"""
PD-Defect / Pole Lean Assessment: geometry-aware multi-task model
(3-way classification + pole-axis keypoints), Kaggle-ready.

Protocol (keeps results comparable to the dataset paper's DINOv3 baseline):
  * classification/{train,val,test} official split is the ONLY split used.
  * keypoint labels are used ONLY for images that sit in classification/train
    (matched by file stem, verified by image size). Keypoint labels of
    classification val/test images are never used for training -> no leakage,
    even though the keypoint subset has its own, different split.
  * 'Rejected' images: keypoint loss is masked (34 Rejected train images do have matching
    keypoint labels, but they are deliberately ignored).

Usage (Kaggle, internet on):
  python train_multitask.py --download --mode mt_geo --seeds 0 1 2
  modes: cls (baseline) | mt (aux keypoint loss) | mt_geo (+ geometry-conditioned head)
"""
import os, glob, math, json, argparse, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report

CLASSES = ["Leaned", "Straight", "Rejected"]
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


# ---------------------------------------------------------------- data
def read_kp_labels(label_dir):
    """stem -> (cx,cy,w,h,[x_bottom,y_bottom,x_top,y_top]) of the LARGEST pole."""
    out = {}
    for f in glob.glob(os.path.join(label_dir, "*.txt")):
        best, best_area = None, -1
        for line in open(f).read().strip().splitlines():
            v = [float(t) for t in line.split()]
            if len(v) < 11:
                continue
            area = v[3] * v[4]
            if area > best_area:
                best_area, best = area, v[5:11]
        if best is not None:  # [kx1 ky1 v1 kx2 ky2 v2]
            out[os.path.splitext(os.path.basename(f))[0]] = [best[0], best[1], best[3], best[4]]
    return out


def build_index(root):
    kp = {}
    for sp in ["train", "valid", "test"]:
        kp.update(read_kp_labels(os.path.join(root, "keypoint_detection", "labels", sp)))
    kp_img = {}
    for sp in ["train", "valid", "test"]:
        for p in glob.glob(os.path.join(root, "keypoint_detection", "images", sp, "*")):
            kp_img[os.path.splitext(os.path.basename(p))[0]] = p
    idx = {"train": [], "val": [], "test": []}
    n_match = n_size_mismatch = 0
    for sp in idx:
        for ci, c in enumerate(CLASSES):
            for p in sorted(glob.glob(os.path.join(root, "classification", sp, c, "*"))):
                stem = os.path.splitext(os.path.basename(p))[0]
                k = None
                if sp == "train" and ci != 2 and stem in kp and stem in kp_img:   # kp only for cls-train, never Rejected
                    n_match += 1
                    if Image.open(p).size == Image.open(kp_img[stem]).size:
                        k = kp[stem]
                    else:
                        n_size_mismatch += 1
                idx[sp].append((p, ci, k))
    print({s: len(v) for s, v in idx.items()}, "train imgs with kp:", n_match - n_size_mismatch,
          "| size mismatches dropped:", n_size_mismatch)
    return idx


class PoleDS(Dataset):
    def __init__(self, items, size, train):
        self.items, self.size, self.train = items, size, train
        import torchvision.transforms as T
        self.cj = T.ColorJitter(0.3, 0.3, 0.2, 0.02)
        self.norm = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y, k = self.items[i]
        img = Image.open(p).convert("RGB").resize((self.size, self.size), Image.BICUBIC)
        kp = np.array(k if k is not None else [0, 0, 0, 0], dtype=np.float32)
        m = np.float32(k is not None)
        if self.train:
            img = self.cj(img)
            if random.random() < 0.5:           # hflip is label-preserving (lean is symmetric)
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                kp[[0, 2]] = 1 - kp[[0, 2]]
            # NOTE: no rotation / shear: they would change the lean label itself.
        return self.norm(img), y, torch.from_numpy(kp), torch.tensor(m)


# ---------------------------------------------------------------- model
class MTNet(nn.Module):
    def __init__(self, name, size, mode, n_cls=3):
        super().__init__()
        from transformers import AutoModel
        self.bb = AutoModel.from_pretrained(name)
        d = self.bb.config.hidden_size
        self.patch = self.bb.config.patch_size
        self.n_patch = (size // self.patch) ** 2
        self.mode = mode
        self.q = nn.Parameter(torch.randn(2, d) * 0.02)           # bottom / top queries
        self.attn = nn.MultiheadAttention(d, 8, batch_first=True)
        self.kp_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 2))
        geo_dim = {"mt_geo": 4, "mt_geo_d": 3}.get(mode, 0)
        self.cls_head = nn.Sequential(nn.LayerNorm(2 * d + geo_dim), nn.Dropout(0.2),
                                      nn.Linear(2 * d + geo_dim, n_cls))

    @staticmethod
    def geo_feats(kp):
        xb, yb, xt, yt = kp.unbind(-1)                            # image coords, y grows downward
        dx, dy = xt - xb, yb - yt                                 # axis vector, bottom -> top
        L = torch.sqrt(dx ** 2 + dy ** 2 + 1e-6)
        s = dx / L                                                # sin(theta from vertical)
        return torch.stack([s, s.abs(), dy / L, L], -1)

    def forward(self, x):
        h = self.bb(pixel_values=x).last_hidden_state
        cls_t, patches = h[:, 0], h[:, -self.n_patch:]            # skips register tokens if any
        kq, _ = self.attn(self.q.unsqueeze(0).expand(x.size(0), -1, -1), patches, patches)
        kp = torch.sigmoid(self.kp_head(kq)).flatten(1)           # [B,4] = xb,yb,xt,yt in [0,1]
        feat = [cls_t, patches.mean(1)]
        if self.mode == "mt_geo":
            feat.append(self.geo_feats(kp))                       # predicted (not GT) geometry
        elif self.mode == "mt_geo_d":                             # detached: cls loss cannot pull the kp head;
            feat.append(self.geo_feats(kp.detach())[:, :3])       # drop un-normalised length L
        return self.cls_head(torch.cat(feat, -1)), kp


# ---------------------------------------------------------------- train / eval
@torch.no_grad()
def evaluate(model, dl, dev, full=False):
    model.eval(); ys, ps, probs, kps = [], [], [], []
    for x, y, _, _ in dl:
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            lg, kp = model(x.to(dev))
        ys += y.tolist(); ps += lg.argmax(-1).cpu().tolist()
        if full:
            probs += lg.float().softmax(-1).cpu().numpy().round(5).tolist()
            kps += kp.float().cpu().numpy().round(5).tolist()   # xb,yb,xt,yt normalised (unused head in cls mode)
    return (ys, ps, probs, kps) if full else (ys, ps)


def run(args, seed, idx):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mk = lambda sp, tr: DataLoader(PoleDS(idx[sp], args.size, tr), batch_size=args.bs, shuffle=tr,
                                   num_workers=args.workers, drop_last=tr, persistent_workers=args.workers > 0)
    tr, va = mk("train", True), mk("val", False)
    te = None if args.no_test else mk("test", False)
    model = MTNet(args.backbone, args.size, args.mode).to(dev)
    heads = [p for n, p in model.named_parameters() if not n.startswith("bb.")]
    opt = torch.optim.AdamW([{"params": model.bb.parameters(), "lr": args.lr_bb},
                             {"params": heads, "lr": args.lr_head}], weight_decay=0.05)
    total = args.epochs * len(tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [args.lr_bb, args.lr_head], total_steps=total, pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=dev == "cuda")
    cnt = np.bincount([y for _, y, _ in idx["train"]], minlength=3)
    w = torch.tensor(cnt.sum() / (3 * cnt), dtype=torch.float32, device=dev)
    best, best_ep = (-1, None), -1
    for ep in range(args.epochs):
        model.train()
        for x, y, kp, m in tr:
            x, y, kp, m = x.to(dev), y.to(dev), kp.to(dev), m.to(dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                lg, kp_hat = model(x)
                loss = F.cross_entropy(lg, y, weight=w)
                if args.mode != "cls" and m.sum() > 0:
                    l_kp = (F.smooth_l1_loss(kp_hat.float(), kp, reduction="none", beta=0.02).mean(1) * m).sum() / m.sum()
                    loss = loss + args.lam * l_kp
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
        ys, ps = evaluate(model, va, dev)
        f1 = f1_score(ys, ps, average="macro")
        print(f"seed{seed} ep{ep} val acc {accuracy_score(ys, ps):.4f} macroF1 {f1:.4f}")
        if f1 > best[0]:
            best, best_ep = (f1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}), ep
    model.load_state_dict(best[1])
    vys, vps, vpr, vkq = evaluate(model, va, dev, full=True)
    res = {"seed": seed, "mode": args.mode, "tag": args.tag or args.mode, "backbone": args.backbone, "best_epoch": best_ep,
           "val_macroF1": best[0], "val_acc": accuracy_score(vys, vps),
           "val_y_true": vys, "val_y_pred": vps, "val_probs": vpr, "val_kp": vkq,
           "val_files": [os.path.basename(i[0]) for i in idx["val"]], "args": vars(args)}
    if not args.no_test:   # hyper-parameter search runs use --no_test: the test split is never looked at
        ys, ps, pr, kq = evaluate(model, te, dev, full=True)
        res.update({"test_acc": accuracy_score(ys, ps), "test_macroF1": f1_score(ys, ps, average="macro"),
                    "y_true": ys, "y_pred": ps, "test_files": [os.path.basename(i[0]) for i in idx["test"]],
                    "test_probs": pr, "test_kp": kq})
    if args.ckpt_dir:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        ck = os.path.join(args.ckpt_dir, f"{args.tag or args.mode}_s{seed}.pt")
        torch.save({"state_dict": {k: (v.half() if v.is_floating_point() else v) for k, v in best[1].items()},
                    "args": vars(args), "seed": seed, "best_epoch": best_ep, "val_macroF1": best[0],
                    "classes": CLASSES}, ck)
        res["checkpoint"] = os.path.basename(ck)
    if not args.no_test:
        print(classification_report(ys, ps, labels=[0, 1, 2], target_names=CLASSES, digits=4, zero_division=0))
    print({k: v for k, v in res.items() if not isinstance(v, (list, dict))})
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/kaggle/working/pd-defect/pole_lean_assessment")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--backbone", default="facebook/dinov2-base")   # baseline paper: facebook/dinov3-vitb16-pretrain-lvd1689m (gated)
    ap.add_argument("--mode", default="mt_geo", choices=["cls", "mt", "mt_geo", "mt_geo_d"])
    ap.add_argument("--size", type=int, default=448)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr_bb", type=float, default=2e-5)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=5.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tag", default=None, help="experiment name stored in results (default: mode)")
    ap.add_argument("--ckpt_dir", default=None, help="save best checkpoint (fp16) per run here")
    ap.add_argument("--no_test", action="store_true", help="skip the test split entirely (hyper-parameter search)")
    ap.add_argument("--data_dir", default="/kaggle/working/pd-defect", help="download target (--root should be <data_dir>/pole_lean_assessment)")
    a = ap.parse_args()
    if a.download:
        from huggingface_hub import snapshot_download
        snapshot_download("EPDCL/pd-defect", repo_type="dataset", local_dir=a.data_dir,
                          allow_patterns=["pole_lean_assessment/classification/**",
                                          "pole_lean_assessment/keypoint_detection/images/**",
                                          "pole_lean_assessment/keypoint_detection/labels/**"])
    index = build_index(a.root)
    results = [run(a, s, index) for s in a.seeds]
    json.dump(results, open(a.out, "w"), indent=1)
    for k in (["val_macroF1"] if a.no_test else ["val_macroF1", "test_acc", "test_macroF1"]):
        v = np.array([r[k] for r in results]); print(k, f"{v.mean():.4f} +- {v.std():.4f}")
