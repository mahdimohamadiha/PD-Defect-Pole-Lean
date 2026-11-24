"""
XAI (pre-registered in README): does the classifier look at the pole?
For Leaned/Straight test images with a GT pole box (keypoint labels, same image size; evaluation only), compute
  * Grad-CAM at the input of the last transformer block (relevance = ReLU(sum_d grad * act) per patch token,
    gradient of the predicted-class logit), and
  * last-layer CLS-token attention (mean over heads, patch tokens only; register tokens excluded),
then the share of relevance inside the GT pole box ("energy-in-box"), the box area share (chance level), and the
pointing-game hit (top patch inside the box). One row per (model, seed, image).
Usage (GPU): python xai_attention.py --root <pole_lean_assessment> --ckpt_root <dir with b*/<tag>_s<seed>.pt>
             --out <dir> [--maps_for dinov2_tuned_cls_s0 ...]
"""
import os, sys, glob, json, argparse
import numpy as np, torch
from PIL import Image

TAGS = {"b4a_final_dinov2": ["dinov2_tuned_cls", "dinov2_tuned_mt"],
        "b4b_final_dinov2reg": ["dinov2reg_tuned_cls", "dinov2reg_tuned_mt"],
        "b7_h2_soft_dinov2": ["dinov2_soft_cls", "dinov2_soft_mt"]}
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def gt_boxes(root):
    """test file -> (cx, cy, w, h) normalised, of the largest pole; Leaned/Straight test images whose keypoint image has the same size."""
    lab, img = {}, {}
    for sp in ["train", "valid", "test"]:
        for f in glob.glob(os.path.join(root, "keypoint_detection", "labels", sp, "*.txt")):
            best, area = None, -1
            for line in open(f).read().strip().splitlines():
                v = [float(t) for t in line.split()]
                if len(v) >= 11 and v[3] * v[4] > area:
                    area, best = v[3] * v[4], v[1:5]
            if best is not None:
                lab[os.path.splitext(os.path.basename(f))[0]] = best
        for p in glob.glob(os.path.join(root, "keypoint_detection", "images", sp, "*")):
            img[os.path.splitext(os.path.basename(p))[0]] = p
    out = {}
    for ci, c in enumerate(["Leaned", "Straight"]):
        for p in sorted(glob.glob(os.path.join(root, "classification", "test", c, "*"))):
            st = os.path.splitext(os.path.basename(p))[0]
            if st in lab and st in img and Image.open(p).size == Image.open(img[st]).size:
                out[p] = (ci, lab[st])
    return out


def box_mask(box, g):
    cx, cy, w, h = box
    ys, xs = (np.arange(g) + 0.5) / g, (np.arange(g) + 0.5) / g
    mx = (xs >= cx - w / 2) & (xs <= cx + w / 2)
    my = (ys >= cy - h / 2) & (ys <= cy + h / 2)
    m = my[:, None] & mx[None, :]
    if not m.any():   # degenerate thin box: the patch containing the centre
        m[min(int(cy * g), g - 1), min(int(cx * g), g - 1)] = True
    return m


def explain(model, x):
    """returns (pred, gradcam [g,g], cls_attn [g,g] or None)"""
    bb = model.bb
    out = bb(pixel_values=x, output_hidden_states=True, output_attentions=True)
    hs = out.hidden_states
    h_in = hs[-2]                       # input of the last block
    h_in.retain_grad()
    h = out.last_hidden_state
    n = model.n_patch
    cls_t, patches = h[:, 0], h[:, -n:]
    kq, _ = model.attn(model.q.unsqueeze(0).expand(x.size(0), -1, -1), patches, patches)
    feat = torch.cat([cls_t, patches.mean(1)], -1)
    lg = model.cls_head(feat)
    pred = int(lg.argmax(-1))
    model.zero_grad(set_to_none=True)
    lg[0, pred].backward()
    g = int(round(n ** 0.5))
    cam = torch.relu((h_in.grad[0, -n:] * h_in[0, -n:]).sum(-1)).detach().float().cpu().numpy().reshape(g, g)
    att = None
    if out.attentions is not None and out.attentions[-1] is not None:
        att = out.attentions[-1][0, :, 0, -n:].mean(0).detach().float().cpu().numpy().reshape(g, g)
    return pred, cam, att


def metrics(m, box):
    s = m.sum()
    if s <= 0:
        return np.nan, np.nan
    return float(m[box].sum() / s), float(box.ravel()[m.argmax()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--ckpt_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train_script_dir", default=".")
    ap.add_argument("--maps_for", nargs="*", default=["dinov2_tuned_cls_s0", "dinov2_tuned_mt_s0", "dinov2_soft_cls_s0"])
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N images")
    a = ap.parse_args()
    sys.path.insert(0, a.train_script_dir)
    from train_multitask import MTNet
    os.makedirs(a.out, exist_ok=True)
    boxes = gt_boxes(a.root)
    files = sorted(boxes)[: a.limit or None]
    print("test L/S images with a GT pole box:", len(files))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows, maps = [], {}
    for batch, tags in TAGS.items():
        for tag in tags:
            for seed in [0, 1, 2]:
                ck = os.path.join(a.ckpt_root, batch, f"{tag}_s{seed}.pt")
                if not os.path.exists(ck):
                    print("missing", ck); continue
                c = torch.load(ck, map_location="cpu")
                ar = c["args"]
                model = MTNet(ar["backbone"], ar["size"], ar["mode"])
                model.load_state_dict({k: v.float() for k, v in c["state_dict"].items()})
                model.bb.config._attn_implementation = "eager"
                model.to(dev).eval()
                for p in files:
                    y, bx = boxes[p]
                    im = Image.open(p).convert("RGB").resize((ar["size"], ar["size"]), Image.BICUBIC)
                    x = torch.tensor(((np.asarray(im) / 255.0 - MEAN) / STD).transpose(2, 0, 1), dtype=torch.float32)[None].to(dev)
                    pred, cam, att = explain(model, x)
                    bm = box_mask(bx, cam.shape[0])
                    e_cam, hit_cam = metrics(cam, bm)
                    e_att, hit_att = metrics(att, bm) if att is not None else (np.nan, np.nan)
                    rows.append({"tag": tag, "seed": seed, "file": os.path.basename(p), "y": y, "pred": pred,
                                 "box_area": float(bm.mean()), "cam_in_box": e_cam, "cam_hit": hit_cam,
                                 "att_in_box": e_att, "att_hit": hit_att})
                    if f"{tag}_s{seed}" in a.maps_for:
                        maps.setdefault(f"{tag}_s{seed}", {})[os.path.basename(p)] = {
                            "cam": np.round(cam / (cam.max() + 1e-12), 3).tolist(),
                            "att": None if att is None else np.round(att / (att.max() + 1e-12), 3).tolist()}
                r = [x for x in rows if x["tag"] == tag and x["seed"] == seed]
                print(f"{tag} s{seed}: cam_in_box {np.nanmean([x['cam_in_box'] for x in r]):.3f} "
                      f"att_in_box {np.nanmean([x['att_in_box'] for x in r]):.3f} box_area {np.mean([x['box_area'] for x in r]):.3f}", flush=True)
                del model; torch.cuda.empty_cache()
    json.dump(rows, open(os.path.join(a.out, "xai_rows.json"), "w"))
    json.dump(maps, open(os.path.join(a.out, "xai_maps.json"), "w"))
    print("done", len(rows))


if __name__ == "__main__":
    main()
