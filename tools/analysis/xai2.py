"""
X2 (pre-registered in README): does the decision depend on the pole?
For the 143 Leaned/Straight test images with a GT pole box (evaluation only) and the 18 X1 checkpoints:
  * occlusion: blur the pole box / 5 random non-overlapping boxes of the same size / everything except the pole box,
    and record the log-probability of the originally predicted class and the prediction;
  * attributions: last-layer CLS attention, attention rollout, Chefer et al. (2021) relevance, last-block Grad-CAM;
    energy-in-box and a deletion test (blur patches in attribution order, 10 steps up to 50% of the patches) for each
    method and for a random order;
  * sanity check: re-initialise the classification head and the last 4 blocks of dinov2_tuned_cls seed 0 and
    correlate the maps.
Usage (GPU): python xai2.py --root <pole_lean_assessment> --ckpt_root <dir> --out <dir> --train_script_dir <dir>
"""
import os, sys, json, argparse, copy
import numpy as np, torch
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xai_attention import gt_boxes, box_mask, metrics, TAGS, MEAN, STD

BLUR = 24          # Gaussian blur radius in pixels (at 448 px)
N_RAND = 5         # random control boxes per image
STEPS = 10         # deletion steps
MAX_FRAC = 0.5     # deletion up to 50% of the patches
METHODS = ["att", "rollout", "chefer", "gradcam"]
FIG_MODELS = ["convnext_tuned_cls_s0", "convnext_tuned_mt_s0", "dinov2_tuned_cls_s0", "dinov2_tuned_mt_s0", "dinov2reg_tuned_cls_s0", "dinov2reg_tuned_mt_s0",
              "dinov2_soft_cls_s0"]


def to_x(im, dev):
    a = (np.asarray(im, dtype=np.float32) / 255.0 - MEAN) / STD
    return torch.tensor(a.transpose(2, 0, 1), dtype=torch.float32)


def pix_rect(box, T):
    cx, cy, w, h = box
    x0, y0 = int(round((cx - w / 2) * T)), int(round((cy - h / 2) * T))
    x1, y1 = int(round((cx + w / 2) * T)), int(round((cy + h / 2) * T))
    return max(0, x0), max(0, y0), min(T, max(x1, x0 + 1)), min(T, max(y1, y0 + 1))


def random_boxes(rect, T, rng, n=N_RAND, tries=500):
    x0, y0, x1, y1 = rect
    bw, bh = x1 - x0, y1 - y0
    out = []
    for _ in range(tries):
        if len(out) == n or bw >= T or bh >= T:
            break
        rx, ry = int(rng.integers(0, T - bw + 1)), int(rng.integers(0, T - bh + 1))
        if rx < x1 and rx + bw > x0 and ry < y1 and ry + bh > y0:
            continue                                   # overlaps the pole box
        out.append((rx, ry, rx + bw, ry + bh))
    return out


def paste(base, blurred, rect):
    a = np.array(base); b = np.asarray(blurred)
    x0, y0, x1, y1 = rect
    a[y0:y1, x0:x1] = b[y0:y1, x0:x1]
    return a


def head(model, h):
    n = model.n_patch
    cls_t, patches = h[:, 0], h[:, -n:]
    return model.cls_head(torch.cat([cls_t, patches.mean(1)], -1))


def explain_conv(model, x):
    """CNN backbone (H4): standard Grad-CAM on the last-stage feature map; attention-based methods do not apply."""
    out = model.bb(pixel_values=x)
    A = out.last_hidden_state; A.retain_grad()
    patches = A.flatten(2).transpose(1, 2)
    lg = model.cls_head(torch.cat([out.pooler_output, patches.mean(1)], -1))
    lp = torch.log_softmax(lg.float(), -1)[0]
    pred = int(lg.argmax(-1))
    model.zero_grad(set_to_none=True)
    lg[0, pred].backward()
    with torch.no_grad():
        w = A.grad[0].mean((1, 2))                                  # channel weights = spatially averaged gradients
        cam = torch.relu((w[:, None, None] * A[0]).sum(0))
    return pred, lp.detach().cpu().numpy(), {"gradcam": cam.float().cpu().numpy()}


def explain(model, x):
    """one forward + backward: prediction, log-probs, and the four attribution maps (g x g, numpy)."""
    if getattr(model, "conv", False):
        return explain_conv(model, x)
    out = model.bb(pixel_values=x, output_hidden_states=True, output_attentions=True)
    atts = out.attentions
    for a in atts:
        a.retain_grad()
    h_in = out.hidden_states[-2]; h_in.retain_grad()
    lg = head(model, out.last_hidden_state)
    lp = torch.log_softmax(lg.float(), -1)[0]
    pred = int(lg.argmax(-1))
    model.zero_grad(set_to_none=True)
    lg[0, pred].backward()
    n = model.n_patch; g = int(round(n ** 0.5)); N = atts[0].shape[-1]
    with torch.no_grad():
        att = atts[-1][0, :, 0, -n:].mean(0)
        eye = torch.eye(N, device=x.device)
        R = eye.clone()
        for a in atts:                                     # rollout (Abnar & Zuidema, 2020)
            A = 0.5 * a[0].mean(0) + 0.5 * eye
            A = A / A.sum(-1, keepdim=True)
            R = A @ R
        roll = R[0, -n:]
        R = eye.clone()
        for a in atts:                                     # Chefer et al. (2021), gradient-weighted relevance
            cam = (a.grad[0] * a[0]).clamp(min=0).mean(0)
            R = R + cam @ R
        chef = R[0, -n:]
        gcam = torch.relu((h_in.grad[0, -n:] * h_in[0, -n:]).sum(-1))
    f = lambda t: t.float().cpu().numpy().reshape(g, g)
    return pred, lp.detach().cpu().numpy(), {"att": f(att), "rollout": f(roll), "chefer": f(chef), "gradcam": f(gcam)}


@torch.no_grad()
def batch_logp(model, arrs, dev, bs=32):
    """log-probs [len(arrs), 3] for a list of HxWx3 uint8 arrays."""
    out = []
    for i in range(0, len(arrs), bs):
        x = torch.stack([to_x(a, dev) for a in arrs[i:i + bs]]).to(dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            lg, _ = model(x)
        out.append(torch.log_softmax(lg.float(), -1).cpu().numpy())
    return np.concatenate(out)


def deletion_set(im_a, bl_a, order, T, g):
    """images with the top-k patches (in `order`) replaced by the blurred image, k over STEPS steps up to MAX_FRAC."""
    p = T // g; out = []
    for s in range(1, STEPS + 1):
        k = int(round(s / STEPS * MAX_FRAC * g * g))
        a = im_a.copy()
        for idx in order[:k]:
            r, c = divmod(int(idx), g)
            a[r * p:(r + 1) * p, c * p:(c + 1) * p] = bl_a[r * p:(r + 1) * p, c * p:(c + 1) * p]
        out.append(a)
    return out


def load(ck, MTNet, dev):
    c = torch.load(ck, map_location="cpu"); ar = c["args"]
    model = MTNet(ar["backbone"], ar["size"], ar["mode"])
    model.load_state_dict({k: v.float() for k, v in c["state_dict"].items()})
    model.bb.config._attn_implementation = "eager"
    return model.to(dev).eval(), ar


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a.ravel())), np.argsort(np.argsort(b.ravel()))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--ckpt_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train_script_dir", default=".")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=None, help="debug: restrict to these <tag>_s<seed>")
    ap.add_argument("--convnext", action="store_true", help="H4: run on the ConvNeXt checkpoints (b10) instead of the X1 set")
    a = ap.parse_args()
    sys.path.insert(0, a.train_script_dir)
    from train_multitask import MTNet
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    boxes = gt_boxes(a.root)
    files = sorted(boxes)[: a.limit or None]
    T = 448
    prep = {}
    for p in files:                                        # model-independent image variants
        y, box = boxes[p]
        im = Image.open(p).convert("RGB").resize((T, T), Image.BICUBIC)
        bl = im.filter(ImageFilter.GaussianBlur(BLUR))
        im_a, bl_a = np.array(im), np.array(bl)
        rect = pix_rect(box, T)
        rb = random_boxes(rect, T, np.random.default_rng(0))       # rng 0 per image (pre-registered)
        ctx = bl_a.copy(); x0, y0, x1, y1 = rect; ctx[y0:y1, x0:x1] = im_a[y0:y1, x0:x1]
        prep[p] = dict(y=y, box=box, im=im, im_a=im_a, bl_a=bl_a, rect=rect, rand=rb,
                       occ=[paste(im_a, bl_a, rect)] + [paste(im_a, bl_a, r) for r in rb] + [ctx])
    print("images:", len(files), "| with 5 random boxes:", sum(len(v["rand"]) == N_RAND for v in prep.values()), flush=True)
    rows, maps = [], {}
    tagset = {"b10_final_convnext": ["convnext_tuned_cls", "convnext_tuned_mt"]} if a.convnext else TAGS
    for batch, tags in tagset.items():
        for tag in tags:
            for seed in [0, 1, 2]:
                name = f"{tag}_s{seed}"
                if a.only and name not in a.only:
                    continue
                ck = os.path.join(a.ckpt_root, batch, f"{name}.pt")
                if not os.path.exists(ck):
                    print("missing", ck); continue
                model, ar = load(ck, MTNet, dev)
                for p in files:
                    d = prep[p]
                    x = to_x(d["im_a"], dev)[None].to(dev)
                    pred, lp0, mp = explain(model, x)
                    g = next(iter(mp.values())).shape[0]
                    bm = box_mask(d["box"], g)
                    rng = np.random.default_rng(1)
                    meths = [m for m in METHODS if m in mp]
                    orders = {m: np.argsort(-mp[m].ravel(), kind="stable") for m in meths}
                    orders["random"] = rng.permutation(g * g)
                    dels = [im for m in meths + ["random"] for im in deletion_set(d["im_a"], d["bl_a"], orders[m], T, g)]
                    LP = batch_logp(model, d["occ"] + dels, dev)
                    k = len(d["occ"]); nr = len(d["rand"])
                    occ_lp = LP[:k]; del_lp = LP[k:].reshape(len(meths) + 1, STEPS, 3)
                    r = {"tag": tag, "seed": seed, "file": os.path.basename(p), "y": d["y"], "pred": pred,
                         "lp0": float(lp0[pred]), "n_rand": nr,
                         "drop_pole": float(lp0[pred] - occ_lp[0, pred]),
                         "drop_rand": float(np.mean(lp0[pred] - occ_lp[1:1 + nr, pred])) if nr else None,
                         "flip_pole": int(occ_lp[0].argmax() != pred),
                         "flip_rand": float(np.mean(occ_lp[1:1 + nr].argmax(1) != pred)) if nr else None,
                         "ctx_pred": int(occ_lp[-1].argmax()), "box_area": float(bm.mean())}
                    for j, m in enumerate(meths + ["random"]):
                        r[f"aoc_{m}"] = float(np.mean(lp0[pred] - del_lp[j, :, pred]))
                    for m in meths:
                        r[f"in_box_{m}"] = metrics(mp[m], bm)[0]
                        r[f"hit_{m}"] = metrics(mp[m], bm)[1]
                    rows.append(r)
                    if name in FIG_MODELS:
                        maps.setdefault(name, {})[os.path.basename(p)] = {m: (mp[m] / (mp[m].max() + 1e-12)).astype(np.float16) for m in meths}
                rr = [x for x in rows if x["tag"] == tag and x["seed"] == seed and x["n_rand"] == N_RAND]
                print(f"{name}: pole reliance {np.mean([x['drop_pole'] - x['drop_rand'] for x in rr]):+.3f} | "
                      + " ".join(f"aoc_{m} {np.mean([x['aoc_' + m] for x in rr]):.3f}" for m in meths + ["random"])
                      + " | " + " ".join(f"box_{m} {np.nanmean([x['in_box_' + m] for x in rr]):.3f}" for m in meths), flush=True)
                del model; torch.cuda.empty_cache()
    json.dump(rows, open(os.path.join(a.out, "xai2_rows.json"), "w"))
    np.savez_compressed(os.path.join(a.out, "xai2_maps.npz"),
                        **{f"{k}|{f}|{m}": v[m] for k, d in maps.items() for f, v in d.items() for m in v})
    # ---- sanity check (Adebayo et al., 2018)
    ck = os.path.join(a.ckpt_root, "b4a_final_dinov2", "dinov2_tuned_cls_s0.pt")
    if os.path.exists(ck) and not a.only and not a.convnext:
        model, ar = load(ck, MTNet, dev)
        rand_model = copy.deepcopy(model)
        torch.manual_seed(0)
        for mod in list(rand_model.cls_head.modules()) + [m for blk in rand_model.bb.encoder.layer[-4:] for m in blk.modules()]:
            if isinstance(mod, torch.nn.Linear):
                torch.nn.init.normal_(mod.weight, std=0.02); torch.nn.init.zeros_(mod.bias)
            elif isinstance(mod, torch.nn.LayerNorm):
                torch.nn.init.ones_(mod.weight); torch.nn.init.zeros_(mod.bias)
        sc = {m: [] for m in METHODS}
        for p in files:
            x = to_x(prep[p]["im_a"], dev)[None].to(dev)
            _, _, m0 = explain(model, x); _, _, m1 = explain(rand_model, x)
            for m in METHODS:
                sc[m].append(spearman(m0[m], m1[m]))
        sanity = {m: {"median_spearman": float(np.median(v)), "mean_spearman": float(np.mean(v))} for m, v in sc.items()}
        json.dump(sanity, open(os.path.join(a.out, "xai2_sanity.json"), "w"), indent=1)
        print("sanity:", sanity, flush=True)
    print("done", len(rows))


if __name__ == "__main__":
    main()
