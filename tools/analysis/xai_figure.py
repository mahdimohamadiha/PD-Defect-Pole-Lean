"""X1 descriptive figure (pre-registered selection): every image in the X1 set misclassified by dinov2_tuned_cls
seed 0, plus 8 images drawn at random (rng 0). Columns: image + GT box | cls Grad-CAM | cls attention |
mt Grad-CAM | mt attention | soft-cls attention. Artifact tokens are shown as they are (not removed).
Usage: python tools/analysis/xai_figure.py --root <pole_lean_assessment> --dir results/marimo/xai_x1 --out results/analysis/xai"""
import os, sys, json, argparse
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(__file__))
from xai_attention import gt_boxes

COLS = [("dinov2_tuned_cls_s0", "cam"), ("dinov2_tuned_cls_s0", "att"), ("dinov2_tuned_mt_s0", "cam"),
        ("dinov2_tuned_mt_s0", "att"), ("dinov2_soft_cls_s0", "att")]
CL = ["Leaned", "Straight", "Rejected"]


def overlay(im, a, box, T):
    h = Image.fromarray((np.clip(np.array(a), 0, 1) * 255).astype("uint8")).resize((T, T), Image.NEAREST)
    z = Image.new("L", (T, T))
    o = Image.blend(im, Image.merge("RGB", (h, z, z)), 0.55)
    cx, cy, w, hh = box
    ImageDraw.Draw(o).rectangle([(cx - w / 2) * T, (cy - hh / 2) * T, (cx + w / 2) * T, (cy + hh / 2) * T], outline="lime", width=2)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--dir", default="results/marimo/xai_x1")
    ap.add_argument("--out", default="results/analysis/xai")
    a = ap.parse_args()
    rows = json.load(open(os.path.join(a.dir, "xai_rows.json")))
    maps = json.load(open(os.path.join(a.dir, "xai_maps.json")))
    boxes = {os.path.basename(p): (p, v) for p, v in gt_boxes(a.root).items()}
    r0 = {r["file"]: r for r in rows if r["tag"] == "dinov2_tuned_cls" and r["seed"] == 0}
    pmt = {r["file"]: r["pred"] for r in rows if r["tag"] == "dinov2_tuned_mt" and r["seed"] == 0}
    files = sorted(r0)
    wrong = [f for f in files if r0[f]["pred"] != r0[f]["y"]]
    rng = np.random.default_rng(0)
    rand = [files[i] for i in rng.choice(len(files), 8, replace=False)]
    for name, sel in [("misclassified_by_cls_s0", wrong), ("random8_rng0", rand)]:
        T = 192; W = T * (1 + len(COLS)); rows_img = []
        for f in sel:
            p, (y, box) = boxes[f]
            im = Image.open(p).convert("RGB").resize((T, T))
            base = im.copy(); d = ImageDraw.Draw(base)
            cx, cy, w, hh = box
            d.rectangle([(cx - w / 2) * T, (cy - hh / 2) * T, (cx + w / 2) * T, (cy + hh / 2) * T], outline="lime", width=2)
            d.rectangle([0, 0, T, 14], fill="black")
            d.text((2, 1), f"{f} GT {CL[y]} | cls {CL[r0[f]['pred']]} | mt {CL[pmt[f]]}"[:34], fill="white")
            tiles = [base] + [overlay(im, maps[k][f][t], box, T) for k, t in COLS]
            r = Image.new("RGB", (W, T), "white"); [r.paste(t, (T * i, 0)) for i, t in enumerate(tiles)]; rows_img.append(r)
        hdr = Image.new("RGB", (W, 16), "white"); d = ImageDraw.Draw(hdr)
        for i, lab in enumerate(["image + GT box", "cls Grad-CAM", "cls attention", "mt Grad-CAM", "mt attention", "soft-cls attention"]):
            d.text((T * i + 4, 2), lab, fill="black")
        sheet = Image.new("RGB", (W, 16 + T * len(rows_img)), "white"); sheet.paste(hdr, (0, 0))
        for i, r in enumerate(rows_img): sheet.paste(r, (0, 16 + T * i))
        os.makedirs(a.out, exist_ok=True); sheet.save(os.path.join(a.out, f"xai_{name}.jpg"), quality=85)
        print(name, len(sel), "images")


if __name__ == "__main__":
    main()
