"""
H1 secondary outcomes: Rejected F1 and false-'Rejected' rate on dark test images (base vs photo augmentation).
Low-light = test images in the lowest quartile of mean grey value (64x64 thumbnail). Only 3 test images fall below
the phase-1b train threshold (70), and none of them is Leaned/Straight, so the quartile definition was fixed before any
H1 result was inspected. The 4 dusk/night images named in phase 1b are also reported individually.
Usage: python tools/analysis/h1_dark.py --root <pole_lean_assessment> --results results/marimo
"""
import os, glob, json, argparse
import numpy as np
from PIL import Image

CLASSES = ["Leaned", "Straight", "Rejected"]
NAMED = ["1024.jpg", "1460.jpg", "1560.jpg", "933.jpg"]   # phase-1b dusk/night false-Rejected images
TAGS = {"b4a_final_dinov2": ["dinov2_tuned_cls", "dinov2_tuned_mt"],
        "b5_h1_photo_dinov2": ["dinov2_photo_cls", "dinov2_photo_mt"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--results", default="results/marimo")
    ap.add_argument("--out", default="results/analysis/h1")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    grey = {}
    for c in CLASSES:
        for p in glob.glob(os.path.join(a.root, "classification", "test", c, "*")):
            grey[os.path.basename(p)] = float(np.asarray(Image.open(p).convert("L").resize((64, 64))).mean())
    DARK = float(np.percentile(list(grey.values()), 25))
    L, rep = [], {}
    def P(s): print(s); L.append(s)
    for batch, tags in TAGS.items():
        for tag in tags:
            rs = []
            for f in sorted(glob.glob(os.path.join(a.results, batch, f"res_{tag}_s*.json"))):
                rs += json.load(open(f))
            if not rs:
                continue
            m = {"rej_F1": [], "rej_recall": [], "rej_precision": [], "false_rej_dark": [], "false_rej_bright": [],
                 "named_rej": []}
            for r in rs:
                yt, yp = np.array(r["y_true"]), np.array(r["y_pred"])
                dk = np.array([grey[f] < DARK for f in r["test_files"]])
                tp = np.sum((yp == 2) & (yt == 2)); fp = np.sum((yp == 2) & (yt != 2)); fn = np.sum((yp != 2) & (yt == 2))
                m["rej_F1"].append(2 * tp / (2 * tp + fp + fn)); m["rej_recall"].append(tp / (tp + fn))
                m["rej_precision"].append(tp / max(tp + fp, 1))
                nr = yt != 2
                m["false_rej_dark"].append(np.mean(yp[nr & dk] == 2) if np.any(nr & dk) else np.nan)
                m["false_rej_bright"].append(np.mean(yp[nr & ~dk] == 2))
                pr = dict(zip(r["test_files"], yp.tolist()))
                m["named_rej"].append(np.mean([pr[f] == 2 for f in NAMED]))
            rep[tag] = {k: [float(np.nanmean(v)), float(np.nanstd(v, ddof=1))] for k, v in m.items()}
            n_dark = int(np.sum([grey[f] < DARK and t != 2 for f, t in zip(rs[0]["test_files"], rs[0]["y_true"])]))
            P(f"{tag:18s} Rejected F1 {100*rep[tag]['rej_F1'][0]:.1f} +- {100*rep[tag]['rej_F1'][1]:.1f} "
              f"(P {100*rep[tag]['rej_precision'][0]:.1f}, R {100*rep[tag]['rej_recall'][0]:.1f}) | false-Rejected: "
              f"low-light L/S {100*rep[tag]['false_rej_dark'][0]:.1f}% (n={n_dark}), other L/S {100*rep[tag]['false_rej_bright'][0]:.1f}%"
              f" | 4 named dusk images predicted Rejected: {100*rep[tag]['named_rej'][0]:.0f}%")
    P(f"(low-light threshold = 25th percentile of test mean grey = {DARK:.1f})")
    json.dump(rep, open(os.path.join(a.out, "h1_dark.json"), "w"), indent=1)
    open(os.path.join(a.out, "h1_dark.txt"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
