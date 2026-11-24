"""
Phase-1 diagnostic analysis (no training). Uses only saved predictions in results/ and the dataset labels.

  A. Label consistency of the binary lean label with the ground-truth pole-axis angle (train split):
     how well can a single angle threshold on the GT axis reproduce the Leaned/Straight labels?
  B. Keypoint quality of the `mt` models on test (angle error vs GT axis).
  C. Rule baselines from predicted keypoints (as in prior work: "angle > 10 deg -> Leaned"):
     - fixed threshold 10 deg, and a threshold selected on validation only;
     - hybrid 3-class: classifier decides Rejected, the angle rule decides Leaned vs Straight.
  D. Error analysis of all tuned classifiers by GT axis angle bin (test images with GT keypoints).

GT keypoints of val/test images are used here for EVALUATION ONLY (never for training or selection,
except C, where the threshold is chosen on validation *predicted* keypoints and validation labels).

Angle = deviation of the bottom->top axis from the image vertical, in pixel space (aspect ratio of the
original image is applied, since keypoints are normalised per axis).

Usage: python tools/analysis/phase1_angle_analysis.py --root <pole_lean_assessment> --results results/marimo
       [--out results/analysis/phase1]
"""
import os, glob, json, argparse, math
import numpy as np
from PIL import Image

CLASSES = ["Leaned", "Straight", "Rejected"]
BINS = [0, 2, 4, 6, 10, 15, 90]
TUNED = {"b4a_final_dinov2": ["dinov2_tuned_cls", "dinov2_tuned_mt"],
         "b4b_final_dinov2reg": ["dinov2reg_tuned_cls", "dinov2reg_tuned_mt"]}
DEFAULT = {"b1_dinov2": ["cls", "mt"], "b2_dinov2reg_geofix": ["dinov2reg_cls", "dinov2reg_mt"]}


def read_kp(root):
    """stem -> (split, [xb, yb, xt, yt]) of the largest pole box, from every keypoint split."""
    out = {}
    for sp in ["train", "valid", "test"]:
        for f in glob.glob(os.path.join(root, "keypoint_detection", "labels", sp, "*.txt")):
            best, area = None, -1
            for line in open(f).read().strip().splitlines():
                v = [float(t) for t in line.split()]
                if len(v) >= 11 and v[3] * v[4] > area:
                    area, best = v[3] * v[4], v[5:11]
            if best is not None:
                out[os.path.splitext(os.path.basename(f))[0]] = (sp, [best[0], best[1], best[3], best[4]])
    return out


def kp_img_sizes(root):
    s = {}
    for p in glob.glob(os.path.join(root, "keypoint_detection", "images", "*", "*")):
        s[os.path.splitext(os.path.basename(p))[0]] = p
    return s


def angle(kp, w, h):
    xb, yb, xt, yt = kp
    dx, dy = (xt - xb) * w, (yt - yb) * h
    return math.degrees(math.atan2(abs(dx), abs(dy) + 1e-9))


def build_gt(root):
    kp, kimg = read_kp(root), kp_img_sizes(root)
    gt = {}   # (split, file) -> dict(label, w, h, angle or None)
    for sp in ["train", "val", "test"]:
        for ci, c in enumerate(CLASSES):
            for p in sorted(glob.glob(os.path.join(root, "classification", sp, c, "*"))):
                fn = os.path.basename(p); stem = os.path.splitext(fn)[0]
                w, h = Image.open(p).size
                a = None
                if stem in kp and stem in kimg and Image.open(kimg[stem]).size == (w, h):
                    a = angle(kp[stem][1], w, h)
                gt[(sp, fn)] = {"label": ci, "w": w, "h": h, "angle": a}
    return gt


def macro_f1(yt, yp, labels):
    f = []
    for c in labels:
        tp = np.sum((yp == c) & (yt == c)); fp = np.sum((yp == c) & (yt != c)); fn = np.sum((yp != c) & (yt == c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def load_runs(results, batch, tag):
    rs = []
    for f in sorted(glob.glob(os.path.join(results, batch, f"res_{tag}_s*.json"))):
        rs += [r for r in json.load(open(f)) if r.get("tag", r["mode"]) == tag]
    return rs


def best_threshold(angles, is_leaned):
    """threshold maximising binary accuracy (Leaned iff angle > t); ties -> the smallest t."""
    cand = np.unique(np.round(angles, 2))
    cand = np.concatenate([[0.0], (cand[:-1] + cand[1:]) / 2, [cand[-1] + 1]])
    acc = [np.mean((angles > t) == is_leaned) for t in cand]
    i = int(np.argmax(acc))
    return float(cand[i]), float(acc[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--results", default="results/marimo")
    ap.add_argument("--out", default="results/analysis/phase1")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    gt = build_gt(a.root)
    rep = {}
    L = []

    def P(*x):
        s = " ".join(str(v) for v in x); print(s); L.append(s)

    # ---------------- A. label consistency with the GT angle
    P("== A. Label consistency: GT axis angle vs binary lean label (Leaned/Straight images with GT keypoints)")
    rep["A"] = {}
    for sp in ["train", "val", "test"]:
        it = [(v["angle"], v["label"]) for (s, _), v in gt.items() if s == sp and v["label"] != 2 and v["angle"] is not None]
        n_all = sum(1 for (s, _), v in gt.items() if s == sp and v["label"] != 2)
        ang = np.array([x[0] for x in it]); lean = np.array([x[1] == 0 for x in it])
        t, acc = best_threshold(ang, lean)
        r = {"n_with_kp": len(it), "n_LS": n_all,
             "leaned_angle_median": float(np.median(ang[lean])), "straight_angle_median": float(np.median(ang[~lean])),
             "leaned_lt3": float(np.mean(ang[lean] < 3)), "straight_gt10": float(np.mean(ang[~lean] > 10)),
             "acc_at_10deg": float(np.mean((ang > 10) == lean)), "best_t": t, "acc_best_t": acc}
        if sp == "train":
            r["hist"] = {f"{BINS[i]}-{BINS[i+1]}": [int(np.sum(lean & (ang >= BINS[i]) & (ang < BINS[i+1]))),
                                                    int(np.sum(~lean & (ang >= BINS[i]) & (ang < BINS[i+1])))]
                         for i in range(len(BINS) - 1)}
        rep["A"][sp] = r
        P(f" {sp:5s}: {len(it)}/{n_all} L/S images with GT kp | median angle Leaned {r['leaned_angle_median']:.1f} deg,"
          f" Straight {r['straight_angle_median']:.1f} deg | Leaned<3deg {100*r['leaned_lt3']:.1f}%,"
          f" Straight>10deg {100*r['straight_gt10']:.1f}%")
        P(f"        rule 'GT angle > 10 deg': acc {100*r['acc_at_10deg']:.1f}% | best threshold {t:.2f} deg: acc {100*acc:.1f}%"
          + ("  (oracle, fitted on this split)" if sp != "train" else ""))
    P(" train histogram (angle bin: #Leaned / #Straight):",
      ", ".join(f"{k}: {v[0]}/{v[1]}" for k, v in rep["A"]["train"]["hist"].items()))
    t_train = rep["A"]["train"]["best_t"]
    for sp in ["val", "test"]:
        it = [(v["angle"], v["label"]) for (s, _), v in gt.items() if s == sp and v["label"] != 2 and v["angle"] is not None]
        ang = np.array([x[0] for x in it]); lean = np.array([x[1] == 0 for x in it])
        rep["A"][sp]["acc_train_t"] = float(np.mean((ang > t_train) == lean))
        P(f" {sp}: GT-angle rule with the train-fitted threshold ({t_train:.2f} deg): acc {100*rep['A'][sp]['acc_train_t']:.1f}%")

    # ---------------- B/C. keypoint quality and rule baselines (mt models)
    P("\n== B. Keypoint quality on test (mt models; images with GT keypoints, Leaned/Straight)")
    P("== C. Rule baselines from predicted keypoints")
    rep["B"], rep["C"] = {}, {}
    for batch, tags in {**TUNED, **DEFAULT}.items():
        for tag in tags:
            runs = load_runs(a.results, batch, tag)
            if not runs or runs[0]["mode"] == "cls" or "test_kp" not in runs[0] or "val_kp" not in runs[0]:
                continue
            B, C = [], []
            for r in runs:
                def pa(files, kps, sp):
                    return np.array([angle(k, gt[(sp, f)]["w"], gt[(sp, f)]["h"]) for f, k in zip(files, kps)])
                ta, va = pa(r["test_files"], r["test_kp"], "test"), pa(r["val_files"], r["val_kp"], "val")
                yt, yp = np.array(r["y_true"]), np.array(r["y_pred"])
                vyt = np.array(r["val_y_true"]); vyp = np.array(r["val_y_pred"])
                g = np.array([gt[("test", f)]["angle"] if gt[("test", f)]["angle"] is not None else np.nan for f in r["test_files"]])
                m = ~np.isnan(g) & (yt != 2)
                B.append({"angle_MAE": float(np.mean(np.abs(ta[m] - g[m]))), "angle_medAE": float(np.median(np.abs(ta[m] - g[m]))),
                          "corr": float(np.corrcoef(ta[m], g[m])[0, 1]), "n": int(m.sum())})
                # threshold on validation (Leaned vs Straight among val images not labelled Rejected)
                vm = vyt != 2
                t_val, _ = best_threshold(va[vm], vyt[vm] == 0)
                res = {"t_val": t_val}
                for name, t in [("t10", 10.0), ("tval", t_val)]:
                    nr = yt != 2   # binary task on GT non-Rejected test images
                    res[f"bin_acc_{name}"] = float(np.mean((ta[nr] > t) == (yt[nr] == 0)))
                    hyb = np.where(yp == 2, 2, np.where(ta > t, 0, 1))
                    res[f"hyb_macroF1_{name}"] = macro_f1(yt, hyb, [0, 1, 2]); res[f"hyb_acc_{name}"] = float(np.mean(hyb == yt))
                res["clf_bin_acc"] = float(np.mean((yp[yt != 2] == 0) == (yt[yt != 2] == 0)))
                res["clf_macroF1"] = macro_f1(yt, yp, [0, 1, 2]); res["clf_acc"] = float(np.mean(yp == yt))
                C.append(res)
            agg = lambda lst, k: (float(np.mean([x[k] for x in lst])), float(np.std([x[k] for x in lst], ddof=1)) if len(lst) > 1 else 0.0)
            rep["B"][tag] = {k: agg(B, k) for k in B[0]}
            rep["C"][tag] = {k: agg(C, k) for k in C[0]}
            b = rep["B"][tag]; c = rep["C"][tag]
            P(f" {tag}: angle MAE {b['angle_MAE'][0]:.2f} deg (median {b['angle_medAE'][0]:.2f}), corr(pred, GT) {b['corr'][0]:.3f}, n={B[0]['n']}")
            P(f"   classifier          : Macro-F1 {100*c['clf_macroF1'][0]:.2f} +- {100*c['clf_macroF1'][1]:.2f}, "
              f"Leaned-vs-Straight acc on non-Rejected {100*c['clf_bin_acc'][0]:.2f}")
            for name, lab in [("t10", "angle > 10 deg"), ("tval", f"angle > t_val (t_val = {c['t_val'][0]:.2f} +- {c['t_val'][1]:.2f})")]:
                P(f"   rule {lab:38s}: hybrid Macro-F1 {100*c[f'hyb_macroF1_{name}'][0]:.2f} +- {100*c[f'hyb_macroF1_{name}'][1]:.2f}, "
                  f"L-vs-S acc {100*c[f'bin_acc_{name}'][0]:.2f}")

    # ---------------- D. errors by GT angle bin
    P("\n== D. Test error rate by GT axis angle bin (Leaned/Straight test images with GT keypoints; mean over seeds)")
    rep["D"] = {}
    test_items = {f: v for (s, f), v in gt.items() if s == "test" and v["label"] != 2 and v["angle"] is not None}
    hdr = [f"{BINS[i]}-{BINS[i+1]}" for i in range(len(BINS) - 1)]
    cnt = {h: [sum(1 for v in test_items.values() if v["label"] == c and BINS[i] <= v["angle"] < BINS[i + 1]) for c in (0, 1)]
           for i, h in enumerate(hdr)}
    P(" bins (deg)      : " + " | ".join(f"{h:>9s}" for h in hdr))
    P(" n Leaned/Straight: " + " | ".join(f"{cnt[h][0]:>4d}/{cnt[h][1]:<4d}" for h in hdr))
    rep["D"]["counts"] = cnt
    for batch, tags in {**TUNED, **DEFAULT}.items():
        for tag in tags:
            runs = load_runs(a.results, batch, tag)
            if not runs:
                continue
            errs = {h: [] for h in hdr}
            for r in runs:
                pred = dict(zip(r["test_files"], r["y_pred"]))
                for i, h in enumerate(hdr):
                    fs = [f for f, v in test_items.items() if BINS[i] <= v["angle"] < BINS[i + 1]]
                    if fs:
                        errs[h].append(np.mean([pred[f] != test_items[f]["label"] for f in fs]))
            rep["D"][tag] = {h: float(np.mean(e)) if e else None for h, e in errs.items()}
            P(f" {tag:20s}: " + " | ".join(f"{100*rep['D'][tag][h]:8.1f}%" if rep['D'][tag][h] is not None else "        -" for h in hdr))
    # Rejected and missing-kp breakdown for context
    n_nokp = sum(1 for (s, f), v in gt.items() if s == "test" and v["label"] != 2 and v["angle"] is None)
    P(f" (test Leaned/Straight images without GT keypoints: {n_nokp}; these are not in the table)")

    # ---------------- E. confusion matrices (summed over seeds)
    P("\n== E. Test confusion matrices, summed over seeds (rows = true L/S/R, cols = predicted L/S/R)")
    rep["E"] = {}
    for batch, tags in TUNED.items():
        for tag in tags:
            Cm = np.zeros((3, 3), int)
            for r in load_runs(a.results, batch, tag):
                for t_, p_ in zip(r["y_true"], r["y_pred"]):
                    Cm[t_, p_] += 1
            rep["E"][tag] = Cm.tolist()
            P(f" {tag:20s}: " + " | ".join(" ".join(f"{v:4d}" for v in row) for row in Cm)
              + f"   recall L/S/R: " + "/".join(f"{100*Cm[i,i]/Cm[i].sum():.1f}" for i in range(3)))

    json.dump(rep, open(os.path.join(a.out, "phase1.json"), "w"), indent=1)
    open(os.path.join(a.out, "phase1.txt"), "w").write("\n".join(L) + "\n")
    print("saved", a.out)


if __name__ == "__main__":
    main()
