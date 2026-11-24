"""
H2 secondary outcomes (pre-registered): soft vs hard targets.
  * test error rate on Leaned/Straight images with GT keypoints in the 2-6 deg GT-angle band (descriptive, n = 37),
    and per angle bin;
  * calibration on the full test split: NLL and ECE (15 equal-width confidence bins), per seed then mean +- std,
    plus the seed-averaged probability ensemble.
GT keypoints of test images are used for evaluation only.
Usage: python tools/analysis/h2_soft.py --root <pole_lean_assessment> --results results/marimo
"""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from phase1_angle_analysis import build_gt, load_runs, BINS

TAGS = {"b4a_final_dinov2": ["dinov2_tuned_cls", "dinov2_tuned_mt"],
        "b7_h2_soft_dinov2": ["dinov2_soft_cls", "dinov2_soft_mt"],
        # H2b: pooled over batches by tag (seeds 0-5), and seeds 3-5 only
        ("b4a_final_dinov2", "b8_h2b_ctrl_dinov2"): ["dinov2_tuned_cls"],
        ("b7_h2_soft_dinov2", "b8_h2b_ctrl_dinov2"): ["dinov2_soft_cls"],
        "b8_h2b_ctrl_dinov2": ["dinov2_tuned_cls", "dinov2_soft_cls", "dinov2_softshuf_cls", "dinov2_softunif_cls"]}


def ece(prob, y, n_bins=15):
    conf, pred = prob.max(1), prob.argmax(1)
    e, edges = 0.0, np.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def nll(prob, y):
    return float(-np.mean(np.log(np.clip(prob[np.arange(len(y)), y], 1e-12, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--results", default="results/marimo")
    ap.add_argument("--out", default="results/analysis/h2")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    gt = build_gt(a.root)
    L, rep = [], {}
    def P(s): print(s); L.append(s)
    P("== H2 secondary outcomes (test)")
    P(f"{'model[seeds]':34s} | {'err 2-6deg':>10s} | " + " | ".join(f"{BINS[i]}-{BINS[i+1]}" .rjust(6) for i in range(len(BINS) - 1))
      + f" | {'NLL':>12s} | {'ECE %':>12s} | ens NLL | ens ECE%")
    for batch, tags in TAGS.items():
        for tag in tags:
            bs = batch if isinstance(batch, tuple) else (batch,)
            rs = sorted([r for b in bs for r in load_runs(a.results, b, tag)], key=lambda r: r["seed"])
            if not rs:
                continue
            name = f"{tag}[{','.join(str(r['seed']) for r in rs)}]"
            files = rs[0]["test_files"]
            y = np.array(rs[0]["y_true"])
            ang = np.array([np.nan if gt[("test", f)]["angle"] is None else gt[("test", f)]["angle"] for f in files])
            ls = (y != 2) & ~np.isnan(ang)
            band = ls & (ang >= 2) & (ang < 6)
            err_band, err_bins, nl, ec, probs = [], [[] for _ in BINS[:-1]], [], [], []
            for r in rs:
                assert r["test_files"] == files
                p = np.array(r["test_probs"]); yp = np.array(r["y_pred"])
                probs.append(p)
                err_band.append(100 * np.mean(yp[band] != y[band]))
                for i in range(len(BINS) - 1):
                    m = ls & (ang >= BINS[i]) & (ang < BINS[i + 1])
                    err_bins[i].append(100 * np.mean(yp[m] != y[m]) if m.any() else np.nan)
                nl.append(nll(p, y)); ec.append(100 * ece(p, y))
            pe = np.mean(probs, 0)
            rep[name] = {"n_seeds": len(rs), "n_band": int(band.sum()), "err_band": err_band,
                        "err_bins": [float(np.mean(b)) for b in err_bins], "nll": nl, "ece": ec,
                        "ens_nll": nll(pe, y), "ens_ece": 100 * ece(pe, y)}
            P(f"{name:34s} | {np.mean(err_band):9.1f}% | " + " | ".join(f"{np.mean(b):5.1f}%" for b in err_bins)
              + f" | {np.mean(nl):.3f}+-{np.std(nl):.3f} | {np.mean(ec):5.2f}+-{np.std(ec):4.2f} |"
              + f" {rep[name]['ens_nll']:.3f}  | {rep[name]['ens_ece']:.2f}")
    P(f"(2-6 deg band: n = {band.sum()} Leaned/Straight test images with GT keypoints; descriptive only)")
    open(os.path.join(a.out, "h2_soft.txt"), "w").write("\n".join(L) + "\n")
    json.dump(rep, open(os.path.join(a.out, "h2_soft.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
