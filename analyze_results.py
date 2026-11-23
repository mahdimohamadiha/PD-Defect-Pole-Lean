"""
Statistical analysis of PD-Defect pole-lean runs.

Reads res_<mode>_s<seed>.json files written by train_multitask.py (each holds y_true, y_pred,
test_files) and reports:
  1. per-mode mean +- std over seeds (Acc, Macro-F1, per-class F1);
  2. 95% paired bootstrap CIs (resampling the 227 test images) for each mode's seed-averaged
     metric, and for the difference between modes (seed-averaged, same resample for both);
  3. exact McNemar tests between modes, per seed and on the seed-majority-vote predictions,
     with Holm correction over the mode pairs;
  4. CI of each mode vs the dataset-card baseline point estimate (no baseline predictions exist,
     so only "is the reference value inside our CI" can be stated).

Usage:
  python analyze_results.py --results /marimo/results [--n_boot 10000] [--out analysis.json]
"""
import os, re, glob, json, math, argparse, itertools
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

CLASSES = ["Leaned", "Straight", "Rejected"]
BASELINE = {"acc": 0.8414, "macroF1": 0.8174}   # dataset card, DINOv3 ViT-B/16


def load(results_dir):
    runs = {}   # mode -> {seed: (y_true, y_pred, files)}
    for f in sorted(glob.glob(os.path.join(results_dir, "res_*.json"))):
        for r in json.load(open(f)):
            if "y_pred" not in r:
                raise ValueError(f"{f}: no predictions saved (older script version)")
            runs.setdefault(r["mode"], {})[r["seed"]] = (np.array(r["y_true"]), np.array(r["y_pred"]),
                                                         r.get("test_files"))
    ref = None
    for mode, d in runs.items():   # every run must be on the same test images in the same order
        for seed, (yt, _, files) in d.items():
            key = (tuple(yt), tuple(files) if files else None)
            if ref is None:
                ref = key
            elif key != ref:
                raise ValueError(f"{mode} seed {seed}: test set differs from the other runs")
    return runs


def macro_f1_fast(yt, yp, k=3):
    """Macro-F1 over k classes; matches sklearn's average='macro' (zero_division=0) on these labels."""
    f1s = []
    for c in range(k):
        tp = np.sum((yp == c) & (yt == c)); fp = np.sum((yp == c) & (yt != c)); fn = np.sum((yp != c) & (yt == c))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def metrics(yt, yp):
    return {"acc": float(np.mean(yt == yp)), "macroF1": macro_f1_fast(yt, yp)}


def majority_vote(preds):
    """preds [S, N] -> [N]; ties broken by the lowest seed's prediction."""
    P = np.stack(preds)
    out = []
    for j in range(P.shape[1]):
        cnt = np.bincount(P[:, j], minlength=3)
        best = np.flatnonzero(cnt == cnt.max())
        out.append(P[0, j] if P[0, j] in best else best[0])
    return np.array(out)


def mcnemar_exact(yt, pa, pb):
    """Exact two-sided McNemar (binomial on discordant pairs). Returns b, c, p."""
    a_ok, b_ok = pa == yt, pb == yt
    b = int(np.sum(a_ok & ~b_ok)); c = int(np.sum(~a_ok & b_ok)); n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return b, c, min(1.0, p)


def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m); running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i]); adj[i] = min(1.0, running)
    return adj.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/marimo/results")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    runs = load(a.results)
    modes = [m for m in ["cls", "mt", "mt_geo"] if m in runs] + sorted(set(runs) - {"cls", "mt", "mt_geo"})
    yt = next(iter(next(iter(runs.values())).values()))[0]
    N = len(yt)
    rng = np.random.default_rng(a.seed)
    idx = rng.integers(0, N, size=(a.n_boot, N))      # one shared resample set -> paired comparisons
    report = {"n_test": N, "n_boot": a.n_boot, "modes": {}, "pairs": []}

    print(f"test images: {N} | bootstrap resamples: {a.n_boot}\n")
    print("== 1-2. per mode (mean +- std over seeds; 95% bootstrap CI of the seed-averaged metric)")
    boot = {}
    for m in modes:
        seeds = sorted(runs[m]); P = [runs[m][s][1] for s in seeds]
        per = [metrics(yt, p) for p in P]
        pcf1 = np.array([f1_score(yt, p, average=None, labels=[0, 1, 2], zero_division=0) for p in P])
        bs = {k: np.zeros(a.n_boot) for k in ["acc", "macroF1"]}
        for p in P:
            ok = (p == yt)
            bs["acc"] += ok[idx].mean(1) / len(P)
            bs["macroF1"] += np.array([macro_f1_fast(yt[i], p[i]) for i in idx]) / len(P)
        boot[m] = bs
        mv = majority_vote(P)
        rec = {"seeds": seeds}
        for k in ["acc", "macroF1"]:
            v = np.array([r[k] for r in per]); lo, hi = np.percentile(bs[k], [2.5, 97.5])
            rec[k] = {"mean": v.mean(), "std": v.std(ddof=1) if len(v) > 1 else 0.0, "ci95": [lo, hi],
                      "baseline_in_ci": bool(lo <= BASELINE[k] <= hi), "per_seed": v.tolist()}
        rec["per_class_F1_mean"] = dict(zip(CLASSES, pcf1.mean(0).tolist()))
        rec["majority_vote"] = metrics(yt, mv)
        report["modes"][m] = rec
        print(f"{m:7s} seeds={seeds}")
        for k in ["acc", "macroF1"]:
            r = rec[k]
            print(f"   {k:8s} {100*r['mean']:.2f} +- {100*r['std']:.2f}  CI95 [{100*r['ci95'][0]:.2f}, {100*r['ci95'][1]:.2f}]"
                  f"  card baseline {100*BASELINE[k]:.2f} {'inside' if r['baseline_in_ci'] else 'OUTSIDE'} CI")
        print("   per-class F1:", {c: round(100 * v, 2) for c, v in rec["per_class_F1_mean"].items()},
              "| majority vote:", {k: round(100 * v, 2) for k, v in rec["majority_vote"].items()})

    print("\n== 2-3. pairwise: bootstrap CI of the difference (B - A) + exact McNemar")
    pairs = list(itertools.combinations(modes, 2))
    p_mv = []
    for A, B in pairs:
        rec = {"A": A, "B": B}
        for k in ["acc", "macroF1"]:
            d = boot[B][k] - boot[A][k]; lo, hi = np.percentile(d, [2.5, 97.5])
            obs = report["modes"][B][k]["mean"] - report["modes"][A][k]["mean"]
            rec[f"diff_{k}"] = {"obs": obs, "ci95": [lo, hi], "p_boot_le0": float(np.mean(d <= 0))}
        common = sorted(set(runs[A]) & set(runs[B]))
        rec["mcnemar_per_seed"] = {s: mcnemar_exact(yt, runs[A][s][1], runs[B][s][1]) for s in common}
        mvA = majority_vote([runs[A][s][1] for s in sorted(runs[A])])
        mvB = majority_vote([runs[B][s][1] for s in sorted(runs[B])])
        rec["mcnemar_majority_vote"] = mcnemar_exact(yt, mvA, mvB)
        p_mv.append(rec["mcnemar_majority_vote"][2])
        report["pairs"].append(rec)
    for rec, padj in zip(report["pairs"], holm(p_mv) if p_mv else []):
        rec["mcnemar_majority_vote_holm"] = padj
        print(f"{rec['B']} vs {rec['A']}:")
        for k in ["acc", "macroF1"]:
            r = rec[f"diff_{k}"]
            print(f"   d{k:8s} {100*r['obs']:+.2f}  CI95 [{100*r['ci95'][0]:+.2f}, {100*r['ci95'][1]:+.2f}]"
                  f"  P(diff<=0)={r['p_boot_le0']:.3f}")
        ps = ", ".join(f"s{s}: b={b} c={c} p={p:.3f}" for s, (b, c, p) in rec["mcnemar_per_seed"].items())
        b, c, p = rec["mcnemar_majority_vote"]
        print(f"   McNemar per seed  {ps}")
        print(f"   McNemar maj.vote  b={b} c={c} p={p:.4f}  Holm-adj p={padj:.4f}")
        print("   (b = A right & B wrong, c = A wrong & B right)")
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1, default=float)
        print("\nsaved", a.out)


if __name__ == "__main__":
    main()
