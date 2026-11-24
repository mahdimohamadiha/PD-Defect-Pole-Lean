"""X2 statistics (pre-registered in README), on xai2_rows.json (+ xai2_sanity.json).
Per image, metrics are averaged over seeds 0-2. Comparisons are paired over images: bootstrap 95% CI of the mean
difference (10k) + Wilcoxon signed-rank test, with Holm correction inside each family.
Usage: python tools/analysis/xai2_stats.py --dir results/marimo/x2_xai --out results/analysis/xai2"""
import os, json, argparse
import numpy as np
from scipy.stats import wilcoxon

METHODS = ["att", "rollout", "chefer", "gradcam"]
PRIMARY = [("dinov2_tuned_cls", "dinov2_tuned_mt"), ("dinov2reg_tuned_cls", "dinov2reg_tuned_mt")]
SECONDARY = [("dinov2_tuned_cls", "dinov2_soft_cls"), ("dinov2_tuned_mt", "dinov2_soft_mt"),
             ("dinov2_tuned_cls", "dinov2reg_tuned_cls"), ("dinov2_tuned_mt", "dinov2reg_tuned_mt")]
CL = ["Leaned", "Straight", "Rejected"]


def holm(ps):
    order = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for k, i in enumerate(order):
        run = max(run, min(1, (len(ps) - k) * ps[i])); adj[i] = run
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/marimo/x2_xai")
    ap.add_argument("--out", default="results/analysis/xai2")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = json.load(open(os.path.join(a.dir, "xai2_rows.json")))
    for r in rows:
        r["reliance"] = r["drop_pole"] - r["drop_rand"] if r["n_rand"] == 5 else np.nan
    tags = sorted({r["tag"] for r in rows})
    files = sorted({r["file"] for r in rows})
    def per_image(tag, key, only_full=False):
        out = []
        for f in files:
            v = [r[key] for r in rows if r["tag"] == tag and r["file"] == f and (r["n_rand"] == 5 or not only_full)]
            v = [x for x in v if x is not None and not (isinstance(x, float) and np.isnan(x))]
            out.append(np.mean(v) if v else np.nan)
        return np.array(out, float)
    rng = np.random.default_rng(0)
    def compare(A, B, key, only_full=False):
        d = per_image(B, key, only_full) - per_image(A, key, only_full); d = d[~np.isnan(d)]
        bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
        p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
        return {"diff": float(d.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "p": p, "n": int(len(d))}
    L, rep = [], {}
    def P(s=""): print(s); L.append(s)
    n_full = len({r["file"] for r in rows if r["n_rand"] == 5})
    P(f"== X2: {len(files)} Leaned/Straight test images with a GT pole box; {n_full} with 5 random control boxes")
    P("\n-- per model (mean over images of the seed-averaged value)")
    P(f"{'model':22s} {'reliance':>9s} {'drop pole':>9s} {'drop rand':>9s} {'flip pole':>9s} {'flip rand':>9s} {'acc':>6s} {'acc ctx':>7s}")
    for t in tags:
        rr = [r for r in rows if r["tag"] == t]
        acc = np.mean([r["pred"] == r["y"] for r in rr]); accc = np.mean([r["ctx_pred"] == r["y"] for r in rr])
        rel = np.nanmean(per_image(t, "reliance", True))
        rep.setdefault("models", {})[t] = {
            "reliance": float(rel), "drop_pole": float(np.nanmean(per_image(t, "drop_pole"))),
            "drop_rand": float(np.nanmean(per_image(t, "drop_rand", True))),
            "flip_pole": float(np.mean([r["flip_pole"] for r in rr])),
            "flip_rand": float(np.mean([r["flip_rand"] for r in rr if r["flip_rand"] is not None])),
            "acc": float(acc), "acc_context_only": float(accc)}
        m = rep["models"][t]
        P(f"{t:22s} {m['reliance']:9.3f} {m['drop_pole']:9.3f} {m['drop_rand']:9.3f} {100*m['flip_pole']:8.1f}% {100*m['flip_rand']:8.1f}% "
          f"{100*acc:5.1f}% {100*accc:6.1f}%")
    P("\n-- PRIMARY: pole reliance, mt - cls (Holm over 2)")
    res = [compare(A, B, "reliance", True) for A, B in PRIMARY]
    for (A, B), r, ad in zip(PRIMARY, res, holm([r["p"] for r in res])):
        r["holm_p"] = float(ad); rep.setdefault("primary", {})[f"{B}-{A}"] = r
        P(f"   {B} - {A}: {r['diff']:+.3f} CI95 [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]  Wilcoxon p={r['p']:.4f}  Holm p={ad:.4f}  (n={r['n']})")
    P("\n-- SECONDARY: pole reliance (two-sided, Holm over 4)")
    res = [compare(A, B, "reliance", True) for A, B in SECONDARY]
    for (A, B), r, ad in zip(SECONDARY, res, holm([r["p"] for r in res])):
        r["holm_p"] = float(ad); rep.setdefault("secondary", {})[f"{B}-{A}"] = r
        P(f"   {B} - {A}: {r['diff']:+.3f} CI95 [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]  Wilcoxon p={r['p']:.4f}  Holm p={ad:.4f}")
    P("\n-- attribution methods: energy-in-box and deletion faithfulness (area over the log-prob curve; higher = more faithful)")
    P(f"{'model':22s} " + " ".join(f"{'box_' + m:>12s}" for m in METHODS) + " | " + " ".join(f"{'aoc_' + m:>12s}" for m in METHODS + ["random"]))
    for t in tags:
        rr = [r for r in rows if r["tag"] == t]
        bx = {m: float(np.nanmean([r[f"in_box_{m}"] for r in rr])) for m in METHODS}
        ao = {m: float(np.mean([r[f"aoc_{m}"] for r in rr])) for m in METHODS + ["random"]}
        rep.setdefault("methods", {})[t] = {"in_box": bx, "aoc": ao}
        P(f"{t:22s} " + " ".join(f"{100*bx[m]:11.1f}%" for m in METHODS) + " | " + " ".join(f"{ao[m]:12.3f}" for m in METHODS + ["random"]))
    P("\n-- faithfulness, method - random order (all models and seeds pooled per image; paired over images)")
    allt = "__all__"
    for m in METHODS:
        d = np.array([np.mean([r[f"aoc_{m}"] - r["aoc_random"] for r in rows if r["file"] == f]) for f in files])
        bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
        rep.setdefault("faithfulness_vs_random", {})[m] = {"diff": float(d.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                                          "p": float(wilcoxon(d).pvalue)}
        r = rep["faithfulness_vs_random"][m]
        P(f"   {m:8s}: {r['diff']:+.3f} CI95 [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]  Wilcoxon p={r['p']:.4g}")
    best = max(METHODS, key=lambda m: rep["faithfulness_vs_random"][m]["diff"])
    rep["most_faithful_method"] = best
    P(f"   most faithful (largest mean gain over random): {best}")
    sp = os.path.join(a.dir, "xai2_sanity.json")
    if os.path.exists(sp):
        s = json.load(open(sp)); rep["sanity"] = s
        P("\n-- sanity check (head + last 4 blocks re-initialised; Spearman original vs randomised map, median over images;"
          " high = insensitive to the model)")
        for m in METHODS:
            P(f"   {m:8s}: {s[m]['median_spearman']:+.3f}")
    open(os.path.join(a.out, "xai2_stats.txt"), "w").write("\n".join(L) + "\n")
    json.dump(rep, open(os.path.join(a.out, "xai2_stats.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
