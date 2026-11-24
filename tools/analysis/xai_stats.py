"""X1 statistics on xai_rows.json (pre-registered in README).
Per image, metrics are averaged over seeds 0-2; comparisons are paired over the 143 images:
bootstrap 95% CI of the mean difference (10k) + Wilcoxon signed-rank; Holm over the 3 pre-registered pairs.
Usage: python tools/analysis/xai_stats.py --rows <xai_rows.json> --out results/analysis/xai"""
import os, json, argparse
import numpy as np
from scipy.stats import wilcoxon

PAIRS = [("dinov2_tuned_cls", "dinov2_tuned_mt"), ("dinov2reg_tuned_cls", "dinov2reg_tuned_mt"),
         ("dinov2_tuned_cls", "dinov2_soft_cls")]
METRICS = ["cam_in_box", "cam_in_box_noart", "att_in_box", "att_in_box_noart", "cam_hit", "att_hit"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", default="results/analysis/xai")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = json.load(open(a.rows))
    tags = sorted({r["tag"] for r in rows})
    files = sorted({r["file"] for r in rows})
    def per_image(tag, m):
        return np.array([np.nanmean([r[m] for r in rows if r["tag"] == tag and r["file"] == f]) for f in files])
    area = np.array([np.mean([r["box_area"] for r in rows if r["file"] == f]) for f in files])
    L, rep = [], {"n_images": len(files), "box_area_mean": float(area.mean())}
    def P(s): print(s); L.append(s)
    P(f"== X1: n = {len(files)} Leaned/Straight test images with a GT pole box; chance level (box area) = {100*area.mean():.1f}%")
    P(f"{'model':22s} " + " ".join(f"{m:>17s}" for m in METRICS) + "   n_art")
    for t in tags:
        v = {m: per_image(t, m) for m in METRICS}
        na = np.mean([r["n_art"] for r in rows if r["tag"] == t])
        rep[t] = {m: float(np.nanmean(v[m])) for m in METRICS} | {"n_art": float(na)}
        P(f"{t:22s} " + " ".join(f"{100*np.nanmean(v[m]):16.1f}%" for m in METRICS) + f"   {na:.1f}")
    rng = np.random.default_rng(0)
    for m in ["cam_in_box", "cam_in_box_noart", "att_in_box", "att_in_box_noart"]:
        P(f"\n-- paired differences (B - A), metric {m}" + (" [PRIMARY]" if m == "cam_in_box" else ""))
        ps = []
        for A, B in PAIRS:
            d = per_image(B, m) - per_image(A, m); d = d[~np.isnan(d)]
            bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
            p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
            ps.append(p)
            rep.setdefault("pairs", {}).setdefault(m, {})[f"{B}-{A}"] = {"diff": float(d.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "p": p}
        order = np.argsort(ps); adj = np.empty(3); run = 0
        for k, i in enumerate(order):
            run = max(run, min(1, (3 - k) * ps[i])); adj[i] = run
        for (A, B), p, ad in zip(PAIRS, ps, adj):
            r = rep["pairs"][m][f"{B}-{A}"]; r["holm_p"] = float(ad)
            P(f"   {B} - {A}: {100*r['diff']:+.2f} pts  CI95 [{100*r['ci'][0]:+.2f}, {100*r['ci'][1]:+.2f}]  Wilcoxon p={p:.4f}  Holm p={ad:.4f}")
    open(os.path.join(a.out, "xai_stats.txt"), "w").write("\n".join(L) + "\n")
    json.dump(rep, open(os.path.join(a.out, "xai_stats.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
