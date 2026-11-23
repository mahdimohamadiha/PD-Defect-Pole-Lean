"""
Summarise an equal-budget hyper-parameter search (validation split only).

Each run is a res_<tag>_s<seed>.json written by train_multitask.py --no_test, with tag = <mode>_t<NN>.
For every (mode, trial) the mean +- std validation Macro-F1 over seeds is reported; the best trial per mode
(highest mean val Macro-F1, ties -> fewer epochs, then lower trial id) is written to --out as the config to
re-train with the final seeds and evaluate once on test.

Usage: python tune_summary.py --results <dir> [--out best_configs.json]
"""
import os, re, glob, json, argparse
import numpy as np

HP = ["lr_bb", "lr_head", "epochs", "lam"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    groups = {}
    for f in sorted(glob.glob(os.path.join(a.results, "res_*.json"))):
        for r in json.load(open(f)):
            assert "test_acc" not in r, f"{f}: search runs must not evaluate on test"
            m = re.fullmatch(r"(.+)_t(\d+)", r["tag"])
            if not m:
                continue
            g = groups.setdefault((r["mode"], int(m.group(2))), {"hp": {}, "f1": {}, "acc": {}, "best_ep": {}})
            g["hp"] = {k: r["args"][k] for k in HP if not (k == "lam" and r["mode"] == "cls")}
            g["f1"][r["seed"]] = r["val_macroF1"]; g["acc"][r["seed"]] = r["val_acc"]; g["best_ep"][r["seed"]] = r["best_epoch"]
    best, report = {}, {}
    for mode in sorted({m for m, _ in groups}):
        rows = []
        for (m, t), g in sorted(groups.items()):
            if m != mode:
                continue
            f1 = np.array(list(g["f1"].values())); acc = np.array(list(g["acc"].values()))
            rows.append({"trial": t, **g["hp"], "n_seeds": len(f1), "val_macroF1_mean": float(f1.mean()),
                         "val_macroF1_std": float(f1.std(ddof=1)) if len(f1) > 1 else 0.0,
                         "val_acc_mean": float(acc.mean()), "best_epochs": list(g["best_ep"].values())})
        n_max = max(r["n_seeds"] for r in rows)
        complete = [r for r in rows if r["n_seeds"] == n_max]   # rank only trials with all seeds finished
        complete.sort(key=lambda r: (-r["val_macroF1_mean"], r["epochs"], r["trial"]))
        report[mode] = rows; best[mode] = complete[0]
        print(f"== {mode}: {len(rows)} trials ({len(complete)} with {n_max} seeds)")
        hdr = ["trial"] + [k for k in HP if k in rows[0]] + ["val F1 mean", "std", "val acc", "best ep"]
        print("  " + " | ".join(f"{h:>11s}" for h in hdr))
        for r in sorted(rows, key=lambda r: -r["val_macroF1_mean"]):
            vals = [str(r["trial"])] + [f"{r[k]:g}" for k in HP if k in r] + [
                f"{100*r['val_macroF1_mean']:.2f}", f"{100*r['val_macroF1_std']:.2f}", f"{100*r['val_acc_mean']:.2f}",
                ",".join(map(str, r["best_epochs"]))]
            print(("* " if r is best[mode] else "  ") + " | ".join(f"{v:>11s}" for v in vals))
    print("\nbest config per mode (selected on validation only):")
    for m, r in best.items():
        print(f"  {m}: trial {r['trial']} " + ", ".join(f"{k}={r[k]:g}" for k in HP if k in r)
              + f"  (val F1 {100*r['val_macroF1_mean']:.2f} +- {100*r['val_macroF1_std']:.2f})")
    if a.out:
        json.dump({"best": best, "all": report}, open(a.out, "w"), indent=1)
        print("saved", a.out)


if __name__ == "__main__":
    main()
