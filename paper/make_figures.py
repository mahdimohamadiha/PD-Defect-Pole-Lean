"""Builds every data figure of the paper from the saved results (no training).
Usage (repo root): python paper/make_figures.py --root <pole_lean_assessment>
Writes vector PDF (plots) and 300-dpi PNG (image panels) to paper/figures/."""
import os, sys, json, glob, math, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

sys.path.insert(0, "tools/analysis")
from phase1_angle_analysis import build_gt
from xai_attention import gt_boxes

OUT = "paper/figures"
C1, C2, C3, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": MUTED,
                     "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK,
                     "text.color": INK, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.5, "axes.axisbelow": True})
R = "results/marimo"
COL1, COL2 = 3.5, 7.2   # inches: single / double column


def runs(batch, tag):
    rs = []
    for f in sorted(glob.glob(f"{R}/{batch}/res_{tag}_s*.json")):
        rs += [r for r in json.load(open(f)) if r.get("tag") == tag]
    return rs


def pair(path, A, B):
    for p in json.load(open(path))["pairs"]:
        if (p["A"], p["B"]) == (A, B):
            return 100 * p["diff_macroF1"]["obs"], [100 * x for x in p["diff_macroF1"]["ci95"]]
        if (p["A"], p["B"]) == (B, A):
            return -100 * p["diff_macroF1"]["obs"], [-100 * x for x in p["diff_macroF1"]["ci95"][::-1]]
    raise KeyError((path, A, B))


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def fig_label_angle(gt):
    a = {0: [], 1: []}
    for (sp, f), v in gt.items():
        if sp == "train" and v["label"] in (0, 1) and v["angle"] is not None:
            a[v["label"]].append(v["angle"])
    bins = np.arange(0, 20.5, 0.5)
    fig, ax = plt.subplots(figsize=(COL1, 1.9))
    for lab, c, name in [(1, C1, "Straight"), (0, C2, "Leaned")]:
        x = np.clip(a[lab], 0, 19.99)
        ax.hist(x, bins=bins, color=c, alpha=0.75, label=f"{name} (n = {len(x)})", edgecolor="white", linewidth=0.4)
    for t, s in [(3.03, "3.03° (fitted)"), (10, "10° (prior rule)")]:
        ax.axvline(t, color=INK if t < 5 else GRAY, ls="--", lw=0.8)
        ax.text(t + 0.3, ax.get_ylim()[1] * 0.92, s, fontsize=6.5, color=MUTED, va="top")
    ax.set_xlabel("ground-truth axis angle from vertical (°; last bin includes ≥ 20°)")
    ax.set_ylabel("train images")
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1, 0.8))
    save(fig, "label_angle_hist")


def fig_error_by_angle():
    ph = json.load(open("results/analysis/phase1/phase1.json"))
    txt = open("results/analysis/phase1/phase1.txt").read().splitlines()
    def row(tag):
        l = [x for x in txt if x.strip().startswith(tag + " ") and "%" in x][0]
        return [float(v.strip().rstrip("%")) for v in l.split(":", 1)[1].split("|")]
    labels = ["0–2", "2–4", "4–6", "6–10", "10–15", "≥15"]
    n = [84, 27, 10, 9, 7, 6]
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(COL1, 1.8))
    ax.axvspan(0.5, 2.5, color="#f3f2ee", zorder=0)
    for i, (tag, c, name) in enumerate([("dinov2_tuned_cls", C1, "cls"), ("dinov2_tuned_mt", C2, "mt")]):
        ax.bar(x + (i - 0.5) * w, row(tag), w * 0.92, color=c, label=name)
    ax.set_xticks(x, [f"{l}\nn={k}" for l, k in zip(labels, n)])
    ax.set_xlabel("ground-truth axis angle (°)"); ax.set_ylabel("test error rate (%)")
    ax.text(1.5, 30.5, "ambiguous band", ha="center", fontsize=6.5, color=MUTED)
    ax.set_ylim(0, 33)
    ax.legend(frameon=False, loc="upper right")
    save(fig, "error_by_angle")


def fig_forest():
    b4a, b4b, b5, b6, b8 = (f"{R}/{b}/analysis.json" for b in
                            ["b4a_final_dinov2", "b4b_final_dinov2reg", "b5_h1_photo_dinov2", "b6_h3_dataeff_dinov2", "b8_h2b_ctrl_dinov2"])
    s345 = "results/analysis/h2b/seeds345.json"
    rows = [("Main comparison: keypoint supervision", None),
            ("DINOv2: mt − cls", pair(b4a, "dinov2_tuned_cls", "dinov2_tuned_mt")),
            ("DINOv2-reg: mt − cls", pair(b4b, "dinov2reg_tuned_cls", "dinov2reg_tuned_mt")),
            ("H1: photometric augmentation", None),
            ("cls: photo − base", pair(b5, "dinov2_tuned_cls", "dinov2_photo_cls")),
            ("mt: photo − base", pair(b5, "dinov2_tuned_mt", "dinov2_photo_mt")),
            ("H3: mt − cls with less training data", None),
            ("10% of train", pair(b6, "dinov2_f10_cls", "dinov2_f10_mt")),
            ("25% of train", pair(b6, "dinov2_f25_cls", "dinov2_f25_mt")),
            ("50% of train", pair(b6, "dinov2_f50_cls", "dinov2_f50_mt")),
            ("H2/H2b: soft targets (cls)", None),
            ("soft − hard, new seeds 3–5", pair(s345, "dinov2_tuned_cls", "dinov2_soft_cls")),
            ("soft − hard, seeds 0–5", pair(b8, "dinov2_tuned_cls", "dinov2_soft_cls")),
            ("soft − shuffled-angle control", pair(b8, "dinov2_softshuf_cls", "dinov2_soft_cls")),
            ("soft − uniform-smoothing control", pair(b8, "dinov2_softunif_cls", "dinov2_soft_cls"))]
    fig, ax = plt.subplots(figsize=(COL1, 3.4))
    y = len(rows)
    yt, yl = [], []
    for name, v in rows:
        y -= 1
        if v is None:
            yt.append(y); yl.append(r"$\bf{" + name.replace(" ", r"\ ").replace("%", r"\%") + "}$")
            continue
        d, (lo, hi) = v
        ax.plot([lo, hi], [y, y], color=C1, lw=1.4, solid_capstyle="round")
        ax.plot(d, y, "o", color=C1, ms=4.5, mec="white", mew=0.8)
        fmt = lambda z: f"{z:+.1f}".replace("-0.0", "0.0").replace("+0.0", "0.0").replace("-", "−")
        ax.text(6.6, y, f"{fmt(d)} [{fmt(lo)}, {fmt(hi)}]", fontsize=6.3, va="center", color=MUTED)
        yt.append(y); yl.append(name)
    ax.axvline(0, color=INK, lw=0.7)
    ax.set_yticks(yt, yl); ax.tick_params(axis="y", length=0)
    ax.set_xlim(-9, 6.4); ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel("Δ test Macro-F1 (points), paired bootstrap 95% CI")
    ax.grid(axis="y", visible=False)
    save(fig, "effects_forest")


def fig_data_efficiency():
    fr = [10, 25, 50, 100]
    fig, ax = plt.subplots(figsize=(COL1, 1.9))
    for mode, c, off in [("cls", C1, -0.012), ("mt", C2, 0.012)]:
        m, s = [], []
        for f in fr:
            rs = runs("b4a_final_dinov2", f"dinov2_tuned_{mode}") if f == 100 else runs("b6_h3_dataeff_dinov2", f"dinov2_f{f}_{mode}")
            v = 100 * np.array([r["test_macroF1"] for r in rs]); m.append(v.mean()); s.append(v.std())
        xs = np.array(fr) * (1 + off)
        ax.errorbar(xs, m, yerr=s, color=c, lw=2, marker="o", ms=4.5, mec="white", mew=0.8, capsize=2, label=mode)
    ax.set_xscale("log"); ax.set_xticks(fr, [f"{f}%\n({n})" for f, n in zip(fr, [170, 426, 853, 1706])])
    ax.minorticks_off()
    ax.set_xlabel("fraction of the training split (images)"); ax.set_ylabel("test Macro-F1 (%)")
    ax.set_xlim(8, 135)
    ax.legend(frameon=False, loc="lower right")
    save(fig, "data_efficiency")


def fig_reliability():
    fig, ax = plt.subplots(figsize=(COL1, 2.6))
    ax.plot([0, 1], [0, 1], color=GRAY, lw=0.8, ls="--")
    sets = [("hard", [("b4a_final_dinov2", "dinov2_tuned_cls"), ("b8_h2b_ctrl_dinov2", "dinov2_tuned_cls")], C1),
            ("soft (GT angle)", [("b7_h2_soft_dinov2", "dinov2_soft_cls"), ("b8_h2b_ctrl_dinov2", "dinov2_soft_cls")], C2),
            ("uniform smoothing", [("b8_h2b_ctrl_dinov2", "dinov2_softunif_cls")], C3)]
    for name, src, c in sets:
        P, Y = [], []
        for b, t in src:
            for r in runs(b, t):
                P.append(np.array(r["test_probs"])); Y.append(np.array(r["y_true"]))
        P, Y = np.concatenate(P), np.concatenate(Y)
        conf, cor = P.max(1), P.argmax(1) == Y
        order = np.argsort(conf); xs, ys = [], []
        for chunk in np.array_split(order, 8):          # equal-mass bins
            xs.append(conf[chunk].mean()); ys.append(cor[chunk].mean())
        ece = sum(((conf > lo) & (conf <= hi)).mean() * abs(cor[(conf > lo) & (conf <= hi)].mean() - conf[(conf > lo) & (conf <= hi)].mean())
                  for lo, hi in zip(np.linspace(0, 1, 16)[:-1], np.linspace(0, 1, 16)[1:]) if ((conf > lo) & (conf <= hi)).any())
        ax.plot(xs, ys, color=c, lw=2, marker="o", ms=4, mec="white", mew=0.8, label=f"{name} (ECE {100*ece:.1f}%)")
    ax.set_xlabel("confidence (max. softmax)"); ax.set_ylabel("accuracy")
    ax.set_xlim(0.3, 1.0); ax.set_ylim(0.3, 1.02); ax.set_aspect("equal")
    ax.legend(frameon=False, loc="lower right")
    save(fig, "reliability")


def overlay(im, a, box, T, color=(235, 104, 52)):
    h = np.array(Image.fromarray((np.clip(np.array(a), 0, 1) * 255).astype("uint8")).resize((T, T), Image.BILINEAR)) / 255.0
    base = np.asarray(im, float) * 0.55
    col = np.array(color, float)
    o = base * (1 - h[..., None]) + col * h[..., None]
    o = Image.fromarray(np.clip(o, 0, 255).astype("uint8"))
    cx, cy, w, hh = box
    ImageDraw.Draw(o).rectangle([(cx - w / 2) * T, (cy - hh / 2) * T, (cx + w / 2) * T, (cy + hh / 2) * T], outline=(40, 200, 90), width=3)
    return o


def fig_xai(root):
    rows = json.load(open(f"{R}/xai_x1/xai_rows.json")); maps = json.load(open(f"{R}/xai_x1/xai_maps.json"))
    boxes = {os.path.basename(p): (p, v) for p, v in gt_boxes(root).items()}
    files = sorted({r["file"] for r in rows if r["tag"] == "dinov2_tuned_cls" and r["seed"] == 0})
    sel = [files[i] for i in np.random.default_rng(0).choice(len(files), 8, replace=False)][:4]   # first 4 of the pre-registered random 8
    T = 360; cols = [None, ("dinov2_tuned_cls_s0", "att"), ("dinov2_tuned_mt_s0", "att"), ("dinov2_tuned_cls_s0", "cam")]
    fig, axs = plt.subplots(len(sel), len(cols), figsize=(COL2 * 0.78, COL2 * 0.78 * len(sel) / len(cols)))
    for i, f in enumerate(sel):
        p, (y, box) = boxes[f]
        im = Image.open(p).convert("RGB").resize((T, T), Image.BICUBIC)
        for j, c in enumerate(cols):
            ax = axs[i, j]
            ax.imshow(im if c is None else overlay(im, maps[c[0]][f][c[1]], box, T)); ax.set_xticks([]); ax.set_yticks([])
            ax.grid(False); [s.set_visible(False) for s in ax.spines.values()]
            if i == 0:
                ax.set_title(["image", "cls: CLS attention", "mt: CLS attention", "cls: Grad-CAM"][j], fontsize=7.5)
    plt.subplots_adjust(wspace=0.03, hspace=0.03)
    fig.savefig(f"{OUT}/xai_examples.png", dpi=300, bbox_inches="tight"); fig.savefig(f"{OUT}/xai_examples.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig); print("saved xai_examples")


def fig_leakage(root):
    pairs = [("1139", "identical"), ("2316", "re-cropped"), ("1290", "re-cropped, rotated")]
    T = 300
    fig, axs = plt.subplots(2, len(pairs), figsize=(COL1, COL1 * 2 / len(pairs) * 1.05))
    for j, (st, kind) in enumerate(pairs):
        a = glob.glob(f"{root}/classification/test/*/{st}.*")
        b = glob.glob(f"{root}/keypoint_detection/images/train/{st}.*")
        if not a or not b:
            continue
        for i, p in enumerate([a[0], b[0]]):
            ax = axs[i, j]; im = Image.open(p).convert("RGB"); im.thumbnail((T, T))
            ax.imshow(im); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if j == 0:
                ax.set_ylabel(["classification test", "keypoint train"][i], fontsize=7)
        axs[0, j].set_title(kind, fontsize=7)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.savefig(f"{OUT}/leakage_examples.png", dpi=300, bbox_inches="tight"); fig.savefig(f"{OUT}/leakage_examples.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig); print("saved leakage_examples")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    fig_forest(); fig_data_efficiency(); fig_reliability(); fig_error_by_angle()
    fig_label_angle(build_gt(a.root)); fig_xai(a.root); fig_leakage(a.root)
