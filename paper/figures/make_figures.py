"""Builds every data figure of the paper from the saved results (no training).
Usage (repo root): python paper/make_figures.py --root <pole_lean_assessment> [--font <times.ttf>]
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
C1, C2, C3, GRAY, INK, MUTED, GRID, BAND = "#2a78d6", "#eb6834", "#1baf7a", "#9a9993", "#1a1a1a", "#5a5955", "#ebeae5", "#f4f3ef"


def setup_style(font):
    """Times New Roman everywhere (text and math). The font file is passed with --font and is NOT shipped
    with the repository (licence); without it, the closest serif font available is used."""
    from matplotlib import font_manager as fm
    family = "serif"
    if font and os.path.exists(font):
        fm.fontManager.addfont(font)
        family = fm.FontProperties(fname=font).get_name()
    plt.rcParams.update({
        "font.family": family, "mathtext.fontset": "custom", "mathtext.rm": family, "mathtext.it": family,
        "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.labelcolor": INK, "text.color": INK, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
        "legend.handlelength": 1.4, "legend.handletextpad": 0.5, "savefig.pad_inches": 0.02})


R = "results/marimo"
COL1, COL2 = 3.4, 7.0   # inches: single / double column


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
    fig, ax = plt.subplots(figsize=(COL1, 1.85))
    for lab, c, name in [(1, C1, "Straight"), (0, C2, "Leaned")]:
        ax.hist(np.clip(a[lab], 0, 19.99), bins=bins, color=c, alpha=0.85, label=name,
                edgecolor="white", linewidth=0.5)
    ax.axvline(3.03, color=INK, lw=0.9)
    ax.axvline(10, color=GRAY, lw=0.9, ls=(0, (3, 2)))
    ax.set_xlim(0, 20); ax.set_xticks([0, 3, 5, 10, 15, 20])
    ax.set_xlabel("Axis angle from vertical (°)"); ax.set_ylabel("Images")
    ax.legend(loc="upper right")
    save(fig, "label_angle_hist")


def fig_error_by_angle():
    txt = open("results/analysis/phase1/phase1.txt").read().splitlines()
    def row(tag):
        l = [x for x in txt if x.strip().startswith(tag + " ") and "%" in x][0]
        return [float(v.strip().rstrip("%")) for v in l.split(":", 1)[1].split("|")]
    labels = ["0–2", "2–4", "4–6", "6–10", "10–15", "≥15"]
    x = np.arange(len(labels)); w = 0.36
    fig, ax = plt.subplots(figsize=(COL1, 1.85))
    ax.axvspan(0.5, 2.5, color=BAND, zorder=0, lw=0)
    for i, (tag, c, name) in enumerate([("dinov2_tuned_cls", C1, "cls"), ("dinov2_tuned_mt", C2, "mt")]):
        ax.bar(x + (i - 0.5) * (w + 0.03), row(tag), w, color=c, label=name)
    ax.set_xticks(x, labels); ax.tick_params(axis="x", length=0)
    ax.set_xlabel("Axis angle (°)"); ax.set_ylabel("Error rate (%)")
    ax.set_ylim(0, 31); ax.set_yticks([0, 10, 20, 30])
    ax.legend(loc="upper right")
    save(fig, "error_by_angle")


def fig_forest():
    b4a, b4b, b5, b6, b8, b10, b11 = (f"{R}/{b}/analysis.json" for b in
                                      ["b4a_final_dinov2", "b4b_final_dinov2reg", "b5_h1_photo_dinov2",
                                       "b6_h3_dataeff_dinov2", "b8_h2b_ctrl_dinov2",
                                       "b10_final_convnext", "b11_convnext_kppos"])
    s345 = "results/analysis/h2b/seeds345.json"
    groups = [
        [("mt − cls, DINOv2", pair(b4a, "dinov2_tuned_cls", "dinov2_tuned_mt")),
         ("mt − cls, DINOv2-reg", pair(b4b, "dinov2reg_tuned_cls", "dinov2reg_tuned_mt")),
         ("mt − cls, ConvNeXt-B", pair(b10, "convnext_tuned_cls", "convnext_tuned_mt")),
         ("mt (+pos) − cls, ConvNeXt-B", pair(b11, "convnext_tuned_cls", "convnext_kppos_mt"))],
        [("Photo aug., cls", pair(b5, "dinov2_tuned_cls", "dinov2_photo_cls")),
         ("Photo aug., mt", pair(b5, "dinov2_tuned_mt", "dinov2_photo_mt"))],
        [("mt − cls, 10% data", pair(b6, "dinov2_f10_cls", "dinov2_f10_mt")),
         ("mt − cls, 25% data", pair(b6, "dinov2_f25_cls", "dinov2_f25_mt")),
         ("mt − cls, 50% data", pair(b6, "dinov2_f50_cls", "dinov2_f50_mt"))],
        [("Soft − hard, seeds 3–5", pair(s345, "dinov2_tuned_cls", "dinov2_soft_cls")),
         ("Soft − hard, seeds 0–5", pair(b8, "dinov2_tuned_cls", "dinov2_soft_cls")),
         ("Soft − shuffled", pair(b8, "dinov2_softshuf_cls", "dinov2_soft_cls")),
         ("Soft − uniform", pair(b8, "dinov2_softunif_cls", "dinov2_soft_cls"))]]
    n = sum(len(g) for g in groups) + len(groups) - 1
    fig, ax = plt.subplots(figsize=(COL1, 3.45))
    y, yt, yl = n, [], []
    for gi, g in enumerate(groups):
        top = y
        for name, (d, (lo, hi)) in g:
            y -= 1
            ax.plot([lo, hi], [y, y], color=C1, lw=1.6, solid_capstyle="round", zorder=3)
            ax.plot(d, y, "o", color=C1, ms=4.8, mec="white", mew=0.9, zorder=4)
            yt.append(y); yl.append(name)
        if gi % 2 == 0:
            ax.axhspan(y - 0.5, top - 0.5, color=BAND, lw=0, zorder=0)
        y -= 1
    ax.axvline(0, color=INK, lw=0.7, zorder=2)
    ax.set_yticks(yt, yl); ax.tick_params(axis="y", length=0)
    ax.set_xlim(-5.2, 7.0); ax.set_xticks([-4, -2, 0, 2, 4, 6]); ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("Δ Macro-F1 (points)")
    ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
    ax.spines["left"].set_visible(False)
    save(fig, "effects_forest")


def fig_data_efficiency():
    fr = [10, 25, 50, 100]
    fig, ax = plt.subplots(figsize=(COL1, 1.85))
    for mode, c, off in [("cls", C1, -0.015), ("mt", C2, 0.015)]:
        m, s = [], []
        for f in fr:
            rs = runs("b4a_final_dinov2", f"dinov2_tuned_{mode}") if f == 100 else runs("b6_h3_dataeff_dinov2", f"dinov2_f{f}_{mode}")
            v = 100 * np.array([r["test_macroF1"] for r in rs]); m.append(v.mean()); s.append(v.std())
        xs = np.array(fr) * (1 + off)
        ax.fill_between(xs, np.array(m) - s, np.array(m) + s, color=c, alpha=0.15, lw=0)
        ax.plot(xs, m, color=c, lw=1.8, marker="o", ms=4.5, mec="white", mew=0.9, label=mode)
    ax.set_xscale("log"); ax.set_xticks(fr, [f"{f}%" for f in fr]); ax.minorticks_off()
    ax.set_xlim(8.5, 118); ax.set_ylim(62, 89)
    ax.set_xlabel("Training data"); ax.set_ylabel("Macro-F1 (%)")
    ax.legend(loc="lower right")
    save(fig, "data_efficiency")


def fig_reliability():
    fig, ax = plt.subplots(figsize=(2.6, 2.6))
    ax.plot([0, 1], [0, 1], color=GRAY, lw=0.8, ls=(0, (3, 2)))
    sets = [("Hard", [("b4a_final_dinov2", "dinov2_tuned_cls"), ("b8_h2b_ctrl_dinov2", "dinov2_tuned_cls")], C1),
            ("Soft (angle)", [("b7_h2_soft_dinov2", "dinov2_soft_cls"), ("b8_h2b_ctrl_dinov2", "dinov2_soft_cls")], C2),
            ("Uniform", [("b8_h2b_ctrl_dinov2", "dinov2_softunif_cls")], C3)]
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
        ax.plot(xs, ys, color=c, lw=1.8, marker="o", ms=4, mec="white", mew=0.9, label=name)
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
    ax.set_xlim(0.4, 1.0); ax.set_ylim(0.4, 1.0); ax.set_aspect("equal")
    ax.set_xticks([0.4, 0.6, 0.8, 1.0]); ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.grid(axis="x", visible=True)
    ax.legend(loc="lower right")
    save(fig, "reliability")


def overlay(im, a, box, T, color=(255, 120, 40)):
    h = np.array(Image.fromarray((np.clip(np.array(a), 0, 1) * 255).astype("uint8")).resize((T, T), Image.BILINEAR)) / 255.0
    base = np.asarray(im, float) * 0.5
    col = np.array(color, float)
    o = base * (1 - h[..., None]) + col * h[..., None]
    o = Image.fromarray(np.clip(o, 0, 255).astype("uint8"))
    cx, cy, w, hh = box
    ImageDraw.Draw(o).rectangle([(cx - w / 2) * T, (cy - hh / 2) * T, (cx + w / 2) * T, (cy + hh / 2) * T], outline=(60, 220, 120), width=3)
    return o


def _clean(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def fig_xai(root):
    rows = json.load(open(f"{R}/xai_x1/xai_rows.json")); maps = json.load(open(f"{R}/xai_x1/xai_maps.json"))
    boxes = {os.path.basename(p): (p, v) for p, v in gt_boxes(root).items()}
    files = sorted({r["file"] for r in rows if r["tag"] == "dinov2_tuned_cls" and r["seed"] == 0})
    sel = [files[i] for i in np.random.default_rng(0).choice(len(files), 8, replace=False)][:4]   # first 4 of the pre-registered random 8
    T = 400; cols = [None, ("dinov2_tuned_cls_s0", "att"), ("dinov2_tuned_mt_s0", "att"), ("dinov2_tuned_cls_s0", "cam")]
    titles = ["Image", "cls attention", "mt attention", "cls Grad-CAM"]
    fig, axs = plt.subplots(len(sel), len(cols), figsize=(COL2 * 0.8, COL2 * 0.8 * len(sel) / len(cols) * 1.02))
    for i, f in enumerate(sel):
        p, (y, box) = boxes[f]
        im = Image.open(p).convert("RGB").resize((T, T), Image.BICUBIC)
        for j, c in enumerate(cols):
            ax = axs[i, j]
            ax.imshow(im if c is None else overlay(im, maps[c[0]][f][c[1]], box, T)); _clean(ax)
            if i == 0:
                ax.set_title(titles[j], fontsize=9, pad=4)
    plt.subplots_adjust(wspace=0.025, hspace=0.025)
    fig.savefig(f"{OUT}/xai_examples.png", dpi=300, bbox_inches="tight"); fig.savefig(f"{OUT}/xai_examples.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig); print("saved xai_examples")


def fig_xai2(root):
    """pre-registered X2 figure: most faithful method per architecture (Chefer for ViT, Grad-CAM for ConvNeXt-B), same 4 random images as X1."""
    z = np.load(f"{R}/x2_xai/xai2_maps.npz")
    zc = np.load(f"{R}/x2_convnext/xai2_maps.npz")
    rows = json.load(open(f"{R}/xai_x1/xai_rows.json"))
    boxes = {os.path.basename(p): (p, v) for p, v in gt_boxes(root).items()}
    files = sorted({r["file"] for r in rows if r["tag"] == "dinov2_tuned_cls" and r["seed"] == 0})
    sel = [files[i] for i in np.random.default_rng(0).choice(len(files), 8, replace=False)][:4]
    T = 400
    cols = [
        ("Image", None, None, None),
        ("DINOv2 cls", z, "dinov2_tuned_cls_s0", "chefer"),
        ("DINOv2 mt", z, "dinov2_tuned_mt_s0", "chefer"),
        ("DINOv2-reg cls", z, "dinov2reg_tuned_cls_s0", "chefer"),
        ("ConvNeXt-B cls", zc, "convnext_tuned_cls_s0", "gradcam"),
    ]
    ncols = len(cols)
    fig, axs = plt.subplots(len(sel), ncols, figsize=(COL2 * 0.96, COL2 * 0.96 * len(sel) / ncols * 1.03))
    for i, f in enumerate(sel):
        p, (y, box) = boxes[f]
        im = Image.open(p).convert("RGB").resize((T, T), Image.BICUBIC)
        for j, (title, npz, model, method) in enumerate(cols):
            ax = axs[i, j]
            if npz is None:
                ax.imshow(im)
            else:
                m = npz[f"{model}|{f}|{method}"].astype(np.float32)
                ax.imshow(overlay(im, m / (m.max() + 1e-12), box, T))
            _clean(ax)
            if i == 0:
                ax.set_title(title, fontsize=8.5, pad=4)
    plt.subplots_adjust(wspace=0.025, hspace=0.025)
    fig.savefig(f"{OUT}/xai_examples.png", dpi=300, bbox_inches="tight"); fig.savefig(f"{OUT}/xai_examples.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig); print("saved xai_examples (Chefer + ConvNeXt-B Grad-CAM, X2)")


def fig_leakage(root):
    pairs = ["1139", "2316", "1290"]    # identical / re-cropped / re-cropped and rotated
    T = 420
    fig, axs = plt.subplots(2, len(pairs), figsize=(2.9, 2.75))
    for j, st in enumerate(pairs):
        a = glob.glob(f"{root}/classification/test/*/{st}.*")[0]
        b = glob.glob(f"{root}/keypoint_detection/images/train/{st}.*")[0]
        for i, p in enumerate([a, b]):
            ax = axs[i, j]; im = Image.open(p).convert("RGB")
            w, h = im.size; k = min(w, h * 0.75)                      # common 3:4 crop for a tidy grid
            im = im.crop(((w - k) / 2, (h - k / 0.75) / 2, (w + k) / 2, (h + k / 0.75) / 2)).resize((int(T * 0.75), T))
            ax.imshow(im); _clean(ax)
        axs[1, j].set_xlabel(f"({'abc'[j]})", fontsize=9, labelpad=3)
    axs[0, 0].set_ylabel("Classification test", fontsize=8.5)
    axs[1, 0].set_ylabel("Keypoint train", fontsize=8.5)
    plt.subplots_adjust(wspace=0.04, hspace=0.04)
    fig.savefig(f"{OUT}/leakage_examples.png", dpi=300, bbox_inches="tight"); fig.savefig(f"{OUT}/leakage_examples.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig); print("saved leakage_examples")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True)
    ap.add_argument("--font", default=None, help="path to a Times New Roman .ttf (not included in the repository)")
    a = ap.parse_args()
    setup_style(a.font)
    os.makedirs(OUT, exist_ok=True)
    fig_forest(); fig_data_efficiency(); fig_reliability(); fig_error_by_angle()
    fig_label_angle(build_gt(a.root)); fig_xai2(a.root); fig_leakage(a.root)
