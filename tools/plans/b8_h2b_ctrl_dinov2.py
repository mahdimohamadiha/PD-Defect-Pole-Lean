# ---- batch 8 (H2b, pre-registered in README): (1) replication of H2 for cls on fresh seeds 3-5 (hard vs soft);
# (2) controls with the same images and the same target mass but no per-image angle information:
#     angle-shuffled soft targets and mass-matched uniform Leaned/Straight smoothing, seeds 0-5.
# DINOv2-base, batch-3 cls hyper-parameters, alpha 0.5 / t0 3.03 / tau 1 (as in b7), test once.
BATCH = "b8_h2b_ctrl_dinov2"
GATED = set()
CKPT_REPO = "northlattice/pd-defect-pole-lean-checkpoints"
SAVE_CKPT = True
BEST = json.load(open(f"{RESULTS}/b3_tune_dinov2/best_configs.json"))["best"]
_b = BEST["cls"]
HP = ["--mode", "cls", f"--lr_bb={_b['lr_bb']}", f"--lr_head={_b['lr_head']}", f"--epochs={_b['epochs']}"]
SOFT = ["--soft_alpha", "0.5", "--soft_t0", "3.03", "--soft_tau", "1.0"]
EXPERIMENTS = {
    "dinov2_tuned_cls": [*HP],                                         # same tag as b4a -> pooled with seeds 0-2
    "dinov2_soft_cls": [*HP, *SOFT],                                   # same tag as b7  -> pooled with seeds 0-2
    "dinov2_softshuf_cls": [*HP, *SOFT, "--soft_ctrl", "shuffle"],
    "dinov2_softunif_cls": [*HP, *SOFT, "--soft_ctrl", "uniform"],
}
SEEDS = [0, 1, 2, 3, 4, 5]
COMMON_ARGS = ["--workers", "12"]
RUN_DIR = f"{RESULTS}/{BATCH}"
CKPT_DIR = f"{WORK}/checkpoints/{BATCH}"
# replication runs first (seed-major), then the controls (seed-major)
RUNS = ([(t, s) for s in [3, 4, 5] for t in ["dinov2_tuned_cls", "dinov2_soft_cls"]]
        + [(t, s) for s in SEEDS for t in ["dinov2_softshuf_cls", "dinov2_softunif_cls"]])
ANALYSIS_CMD = ["analyze_results.py", "--results", f"{RESULTS}/b4a_final_dinov2", f"{RESULTS}/b7_h2_soft_dinov2", RUN_DIR,
                "--n_boot", "10000", "--out", f"{RUN_DIR}/analysis.json",
                "--pairs", "dinov2_tuned_cls:dinov2_soft_cls", "dinov2_softshuf_cls:dinov2_soft_cls",
                "dinov2_softunif_cls:dinov2_soft_cls", "dinov2_tuned_cls:dinov2_softshuf_cls",
                "dinov2_tuned_cls:dinov2_softunif_cls"]
