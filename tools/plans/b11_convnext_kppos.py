# ---- batch 11 (H4b, exploratory, pre-registered in README before running): ConvNeXt mt with a 2-D sin-cos
# position code in the keypoint cross-attention (--kp_pos sincos). Same configuration as b10 mt (selected in b9 on
# validation only; no new search), seeds 0-2, tested once.
BATCH = "b11_convnext_kppos"
CONVNEXT = "facebook/convnext-base-224-22k"
GATED = set()
CKPT_REPO = "northlattice/pd-defect-pole-lean-checkpoints"
SAVE_CKPT = True
_b = json.load(open(f"{RESULTS}/b9_tune_convnext/best_configs.json"))["best"]["mt"]
EXPERIMENTS = {
    "convnext_kppos_mt": ["--mode", "mt", "--backbone", CONVNEXT, "--kp_pos", "sincos", f"--lr_bb={_b['lr_bb']}",
                          f"--lr_head={_b['lr_head']}", f"--epochs={_b['epochs']}", f"--lam={_b['lam']}"],
}
SEEDS = [0, 1, 2]
COMMON_ARGS = ["--workers", "12"]
RUN_DIR = f"{RESULTS}/{BATCH}"
CKPT_DIR = f"{WORK}/checkpoints/{BATCH}"
RUNS = [(t, s) for s in SEEDS for t in EXPERIMENTS]
ANALYSIS_CMD = ["analyze_results.py", "--results", f"{RESULTS}/b10_final_convnext", RUN_DIR, "--n_boot", "10000",
                "--out", f"{RUN_DIR}/analysis.json",
                "--pairs", "convnext_tuned_cls:convnext_kppos_mt", "convnext_tuned_mt:convnext_kppos_mt"]
