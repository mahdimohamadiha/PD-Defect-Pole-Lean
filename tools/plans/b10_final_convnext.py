# ---- batch 10 (H4, pre-registered in README): FINAL evaluation of ConvNeXt-Base with the hyper-parameters selected
# in batch 9 (validation only). Each mode is re-trained with the final seeds 0-2 and evaluated ONCE on test.
BATCH = "b10_final_convnext"
CONVNEXT = "facebook/convnext-base-224-22k"
GATED = set()
CKPT_REPO = "northlattice/pd-defect-pole-lean-checkpoints"
SAVE_CKPT = True
BEST = json.load(open(f"{RESULTS}/b9_tune_convnext/best_configs.json"))["best"]

def _hp(mode):
    _b = BEST[mode]
    return ([f"--lr_bb={_b['lr_bb']}", f"--lr_head={_b['lr_head']}", f"--epochs={_b['epochs']}"]
            + ([f"--lam={_b['lam']}"] if mode != "cls" else []))

EXPERIMENTS = {
    "convnext_tuned_cls": ["--mode", "cls", "--backbone", CONVNEXT, *_hp("cls")],
    "convnext_tuned_mt": ["--mode", "mt", "--backbone", CONVNEXT, *_hp("mt")],
}
SEEDS = [0, 1, 2]
COMMON_ARGS = ["--workers", "12"]
RUN_DIR = f"{RESULTS}/{BATCH}"
CKPT_DIR = f"{WORK}/checkpoints/{BATCH}"
RUNS = [(t, s) for s in SEEDS for t in EXPERIMENTS]
ANALYSIS_CMD = ["analyze_results.py", "--results", f"{RESULTS}/b4a_final_dinov2", RUN_DIR, "--n_boot", "10000",
                "--out", f"{RUN_DIR}/analysis.json",
                "--pairs", "convnext_tuned_cls:convnext_tuned_mt", "dinov2_tuned_cls:convnext_tuned_cls",
                "dinov2_tuned_mt:convnext_tuned_mt"]
