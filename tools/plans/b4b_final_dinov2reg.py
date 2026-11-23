# ---- batch 4b: FINAL evaluation of DINOv2-with-registers with the hyper-parameters selected in batch 3b
# (validation only). Each mode is re-trained with the final seeds 0-2 and evaluated ONCE on test.
BATCH = "b4b_final_dinov2reg"
DINOV2_REG = "facebook/dinov2-with-registers-base"
GATED = set()
CKPT_REPO = "northlattice/pd-defect-pole-lean-checkpoints"
SAVE_CKPT = True
BEST = json.load(open(f"{RESULTS}/b3b_tune_dinov2reg/best_configs.json"))["best"]

def _hp(mode):
    _b = BEST[mode]
    return ([f"--lr_bb={_b['lr_bb']}", f"--lr_head={_b['lr_head']}", f"--epochs={_b['epochs']}"]
            + ([f"--lam={_b['lam']}"] if mode != "cls" else []))

EXPERIMENTS = {
    "dinov2reg_tuned_cls": ["--mode", "cls", "--backbone", DINOV2_REG, *_hp("cls")],
    "dinov2reg_tuned_mt": ["--mode", "mt", "--backbone", DINOV2_REG, *_hp("mt")],
}
SEEDS = [0, 1, 2]
COMMON_ARGS = ["--workers", "12"]
RUN_DIR = f"{RESULTS}/{BATCH}"
CKPT_DIR = f"{WORK}/checkpoints/{BATCH}"
RUNS = [(t, s) for s in SEEDS for t in EXPERIMENTS]
ANALYSIS_CMD = ["analyze_results.py", "--results", f"{RESULTS}/b2_dinov2reg_geofix", f"{RESULTS}/b4a_final_dinov2", RUN_DIR, "--n_boot", "10000",
                "--out", f"{RUN_DIR}/analysis.json",
                "--pairs", "dinov2reg_tuned_cls:dinov2reg_tuned_mt", "dinov2reg_cls:dinov2reg_tuned_cls",
                "dinov2reg_mt:dinov2reg_tuned_mt", "dinov2_tuned_cls:dinov2reg_tuned_cls", "dinov2_tuned_mt:dinov2reg_tuned_mt"]
