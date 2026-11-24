# ---- batch 7 (H2, pre-registered in README): ambiguity-aware soft Leaned/Straight targets from the GT axis
# angle (train images with keypoints only). alpha = 0.5, t0 = 3.03 deg (train-fitted, phase 1), tau = 1 deg,
# all fixed a priori (not tuned). DINOv2-base, batch-3 hyper-parameters (not re-tuned), seeds 0-2, test once.
BATCH = "b7_h2_soft_dinov2"
GATED = set()
CKPT_REPO = "northlattice/pd-defect-pole-lean-checkpoints"
SAVE_CKPT = True
BEST = json.load(open(f"{RESULTS}/b3_tune_dinov2/best_configs.json"))["best"]

def _hp(mode):
    _b = BEST[mode]
    return ([f"--lr_bb={_b['lr_bb']}", f"--lr_head={_b['lr_head']}", f"--epochs={_b['epochs']}"]
            + ([f"--lam={_b['lam']}"] if mode != "cls" else []))

SOFT = ["--soft_alpha", "0.5", "--soft_t0", "3.03", "--soft_tau", "1.0"]
EXPERIMENTS = {
    "dinov2_soft_cls": ["--mode", "cls", *SOFT, *_hp("cls")],
    "dinov2_soft_mt": ["--mode", "mt", *SOFT, *_hp("mt")],
}
SEEDS = [0, 1, 2]
COMMON_ARGS = ["--workers", "12"]
RUN_DIR = f"{RESULTS}/{BATCH}"
CKPT_DIR = f"{WORK}/checkpoints/{BATCH}"
RUNS = [(t, s) for s in SEEDS for t in EXPERIMENTS]
ANALYSIS_CMD = ["analyze_results.py", "--results", f"{RESULTS}/b4a_final_dinov2", RUN_DIR, "--n_boot", "10000",
                "--out", f"{RUN_DIR}/analysis.json",
                "--pairs", "dinov2_tuned_cls:dinov2_soft_cls", "dinov2_tuned_mt:dinov2_soft_mt",
                "dinov2_soft_cls:dinov2_soft_mt"]
