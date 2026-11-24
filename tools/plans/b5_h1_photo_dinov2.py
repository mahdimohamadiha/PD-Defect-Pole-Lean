# ---- batch 5 (H1, pre-registered in README): extra photometric augmentation (random gamma + low-light
# simulation, --aug photo) against the darkness shortcut for 'Rejected'. DINOv2-base, the hyper-parameters
# selected in batch 3 (NOT re-tuned for the new augmentation, which is conservative for H1), seeds 0-2, test once.
BATCH = "b5_h1_photo_dinov2"
GATED = set()
CKPT_REPO = "northlattice/pd-defect-pole-lean-checkpoints"
SAVE_CKPT = True
BEST = json.load(open(f"{RESULTS}/b3_tune_dinov2/best_configs.json"))["best"]

def _hp(mode):
    _b = BEST[mode]
    return ([f"--lr_bb={_b['lr_bb']}", f"--lr_head={_b['lr_head']}", f"--epochs={_b['epochs']}"]
            + ([f"--lam={_b['lam']}"] if mode != "cls" else []))

EXPERIMENTS = {
    "dinov2_photo_cls": ["--mode", "cls", "--aug", "photo", *_hp("cls")],
    "dinov2_photo_mt": ["--mode", "mt", "--aug", "photo", *_hp("mt")],
}
SEEDS = [0, 1, 2]
COMMON_ARGS = ["--workers", "12"]
RUN_DIR = f"{RESULTS}/{BATCH}"
CKPT_DIR = f"{WORK}/checkpoints/{BATCH}"
RUNS = [(t, s) for s in SEEDS for t in EXPERIMENTS]
ANALYSIS_CMD = ["analyze_results.py", "--results", f"{RESULTS}/b4a_final_dinov2", RUN_DIR, "--n_boot", "10000",
                "--out", f"{RUN_DIR}/analysis.json",
                "--pairs", "dinov2_tuned_cls:dinov2_photo_cls", "dinov2_tuned_mt:dinov2_photo_mt",
                "dinov2_photo_cls:dinov2_photo_mt"]
