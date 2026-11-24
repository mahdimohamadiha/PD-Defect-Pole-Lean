# ---- batch 6 (H3, pre-registered in README): data efficiency of cls vs mt on DINOv2-base.
# Train on a stratified 10 / 25 / 50% subset of the classification-train split (subset drawn with rng 1000+seed,
# so cls and mt of the same seed see the same images). Hyper-parameters = batch-3 selection; the number of
# optimisation steps is held equal to the full-data runs (epochs = 15 / frac). Seeds 0-2, test evaluated once.
# Full-data reference = b4a_final_dinov2.
BATCH = "b6_h3_dataeff_dinov2"
GATED = set()
CKPT_REPO = "northlattice/pd-defect-pole-lean-checkpoints"
SAVE_CKPT = False                      # low-data models are not worth hosting
BEST = json.load(open(f"{RESULTS}/b3_tune_dinov2/best_configs.json"))["best"]
FRACS = {"f10": 0.10, "f25": 0.25, "f50": 0.50}

def _hp(mode, frac):
    _b = BEST[mode]
    return ([f"--lr_bb={_b['lr_bb']}", f"--lr_head={_b['lr_head']}", f"--epochs={round(_b['epochs'] / frac)}",
             f"--train_frac={frac}"] + ([f"--lam={_b['lam']}"] if mode != "cls" else []))

EXPERIMENTS = {}
for _k, _f in FRACS.items():
    for _m in ["cls", "mt"]:
        EXPERIMENTS[f"dinov2_{_k}_{_m}"] = ["--mode", _m, *_hp(_m, _f)]
SEEDS = [0, 1, 2]
COMMON_ARGS = ["--workers", "12"]
RUN_DIR = f"{RESULTS}/{BATCH}"
CKPT_DIR = f"{WORK}/checkpoints/{BATCH}"
RUNS = [(t, s) for s in SEEDS for t in EXPERIMENTS]   # seed-major: every finished seed is a paired cls/mt set
ANALYSIS_CMD = ["analyze_results.py", "--results", f"{RESULTS}/b4a_final_dinov2", RUN_DIR, "--n_boot", "10000",
                "--out", f"{RUN_DIR}/analysis.json",
                "--pairs", "dinov2_f10_cls:dinov2_f10_mt", "dinov2_f25_cls:dinov2_f25_mt", "dinov2_f50_cls:dinov2_f50_mt",
                "dinov2_tuned_cls:dinov2_tuned_mt"]
