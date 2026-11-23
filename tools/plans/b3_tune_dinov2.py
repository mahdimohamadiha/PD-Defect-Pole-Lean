# ---- batch 3: equal-budget hyper-parameter search on the VALIDATION split (test is never evaluated)
# Paired random search: cls and mt get the SAME 8 draws of the shared hyper-parameters (trial 0 = the
# default used in batches 1-2); mt additionally gets lam (balanced: each of 1, 2, 5, 10 twice). 2 search seeds (100, 101), distinct from the
# final seeds 0-2. Best config per mode = highest mean val Macro-F1 -> re-trained in batch 4 and tested once.
BATCH = "b3_tune_dinov2"
GATED = set()
CKPT_REPO = "northlattice/pd-defect-pole-lean-checkpoints"
SAVE_CKPT = False                      # search runs: no checkpoints
N_TRIALS = 8
_rng = np.random.default_rng(2026)
_space = {"lr_bb": [1e-5, 2e-5, 3e-5, 5e-5], "lr_head": [3e-4, 1e-3, 3e-3], "epochs": [15, 25]}
TRIALS = [{"lr_bb": 2e-5, "lr_head": 1e-3, "epochs": 15}]
while len(TRIALS) < N_TRIALS:
    _c = {k: _v[int(_rng.integers(len(_v)))] for k, _v in _space.items()}
    if _c not in TRIALS:
        TRIALS.append(_c)
LAMS = [5.0] + [float(x) for x in _rng.permutation([1.0, 1.0, 2.0, 2.0, 5.0, 10.0, 10.0])]   # each lam twice
EXPERIMENTS = {}
for _i, _c in enumerate(TRIALS):
    _hp = [f"--{k}={v}" for k, v in _c.items()]
    EXPERIMENTS[f"cls_t{_i:02d}"] = ["--mode", "cls", *_hp, "--no_test"]
    EXPERIMENTS[f"mt_t{_i:02d}"] = ["--mode", "mt", *_hp, f"--lam={LAMS[_i]}", "--no_test"]
SEEDS = [100, 101]
COMMON_ARGS = ["--workers", "12"]
RUN_DIR = f"{RESULTS}/{BATCH}"
CKPT_DIR = f"{WORK}/checkpoints/{BATCH}"
RUNS = [(t, s) for s in SEEDS for t in EXPERIMENTS]   # seed-major: an interrupted search still has 1 seed/trial
ANALYSIS_CMD = ["tune_summary.py", "--results", RUN_DIR, "--out", f"{RUN_DIR}/best_configs.json"]
