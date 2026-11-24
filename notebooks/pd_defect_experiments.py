import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", auto_download=["html"])

with app.setup:
    import os, sys, json, glob, re, subprocess, time
    import numpy as np
    import marimo as mo
    from huggingface_hub import HfApi

    WORK = "/marimo"
    DATA_DIR = f"{WORK}/data/pd-defect"
    ROOT = f"{DATA_DIR}/pole_lean_assessment"
    RESULTS = f"{WORK}/results"
    SCRIPT = f"{WORK}/train_multitask.py"
    os.makedirs(RESULTS, exist_ok=True)


@app.cell(hide_code=True)
def intro():
    mo.md(r"""
    # PD-Defect · Pole Lean: multi-task experiments
    Official **classification** split only (train 1706 / val 341 / test 227), comparable with the dataset-card
    DINOv3 baseline (test Acc 84.14, Macro-F1 81.74). Keypoints used only for cls-train images (stem + size match),
    masked for *Rejected*. Modes: `cls` · `mt` (aux keypoint loss) · `mt_geo` (+ geometry-conditioned head).

    Runs execute in a background process (`run_all`), so the notebook stays responsive; each run writes
    `results/res_<mode>_s<seed>.json` (metrics + test predictions) and `results/log_<mode>_s<seed>.txt`.
    """)
    return


@app.cell
def download():
    from huggingface_hub import snapshot_download

    def _count():
        return (len(glob.glob(f"{ROOT}/classification/*/*/*"))
                + len(glob.glob(f"{ROOT}/keypoint_detection/images/*/*")))

    if _count() != 4072:
        snapshot_download("EPDCL/pd-defect", repo_type="dataset", local_dir=DATA_DIR, max_workers=16,
                          allow_patterns=["pole_lean_assessment/classification/**",
                                          "pole_lean_assessment/keypoint_detection/images/**",
                                          "pole_lean_assessment/keypoint_detection/labels/**"])
    n_images = _count()
    mo.md(f"**Images on disk:** {n_images} (expected 4072 = 2274 cls + 1798 kp)")
    return (n_images,)


@app.cell
def plan():
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

    return (
        ANALYSIS_CMD,
        BATCH,
        CKPT_DIR,
        CKPT_REPO,
        COMMON_ARGS,
        EXPERIMENTS,
        GATED,
        RUNS,
        RUN_DIR,
        SAVE_CKPT,
    )


@app.cell
def hf_token():
    hf_token = mo.ui.text(kind="password", full_width=True,
                          label="Hugging Face token (write access to the checkpoints repo). Never saved in the notebook file.")
    hf_token
    return (hf_token,)


@app.cell
def refresh():
    refresh = mo.ui.refresh(default_interval="30s", label="auto-refresh")
    refresh
    return (refresh,)


@app.cell
def launch(
    BATCH,
    CKPT_DIR,
    CKPT_REPO,
    COMMON_ARGS,
    EXPERIMENTS,
    GATED,
    RUNS,
    RUN_DIR,
    SAVE_CKPT,
    hf_token,
    n_images,
    refresh,
):
    refresh.value   # re-check every refresh tick: picks up DINOv3 as soon as access is granted

    def _alive(pid):
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def _failed(t, s):
        _log = f"{RUN_DIR}/log_{t}_s{s}.txt"
        return os.path.exists(_log) and "Traceback" in open(_log).read()

    _token = hf_token.value.strip()
    _pid_file = f"{RUN_DIR}/runner.pid"
    _running = os.path.exists(_pid_file) and _alive(int(open(_pid_file).read()))
    _todo = [(t, s) for t, s in RUNS if not os.path.exists(f"{RUN_DIR}/res_{t}_s{s}.json")]
    _blocked, _msg = [], ""
    if not _running and _todo and not _token:
        _msg = "Paste the HF token above to start this batch."
    elif not _running and _todo and n_images == 4072:
        _api = HfApi(token=_token)
        try:
            _api.repo_info(CKPT_REPO)
            _ok_repo = True
        except Exception as _e:
            _ok_repo, _msg = False, f"checkpoint repo not accessible ({type(_e).__name__}): {str(_e)[:200]}"
        _gated_ok = {}
        for _bb in GATED:
            try:
                _api.auth_check(_bb)   # model_info succeeds even without gated access
                _gated_ok[_bb] = True
            except Exception:
                _gated_ok[_bb] = False
        _ready = [(t, s) for t, s in _todo if not _failed(t, s)
                  and all(_gated_ok.get(b, True) for b in EXPERIMENTS[t] if b in _gated_ok)]
        _blocked = sorted({t for t, s in _todo if (t, s) not in _ready})
        if _ok_repo and _ready:
            os.makedirs(RUN_DIR, exist_ok=True)
            os.makedirs(CKPT_DIR, exist_ok=True)
            _cmd = " ; ".join(
                f"{sys.executable} -u {SCRIPT} --root {ROOT} --tag {t} --seeds {s} "
                f"{' '.join(EXPERIMENTS[t] + COMMON_ARGS + (['--ckpt_dir', CKPT_DIR] if SAVE_CKPT else []))} "
                f"--out {RUN_DIR}/res_{t}_s{s}.json > {RUN_DIR}/log_{t}_s{s}.txt 2>&1"
                for t, s in _ready)
            _p = subprocess.Popen(["bash", "-c", _cmd], cwd=WORK, start_new_session=True,
                                  env=dict(os.environ, HF_TOKEN=_token))
            open(_pid_file, "w").write(str(_p.pid))
            _running = True
        if _blocked and not _msg:
            _msg = f"waiting (gated access not granted yet, or failed run): {', '.join(_blocked)}"
    run_all = {"running": _running, "todo": _todo}
    mo.md(f"**Runner ({BATCH}):** {'running' if _running else 'idle'} · remaining runs: {len(_todo)} / {len(RUNS)}"
          + (f"<br>⚠️ {_msg}" if _msg else ""))
    return


@app.cell
def status(BATCH, RUNS, RUN_DIR, refresh):
    refresh.value
    _rows = []
    for _t, _s in RUNS:
        _res, _log = f"{RUN_DIR}/res_{_t}_s{_s}.json", f"{RUN_DIR}/log_{_t}_s{_s}.txt"
        _row = {"experiment": _t, "seed": _s, "state": "pending", "last epoch": "", "val macroF1": "", "test acc": "", "test macroF1": ""}
        if os.path.exists(_res):
            _r = json.load(open(_res))[0]
            _row.update(state="done", **{"val macroF1": round(100 * _r["val_macroF1"], 2)})
            if "test_acc" in _r:
                _row.update(**{"test acc": round(100 * _r["test_acc"], 2), "test macroF1": round(100 * _r["test_macroF1"], 2)})
        if os.path.exists(_log):
            _txt = open(_log).read()
            _ep = re.findall(r"ep(\d+) val acc ([\d.]+) macroF1 ([\d.]+)", _txt)
            if _ep:
                _row["last epoch"] = f"ep{_ep[-1][0]}: val F1 {100 * float(_ep[-1][2]):.1f}"
            if _row["state"] == "pending":
                _row["state"] = "error" if "Traceback" in _txt else "running"
        _rows.append(_row)
    mo.ui.table(_rows, selection=None, label=f"Runs · {BATCH}")
    return


@app.cell
def upload(BATCH, CKPT_DIR, CKPT_REPO, RUNS, RUN_DIR, hf_token, refresh):
    refresh.value
    _tok = hf_token.value.strip()
    _uploaded, _err = [], ""
    if _tok and os.path.isdir(CKPT_DIR):
        _api = HfApi(token=_tok)
        for _f in sorted(glob.glob(f"{CKPT_DIR}/*.pt")):
            _name = os.path.basename(_f)[:-3]
            _mark = _f + ".uploaded"
            # the result json is written after the checkpoint, so its presence means the .pt is complete
            if not os.path.exists(_mark) and os.path.exists(f"{RUN_DIR}/res_{_name}.json"):
                try:
                    _api.upload_file(path_or_fileobj=_f, path_in_repo=f"{BATCH}/{os.path.basename(_f)}",
                                     repo_id=CKPT_REPO, commit_message=f"{BATCH}: {_name}")
                    open(_mark, "w").write(time.ctime())
                except Exception as _e:
                    _err = f"{type(_e).__name__}: {str(_e)[:200]}"
                    break
            if os.path.exists(_mark):
                _uploaded.append(_name)
    mo.md(f"**Checkpoints on HF** (`{CKPT_REPO}/{BATCH}`): {len(_uploaded)} / {len(RUNS)}"
          + (f"<br>⚠️ upload error: {_err}" if _err else ""))
    return


@app.cell
def analysis(ANALYSIS_CMD, BATCH, RUNS, RUN_DIR, refresh):
    refresh.value
    _out = f"{RUN_DIR}/analysis.txt"
    _n_done = len(glob.glob(f"{RUN_DIR}/res_*.json"))
    if _n_done == len(RUNS) and not os.path.exists(_out):
        _r = subprocess.run([sys.executable, f"{WORK}/{ANALYSIS_CMD[0]}", *ANALYSIS_CMD[1:]], capture_output=True, text=True)
        open(_out, "w").write(_r.stdout + _r.stderr)
    analysis_text = open(_out).read() if os.path.exists(_out) else None
    mo.md(f"## Analysis · {BATCH}\n" + (f"```\n{analysis_text}\n```" if analysis_text
          else f"_Waiting for all runs: {_n_done} / {len(RUNS)} done._"))
    return


if __name__ == "__main__":
    app.run()
