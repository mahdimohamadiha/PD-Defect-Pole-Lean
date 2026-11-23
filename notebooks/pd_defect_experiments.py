import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", auto_download=["html"])

with app.setup:
    import os, sys, json, glob, re, subprocess, time
    import numpy as np
    import marimo as mo

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
    MODES = ["cls", "mt", "mt_geo"]
    SEEDS = [0, 1, 2]
    EXTRA_ARGS = ["--workers", "12"]   # all other hyper-parameters = script defaults (15 ep, 448 px, bs 16)
    RUNS = [(m, s) for s in SEEDS for m in MODES]   # seed-major: partial results stay balanced across modes
    return EXTRA_ARGS, RUNS


@app.cell
def launch(EXTRA_ARGS, RUNS, n_images):
    def _alive(pid):
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    _pid_file = f"{RESULTS}/runner.pid"
    _running = os.path.exists(_pid_file) and _alive(int(open(_pid_file).read()))
    _todo = [(m, s) for m, s in RUNS if not os.path.exists(f"{RESULTS}/res_{m}_s{s}.json")]
    if n_images == 4072 and _todo and not _running:
        _cmd = " ; ".join(
            f"{sys.executable} -u {SCRIPT} --root {ROOT} --mode {m} --seeds {s} "
            f"--out {RESULTS}/res_{m}_s{s}.json {' '.join(EXTRA_ARGS)} > {RESULTS}/log_{m}_s{s}.txt 2>&1"
            for m, s in _todo)
        _p = subprocess.Popen(["bash", "-c", _cmd], cwd=WORK, start_new_session=True)
        open(_pid_file, "w").write(str(_p.pid))
        _running = True
    run_all = {"running": _running, "todo": _todo}
    mo.md(f"**Runner:** {'running' if _running else 'idle'} · remaining runs: {len(_todo)} / {len(RUNS)}")
    return


@app.cell
def refresh():
    refresh = mo.ui.refresh(default_interval="30s", label="auto-refresh")
    refresh
    return (refresh,)


@app.cell
def status(RUNS, refresh):
    refresh.value
    _rows = []
    for _m, _s in RUNS:
        _res, _log = f"{RESULTS}/res_{_m}_s{_s}.json", f"{RESULTS}/log_{_m}_s{_s}.txt"
        _row = {"mode": _m, "seed": _s, "state": "pending", "last epoch": "", "test acc": "", "test macroF1": ""}
        if os.path.exists(_res):
            _r = json.load(open(_res))[0]
            _row.update(state="done", **{"test acc": round(100 * _r["test_acc"], 2),
                                         "test macroF1": round(100 * _r["test_macroF1"], 2)})
        if os.path.exists(_log):
            _txt = open(_log).read()
            _ep = re.findall(r"ep(\d+) val acc ([\d.]+) macroF1 ([\d.]+)", _txt)
            if _ep:
                _row["last epoch"] = f"ep{_ep[-1][0]}: val F1 {100 * float(_ep[-1][2]):.1f}"
            if _row["state"] == "pending":
                _row["state"] = "error" if "Traceback" in _txt else "running"
        _rows.append(_row)
    mo.ui.table(_rows, selection=None, label="Runs")
    return


@app.cell
def analysis(RUNS, refresh):
    refresh.value
    _out = f"{RESULTS}/analysis.txt"
    _n_done = len(glob.glob(f"{RESULTS}/res_*.json"))
    if _n_done == len(RUNS) and not os.path.exists(_out):
        _r = subprocess.run([sys.executable, f"{WORK}/analyze_results.py", "--results", RESULTS,
                             "--n_boot", "10000", "--out", f"{RESULTS}/analysis.json"],
                            capture_output=True, text=True)
        open(_out, "w").write(_r.stdout + _r.stderr)
    analysis_text = open(_out).read() if os.path.exists(_out) else None
    mo.md("## Statistical analysis\n" + (f"```\n{analysis_text}\n```" if analysis_text
          else f"_Waiting for all runs: {_n_done} / {len(RUNS)} done._"))
    return


if __name__ == "__main__":
    app.run()
