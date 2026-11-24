# How to resume this project

Last updated: 2026-11-23, after batch `b4a_final_dinov2`.

## Where everything is
| what | where |
|---|---|
| code | `train_multitask.py`, `analyze_results.py`, `tune_summary.py` (repo root) |
| all run outputs (metrics, predictions, logs, analyses) | `results/marimo/<batch>/` |
| marimo notebook that drives the runs | `notebooks/pd_defect_experiments.py` |
| batch definitions (the `plan` cell of each batch) | `tools/plans/<batch>.py` |
| fine-tuned checkpoints | HF `northlattice/pd-defect-pole-lean-checkpoints` (`<batch>/<tag>_s<seed>.pt`) |
| literature search | `docs/literature_search_2026-11-23.md` (+ browser report) |
| project rules | `CLAUDE.md` |

## Done
- `b1_dinov2`: DINOv2 × {cls, mt, mt_geo}, default hyper-parameters
- `b2_dinov2reg_geofix`: DINOv2-reg × {cls, mt} + DINOv2 `mt_geo_d` (checkpoints on HF)
- `b3_tune_dinov2`: equal-budget search, validation only → `best_configs.json` (trial 7 for both modes)
- `b4a_final_dinov2`: tuned DINOv2 cls / mt, seeds 0–2, test once (checkpoints on HF). Result: mt 84.72 ± 2.29 vs
  cls 83.50 ± 0.67 Macro-F1, Δ +1.23, CI [−2.0, +4.7], not significant. See its `analysis.txt` and the README headline.

## Next
1. ✅ `b3b_tune_dinov2reg` done (cls → trial 0 default, mt → trial 4, λ=1). Plan: `tools/plans/b3b_tune_dinov2reg.py`.
2. ✅ `b4b_final_dinov2reg` done (06:18 UTC 2026-11-24); 6 checkpoints on HF; README + HF card updated.
   It reads `results/b3b_tune_dinov2reg/best_configs.json` automatically.
3. Paper-side analyses (no GPU needed):
   - the rule baseline "predicted keypoint angle > 10°" (as in arXiv:2610.17620) vs. the classifiers;
   - error analysis using the GT axis angle;
   - read Wang & Wei (EAAI 2026) before writing related work.
4. Then: paper (LaTeX), arXiv preprint, choose a licence for code and weights.

## Restarting compute in a new molab sandbox
1. Open molab, create or open a notebook, and import `notebooks/pd_defect_experiments.py`.
2. Copy `train_multitask.py`, `analyze_results.py` and `tune_summary.py` into the sandbox working dir (`/marimo`).
3. Copy previous results into `/marimo/results/` if a batch needs them. `b4b` and the analyses read earlier
   batches: `b1_dinov2`, `b2_dinov2reg_geofix`, `b4a_final_dinov2`, `b3b_tune_dinov2reg`.
4. Run the notebook. The `download` cell fetches the dataset (~2.5 GB, ~3 min).
5. Replace the `plan` cell body with the next file from `tools/plans/`.
6. Paste a fine-grained HF token (write access to the checkpoints repo) into the token box. The batch then starts.
7. In Claude Code, attach the new sandbox URL and marimo token (`uvx marimo@latest pair prompt --url <URL> --with-token --claude`).
   Then run `tools/sync.sh` in a loop to mirror results to GitHub (see the env vars in `tools/mx.sh`).

The notebook's `launch` cell is idempotent: finished runs (existing `res_*.json`) are skipped. A batch
interrupted by a sandbox shutdown can therefore be resumed by re-running the notebook, after copying
the batch's existing results back into `/marimo/results/<batch>/`.
