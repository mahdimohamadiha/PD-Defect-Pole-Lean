# How to resume this project

Last updated: 2026-11-26, after ConvNeXt (b9–b11, xai2_convnext) completion and handover to Gemini.

## Where everything is
| what | where |
|---|---|
| code | `train_multitask.py`, `analyze_results.py`, `tune_summary.py` (repo root) |
| all run outputs (metrics, predictions, logs, analyses) | `results/marimo/<batch>/` |
| marimo notebook that drives the runs | `notebooks/pd_defect_experiments.py` |
| batch definitions (the `plan` cell of each batch) | `tools/plans/<batch>.py` |
| fine-tuned checkpoints | HF `northlattice/pd-defect-pole-lean-checkpoints` (`<batch>/<tag>_s<seed>.pt`) |
| literature search | `docs/literature_search_2026-11-23.md` (+ browser report) |
| project rules | `CLAUDE.md`, `GEMINI.md` |
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

## Status 2026-11-24 (end of session)
- All main runs done (b1, b2, b3, b4a, b3b, b4b). Checkpoints on HF. README + HF card are up to date.
- Phase 1 diagnostics done (no GPU): `results/analysis/phase1/phase1.txt`, `docs/phase1_error_analysis.md`.
  Labels follow an implicit ~3° threshold. Leaned vs Straight is near the label-consistency ceiling. The headroom
  is in `Rejected`, which uses darkness as a shortcut.
- Next, in this order. Each hypothesis is pre-registered in the README before running, uses equal-budget tuning
  and a single test evaluation:
  1. H1: stronger photometric augmentation (wider brightness, gamma, low-light simulation) against the darkness
     shortcut. This needs the user's OK to extend the augmentation protocol in CLAUDE.md.
  2. H3: data efficiency (10 / 25 / 50% of train) for cls vs mt.
  3. H2: GT-angle soft labels in the ambiguous 2–6° band.
  Then: LaTeX preprint → supervisor → arXiv.
- Restart: new molab sandbox → user sends URL + token → restore `results/marimo/*` and the scripts into
  `/marimo`, download the data, and keep the HF token in the notebook box (see the procedure above).

## Status 2026-11-24 (afternoon)
- H1 `b5_h1_photo_dinov2` done: null (photo aug does not raise Macro-F1 / Rejected F1). 6 checkpoints on HF.
- H3 `b6_h3_dataeff_dinov2` done: null (Δ mt−cls ≈ 0 at 10/25/50%; no growth with less data). No checkpoints (user's choice).
- TODO: HF card update with H1 + H3 (needs the HF token in the sandbox box); then design and pre-register H2.
- H2 `b7_h2_soft_dinov2` done (16:35 UTC): soft cls 85.84 (Δ +2.34 [−1.29, +6.26], n.s.); NLL 0.609 → 0.461,
  ECE 10.4 → 6.7%; 2–6° errors 26.1 → 17.1%. 6 checkpoints on HF. Open confound: label smoothing (control needed).
- H2b `b8_h2b_ctrl_dinov2` done (18:00 UTC): replication +1.49 [−1.28, +4.44] n.s.; soft ≈ shuffled control → no angle-specific claim; calibration gain = smoothing. 12 ckpts on HF.
- X1 (XAI energy-in-box) pre-registered; running in sandbox (`results/xai_x1`).
- X1 done: mt does not increase energy-in-box (DINOv2 −1.5 pts, reg +0.1); Grad-CAM diffuse; CLS attention on pole 41–55% (chance 9%); DINOv2-reg attention pointing 86–88% vs DINOv2 26–49%, yet reg is less accurate.

## Status 2026-11-25
- All GPU work is done (b1–b8, X1, X2). Sandbox shut down by the user; everything is in `results/` and the
  checkpoints are on HF (not linked in the paper yet).
- Paper draft: `paper/` (arXiv `article` layout, Times New Roman figures from `paper/make_figures.py --font`).
  Open items: code repository URL (placeholder, on purpose), Wang & Wei (EAAI 2026) summary, unverified bib
  entries.
- Next: the user runs `docs/citation_check_prompt.md` in Claude chat (browser) with the paper zip, then sends
  back main.tex, references.bib and the report. Review every change against the report before merging, compile
  and commit. Then the supervisor reviews the draft.
## Status 2026-11-26 (handover to Gemini)
- **ConvNeXt-Base (22k) complete (`b9_tune_convnext`, `b10_final_convnext`, `b11_convnext_kppos`, `xai2_convnext`):**
  - `b10 cls`: Acc `82.82 ± 0.76`, Macro-F1 `79.54 ± 0.77` (MV `80.70`).
  - `b10 mt` (`H4`): Acc `82.53 ± 0.67`, Macro-F1 `78.97 ± 0.66` (Δ `-0.57`, n.s.); keypoint head failed to learn pole axis (angle corr `0.13`, `MAE 3.28°`) due to lack of absolute position codes in ConvNeXt last-stage features.
  - `b11 mt + kp_pos` (`H4b`, `--kp_pos sincos`): Acc `81.20 ± 1.67`, Macro-F1 `77.53 ± 1.53` (MV `80.47`, Δ vs `cls` `-2.01`, n.s.); angle corr recovered to `0.87 ± 0.08` (`MAE 1.79°`, median `1.05°`), confirming keypoint supervision is still null on CNN.
  - `xai2_convnext`: `mt - cls` pole reliance difference `+0.205` (`Wilcoxon p = 0.7529`, n.s.).
- **Hugging Face (`northlattice/pd-defect-pole-lean-checkpoints`):** Verified all 9 ConvNeXt checkpoints (`b10_final_convnext/` ×6, `b11_convnext_kppos/` ×3) and earlier final batches (`b2`, `b4a`, `b4b`, `b5`, `b7`, `b8`) are uploaded (`10.8 GB`, 72 commits) and HF model card updated.
- **Citation check (`docs/citation_check_report.md`):** 3 passes complete; open items remain for Step 2.
- **Next (tracked in `GEMINI.md`):**
  - Step 2: Resolve open items from `docs/citation_check_report.md` in `paper/references.bib` and `paper/main.tex`.
  - Step 3: Integrate ConvNeXt (`H4`, `H4b`, `xai2_convnext`) into `paper/main.tex` (and check `paper/make_figures.py` / figures).
  - Step 4: Final cleanup (`\todo`, Data & Code Availability, consistency check) for supervisor review and arXiv.
