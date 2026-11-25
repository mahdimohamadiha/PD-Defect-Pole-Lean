# Keypoint-supervised pole-lean classification on PD-Defect

Multi-task learning for **utility-pole lean assessment** (Leaned / Straight / Rejected) on the
[PD-Defect](https://huggingface.co/datasets/EPDCL/pd-defect) dataset (`pole_lean_assessment` subset,
CC BY 4.0). A ViT backbone is fine-tuned for 3-way classification, with an auxiliary head that
regresses the pole axis (bottom and top keypoints).

> **Status:** all main experiments are complete (default runs, validation-only equal-budget tuning and the
> final tuned test evaluation, for both DINOv2-base and DINOv2-with-registers), as are the diagnostic analyses and
> the pre-registered follow-ups: H1 (photometric augmentation) and H3 (data efficiency) are null; H2 (GT-angle soft
> labels) improves calibration, but the H2b controls show that this comes from target smoothing, not from the
> angle. The explainability test (X1) finds that both models attend to the pole, but keypoint supervision does not make attention more pole-focused. The perturbation test (X2) shows that all models depend causally on the pole and partly on its context. A paper
> draft is in [`paper/`](paper).

## How this project was carried out

**AI assistance.** The study was carried out with Claude, an AI assistant developed by Anthropic, working as a
research assistant under the authors' direction. Claude searched the literature, wrote and ran the experiment and
analysis code, proposed experimental designs, performed the statistical analyses, produced the figures and drafted
the manuscript. The authors chose the topic and the dataset, reviewed every design and approved it before it was
run, made the key decisions (for example the target venue, the follow-up hypotheses to pursue, the deeper
explainability analysis and the structure of the paper), and checked the results and the text. The paper states
this in its "Use of AI assistance" section. No other AI tool was used.

**Timeline and compute.** All experiments ran between 23 and 25 November 2026 on single-GPU cloud notebooks.
One training run takes about 4–7 minutes, and runs were queued back to back, so the whole study needed a few
tens of GPU hours. For every run, the training log, the selected epoch and the prediction for every validation
and test image are in [`results/`](results), so all reported numbers can be recomputed.

**The commit history is the pre-registration record.** Each follow-up hypothesis, its prediction, its outcomes and,
where relevant, its decision rule were committed *before* the corresponding runs, and the results were committed
afterwards. The single amendment made after a code check (artifact tokens in X1) is documented where it occurred.
The history has not been rewritten, so the order of design and results can be verified from the commit
timestamps. Commits made by the assistant carry its name as author or co-author.

**Labels used here vs. the paper.** The hypothesis labels in this README are internal; the paper uses
descriptive section titles instead:

| README label | batch | paper section |
|---|---|---|
| main comparison | `b1`–`b4b` | Main comparison: does keypoint supervision help? |
| phase 1 / 1b | (analysis only) | What do the labels encode? · Where do the errors come from? |
| H1 | `b5_h1_photo_dinov2` | Photometric augmentation |
| H3 | `b6_h3_dataeff_dinov2` | Training with less data |
| H2, H2b | `b7_h2_soft_dinov2`, `b8_h2b_ctrl_dinov2` | Angle-aware soft targets |
| X1, X2 | `xai_x1`, `x2_xai` | Where does the model look? |

## Headline: tuned models (final)

Hyper-parameters selected on validation only (batches 3 / 3b, equal budget per mode), re-trained with seeds
0–2 and evaluated **once** on the official test split (227 images). Mean ± std over seeds:

| backbone | model | selected config (lr bb / lr head / epochs / λ) | Acc | Macro-F1 | F1 Leaned | F1 Straight | F1 Rejected | maj.-vote Macro-F1 |
|---|---|---|---|---|---|---|---|---|
| DINOv2-base | `cls` | 1e-5 / 3e-4 / 15 / – | 86.34 ± 0.76 | 83.50 ± 0.67 | 78.8 | **91.3** | 80.4 | 83.30 |
| DINOv2-base | **`mt`** | 1e-5 / 3e-4 / 15 / 2 | **86.93 ± 2.17** | **84.72 ± 2.29** | **81.5** | 91.2 | **81.4** | **87.37** |
| DINOv2-reg-base | `cls` | 2e-5 / 1e-3 / 15 / – (= default) | 83.99 ± 2.22 | 81.31 ± 2.64 | 77.9 | 89.1 | 76.9 | 81.47 |
| DINOv2-reg-base | `mt` | 2e-5 / 3e-3 / 25 / 1 | 83.26 ± 0.88 | 79.97 ± 0.97 | 75.6 | 89.2 | 75.1 | 83.60 |
| DINOv3 ViT-B/16 | dataset-card baseline\* | – | 84.14 | 81.74 | – | – | – | – |

| comparison (B vs A) | Δ Macro-F1 | 95% CI | McNemar p (Holm) |
|---|---|---|---|
| DINOv2: tuned `mt` vs tuned `cls` | +1.23 | [−2.01, +4.70] | 0.24 (0.29) |
| DINOv2-reg: tuned `mt` vs tuned `cls` | −1.34 | [−3.93, +1.15] | 0.24 (0.71) |
| DINOv2: tuned `cls` vs default `cls` | **+4.72** | [+1.50, +8.37] | 0.041 (0.12) |
| DINOv2-reg: re-run of the default `cls` config (b4b vs b2) | −1.06 | [−3.63, +1.61] | 0.33 (0.71) |
| `cls`: DINOv2-reg vs DINOv2 (both tuned) | −2.19 | [−5.48, +1.09] | 0.25 (0.71) |
| `mt`: DINOv2-reg vs DINOv2 (both tuned) | **−4.76** | [−7.98, −1.94] | 0.12 (0.47) |

Takeaways:
- **After equal-budget tuning, the auxiliary keypoint loss gives no significant change in classification**
  on either backbone (+1.2 on DINOv2, −1.3 on DINOv2-reg; both CIs contain 0). The large default-setting
  effects (+4.6 / −3.7, below) were mostly caused by untuned hyper-parameters.
- **Run-to-run noise is of the same size as these effects.** For DINOv2-reg, the search selected the default
  configuration for `cls`, so b4b re-ran exactly the b2 setting (same seeds). Non-deterministic GPU kernels
  alone changed the test Macro-F1 by −1.06 on average (per seed: +1.9, −2.6, −2.5).
- After tuning, **DINOv2-base is the better backbone** (significant for `mt` under the bootstrap), reversing
  the default-setting ranking of the `cls` baselines.
- The joint model still gives pole-axis keypoints at no extra inference cost, with equal classification
  quality. The best single configuration is tuned DINOv2 `mt` (Macro-F1 84.72; 87.37 with a 3-seed majority vote).
- All tuned models are within the 95% CI of the dataset-card baseline, so we make no claim of beating it.
- Output: [`b4a_final_dinov2/analysis.txt`](results/marimo/b4a_final_dinov2/analysis.txt),
  [`b4b_final_dinov2reg/analysis.txt`](results/marimo/b4b_final_dinov2reg/analysis.txt).

## Results with default hyper-parameters (batches 1–2)

Full fine-tuning, official test split (227 images), mean ± std over 3 seeds, default hyper-parameters (not tuned):

| backbone | model | Acc | Macro-F1 | F1 Leaned | F1 Straight | F1 Rejected |
|---|---|---|---|---|---|---|
| DINOv2-base | `cls` (classification only) | 82.23 ± 2.08 | 78.78 ± 2.39 | 75.3 | 88.4 | 72.7 |
| DINOv2-base | **`mt` (+ auxiliary keypoint loss)** | **85.90 ± 0.44** | **83.41 ± 0.94** | **80.5** | **90.4** | **79.3** |
| DINOv2-base | `mt_geo` (+ head on predicted geometry) | 83.55 ± 1.67 | 80.29 ± 1.95 | 76.6 | 89.3 | 75.0 |
| DINOv2-base | `mt_geo_d` (detached geometry, no length) | 83.41 ± 0.67 | 80.38 ± 1.79 | 75.1 | 88.8 | 77.2 |
| DINOv2-reg-base | `cls` | 84.88 ± 1.11 | 82.37 ± 0.97 | 80.4 | 89.2 | 77.5 |
| DINOv2-reg-base | `mt` | 81.50 ± 1.17 | 78.66 ± 0.99 | 73.7 | 87.0 | 75.3 |
| DINOv3 ViT-B/16 | dataset-card baseline\* | 84.14 | 81.74 | – | – | – |

\* Taken from the dataset card and not reproduced here, because DINOv3 is gated and access was not
granted to us. The card value lies inside our 95% CIs, so we make **no claim of beating it**. All
statistical claims compare our own models, trained with the same code.

**Statistics** (paired bootstrap over test images, 10,000 resamples; exact McNemar on seed-majority votes, Holm-corrected):

| comparison | Δ Macro-F1 | 95% CI | McNemar p (Holm) |
|---|---|---|---|
| DINOv2: `mt` vs `cls` | **+4.63** | [+1.98, +7.46] | 0.035 (0.105) |
| DINOv2-reg: `mt` vs `cls` | **−3.71** | [−6.61, −0.99] | 0.14 (0.43) |
| `cls`: DINOv2-reg vs DINOv2 | +3.58 | [+0.22, +7.07] | 0.096 (0.39) |
| DINOv2: `mt_geo` vs `mt` | −3.12 | [−5.70, −0.73] | 0.79 |
| DINOv2: `mt_geo_d` vs `mt_geo` | +0.09 | [−2.48, +2.59] | 1.0 |

Takeaways so far (default hyper-parameters):
- **The effect of the auxiliary keypoint loss depends on the backbone.** It helps DINOv2 (+4.6 Macro-F1)
  but hurts DINOv2-with-registers (−3.7). Both CIs exclude 0. With λ = 5 and an untuned learning rate, the
  benefit is therefore *not* robust. This is why batch 3 tunes both models with an equal budget before any
  claim is made.
- DINOv2-with-registers is a stronger classification-only baseline than DINOv2.
- Feeding predicted geometry into the classifier does not help: `mt_geo` and its fixed variant `mt_geo_d`
  perform the same, and both are below `mt`.
  These variants are therefore reported as **ablations at the default hyper-parameters** and are not
  included in the hyper-parameter search, which is spent on the main `cls` vs `mt` comparison.
- No McNemar test survives Holm correction. With 227 test images, the paired bootstrap is the more
  sensitive test, and both are reported.

Full outputs: [`b1_dinov2/analysis.txt`](results/marimo/b1_dinov2/analysis.txt),
[`b2_dinov2reg_geofix/analysis.txt`](results/marimo/b2_dinov2reg_geofix/analysis.txt).

## Hyper-parameter search (DINOv2-base, validation only)

Equal-budget paired random search: 8 shared configurations for `cls` and `mt` (trial 0 = default),
`mt` additionally gets λ (each of 1, 2, 5, 10 twice), and 2 search seeds (100, 101). The test split is
never loaded (`--no_test`). Validation Macro-F1, mean over the 2 seeds:

| trial | lr backbone | lr head | epochs | λ (mt) | `cls` | `mt` |
|---|---|---|---|---|---|---|
| 0 (default) | 2e-5 | 1e-3 | 15 | 5 | 79.47 (70.94 / 88.01) | 83.43 (83.45 / 83.41) |
| 1 | 5e-5 | 3e-4 | 15 | 10 | 79.33 | 81.78 |
| 2 | 3e-5 | 1e-3 | 15 | 10 | 81.78 | 82.87 |
| 3 | 1e-5 | 1e-3 | 25 | 2 | 84.36 | 84.60 |
| 4 | 2e-5 | 3e-3 | 25 | 1 | 81.02 | 82.44 |
| 5 | 3e-5 | 3e-3 | 25 | 5 | 82.66 | 82.61 |
| 6 | 1e-5 | 3e-3 | 25 | 1 | 84.64 | 83.56 |
| **7** | **1e-5** | **3e-4** | **15** | **2** | **85.17** | **85.10** |

- Both modes select **the same configuration** (trial 7), with λ = 2 for `mt`. Their validation scores are tied.
- `cls` is far more sensitive to hyper-parameters (79.3–85.2) and to the seed: at the default setting the two
  seeds differ by 17 points. `mt` stays within 81.8–85.1, and its seed spread is ≤ 2.6 points in every trial.
- Output: [`b3_tune_dinov2/analysis.txt`](results/marimo/b3_tune_dinov2/analysis.txt),
  [`best_configs.json`](results/marimo/b3_tune_dinov2/best_configs.json).

## Hyper-parameter search (DINOv2-with-registers, validation only)

Same search as above (same 8 configurations, same λ draws, same search seeds), backbone
`facebook/dinov2-with-registers-base`. Validation Macro-F1, mean over the 2 seeds:

| trial | lr backbone | lr head | epochs | λ (mt) | `cls` | `mt` |
|---|---|---|---|---|---|---|
| 0 (default) | 2e-5 | 1e-3 | 15 | 5 | **84.46** | 84.39 |
| 1 | 5e-5 | 3e-4 | 15 | 10 | 82.20 | 81.92 |
| 2 | 3e-5 | 1e-3 | 15 | 10 | 83.15 | 83.75 |
| 3 | 1e-5 | 1e-3 | 25 | 2 | 83.35 | 84.11 |
| 4 | 2e-5 | 3e-3 | 25 | 1 | 83.74 | **84.57** |
| 5 | 3e-5 | 3e-3 | 25 | 5 | 83.02 | 81.64 |
| 6 | 1e-5 | 3e-3 | 25 | 1 | 83.73 | 83.03 |
| 7 | 1e-5 | 3e-4 | 15 | 2 | 83.90 | 83.72 |

- `cls` selects the default configuration (trial 0); `mt` selects trial 4 (λ = 1, 25 epochs).
- Unlike DINOv2-base, DINOv2-reg is stable across configurations and seeds for both modes
  (all trials within 81.6–84.6, seed spread ≤ 1.5 points), and the two modes are tied on validation.
- Output: [`b3b_tune_dinov2reg/analysis.txt`](results/marimo/b3b_tune_dinov2reg/analysis.txt),
  [`best_configs.json`](results/marimo/b3b_tune_dinov2reg/best_configs.json).

## Diagnostic analysis (phase 1, no training)

Script: [`tools/analysis/phase1_angle_analysis.py`](tools/analysis/phase1_angle_analysis.py); output:
[`results/analysis/phase1/phase1.txt`](results/analysis/phase1/phase1.txt). GT keypoints of val/test images
are used here **for evaluation only**.

- **The binary lean label follows an implicit ≈ 3° threshold, not 10°.** On train, a single threshold on the
  GT axis angle (3.03°, fitted on train) reproduces the Leaned/Straight labels with 90.9% accuracy, and
  91.6% on test. The "angle > 10°" rule used in prior work reaches only 77.6% on test, even with GT keypoints.
  No Straight pole has a GT angle above 10°, while 16.8% of Leaned train poles are below 3°.
- **Remaining errors sit in the ambiguous 2–6° band.** 37 of the 143 Leaned/Straight test images with GT
  keypoints lie in 2–6°, where every model errs on 10–37% of images; outside this band errors are rare.
- **The classifiers are close to the label-consistency ceiling on Leaned vs Straight** (tuned DINOv2 `mt`:
  90.4% vs 91.6% for the GT-angle oracle rule). The headroom for higher Macro-F1 is mainly in `Rejected`
  (recall 77–87%) and in the 2–6° band.
- **The `mt` keypoint head is accurate**: angle MAE 0.97° (median 0.63°), correlation 0.95 with the GT angle
  (tuned DINOv2). A rule on the predicted angle with a validation-selected threshold (≈ 3.2°) gives hybrid
  Macro-F1 81.9 (classifier decides `Rejected`), and 68.5 with the 10° rule; both are below the classifier (84.7).
- Tuned DINOv2 `mt` vs `cls` (test, summed over seeds): recall Leaned 79.6 vs 74.7, Rejected 86.5 vs 77.5,
  Straight 90.6 vs 94.5.

## Pre-registered follow-up hypotheses

Registered here **before** any run. Each uses the fixed protocol (official split, selection on validation,
test evaluated once) and reports every outcome, including null results.

**H1: darkness shortcut (batch `b5_h1_photo_dinov2`).** Phase 1b found that dark images are about 10× more
frequent in train `Rejected`, and that valid dusk or night poles are wrongly predicted `Rejected`.
- *Intervention:* `--aug photo`, which adds random gamma (0.5–2.0, p = 0.5) and low-light simulation
  (intensity × 0.15–0.5 plus Gaussian noise, p = 0.25) on top of the base augmentation. The parameters were fixed
  a priori and are not tuned.
- *Setting:* DINOv2-base with the batch-3 hyper-parameters (not re-tuned, which is conservative), `cls` and `mt`,
  seeds 0–2, with checkpoints.
- *Primary outcome:* test Macro-F1 versus the tuned models of `b4a` (paired bootstrap, McNemar, Holm over 3 pairs).
- *Secondary outcomes:* `Rejected` F1, and the false-`Rejected` rate on low-light Leaned/Straight test images
  (lowest quartile of mean grey value; [`tools/analysis/h1_dark.py`](tools/analysis/h1_dark.py)).
- *Prediction:* `Rejected` F1 and Macro-F1 increase.
- **Result: not supported (null).** Test, mean ± std over seeds 0–2 ([`analysis.txt`](results/marimo/b5_h1_photo_dinov2/analysis.txt),
  [`h1_dark.txt`](results/analysis/h1/h1_dark.txt)):

  | model | Macro-F1 | Δ vs base aug [95% CI] | McNemar Holm p | `Rejected` F1 | false-`Rejected`, low-light L/S |
  |---|---|---|---|---|---|
  | `cls`, base aug (b4a) | 83.50 ± 0.67 | – | – | 80.4 | 7.6% |
  | `cls`, photo aug | 83.47 ± 1.72 | −0.03 [−3.03, +3.03] | 1.00 | 79.0 | 7.6% |
  | `mt`, base aug (b4a) | 84.72 ± 2.29 | – | – | 81.4 | 10.4% |
  | `mt`, photo aug | 83.42 ± 1.22 | −1.30 [−4.12, +1.53] | 0.63 | 78.9 | 8.3% |

  Photometric augmentation did not raise Macro-F1 or `Rejected` F1, and did not remove the low-light false-`Rejected`
  errors (7.6% → 7.6% for `cls`). This suggests that the darkness association is not a simple intensity shortcut.
  Low-light images also tend to be blurred, backlit or noisy in ways that resemble the true `Rejected` images. With
  photo aug, `mt` and `cls` are equal (−0.04 [−2.24, +2.16]). The main results keep `--aug base`.

**H3: data efficiency (batch `b6_h3_dataeff_dinov2`).** Auxiliary keypoint supervision should matter more when
labelled images are scarce.
- *Intervention:* train on a stratified 10 / 25 / 50% subset of the classification-train split (`--train_frac`).
  The subset is drawn with rng `1000 + seed`, so `cls` and `mt` of the same seed see the same images, and the
  keypoints are those of the subset only.
- *Setting:* DINOv2-base with the batch-3 hyper-parameters. The number of optimisation steps is held equal to the
  full-data runs (epochs = 15 / fraction). Seeds 0–2. The full-data reference is `b4a`.
- *Primary outcome:* Δ Macro-F1 (`mt` − `cls`) at each fraction (paired bootstrap and McNemar, Holm over the 4 pairs
  including full data).
- *Prediction:* Δ is positive and grows as the fraction shrinks.
- **Result: not supported (null).** Test Macro-F1, mean ± std over seeds 0–2
  ([`analysis.txt`](results/marimo/b6_h3_dataeff_dinov2/analysis.txt)):

  | train fraction | images | `cls` | `mt` | Δ (`mt` − `cls`) [95% CI] | McNemar maj. vote, Holm p |
  |---|---|---|---|---|---|
  | 10% | 170 | 68.82 ± 2.72 | 68.46 ± 4.68 | −0.37 [−3.34, +2.51] | 1.00 |
  | 25% | ≈ 426 | 78.83 ± 0.73 | 78.55 ± 3.05 | −0.28 [−3.06, +2.40] | 1.00 |
  | 50% | ≈ 853 | 82.13 ± 0.46 | 81.36 ± 0.66 | −0.78 [−3.96, +2.45] | 1.00 |
  | 100% (b4a) | 1706 | 83.50 ± 0.67 | 84.72 ± 2.29 | +1.23 [−2.01, +4.70] | 0.95 |

  The learning curve is steep (−15 Macro-F1 from 100% to 10%, mostly in `Rejected` F1: 80 → 51), so the setting
  could detect a data-efficiency effect. No such effect appears: Δ is ≈ 0 at every fraction and does not grow as
  data shrinks. Keypoint supervision does not act as a sample-efficiency regulariser for a DINOv2 backbone here.

**H2: ambiguity-aware soft labels (batch `b7_h2_soft_dinov2`).** Phase 1 showed that the Leaned/Straight labels follow
an implicit ≈ 3° threshold, that 16.8% of train `Leaned` images have a GT angle < 3°, and that the test errors
concentrate in the 2–6° band. Hard one-hot targets force the model to be confident on these inconsistent,
near-threshold images. Soft targets that encode the measured angle should reduce this.
- *Intervention:* for Leaned/Straight train images with GT keypoints (1,149 of 1,426), the target becomes
  (1 − α)·onehot(y) + α·[p, 1 − p, 0], where p = sigmoid((θ − θ₀)/τ) and θ is the GT axis angle (original
  pixel aspect). The hard label always keeps ≥ 50% of the mass, so the label is softened and never flipped.
  `Rejected` images and images without keypoints keep one-hot targets. The loss is the class-weighted soft
  cross-entropy, which equals the baseline loss for one-hot targets.
- *Fixed a priori, not tuned:* α = 0.5, θ₀ = 3.03° (fitted on train only in phase 1), τ = 1°. Validation and test
  keypoints are never used.
- *Setting:* DINOv2-base with the batch-3 hyper-parameters (not re-tuned, which is conservative), `cls` and `mt`,
  seeds 0–2, with checkpoints.
- *Primary outcome:* test Macro-F1 versus the tuned models of `b4a` (paired bootstrap, McNemar, Holm over 3 pairs).
- *Secondary outcomes:*
  - test error rate in the 2–6° GT-angle band. Only 37 test images fall in this band, so this has low power and is
    descriptive;
  - calibration on test: NLL and ECE (15 equal-width bins), seed-averaged
    ([`tools/analysis/h2_soft.py`](tools/analysis/h2_soft.py), written before any H2 result was seen).
- *Prediction:* Macro-F1 increases, errors in the 2–6° band decrease, and NLL / ECE decrease.
- *Expected limitation:* the test labels share the same ≈ 3° ambiguity, so the attainable gain in accuracy is
  bounded. The GT-angle rule reaches 91.6% Leaned-vs-Straight accuracy on test, and the classifiers reach ≈ 90%.
- **Result: primary outcome not significant; calibration improves.** Test, mean ± std over seeds 0–2
  ([`analysis.txt`](results/marimo/b7_h2_soft_dinov2/analysis.txt), [`h2_soft.txt`](results/analysis/h2/h2_soft.txt)):

  | model | Macro-F1 | Δ vs hard labels [95% CI] | McNemar Holm p | error 2–6° (n = 37) | NLL | ECE |
  |---|---|---|---|---|---|---|
  | `cls`, hard (b4a) | 83.50 ± 0.67 | – | – | 26.1% | 0.609 ± 0.007 | 10.4 ± 0.6% |
  | `cls`, soft | **85.84 ± 1.33** | +2.34 [−1.29, +6.26] | 1.00 | 17.1% | **0.461 ± 0.037** | **6.7 ± 1.0%** |
  | `mt`, hard (b4a) | 84.72 ± 2.29 | – | – | 21.6% | 0.543 ± 0.036 | 9.9 ± 1.8% |
  | `mt`, soft | 85.17 ± 1.09 | +0.44 [−2.49, +3.23] | 1.00 | 20.7% | 0.477 ± 0.031 | 7.6 ± 1.0% |

  - Every pre-registered outcome moves in the predicted direction for `cls`: Macro-F1 (+2.3, the largest gain in
    this study, but its CI includes 0), errors in the 2–6° band (26.1% → 17.1%, descriptive), NLL and ECE.
    The `Rejected` F1 also rises (80.4 → 84.3), although the targets of `Rejected` images are unchanged.
  - *Exploratory (not pre-registered):* a paired bootstrap of the per-image NLL (seed-averaged probabilities) gives
    ΔNLL = −0.148 [−0.273, −0.036] for `cls` and −0.066 [−0.172, +0.028] for `mt`.
  - For `mt` the effects are smaller. Soft `cls` and soft `mt` are equal (−0.67 [−2.66, +1.27]).
  - *Open confound:* soft targets also lower the target confidence in general, like label smoothing. Part of the
    calibration gain could therefore come from smoothing alone, not from the angle information. A uniform
    label-smoothing control with the same target mass is needed to separate the two.

**H2b: replication and smoothing controls for H2 (batch `b8_h2b_ctrl_dinov2`).** Pre-registered after the H2 result,
before any of these runs. `cls` only, because the H2 effect was seen there. Hyper-parameters and α / θ₀ / τ are as
in H2.
- *Replication:* hard (`dinov2_tuned_cls`) and soft (`dinov2_soft_cls`) targets on **new seeds 3–5**.
  - *Primary outcome:* Δ Macro-F1 (soft − hard) on seeds 3–5 only. These seeds are fresh, so this is the
    confirmatory test.
  - The pooled 6-seed result (seeds 0–5) is also reported. It is not independent of H2, because seeds 0–2 are reused.
- *Controls* (seeds 0–5): same images and the same target mass, without the per-image angle:
  - `softshuf`: the H2 recipe, but the GT angles are permuted among the images of the same class (rng 2000 + seed).
    The per-class distribution of targets is identical to H2; only the image-angle link is broken.
  - `softunif`: label smoothing between Leaned and Straight only. Each eligible image moves the class-wise mean
    H2 mass off its hard label (Leaned 0.099, Straight 0.084).
  - *Outcomes:* Macro-F1 (paired bootstrap, McNemar; Holm over the 5 pairs in the plan), NLL / ECE, and errors in
    the 2–6° band, soft vs each control.
- *Prediction:* the Macro-F1 gain replicates on seeds 3–5 (Δ > 0). Soft beats both controls on Macro-F1 and on
  2–6° errors. For calibration, part of the gain is expected from smoothing alone, so soft ≈ controls is plausible
  for NLL / ECE.
- *Decision rule (fixed now):* the paper claims an angle-specific benefit only if soft > both controls in
  Macro-F1 with a 95% CI above 0. Otherwise it reports a smoothing / calibration effect.
- **Result: no angle-specific benefit; the calibration gain comes from target smoothing.**
  Test, `cls` only ([`analysis.txt`](results/marimo/b8_h2b_ctrl_dinov2/analysis.txt),
  [`seeds345_primary.txt`](results/analysis/h2b/seeds345_primary.txt), [`h2_soft.txt`](results/analysis/h2/h2_soft.txt)):

  | targets | seeds | Macro-F1 | NLL | ECE | error 2–6° |
  |---|---|---|---|---|---|
  | hard | 0–5 | 83.18 ± 1.13 | 0.577 | 10.1% | 24.8% |
  | soft, GT angle (H2) | 0–5 | **85.10 ± 1.98** | 0.475 | 7.5% | 22.1% |
  | soft, angles shuffled within class | 0–5 | 84.30 ± 1.66 | 0.476 | 6.4% | 20.7% |
  | uniform L/S smoothing, same mass | 0–5 | 82.91 ± 2.05 | 0.492 | 7.3% | 24.3% |

  | comparison (Δ Macro-F1, paired bootstrap 95% CI) | Δ | McNemar maj. vote, Holm p |
  |---|---|---|
  | **soft − hard, new seeds 3–5 (primary, confirmatory)** | **+1.49 [−1.28, +4.44]** | 0.33 (single pair) |
  | soft − hard, seeds 0–5 (pooled, not independent of H2) | +1.91 [−0.61, +4.60] | 0.84 |
  | soft − shuffled-angle control | +0.80 [−1.33, +2.89] | 0.85 |
  | soft − uniform control | +2.19 [+0.13, +4.35] | 0.84 |
  | shuffled-angle control − hard | +1.11 [−0.97, +3.43] | 0.11 |
  | uniform control − hard | −0.28 [−2.22, +1.68] | 1.00 |

  - The Macro-F1 gain of H2 keeps its sign on fresh seeds but shrinks (+2.3 → +1.5), and its CI includes 0.
  - The decision rule is **not met**: soft targets do not beat the shuffled-angle control. So no claim is
    made that the per-image angle helps. The drop in 2–6° errors seen in H2 (26 → 17%) also does not replicate
    on seeds 3–5 (23.4 → 27.0%).
  - The calibration gain is robust (NLL 0.577 → 0.475 over 6 seeds; 0.545 → 0.489 on seeds 3–5 alone), but both
    controls reach the same NLL / ECE. It is therefore a property of softened targets, not of the angle.
  - *Exploratory:* heterogeneous soft targets (GT or shuffled angles) do slightly better than uniform smoothing
    with the same mean mass, but the evidence is weak (only soft − uniform has a CI above 0, uncorrected, and
    McNemar is not significant).

**X1: does keypoint supervision make the classifier look at the pole? (explainability, no training).**
Keypoint supervision did not raise accuracy. It could still change *what* the classifier relies on. This is
tested on the saved checkpoints ([`tools/analysis/xai_attention.py`](tools/analysis/xai_attention.py), written before
any map was inspected).
- *Images:* the 143 Leaned/Straight test images with a GT pole box (keypoint labels, same image size). The
  boxes are used for evaluation only.
- *Maps:*
  - Grad-CAM at the input of the last transformer block: ReLU(Σ grad · activation) per patch token, for the
    gradient of the predicted-class logit;
  - last-layer CLS attention, averaged over heads, over patch tokens only (register tokens excluded).
- *Primary metric:* Grad-CAM energy-in-box, i.e. the share of relevance inside the GT pole box (chance level =
  box area share, mean ≈ 9%). Computed per image and averaged over seeds 0–2.
- *Models:* tuned `cls` / `mt` for DINOv2 (b4a) and DINOv2-reg (b4b), and soft `cls` / `mt` (b7).
- *Tests:* paired over images, with a bootstrap 95% CI of the mean difference and a Wilcoxon signed-rank test.
  Holm correction over 3 comparisons: DINOv2 `mt` vs `cls`, DINOv2-reg `mt` vs `cls`, and DINOv2 soft `cls` vs hard
  `cls`.
- *Prediction:* `mt` > `cls` in energy-in-box for both backbones. Soft vs hard is two-sided, with no directional
  prediction.
- *Secondary metrics:* CLS-attention energy-in-box and the pointing-game hit rate (top patch inside the box).
- *Descriptive figure:* the maps of every image in this set misclassified by `dinov2_tuned_cls` seed 0, plus 8
  images drawn at random (rng 0). No hand-picked examples.
- *Caveat:* saliency shows correlation, not causation. A higher energy-in-box does not prove that the decision
  is made for the right reason.
- *Amendment, made before the full run:* a sanity check of the code on 4 images (maps only, no model
  comparison) showed a hot spot on a few high-norm background tokens. This is the known ViT "artifact token"
  effect (Darcet et al., 2024). A robustness variant is therefore added, which excludes tokens whose last-layer
  norm is more than 3× the median. The primary metric is unchanged.
- **Result: prediction not supported. Both models attend to the pole; keypoint supervision does not make attention more pole-focused.**
  143 images, mean over seeds 0–2, chance level (box area) 9.1%
  ([`xai_stats.txt`](results/analysis/xai/xai_stats.txt), per-image rows in
  [`results/marimo/xai_x1/`](results/marimo/xai_x1)):

  | model | Grad-CAM in box | CLS attention in box | CLS attention in box, no artifact tokens | attention pointing hit | artifact tokens / image |
  |---|---|---|---|---|---|
  | DINOv2 `cls` | 9.8% | 41.1% | 50.8% | 37.8% | 7.5 |
  | DINOv2 `mt` | 8.3% | 37.7% | 49.7% | 25.6% | 5.8 |
  | DINOv2 soft `cls` | 8.1% | 41.9% | 52.1% | 48.7% | 9.5 |
  | DINOv2-reg `cls` | 10.9% | 54.9% | 54.9% | 87.6% | 0 |
  | DINOv2-reg `mt` | 10.9% | 54.7% | 54.7% | 85.5% | 0 |

  - *Primary* (Grad-CAM energy-in-box, paired over images, Holm):
    - DINOv2 `mt` − `cls`: −1.5 pts [−2.1, −0.9], p < 0.001;
    - DINOv2-reg `mt` − `cls`: +0.1 [−1.2, +1.3], p = 0.35;
    - soft − hard `cls`: −1.7 [−2.2, −1.2], p < 0.001.

    The differences are statistically detectable but tiny. In the predicted direction there is no effect.
  - The last-block Grad-CAM is diffuse: it is at chance level for all models, and its top patch falls in the box in
    only 1–13% of images. It is therefore a weak instrument here. The CLS attention, by contrast, concentrates on the
    pole for every model (41–55% of the mass in a box that covers 9% of the image).
  - The clearest difference is between backbones, not between training modes. DINOv2-reg has no artifact tokens,
    and its attention peak lies on the pole in 86–88% of images, against 26–49% for DINOv2. Yet DINOv2-reg is the
    *less* accurate backbone here (79.97–81.31 vs 83.50–84.72 Macro-F1). Better-localised attention therefore did
    not translate into better lean classification.
  - Pre-registered descriptive figures (seed 0, no hand-picking): [all 14 images misclassified by `cls`](results/analysis/xai/xai_misclassified_by_cls_s0.jpg)
    and [8 random images](results/analysis/xai/xai_random8_rng0.jpg). The attention traces the pole shaft, while
    Grad-CAM is faint and diffuse. The small red squares in the corners are artifact tokens.
  - Caveat: these are correlational saliency measures, not causal evidence.

**X2: does the decision depend on the pole? (perturbation-based explainability, no training).** Pre-specified
after X1 and before any X2 computation. X1 relied on saliency maps, which are correlational, and its primary map
(last-block Grad-CAM) turned out to be diffuse for every model. X2 adds a perturbation test, which asks whether the
decision actually changes when the pole is hidden, and transformer-specific attribution methods whose
faithfulness is measured instead of assumed. Script: [`tools/analysis/xai2.py`](tools/analysis/xai2.py).
- *Images and models:* the same 143 Leaned/Straight test images with a GT pole box (evaluation only), and the
  same 18 checkpoints as X1 (tuned `cls` / `mt` for DINOv2 and DINOv2-reg, soft `cls` / `mt`; seeds 0–2).
- *Primary metric, pole reliance:* the image region inside the GT pole box is replaced by a strongly blurred
  version (Gaussian, radius 24 px at 448 px), and the drop in log-probability of the originally predicted class
  is measured. The same is done for 5 boxes of the same size placed at random without overlap with the pole box
  (rng 0 per image; images where no such box fits are excluded). Pole reliance = drop(pole) − mean drop(random).
  Per image, averaged over seeds.
- *Primary test:* `mt` vs `cls` for DINOv2 and for DINOv2-reg. Paired over images, with a bootstrap 95% CI of the
  mean difference and a Wilcoxon test, Holm over these 2 comparisons.
- *Prediction:* pole reliance is higher for `mt` than for `cls` on both backbones.
- *Secondary analyses:*
  1. Pole reliance for soft vs hard `cls`, and for DINOv2-reg vs DINOv2. These are two-sided, with no
     prediction.
  2. Prediction flip rate when the pole is blurred, against when a random box is blurred.
  3. Context test (descriptive): accuracy when only the pole box is kept and everything else is blurred, against
     the intact image.
  4. Attribution methods: last-layer CLS attention, attention rollout (Abnar & Zuidema, 2020), the
     gradient-weighted relevance of Chefer et al. (2021), and the X1 Grad-CAM. For each, report energy-in-box,
     and faithfulness by a deletion test: patches are blurred in order of attribution, in 10 steps up to 50% of
     the image, and the mean drop in log-probability of the predicted class (area over the curve) is compared
     with a random order.
  5. Sanity check (Adebayo et al., 2018) on `dinov2_tuned_cls` seed 0: the classification head and the last
     4 blocks are re-initialised, and the Spearman correlation between the original and the randomised map is
     reported for each method.
- *Figure rule:* the explanation figure uses the attribution method with the best deletion faithfulness, the same
  4 random images as X1, and the models DINOv2 `cls`, DINOv2 `mt` and DINOv2-reg `cls`. Grad-CAM is reported in
  the tables and the text only.
- **Result: partly supported. All models depend strongly on the pole; keypoint supervision increases this
  dependence on DINOv2-reg but not significantly on DINOv2.** 143 images (137 with 5 control boxes), mean over
  seeds 0–2 ([`xai2_stats.txt`](results/analysis/xai2/xai2_stats.txt), per-image rows in
  [`results/marimo/x2_xai/`](results/marimo/x2_xai)):

  | model | pole reliance | flip, pole blurred | flip, random box blurred | accuracy, intact | accuracy, pole only |
  |---|---|---|---|---|---|
  | DINOv2 `cls` | 1.87 | 49.2% | 1.8% | 90.0% | 49.9% |
  | DINOv2 `mt` | 2.01 | 49.4% | 3.2% | 89.3% | 43.8% |
  | DINOv2 soft `cls` | 1.64 | 52.0% | 2.1% | 91.4% | 47.3% |
  | DINOv2 soft `mt` | 1.71 | 51.7% | 2.4% | 89.7% | 49.4% |
  | DINOv2-reg `cls` | 2.26 | 49.4% | 2.1% | 88.1% | 53.4% |
  | DINOv2-reg `mt` | 2.58 | 45.9% | 1.8% | 87.9% | 55.0% |

  Pole reliance is the drop in log-probability of the predicted class when the pole box is blurred, minus the
  drop for random boxes of the same size. Accuracy is on these 143 Leaned/Straight images.
  - *Primary* (paired over 137 images, Holm over 2):
    - DINOv2 `mt` − `cls`: +0.14 [−0.04, +0.32], Wilcoxon p = 0.32, which is not significant;
    - DINOv2-reg `mt` − `cls`: +0.33 [+0.08, +0.58], p = 0.003 (Holm 0.005).
  - *Secondary* (Holm over 4):
    - soft − hard: `cls` −0.23 [−0.42, −0.04] (Holm 0.08), `mt` −0.30 [−0.49, −0.11] (Holm 0.002). Soft targets
      *reduce* the dependence on the pole;
    - DINOv2-reg − DINOv2: `cls` +0.38 [+0.13, +0.67] (Holm 0.19), `mt` +0.57 [+0.26, +0.89] (Holm 0.08).
  - Blurring the pole flips about half of the predictions, while blurring a random region of the same size flips
    2–3%. The decision therefore depends causally on the pole region. Keeping *only* the pole (everything else
    blurred) drops accuracy from about 89% to 44–55%, so the context is needed as well. A plausible reason is
    that the surroundings supply a vertical reference, but heavy blurring is also a distribution shift.
  - Attribution methods: deletion faithfulness (area over the log-prob curve, gain over a random order, all
    models pooled) is Chefer +5.80 > CLS attention +4.66 > rollout +2.44 > Grad-CAM +0.29. In the sanity check
    (median Spearman between the original map and the map after re-initialising the head and the last 4 blocks;
    high = insensitive), attention (0.03) and Grad-CAM (0.08) pass, Chefer is partly insensitive (0.70), and
    rollout fails (0.96). CLS attention is thus both faithful and model-sensitive. Following the figure rule, the
    figure uses Chefer.

## Data protocol

- **Only the official classification split** is used: train 1706 / val 341 / test 227
  (Leaned 619, Straight 1281, Rejected 374).
- **Leakage check.** The keypoint subset has its *own* split, so it overlaps with the classification split:
  - 1,676 classification images share a file stem with a keypoint image.
  - Pixel comparison shows that 1,564 of these pairs are the same photo re-encoded (max mean abs.
    difference 0.14/255). The other 112 are re-cropped versions with a different size.
  - **125 test and 189 val classification images appear in the keypoint *train* split.** Using the
    keypoint split as-is would leak.
- A content-based re-check (thumbnails, independent of file names) confirms this: 113 of the 125 test images are
  near-identical to a keypoint-train image, and the other 12 are re-cropped or slightly rotated copies of the same
  photo. The official classification split itself has only 2–3 near-duplicates between test and train (≈ 1%).
  See [`results/analysis/leakage/`](results/analysis/leakage/leakage_recheck.txt).
- Keypoint labels are therefore used **only for classification-train images**, matched by stem *and*
  image size. Cropped copies are dropped, because normalised coordinates would not transfer to them.
  This leaves 1,149 train images with keypoints.
- The keypoint loss is **masked for `Rejected`** images. 34 of them have keypoint labels, which are ignored on purpose.
- Augmentation is horizontal flip (keypoints mirrored) and colour jitter only. Rotation or shear would change the lean label.

GT geometry vs label (train split only):
- The axis angle separates Leaned from Straight with AUC 0.965. A 3° threshold agrees with the label on 90.8% of images.
- 16.8% of Leaned poles are below 3° and 5.3% of Straight poles are above it. This is evidence of subjective
  or borderline labels, which matches the caveats on the dataset card.

## Model

`MTNet` (in `train_multitask.py`) is a Hugging Face ViT backbone with two heads:
- (a) a classification head on `[CLS ; mean(patches)]`;
- (b) two learned queries that cross-attend to the patch tokens and predict the bottom/top keypoints
  (sigmoid, normalised coordinates).

| mode | description |
|---|---|
| `cls` | classification loss only |
| `mt` | + λ · SmoothL1 keypoint loss (masked where there are no keypoints) |
| `mt_geo` | `mt`, plus the classifier also sees the geometry of the *predicted* axis (sin θ, \|sin θ\|, cos θ, length) |
| `mt_geo_d` | as `mt_geo`, but the geometry comes from **detached** keypoints and has no length term |

Default training settings:
- 448 px input, batch 16
- AdamW (weight decay 0.05) with OneCycle; lr 2e-5 for the backbone and 1e-3 for the heads
- 15 epochs, λ = 5, class-weighted cross-entropy
- the best epoch is chosen on **val** Macro-F1, and test is evaluated once

## Experiment log

| batch | content | status |
|---|---|---|
| `b1_dinov2` | DINOv2-base × {cls, mt, mt_geo} × seeds 0–2 | ✅ done |
| `b2_dinov2reg_geofix` | DINOv2-with-registers × {cls, mt}; DINOv2 `mt_geo_d`; × seeds 0–2; checkpoints saved | ✅ done |
| `b3_tune_dinov2` | DINOv2: equal-budget paired random search, 8 shared configs for cls & mt (mt adds λ ∈ {1, 2, 5, 10}), 2 search seeds, **val only** | ✅ done |
| `b4a_final_dinov2` | DINOv2: best config per mode from `b3`, seeds 0–2, single test evaluation, checkpoints saved | ✅ done |
| `b3b_tune_dinov2reg` | same search (same 8 configs, same λ draws, same seeds) on DINOv2-with-registers | ✅ done |
| `b4b_final_dinov2reg` | DINOv2-reg: best config per mode from `b3b`, seeds 0–2, single test evaluation, checkpoints saved | ✅ done |
| `b5_h1_photo_dinov2` | H1: DINOv2 × {cls, mt} with `--aug photo`, seeds 0–2, checkpoints saved | ✅ done (null) |
| `b6_h3_dataeff_dinov2` | H3: DINOv2 × {cls, mt} × train fraction {10, 25, 50%}, seeds 0–2 (no checkpoints) | ✅ done (null) |
| `b7_h2_soft_dinov2` | H2: DINOv2 × {cls, mt} with GT-angle soft targets, seeds 0–2, checkpoints saved | ✅ done (F1 n.s., calibration ↑) |
| `b8_h2b_ctrl_dinov2` | H2b: cls hard / soft on new seeds 3–5; shuffled-angle and uniform-smoothing controls, seeds 0–5; checkpoints saved | ✅ done (no angle-specific effect) |

Per-run outputs are in [`results/marimo/`](results/marimo):
- `res_<tag>_s<seed>.json`: metrics, test/val predictions, probabilities and predicted keypoints
- `log_*.txt`: training logs

Fine-tuned checkpoints (from batch 2 onwards) are on the Hugging Face Hub:
[`northlattice/pd-defect-pole-lean-checkpoints`](https://huggingface.co/northlattice/pd-defect-pole-lean-checkpoints).

## Repository layout

```
train_multitask.py      training / evaluation (all modes, any HF ViT backbone)
analyze_results.py      bootstrap CIs, exact McNemar + Holm, comparison with the card baseline
tune_summary.py         ranks hyper-parameter search trials (validation only)
notebooks/              marimo notebook that drives the runs (synced from the compute sandbox)
results/marimo/<batch>/ per-run results, logs and analysis
```

## Reproduce

```bash
pip install torch torchvision transformers huggingface_hub scikit-learn pillow
```
```bash
python train_multitask.py --download --data_dir data/pd-defect --root data/pd-defect/pole_lean_assessment --mode mt --seeds 0 1 2 --workers 8 --out res_mt.json
```
```bash
python analyze_results.py --results <dir with res_*.json> --n_boot 10000
```

Hardware: one NVIDIA RTX PRO 6000 (96 GB). A 15-epoch run takes about 4.5 min.

## Data and citation

PD-Defect is released by its authors under CC BY 4.0:

```bibtex
@misc{pddefect,
  author    = {Masabathula, Sampath Balaji and Vanapalli, Srija and Kanchi, Gangadhar and
               V.A.N, Sreenivasa Rao and V, Vijaya Lalitha and Chilamkurthy, Sasank},
  title     = {{PD-Defect: A Field-Captured Smartphone Image Dataset for AI-Based Inspection of Power Distribution Infrastructure}},
  year      = {2026}, publisher = {figshare}, doi = {10.6084/m9.figshare.34329402}
}
```
