# Keypoint-supervised pole-lean classification on PD-Defect

Multi-task learning for **utility-pole lean assessment** (Leaned / Straight / Rejected) on the
[PD-Defect](https://huggingface.co/datasets/EPDCL/pd-defect) dataset (`pole_lean_assessment` subset,
CC BY 4.0). A ViT backbone is fine-tuned for 3-way classification, with an auxiliary head that
regresses the pole axis (bottom and top keypoints).

> **Status:** all main experiments are complete (default runs, validation-only equal-budget tuning and the
> final tuned test evaluation, for both DINOv2-base and DINOv2-with-registers). Remaining work is analysis
> without new training (rule baseline on predicted keypoint angle, label-subjectivity analysis) and the paper.

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

## Data protocol

- **Only the official classification split** is used: train 1706 / val 341 / test 227
  (Leaned 619, Straight 1281, Rejected 374).
- **Leakage check.** The keypoint subset has its *own* split, so it overlaps with the classification split:
  - 1,676 classification images share a file stem with a keypoint image.
  - Pixel comparison shows that 1,564 of these pairs are the same photo re-encoded (max mean abs.
    difference 0.14/255). The other 112 are re-cropped versions with a different size.
  - **125 test and 189 val classification images appear in the keypoint *train* split.** Using the
    keypoint split as-is would leak.
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
