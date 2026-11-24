---
tags:
- image-classification
- keypoint-detection
- multi-task-learning
- vision-transformer
- infrastructure-inspection
datasets:
- EPDCL/pd-defect
base_model:
- facebook/dinov2-base
- facebook/dinov2-with-registers-base
pipeline_tag: image-classification
---

# Pole-lean classification with keypoint supervision (PD-Defect)

Fine-tuned checkpoints for **utility-pole lean assessment** (3 classes: `Leaned`, `Straight`, `Rejected`)
on the `pole_lean_assessment` subset of [PD-Defect](https://huggingface.co/datasets/EPDCL/pd-defect).
A ViT backbone is fully fine-tuned for classification, optionally with an auxiliary head that regresses
the pole axis (bottom and top keypoints).

> **Status:** all main experiments are complete. The card may still change while the accompanying paper is written.

## Checkpoints in this repository

| tag (file prefix) | backbone | training |
|---|---|---|
| **`dinov2_tuned_mt`** (`b4a_final_dinov2/`) | DINOv2 (base) | classification + keypoint loss, tuned (lr 1e-5 / 3e-4, 15 ep, λ = 2) |
| `dinov2_tuned_cls` (`b4a_final_dinov2/`) | DINOv2 (base) | classification only, tuned (lr 1e-5 / 3e-4, 15 ep) |
| `dinov2reg_tuned_cls` (`b4b_final_dinov2reg/`) | DINOv2 with registers (base) | classification only, tuned (selected config = default: lr 2e-5 / 1e-3, 15 ep) |
| `dinov2reg_tuned_mt` (`b4b_final_dinov2reg/`) | DINOv2 with registers (base) | classification + keypoint loss, tuned (lr 2e-5 / 3e-3, 25 ep, λ = 1) |
| `dinov2reg_cls` | DINOv2 with registers (base) | classification loss only |
| `dinov2reg_mt` | DINOv2 with registers (base) | + auxiliary keypoint loss (λ = 5) |
| `dinov2_mt_geo_d` | DINOv2 (base) | `mt` + classifier also sees the geometry of the predicted axis (detached keypoints) |
| `dinov2_soft_cls` / `dinov2_soft_mt` (`b7_h2_soft_dinov2/`) | DINOv2 (base) | tuned config + angle-aware soft Leaned/Straight targets (H2) |
| `dinov2_photo_cls` / `dinov2_photo_mt` (`b5_h1_photo_dinov2/`) | DINOv2 (base) | tuned config + extra photometric augmentation (H1) |
| `b8_h2b_ctrl_dinov2/`: `dinov2_tuned_cls`, `dinov2_soft_cls` (seeds 3–5), `dinov2_softshuf_cls`, `dinov2_softunif_cls` (seeds 0–5) | DINOv2 (base) | replication and smoothing controls for the soft targets (H2b) |

The batch-1 DINOv2 `cls` / `mt` models (default settings) were trained before checkpoint saving was enabled.
The tuned DINOv2 checkpoints above replace them.

## Results

Official classification test split (227 images), mean ± std over 3 seeds (0, 1, 2).

**Tuned models** (hyper-parameters selected on validation only, with an equal budget for both modes):

| tag | Accuracy | Macro-F1 | Macro-F1 of 3-seed majority vote |
|---|---|---|---|
| **`dinov2_tuned_mt`** | **86.93 ± 2.17** | **84.72 ± 2.29** | **87.37** |
| `dinov2_tuned_cls` | 86.34 ± 0.76 | 83.50 ± 0.67 | 83.30 |
| `dinov2reg_tuned_cls` | 83.99 ± 2.22 | 81.31 ± 2.64 | 81.47 |
| `dinov2reg_tuned_mt` | 83.26 ± 0.88 | 79.97 ± 0.97 | 83.60 |

`mt` vs `cls` (paired bootstrap 95% CI of Δ Macro-F1): DINOv2 +1.23 [−2.0, +4.7]; DINOv2 with registers
−1.34 [−3.9, +1.2]. Neither difference is significant: after tuning, the keypoint loss leaves classification
quality unchanged while adding a pole-axis output. Re-running an identical configuration (same seeds) changed
Macro-F1 by about 1 point on average, which is the same size as these differences.

**Default hyper-parameters** (lr 2e-5 / 1e-3, 15 epochs, λ = 5):

| tag | Accuracy | Macro-F1 |
|---|---|---|
| `dinov2reg_cls` | 84.88 ± 1.11 | 82.37 ± 0.97 |
| `dinov2reg_mt` | 81.50 ± 1.17 | 78.66 ± 0.99 |
| `dinov2_mt_geo_d` | 83.41 ± 0.67 | 80.38 ± 1.79 |
| DINOv2 `cls` (no checkpoint) | 82.23 ± 2.08 | 78.78 ± 2.39 |
| DINOv2 `mt` (no checkpoint) | 85.90 ± 0.44 | 83.41 ± 0.94 |

The auxiliary keypoint loss helps DINOv2 (+4.6 Macro-F1, 95% CI [+2.0, +7.5]) but hurts DINOv2 with
registers (−3.7, CI [−6.6, −1.0]) at these untuned settings. Both effects largely disappear after tuning (see above).

**Follow-up experiments** (DINOv2, tuned config, planned before running; Δ = paired bootstrap 95% CI vs the
tuned model with the same mode):

| tag | Macro-F1 | Δ Macro-F1 | test NLL | test ECE |
|---|---|---|---|---|
| `dinov2_tuned_cls` (reference) | 83.50 ± 0.67 | – | 0.609 | 10.4% |
| **`dinov2_soft_cls`** | **85.84 ± 1.33** | +2.34 [−1.3, +6.3] | **0.461** | **6.7%** |
| `dinov2_tuned_mt` (reference) | 84.72 ± 2.29 | – | 0.543 | 9.9% |
| `dinov2_soft_mt` | 85.17 ± 1.09 | +0.44 [−2.5, +3.2] | 0.477 | 7.6% |
| `dinov2_photo_cls` | 83.47 ± 1.72 | −0.03 [−3.0, +3.0] | – | – |
| `dinov2_photo_mt` | 83.42 ± 1.22 | −1.30 [−4.1, +1.5] | – | – |

- *Soft targets:* for Leaned/Straight train images with keypoints, the target is
  0.5·onehot + 0.5·[p, 1 − p, 0], with p = sigmoid((θ − 3.03°)/1°) and θ the ground-truth axis angle.
- *Replication and controls (6 seeds, `cls`):*

  | targets | Macro-F1 | NLL | ECE |
  |---|---|---|---|
  | hard | 83.18 ± 1.13 | 0.577 | 10.1% |
  | soft, GT angle | 85.10 ± 1.98 | 0.475 | 7.5% |
  | soft, angles shuffled within class | 84.30 ± 1.66 | 0.476 | 6.4% |
  | uniform smoothing, same mass | 82.91 ± 2.05 | 0.492 | 7.3% |

  On new seeds 3–5, soft − hard is +1.49 [−1.3, +4.4], and soft − shuffled is +0.80 [−1.3, +2.9]. So there is no
  evidence that the per-image angle helps. The calibration gain is real, but it comes from softened targets in
  general.
- *Explainability:* the auxiliary keypoint loss does not concentrate Grad-CAM or CLS attention on the pole. The
  DINOv2-with-registers models attend to the pole much more consistently (the attention peak falls on the pole in
  86–88% of images vs 26–49%), yet they are the less accurate backbone here.
- *Occlusion test:* blurring the pole flips about half of the predictions of every model, against 2–3% for a
  random region of the same size. Keypoint supervision increases this reliance on the pole for DINOv2 with
  registers (significant), but not significantly for DINOv2.
- *Photometric augmentation* (random gamma and low-light simulation) does not help.
- *Data efficiency:* with 10 / 25 / 50% of the training images, `mt` and `cls` stay within 1 point of each other
  (no checkpoints).

The dataset card reports a DINOv3 ViT-B/16 baseline of 84.14 Acc / 81.74 Macro-F1. This value lies inside
our confidence intervals, so no claim of outperforming it is made.

## Files

```
b4a_final_dinov2/<tag>_s<seed>.pt        tuned DINOv2, seeds 0, 1, 2
b4b_final_dinov2reg/<tag>_s<seed>.pt     tuned DINOv2 with registers, seeds 0, 1, 2
b2_dinov2reg_geofix/<tag>_s<seed>.pt     default settings, seeds 0, 1, 2
b5_h1_photo_dinov2/<tag>_s<seed>.pt      DINOv2 + photometric augmentation, seeds 0, 1, 2
b7_h2_soft_dinov2/<tag>_s<seed>.pt       DINOv2 + soft targets, seeds 0, 1, 2
b8_h2b_ctrl_dinov2/<tag>_s<seed>.pt      DINOv2 cls: hard / soft seeds 3-5, smoothing controls seeds 0-5
```

Each file is a `torch.save` dict with these keys:
- `state_dict`: fp16 weights
- `args`: the training arguments (backbone, mode, image size, ...)
- `seed`, `best_epoch`, `val_macroF1`
- `classes`: `["Leaned", "Straight", "Rejected"]`

## Usage

The checkpoints use the `MTNet` class from the project's training script. Inputs are RGB images resized to
448×448 (bicubic), normalised with ImageNet mean/std.

```python
import torch
from train_multitask import MTNet   # project training script

ck = torch.load("dinov2_tuned_mt_s0.pt", map_location="cpu")
a = ck["args"]
model = MTNet(a["backbone"], a["size"], a["mode"])
model.load_state_dict({k: v.float() for k, v in ck["state_dict"].items()})
model.eval()
logits, keypoints = model(pixel_values)   # keypoints: [B, 4] = x_bottom, y_bottom, x_top, y_top in [0, 1]
```

## Training data and protocol

- Only the official classification split is used (train 1706 / val 341 / test 227).
- Keypoint labels come from the dataset's keypoint subset. They are used **only for classification-train
  images**, because that subset's own split places 125 classification-test images in its train split.
- The keypoint loss is masked for `Rejected` images.
- Augmentation is horizontal flip and colour jitter only.
- The best epoch is selected on validation Macro-F1, and test is evaluated once.

## Limitations

- Lean labels in the dataset are binary and subjective, with no measured angle. On the train split, 16.8%
  of `Leaned` poles have a ground-truth axis angle below 3°.
- The test set is small (227 images), so differences of 1–2 points are within noise.
- The images come from one region (Eastern Andhra Pradesh, India), so performance elsewhere is unknown.

## License and citation

The training data, PD-Defect, is CC BY 4.0. Please cite it:

```bibtex
@misc{pddefect,
  author    = {Masabathula, Sampath Balaji and Vanapalli, Srija and Kanchi, Gangadhar and
               V.A.N, Sreenivasa Rao and V, Vijaya Lalitha and Chilamkurthy, Sasank},
  title     = {{PD-Defect: A Field-Captured Smartphone Image Dataset for AI-Based Inspection of Power Distribution Infrastructure}},
  year      = {2026}, publisher = {figshare}, doi = {10.6084/m9.figshare.34329402}
}
```
