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

> **Work in progress.** This card is updated as new experiment batches finish.

## Checkpoints in this repository

| tag (file prefix) | backbone | training |
|---|---|---|
| `dinov2reg_cls` | DINOv2 with registers (base) | classification loss only |
| `dinov2reg_mt` | DINOv2 with registers (base) | + auxiliary keypoint loss (λ = 5) |
| `dinov2_mt_geo_d` | DINOv2 (base) | `mt` + classifier also sees the geometry of the predicted axis (detached keypoints) |

The batch-1 DINOv2 `cls` / `mt` models reported below were trained before checkpoint saving was enabled.
Their re-trained checkpoints, with tuned hyper-parameters, will be added later.

## Results

Official classification test split (227 images), mean ± std over 3 seeds, DINOv2-base:

| model | Accuracy | Macro-F1 |
|---|---|---|
| `cls` | 82.23 ± 2.08 | 78.78 ± 2.39 |
| **`mt`** | **85.90 ± 0.44** | **83.41 ± 0.94** |

The keypoint loss improves Macro-F1 by +4.63 points (paired bootstrap 95% CI [+1.98, +7.46]).
The dataset card reports a DINOv3 ViT-B/16 baseline of 84.14 Acc / 81.74 Macro-F1. That value lies inside
our confidence intervals, so no claim of outperforming it is made.

Results for the batch-2 checkpoints in this repository will be added when the batch completes.

## Files

```
b2_dinov2reg_geofix/<tag>_s<seed>.pt     seeds 0, 1, 2
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

ck = torch.load("dinov2reg_mt_s0.pt", map_location="cpu")
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
