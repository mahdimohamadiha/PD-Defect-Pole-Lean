# PD-Defect – Pole Lean: geometry-aware multi-task model

Classification (Leaned / Straight / Rejected) + pole-axis keypoints (bottom, top) on the
[EPDCL/pd-defect](https://huggingface.co/datasets/EPDCL/pd-defect) dataset (`pole_lean_assessment`).

## Protocol
- Only the official **classification** split (train/val/test) is used, so results are comparable
  with the dataset paper's DINOv3 baseline (test Acc 84.14, Macro-F1 81.74).
- Keypoint labels are used **only for images in classification/train** (matched by file stem and
  verified by image size). The keypoint subset has its own, different split, so using its labels
  for classification val/test images would leak.
- `Rejected` images have no keypoints: the keypoint loss is masked for them.
- Augmentation: horizontal flip + color jitter only (rotation/shear would change the lean label).

## Modes
| mode | description |
|---|---|
| `cls` | baseline, classification only |
| `mt` | + auxiliary keypoint loss |
| `mt_geo` | + classification head conditioned on geometry of the *predicted* axis |

## Run on Kaggle (GPU T4/P100, Internet ON)
```bash
!git clone https://github.com/mahdimohamadiha/pd-defect-pole-lean.git
%cd pd-defect-pole-lean
!pip -q install transformers huggingface_hub scikit-learn
!python train_multitask.py --download --mode cls    --seeds 0 1 2 --out res_cls.json
!python train_multitask.py --download --mode mt     --seeds 0 1 2 --out res_mt.json
!python train_multitask.py --download --mode mt_geo --seeds 0 1 2 --out res_mt_geo.json
```
For a private repo, store a GitHub token in Kaggle Secrets and clone with
`https://<TOKEN>@github.com/mahdimohamadiha/pd-defect-pole-lean.git`.

Baseline backbone in the dataset paper: `facebook/dinov3-vitb16-pretrain-lvd1689m` (gated on HF);
pass it via `--backbone`. Default here is `facebook/dinov2-base`.

## Status
Script is syntax-checked but **not yet executed** (no torch in the authoring environment).
