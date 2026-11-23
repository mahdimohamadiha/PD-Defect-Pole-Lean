# Literature / prior-use search for PD-Defect (pole lean): 2026-11-23

Goal: find any published or public work that uses the PD-Defect dataset, especially its
`pole_lean_assessment` subset, and any work close to our method (joint classification +
keypoint supervision for pole lean).

## Sources checked

| source | query / method | result |
|---|---|---|
| OpenAlex | works citing the dataset DOI (10.6084/m9.figshare.34329402) | **1**: the authors' own arXiv paper |
| OpenAlex | works citing arXiv 2610.17620 | **0** |
| OpenAlex | title/abstract search "PD-Defect" | only the dataset itself (other hits are partial-discharge "PD defects", unrelated) |
| arXiv API | `all:"PD-Defect"`, `abs:"pole lean"`, `abs:"utility pole" AND lean`, pole inclination / tilt | 2610.17620 (authors) and 2608.04061 (Strathclyde, different data) |
| Semantic Scholar API | paper + citations | rate-limited (HTTP 429); citations endpoint returned an empty list |
| Hugging Face | models trained on `EPDCL/pd-defect`; model/dataset/space search; dataset discussions; arXiv-linked repos | no third-party models, spaces or discussions; mirror `sampath-balaji/pd-defect` (dataset author) |
| GitHub code search | `"EPDCL/pd-defect"`, `"pd-defect" pole`, `pole_lean_assessment` | only metadata dumps (Croissant JSON, dataset lists) and arXiv mirrors, no training code |
| GitHub repo search | pd-defect, pole lean, utility pole lean, APEPDCL | `briangnj/utility-pole-lean-detector` and `Eswarnath-jeyaram/PoleSentinel` were read: neither uses PD-Defect (own drive video / IMU sensors, no ML metrics). APEPDCL repos are billing/ops tools |
| Kaggle | datasets and notebooks: pd-defect, pole lean, EPDCL | nothing relevant |
| Web search (several queries) | dataset name, authors, pole lean + keypoint / multi-task | see below |

## Findings

1. **Companion paper by the dataset authors:** Masabathula et al., *Smartphone-Based, Field-Deployed AI
   Inspection System for Power Distribution Infrastructure*, arXiv:2610.17620 (13 Oct 2026), CC BY 4.0.
   This is the paper that introduces PD-Defect. It was not linked from the dataset card. Pole lean in this paper:
   - a **YOLO12m object detector** classifies each pole instance as leaned/straight
     (1,801 images, 1,441/181/179);
   - a separate **YOLO12m pose** model predicts two keypoints (bottom/top), and a **rule-based** apparent
     angle with a 10° threshold gives an interpretable lean estimate (1,798 images);
   - DINOv3 ViT-B/16 is used in the paper only for perspective validation (valid vs distorted), with its
     last 4 blocks fine-tuned. The 3-class image-level classification baseline (84.14 Acc / 81.74
     Macro-F1) appears on the dataset card, not in the paper's method section.
   - So detection, keypoints and the angle rule are **separate models in a pipeline**. Nothing is learned
     jointly, and nothing is reported on the effect of keypoint supervision on classification.
2. **No third-party use found.** OpenAlex shows one citation of the dataset (the authors' paper) and zero
   citations of the paper. There are no HF models or spaces, no Kaggle notebooks, and no GitHub training code.
3. **Closest related work on other data:**
   - Dickinson & Di Caterina, *Advancing Utility Pole and Sign Detection Through Deep Learning*, BMVC 2025 /
     arXiv:2608.04061: a DETR detector + segmentation head on Google Street View; lean angle from a line
     fitted to the mask, with 1.01° MAE.
   - UAV-based pole inclination measurement (segmentation + classical geometry; NSF PAR 10114064).
   - A YOLOv11/BiFPN pole tilt framework for complex backgrounds (ResearchGate 406899790; full text not checked).

## Implications for our paper

- No one appears to have published on this subset yet, apart from the dataset authors. The dataset is about
  6 weeks old, and the HF repo has been downloaded ~9.2k times, so early submission or an arXiv preprint matters.
- Our contributions differ from 2610.17620:
  - a single network with joint classification + keypoint supervision;
  - the leakage analysis between the classification and keypoint splits, which the authors' pipeline does not discuss;
  - equal-budget tuning with paired statistics;
  - an analysis of label subjectivity using GT angles.
- We must cite 2610.17620 and compare with its rule-based keypoint-angle approach. A natural extra baseline is
  "predicted-keypoint angle > threshold" (as in their deployed system) vs our learned classifier.
- Limitation of this search: Google Scholar / Scopus / IEEE Xplore were not queried directly (no API access).
  Semantic Scholar was rate-limited. Re-check before submission.

## Links
- https://arxiv.org/abs/2610.17620
- https://huggingface.co/datasets/EPDCL/pd-defect
- https://doi.org/10.6084/m9.figshare.34329402
- https://arxiv.org/abs/2608.04061
- https://par.nsf.gov/biblio/10114064-automatic-utility-pole-inclination-angle-measurement-using-unmanned-aerial-vehicle-deep-learning
- https://www.researchgate.net/publication/406899790_A_Deep_Learning-Based_Framework_for_Automated_Power_Distribution_Pole_Inspection_in_Complex_Backgrounds
