# Literature check: PD-Defect / pole lean (classification + keypoints) — 2026-11-23

Scope: dataset PD-Defect (EPDCL/pd-defect; DOI 10.6084/m9.figshare.34329402), subset `pole_lean_assessment`.
Topic: a single network trained jointly for 3-class lean classification (Leaned / Straight / Rejected) + pole-axis keypoints (bottom, top).
Everything below was seen on pages opened in Chrome on 2026-11-23. Items marked (snippet) or (title only) were NOT opened in full.

## Key finding
The dataset's own companion paper (arXiv:2610.17620) already combines classification with keypoint geometry, but as SEPARATE models plus a rule:
- lean class from a YOLO12m detector (Leaned_Pole / Straight_Pole);
- a supplementary rule-based module computes the apparent lean angle from two keypoints (top, bottom): theta = atan2(x1 - x2, y2 - y1) in degrees; in the deployed system a pole counts as leaned when theta > 10 degrees. The paper states this is image-space inclination, not physical lean.
=> "first to use keypoints with lean classification" is NOT a safe claim. The defensible claim is: ONE jointly trained network (shared backbone, multi-task), under a leakage-free protocol.

Also relevant: the splits of the classification subset and the keypoint subset differ (checked by file stem on the HF repo: e.g. 125 classification-test images appear in keypoint-train). Project Talon (below) independently writes that split leakage in PD-Defect is not ruled out.

## Section A — Works that USE PD-Defect
1. Project Talon (Devpost, "CodeCoogs x CenterPoint Hackathon"; Wali Lambert, Aayan Khan, Ausnguy, Sakina Saifee). https://devpost.com/software/project-talon-12twbi
   - Subsets: pole lean; vegetation/conductor.
   - Pole lean (single seed): Leaned_Pole detector recall 0.818, precision 0.864, mAP50 0.915 on 179 images; 3-class lean classifier top-1 0.714, Leaned recall 0.516 on 227 images. They state neither met their own gate (defect recall >= 0.85), so lean is "unsupported" in their system. No lean angle measured.
   - Vegetation/conductor, PD-Defect test: mIoU 0.734 (vegetation IoU 0.840, conductor IoU 0.427).
   - They note VEPL and PD-Defect scores may be optimistic because split leakage is not ruled out.
   - Type: hackathon project page, not peer reviewed.
2. SpanWitness (Devpost; Kumar Abhinav; "OpenCV AI Competition 2026 powered by AWS"). https://devpost.com/software/spanwitness
   - Uses two PD-Defect photos as functional demo (OpenCV Canny/Hough/Huber line fitting; publisher labels not used as inputs). Reports two local learned pole classifiers failed their acceptance boundary and were excluded; no numbers given.
3. OwnEval "Utility pole crossarms". https://owneval.com/fcakyon/utility-pole-crossarm-keypoints
   - 24 samples from EPDCL/pd-defect, crossarm/top-cleat keypoint subset (NOT lean). Only one sample page seen: model "gemini-4-neon", F1 100.0% at 0.1 arm length on that sample (7 matched, 0 extra, 0 missed). Overall results not seen.

Per source, nothing found:
- Google Scholar: none (only the dataset's own companion paper). Queries "PD-Defect" return partial-discharge / metallurgy papers.
- Semantic Scholar: none. IEEE Xplore: none. MDPI: none. Springer: none. ScienceDirect: none.
- GitHub / Kaggle: none on Google page 1; repos linked from the Devpost pages were not opened.

## Section B — Works that CITE the dataset or arXiv:2610.17620
- Google Scholar: no "Cited by" link shown for arXiv:2610.17620 (only Save / Cite / Related articles), so nothing to list.
- Semantic Scholar paper page (https://www.semanticscholar.org/paper/05d766dd9d4b2ca32796cc1435afbeb75c1edf03): shows 13 References; no Citations list or count was shown.
- Hugging Face Papers (https://huggingface.co/papers/2610.17620): 0 models, 0 datasets, 0 Spaces, 0 collections link to it.
- Mentions of the EPDCL deployment seen only in Scholar snippets (not opened; cannot say if they cite the paper):
  - Krishnapriya & Manohar, "ROLE OF ARTIFICIAL INTELLIGENCE IN ELECTRICAL ENGINEERING AND ITS FUTURE IMPACT", IERJ, 2026 (academia.edu).
  - Lizy & Wiryakusuma, "AI for Accident Detection and Emergency Response" (Taylor & Francis chapter).
- Dataset mirrors: author-maintained fork https://huggingface.co/datasets/sampath-balaji/pd-defect (not a user of the dataset).

Dataset companion paper (context): Masabathula, Vanapalli, Kanchi, Sreenivasa Rao V.A.N., Vijaya Lalitha V., Chilamkurthy, "Smartphone-Based, Field-Deployed AI Inspection System for Power Distribution Infrastructure", arXiv:2610.17620 [cs.CV], submitted 13 Oct 2026. https://arxiv.org/abs/2610.17620

## Section C — Closely related work on pole lean/tilt (other datasets)
Classification + keypoints in ONE model for pole lean: none found. Scholar queries (g) and (h) returned only the arXiv:2610.17620 paper plus unrelated items.

- Dickinson & Di Caterina, "Advancing Utility Pole and Sign Detection Through Deep Learning", arXiv:2608.04061 (4 Aug 2026); BMVC 2025, Paper 976. https://arxiv.org/abs/2608.04061 — DETR modified with a segmentation head; lean angle estimated from mask-derived line fitting; mean absolute error 1.01 deg (1,367 of 1,433 test poles estimated accurately); custom 4,570-image Google Street View dataset released. Detection+segmentation in one model; no keypoints, no lean classification.
- Luo, Lin, Miu, "A Deep Learning-Based Framework for Automated Power Distribution Pole Inspection in Complex Backgrounds", IEEE Access 14 (2026) — see Section D. Detection + geometric (PCA/line fitting) angle; no keypoints.
- Wang, Liu, ..., Wei, "Understanding of leaning utility poles for visual monitoring of power distribution infrastructure", Journal of Civil Structural Health Monitoring, 21 June 2024 (seen in Springer listing; arXiv:2610.17620 describes it as a low-cost monocular method for lean posture from geometric image features, without model training or calibration).
- Wang & Wei, "Understanding leaning utility poles with overhead wires for monitoring power distribution infrastructure", Engineering Applications of Artificial Intelligence, 1 Oct 2026 (title/date only; article page blocked by captcha).
- Elsayed, Hashim, Rahman et al., "Deep Learning-based Utility Pole Safety Assessment from Visual Data", JOIV, 2024 (snippet: detection-based, flags pole lean; "Cited by 3").
- Zhang, Qu, Sun, Zhang et al., "A Deep Learning-based Method For Utility Pole Image Segmentation", 2025 IEEE 7th ... (snippet).
- Xu, Zhang, Qu, Zhang et al., "A Deep Learning-Based Method for Real-Time Monitoring and Anomaly Detection of Electric Pole Force States", 2025 (EPUSNet; snippet).
- Zaikin, Astapova et al., "Method for determining the inclination of a communication line pole based on UAV images", Izvestiya Southwest State University, 2025 (snippet: detects key support points and computes tilt angle).
- Zhang, Liu, Zhong, Huang, Du et al., "Multimodal fusion network for power tower semantic segmentation and inclination detection", IEEE Transactions ..., 2025 (snippet; "Cited by 8").
- Ngamsamer, Phornthepkasemsant et al., "Robust Automated Inclination Assessment of Urban Electric Poles using Multi-Algorithmic Point Cloud Analysis and RANSAC-PCA Integration", Research Square, 2026 (preprint, point clouds; snippet).
- Yu, Yang, Huang et al., "MAE-TPFNet: Pole-fall state recognition in distribution network construction surveillance videos", Displays, Dec 2026 (title only).
- Older, snippet only: Dosso et al., "Deep learning for segmentation of critical electrical infrastructure from vehicle-based images" (2022); Russell et al., "Computer vision-assisted human-in-the-loop measurements..." (2018).

## Section D — Paper requested in task 5
- Title: "A Deep Learning-Based Framework for Automated Power Distribution Pole Inspection in Complex Backgrounds"
- Authors: Jinbin Luo, Junsheng Lin, Zhongling Miu
- Venue: IEEE Access, Volume 14, pp. 88753-88764; published 10 June 2026; eISSN 2169-3536; open access (CC BY)
- DOI: 10.1109/ACCESS.2026.3702160 ; https://ieeexplore.ieee.org/document/11557132
- Abstract (summary in my words; full text on the IEEE page): UAV-based framework for pole tilt detection and tilt-angle estimation under small objects, clutter, viewpoint change and occlusion. An improved YOLOv11n (C3k2-SimAM attention + weighted BiFPN) localizes poles; a Hybrid Geometric Angle Estimation Module (HGAEM: PCA + weighted line fitting) estimates tilt; a temporal consistency strategy suppresses transient errors in UAV video. On a self-built UAV dataset: F1 82.1%, tilt-angle MAE 4.05 deg.
- Details from the paper body: ~600 UAV video sequences, ~48,000 annotated images (70% real, 30% synthetic), video-level split 7:2:1; accuracy 0.930, FPR 0.040, 86.7 FPS, 1.78M params, 6.14 GFLOPs; tilt event when angle deviation from a reference frame exceeds 15 deg for k consecutive frames; angle is image-plane, not 3D inclination.
- Does NOT use PD-Defect. The IEEE page showed "References is not available for this document".

## Section E — Queries run / not accessible
Run:
- Google Scholar: a, b, c, d, e, f, g, h (a also with "since 2025"; e and f page 2 also checked); exact-title query for the IEEE Access paper.
- Semantic Scholar: a, b, and the paper page of arXiv:2610.17620.
- Google web: "EPDCL/pd-defect"; "pd-defect" "pole_lean"; "PD-Defect" pole dataset github OR kaggle (first result page only).
- Hugging Face: papers/2610.17620 page; datasets/sampath-balaji/pd-defect.
- IEEE Xplore: a; e and f (filter 2025-2026); exact-title query; document page 11557132.
- MDPI: a, b, e, f. Springer: a (+pole), e, f (2025-2026). ScienceDirect: a (+pole), e, f (2025-2026).
- arXiv: abs + HTML of 2610.17620; abs of 2608.04061.

Not accessible / inconclusive:
- ScienceDirect article page (pii S0952197626021147): Cloudflare captcha; not bypassed.
- ScienceDirect first request: "network issue"; second attempt worked.
- IEEE Xplore query b: page stayed on "Getting results..." — inconclusive.
- IEEE Xplore e and f with 2025-2026 filter returned "No results" although the Luo paper exists — IEEE search unreliable for these phrasings.
- Springer and ScienceDirect: query b not run. Springer f was very noisy.
- No "Cited by" for arXiv:2610.17620 shown on Scholar; Semantic Scholar showed no Citations section. Absence of citations is not proven, only "not shown".
- GitHub/Kaggle not searched directly; repos linked from the Devpost pages not opened.
