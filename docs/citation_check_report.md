# Citation check report: pole-lean paper

Files returned: `main.tex`, `references.bib`. The paper compiles with pdflatex + bibtex (`unsrtnat`), 18 pages, no BibTeX warnings, no undefined citations. All `TODO verify` notes are removed. Citation keys are unchanged (two new keys were added, see below). The `\todo{repository URL}` placeholder is untouched.

How "verified" is used below:
- **verified (read)**: I read the full text you sent or the full arXiv text.
- **verified (page)**: I checked title, authors and metadata on the publisher, arXiv, SPIE or MDPI page. Claims about the work were checked against the abstract or page summary only.
- **venue not opened**: title and authors checked on arXiv, but the venue, volume and pages come from my own knowledge, because the arXiv page does not list them or the proceedings page was not reachable. Please spot-check these before submission.

## 1. References

| key | status | what was corrected in the bib entry | source used |
|---|---|---|---|
| pddefect_data | corrected | Authors normalised: "V.A.N, Sreenivasa Rao" became "Sreenivasa Rao, V. A. N."; "V, Vijaya Lalitha" became "Lalitha, V. Vijaya". Month added. | Hugging Face card https://huggingface.co/datasets/EPDCL/pd-defect (its citation block); DOI 10.6084/m9.figshare.34329402. The figshare page itself would not load. |
| masabathula2026smartphone | corrected | Full six-author list replaces "and others"; `primaryClass` added; note removed. | https://arxiv.org/abs/2610.17620 (read in full) |
| wang2026overheadwires | corrected | "Wang and Wei" became Luping Wang and Hui Wei; volume 181, article number 115830, DOI added; note removed. | Full text PDF you sent; DOI 10.1016/j.engappai.2026.115830. Accepted 27 Jul 2026, online 31 Jul 2026. |
| wang2024leaning | corrected | Four authors: Luping Wang, Gang Liu, Shanshan Wang, Hui Wei. Volume 14, issue 8, pages 1857–1868, DOI added. | Full text PDF you sent; DOI 10.1007/s13349-024-00820-x |
| luo2026pole | corrected | Full names: Jinbin Luo, Junsheng Lin, Zhongling Miu. Volume, pages and DOI were already right. | Full text PDF you sent; DOI 10.1109/ACCESS.2026.3702160 |
| dickinson2025pole | corrected | Changed from `@misc` (arXiv) to `@inproceedings`: 36th BMVC, Sheffield, BMVA, 2025, with the proceedings URL. Full names unchanged. | https://bmvc2025.bmva.org/proceedings/976/ ; https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_976/paper.pdf ; arXiv 2608.04061. Note: the proceedings page header says "35th", the title block and BibTeX say 36th; I used 36th. |
| alam2020resilience (new) | added | New entry. | DOI 10.1007/s13753-020-00254-1 (Springer and journal page snippets). |
| chen2023inclination (new) | added | New entry. | https://www.mdpi.com/2076-3417/13/3/1688 ; DOI 10.3390/app13031688 |
| oquab2024dinov2 | corrected, venue not opened | "and others" replaced by the full 26-author list. | https://arxiv.org/abs/2304.07193. The arXiv page does not list TMLR, and OpenReview showed a browser check, so TMLR 2024 is from my knowledge. |
| darcet2024registers | verified, venue not opened | None. | https://arxiv.org/abs/2309.16588. The page does not name ICLR 2024. |
| simeoni2025dinov3 | corrected | "and others" replaced by the full 26-author list; `primaryClass` added. | https://arxiv.org/abs/2508.10104 |
| dosovitskiy2021vit | verified, venue not opened | None. | https://arxiv.org/abs/2010.11929 |
| caruana1997multitask | not checked online | None. Details are from my knowledge (Machine Learning 28(1):41–75). | none opened |
| ruder2017overview | verified | `primaryClass` added. | https://arxiv.org/abs/1706.05098 |
| szegedy2016inception | corrected | "Jon" changed to "Jonathon" Shlens. | https://arxiv.org/abs/1512.00567 |
| muller2019labelsmoothing | corrected | "Geoffrey E. Hinton" changed to "Geoffrey Hinton" (arXiv spelling); volume 32 added. | https://arxiv.org/abs/1906.02629 |
| guo2017calibration | verified | None. | https://arxiv.org/abs/1706.04599 (lists ICML 2017) |
| peterson2019human | verified | None. | https://arxiv.org/abs/1908.07086 (comment says ICCV 2019) |
| northcutt2021confident | verified, venue not opened | None. | https://arxiv.org/abs/1911.00068 |
| selvaraju2017gradcam | verified | None. A later IJCV version exists; the ICCV 2017 entry stays. | https://arxiv.org/abs/1610.02391 |
| abnar2020attention | verified, venue not opened | None. | https://arxiv.org/abs/2005.00928 |
| chefer2021transformer | verified, venue not opened | None. | https://arxiv.org/abs/2012.09838 |
| zhang2018excitation | verified, authors differ by version | None. I kept Bargal as second author (the IJCV version). The arXiv v1 lists five authors without her. | https://arxiv.org/abs/1608.00507 |
| adebayo2018sanity | verified, venue not opened | Volume 31 added. | https://arxiv.org/abs/1810.03292 |
| kapoor2023leakage | not checked online | None. | none opened |
| bouthillier2021variance | corrected | Full 17-author list; "Mohammadi Sepahvand, Naz" changed to "Sepah, Naz" (arXiv spelling); volume 3 added. Pages were not added because I could not confirm them. | https://arxiv.org/abs/2103.03098. The arXiv comment says "Submitted to MLSys2021" and does not confirm acceptance. |
| dietterich1998tests, efron1993bootstrap, mcnemar1947, holm1979, wilcoxon1945, nosek2018preregistration, bergstra2012random | not checked online | None. Details are from my knowledge. | none opened |
| loshchilov2019adamw | verified | None. | https://arxiv.org/abs/1711.05101 ("Published as a conference paper at ICLR 2019") |
| smith2019superconvergence | corrected | Article number 1100612 and DOI added. | https://spiedigitallibrary.org/conference-proceedings-of-spie/11006/1100612/Super-convergence--very-fast-training-of-neural-networks-using/10.1117/12.2520589.full |
| toshev2014deeppose | verified | None. | https://arxiv.org/abs/1312.4659 (DOI 10.1109/CVPR.2014.214) |
| carion2020detr | verified, venue not opened | None. | https://arxiv.org/abs/2005.12872 |

## 2. Claims about cited work

1. Introduction, first paragraph
   - Old: "...could reduce the inspection workload considerably \citep{masabathula2026smartphone,luo2026pole,wang2024leaning}."
   - New: "...could reduce the inspection workload, and recent systems address this and related tasks from smartphone, UAV and other monocular images \citep{...}."
   - Reason: none of the three papers shows a considerable workload reduction. Wang et al. 2024 is a monocular-camera method.

2. Introduction, third paragraph
   - Old: "...predicted angle is thresholded at $10^\circ$ \citep{masabathula2026smartphone}, while the image-level baseline on the dataset card is a fine-tuned DINOv3 classifier."
   - New: "...thresholded at $10^\circ$ in a supplementary rule \citep{...}, while the image-level baseline on the dataset card is a DINOv3 ViT-B/16 classifier."
   - Reason: Masabathula et al. call the angle module "supplementary", and the paper does not say how it combines with the detector. The dataset card says "DINOv3 ViT-B/16" and does not say "fine-tuned".

3. Abstract
   - Old: "...not the $10^\circ$ rule used in earlier work..."
   - New: "...not the $10^\circ$ rule used by the dataset authors..."
   - Reason: "earlier work" was imprecise. Luo et al. use a 15° threshold.

4. Related work, first paragraph (rewritten, includes the Wang & Wei text)
   - Old: "...from UAV images with detectors followed by line fitting \citep{luo2026pole}, ... In all these systems, geometry and classification are handled by separate components. ... \todo{...}"
   - New: see the full text under section 3 below, plus the PD-Defect description: "a YOLO12m detector labels poles as leaned or straight and a separate YOLO12m pose model predicts the axis keypoints, whose angle is compared with a $10^\circ$ threshold in a supplementary rule".
   - Reason:
     - Luo et al. use YOLOv11n, then PCA plus weighted line fitting, then temporal checks on UAV video. "Line fitting" alone was incomplete.
     - Dickinson & Di Caterina share one network for detection and segmentation, so "all these systems separate geometry and classification" was too strong. The accurate statement is that the angle comes from a separate geometric step.
     - The PD-Defect model types (YOLO12m detector, YOLO12m pose) are stated in Sec. III-C.1 and III-D.2 of the paper.

5. Related work, auxiliary tasks
   - Old: "keypoint regression from deep features is standard in pose estimation \citep{toshev2014deeppose}"
   - New: "keypoints can be regressed directly from deep features \citep{toshev2014deeppose}"
   - Reason: DeepPose introduced direct regression; "standard" overstates it today.

6. Related work, label smoothing
   - Old: "label errors are common in public datasets \citep{northcutt2021confident}"
   - New: "label errors can be found even in widely used benchmark datasets"
   - Reason: Confident Learning is a method for finding label errors. "Common" overstates it.

7. Related work, leakage and variance
   - Old: "Seed and data-order variance can be as large as the effects being reported \citep{bouthillier2021variance}"
   - New: "Seed and data-order variance can make small benchmark differences unreliable"
   - Reason: I only read the arXiv abstract. The old wording was a stronger quantitative claim than I could confirm.

8. Related work, pre-registration
   - Old: "is established in other sciences" → New: "is advocated in other sciences"
   - Reason: Nosek et al. argue for preregistration. "Established" overstates it.

Sentences I checked and left unchanged:
- Wang et al. 2024 as "training-free monocular geometry": confirmed in the full text (no prior training, no calibration, low-cost monocular camera).
- Dickinson & Di Caterina as "detection transformer plus segmentation head on street-level imagery": confirmed. The data are Google Street View images, and lean comes from a line fitted to the mask.
- Luo et al. as "UAV detection plus line fitting": confirmed, and extended in item 4.
- All method-paper attributions (DINOv2, registers, DINOv3, ViT, label smoothing, calibration, Grad-CAM, rollout, Chefer, sanity checks, AdamW, one-cycle, random search, DeepPose, DETR). Their abstracts or arXiv pages support what the text says. Exceptions that I could not confirm from an abstract:
  - The pointing game in Zhang et al. (the arXiv abstract does not mention it).
  - Label smoothing in Szegedy et al. (the arXiv abstract does not mention it).
  - Both are from my knowledge of the full texts and I did not change them.

## 3. Wang & Wei (EAAI 2026)

**What the paper does** (full text read):
- Input: one monocular, uncalibrated RGB image of a roadside scene.
- Method: purely geometric and training-free. It extracts line segments and vanishing points, forms "spatial texture projections" from pairs of segments, clusters them by position and orientation into eight classes, and composes pairs from different clusters into pole faces. Overhead wires are found as projections close to the pole surface.
- Output: tilt angles of poles and wires and a relative (not metric) 3D reconstruction.
- Evaluation: pixel accuracy and IoU against ground-truth masks, plus angle errors, on the authors' FDWW dataset. A comparison on Southern Ontario poles reports 83.67% for their method against 60.93% for HRNet-OCR. Baselines are YOLO, DeepLabv3+ and HRNet-OCR. Mean frame time is 2.9 s in MATLAB on a CPU.
- Limitation they state: failure when textures cannot be projected to lines.
- It has no image-level lean classification.

**Is our distinction true?** Yes. The abstract says the model "achieves multi-task capabilities", but that means one hand-designed geometric procedure for both poles and wires. There is no learned network, no joint training, and no controlled or pre-specified experiments.

**Exact text inserted** (replaces the `\todo`):

> \citet{wang2026overheadwires} extend their earlier method to poles together with overhead wires: from a single uncalibrated image they extract line segments and vanishing points, group pairs of segments by position and orientation into candidate pole faces and wires, and reconstruct both, with their tilt angles, in a relative 3D scene without any training. They compare the estimated poles with ground-truth annotations on leaning-pole datasets, using deep segmentation networks as baselines, and do not address image-level lean classification; their model is hand-designed geometry, whereas ours is trained jointly for classification and keypoints and evaluated in controlled, pre-specified experiments.

## 4. Dataset facts

| fact | status | where |
|---|---|---|
| Classification split 1706 / 341 / 227 | confirmed | HF card, split table (total 2,274) |
| Keypoint split 1438 / 181 / 179 | confirmed | HF card and arXiv 2610.17620 Sec. III-D.2 (total 1,798) |
| Licence CC BY 4.0 | confirmed | HF card ("Attribution is required") and the HF file listing (`cc-by-4.0`). The arXiv paper states no licence. |
| Class counts: Leaned 619, Straight 1281, Rejected 374 | **not confirmed** | The HF card gives no per-class counts and does not define "Rejected". The arXiv paper reports no class counts for lean and has no Rejected class. The figshare page rendered blank in my browser. |
| DINOv3 baseline 84.14 accuracy / 81.74 Macro-F1 | confirmed on the card only | HF card: DINOv3 ViT-B/16, test accuracy 84.14%, Macro-F1 81.74% (val 83.58% / 80.57%). These numbers do **not** appear in the arXiv paper. There DINOv3 is used only for perspective validation. "Reported on the dataset card" is therefore correct. |

Related checks:
- Masabathula et al. use separate YOLO12m models: a detector that labels leaned or straight (Sec. III-C.1) and a YOLO12m pose model for keypoints (Sec. III-D.2).
- The 10° threshold on the apparent inclination is confirmed (Sec. III-C.1).
- The paper calls the angle module "supplementary" and does not say which output decides when the two disagree.
- Their reported lean angle MAE on test is 0.29°.

## 5. Possible missing related work

Added to Related work (checked on publisher pages; claims rest on the abstract or page summary, not the full text):
- Alam, M. M., Zhu, Z., Tokgoz, B. E., Zhang, J., Hwang, S. (2020). Automatic Assessment and Prediction of the Resilience of Utility Poles Using Unmanned Aerial Vehicles and Computer Vision Techniques. *Int. J. Disaster Risk Science* 11(1):119–132. doi:10.1007/s13753-020-00254-1. UAV images, a deep network to segment poles, then computed inclination angles.
- Chen, L., Chang, J., Xu, J., Yang, Z. (2023). Automatic Measurement of Inclination Angle of Utility Poles Using 2D Image and 3D Point Cloud. *Applied Sciences* 13(3):1688. doi:10.3390/app13031688. Improved Mask R-CNN, then PointNet, then an axis fitted by RANSAC.

Candidates not added:
- Zhu, Z. and Zhang, J. (2019). Automatic Utility Pole Inclination Angle Measurement Using Unmanned Aerial Vehicle and Deep Learning. *Proc. 2019 IISE Annual Conference*. The conference abstract is clear (segmentation plus inclination angle, 64 images), but it overlaps with Alam et al. 2020 and I did not want a conference note in a short paragraph.
- Prabha, R. and Nihar, K. (2024). Utility Pole Fire Risk Inspection from 2D Street-Side Images. arXiv:2406.13158. Not peer-reviewed, and it is about vegetation proximity as well as inclination.
- Lee, S. et al. (2021). Large-Scale Visual Data–Driven Probabilistic Risk Assessment of Utility Poles… *J. Constr. Eng. Manage.* 147(10). I could not get the full author list or DOI from the page.

PD-Defect users: I found no paper other than the dataset paper (arXiv 2610.17620) that uses PD-Defect. The dataset is from October 2026, so this is expected. A Google Scholar "cited by" check just before submission is worthwhile.

## 6. Open issues

1. **Class counts** (619 / 1281 / 374) and the definition of "Rejected" have no source I could reach. If you counted them from the files, say so in the paper, or cite the figshare module card once you can open it.
2. **Venue details I did not open**: TMLR for DINOv2, ICLR 2024 for registers, ICLR 2021 for ViT, CVPR 2021 for Chefer, ACL 2020 for Abnar, MLSys 2021 for Bouthillier (volume 3, no pages added), and the older classics (Caruana, Efron, McNemar, Holm, Wilcoxon, Nosek, Kapoor, Dietterich, Bergstra). Please spot-check them.
3. **Added references**: I checked Alam et al. 2020 and Chen et al. 2023 from abstracts and publisher pages only. Remove them if you want only works you have read in full.
4. **Wording you may want to decide**: "controlled, pre-specified experiments" in the Wang & Wei text echoes your own paper's claims. It is accurate for your work, but it describes what Wang & Wei did not do, so it should stay evidence-based if a reviewer pushes back.
5. **Dickinson & Di Caterina**: the arXiv posting (Aug 2026) is later than the BMVC date (Nov 2025). I cite the BMVC proceedings and give no arXiv ID.
6. **Unchanged own claims**: all numbers, tables, figures, class counts and statistics are untouched, as requested.

## Second pass (full-text check)
Checked against full text (arXiv/publisher PDFs): Zhang (Pointing Game, Sec. 4.1), Szegedy (label smoothing, Sec. 7), Müller (calibration, tighter clusters), Bouthillier (variance of the order of the increments; data sampling dominates), Darcet, Northcutt, Ruder, DINOv2, Guo, Peterson, DeepPose, DETR, Smith & Topin, Loshchilov, Chefer, Abnar & Zuidema, Adebayo, Dietterich, ViT.
Changes: Bouthillier sentence rewritten to match the paper; Dietterich sentence narrowed to McNemar; Peterson pages 9616–9625; Bouthillier pages 747–769, author "Mohammadi Sepahvand, Nazanin" (proceedings form); Chefer (782–791), Holm (6(2):65–70), Wilcoxon (1(6):80–83), Efron & Tibshirani (Chapman & Hall, 1993), Adebayo (vol. 31), Müller (vol. 32) metadata confirmed.
Not checked in full text (paywalled/books): Holm, Wilcoxon, Efron & Tibshirani, Dietterich beyond abstract.
Still open: class counts 619/1281/374 and the definition of "Rejected" (not found on the HF card, arXiv paper, or a readable figshare page).

## Third pass (full text supplied by the author)
Read in full: Holm 1979 (valid level protection for any combination of true hypotheses: confirmed), Dietterich 1998 (McNemar only test with acceptable Type I error for once-run algorithms; it does not measure training-set variation: confirmed), Alam 2020 (SegNet segmentation of UAV images, then skeleton and Hough-transform line detection: sentence made precise), Chen 2023 (improved Mask R-CNN skeleton mask + PointNet on 3D point cloud + axis fitting: confirmed). Efron & Tibshirani: only front matter/TOC available (bootstrap CIs and hypothesis-testing chapters exist); Wilcoxon 1945 still not read.
