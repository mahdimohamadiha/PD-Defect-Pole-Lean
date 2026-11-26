# Project rules & handover notes (Gemini)

This file tracks the active rules, decisions, and step-by-step progress after handover from Claude on 2026-11-26. All original rules in `CLAUDE.md` remain in effect.

## Core rules
- Communicate with the user in Persian (Farsi).
- **Save and commit every change to GitHub (`main`) immediately.** Do not wait for multiple stages to accumulate.
- **Keep READMEs public-ready.** Keep GitHub `README.md` and the HF model card (`northlattice/pd-defect-pole-lean-checkpoints`) aligned with the latest verified results.
- **HF anonymity rule (strict):** The HF account (`northlattice`) is deliberately unrelated to the author. Never put the GitHub username, real name, or affiliation on the HF model card or checkpoints repo.
- Keep a single compute notebook per platform (`notebooks/pd_defect_experiments.py`) and never shut down a compute session without asking the user first.
- **Experimental protocol (locked):**
  - Official `classification/{train,val,test}` split only; keypoints only for `cls-train` images (stem + size match), masked for `Rejected`.
  - Default augmentation (`--aug base`): horizontal flip + color jitter only.
  - Approved exception (H1, `b5_h1_photo_dinov2`): `--aug photo` adds photometric-only random gamma + low-light simulation (no geometric change).
  - Approved architectural fix (H4b, `b11_convnext_kppos`): `--kp_pos sincos` adds fixed 2D sin-cos position codes only to the LayerNormed tokens entering the keypoint cross-attention (classification features, loss, data, and augmentation unchanged).

## Verified handover state (2026-11-26)
- **GPU experiments:** All batches `b1`–`b11` and analyses (`phase1`, `h1`, `h2`, `h2b`, `xai`, `xai2`, `xai2_convnext`) are complete and synced in `results/`.
- **ConvNeXt-Base (22k) (`H4` / `H4b`):**
  - `b9_tune_convnext` (tuning), `b10_final_convnext` (`cls` & `mt`, 3 seeds), `b11_convnext_kppos` (`mt + kp_pos sincos`, 3 seeds), and `xai2_convnext` are all complete.
  - Results (227 test images, seeds 0–2): `b10 cls` Acc `82.82 ± 0.76`, Macro-F1 `79.54 ± 0.77` (MV `80.70`); `b10 mt` Acc `82.53 ± 0.67`, Macro-F1 `78.97 ± 0.66` (angle corr `0.13`, `MAE 3.28°`); `b11 mt + kp_pos` Acc `81.20 ± 1.67`, Macro-F1 `77.53 ± 1.53` (MV `80.47`, angle corr `0.87 ± 0.08`, `MAE 1.79°`). Keypoint supervision remains null on CNN.
- **Hugging Face checkpoints (`northlattice/pd-defect-pole-lean-checkpoints`):** `10.8 GB`, 72 commits; all final batches (`b2`, `b4a`, `b4b`, `b5`, `b7`, `b8`, `b10_final_convnext`, `b11_convnext_kppos`) are uploaded and the HF model card is updated with `H4`/`H4b`.
- **Citation check (`docs/citation_check_report.md`):** 3 passes completed; remaining open items to resolve in Step 2 include explicit sourcing/wording for class counts (`619/1281/374`) and `Rejected`, keep/drop decision on `alam2020resilience` & `chen2023inclination`, Wang & Wei (EAAI 2026) phrasing, spot-checking remaining venue details, and replacing `\todo{repository URL}`.
- **Paper draft (`paper/main.tex`):** ConvNeXt (`H4`, `H4b`, `xai2_convnext`) results are not yet integrated into `paper/main.tex` (abstract, main table, text, discussion, limitations still reflect ViT-only).

## Completion roadmap & step log
- [x] **Step 1 (2026-11-26):** Handover alignment — reviewed `CLAUDE.md`, `docs/RESUME.md`, `docs/citation_check_report.md`, ConvNeXt results (`b9`–`b11`, `xai2_convnext`), and HF checkpoints; created `GEMINI.md` and updated `docs/RESUME.md`.
- [x] **Step 2 (2026-11-26):** Verified official `Rejected` definition (374 images: 280/57/37, classification-only) directly from `EPDCL/pd-defect/pole_lean_assessment/README.md` and updated `paper/main.tex`; independently verified and kept `alam2020resilience` and `chen2023inclination`; neutralized Wang & Wei (EAAI 2026) comparison wording in `paper/main.tex`; spot-checked remaining `references.bib` venues (TMLR 2024, ICLR 2024, MLSys 2021, JAIR 2021, ACL 2020, NeurIPS 2018).
- [x] **Step 3 (2026-11-26):** Integrated ConvNeXt-Base (H4, H4b, xai2_convnext) across abstract, intro, method, main table, dedicated Results subsection, XAI, discussion, limitations, and conclusion in `paper/main.tex` (`d6ac1e1`) and added `liu2022convnext` to `paper/references.bib` (`a8d7213`). Note: `paper/make_figures.py` (`fig_forest`) does not yet include `b10`/`b11`.
- [x] **Step 4:** Final cleanup (`\todo` removal, Data & Code Availability links, consistency audit) for supervisor review and arXiv submission.
- [x] **Step 5 (2026-11-26):** Formatted manuscript into official *Computer Optics* DOCX template (`CO Template_en 2026-06-30.docx`) under `paper/computer_optics/`, shortened abstract to 214 words (no formulas/abbreviations), converted references to numbered AMA style with DOIs, prepared 8 separate 300-dpi JPEG figures (`01.jpg`–`08.jpg`, 1000–2000 px) in `figures_computer_optics.zip`, and generated PDF via LibreOffice.
