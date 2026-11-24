You are helping me prepare an arXiv preprint for submission. I have attached a zip with the LaTeX sources of
the paper: `main.tex`, `references.bib` and the `figures/` folder. Please use your browser to check every
reference in the paper, fix the bibliography and any wrong or overstated claims about other work, and then
return the corrected files together with a report of what you changed.

## What the paper is about (for context)

The paper evaluates whether pole-axis keypoints improve image-level lean classification of utility poles
(Leaned / Straight / Rejected) on the public PD-Defect dataset, using fine-tuned DINOv2 vision transformers.
All numbers, tables and figures come from our own experiments. **Do not change any of our results, numbers,
statistics or conclusions about our own experiments.** Your job concerns other people's work only.

## Step 1: find and read every cited work

For each entry in `references.bib`:

1. Find the work online (publisher page, arXiv, DOI, proceedings, Google Scholar, Semantic Scholar).
2. Read it, in full where the paper makes a specific claim about it, and at least the abstract and the
   relevant section otherwise.
3. Check the bibliographic details: full author list with correct spelling, title, venue, year, volume,
   issue, pages or article number, and DOI. Entries marked `TODO verify` in `references.bib` are known to be
   incomplete. Several entries describe 2025–2026 works whose details I took from search snippets.
4. Find every place in `main.tex` where the key is cited (`\citep`, `\citet`, `\citealp`), and check that the
   sentence describes the cited work correctly. Pay special attention to:
   - `masabathula2026smartphone` (arXiv 2610.17620, the paper that introduces PD-Defect). Check that it uses
     separate YOLO-based detector and pose models and that the keypoint angle is compared with a 10° threshold.
     Also check how it describes the DINOv3 classification baseline. Our paper says that 84.14 accuracy /
     81.74 Macro-F1 is "reported on the dataset card"; check where these numbers actually appear.
   - `pddefect_data` (figshare DOI 10.6084/m9.figshare.34329402) and the Hugging Face dataset card
     `EPDCL/pd-defect`: check the split sizes and class counts we state (train 1706 / val 341 / test 227;
     Leaned 619, Straight 1281, Rejected 374; keypoint split 1438 / 181 / 179) against what the dataset
     publishes, and check its licence (CC BY 4.0).
   - `wang2026overheadwires` (Wang & Wei, Engineering Applications of Artificial Intelligence, 2026). **Read
     this paper in full.** There is a `\todo{Summarise Wang \& Wei (EAAI 2026) after reading the full text.}`
     in the Related work section. Replace it with two or three sentences that describe what that paper does
     and how our work differs from it. Our work differs in that it jointly trains one network for
     classification and keypoints and evaluates it with controlled, pre-specified experiments; check whether
     this distinction is actually true given what Wang & Wei did.
   - `wang2024leaning`, `luo2026pole` and `dickinson2025pole`: check the one-line descriptions in Related work
     (training-free monocular geometry; UAV detection plus line fitting; detection transformer plus
     segmentation head on street-level imagery).
   - The methods papers (DINOv2, registers, DINOv3, ViT, label smoothing, calibration, Grad-CAM, attention
     rollout, Chefer et al., pointing game, sanity checks for saliency maps, leakage, benchmark variance,
     statistical tests, pre-registration, AdamW, one-cycle, random search, DeepPose, DETR): check that what we
     attribute to each one is what the paper says.
5. Check whether we have missed closely related work: other papers on utility-pole lean or inclination
   estimation from images, or other papers that use PD-Defect. Do not add a reference you have not found and
   checked yourself. List the candidates in the report and add only those that clearly belong in Related work.

## Step 2: tell me what you could not access

If you cannot find a work, or cannot read the full text where it is needed (for example because of a paywall
or a captcha), **stop before editing and give me a list** of these works with the exact title and, if known,
the DOI or URL. I will send you the PDFs. Then continue.

## Step 3: edit the files

- Fix `references.bib` so that every entry is complete and correct, and remove all `TODO verify` notes.
  Keep the existing citation keys. If you must rename a key, update every `\cite` in `main.tex` as well.
- In `main.tex`, change only:
  - sentences that describe other work, where they are wrong, imprecise or overstated;
  - the Wang & Wei `\todo`, which you replace with the summary;
  - new citations to related work you found and verified.
- Keep the style of the paper: flowing paragraphs, no bullet lists, no bold text in the body, short figure and
  table captions, British spelling, and no new section or paragraph titles.
- Leave the `\todo{repository URL}` placeholder in the "Data and code availability" section exactly as it is.
  The code link will be added later. Do not add links to code or checkpoints.
- Do not change author names, affiliations, numbers, tables or figures.
- The paper must still compile with pdflatex and bibtex (`natbib` with `unsrtnat`). Only use standard LaTeX
  packages that are already loaded.

## Step 4: return

1. The corrected `main.tex` and `references.bib` as downloadable files.
2. A report in English, as a Markdown file, with these parts:
   - **References**: a table with one row per reference: key | status (verified / corrected / not
     accessible) | what was corrected in the bib entry | source used (URL or DOI).
   - **Claims about cited work**: every sentence in `main.tex` you changed, with the old text, the new text
     and the reason.
   - **Wang & Wei (EAAI 2026)**: a short summary of the paper and the exact text you inserted.
   - **Dataset facts**: whether the split sizes, class counts, licence and the location of the DINOv3
     baseline numbers were confirmed, and where.
   - **Possible missing related work**: candidates with full references, and which ones you added.
   - **Open issues**: anything you could not verify, or anything I should decide myself.
