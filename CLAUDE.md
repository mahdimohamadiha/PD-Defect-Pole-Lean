# Project rules

- Communicate with the user in Persian (Farsi).
- **Save every result to GitHub as soon as it exists.** Compute sandboxes (marimo/molab, Kaggle) can
  disappear at any moment. After each run, analysis or any other output, copy it into this repo
  (`results/<platform>/<batch>/`, notebooks in `notebooks/`) and commit + push to `main` right away.
  Do not wait for a whole batch to finish.
- **Keep READMEs public-ready.** At every stage, update `README.md` (GitHub) and the model card of
  `northlattice/pd-defect-pole-lean-checkpoints` (HF) with the current results and status. The HF
  account is deliberately unrelated to the author: never put the GitHub username, real name or
  affiliation on the HF card.
- Keep a single compute notebook per platform and add new work to it. Do not create new notebooks.
- Never stop or shut down a compute session without asking the user first.
- Protocol: official classification split only; keypoints only for cls-train images (stem + size
  match), masked for `Rejected`; augmentation = hflip + color jitter only
  (`--aug base`, all main results). Exception approved by the user (2026-11-24) for the pre-registered H1
  experiment: `--aug photo` adds photometric-only random gamma + low-light simulation (no geometric change).
