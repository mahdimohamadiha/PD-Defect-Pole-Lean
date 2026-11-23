# Project rules

- Communicate with the user in Persian (Farsi).
- **Save every result to GitHub as soon as it exists.** Compute sandboxes (marimo/molab, Kaggle) can
  disappear at any moment. After each run, analysis or any other output, copy it into this repo
  (`results/<platform>_<backbone>/`, notebooks in `notebooks/`) and commit + push to `main` right away.
  Do not wait for a whole batch to finish.
- Keep a single compute notebook per platform and add new work to it. Do not create new notebooks.
- Never stop or shut down a compute session without asking the user first.
- Protocol: official classification split only; keypoints only for cls-train images (stem + size
  match), masked for `Rejected`; augmentation = hflip + color jitter only.
