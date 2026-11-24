# Paper draft (LaTeX)

- `main.tex`: manuscript (plain `article` class, arXiv preprint layout); `references.bib`: bibliography.
- `figures/`: all figures, as vector PDF (plots) and 300-dpi PNG. LaTeX picks the PDF versions.
- `make_figures.py`: rebuilds every data figure from the saved results in `results/` (no training):
  `python paper/make_figures.py --root <pole_lean_assessment> --font <times.ttf>` (run from the repo root; needs
  matplotlib). The figures use Times New Roman; the font file is not included in the repository.

Build: `pdflatex main && bibtex main && pdflatex main && pdflatex main` (or upload `main.tex`,
`references.bib` and `figures/` to an online LaTeX editor). Every number in the text comes from
`README.md` / `results/`. Open items are marked `\todo{...}` (red) in the PDF, and unverified bibliography
entries are marked `TODO verify` in `references.bib`.
