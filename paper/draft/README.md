# Paper draft

`main.tex` (LNCS) and `supplementary.tex`. No number in the text is typed by hand:

- `\res{key}`: the key is defined in `numbers.txt` and filled from `results/` by `tools/paper_numbers.py`.
- Tables and Fig. 1 are written to `generated/` by `tools/paper_tables.py`.
- A missing result shows as a red `??key`, and a missing table as a red box.

## Build

From the repository root, once the Slurm chain (`tools/run_isic_benchmarks.sh`) has finished:

```bash
python tools/annotator_agreement.py   # results/agreement/summary.json
bash tools/paper_outputs.sh           # paper/draft/generated/: tables, curves.pdf, numbers.tex
```

Then, in `paper/draft/`, with Springer's `llncs.cls` and `splncs04.bst` beside the sources:

```bash
latexmk -pdf supplementary.tex        # first: main.tex reads its table numbers (xr)
latexmk -pdf main.tex
```

## Before submission

- `\verify{...}`: red margin notes on sentences whose direction (which model is better) was written from the pilot runs. Check each one against the final numbers and reword it where needed. Then set `\finaltrue` in `main.tex` to hide the notes.
- `\tbd{...}`: facts that are not in any results file: co-authors, affiliation, email, and the size of the clean IMA++ pool (`datasets/imapp/raw/clean_pool.json`, length of `clean`).
- `references.bib`: entries marked `VERIFY` have unchecked author lists or titles.
- Spare keys in `numbers.txt`, for example the ISIC box results and the RITM rows, are ready for any sentence that needs them.
