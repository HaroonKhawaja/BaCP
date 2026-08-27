# Which paper is which

Two different papers live under `Paper/`, plus one third-party style file.
This note says which is which so nothing gets confused for anything else.

## The workshop paper — ours

**`neurips2026/`** is the NeurIPS 2026 workshop submission. Everything in that
directory is our own work **except `neurips_2026.sty`** (see below).

| Path | What it is |
|---|---|
| `main.tex` | the file to compile |
| `sections/01_intro.tex` … `05_limitations.tex` | the 4-page body |
| `sections/A1_appendix.tex` | appendix — does not count to the page limit |
| `statements/*.tex` | reproducibility, compute, licensing, ethics, broader impact |
| `checklist.tex` | the NeurIPS paper checklist |
| `results_macros.tex` | **generated** — every number in the paper |
| `refs.bib` | bibliography |
| `bacp_framework.png` | the framework figure |
| `tools/make_results_macros.py` | generates `results_macros.tex` from run records |
| `tools/significance.py` | the paired sign / Wilcoxon tests |
| `tools/anonymization_check.py` | double-blind check |
| `BRIEF.md` | the declared source of factual claims |
| `README.md`, `SUBMISSION.md` | format decisions and the submission checklist |
| `overleaf_flat/` | **build artefact**, git-ignored — a flattened copy for Overleaf |

### Not ours, inside that directory

`neurips2026/neurips_2026.sty` is the official NeurIPS 2026 style kit, used
byte-for-byte. It is not edited and must stay beside `main.tex` for the
document to build — which is why the workshop paper is not nested one level
deeper.

## The earlier submission — not this paper

`BaCP__Appendix.pdf` and
`BaCP__Backbone_Contrastive_Pruning_for_Preserving_Representations_in_Extremely_Sparse_Neural_Networks.pdf`
are an earlier AAAI submission. They are kept for reference and are **not** the
paper under `neurips2026/`.

Do not take numbers from them. Per `neurips2026/BRIEF.md`, every ResNet-34
figure in that submission is **void**: it was produced under a `drop_last=True`
evaluation defect that scored 9,728 of 10,000 test images, alongside an EAST
pruner whose masks never updated.

## The Overleaf project

The Overleaf project is **flat** — it has no folders — and holds files from
both papers side by side. The staging script (`overleaf_flat/`) deliberately
renames ours so they cannot overwrite yours:

| Ours in Overleaf | Yours in Overleaf, untouched |
|---|---|
| `main_workshop4p.tex` | `main.tex` |
| `checklist_workshop.tex` | `checklist.tex` |
| `st_*.tex` (statements) | `methodology.tex`, `results.tex` |
| `01_intro.tex` … `A1_appendix.tex` | `references.bib`, `methodology_references.bib` |
| `results_macros.tex`, `refs.bib` | `neurips_2026.tex` |
| `bacp_framework.png` | |

**Compile `main_workshop4p.tex`** for the workshop paper. `main.tex` there is
the earlier full-length draft and is left alone.
