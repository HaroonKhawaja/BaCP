# Which paper is which

Two papers live under `Paper/`, plus one third-party style file. This note says
which is which so nothing gets confused for anything else.

## The workshop paper — ours

**`neurips2026/workshop/`** is the NeurIPS 2026 workshop submission, and it is
self-contained: everything the build needs is inside it.

**Compile `neurips2026/workshop/main_workshop.tex`.**

| Path | What it is |
|---|---|
| `main_workshop.tex` | **the file to compile** |
| `sections/01_intro.tex` … `05_limitations.tex` | the 4-page body |
| `sections/A1_appendix.tex` | appendix — does not count to the page limit |
| `statements/*.tex` | reproducibility, compute, licensing, ethics, broader impact |
| `checklist.tex` | the NeurIPS paper checklist |
| `results_macros.tex` | **generated** — every number in the paper |
| `refs.bib` | bibliography |
| `bacp_framework.png` | the framework figure |
| `tools/make_results_macros.py` | regenerates `results_macros.tex` from run records |
| `tools/significance.py` | the paired sign / Wilcoxon tests |
| `tools/anonymization_check.py` | double-blind check |
| `BRIEF.md` | the declared source of factual claims |
| `README.md`, `SUBMISSION.md` | format decisions and the submission checklist |
| `neurips_2026.sty` | **not ours** — a copy of the official kit, needed beside the main file |

Regenerate the numbers with:

    python Paper/neurips2026/workshop/tools/make_results_macros.py results/runs_grid

Never point that at `results/runs`, which holds only local smoke runs — it would
blank every value in the paper to a red `??`. See that script's header.

## Not ours — untouched

| Path | What it is |
|---|---|
| `neurips2026/neurips_2026.sty` | the official NeurIPS 2026 style kit, byte-for-byte |
| `BaCP__Appendix.pdf` | earlier AAAI submission |
| `BaCP__Backbone_Contrastive_Pruning_…​.pdf` | earlier AAAI submission |

The kit is duplicated into `workshop/` because LaTeX resolves `\usepackage`
relative to the main file. The original is left where it was.

**Do not take numbers from the two PDFs.** Per `workshop/BRIEF.md`, every
ResNet-34 figure in that submission is **void**: produced under a
`drop_last=True` evaluation defect that scored 9,728 of 10,000 test images,
alongside an EAST pruner whose masks never updated.

## Build artefact

`neurips2026/overleaf_flat/` is generated, git-ignored, and safe to delete. It is
a flattened copy of `workshop/` for the Overleaf project, which has no folders.

## The Overleaf project

Overleaf is **flat** and holds files from both papers side by side. The staging
script renames ours so they cannot overwrite the earlier draft:

| Ours in Overleaf | Yours in Overleaf, untouched |
|---|---|
| `main_workshop4p.tex` | `main.tex` |
| `checklist_workshop.tex` | `checklist.tex` |
| `st_*.tex` (statements) | `methodology.tex`, `results.tex` |
| `01_intro.tex` … `A1_appendix.tex` | `references.bib`, `methodology_references.bib` |
| `results_macros.tex`, `refs.bib` | `neurips_2026.tex` |
| `bacp_framework.png` | |

**Compile `main_workshop4p.tex`** there. `main.tex` in Overleaf is the earlier
full-length draft and is left alone.
