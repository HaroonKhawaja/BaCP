# SUBMISSION.md — the master guide for submission day

This is the operational document. Follow it top to bottom.

- **What may be claimed:** [`BRIEF.md`](BRIEF.md) — the only source of facts.
- **Why the format is what it is:** [`README.md`](README.md) — format research,
  author-kit provenance, the results-macro system.
- **This file:** what to build, what to check, and what still needs a human.

---

## 0. Read this first — three things that are true right now

1. **Nothing in this directory has ever been compiled.** There is no TeX
   toolchain on the machine this was written on: `pdflatex`, `xelatex`,
   `lualatex`, `latexmk`, `bibtex`, `biber` and `tectonic` are all absent from
   `PATH`, and no TeX Live or MiKTeX installation exists. Every file here was
   verified by static inspection only — brace balance, environment balance,
   `\input` targets resolving, `\cite` keys resolving against `refs.bib`, no
   unescaped `%`, balanced math delimiters. **The first real build will be the
   first build. Budget time for it.**
2. **The sweep is unfinished.** At the time of writing, most result cells are
   unmeasured. They render as a loud red `??` in the PDF and log a
   `MISSING RESULT KEY` line — by design, so the paper always compiles and
   becomes correct automatically. **A `??` in the compiled PDF is a stop sign,
   not a cosmetic issue.**
3. **The submission is double-blind.** `main.tex` is set to
   `\usepackage[dblblindworkshop]{neurips_2026}`. See §5 for the anonymity trap
   this project specifically has, and run the checker before you upload.

---

## 1. Build

Nothing here can be built locally today. Two workable routes:

### Overleaf (recommended — no install)

Upload the whole `Paper/neurips2026/` directory **except** `tools/` and the
Markdown files. Set the compiler to **pdfLaTeX** and the main document to
`main.tex`. Overleaf runs the bibliography pass automatically.

### A local TeX installation

Install TeX Live (any platform) or MiKTeX (Windows), then from inside
`Paper/neurips2026`:

```
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

or, equivalently and more reliably:

```
latexmk -pdf main.tex
```

Four passes matter: `natbib` author–year citations and the `\result{}` registry
both need the extra runs to settle cross-references and page numbers.

### After every build, check the log

```
grep "MISSING RESULT KEY" main.log | sort -u     # unmeasured cells -> red ?? in the PDF
grep -i "undefined" main.log                     # undefined citations / references
grep -i "overfull" main.log                      # boxes running into the margin
```

An empty first command is the condition for submitting. A non-empty one means
the PDF contains `??` where a number should be.

### Packages the preamble requires

`amsmath`, `amssymb`, `amsfonts`, `booktabs`, `multirow`, `graphicx`, `xcolor`,
`nicefrac`, `microtype`, `siunitx`, `hyperref`, `url`, plus `natbib` (loaded by
the style file itself). All are in a full TeX Live / MiKTeX install and on
Overleaf. A minimal install will need `siunitx` and `microtype` added.

---

## 2. File inventory

| File | What it is | Owner |
|---|---|---|
| `main.tex` | The document. Preamble, package option, title, author block, abstract, section wiring, statements, checklist. | edit with care |
| `neurips_2026.sty` | The official 2026 style file, byte-for-byte from the author kit. | **do not edit** — a modified style file is a desk-reject risk |
| `sections/*.tex` | The paper body: intro, related work, method, experiments, results, discussion, conclusion, appendix. | being rewritten to fit the page limit |
| `refs.bib` | 36 bibliography entries, cross-checked against primary sources. | |
| `results_macros.tex` | **Generated.** Every number in the paper. Never hand-edit — regenerate (§4). | generated |
| `checklist.tex` | The NeurIPS Paper Checklist, filled in for this paper. Questions and guidelines are the kit's own text, unmodified; only the answers and justifications are ours. | |
| `statements/reproducibility.tex` | Protocol source, seeds, splits, hardware, record idempotence, how the numbers are generated, and the code-availability promise. | |
| `statements/compute.tex` | Hardware, wall-clock, experiment count, and the compute the paper does *not* report. | |
| `statements/licensing.tex` | Attribution and licences for every asset used. | |
| `statements/ethics.tex` | No human subjects, no personal data, public benchmark only. | |
| `statements/broader_impact.tex` | Short, honest, uninflated. | |
| `tools/make_results_macros.py` | Regenerates `results_macros.tex` from the run records. The source of truth for the macro system. | |
| `tools/anonymization_check.py` | Scans every `.tex` and `.md` here for identifying strings. Exits non-zero on a hit. | |
| `BRIEF.md` / `README.md` / `SUBMISSION.md` | Repo-local documentation. **Not part of the submission.** | |

The statements are appendix material and **do not count towards the content
page limit**. Neither does the checklist. Neither do the references.

---

## 3. Pre-submission checklist — in order

Everything with a ☐ needs a human decision or a human action. Nothing below can
be resolved from inside the repository.

### Blocking decisions

- [ ] **1. Which workshop?** Not yet chosen. `main.tex` line ~34 carries
      `\workshoptitle{FIXME -- WORKSHOP NAME NOT YET CHOSEN}`, and that string
      **prints in the first-page footer**, so it cannot be missed. Replace it
      with the workshop's exact name as the CFP writes it.
- [ ] **2. That workshop's actual page limit.** The NeurIPS workshop *norm* is
      4 pages excluding references, but the limit is set by each workshop's own
      CFP, not centrally. **Read the CFP and confirm.** The draft was originally
      sized for the 9-page main track; if the limit is 4, see §6.
- [ ] **3. Does that workshop require the checklist?** Main track: mandatory,
      desk-reject without it. Workshops: usually not. If not required, comment
      out `\input{checklist}` at the end of `main.tex`. If required, leave it —
      it is filled in.
- [ ] **4. The author block.** `main.tex` has a neutral placeholder. Under
      `dblblindworkshop` the style file suppresses it and prints
      "Anonymous Author(s)", so it is safe as-is **for the anonymous
      submission**. It must be filled in for a camera-ready or a preprint build.
- [ ] **5. The title.** It is a proposal, not a decision.
- [ ] **6. The three `summary.*` abstract keys.** `summary.delta.mean`,
      `summary.delta.max` and `summary.delta.mean.bycriterion` are
      *deliberately unregistered* and render as red `??`. They are the
      sweep-wide claims. Once the sweep lands, decide what the paper claims
      across the whole grid, register those keys, **or rewrite those sentences
      to claim only what the per-cell results support.** They will not fill
      themselves in.
- [ ] **7. Has the sweep finished, and were the macros regenerated?** See §4.
      Then rebuild and confirm `grep "MISSING RESULT KEY" main.log` is empty.
- [ ] **8. SNIP-it attribution.** `README.md` records a conflict: `BRIEF.md`
      attributes the name "SNIP-it" to de Jorge et al., but that paper is
      "Progressive Skeletonization"; the sections cite Verdenius et al., which
      appears correct. **Confirm which method the code implements** and delete
      the unused `refs.bib` entry. Do not cite both as one method.

### Re-read once the sweep lands

- [ ] **9. Checklist Q1 (Claims), Q7 (Statistical significance), Q8 (Compute).**
      All three were written for the pre-sweep state and say so in a comment at
      the top of `checklist.tex`. **Q7 is currently `[No]`** — it should become
      `[Yes]` once the three-seed headline cells exist and carry error bars.
      Q1 and Q8 need a re-read, not necessarily a change.
- [ ] **10. `statements/compute.tex`** states 97 archived pre-protocol records.
      If more runs are archived before submission, update the number from the
      archive — do not estimate it.

### Final gate, in this order

- [ ] **10a. Confirm `main.tex`'s `\inputsection{...}` lines match the files
      actually in `sections/`.** The body is being restructured, and section
      files are being renamed as it happens. A stale `\inputsection` name does
      **not** break the build — it prints a red `[MISSING SECTION FILE]` marker
      instead — and a section file that exists but is not listed is silently
      omitted from the PDF. Diff the two lists by eye before building:
      ```
      ls Paper/neurips2026/sections/
      grep inputsection Paper/neurips2026/main.tex
      ```
- [ ] **11. Run the anonymity check.** Must exit 0:
      ```
      python Paper/neurips2026/tools/anonymization_check.py
      ```
      And with placeholder detection, which will list the remaining `FIXME`s:
      ```
      python Paper/neurips2026/tools/anonymization_check.py --strict
      ```
- [ ] **12. Build clean.** No `MISSING RESULT KEY`, no undefined citations, no
      undefined references.
- [ ] **13. Open the PDF and look at it.** Specifically: no red `??` anywhere;
      no red `[MISSING SECTION FILE]` or `[MISSING STATEMENT FILE]` markers; the
      footer names the right workshop; the author block is anonymous; line
      numbers are present (the double-blind workshop option turns them on).
- [ ] **14. Count the content pages** against the limit from item 2.
      References, appendix, statements and checklist do not count.
- [ ] **15. Upload only the PDF** (plus supplementary material if the workshop
      accepts it). **Never upload `tools/`** — `anonymization_check.py`
      necessarily contains the identifying strings it searches for.

---

## 4. Regenerating the results after the sweep

```
python Paper/neurips2026/tools/make_results_macros.py results/runs
```

Useful variants:

```
python Paper/neurips2026/tools/make_results_macros.py results/runs --dry-run
python Paper/neurips2026/tools/make_results_macros.py results/runs --group sweep_v1
```

Then rebuild (§1) and check the log.

What the generator does, and why you should trust it over a hand-edit: it reads
the per-run JSON records, keys each on model / method / requested sparsity,
picks one record per cell and seed by phase priority, and aggregates across
seeds into a mean and a sample standard deviation. It **skips** records whose
status is not `ok`, records marked as smoke tests, and records with a non-zero
`eval_samples_dropped` — that last being the `drop_last` defect that voided the
earlier ResNet-34 figures. It prints the skip counts by reason to stderr; read
them.

`results_macros.tex` is a generated artefact. Delete it and re-run the script and
you get it back byte-for-byte. **Never hand-edit it** — a hand-edited number is
exactly the failure mode the whole macro system exists to prevent.

☐ **Open question for a human:** is `results/runs` the right directory for the
final sweep, or is a `--group` filter needed to exclude superseded records?

---

## 5. Anonymity — the trap this project has

**The git remote URL contains the author's account name.** Any code-availability
or reproducibility sentence that cites it de-anonymises the submission
instantly. This is the single highest-risk item in the package.

How it is handled:

- `statements/reproducibility.tex` and `checklist.tex` Q5 both promise release
  **on acceptance**, and offer an **anonymized mirror** on request during the
  discussion period. Neither names a URL. Checklist Q5 is therefore `[No]` on
  open access, which the checklist guidelines explicitly permit with a
  justification.
- `statements/licensing.tex` names the code licence (MIT) **without its
  copyright line**, because that line contains the author's name.
- `tools/anonymization_check.py` scans every `.tex` and `.md` in this directory
  for: the author's given and family name, any email address, any GitHub or
  GitLab account URL, the compute workspace host id <!-- anon-check: ignore, i.e. the Databricks id -->, workspace and DBFS
  paths, absolute local Windows user paths, `\thanks`, acknowledgement sections,
  funding phrases, and undisguised self-references. It additionally derives
  patterns at run time from the checkout's git remote and git identity, so it
  keeps working if the hard-coded list goes stale.

**Current state: the scan is clean.** No identifying string was found in any
`.tex` or `.md` file in this directory, including `sections/`.

If you need to mention the trap in prose (as this file does), put
`anon-check: ignore` on the line and the scanner will skip it. Use that only for
prose that *discusses* the problem, never to silence a real leak.

Two further anonymity notes:

- Non-anonymous preprints on arXiv are explicitly permitted by NeurIPS and do
  not cause rejection. If you post one, build it with `[preprint]` — the option
  is commented out and ready in `main.tex` — and fill in the author block.
- Self-citations must be disguised ("Smith et al.", not "our previous work").
  The scanner flags the undisguised form.

---

## 6. If the paper overruns the page limit

Fix it by removing content, in this order. Do **not** fix it by changing the
style file: `neurips_2026.sty` forbids altering any formatting parameter, and a
modified style file is a desk-reject risk. That rules out `\vspace` surgery,
smaller fonts, changed margins, and `\setlength` on the float parameters.

Legitimate moves, cheapest first:

1. **Move material to the appendix.** The appendix does not count. Full
   hyperparameter tables, the schedule derivation, per-model breakdowns and
   secondary ablations all belong there rather than in the body.
2. **Cut a table to its headline rows.** A four-model × three-criterion ×
   four-sparsity grid does not have to appear in full in the body; one
   representative block plus "the full grid is in Appendix A" is standard.
3. **Cut the related-work survey, not the CAP paragraph.** The explicit
   "Differences from CAP" paragraph is load-bearing — a reviewer who discovers
   the overlap unaided discounts the whole paper. Cut around it.
4. **Merge Discussion into Results**, keeping the Limitations paragraphs.
   Limitations are a checklist commitment; do not cut them to fit.
5. **Drop a figure** before dropping a limitation or a caveat.
6. **Tighten prose.** Last, not first — it is the slowest way to recover a page.

What must survive any cut: the matched-budget description (it is the fairness
claim), the labelling of single-seed and earlier-protocol numbers as such, the
statement that absolute accuracies sit below the published comparators because
the budget is smaller, and the Limitations discussion.

---

## 7. Everything still unresolved, in one list

Blocking a submission:

1. Workshop name (and therefore `\workshoptitle`).
2. That workshop's page limit, and whether it wants the checklist.
3. Whether the sweep has completed and the macros been regenerated.
4. The three `summary.*` abstract keys — register them or rewrite the sentences.
5. Title and, for any non-anonymous build, the author block.

Should be resolved but will not block:

6. SNIP-it attribution: confirm which method the code implements, delete the
   other `refs.bib` entry.
7. Whether `results/runs` needs a `--group` filter for the final sweep.
8. Checklist Q7 should flip to `[Yes]` if and when the three-seed headline
   cells land with error bars.

Could not be verified from this machine, and should be checked by a human:

9. **The document has never been compiled.** No TeX toolchain exists here.
10. The size of the archived pre-protocol record set is taken from the archive
    snapshot in the repository (97 records). An additional off-machine archive
    is referenced in the project documentation and was not counted.
11. No formal licence declaration could be established for the CIFAR-10
    distribution. `statements/licensing.tex` says so rather than asserting one,
    which the checklist guidelines permit — but if a licence is found, name it.
