# NeurIPS 2026 workshop submission — build and format notes

Ground truth for every factual claim is [`BRIEF.md`](BRIEF.md). Ground truth for
every citation is [`../../docs/citations.md`](../../docs/citations.md).
This file records the *format* decisions and where they came from.

## Build

```
cd Paper/neurips2026
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

No TeX toolchain was installed on the machine this scaffold was written on
(`pdflatex`, `bibtex`, `latexmk` all absent from PATH, and no MiKTeX/TeX Live
directory found), so **the document has never actually been compiled.** It was
verified instead by a static pass: brace balance, environment balance, every
`\input` target resolving, every `\cite` key resolving against `refs.bib`, no
`\defresultsd`/`\defresultn` without a matching `\defresult`, and an audit that
every command and environment used in `sections/*.tex` is provided by a package
`main.tex` loads. **Someone with LaTeX must run the build before trusting it.**

Regenerate the results macros:

```
python Paper/neurips2026/tools/make_results_macros.py results/runs
```

---

## Format research

All URLs fetched **2026-08-23**.

### The author kit is published for 2026 — no substitution was needed

Downloaded from the link on the official Call for Papers:

- <https://neurips.cc/Conferences/2026/CallForPapers>
- <https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip>

The ZIP contains `neurips_2026.tex`, `neurips_2026.sty` and `checklist.tex`,
dated 2026-06-23; the style file self-identifies as
`[2026-01-29 NeurIPS 2026 submission/camera-ready style file]`.

`neurips_2026.sty` in this directory is that file **byte-for-byte**
(md5 `f447d3302c8719cb27619a074c876b44`). It was not reimplemented. Do not edit
it — the instructions forbid changing any formatting parameter, and a modified
style file is a desk-reject risk.

### Package options

Verified by reading the shipped `.sty` and the header comments of the shipped
`.tex`. The option determines anonymity, line numbers, and the footer track
string:

| Option | Anonymous | Line numbers | Use for |
|---|---|---|---|
| *(none)* / `main` | yes | yes | main-track submission |
| `preprint` | no | no | arXiv / preprint — **currently set** |
| `sglblindworkshop` | no | yes | single-blind workshop submission |
| `dblblindworkshop` | **yes** | yes | double-blind workshop submission |
| `+ final` | no | no | camera-ready |
| `nonatbib` | — | — | suppress the automatic natbib load |

Both workshop options additionally **require** `\workshoptitle{...}`; the style
file uses it for the first-page footer. `main.tex` carries the commented switch
and the `\workshoptitle` line right next to it.

> **A human must choose the option before submitting.** `preprint` is set
> because it compiles standalone and reveals nothing, but it is *not* the right
> option for a double-blind workshop submission.

### Page limit

- Main track: **9 content pages** including figures and tables. References,
  appendices and the checklist do not count.
  <https://neurips.cc/Conferences/2026/MainTrackHandbook>
- Workshops: the NeurIPS workshop norm is **4 pages excluding references**,
  with unlimited references and supplementary material, and workshop papers are
  **non-archival**. This is set by each workshop's own CFP, not centrally by
  NeurIPS — the figure above is the common convention across the 2026 workshop
  CFPs, not a rule quoted from a NeurIPS-wide page.

> **A human must check the target workshop's own CFP for its page limit.**
> At 4 pages the current draft will not fit; the 9-page main-track budget is
> what the scaffold is currently sized for.

### Anonymity

Submissions are double-blind: no identifying information, self-citations
disguised ("Smith et al.", not "our previous work"), acknowledgments omitted at
submission. Non-anonymous preprints on arXiv are explicitly permitted and do not
cause rejection. Source: Main Track Handbook, above.

The `\author` block in `main.tex` is a neutral placeholder. Under
`dblblindworkshop` the style file suppresses it entirely; under `preprint` it is
typeset as written, so it must not name anyone until a human fills it in.

### Checklist

The NeurIPS Paper Checklist is **mandatory for main-track** submissions — the
kit's own `checklist.tex` says papers without it are desk-rejected. It goes
after the references and does not count toward the page limit. Workshops
generally do not require it.

`checklist.tex` is **not** in this directory. To add it: copy it out of the
author kit ZIP, delete its instruction block, keep the section heading and the
questions, answer with the `\answerYes` / `\answerNo` / `\answerNA` macros
(defined in `neurips_2026.sty`, lines 384–386), and uncomment the
`\input{checklist}` line already present in `main.tex`.

### Citation style

`neurips_2026.sty` loads `natbib` automatically unless the `nonatbib` option is
given, so `main.tex` deliberately does **not** load it again. Bibliography style
is `plainnat`, giving author–year citations via `\citep` / `\citet` — which is
what `sections/*.tex` uses.

---

## The results macro system

**No accuracy number is ever typed into the paper.** Every figure in prose, in a
table and in the abstract is a `\result{}` call. This is what lets the document
compile before the sweep finishes and become correct automatically afterwards.

### Contract

| Macro | Behaviour |
|---|---|
| `\defresult{key}{value}` | register a value |
| `\defresultsd{key}{sd}` | register a standard deviation |
| `\defresultn{key}{n}` | register the seed count |
| `\result{key}` | the value, or a loud red **??** if unregistered |
| `\resultpm{key}` | `value ± sd`; falls back to `value` if no sd; **??** if unregistered |
| `\resultn{key}` | the seed count, or **??** |
| `\resultif{key}{yes}{no}` | branch on whether the key is registered |

A missing key **never errors**. It renders red **??** and writes one
`MISSING RESULT KEY: <key>` line to `main.log`:

```
grep "MISSING RESULT KEY" main.log | sort -u
```

`\result` and friends are `\DeclareRobustCommand`s, so they are safe in captions
and section headings.

### Key namespaces

```
<model>.<arm>.<pruner>.<sparsity>    resnet50.bacp.magnitude.0.95
                                     resnet50.ip.wanda.0.999
<model>.delta.<pruner>.<sparsity>    bacp minus ip, explicit sign
<model>.dense.none.0.0               dense baseline (alias: <model>.dense)
<model>.abl{ip,control,legacy,delta}.<pruner>.<sparsity>
                                     the objective ablation
<model>.probe*                       vgg19.probelr010.magnitude.0.95
cifar10.noise.floor.{lo,hi}          run-to-run noise floor
lit.*                                published comparators
summary.*                            cross-cell claims — nothing registered
```

`arm` is `bacp` or `ip`. Sparsity is formatted `%g`, so `0.95` and `0.999`.

Two naming conventions collided while this was being written in parallel with
`sections/*.tex`. Rather than edit the sections, the generator emits **both
spellings with the same value** for the cases that differ (`<model>.dense` vs
`<model>.dense.none.0.0`, `<model>.ablip...` vs `abl.<model>.ip...`,
`cifar10.noise.floor.lo` vs `probe.noisefloor.lo`), so either resolves. If the
sections are ever normalised, the aliases can be dropped.

### Why the ablation and the probes are in separate namespaces

The objective ablation was run under the **previous** protocol — 60 epochs,
`delta_T` 88, no validation split, one seed. Putting it in the headline
namespace would let an old-protocol number silently populate a matched-sweep
table. The `abl` prefix on the *arm* makes that impossible.

Likewise the 5-seed dense ResNet-34 figure (91.64 ± 0.30) was obtained with a
defective stem and is registered as `probe.resnet34.densescratch`, **not** as
`resnet34.dense.none.0.0`. It is a measurement of run-to-run spread, not a dense
baseline. `resnet34.dense.none.0.0` stays unregistered and renders **??** until
a clean dense run lands.

`resnet50.ablcontrol.magnitude.0.99` is also deliberately unregistered — that
run was killed. It must be reported as *not run*, not as a gap.

### The generator

`tools/make_results_macros.py` is the **source of truth**, not the generated
`.tex`. The macro definitions and the hand-entered MEASURED and LITERATURE
blocks are string constants inside the script, so the output is fully
reproducible: delete `results_macros.tex`, run the script, get it back. **Never
hand-edit `results_macros.tex`.**

```
python Paper/neurips2026/tools/make_results_macros.py results/runs
python Paper/neurips2026/tools/make_results_macros.py results/runs --group sweep_v1
python Paper/neurips2026/tools/make_results_macros.py results/runs --dry-run
```

It reads the per-run JSON records written by `project/results.py`, keys each on
`model_name` / `method` / `sparsity_requested`, picks one record per (cell,
seed) by phase priority (`finetune` > `prune` > `dense` > …, latest wins within
a phase), and aggregates across seeds into mean and sample standard deviation.
Deltas are computed for every cell where both arms landed.

Records are **skipped** when `status != 'ok'`, when `is_smoke` is set, or when
`eval_samples_dropped` is non-zero — the last being the `drop_last` defect that
voided the earlier ResNet-34 figures. Counts by reason are printed to stderr.
Run against the current `results/runs`, all 26 records are smoke runs and
nothing is emitted, which is the correct state for a sweep that has not run.

The metric preferred is **`test_acc_exact_pct`**, falling back to
`test_acc_pct`, because the former is the sample-weighted figure and the latter
is a batch-mean of batch-means. Override with `--metric`. The field actually
used is written into a comment above each cell.

---

## Bibliography

`refs.bib` has **36 entries**, built from `docs/citations.md` and cross-checked
against the arXiv API on 2026-08-23 for author lists, titles and venues. Where
`docs/citations.md` records only "et al.", the entry says `and others` rather
than inventing names. Where no venue could be established from a primary source,
the entry is the arXiv preprint rather than a guessed conference.

Two corrections were applied against `docs/citations.md` and `BRIEF.md`:

1. **CAP's first author is Runxin Xu, not "Ruiqi Xu".** `docs/citations.md`
   §10 has the latter; arXiv:2112.07198 has the former. The arXiv record wins.
2. **"SNIP-it" is Verdenius, Stol & Forré (arXiv:2006.00896), not de Jorge et
   al.** `BRIEF.md` attributes the name to de Jorge et al., whose paper is
   actually "Progressive Skeletonization" (arXiv:2006.09081) — an adjacent but
   differently-named method. Both entries are in `refs.bib`;
   `sections/*.tex` currently cites `verdenius2020snipit`, which is correct.
   **A human should confirm which method the code actually implements** and
   delete the other entry. Do not cite both as if they were one method.

Read `docs/citations.md` before citing any entry for a *claim* — several carry
mis-citation flags (WANDA must not be cited for per-output ranking in a vision
setting; Hinton must not be cited for the KL form; FitNets must not be cited for
cosine feature matching; CRD must not be cited for attributing its gain to the
negatives; Wang & Isola must not be cited for the distillation reading; EAST
must be cited from the TMLR record and v4 for numbers).

Entries present but not yet cited, kept because `BRIEF.md` or the task named
them: `dejorge2021snipit`, `devlin2019bert`, `dosovitskiy2021vit`,
`grill2020byol`, `liu2019roberta`, `sanh2019distilbert`, `romero2015fitnets`,
`wang2020alignment`, `picard2021seed`, `dodge2019showyourwork`. `plainnat` only
prints cited entries, so these cost nothing.

---

## Open decisions for a human

1. **Package option.** `preprint` is set. A double-blind workshop submission
   needs `dblblindworkshop` plus `\workshoptitle{...}`.
2. **Page limit.** Sized for 9 pages; the workshop norm is 4 excluding
   references. Check the target workshop's CFP.
3. **Title and author block.** The title is a proposal. The author block is a
   placeholder.
4. **The checklist.** Not present. Required for main track, usually not for
   workshops.
5. **SNIP-it attribution.** See above.
6. **The abstract's `summary.*` keys** (`summary.delta.mean`,
   `summary.delta.max`, `summary.delta.mean.bycriterion`) are intentionally
   unregistered and render **??**. They are the sweep-wide claims; a human must
   decide what to claim and register them, or rewrite those sentences.
7. **Whether `results/runs` is the right directory** for the final sweep, and
   whether an `--group` filter is needed to exclude superseded records.
