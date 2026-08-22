# Ground truth for the NeurIPS 2026 workshop submission

Deadline **29 Aug 2026**. This file is the ONLY source of factual claims.
Do not invent numbers. Do not copy numbers from the two PDFs in `Paper/` —
every ResNet-34 figure in that AAAI submission is **VOID** (produced under a
`drop_last=True` eval defect that scored 9,728 of 10,000 test images, plus an
EAST pruner whose masks never updated).

## The contribution, stated honestly

BaCP applies a contrastive objective during gradual static pruning so that the
sparse student's representations stay aligned with the dense model's. The
module decomposition (PrC / FiC / SnC) is **CAP's** (Xu et al., AAAI 2022,
arXiv:2112.07198), NOT this project's. The defensible deltas are:

1. Extension from language-only (CAP) to **vision backbones** (ResNet, VGG).
2. Demonstration that the framework is **pruning-criterion-agnostic** —
   magnitude, SNIP-it and WANDA all improve under it.
3. The **extreme-sparsity regime** (up to 99.9%), where CAP reported 97%.

Reviewer objection **B7** requires an explicit "Differences from CAP"
paragraph in Related Work. Write it early and plainly — a reviewer who
discovers the overlap unaided discounts the whole paper.

## The objective

L = λ1·PrC + λ2·FiC + λ3·SnC + λ4·CE,  λ = 0.25 each,  τ = 0.15.

Per-term functional (CAP Eq. 1), anchor i with positive set P(i):

L_i = -(1/|P(i)|) Σ_{j∈P(i)} log[ exp(z_i·ẑ_j/τ) / Σ_{k=1..N} exp(z_i·ẑ_k/τ) ]

z = normalize(g(f(x))), so z_i·z_j is cosine similarity.

Two documented code-vs-paper deltas that MUST be fixed in the paper, not the
code:
- **SnC aggregation**: with K snapshots the code averages (1/K)Σ_k; the AAAI
  submission wrote a plain sum. Averaging keeps the term's scale independent
  of K. Write the average.
- The optional λ_KD sits **outside** the simplex, by design.

The implemented objective is the `legacy` variant: SupCon + NT-Xent over a
square 2B×2B matrix on cat([z_student, z_teacher]), SimCLR-style all-pairs,
with per-model trainable projection heads (`Linear(d,128)`, never pruned).
This differs from CAP's rectangular B×N single-functional form with no
projection head and a 4096-entry memory bank. State this difference.

## Pruning

Gradual static pruning, Zhu & Gupta (arXiv:1710.01878) cubic schedule:
    s_t = s_f · (1 − (1 − t/T)³)
Ramp over the first 80% of steps, mask frozen for the final 20% (recovery).
Mask recomputed every delta_T = 100 optimizer steps. Prune–retrain paradigm
of Han et al. (NeurIPS 2015, arXiv:1506.02626).

Criteria compared (all at a single **global** threshold, so the columns differ
only in score):
- **magnitude**: |W|
- **SNIP-it**: |W ⊙ grad W| -- the SCORE is SNIP's connection sensitivity
  (Lee, Ajanthan & Torr, ICLR 2019, arXiv:1810.02340); the ITERATIVE
  re-application every delta_T steps is SNIP-it (Verdenius, Stol & Forre,
  2020, arXiv:2006.00896). Cite BOTH. de Jorge et al. (arXiv:2006.09081) is
  the adjacent 'Progressive Skeletonization'/FORCE line and is NOT this.
  Confirmed against the docstring at project/pruning_factory.py:438.
- **WANDA**: |W| · ‖X‖₂   (Sun et al., arXiv:2306.11695)

The classifier head IS pruned (`prune_task_head=True`); the projection head is
never pruned and is discarded before the final fine-tune.

## Protocol — identical across both arms (this is the paper's fairness claim)

CIFAR-10, **45,000 train / 5,000 val / 10,000 held-out test** (9:1 train/val
split of the 50k training set). Batch 512 → 87 optimizer steps/epoch.
bf16 autocast; gradient clipping at global norm 10.0 (SNIP-it's published
value) on BOTH arms.

| | dense | I.P. baseline | BaCP |
|---|---|---|---|
| init | ImageNet-pretrained | dense ckpt | dense ckpt |
| optimizer | SGD 0.01 | SGD 0.01 | SGD 0.1 (**VGG: 0.05**) |
| epochs | 100, patience 20 | 50 | 50 |
| fine-tune | — | AdamW 1e-4 × 25 | AdamW 1e-4 × 25 |
| delta_T | — | 100 | 100 |
| objective | CE | CE | λ-weighted PrC+FiC+SnC+CE |

"I.P." = iterative pruning baseline: the SAME schedule, SAME criterion, SAME
epoch budget, SAME fine-tune — differing from BaCP **only in the objective**.
That matched budget is what makes the comparison clean; say so explicitly.

**Model selection**: BaCP's contrastive phase selects on training loss (it has
no supervised val signal); both fine-tune phases select on validation
accuracy. **No test-set model selection anywhere.** State this.

**VGG learning rate.** Both VGGs carry zero BatchNorm layers (ResNet-50 has
53) and hold ~85% of their parameters in the classifier MLP, so they cannot
absorb the contrastive phase's 0.1 step size. Measured on vgg19/magnitude/0.95
with everything else held identical: LR 0.10 → 89.99, LR 0.05 → 91.88,
I.P. → 91.02. This is a reported protocol detail, not a hidden tweak.

## Grid

4 models (resnet34, resnet50, vgg11, vgg19) × 3 criteria × 4 sparsities
(0.95, 0.97, 0.99, 0.999) × 2 arms + 4 dense = 100 cells.
Planned: **3 seeds on the headline cells**.

## MEASURED numbers (the only ones that may be stated as results)

Objective ablation, ResNet-50 / CIFAR-10 / magnitude, seed 1, under the
PREVIOUS protocol (60 epochs, delta_T 88, no val split) — report as an
ablation, flagged as single-seed and under the earlier protocol:

| sparsity | I.P. | control (CAP-form, frozen tied head) | legacy | legacy − I.P. |
|---|---|---|---|---|
| 0.95 | 91.71 | 91.78 | 92.43 | +0.72 |
| 0.97 | 91.35 | 91.40 | 92.04 | +0.69 |
| 0.99 | 89.80 | (run killed) | 91.24 | +1.44 |

Dense ResNet-34 / CIFAR-10 from scratch, 5 seeds: **91.64 ± 0.30**
(per-seed 91.6535, 91.5447, 91.5843, 91.2876, 92.1064). Obtained with a
defective stem — do not use as a headline dense baseline.

Run-to-run noise floor measured at ~0.1–0.3 points.

**Everything else is UNMEASURED.** The full matched sweep has not completed.

## Published comparators (quote exactly; these are real)

ResNet-50 / CIFAR-10, GraNet Table 2 (Liu et al., NeurIPS 2021,
arXiv:2106.10404). Recipe: 160 epochs, batch 128, SGD(0.9) LR 0.1, ×10 at
[80,120], wd 5e-4.
  dense 94.75 ± 0.01 · **GMP @95% = 94.52 ± 0.08** (our exact cell) ·
  SNIP 90.86 · GraSP 91.32 · SynFlow 91.22 · RigL 93.86 ± 0.25 ·
  GraNet 94.44 ± 0.01

VGG-19 / CIFAR-10 (GraNet dense 93.85 ± 0.05; GraSP dense 94.23 — the papers
disagree, note it):
  GMP@90 93.59 ± 0.10 · SNIP@95 93.43 ± 0.20 · SNIP@98 92.05 ± 0.28 ·
  GraSP@95 93.04 ± 0.18 · GraSP@98 92.19 ± 0.12 · RigL@90 93.38 ± 0.11

Note the protocol gap honestly: our budget is 50+25 epochs at batch 512, not
160 at batch 128, so our absolute numbers sit below theirs. The controlled
claim is the **I.P.-vs-BaCP delta under a matched budget**, not a
state-of-the-art absolute.

## Threats to validity that MUST appear

1. Single dataset (CIFAR-10) and one modality in the headline grid.
2. The trainable projection head means the contrastive loss can be lowered by
   moving 262k unpruned parameters, so "preserves representations" is not
   supported by the loss alone. The defensible claim is the measured one:
   this objective yields better sparse models. The frozen-shared-head variant
   is the ablation and it costs ~0.65.
3. Compute budget below the published comparators' (above).
4. Seeds: 3 on headline cells only.

## Writing rules

- Past tense for method, protocol and setup.
- **Every numeric result must come from a `\result{...}` macro** defined in
  `results_macros.tex`. Unmeasured cells resolve to a visible placeholder.
  NEVER type a bare accuracy number into prose or a table.
- Cite from `docs/citations.md` (481 lines, audited). A wrong citation is
  worse than none.
