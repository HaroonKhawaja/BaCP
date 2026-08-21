# The static BaCP framework — every equation, with its source

This is the reference for what the code computes and where each formula comes
from. Every entry carries the implementing `file:line` and either a citation or
an explicit **CHOICE** label (a definition this project made, which the paper
must derive rather than attribute). If code and a published equation differ,
the difference is stated. Deeper history and the full citation audit live in
`docs/citations.md` and `docs/math.md`.

Notation: $f_\theta$ is the sparse student, $g$ the projection head,
$z = \mathrm{normalize}(g(f(x)))$ so $z_i^\top z_j$ is cosine similarity,
$\tau > 0$ the temperature, $B$ the batch size, $N$ the candidate count.

---

## 1. The pipeline

Pretrained model → dense finetune on the task → gradual static pruning with
recovery → BaCP → finetune. This is the prune–retrain paradigm of Han et al.
(NeurIPS 2015, arXiv 1506.02626) with the gradual schedule of Zhu & Gupta
(2017, arXiv 1710.01878), and the contrastive-pruning objective of CAP
(Xu et al., AAAI 2022, arXiv 2112.07198) applied to vision backbones.
Protocols (optimizers, LRs, epochs, batch sizes) are in
`project/test_notebooks/nb_common.py` `FAMILIES`, sourced from the paper's
appendix Tables 1–3.

## 2. The BaCP objective

$$\mathcal{L} = \lambda_1\,\mathcal{L}_{PrC} + \lambda_2\,\mathcal{L}_{FiC}
+ \lambda_3\,\mathcal{L}_{SnC} + \lambda_4\,\mathcal{L}_{CE},
\qquad \textstyle\sum_i \lambda_i = 1$$

- Implemented in `_combine_losses` (`project/bacp.py:783`); the λ→term mapping
  is **λ1→PrC, λ2→FiC, λ3→SnC, λ4→CE**, pinned by
  `test_lambda_to_loss_mapping` (`project/tests/test_losses.py`).
- The four-term decomposition is **CAP's Eq. 5** (Xu et al. 2022) with λ
  indices permuted; the PrC/SnC/FiC teacher decomposition is CAP's
  contribution, not this project's.
- **SnC aggregation — code vs paper**: with $K$ snapshots the code **averages**
  ($\frac{1}{K}\sum_k$, `project/bacp.py:516-524`); the submitted paper wrote a
  plain sum. Averaging keeps the term's scale independent of $K$ so
  $\lambda_3$ needs no retuning as snapshots accumulate. Fix the equation in
  the paper, not the code.
- The optional distillation weight $\lambda_{KD}$ sits **outside** the simplex
  (**CHOICE**, reasoned in the `_combine_losses` docstring): turning KD on
  must not silently rescale the contrastive terms.

## 3. The contrastive functional (PrC, FiC, SnC)

$$\mathcal{L}_i = -\frac{1}{|P(i)|}\sum_{j\in P(i)}
\log\frac{\exp(z_i^\top \hat z_j/\tau)}{\sum_{k=1}^{N}\exp(z_i^\top \hat z_k/\tau)}$$

- `cap_contrastive_loss` (`project/loss_functions.py:5`) — **CAP Eq. 1**
  (Xu et al. 2022). Anchors $z$ come from the student; candidates $\hat z$
  from a frozen teacher. The similarity matrix is rectangular $B \times N$,
  exactly the student–teacher block and nothing else.
- Each module is this functional against a different teacher (CAP's Table 1):
  **PrC** = frozen pretrained model, **FiC** = finetuned model, **SnC** =
  saved sparse snapshots. Unsupervised term: $P(i)$ = the teacher's view of
  sample $i$. Supervised term: $P(i)$ = every candidate sharing $i$'s label.
  Both terms share one denominator (`CAPContrastiveLoss`,
  `project/loss_functions.py:67`).
- Anchors with no positives are excluded from the mean, not divided by a
  clamped 1 (**CHOICE**, documented at the function).

Legacy losses, kept for `contrastive_mode='legacy'` only:
- `SupConLoss` (`project/loss_functions.py:108`) — Khosla et al., NeurIPS
  2020, arXiv 2004.11362, their $L^{sup}_{out}$ (Eq. 2).
- `NTXentLoss` (`project/loss_functions.py:155`) — Chen et al., ICML 2020
  (SimCLR), arXiv 2002.05709, their Eq. 1.

## 4. Distillation controls (comparison arms, not part of BaCP)

- `kd_kl_loss` (`project/loss_functions.py:229`):
  $T^2 \cdot \mathrm{KL}(p^{teacher}_T \,\|\, p^{student}_T)$ — Hinton,
  Vinyals & Dean 2015, arXiv 1503.02531; the $T^2$ factor is their Sec. 2
  gradient-scale argument. The KL *form* is the standard implementation
  convention (Hinton's paper writes soft-target cross-entropy, not KL — see
  the note at `project/loss_functions.py:210`), and the two differ by a
  constant that does not affect gradients.
- `feature_distill_loss` (`project/loss_functions.py:277`): cosine or MSE
  matching on normalized embeddings — **CHOICE**; deliberately *not* cited as
  FitNets (Romero et al. 2015), which regresses intermediate activations
  through a learned regressor and contains no cosine matching.

## 5. Pruning

All pruners share the mask mechanics of `BasePruner`
(`project/pruning_factory.py`): binary masks over the prunable scope,
re-applied after every optimizer step.

**Sparsity schedule** (`BasePruner.step`, `project/pruning_factory.py:280`):

$$s_t = s_f\left(1 - \left(1 - \tfrac{t}{T}\right)^3\right)$$

Zhu & Gupta 2017, arXiv 1710.01878, Eq. 1 with $s_i = 0$, $t_0 = 0$. A mask
update fires every `delta_T` steps (~one epoch: 88 CV / 176 ViT / 1052 LLM)
over the first 80% of the phase; the final 20% holds the mask fixed
(recovery), the ramp-then-recover proportion of Gale, Elsen & Hooker 2019
(arXiv 1902.09574). Budgets: 60 total epochs (CV/ViT), 15 (LLM), with
`recovery_epochs = 0` — the trainer's legacy interleaved per-epoch recovery
is disabled as a **declared deviation** from the appendix's "5 + 10"
protocol, whose implementation re-ran a 10-epoch recovery block after every
pruning epoch.

**Criteria** (score → global or per-layer threshold):

| pruner | score | source | file:line |
|---|---|---|---|
| `magnitude` (global) | $\|w\|$ | Han et al. 2015, arXiv 1506.02626 | `pruning_factory.py:348` |
| `local_magnitude` | $\|w\|$ per layer, ERK/uniform budget | Han et al. 2015; ERK: Evci et al. 2020, arXiv 1911.11134 | `pruning_factory.py:382` |
| `snip` | $\|w \cdot \partial\mathcal{L}/\partial w\|$ | criterion: Lee et al., ICLR 2019, arXiv 1810.02340 (Eq. 6, unnormalized); iterative application: SNIP-it, Verdenius et al. 2020, arXiv 2006.00896 | `pruning_factory.py:438` |
| `wanda` | $\|w_{ij}\| \cdot \\|X_j\\|_2$ | Sun et al. 2023, arXiv 2306.11695; grouping choice documented at the class (their own Appendix A finds per-output ranking does **not** transfer to vision) | `pruning_factory.py:481` |

Retained but **not part of the static pipeline** (dynamic sparse training,
cited-not-run): `rigl` (`:725`, Evci et al. 2020) and `east` (`:788`, Li et
al., "Pushing the Limits of Sparsity: A Bag of Tricks for Extreme Pruning",
TMLR 2025, arXiv 2411.13545).

**Prunable scope** (`set_prunable_scope`, `pruning_factory.py`): 1-D tensors,
frozen tensors, the projection head, and (by default) embeddings and task
heads are excluded. The task-head keyword set is model-type-aware — for CV the
head is `cls_head`; for LLMs it includes `classifier`/`lm_head` etc. VGG's
backbone MLP is **in** scope (defect M2, fixed; pinned by
`test_vgg_backbone_mlp_is_prunable_but_llm_classifier_is_not`).

**Head convention by family** (set in `nb_common.FAMILIES`, recorded per run
as `prune_task_head`): vision arms run with the classifier **pruned**
(`prune_task_head=True`), matching the vision-line comparators (Han et al.
prune the output fc; SNIP thresholds globally; RigL/GraNet mask the
classifier — `docs/citations.md` §4). LLM arms keep heads and embeddings
dense, the PLM convention of CAP. The scope is identical across I.P. and
BaCP within every table. The contrastive projection head is excluded
everywhere, unconditionally.

**Reported sparsity** (`check_model_sparsity` `:955`,
`check_mask_sparsity` `:976`): fraction of *prunable* weights that are zero
(value-based) or masked (mask-based); `report_sparsity` prefers the mask when
a pruner owns one. The denominator is the prunable scope, not the full
parameter count — **CHOICE**, and it must be stated with any headline number.

## 6. Sparsity levels and comparators

Levels: **0.95 / 0.97 / 0.99**, the paper's own grid. Published comparators
(paper Table 1, CIFAR-10 + SST-2) are embedded in
`project/test_notebooks/nb_common.py` `PAPER` and printed by
`results_table()` next to fresh numbers. ResNet-34 has no published row and
is reported standalone.

## 7. Provenance rules the framework enforces

- `init_source` records what the weights actually are — `imagenet_checkpoint`
  / `hf_hub` / `random` (`project/model_factory.py`); the notebooks' preflight
  refuses to train when it is not the requested one.
- A run is complete iff a record carrying its cell key exists
  (`project/runner.py`); deleting the record is the re-run mechanism.
- Failures write `status='failed'` records; a crashed run is never silently
  indistinguishable from an unscheduled one.
