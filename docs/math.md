# BaCP: objectives, gradients, and the relationship to CAP

Written so that sections lift directly into the paper's method section. Code carries
one-line docstring pointers here rather than duplicating derivations.

Notation throughout: $f_\theta$ is the sparse model being pruned (the *student*),
$\hat f$ a frozen reference (*teacher*), $g$ the projection head, and
$z = \mathrm{normalize}(g(f(x))) \in \mathbb{R}^{D}$ with $\lVert z \rVert_2 = 1$ so
that $z_i^\top z_j$ is a cosine similarity. $\tau > 0$ is the temperature. $B$ is the
batch size and $N$ the number of candidate embeddings.

---

## 1. Relationship to CAP

BaCP's module decomposition is **CAP's** (Xu et al. 2022). This must be stated
explicitly in Related Work — CAP's three modules are literally PrC, FiC and SnC, and
its objective is the same four-term sum.

| | CAP | BaCP (as originally implemented) |
|---|---|---|
| Modules | PrC, FiC, SnC | PrC, FiC, SnC — identical |
| Objective | $\lambda_1\mathcal{L}^{CE} + \lambda_2\mathcal{L}^{PrC} + \lambda_3\mathcal{L}^{SnC} + \lambda_4\mathcal{L}^{FiC}$ | same four terms, $\lambda$ indices permuted |
| Loss | one functional, two positive sets, shared denominator | two implementations summed, different denominators |
| Similarities | rectangular $B \times N$, student anchors × teacher candidates | square $2B \times 2B$ over $\mathrm{cat}[z_s, z_t]$ |
| Projection head | none (contrasts the `[CLS]` state) | `Linear(d, 128)`, trainable, unpruned |
| Negatives | memory bank, $N = 4096$ pre-encoded on CPU | in-batch only, $2B$ |
| Sparsity reported | 97% | 99% |

**BaCP's defensible delta over CAP:** extension to vision backbones (CNNs and ViTs,
where CAP was language-only); the higher 99% sparsity regime; demonstration that the
framework is pruning-criterion-agnostic across magnitude, SNIP-it and WANDA; and
composition with EAST. The module decomposition itself is not new and should not be
presented as such.

---

## 2. The unified objective (CAP Eq. 1)

For anchor $i$ with positive set $P(i)$ drawn from the candidates:

$$\mathcal{L}_i = -\frac{1}{|P(i)|}\sum_{j \in P(i)} \log \frac{\exp(z_i^\top \hat z_j / \tau)}{\sum_{k=1}^{N} \exp(z_i^\top \hat z_k / \tau)}$$

and $\mathcal{L} = \frac{1}{|\mathcal{A}|}\sum_{i \in \mathcal{A}} \mathcal{L}_i$ over
the anchors $\mathcal{A} = \{i : |P(i)| > 0\}$.

Each module instantiates this twice, over the **same** candidate set and therefore
the same denominator (CAP Table 1):

| Module | teacher $\hat f$ | unsupervised $P(i)$ | supervised $P(i)$ |
|---|---|---|---|
| PrC | pretrained $\phi_{pre}$ | $\{\hat z_i\}$ | $\{\hat z_j : y_j = y_i\}$ |
| FiC | fine-tuned $\phi_{fine}$ | $\{\hat z_i\}$ | $\{\hat z_j : y_j = y_i\}$ |
| SnC | snapshots $\phi_{r'},\ r' < r$ | $\{\hat z_i\}$ | $\{\hat z_j : y_j = y_i\}$ |

$\mathcal{L}^{\text{module}} = \mathcal{L}_{\text{unsup}} + \mathcal{L}_{\text{sup}}$.

Implemented in `loss_functions.cap_contrastive_loss` and `CAPContrastiveLoss`.

### 2.1 Gradient

Write $s_{ik} = z_i^\top \hat z_k / \tau$ and $p_{ik} = \mathrm{softmax}_k(s_{ik})$.
Then

$$\frac{\partial \mathcal{L}_i}{\partial z_i} = \frac{1}{\tau}\left[\sum_{k=1}^{N} p_{ik}\,\hat z_k \;-\; \frac{1}{|P(i)|}\sum_{j \in P(i)} \hat z_j\right]$$

The bracket is *predicted candidate centroid minus positive centroid*: the update
moves $z_i$ toward its positives and away from the softmax-weighted average of all
candidates. Two properties follow directly:

- **$\hat z$ receives no gradient.** The teacher forward runs under `torch.no_grad`,
  so $\hat z$ is a constant and only $z_i$ is updated. This is why the loss is a
  distillation objective rather than a joint embedding objective.
- **$1/\tau$ scales the whole gradient.** Lowering $\tau$ sharpens $p_{ik}$ *and*
  amplifies the update, so $\tau$ and the learning rate are not independent knobs.

### 2.2 Analytic fixed points

Used as tests rather than as regression baselines (`test_cap_contrastive.py`):

- $z = \hat z = 0 \Rightarrow \mathcal{L} = \log N$ exactly. All logits vanish, the
  softmax is uniform over $N$ candidates.
- Note $\log N$, **not** $\log(2N-1)$. There is no self-similarity to mask: the
  anchor's own index in the candidate set is a legitimate positive — the same input
  seen through a different model — unlike the degenerate self-match NT-Xent must
  exclude with a $-\infty$ diagonal.

### 2.3 Numerical stability

$\mathcal{L}_i$ is invariant to a per-row constant shift of the logits, so the
implementation subtracts $\max_k s_{ik}$ before the log-sum-exp. The shift is
**detached**: it is a constant of the optimisation, and letting gradient flow through
it would add a spurious term.

---

## 3. Why the original two-loss composition worked against itself

The original code computed, per module,

$$\mathcal{L}^{\text{module}} = \underbrace{\mathcal{L}_{\text{SupCon}}(z_s, z_t, y)}_{\text{Khosla et al. 2020}} + \underbrace{\mathcal{L}_{\text{NT-Xent}}(z_s, z_t)}_{\text{Chen et al. 2020}}$$

with both losses built on $Z = \mathrm{cat}[z_s, z_t] \in \mathbb{R}^{2B \times D}$ and
$Z Z^\top$.

### 3.1 The square matrix carries three unintended blocks

$$Z Z^\top = \begin{bmatrix} z_s z_s^\top & z_s z_t^\top \\ z_t z_s^\top & z_t z_t^\top \end{bmatrix}$$

Only $z_s z_t^\top$ appears in CAP, and only $z_s z_t^\top$ appears in BaCP's own
written equations, which define $\mathcal{L}_{PrC} = \mathcal{L}^{sup}(z_{curr}, z_{pre}) + \mathcal{L}^{self}(z_{curr}, z_{pre})$
as a function of the cross-model pair. The other three blocks are artefacts of the
implementation:

- $z_s z_s^\top$ adds a within-student SimCLR term that is in neither the antecedent
  nor the paper.
- $z_t z_t^\top$ contributes constants to the denominators of teacher-anchored rows:
  it shifts the loss and carries no learning signal.
- $z_t z_s^\top$ makes teacher rows anchors. Averaging over all $2B$ rows therefore
  dilutes the loss magnitude by roughly $2\times$, which silently halves the
  effective $\lambda$ on every contrastive term.

**The implementation did not match the paper's method section.** This has to be
reconciled in either direction.

### 3.2 The two positive sets are inconsistent, and the conflict is quantitative

Because labels are repeated across views, $y_{i+B} = y_i$ by construction, so the
paired view is **always** in SupCon's positive set. Hence NT-Xent's single positive is
a *strict subset* of SupCon's — Khosla et al. §3 note that SupCon reduces to NT-Xent
at $|P(i)| = 1$. Adding NT-Xent therefore contributes no new positive; it re-weights
one SupCon already had and reclassifies every other same-class sample.

For a same-class, different-instance pair $(i, j)$ with $y_j = y_i$, $j \neq i+B$:

| | treats $j$ as | gradient weight on that pair |
|---|---|---|
| SupCon | positive → pull | $1/\lvert P(i)\rvert$ |
| NT-Xent | negative → push | full softmax weight $p_{ij}$ |

SupCon's pull on each individual positive is divided by $|P(i)|$; NT-Xent's push,
coming from a single-target cross-entropy, is not divided at all. At $B = 512$ on
CIFAR-10, $|P(i)| \approx 2B/C = 102$, so **the push dominates the pull by roughly two
orders of magnitude.** The composition converts supervised contrastive learning into
instance discrimination that actively fights its own class structure.

This is measurable in the paper's own Appendix Table 6, at 99% sparsity:

| Configuration | Accuracy |
|---|---|
| supervised only | **92.96** |
| self-supervised only | 92.66 |
| neither | 92.63 |
| both (as shipped) | **92.32** |

Best alone, worst together — the ordering a dominated pull term predicts.

### 3.3 A note on the empty-positive branch

`SupConLoss` divides by `num_pos.clamp(min=1.0)`, which looks like a dilution bug for
anchors with no positives. It is **unreachable** under the in-batch call pattern, for
the same reason as above: $|P(i)| \geq 1$ always. It becomes live only once candidates
come from a memory bank that may not contain a given label, which is why
`cap_contrastive_loss` excludes such anchors rather than contributing zeros.

---

## 4. The projection head

CAP has no projection head. BaCP added one, and the addition is why the contrastive
terms could not constrain the backbone.

`layer_check` excludes `encoder_head` from pruning, and for ResNet-50 it is a
$\mathrm{Linear}(2048, 128)$ carrying $\approx 262{,}000$ dense parameters. All three
contrastive losses are functions of $z = \mathrm{normalize}(g(r))$ only. With the
teacher frozen, the student can therefore reduce PrC/FiC/SnC by adjusting $g$ alone:

$$\min_{\theta, g} \mathcal{L}^{PrC}(g(f_\theta(x)), \hat g(\hat f(x))) \quad\text{admits solutions that leave } \theta \text{ unchanged.}$$

Ordinary contrastive learning is immune because both branches share the encoder, so
the only route to a lower loss is a better backbone. Freezing the teacher branch
introduces a free 262k-parameter adapter between the optimised weights and the loss.

Compounding this, the teachers' heads were **never trained or loaded**: `encoder_head`
is constructed in `ClassificationAndEncoderNetwork.__init__`, which runs *after* the
`load_weights` call in `BaseModelWrapper.__init__`, and the partial-load path
explicitly filters `encoder_head` keys. The target was therefore
$\mathrm{normalize}(W_{\text{rand}}\,\hat f(x))$. Not vacuous — by Johnson–Lindenstrauss
(Dasgupta & Gupta 2003) a random projection to $d = 128$ preserves pairwise inner
products to $O(1/\sqrt{d})$ distortion, which is plausibly why BaCP works at all — but
lossy and unintended.

**Together, §3 and §4 predict the ablation.** Removing PrC, SnC or CE each improves
99%-sparsity accuracy, and learnable $\lambda$ drives the three contrastive weights to
$\approx 0.003$, because those terms were never constraining the backbone.

### 4.1 `proj_mode`

- **`tied_frozen` (default)** — one frozen $W$ shared by student and every teacher:
  $z_s = \mathrm{normalize}(W f_\theta(x_1))$, $z_t = \mathrm{normalize}(W \hat f(x_2))$.
  The head is retained but cannot absorb the objective, so the only remaining route
  is $\theta$. Sharing also resolves the random-head problem by construction: one map
  applies on both sides, so its JL distortion cancels in the comparison.
- **`none`** — contrast $r$ directly, as CAP does.
- **`current`** — the original per-model trainable heads, for reproducing published
  numbers.

Verified by `test_projection_head_cannot_absorb_the_objective`: with $\lambda_{CE} = 0$,
`tied_frozen` gives zero gradient on the head and non-zero gradient on `layer4`, while
`current` puts gradient into the head.

---

## 5. Memory bank

CAP holds $N = 4096$ pre-encoded teacher examples on CPU and fetches them per batch.
Because the teachers are **frozen**, pre-encoding is exact — unlike MoCo's queue
(He et al. 2020), which tolerates drift because its key encoder moves. The only
variation is the augmentation draw.

This is a CAP component that BaCP omitted, not a contribution to claim. It slots
directly into the rectangular formulation: $z_{\text{cand}}$ grows from $B$ rows to
$B + 4096$, with no change to the loss.

---

## 6. EAST components

From Li et al. 2024 (arXiv 2411.13545).

### 6.1 DyReLU phasing

$$\mathrm{out} = \beta_t \cdot \mathrm{DyReLU}(x) + (1 - \beta_t) \cdot \mathrm{ReLU}(x), \qquad \beta_t = \mathrm{clip}\!\left(1 - \frac{t - t_{\text{start}}}{t_{\text{end}} - t_{\text{start}}},\, 0,\, 1\right)$$

DyReLU-B (Chen et al. 2020) computes
$\mathrm{out}_c = \max_k (a_c^k x_c + b_c^k)$ with $(a, b)$ predicted per-sample by a
hyperfunction. With the initialisation $a = [1, 0]$, $b = [0, 0]$ the module is
$\max(x, 0)$ — **exactly** ReLU — so the anneal introduces no discontinuity. That is
what makes phasing coherent, and it is worth stating in the paper.

The hyperfunction parameters are excluded from the pruning mask, which is correct
because phasing removes them from the forward pass. But note two consequences: at
$\beta = 0$ the adapter only *bypasses* the DyReLU submodule rather than deleting it,
so it remains in the state dict; and a 2048-channel DyReLU-B hyperfunction is
$\approx 5.2$M parameters. Any reported parameter count or checkpoint size must account
for this separately from the sparsity figure.

### 6.2 Weight sharing

Blocks $R, R{+}1, \dots$ of each residual layer reuse block $R{-}1$'s convolution
weights, each with its own scalar gain: $W_i = \sigma_i W_{R-1}$.

The target sparsity must be re-expressed against the unique parameter count, or a
shared model would retain far fewer live weights than its unshared counterpart at the
same nominal sparsity:

$$s_{\text{adj}} = 1 - \frac{(1 - s)\,n_{\text{total}}}{n_{\text{unique}}}$$

This is well defined only when $s > 1 - n_{\text{unique}}/n_{\text{total}}$; below that
threshold the requested weight budget exceeds the unique weights available and
$s_{\text{adj}} < 0$.

The gains $\sigma_i$ must be exempt from weight decay. Under the default
$5\times10^{-4}$ they decay toward zero across training, progressively silencing the
shared blocks.

### 6.3 Cyclic sparsity

$$s_{\text{target}}(t) = s_{\max} - (s_{\max} - s_{\min}) \cdot \tfrac{1}{2}\left(1 - \cos\frac{2\pi t}{T_c}\right), \qquad t \le T_c$$

Anchored at $s_{\max}$: the cycle starts and ends at $s_{\max}$ and dips to $s_{\min}$
at the half-cycle, relaxing the constraint mid-cycle to allow parameter exploration
before re-tightening. The phase matters — anchoring at $s_{\min}$ instead makes the
first update *grow* the network from $s_{\max}$ back toward $s_{\min}$, abandoning the
sparsity budget entirely.

Past $T_c$ the level is held at $s_{\max}$ and the pruner only rewires, dropping and
regrowing equal counts — the drop-and-regrow behaviour inherited from RigL
(Evci et al. 2020), with regrown weights zero-initialised and their optimizer state
cleared.

Per-layer counts must be clamped to what each layer can supply: $k$ is derived from a
*global* sparsity ratio but spent per-layer, so a layer whose ERK density differs from
the global average can otherwise be asked to prune more weights than it has active.

---

## 7. Reporting corrections

Three items where the code and the paper disagree, and the paper is wrong:

1. **SnC aggregation.** The paper writes $\mathcal{L}^{SnC} = \sum_{k=1}^{K}(\cdot)$;
   the code averages. Averaging is correct — it keeps the term's scale independent of
   $K$, so $\lambda_{SnC}$ does not need retuning as snapshots accumulate. Fix the
   equation.
2. **$\lambda$ ordering.** `_combine_losses` maps $\lambda_1 \to$ PrC,
   $\lambda_2 \to$ FiC, $\lambda_3 \to$ SnC, $\lambda_4 \to$ CE. The plotting helper
   labelled its series `['CE', 'PrC', 'SnC', 'FiC']`, so every published
   dynamic-lambda figure had three of four series mislabelled. Pinned by
   `test_lambda_to_loss_mapping`.
3. **MLM perplexity.** RoBERTa and DistilBERT are **encoder-only**, not
   "decoder-only" as the appendix states, and pseudo-perplexity for masked language
   modelling is not the standard causal quantity — its computation needs defining
   explicitly.

---

## References

- Xu, R., Luo, F., Wang, C., Chang, B., Huang, J., Huang, S., Huang, F. (2022). *From Dense to Sparse: Contrastive Pruning for Better Pre-trained Language Model Compression.* AAAI 36, 11547–11555. arXiv:2112.07198.
- Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P., Maschinot, A., Liu, C., Krishnan, D. (2020). *Supervised Contrastive Learning.* NeurIPS 33.
- Chen, T., Kornblith, S., Norouzi, M., Hinton, G. (2020). *A Simple Framework for Contrastive Learning of Visual Representations.* ICML.
- He, K., Fan, H., Wu, Y., Xie, S., Girshick, R. (2020). *Momentum Contrast for Unsupervised Visual Representation Learning.* CVPR.
- Tian, Y., Krishnan, D., Isola, P. (2020). *Contrastive Representation Distillation.* ICLR.
- Grill, J.-B., et al. (2020). *Bootstrap Your Own Latent.* NeurIPS 33.
- Dasgupta, S., Gupta, A. (2003). *An elementary proof of a theorem of Johnson and Lindenstrauss.* Random Structures & Algorithms, 22(1), 60–65.
- Li, A., Durrant, A., Markovic, M., Huang, T., Kundu, S., Chen, T., Yin, L., Leontidis, G. (2024). *Pushing the Limits of Sparsity: A Bag of Tricks for Extreme Pruning.* arXiv:2411.13545.
- Chen, Y., Dai, X., Liu, M., Chen, D., Yuan, L., Liu, Z. (2020). *Dynamic ReLU.* ECCV.
- Evci, U., Gale, T., Menick, J., Castro, P. S., Elsen, E. (2020). *Rigging the Lottery: Making All Tickets Winners.* ICML.
- Zhu, M., Gupta, S. (2017). *To prune, or not to prune: exploring the efficacy of pruning for model compression.* arXiv:1710.01878.
