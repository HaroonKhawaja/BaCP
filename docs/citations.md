# Citation dossier

Every implementation decision in this repository, paired with the primary source
that supports it — and, where the obvious source does **not** support it, an
explicit note saying so.

The organising rule: **a wrong citation is worse than none.** A reviewer who
checks one appendix and finds it says the opposite of what was claimed will
discount the rest of the paper. Three such cases were found and are flagged
below; in each, the fix is to narrow the claim, not to find a friendlier source.

Status legend: **✓** verified against the primary source · **⚠** mis-citation
risk, read the note · **◻** design choice with no canonical source · **…**
research still in progress.

---

## 1. WANDA pruning — `pruning_factory.WandaPruner`

**✓ Cite for the metric.**
Mingjie Sun, Zhuang Liu, Anna Bair, J. Zico Kolter. *A Simple and Effective
Pruning Approach for Large Language Models.* ICLR 2024, Vienna.
arXiv:2306.11695. Code: <https://github.com/locuslab/wanda>.

The importance score is $S_{ij} = |W_{ij}| \cdot \lVert X_j \rVert_2$, with
$\lVert X_j \rVert_2$ aggregated over a calibration set of 128 sequences sampled
from C4. Verified from the camera-ready PDF; the header reads "Published as a
conference paper at ICLR 2024". Do **not** append a presentation tier
(poster/spotlight/oral) — it could not be established.

### ⚠ Do **not** cite WANDA for per-output-row ranking in a vision setting

WANDA's headline configuration ranks weights within each output row, and the
paper argues that comparison group is crucial. **For LLMs.** Its own Appendix A
runs the identical comparison on image classifiers and reports the opposite:

> "we do not observe similar trend in image classification models, suggesting
> that our observations regarding pruning per output might be unique to LLMs"

with layer-wise *slightly better* than per-output for both ConvNeXt-B and
DeiT-B. That text survives into the ICLR 2024 camera-ready, so it is not a
preprint artefact.

This project prunes ResNet/VGG on CIFAR. There is therefore **no WANDA evidence
for per-output ranking in our setting in either direction**, and the only vision
evidence that exists points the other way. Two further scope limits make the
flag stronger rather than weaker: WANDA's vision experiments use 4096
calibration images at 70–80% sparsity, nowhere near 99%+; and even the
"the metric transfers" concession is demonstrated only in that range.

**Recommended wording:** *"WANDA reports the per-output comparison group as
crucial for LLMs but explicitly finds it does not transfer to image classifiers;
we adopt per-output ranking as a design choice, not on WANDA's authority."*

The code makes this a recorded knob (`--wanda_group {output,layer}`) rather than
a buried default, so the paper can report which was used.

### ◻ The Conv2d convention is ours

WANDA's reference implementation handles `nn.Linear` only — `WrappedGPT`'s
reshape branch is gated on `isinstance(layer, nn.Linear)`. There is no published
convention for 4-D weights. We fold the kernel into the input dimension so a
weight of shape `(out, in, kh, kw)` is scored against `F.unfold`'d input columns,
producing the same matrix WANDA scores for a Linear layer. State this in the
paper; do not imply it came from the source.

---

## 2. ERK layer-wise sparsity — `BasePruner.calculate_erk_densities`

**✓** Original Erdős–Rényi: Decebal Constantin Mocanu, Elena Mocanu, Peter
Stone, Phuong H. Nguyen, Madeleine Gibescu, Antonio Liotta. *Scalable training
of artificial neural networks with adaptive sparse connectivity inspired by
network science.* Nature Communications 9, 2383 (2018).

**✓** The kernel extension (ERK): Utku Evci, Trevor Gale, Jacob Menick, Pablo
Samuel Castro, Erich Elsen. *Rigging the Lottery: Making All Tickets Winners.*
ICML 2020. arXiv:1911.11134. Code: <https://github.com/google-research/rigl>.

RigL and EAST both use ERK by default, which is why the ladder's main path does.

---

## 3. RigL drop-and-regrow — zero-initialised regrowth

**✓** Evci et al. 2020, as above. Regrown connections are initialised to
**exactly zero**, and the paper's stated reason is that zero-valued new weights
do not perturb the current function.

This is why `results.measure_sparsity` records mask sparsity **and** value
sparsity separately: under dynamic sparse training a structurally active weight
can read as numerically zero, so the value-based measure *overstates* sparsity —
measured at 0.9277 against a true 0.8976. Overstating is the flattering
direction, which is exactly why it needs saying.

---

## 4. Whether the classifier is pruned — the sparsity denominator

**✓ This contradicts the common assumption, and it changes the denominator by a
large factor at 99.9%+.** Both reference codebases prune the final classifier by
default:

- **RigL** keeps bias and batch-norm parameters dense under all distributions,
  and keeps the *first* layer dense **only under the Uniform distribution** —
  bullets 2 (ER) and 3 (ERK) carry no such carve-out. It assigns ResNet-50's
  `fc1000` an ERK sparsity of **0.957**, and `resnet_model.py` defaults
  `prune_last_layer=True`. Cross-checked numerically: ER density for 2048→1000
  is $(2048+1000)/(2048 \times 1000) = 0.001488$; times $\varepsilon \approx 28.9$ gives
  density 0.043, i.e. sparsity 0.957 — reproducing the figure to three
  decimals, so it is not an extraction artefact.
- **GraNet** registers *every* 2-D and 4-D parameter tensor into the mask set
  with no name-based classifier exclusion; the only optional exclusion is the
  first conv, behind `--rm_first`.

**Consequence for the ladder:** rung A5 (`+ RigL`) now inherits A3 — ERK with the
head **pruned** — rather than the head-dense corner. Since Stage C is built on
that substrate, the earlier arrangement would have made every downstream number
incomparable to the published RigL/EAST figures the paper cites.

The head-dense arms (A2, A4) remain as corners of the 2×2, measuring what the
alternative convention buys, rather than being adopted silently.

### ◻ One deliberate divergence

RigL's *Uniform* distribution keeps the first layer dense; our `uniform`
allocation does not. Baking that in would put a third difference inside the
uniform arms and destroy the 2×2. RigL applies the carve-out only to Uniform, not
to ERK, so rungs A3–A6 — the comparable ones — are unaffected.

---

## 5. Knowledge distillation on logits — `loss_functions.kd_kl_loss`

**✓ Cite for the soft-target idea and the $T^2$ rescaling.**
Geoffrey Hinton, Oriol Vinyals, Jeff Dean. *Distilling the Knowledge in a Neural
Network.* NIPS 2014 Deep Learning Workshop. arXiv:1503.02531.

The $T^2$ justification is at the end of Section 2: soft-target gradients scale
as $1/T^2$, so multiplying by $T^2$ keeps the relative contribution of soft and
hard targets roughly unchanged as $T$ varies.

### ⚠ Two narrowings

**(a) $1/T^2$ is a high-temperature approximation, not an identity.** The exact
gradient is $\partial C/\partial z_i = (1/T)(q_i - p_i)$ (Eq. 2); the $1/T^2$
form follows only under high temperature relative to logit magnitude (Eq. 3) plus
per-case zero-meaned logits (Eq. 4). Writing "gradients scale as $1/T^2$ [Hinton
et al. 2015]" is faithful — the unqualified statement appears in Section 2 prose.
Writing "*exactly* $1/T^2$", "at all temperatures", or presenting $T^2$ as an
exact normaliser rather than a balancing heuristic overstates the source.

**(b) Hinton et al. never write the objective as a KL divergence.** Section 2
defines it purely as cross-entropy with soft targets. "Kullback" and "relative
entropy" appear zero times; "KL" appears only in Section 5.4, on ensembles of
specialists. **Cite Hinton for $T^2$ and soft targets; cite implementations, not
Hinton, for the KL form and its direction.**

**✓ Direction and reduction, established at implementation level.**
`F.kl_div(log_softmax(student/T), softmax(teacher/T))` computes
KL(teacher ‖ student) — forward KL, teacher as reference — and is the accepted
convention. It is gradient-identical to Hinton's formulation because
$H(p_t, q_s) = \mathrm{KL}(p_t \Vert q_s) + H(p_t)$ and $H(p_t)$ is constant in
the student's parameters.

`reduction='batchmean'` matters and should be stated. PyTorch's **default** is
`'mean'`, which divides by `numel` (batch × classes) and which PyTorch's own
documentation notes does not return the true KL. The two differ by exactly the
class count — 10 on CIFAR-10, 100 on CIFAR-100 — and that factor composes
multiplicatively with $T^2$ and with `lambda_kd`, silently detuning the KD/CE
balance across datasets. RepDistiller, mmrazor, the PyTorch tutorial and
HuggingFace all use `batchmean` or hand-roll it.

---

## 6. Feature distillation — `loss_functions.feature_distill_loss`

**✓ FitNets, cited accurately.** Adriana Romero, Nicolas Ballas, Samira Ebrahimi
Kahou, Antoine Chassang, Carlo Gatta, Yoshua Bengio. *FitNets: Hints for Thin
Deep Nets.* ICLR 2015. arXiv:1412.6550. It uses a squared Euclidean **hint** loss
on **intermediate** activations through a **learned regressor**; its second stage
operates on softened logits.

### ⚠ FitNets does **not** support cosine feature matching

It contains no cosine similarity and no discussion of teacher/student
feature-norm mismatch, and neither of its stages operates on penultimate
features. It is not a valid source for either the metric or the justification.

The usual alternatives do not carry it either:

| Candidate | What it actually does |
|---|---|
| SP (Tung & Mori, ICCV 2019) | squared Frobenius norm on $b \times b$ **row-wise** similarity matrices — *relational*, not student-feature-to-teacher-feature |
| RKD (Park et al., CVPR 2019) | distance and angle **between samples** — also relational |
| PKT (Passalis & Tefas) | builds a cosine-**kernel** probability distribution, matched by KL |
| CRD (Tian et al., ICLR 2020) | normalises embeddings for its NCE critic, but argues from mutual information, not norm mismatch |

### ◻ Cosine-on-penultimate is our design choice

There is **no verified primary source** for "cosine feature matching on
penultimate features *because* student and teacher norms differ substantially".
State it as this paper's justification. One candidate surfaced during the search
had all three of its supporting claims refuted under adversarial vote and must
not be cited on the strength of this dossier.

### ◻ And the two modes are not independent here

Every branch of `get_embeddings` ends in `F.normalize`, so both arguments
reaching `feature_distill_loss` are unit-norm — and on unit-norm vectors
$\lVert z_s - z_t \rVert^2 = 2(1 - \cos)$ exactly. The MSE branch is therefore
$(2/D)$ times the cosine branch: a monotone rescaling that changes only the
effective `lambda_kd`. **Do not present a cosine/MSE ablation as evidence about
norm versus direction** — norm is a constant on both sides. Pinned by
`test_mse_and_cosine_are_the_same_metric_on_normalised_features`.

---

## 7. Cubic sparsity schedule — `BasePruner.step`

**✓** Michael Zhu, Suyog Gupta. *To prune, or not to prune: exploring the efficacy
of pruning for model compression.* ICLR 2018 Workshop track. arXiv:1710.01878.

The schedule is defined over **training steps**, not epochs. Our implementation
applies it epoch-wise for the monotone pruners and step-wise for the DST ones;
that is a deviation and should be stated rather than assumed equivalent.

---

## 8. Contrastive vs feature-regression distillation — the C2 → C3 rung

### ✓ CRD, cited for what it actually shows

Yonglong Tian, Dilip Krishnan, Phillip Isola. *Contrastive Representation
Distillation.* ICLR 2020. arXiv:1910.10699 — **cite v3**; the v3 comment reads
"Typo fixed in the newest version". Code:
<https://github.com/HobbitLong/RepDistiller>.

The strongest published evidence that a contrastive objective beats hint-style
regression is CRD's **Table 2** (cross-architecture pairs, 3 runs). On
vgg13 → MobileNetV2 the regression-style methods fall *below the vanilla student*
(64.6): **AT 59.40, NST 58.16** — while **CRD reaches 69.73**.

⚠ **State the caveat.** Tables 1 and 2 are a cross-method benchmark, not a
controlled ablation: the methods differ in *which layers they act on*, not only
in objective form. That is exactly the confound the C1/C2/C3 rungs remove by
holding the teacher and the injection point fixed — which is the argument for
running them rather than citing CRD and stopping.

### ⚠ CRD does **not** attribute its gain to the negatives

Its only ablation (Table 6) varies two things: the negative **sampling policy**
and the **objective form**. The sampling policy is the larger effect and the only
quantified one — **+0.81%** for their objective, **+0.62%** for InfoNCE. The
objective form is nearly a wash: their bound wins 4 of 5 pairs, every gap ≤ 0.33
points, and it *loses* on resnet110/resnet32 (73.48 vs 73.53).

So CRD's own evidence attributes the gain to *which negatives you draw*, not to
the particular MI bound. The mutual-information story is motivation (§1, §3.1)
and is never isolated by an experiment. **L2 normalisation is stated once as an
implementation detail and is never ablated** — do not claim CRD attributes
anything to it.

**Missing and uncitable:** the number-of-negatives sweep runs
N ∈ {16, 64, 256, 1024, 4096, 16384}, but the only published numeric fact is
"the difference of error rate between N=4096 and N=16384 is less than 0.1%".
Figure 5(a) is a plot; **no per-N accuracy values exist** anywhere in the paper
or its appendix. They cannot be quoted.

### ◻ "Cosine matching is InfoNCE without the denominator" — nobody has published this

This is the framing the C2 → C3 rung rests on, so it matters that it is ours.

**✓ Cite Wang & Isola for the decomposition and the identity.** Tongzhou Wang,
Phillip Isola. *Understanding Contrastive Representation Learning through
Alignment and Uniformity on the Hypersphere.* ICML 2020. arXiv:2005.10242.
Theorem 1 decomposes the limiting normalised InfoNCE into an **alignment** term
and a **uniformity** term; the encoder is defined onto the sphere, so the
alignment loss is a distance on L2-normalised features; and their appendix says
in these words that it *is* cosine "up to a constant and a scaling", via the
identity ‖u − v‖² = 2 − 2·uᵀv for unit vectors.

⚠ **Their theorem is asymptotic** (batch size → ∞, with O(M^−1/2) deviation), so
it does not license the word "exactly" at finite batch size — and the paper never
mentions distillation, teachers, students, hint losses or pruning. It is a pure
self-supervised analysis.

**✓ The closest published statement is BYOL's, not Wang & Isola's.** Grill et al.
*Bootstrap Your Own Latent.* NeurIPS 2020. arXiv:2006.07733. BYOL's Eq. 6 writes
an InfoNCE whose log-sum-exp denominator is weighted by β, defines the similarity
as a normalised dot product (Eq. 7 — cosine), and states: **"We recover the BYOL
loss … with β = 0"** — while BYOL's own Eq. 2 is 2 − 2·cos. That *is* "the
positive-pair term of InfoNCE with the denominator removed", asserted by the
authors, at finite batch size.

**✗ SimSiam does not support this.** The string "InfoNCE" appears **zero times**
in the SimSiam paper; it frames itself only as "SimCLR without negative pairs" —
an architectural comparison, never an objective decomposition.

**The single most useful citable number**, and it is a *warning* as much as
support: BYOL Table 5(b), bottom row — β = 0 with no predictor and no target
network gives **0.1% ImageNet top-1**, against SimCLR's **69.4%** at β = 1.
Published, controlled, same architecture, and precisely "what happens when you
delete the denominator".

⚠ **Do not over-read it.** BYOL's collapse is between a *trainable* online/target
pair with no predictor. Rung C2 regresses onto a **frozen** teacher, which cannot
drift, so the collapse mechanism does not transfer directly. Cite it as evidence
that the denominator does real work — not as a prediction for what C2 will do.

**Recommendation:** present the equivalence as a **one-line derivation of our
own**; cite Wang & Isola for the decomposition and the identity, and BYOL Eq. 6–7
for the β = 0 parameterisation. Do not write "as shown by [Wang & Isola]" for the
distillation reading.

---

## 9. EAST — the external baselines

**✓** Andy Li, Aiden Durrant, Milan Markovic, Tianjin Huang, Souvik Kundu,
Tianlong Chen, Lu Yin, Georgios Leontidis. *Pushing the Limits of Sparsity: A Bag
of Tricks for Extreme Pruning.* **TMLR, volume 2025.** arXiv:2411.13545.

⚠ **Your `.bib` may carry the wrong author list.** arXiv **v1 and v2 have only
five authors** (Li, Durrant, Markovic, Yin, Leontidis); v3 onward has all eight.
DBLP's CoRR record `journals/corr/abs-2411-13545` is **still frozen at the
five-author v1**, so a citation auto-generated from it is wrong. Use the TMLR
record `journals/tmlr/LiDMHKCYL25`.

⚠ **Cite v4 for the numbers.** v1/v3 report ResNet-34 / CIFAR-10 as
93.03 / 86.76 / 83.83 / 70.57 with **no standard deviations**. v4 reports
93.51 / 86.99 / 83.83 / 62.12 with them — the 99.99% cell moved by **8.45
points** between versions.

### ResNet-34 / CIFAR-10, top-1, v4 Table 1 (3 runs)

| Sparsity | RigL | EAST | SET | SynFlow |
|---|---|---|---|---|
| 99% | 92.92 ± 0.18 | 93.51 ± 0.13 | 93.09 ± 0.15 | 86.03 ± 0.71 |
| **99.9%** | **85.71 ± 0.23** | **86.99 ± 0.32** | 82.70 ± 0.91 | 61.61 ± 10.76 |
| 99.95% | 81.47 ± 0.32 | 83.83 ± 0.02 | 70.84 ± 1.51 | 56.33 ± 3.44 |
| 99.99% | 10.03 ± 0.19 | 62.12 ± 0.90 | 10.00 ± 0.00 | 10.00 ± 0.00 |

The pair the ladder anchors on — RigL 85.71 ± 0.23 vs EAST 86.99 ± 0.32 — is
**confirmed, and it is the 99.9% level.**

Note that RigL **collapses to random** at 99.99% (10.03) while EAST holds 62.12.
That column is a mixture, not a Gaussian, which is why the plan reports median,
IQR and collapse rate there rather than mean ± std.

### Protocol (v4, Table B.1 and §4 "Experiments")

250 epochs · batch 128 · SGD momentum 0.9 · LR 0.1 · **weight decay 1e-4** ·
ERK · mask-update interval **4000** for RigL (1500 for SET) · single A100 ·
**3 runs** behind every ±.

⚠ **EAST contradicts itself on the LR schedule.** Appendix Table B.1 says
`10×[75,150]`; the main text says ×10 at halfway and three-quarters, i.e. epochs
125 and 187. The paper does not say which produced Table 1.

⚠ **Two things the paper does not state**, and which must not be asserted on its
authority: whether training starts **from scratch or from ImageNet**, and whether
the **final classifier is in the pruned set** — "classifier", "last layer",
"final layer" and "output layer" have **zero occurrences** in v4. Both have to be
read off the ES2/ITOP code.

**Two fidelity gaps this closed in our code.** `weight_decay` was hardcoded at
GraNet's 5e-4 — now a recorded field, set to EAST's 1e-4 for the anchor. And
`delta_T` defaulted to 100, updating masks **40× more often** than the baseline
being reproduced: a different algorithm, not a different setting.

---

## 10. CAP — the antecedent

Ruiqi Xu, Fuli Luo, Chengyu Wang, Baobao Chang, Jun Huang, Songfang Huang, Fei
Huang. *From Dense to Sparse: Contrastive Pruning for Better Pre-trained Language
Model Compression.* AAAI 2022, vol. 36, pp. 11547–11555. arXiv:2112.07198.

The point the paper most needs to state plainly: **CAP reports on pre-trained
language models only.** BaCP is CAP applied to vision, and Related Work must say
so rather than let a reviewer discover it — this was reviewer objection B7, and
honest framing is stronger here than the alternative.

---

## 11. SupCon $L_{out}$ vs $L_{in}$, and NT-Xent

Prannay Khosla et al. *Supervised Contrastive Learning.* NeurIPS 2020.
arXiv:2004.11362 — the $L_{out}$ (sum over positives **outside** the log) vs $L_{in}$
(inside) distinction, with $L_{out}$ the superior formulation. Ting Chen, Simon
Kornblith, Mohammad Norouzi, Geoffrey Hinton. *A Simple Framework for Contrastive
Learning of Visual Representations* (SimCLR). ICML 2020. arXiv:2002.05709 —
NT-Xent.

◻ **No published analysis exists** of what summing two contrastive losses over a
shared denominator does to the fixed point. That derivation is ours;
`docs/math.md` carries it, along with the measured crossover that confirms it.

---

## 12. Experimental hygiene

| Claim the methodology section needs | Source |
|---|---|
| Pruning papers are not mutually comparable: protocol drift, missing baselines | Blalock, Gonzalez Ortiz, Frankle, Guttag. *What is the State of Neural Network Pruning?* MLSys 2020. arXiv:2003.03033 |
| Seed variance in DL benchmarks; one seed is insufficient | Bouthillier et al. *Accounting for Variance in Machine Learning Benchmarks.* MLSys 2021. arXiv:2103.03098 |
| Measured seed spread on CIFAR-10 | Picard. *torch.manual_seed(3407) is all you need.* arXiv:2109.08203 |
| Max over a noisy sequence is upward-biased; report expected max vs budget | Dodge, Gururangan, Card, Schwartz, Smith. *Show Your Work.* EMNLP 2019. arXiv:1909.03004 |
| TOST equivalence testing, so a null result is publishable | Schuirmann 1987; Lakens 2017, *Social Psychological and Personality Science* |
| Fixed-sequence testing at full α with no correction | Westfall & Krishen 2001; Maurer, Hothorn & Lehmacher 1995 |

⚠ Dodge et al. concerns the maximum over a **hyperparameter search**. We need the
best-epoch-of-a-single-run case, which is adjacent but not identical — make the
argument in our own terms and cite Dodge for the estimator, not for our case.

---

## 13. SynFlow and layer collapse

Hidenori Tanaka, Daniel Kunin, Daniel L. K. Yamins, Surya Ganguli. *Pruning
neural networks without any data by iteratively conserving synaptic flow.*
NeurIPS 2020. arXiv:2006.05467 — layer collapse, the maximal-critical-compression
axiom, and the argument that iterative pruning avoids collapse where single-shot
does not. Relevant to any decision about protecting layers at 99.9%+.

---

## 14. Published dense baselines — `results/reference/dense_reference.json`

**✓** GraNet: Shiwei Liu, Tianlong Chen, Xiaohan Chen, Zahra Atashgahi, Lu Yin,
Huanyu Kou, Li Shen, Mykola Pechenizkiy, Zhangyang Wang, Decebal Constantin
Mocanu. *Sparse Training via Boosting Pruning Plasticity with Neuroregeneration.*
NeurIPS 2021. arXiv:2106.10404.

| Model / dataset | Dense top-1 |
|---|---|
| ResNet-50 / CIFAR-10 | 94.75 ± 0.01 |
| ResNet-50 / CIFAR-100 | 78.23 ± 0.18 |
| VGG-19 / CIFAR-10 | 93.85 ± 0.05 |
| VGG-19 / CIFAR-100 | 73.43 ± 0.08 |

Recipe: 160 epochs · batch 128 · SGD(0.9) LR 0.1 · ×10 at [80, 120] · wd 5e-4.

These are printed **identically in four separate GraNet tables** (1, 2, 6 and 9),
and all 24 shared sparse cells agree with GraSP's Table 2 — so the transcription
is four-way internally consistent and externally cross-checked.

⚠ GraSP reports **94.23 / 74.16** for VGG-19, disagreeing with GraNet, and gives
no standard deviations. GraSP's ResNet-32 rows are **2×-width** and must not be
used as a ResNet-50 baseline.

⚠ GraNet's own CIFAR pipelines are internally inconsistent: CIFAR-10 pads with
**reflect** before cropping, CIFAR-100 uses zero padding — and GraSP uses zero
padding for both. It also trains CIFAR-10 on 45,000 images (a 0.1 validation
split) and uses Nesterov momentum, neither of which is stated in the paper. If
our dense numbers land below parity, check these before concluding the recipe is
wrong.

### ⚠ There is no published dense ResNet-34 / CIFAR-10 baseline

That is the **ladder's anchor cell**. GraNet covers ResNet-20/50 and VGG-19; EAST
reports no dense row at all. So the anchor's dense parity **cannot be checked
against the literature** — state it as a limitation rather than borrowing a figure
from a different architecture. `dense_reference.json` records this explicitly as
`"source": "NOT FOUND"`, so the parity gate skips that cell instead of silently
passing it.

---

## Summary of mis-citation flags

| # | The tempting citation | Why it fails | What to do |
|---|---|---|---|
| 1 | WANDA → per-output ranking on CIFAR | Its Appendix A finds the opposite for image classifiers, and says the effect "might be unique to LLMs" | Cite for the metric only; present the grouping as a design choice |
| 2 | Hinton → the KL form and its direction | The paper never writes a KL in that objective; it is cross-entropy with soft targets | Cite for T² and soft targets; cite implementations for the form |
| 3 | FitNets → cosine penultimate matching | Squared Euclidean, intermediate layers, learned regressor, no norm discussion | State the metric and its rationale as our own |
| 4 | CRD → "the negatives cause the gain" | CRD's own ablation attributes it to the sampling policy; the objective form is within 0.33 points and loses one pair | Cite the benchmark tables, not a mechanism |
| 5 | Wang & Isola → "cosine is InfoNCE without the denominator" | Asymptotic, self-supervised, never mentions distillation | Cite for the decomposition; derive the rest ourselves; BYOL Eq. 6–7 is closer |
| 6 | EAST via the DBLP CoRR record | That record is frozen at the five-author v1 | Use the TMLR record; cite v4 for the numbers |
| 7 | Any dense ResNet-34 / CIFAR-10 figure | None is published | State the limitation |
