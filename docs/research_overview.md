# BaCP - Research Overview

## Status

- **Target:** NeurIPS 2026 workshop deadline, **29 August 2026**.
- **All prior ResNet-34 numbers are VOID.** Every figure in the AAAI submission predates 28 bug fixes and was produced under, among others, the `drop_last=True` eval defect (§2.7) and an EAST pruner whose masks never updated (§7.10).
- **The only completed measurement is the dense anchor `A0`: 91.64 ± 0.30 top-1**, ResNet-34 / CIFAR-10, 5 seeds, from scratch, zero NaN (per-seed 91.6535, 91.5447, 91.5843, 91.2876, 92.1064). Recomputed from those five values: mean **91.6353**, sample sd ($ddof=1$) **0.2975**, population sd 0.2661, range 0.8188, 95% CI half-width 0.369. **A figure of `91.68 ± 0.44` circulated earlier and does not reconcile with the per-seed data; it is superseded throughout this document.** The measurement has **no published comparator** (§6.5) and it was obtained with the defective stem (§1.3.3), so it is superseded as a final number in any case.
- **Sparse training currently diverges to NaN at 0.999 and is under investigation.** 5 of 5 `C-null` seeds and 2 of 3 completed `C0` seeds collapse to exactly 10.0969% (chance on CIFAR-10). **The cause is not established.** See §8.1.
- The first tier-1 attempt was stopped and its records archived to `/workspace/ARCHIVE_results_nan_20260819`. **Nothing has been published.**

Labels used throughout: **MEASURED** = computed this session from a run or from constructing the module; **PLANNED** = declared in `project/experiments/manifest.py` and not run; **PUBLISHED** = appears in an external paper or the AAAI submission. Where a quantity is unknown it is written *not measured* or *not stated in the source*. Nothing is inferred silently.

---

## Table of contents

- [1. Models](#1-models)
  - [1.1 The registry](#11-the-registry)
  - [1.2 Defined but not registered](#12-defined-but-not-registered-dead-code)
  - [1.3 The anchor model: resnet34](#13-the-anchor-model-resnet34)
  - [1.4 Projection-head arithmetic](#14-projection-head-arithmetic-why-proj_modetied_frozen-is-a-fairness-requirement)
  - [1.5 DyReLU and the phasing adapter](#15-dyrelu-and-the-phasing-adapter)
  - [1.6 Weight sharing](#16-weight-sharing)
  - [1.7 The other registered backbones](#17-the-other-registered-backbones)
- [2. Datasets](#2-datasets)
  - [2.1 The anchor cell, data side](#21-the-anchor-cell-data-side)
  - [2.2 Registered datasets](#22-registered-datasets)
  - [2.3 Dataset construction: signature introspection](#23-dataset-construction-signature-introspection)
  - [2.4 Train / validation / test split](#24-train--validation--test-split)
  - [2.5 Augmentation](#25-augmentation)
  - [2.6 Eval transform](#26-eval-transform)
  - [2.7 Loader construction](#27-loader-construction)
  - [2.8 Tier-0 truncation](#28-tier-0-truncation-_limitedloader)
  - [2.9 Text pipeline](#29-text-pipeline)
- [3. Test parameters](#3-test-parameters)
  - [3.1 The anchor cell](#31-the-anchor-cell)
  - [3.2 The east250 protocol](#32-the-east250-protocol)
  - [3.3 The smoke protocol](#33-the-smoke-protocol-tier-0)
  - [3.4 Sparsity levels](#34-sparsity-levels-two-sets-for-two-questions)
  - [3.5 Surviving weights at each sparsity](#35-surviving-weights-at-each-sparsity)
  - [3.6 Primary endpoint](#36-primary-endpoint)
  - [3.7 Seeds, tiers, and cell-key scoping](#37-seeds-tiers-and-cell-key-scoping)
- [4. Hardware](#4-hardware)
  - [4.1 The machine](#41-the-machine)
  - [4.2 Scheduling](#42-scheduling)
  - [4.3 Measured throughput](#43-measured-throughput)
  - [4.4 Budget](#44-budget)
  - [4.5 Every timing above is stale](#45-every-timing-in-43-and-44-is-now-stale)
- [5. Experiments](#5-experiments)
  - [5.1 The design principle: one rung, one change](#51-the-design-principle-one-rung-one-change)
  - [5.2 Stage A - the sparse substrate](#52-stage-a--the-sparse-substrate)
  - [5.3 Stage B - EAST's architecture knobs](#53-stage-b--easts-architecture-knobs)
  - [5.4 Stage C - the scientific heart](#54-stage-c--the-scientific-heart)
  - [5.5 Stage D - BaCP's teachers, forward](#55-stage-d--bacps-teachers-forward)
  - [5.6 Stage D4 - backward leave-one-out](#56-stage-d4--backward-leave-one-out-with-renormalisation)
  - [5.7 Forward vs backward](#57-forward-sufficiency-vs-backward-necessity)
  - [5.8 Tiers and cells](#58-tiers-and-cells)
  - [5.9 The gates](#59-the-gates)
  - [5.10 Statistical design](#510-statistical-design)
  - [5.11 Experiment-level invariants](#511-experiment-level-invariants)
  - [5.12 Execution](#512-execution)
- [6. Citations](#6-citations)
  - [6.1 How to read this section](#61-how-to-read-this-section)
  - [6.2 Complete reference table](#62-complete-reference-table)
  - [6.3 The mis-citation flags](#63-the-mis-citation-flags--the-part-that-matters)
  - [6.4 The EAST citation hazard](#64-the-east-citation-hazard-in-full)
  - [6.5 Published baseline tables](#65-published-baseline-tables)
  - [6.6 The sparsity-denominator citation](#66-the-sparsity-denominator-citation)
  - [6.7 CAP: the direct antecedent](#67-cap-the-direct-antecedent-and-the-honest-framing)
- [7. Mathematical formulas](#7-mathematical-formulas)
- [8. Open defects](#8-open-defects)

---

# 1. Models

## 1.1 The registry

Models are declared as `ModelSpec` records (`project/model_factory.py:25-32`) rather than bare constructors, because the project's own vision factories take `dyrelu_en` / `dyrelu_phasing_en` and no HuggingFace constructor accepts those arguments:

| field | meaning |
|---|---|
| `builder` | `callable(num_classes, dyrelu_en, dyrelu_phasing_en) -> nn.Module` |
| `weight` | local checkpoint path; `''` when the builder arrives pretrained |
| `family` | `'resnet' \| 'vgg' \| 'vit' \| 'bert'` |
| `type` | `'cv' \| 'llm'` (matches `--model_type`) |
| `task` | `'classification' \| 'mlm'` |

Four builder factories exist: `_local_vision` (`model_factory.py:34-42`), `_hf_image` (`:45-52`), `_hf_seqcls` (`:55-62`), `_hf_mlm` (`:65-70`). All three HF factories call `_reject_dyrelu` (`:73-86`), which raises `ValueError` rather than silently training a plain-ReLU network while the logs claim phasing is active.

**Architecture references.** Until this revision the reference table (§6.2) contained no entry for any backbone the project builds. They are now R30 (ResNet), R31 (VGG), R32 (WideResNet), R33 (ViT), R34 (BERT), R35 (DistilBERT), R36 (RoBERTa) - **all `UNVERIFIED`**, because none appears in `docs/citations.md` and none has been checked against a primary source by this project (§6.2).

`_local_vision` **discards `num_classes`** (`model_factory.py:40-41` calls `fn(dyrelu_en=…, dyrelu_phasing_en=…)` only), so every local backbone is built with its default 1000-way head, which is then replaced by `nn.Identity()` in `remove_last_layer` and superseded by a fresh `cls_head`. The 1000-way head exists transiently at construction and contributes nothing to the trained model.

### The `MODELS` table (`model_factory.py:89-110`) - all 11 entries are REGISTERED and CLI-reachable

| key | builder | family | type | task | weight path | DyReLU |
|---|---|---|---|---|---|---|
| `resnet34` | `_local_vision(resnet34)` | resnet | cv | classification | `/dbfs/research/bacp/resnet34_imagenet.pth` | yes |
| `resnet50` | `_local_vision(resnet50)` | resnet | cv | classification | `/dbfs/research/bacp/resnet50_imagenet.pth` | yes |
| `resnet101` | `_local_vision(resnet101)` | resnet | cv | classification | `/dbfs/research/bacp/resnet101_imagenet.pth` | yes |
| `vgg11` | `_local_vision(vgg11)` | vgg | cv | classification | `/dbfs/research/bacp/vgg11_imagenet.pth` | yes |
| `vgg19` | `_local_vision(vgg19)` | vgg | cv | classification | `/dbfs/research/bacp/vgg19_imagenet.pth` | yes |
| `vit-tiny` | `_hf_image('WinKawaks/vit-tiny-patch16-224')` | vit | cv | classification | `''` (hub) | **raises** |
| `vit-small` | `_hf_image('WinKawaks/vit-small-patch16-224')` | vit | cv | classification | `''` (hub) | **raises** |
| `distilbert-base-uncased` | `_hf_seqcls(...)` | bert | llm | classification | `''` (hub) | **raises** |
| `roberta-base` | `_hf_seqcls(...)` | bert | llm | classification | `''` (hub) | **raises** |
| `distilbert-base-uncased-mlm` | `_hf_mlm(...)` | bert | llm | **mlm** | `''` (hub) | **raises** |
| `roberta-base-mlm` | `_hf_mlm(...)` | bert | llm | **mlm** | `''` (hub) | **raises** |

The CLI `--model_name` choices list (`project/scripts/scripting_utils.py:26-35`) contains exactly these 11 keys and is pinned to the registry by `test_imports.py::test_cli_model_choices_are_all_registered` - the two had drifted before, with the CLI accepting `vgg11`/`vgg19` while both were unregistered (`scripting_utils.py:22-25`). `manifest.validate()` asserts `ANCHOR['model_name'] in MODELS` (`manifest.py:740-741`).

### Weight provenance is recorded, not assumed

`initialize_model_components` returns `init_source` describing what the weights **actually are** (`model_factory.py:164-169`): `'hf_hub'` when `spec.weight == ''`, `'imagenet_checkpoint'` only when `load_weights` returned truthy, otherwise `'random'`. This matters because `load_weights` fails soft - a missing checkpoint warns and the run continues - and **every** local `weight` path points into `/dbfs`, which does not exist on the rented Vast.ai box. On the current hardware all five local vision entries therefore resolve to `init_source='random'`.

⚠ **The source comment justifying that goes further than the dossier allows.** `model_factory.py:160-163` reads "RigL, GraNet and EAST all train CIFAR from SCRATCH, so 'random' is the setting that makes this project's numbers comparable to their published ones." For RigL (R3) and GraNet (R25) that is supported. **For EAST it is not:** `docs/citations.md` §9 records, among the two things EAST does not state and "which must not be asserted on its authority", "whether training starts **from scratch or from ImageNet**" - the same caveat this document repeats verbatim at §6.4(c). So the defensible statement is: *`random` matches RigL's and GraNet's published setting; EAST's initialisation is not stated in its paper and has to be read off the ES2/ITOP code before any comparison against EAST's Table 1 claims a matched setting.* The code comment should be narrowed to match.

## 1.2 Defined but NOT registered (dead code)

`model_factory.py:8-9` imports only `resnet34, resnet50, resnet101` and `vgg11, vgg19`. Everything else in the model files is unreachable from the CLI, the manifest and the runner. A repo-wide grep (excluding `.venv`) finds **zero** references to any of the following outside their own definition site.

| symbol | file:line | status |
|---|---|---|
| `resnet18` | `project/models/resnet.py:258` | unregistered; builds fine (MEASURED 11,689,512 params, 7×7 stem, 1000-way `fc`) |
| `resnet34_wide` | `project/models/resnet.py:273` | unregistered **and broken** - defect M3, §8.6 |
| `WideResNet22` / `wrrn22` | `project/models/resnet.py:368`, `:414` | unregistered; builds, but incompatible with the wrapper - defect M4, §8.6 |
| `vgg11_bn`, `vgg13`, `vgg13_bn`, `vgg16`, `vgg16_bn`, `vgg19_bn` | `project/models/vgg.py:209-275` | unregistered |

`wrrn22` also uses `n_groups` both as the number of groups and as the per-group block count (`models/resnet.py:381` passes `n_groups` into the `n_blocks` slot of `_wrrn22_make_group`), giving 3 groups × 3 blocks, with channels hardcoded `[channel_start, 96, 192, 384]` (`:374`). Which published WRN variant this corresponds to is **not stated in the source**; commit `3333f03` calls it "official WideResNet22" but the source contains no citation. Nor does this document supply one: the WRN family is R32 (§6.2), but R32 is `UNVERIFIED` and **nobody has checked that these hyperparameters reproduce any variant in it**, so citing R32 for `wrrn22` would assert exactly the correspondence that is open. The question stands in both places until someone reconciles the two - and since the entry is unregistered dead code carrying defect M4 (§8.6), deleting it is the cheaper resolution.

## 1.3 The anchor model: `resnet34`

`ANCHOR` (`project/experiments/manifest.py:57-63`) is ResNet-34 / CIFAR-10, `num_classes=10`, `image_size=32`, at `LADDER_SPARSITY = 0.999` (`manifest.py:65`) - EAST's own published setting (Li et al. 2024, arXiv 2411.13545), PUBLISHED comparators RigL 85.71 ± 0.23 and EAST 86.99 ± 0.32 (`manifest.py:46-49`).

Built by `resnet34()` (`models/resnet.py:265-271`) = `ResNet(BasicBlock, [3,4,6,3])` - the ResNet-34 of He et al. (**R30**, §6.2), with the CIFAR stem adaptation of §1.3.3, which is a convention rather than anything R30 specifies.

### 1.3.1 Layer-by-layer - MEASURED (CIFAR stem, `fc` replaced by `Identity`)

| component | definition | shape / blocks | params | spatial out @32×32, **current** | spatial out, **after stem fix** |
|---|---|---|---:|---|---|
| `conv1` | `model_factory.py:183` (replaces `resnet.py:162`) | `Conv2d(3,64,k=3,s=1,p=1)` | 1,728 | 32×32 | 32×32 |
| `bn1` | `resnet.py:163` | `BatchNorm2d(64)` | 128 | 32×32 | 32×32 |
| `relu_v1` | `resnet.py:167-172` | `ReLU` (or DyReLU) | 0 | 32×32 | 32×32 |
| `maxpool` | `resnet.py:174` | `MaxPool2d(k=3,s=2,p=1)` | 0 | **16×16** | **32×32** (→ `Identity`) |
| `layer1` | `resnet.py:175` | 3 × BasicBlock, 64ch, no downsample | 221,952 | 16×16 | 32×32 |
| `layer2` | `resnet.py:176` | 4 × BasicBlock, 128ch, downsample @block 0 | 1,116,416 | 8×8 | 16×16 |
| `layer3` | `resnet.py:177` | 6 × BasicBlock, 256ch, downsample @block 0 | 6,822,400 | 4×4 | 8×8 |
| `layer4` | `resnet.py:178` | 3 × BasicBlock, 512ch, downsample @block 0 | 13,114,368 | **2×2** | 4×4 |
| `avgpool` | `resnet.py:179` | `AdaptiveAvgPool2d((1,1))` | 0 | 1×1 | 1×1 |
| `fc` | `resnet.py:180` → `model_factory.py:221` | `Identity` | 0 | - | - |
| **backbone total** | | | **21,276,992** | | |

Each `BasicBlock` is two 3×3 convs with BN (`resnet.py:26-31`), pre-add ReLU on the first and post-add ReLU on the second (`:49-60`). Downsample is a 1×1 conv + BN, created only when stride ≠ 1 or channels change (`resnet.py:208-212`); MEASURED downsample tensors: `layer2.0` 8,192, `layer3.0` 32,768, `layer4.0` 131,072.

### 1.3.2 The parameter inventory and the prunable denominator - AUTHORITATIVE

**Every other section of this document defers to this table.** All figures MEASURED by constructing the module and counting `named_parameters()`.

| set | count |
|---|---:|
| ≥2-D conv weights in the backbone, across **36** tensors | **21,259,968** |
| 1-D backbone params (BN weight + bias) - never prunable | 17,024 |
| backbone total (`fc` = `Identity`) | 21,276,992 |
| `cls_head` `Linear(512,10)`: 5,120 W + 10 b | 5,130 |
| **total, as trained (backbone + `cls_head`)** | **21,282,122** |
| `encoder_head` `Linear(512,128)`: 65,536 W + 128 b (BaCP runs only) | 65,664 |
| **total with `encoder_head`** | **21,347,786** |

Check: $21{,}259{,}968 + 17{,}024 + 5{,}130 = 21{,}282{,}122$. Both totals match the parameter arithmetic recorded against real run records exactly.

**The prunable set depends on one flag.** `layer_check` (`project/pruning_factory.py:127-146`) excludes anything with `dim() <= 1`, anything frozen, and anything whose lowercased name contains a keyword from `excluded_keywords()` (`:118-125`) - `'hyperfunction'` (`:22`), `'relu'`, `'encoder_head'` (`:27`), `'embedding'` (`:34`), and the task-head set `'cls_head'`, `'vocab_projector'`, `'vocab_transform'`, `'lm_head'`, `'classifier'` (`:41-45`). The last two groups are gated by `set_prunable_scope` (`:57-95`), whose module defaults are `_PRUNE_TASK_HEAD = False`, `_PRUNE_EMBEDDINGS = False` (`:56-57`).

| scope | flag | prunable weights on the anchor | used by |
|---|---|---:|---|
| backbone only (heads kept dense) | `prune_task_head=False` (module default) | **21,259,968** | rungs A2, A4 |
| backbone + task head | `prune_task_head=True` | **21,265,088** ( = 21,259,968 + 5,120) | **rungs A1, A3, A5, A6 and therefore all of Stage B, C, D** |

The two figures that circulated separately - 21,259,968 and 21,265,088 - are **both correct, for different scopes**. The main path (`A3 → A5 = BEST_SUBSTRATE` → all of Stage C and D) runs `prune_task_head=True`, matching RigL and GraNet (§6.6), so **21,265,088 is the denominator behind every headline number**. The `cls_head` **bias** (10), the `encoder_head` (65,664) and all 17,024 BN parameters are outside every scope. The stated convention (`pruning_factory.py:38-40`, matching the paper's appendix D.4) is "heads kept dense" for the *default*; the ladder deliberately overrides it.

At $s = 0.999$ on the main path:

$$\big\lfloor 21{,}265{,}088 \times (1 - 0.999)\big\rfloor = \mathbf{21{,}265\ \text{surviving weights}}$$

and 21,259.97 ≈ 21,260 on the head-dense corners. The manifest's "~21,300" (`manifest.py:52`, `:527`) is a rounding of the former. Per-sparsity figures: §3.5. The global parameter budget is held **exactly** (MEASURED `eff_sparsity` 0.999000).

### 1.3.3 The stem - AUTHORITATIVE statement of defect M1

`adapt_resnet_for_small_images` (`model_factory.py:181-183`) is called from `BaseModelWrapper.__init__` whenever `family == 'resnet'` and `adapt` is true (`model_factory.py:264-265`), and `_create_base_model` sets `adapt = args.image_size <= 64` (`training_utils.py:32`). With `ANCHOR['image_size'] = 32`, **every CIFAR run already has a 3×3 stride-1 stem**:

```python
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
```

**But `self.maxpool` (`models/resnet.py:174`) is left in place and is still applied in the forward pass (`resnet.py:241`).** MEASURED spatial trace at 32×32 input:

```
current : conv1 32->32, maxpool 32->16, layer1 16, layer2 8, layer3 4, layer4 2
fixed   : conv1 32->32, maxpool 32->32, layer1 32, layer2 16, layer3 8, layer4 4
```

A correct CIFAR ResNet keeps `layer1@32 / layer2@16 / layer3@8 / layer4@4`. The current model processes `layer4` at **2×2**: the entire final stage - 13,114,368 parameters, 61.6% of the backbone - operates on four spatial positions.

Setting `maxpool = nn.Identity()`:

- changes **ZERO** parameters (MEASURED: 21,276,992 before and after) and **ZERO** `state_dict` keys, so all existing checkpoints remain loadable and every figure in §1.3.2, §3.5 and §7.7 is unaffected;
- multiplies MACs by **3.98** (0.2912 → 1.1594 GMAC at 32×32, MEASURED).

**DECISION TAKEN: switch to the true CIFAR stem (`maxpool → Identity`) and re-run.** The fix is **not yet in the tree**. Consequences: the 5-seed dense `A0` result (91.64 ± 0.30) is superseded, and every wall-clock and cost figure in §4.3-§4.4 is stale (§4.5).

The stem also interacts with the ERK allocator: at target 0.999 `conv1` receives density 0.015352, i.e. ≈26.5 of its 1,728 weights survive. Full arithmetic and the ERK-vs-ER fidelity question: §7.7 and flag F8 (§6.3).

### 1.3.4 The two heads

Both live on the **wrapper** (`ClassificationAndEncoderNetwork`), not on the inner `ResNet`.

| head | created at | size | MEASURED params | in the pruned set? |
|---|---|---|---:|---|
| `cls_head` = `make_classification_head(512, 10)` = `nn.Linear(512, 10)` | `model_factory.py:186-187`, attached `:301` | 512×10 + 10 | 5,130 | **only when `prune_task_head=True`** (the main path); `'cls_head'` is in `_EXCLUDE_TASK_HEAD` (`pruning_factory.py:41`) otherwise |
| `encoder_head` = `nn.Linear(512, 128)` | `model_factory.py:309-310`, only when `num_out_features is not None` | 512×128 + 128 | 65,664 | **never** - `'encoder_head'` is in `_EXCLUDE_PROJECTION` (`pruning_factory.py:27`) in every scope |

`num_out_features` defaults to `128` on `BaCPTrainingArguments` (`project/bacp.py:46`) and to `None` on `TrainingArguments` (`project/trainer.py:35`), and has no CLI flag - so **baseline and pruning runs have no `encoder_head` at all**, and only BaCP runs carry the extra 65,664 parameters.

For language models `cls_head` is `None` (`model_factory.py:302-307`): the pretrained sequence-classifier or vocabulary projection is kept, because replacing it would discard the pretrained LM head. Features come from `hidden_states[-1][:, 0]` - the `[CLS]`/`<s>` token, matching CAP (`model_factory.py:340-350`).

## 1.4 Projection-head arithmetic: why `proj_mode='tied_frozen'` is a fairness requirement

`get_embeddings` (`model_factory.py:312-338`) is the only path into the contrastive losses. Three modes exist, applied by `_apply_proj_mode` (`project/training_utils.py:50-113`):

| mode | behaviour |
|---|---|
| `'tied_frozen'` (default for BaCP, `manifest.py:532`) | ONE `encoder_head` module shared by student and both teachers, `requires_grad = False` on all its parameters (`training_utils.py:107-110`) |
| `'none'` | no head; `F.normalize` the pooled backbone features directly, as CAP does with `[CLS]` (`model_factory.py:335-336`) |
| `'current'` | legacy: every model keeps its own head, the student's trainable |

Non-contrastive runs are forced to `'none'` unconditionally (`training_utils.py:79-82`) - a guard added because `TrainingArguments` does not declare `proj_mode`, so the `'tied_frozen'` default previously reached baseline and pruning runs and raised, crashing every CLI invocation of `baseline_script.py` and `pruning_script.py` during `__post_init__`.

**The argument** (reasoning at `manifest.py:525-532`). The `encoder_head` weight matrix is **65,536** weights against **21,265** surviving weights at $s = 0.999$ - **308%** of the entire surviving network (per-sparsity table: §3.5). A *trainable* projection head would place 65,536 unpruned, unmeasured parameters on the gradient path of PrC/FiC/SnC, in one arm of a comparison and not the other. Because the teacher branch is frozen, the student could drive all three contrastive losses down by moving those 65,536 parameters alone, without ever touching the backbone - the thing being pruned and the thing the method claims to protect. Ordinary contrastive learning is immune because both branches share the encoder. Frozen and tied, the head contributes no trainable capacity, and because student and teachers are pushed through **the same fixed map** $g$, whatever $g$ distorts it distorts identically on both sides of every similarity the loss forms - so the comparison stays a comparison of backbones. **That is a property of tying, derived here, and it is not the Johnson-Lindenstrauss bound.** JL (R28) bounds the $O(1/\sqrt d)$ inner-product distortion of *a single* random projection; it says nothing about distortion cancelling across the two arms of a contrastive loss, and it is not a licence for the cancellation argument. R28's proper and narrower use in this document is §7.14's: quantifying why the never-trained `encoder_head` was **lossy rather than vacuous**. Cite it there and nowhere else.

Tying also fixes a second, independent defect by construction (`training_utils.py:56-63`): the teachers' `encoder_head`s were never trained or loaded - `encoder_head` is created in `ClassificationAndEncoderNetwork.__init__`, which runs *after* the `load_weights` call inside `BaseModelWrapper.__init__`, and the partial-load path explicitly filters `encoder_head` keys - so the student had been aligning to a **random** projection of the teacher's features.

## 1.5 DyReLU and the phasing adapter

### `DyReLUB` (`project/dyrelu_adapter.py:4-60`)

**Citation:** Chen et al., *Dynamic ReLU*, ECCV 2020 (**R29**) for the DyReLU-B form; EAST (**R14**) for the phasing schedule that wraps it. ⚠ **Both are `UNVERIFIED` for these uses** - R29 appears nowhere in `docs/citations.md`, and the dossier's EAST section covers the baselines, the protocol and the citation hazards but not DyReLU (§6.2, §7.14).

Replaces `nn.ReLU`. A per-sample hyperfunction produces $2K$ coefficients per channel:

$$\theta = \sigma\!\big(W_2\,\mathrm{ReLU}(W_1 \bar{x})\big)\cdot 2 - 1, \qquad
\bar{x}_c = \frac{1}{HW}\sum_{h,w} x_{c,h,w}$$

$$y_c = \max_{k \in \{1..K\}} \big(a_{c,k}\, x_c + b_{c,k}\big), \qquad
a = \theta_a\lambda_a + a_{\text{init}},\quad b = \theta_b\lambda_b + b_{\text{init}}$$

with $K = 2$, `reduction = 4`, and buffers `lambdas = [1.0, 0.5]`, `a_init = [1, 0]`, `b_init = [0, 0]` (`dyrelu_adapter.py:17-19`) - so at $\theta = 0$ the unit is exactly the identity on the positive branch. `get_relu_coefs` handles both 4-D (spatial mean, `:23-25`) and 2-D input (`:26-32`, the VGG classifier path where `DyReLUB(4096)` receives `[B, 4096]`; this branch previously raised `UnboundLocalError` from a `thera`/`theta` typo).

Parameters per unit - MEASURED, exact for $C \in \{64,128,256,512\}$:

$$P(C) = \underbrace{\tfrac{C^2}{4} + \tfrac{C}{4}}_{\text{Linear}(C,\,C/4)} +
\underbrace{C^2 + 4C}_{\text{Linear}(C/4,\,4C)} = 1.25\,C^2 + 4.25\,C$$

| $C$ | 64 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|
| params | 5,392 | 21,024 | 83,008 | 329,856 |

**Placement.** ResNet stem `relu_v1` (`models/resnet.py:167-172`); 2 per `BasicBlock` (`:36-44`); 3 per `Bottleneck` (`:94-105`); VGG - one per conv in `make_layers` (`vgg.py:101-106`) plus two at width 4096 in the classifier (`vgg.py:26-34`); WRN-22 blocks (`resnet.py:334-339`, `:310-318`, `:385-390`).

**Cost on the anchor - MEASURED.** `resnet34(dyrelu_en=True)` instantiates **33** `DyReLUB` modules (1 stem + 2 × 16 blocks) and totals **24,463,290** parameters vs 21,282,122 for the plain-ReLU anchor: **+3,181,168 (+14.9%)**, all of it in `hyperfunction` tensors, all of it outside the sparsity denominator (defect M5, §8.6).

### `DyReLUAdapter` (`project/dyrelu_adapter.py:62-104`)

Holds one `DyReLUB` and one `nn.ReLU` side by side and blends them:

$$y = \beta\,\mathrm{DyReLU}(x) + (1-\beta)\,\mathrm{ReLU}(x), \qquad
\beta(t) = \begin{cases}
1 & t < t_{\text{start}}\\[2pt]
1 - \dfrac{t - t_{\text{start}}}{t_{\text{end}} - t_{\text{start}}} & t_{\text{start}} \le t < t_{\text{end}}\\[6pt]
0 & t \ge t_{\text{end}}
\end{cases}$$

`forward` short-circuits at both endpoints so the exact-0 and exact-1 cases cost one branch, not a blend (`:95-98`). A zero-length window is a legal instant switch - the `t >= t_end` guard fires first, so the denominator is strictly positive (`:86-90`). `get_beta` raises `RuntimeError` if the window was never set (`:75-79`).

**Scheduling.** `t` is an integer epoch counter incremented by `step()` (`:102-104`), driven by `step_dyrelu_adapter(model)` at the end of every training epoch (`trainer.py:254-255`), every BaCP pretraining epoch (`bacp.py:559-560`) and every BaCP finetune epoch (`bacp.py:593-594`). `set_t_for_dyrelu_adapter` broadcasts the window to every adapter (`:107-112`). The window is chosen in `_initialize_dyrelu_phasing` (`training_utils.py:165-181`):

| path | $t_{\text{start}}$ | $t_{\text{end}}$ |
|---|---|---|
| non-BaCP (`:171-173`) | $0.2 \cdot (\text{epochs} + \text{epochs}\cdot\text{recovery\_epochs})$ | $0.8 \cdot (\dots)$ |
| BaCP (`:169`, `:175-178`) | **10, hardcoded** | $\min\big(10 + \text{epochs}+\text{recovery\_epochs}+0.25\,\text{epochs\_ft},\ \text{epochs}+\text{recovery\_epochs}+\text{epochs\_ft}\big)$ |

The intent is that by $t_{\text{end}}$ the DyReLU branch has left the forward pass entirely, which is the stated justification for excluding `hyperfunction` parameters from the sparsity denominator (`pruning_factory.py:20-22`). That justification holds for the *phasing* variant (B2) and not for B1, which holds DyReLU on for the whole run.

Ladder rungs: `B1` sets `dyrelu_en=True, dyrelu_phasing_en=False`; `B2` adds `dyrelu_phasing_en=True` (`manifest.py:274`, `:280`). Both are `priority='optional'`. **Neither has been run.**

## 1.6 Weight sharing (`project/weight_sharing.py`)

The mechanism comes from EAST (R14), but ⚠ `docs/citations.md` §9 does **not** cover weight sharing - it verifies EAST for the baseline table, the protocol and the citation hazards only - so R14 is **UNVERIFIED for this use** (§6.2) and the rescale below is project-original in any case (§7.11).

`apply_weight_sharing_resnet(model_wrapper, R=2)` unwraps to the inner backbone (`weight_sharing.py:7`), records `total_params` **before** aliasing (`:8`), then for each of `layer1..layer4` (`:11`) designates block index $R-1 = 1$ as master and aliases every later block's direct-child `Conv2d` weights of matching shape onto the master's (`:20-31`):

```python
del s_child.weight
s_child.weight = m_child.weight
```

**What is aliased - MEASURED on the anchor:**

| stage | blocks | shared block indices (`conv1` and `conv2` each) |
|---|---:|---|
| `layer1` | 3 | {1, 2} |
| `layer2` | 4 | {1, 2, 3} |
| `layer3` | 6 | {1, 2, 3, 4, 5} |
| `layer4` | 3 | {1, 2} |

Block 0 is never shared (it carries the downsample and a different input width). **BatchNorm is not shared** - only `Conv2d` children pass the type test (`:28`). **Downsample is never shared** - it is an `nn.Sequential`, not a `Conv2d`. Stages with `num_blocks <= R` are skipped entirely (`:18`).

Each aliased conv gains one scalar `nn.Parameter` `scaler` (`:33-34`) and a monkeypatched `forward` that uses `weight * scaler` (`:36-43`). MEASURED: 16 scalers on the anchor. Each is 0-dimensional, so `layer_check`'s `param.dim() <= 1` test (`pruning_factory.py:135`) keeps them out of the pruning set. ⚠ The gains must be exempt from weight decay - under the default $5\times10^{-4}$ they decay toward zero and progressively silence the shared blocks (§7.11).

### Effect on the sparsity denominator - MEASURED

`named_parameters()` and `parameters()` de-duplicate aliased tensors by identity; `state_dict()` does not (hence the load-before-share ordering at `training_utils.py:146-155`). So every counter in the project sees the *unique* set:

| quantity | unshared | shared (R=2) | change |
|---|---:|---:|---:|
| inner-backbone trainable params | 21,276,992 | 11,176,272 | −47.47% ("intrinsic sparsity", printed at `weight_sharing.py:49`) |
| prunable set, backbone scope | 21,259,968 | 11,159,232 | −47.51% |

`_calculate_adjusted_sparsity` (`training_utils.py:357-395`) rescales the target so the *absolute* number of surviving weights matches the unshared arm (formula: §7.11). MEASURED on the anchor: $s = 0.999 \Rightarrow \text{keep} = 21{,}277 \Rightarrow s_{\text{adj}} = \mathbf{0.9981}$; $s = 0.99 \Rightarrow s_{\text{adj}} = 0.9810$. An unreachable rescaling raises rather than handing a negative sparsity to the pruner (`:386-393`). This is why `sparsity_requested` and `sparsity_target_adjusted` are separate fields on the run record (`manifest.py:288-290`).

Weight sharing is applied to the pruned model only, never to the frozen teachers (`training_utils.py:126-132`); it was previously a silent no-op on the BaCP path while still tripping `_calculate_adjusted_sparsity` into an `AttributeError`. Ladder rung `B3` enables it (`manifest.py:284-291`), `priority='optional'`, **not run**.

## 1.7 The other registered backbones

| model | MEASURED as-constructed params | embedded_dim | notes |
|---|---:|---:|---|
| `resnet50` | 23,500,352 (CIFAR stem, `fc`=Identity) | 2048 | `Bottleneck`, `[3,4,6,3]` |
| `resnet101` | 42,492,480 (CIFAR stem, `fc`=Identity) | 2048 | `Bottleneck`, `[3,4,23,3]` |
| `vgg11` | 132,863,336 (1000-way head) | 4096 | cfg `"A"`, `batch_norm=False` |
| `vgg19` | 143,667,240 (1000-way head) | 4096 | cfg `"E"`, `batch_norm=False` |
| `vit-tiny` / `vit-small` | not measured (hub download required) | `config.hidden_size` | require `--image_size 224` |
| `distilbert-*`, `roberta-*` | not measured | `config.hidden_size` | `cls_head` is `None`; own head retained |

Two things about the VGG entries matter, since both are registered and CLI-reachable, and both are open defects (M6, M2 - §8.6):

1. **VGG is never adapted for small images.** `adapt_resnet_for_small_images` runs only when `model_family == 'resnet'` (`model_factory.py:264`). At 32×32 the five maxpools in cfg `"A"`/`"E"` collapse the feature map to **1×1** (MEASURED: `features` output `(1,512,1,1)`), and `AdaptiveAvgPool2d((7,7))` (`vgg.py:21`) then *replicates* it back to 7×7, so the `Linear(25088, 4096)` sees 49 copies of the same 512-vector. This is not a crash - the forward pass returns a well-formed `(1,1000)` logit - which is exactly why it would go unnoticed.

2. **The `'classifier'` exclusion keyword swallows VGG's entire MLP.** `_EXCLUDE_TASK_HEAD` contains the bare substring `'classifier'` (`pruning_factory.py:44`), intended for HF sequence-classification heads, and the comment above it cites "the final `classifier.6` layer was kept dense" (`:37-39`). But `layer_check` does a substring match on the full parameter name, so `classifier.0.weight` and `classifier.3.weight` are excluded too. MEASURED, after `remove_last_layer`:

   | model | features | classifier | prunable under `layer_check` |
   |---|---:|---:|---:|
   | `vgg11` | 9,220,480 | 119,545,856 | **9,217,728 (7.2% of the model)** |
   | `vgg19` | 20,024,384 | 119,545,856 | **20,018,880 (14.3% of the model)** |

   A "99% sparse VGG-11" under this scope leaves 92,177 pruned-set weights alongside **119.5M untouched dense parameters** - 92.8% of the network. The anchor is unaffected (ResNet-34 names its head `cls_head`, and `remove_last_layer` leaves no `classifier` attribute), and no VGG run has been performed.

`project/models/vgg.py:5` imports `_ovewrite_named_param` and `WeightsEnum` from `torchvision.models.vgg` - a private symbol, imported at module load, so a torchvision version bump would break `import model_factory` entirely rather than only VGG (defect M7).

**Untested paths.** Only `resnet34` has been exercised at scale. `resnet50`, `resnet101`, `vgg11`, `vgg19`, both ViTs and all four language models have **not** been run under the current codebase. DyReLU (`B1`/`B2`) and weight sharing (`B3`) have **not** been run.

---

# 2. Datasets

## 2.1 The anchor cell, data side

Every rung of the ladder runs on one cell, fixed at the top of the manifest and validated at import against the live registries (`project/experiments/manifest.py:57-63`, checked by `manifest.py:733-747`): `cifar10`, 10 classes, `image_size=32`, `resnet34`, `model_type='cv'`. `--image_size` defaults to 32 in all three parsers (`project/scripts/scripting_utils.py:65`, `:153`, `:298`). Nothing outside CIFAR-10 has been run.

Two protocol values touch the data pipeline directly:

| protocol | `batch_size` | `num_workers` | source |
|---|---|---|---|
| `east250` (tier 1+) | 128 | 8 (overridden at dispatch) | `manifest.py:94`, `:105` |
| `smoke` (tier 0) | 32 | 0 | `manifest.py:110`, `:125` |

`num_workers=0` on the smoke protocol is a MEASURED decision, not a default: a CIFAR-10 loader with `num_workers=8` took 72.9 s to yield its first two batches versus 0.0 s with 0 workers, and that cost is paid three times per cell (`manifest.py:117-124`). ⚠ That measurement was taken on the **local Windows development checkout**, not on the Linux run box - the machine is not otherwise specified in the source, so the 72.9 s must not be quoted as a property of the hardware in §4.1 (§4.1, *Where the 72.9 s dataloader measurement was taken*). At tier 1 the pool overrides the protocol value entirely - the derivation, and why `os.cpu_count()` is the wrong call inside a container, is in §4.2. On the run box it resolves to **30** workers per cell at 8 concurrent cells.

## 2.2 Registered datasets

Two disjoint registries. `CV_DATASETS` (`project/dataset_factory.py:56-64`) maps a name to a `torchvision` class; `TEXT_DATASETS` (`:78-81`) maps a name to a `(hub_repo, config)` pair, kept separate because text needs a tokenizer and a collator rather than an image transform. Disjointness is pinned by `project/tests/test_registry.py:45`.

| name | kind | classes | native size | train | test | reachable? |
|---|---|---|---|---|---|---|
| `cifar10` | CV | 10 | 32x32 RGB | 50,000 | 10,000 | yes - the anchor, exercised at tier 1 |
| `cifar100` | CV | 100 | 32x32 RGB | 50,000 | 10,000 | yes (transform path only; never run) |
| `svhn` | CV | 10 | 32x32 RGB | 73,257 | 26,032 | yes (never run) |
| `mnist` | CV | 10 | 28x28 L | 60,000 | 10,000 | yes (never run) |
| `fmnist` | CV | 10 | 28x28 L | 60,000 | 10,000 | yes (never run) |
| `emnist` (balanced) | CV | 47 | 28x28 L | 112,800 | 18,800 | **train yes, test split BROKEN - defect D2, §8.7** |
| `imagenet` | CV | 1000 | variable | 1,281,167 | 50,000 (val) | **suspect - defect D3, §8.7** |
| `sst2` | LLM | 2 | n/a (128 tokens) | 67,349 | 872 dev (test unlabelled) | yes; GLUE arrow cache present under `project/cache/glue/sst2/` |
| `wikitext2` | LLM | n/a (MLM) | n/a (512-token blocks) | 36,718 rows | 4,358 rows | **BROKEN - defect D1, §8.7** |

**Dataset references.** All nine were uncited until this revision - the split sizes above appeared with no source for the corpora themselves. They are now R37 (CIFAR-10/100), R38 (SVHN), R39 (MNIST), R40 (Fashion-MNIST), R41 (EMNIST), R42 (ImageNet / ILSVRC), R43 (SST) with R44 (GLUE) for the split actually loaded, and R45 (WikiText-2) - **all `UNVERIFIED`**, none recorded in `docs/citations.md` (§6.2). The anchor cell needs two of them, R37 and R30, and a paper reporting ResNet-34 on CIFAR-10 must carry both.

Split sizes are the canonical upstream figures for these corpora; they were **not measured in this session** (torchvision is not importable in the local checkout environment, and no CV dataset other than CIFAR-10 has been instantiated). The 872-example SST-2 dev size is however stated in the code itself (`dataset_factory.py:496-497`).

Class counts live in `DATASET_NUM_CLASSES` (`dataset_factory.py:87-96`) so `--num_classes` cannot silently disagree with the data; `wikitext2` is deliberately absent because MLM predicts over the tokenizer vocabulary and has no fixed class count (`:85-86`, asserted by `project/tests/test_imports.py:326-329`). EMNIST is registered as 47 classes because `split='balanced'` is forced (`:242-243`, `:264-265`).

`GRAYSCALE_DATASETS = ["mnist", "fmnist", "emnist"]` (`:66`) drives a `T.Grayscale(num_output_channels=3)` prepended to normalisation. This is `T.Grayscale`, not a lambda that repeats the channel, specifically because a lambda closure cannot be pickled and therefore could not be shipped to a spawn-based DataLoader worker - it only ever worked because Databricks forks (`:124-128`, pinned by `project/tests/test_data_invariants.py:157-165`).

## 2.3 Dataset construction: signature introspection

`get_dataset_train_fn` (`:221-245`) and `get_dataset_test_fn` (`:248-269`) build a zero-argument thunk by introspecting the torchvision constructor:

```python
sig = inspect.signature(dataset_class.__init__)
if 'train' in sig.parameters:  dataset_args['train']  = True / False
if 'split' in sig.parameters:  dataset_args['split']  = 'train' / 'test'
if dataset_name == 'emnist':   dataset_args['split']  = 'balanced'
if dataset_name == 'imagenet': dataset_args['split']  = 'val'   # test fn only
```

This is what routes `CIFAR10(train=...)` versus `SVHN(split=...)` from one call site. It is also the mechanism behind defects D2 and D3: any dataset whose `train` flag is absorbed by `**kwargs` rather than named in the signature never receives `train=False`.

## 2.4 Train / validation / test split

Built in `load_cv_datasets` (`dataset_factory.py:298-347`).

| | supervised (`t_type='supervised'`) | contrastive (`t_type='contrastive'`) |
|---|---|---|
| train | 80% of the train file | **100% of the train file** |
| validation | 20% of the train file, eval transform | **`None`** |
| test | the test file | the test file |
| CIFAR-10 counts | 40,000 / 10,000 / 10,000 | 50,000 / - / 10,000 |

```
val_size   = int(0.20 * train_size)                                     # :319
generator  = torch.Generator().manual_seed(val_split_seed)              # :335
trainset, valset = random_split(trainset, [train_size, val_size], generator=generator)  # :336
valset = deepcopy(valset)                                               # :338
valset.dataset.transform = get_test_transform(dataset_name, size)       # :339
```

**The `VAL_SPLIT_SEED` mechanism.** `VAL_SPLIT_SEED = 1234` is a module constant (`dataset_factory.py:70`), threaded as a *default parameter* through `load_cv_datasets(val_split_seed=VAL_SPLIT_SEED)` (`:304`) and `load_cv_dataloaders` (`:358`), and read from args as `getattr(args, 'val_split_seed', VAL_SPLIT_SEED)` in `get_dataloaders` (`:585`). It is **never** derived from `args.seed`. Two independent reasons, both recorded at `:322-334`:

1. Without an explicit generator, `random_split` consumed the **global** torch RNG, whose state at that point depends on how much randomness model construction used - `_initialize_models` runs before `_initialize_data_loaders` (`project/training_utils.py:204-205`). Changing the backbone therefore changed the validation set, so two models were never evaluated on the same held-out data.
2. Holding the split fixed across seeds is what makes an $n$-seed error bar mean *initialisation and data-order variance*. If the split moved with `--seed`, the spread would additionally contain split variance and early stopping would select on different images per seed.

This matters directly for the measured noise floor: the A0 dense 5-seed spread of $\hat\sigma = 0.30$ points is a statement about init + data order only, because all five seeds saw the identical 40,000/10,000 partition.

The mechanism is pinned by four tests: independence from prior RNG consumption (`project/tests/test_eval_correctness.py:79-88`), stability across calls (`:91-92`), that the seed is actually wired through and not ignored (`:95-97`), and that the signature default is the module constant and `get_dataloaders` does not thread `args.seed` into it (`:100-116`). `--val_split_seed` is exposed on all three CLIs (default 1234, `scripting_utils.py:124`, `:214`, `:363`), recorded on every run (`project/results.py:855`), and pinned as a scope key (`project/tests/test_experiment_invariants.py:250`). It should be varied **only** for a deliberate split-robustness check.

The `deepcopy` at `:338` is required, not cosmetic: `random_split` returns two `Subset`s sharing one underlying dataset object, so assigning `.dataset.transform` without the copy would replace the *training* transform too. The cost is a full duplicate of the training array in host memory (CIFAR-10: $50{,}000 \times 32 \times 32 \times 3 = 153.6$ MB uint8). Not measured as a bottleneck.

**Asymmetry that must appear in the paper (defect D4).** The contrastive phase has no validation holdout at all (`:340-341`), so BaCP's contrastive stage trains on 50,000 CIFAR-10 images while the CE baseline's supervised stage trains on 40,000 - 25% more data. It is not a leak (the test file is untouched either way), but it is an uncontrolled difference between the arms. It also explains why the contrastive phase saves on minimum training loss rather than on validation accuracy (`test_data_invariants.py:170-179`). BaCP's fine-tuning phase does rebuild supervised loaders and therefore *does* get the 40,000/10,000 split (`project/bacp.py:352`, via `supervised_override=True`).

## 2.5 Augmentation

`get_train_transform(dataset_name, t_type, size)` (`dataset_factory.py:140-205`) has three branches: `cifar10`, `imagenet`, and a shared fallback. `AugmentData` (`:108-117`) then wraps the resulting `Compose` and applies it `n_views` times.

### 2.5.1 Supervised (single view)

| dataset | train transform, in order |
|---|---|
| `cifar10` (`:146-152`) | `Resize((s,s))` then `RandomCrop(s, padding=4)` then `RandomHorizontalFlip()` then norm |
| `imagenet` (`:164-168`) | `RandomResizedCrop(s)` then `RandomHorizontalFlip()` then norm |
| all others (`:184-189`) | `Resize((s,s))` then `RandomCrop(s, padding=4)` then `RandomHorizontalFlip()` then norm |

At $s=32$ the CIFAR-10 branch is exactly the standard CIFAR recipe (the `Resize((32,32))` is an identity on 32x32 input). This is the transform behind the MEASURED $91.64 \pm 0.30$ dense result.

The fallback branch at `:177-200` exists because it previously **raised**, which made five of the seven registered CV datasets - `cifar100`, `svhn`, `mnist`, `fmnist`, `emnist` - unreachable even though the CLI accepted them and the AAAI submission reports numbers on several of them (`:178-183`). Whatever produced those published numbers took a different code path than this one. Coverage of every registered dataset is now pinned by `test_data_invariants.py:134-149`.

### 2.5.2 Contrastive (two views)

| dataset | train transform, in order |
|---|---|
| `cifar10` (`:153-162`) | `Resize((s,s))`, `RandomResizedCrop(s, scale=(0.2,1.0))`, `RandomHorizontalFlip()`, `RandomApply([ColorJitter(0.8,0.8,0.8,0.2)], p=0.8)`, `RandomGrayscale(p=0.2)`, `GaussianBlur(k, sigma=(0.1,2.0))`, norm |
| `imagenet` (`:169-176`) | same minus the leading `Resize` |
| all others (`:190-198`) | identical to `cifar10` |

This is the SimCLR stack (Chen et al., **R17**, §6.2 - the same reference NT-Xent is cited to in §7.2.3), and it is now **the same for every dataset**; there is no per-dataset tuning.

`AugmentData.__call__` (`:113-117`) is a **type switch, not a length change**: at `n_views=1` it returns a bare tensor, at `n_views=2` a Python list of two independently-drawn tensors. `default_collate` then recurses into that inner list and stacks position-wise, so a contrastive batch is

$$\big[\,[\;T_{[B,3,H,W]},\;T_{[B,3,H,W]}\;],\;\;T_{[B]}\,\big]$$

a list of two batched tensors, **not** a single $[B,2,3,H,W]$ tensor and not a list of $B$ pairs. `BaCPTrainer._split_views` (`project/bacp.py:678-697`) unpacks against exactly this, feeding view 1 to the trainable student and view 2 to all three frozen teachers (`bacp.py:482-502`). Both facts are pinned (`test_data_invariants.py:44-90`), including that the two views are genuinely different draws rather than the same tensor twice - if they collapsed to one draw the self-supervised term would compare an image with itself and be trivially satisfiable (`:56-66`). `_split_views` returns `(data, data)` for a non-list batch, which is how the `n_views=1` C-null rung and the whole text path work.

`--n_views` defaults to 2 (`scripting_utils.py:293`) and is forced to 1 whenever `is_bacp` is false (`training_utils.py:253`).

### 2.5.3 Normalisation constants

`normalize_data` (`:120-137`), constants at `DATASET_STATS` (`:97-105`). Applied last, after all augmentation; `T.Grayscale` is applied *before* `ToTensor` so it operates on the PIL image.

| dataset | mean | std |
|---|---|---|
| `cifar10` | `[0.4914, 0.4822, 0.4465]` | `[0.2023, 0.1994, 0.2010]` |
| `cifar100` | `[0.5071, 0.4867, 0.4408]` | `[0.2675, 0.2565, 0.2761]` |
| `svhn` | `[0.4380, 0.4440, 0.4730]` | `[0.1751, 0.1771, 0.1744]` |
| `mnist` | `[0.1307]` | `[0.3081]` |
| `fmnist` | `[0.2860]` | `[0.3530]` |
| `emnist` | `[0.1307]` | `[0.3081]` |
| `imagenet` | `[0.485, 0.456, 0.406]` | `[0.229, 0.224, 0.225]` |

For grayscale datasets the single value is replicated by *list* repetition, `T.Normalize(mean*3, std*3)` giving `[0.1307, 0.1307, 0.1307]` (`:131`). Note `emnist` reuses MNIST's constants verbatim; EMNIST's own channel statistics are **not stated in the source** and were not computed. Every entry of `CV_DATASETS` is required to have stats (`test_data_invariants.py:151-152`).

### 2.5.4 AutoAugment / RandAugment / GaussianBlur / Mixup

- **RandAugment**: not used anywhere in the repository. Not imported.
- **AutoAugment**: imported at `dataset_factory.py:3` but **commented out** at `:161`. Dead in the current pipeline. This is a material change from the code that produced the AAAI submission: `project/dataset_utils_old.py:132-136` used `Resize` + `AutoAugment(CIFAR10)` as the *entire* CIFAR-10 contrastive transform - no crop, flip, jitter, grayscale or blur. The manifest calls this out as one of the confounds bundled into the submitted Table 1 I.P. column (`manifest.py:296-300`).
- **GaussianBlur**: used, contrastive branches only, at `:160`, `:175`, `:197`. Kernel size is derived from the image size at `:141-143`: $k = \lfloor 0.1 s \rfloor$, incremented to the next odd integer if even. At $s=32$ this gives $k=3$; at $s=224$, $k=23$. `sigma=(0.1, 2.0)` throughout.
- **Mixup / CutMix / Cutout**: not present anywhere in the repository.

## 2.6 Eval transform

`get_test_transform(dataset_name, size)` (`dataset_factory.py:208-218`):

| dataset | transform |
|---|---|
| `imagenet` | `Resize(256)`, `CenterCrop(size)`, norm |
| everything else | `Resize((size, size))`, norm |

Fully deterministic - no random op appears in either branch, pinned by `test_data_invariants.py:167-170`. It is used for the test set (`:257`), for the validation subset after the deepcopy (`:339`), and - importantly - is **not** wrapped in `AugmentData`, so eval batches are always single-view even during a BaCP contrastive run.

## 2.7 Loader construction

Assembled in `load_cv_dataloaders` (`:350-392`) from the shared kwargs helper `_loader_args` (`:414-434`).

| loader | `shuffle` | `drop_last` | `salt` | line |
|---|---|---|---|---|
| train | `True` | **`True`** | 0 | `:370-372` |
| test | `False` | **`False`** | 1 | `:373-375` |
| validation | `False` | **`False`** | 2 | `:377-380` |

**`drop_last=False` on val and test is a fix for a real, quantified defect (D7).** It used to be `True` everywhere, so CIFAR-10's 10,000-image test set was scored over $\lfloor 10000/512 \rfloor \times 512 = 9{,}728$ images: **272 test images were silently never scored, and which 272 depended on the batch size**, so changing the batch size changed the reported accuracy (`dataset_factory.py:365-369`, restated in the `_loader_args` docstring at `:418-421`). The same class of bug on the text side would have scored 864 of SST-2's 872 dev examples at batch 32 (`:495-497`). **All prior published ResNet-34 numbers were produced under the broken behaviour.** Guarded now by three tests: `valloader.drop_last is False` and `testloader.drop_last is False` (`test_eval_correctness.py:36-40`) and a direct count that every sample is scored (`:43-46`).

`drop_last=True` on the **train** loader is deliberate and load-bearing: `SupConLoss` does `labels.repeat(n_views, 1)` against a concatenated embedding batch, and a short final batch would silently misalign labels against embeddings (`:418-419`, `test_eval_correctness.py:49-57`, `test_data_invariants.py:182-187`).

**Per-loader generator seeding** (`:424-425`):

```python
generator = torch.Generator()
generator.manual_seed((int(seed) * 1000003) ^ int(salt))
```

The salt keeps train, val and test off a shared shuffle stream while remaining fully deterministic given `(seed, salt)`. Pinned both ways: same `(seed, salt)` gives the same `initial_seed()`, and the three salts give three distinct streams (`test_eval_correctness.py:120-131`).

**`worker_init_fn = _seed_worker`** (`:400-411`, wired at `:432`):

```python
worker_seed = torch.initial_seed() % 2 ** 32
np.random.seed(worker_seed)
random.seed(worker_seed)
```

PyTorch derives each worker's *torch* seed from the parent generator but leaves `random` and `numpy.random` unseeded, so any augmentation reaching for either is not reproducible. This is aggravated by `--num_workers` defaulting to `os.cpu_count()` (`scripting_utils.py:73`), which makes the worker count - and hence the RNG partition - machine-dependent. `_seed_worker` closes that.

Remaining `_loader_args` kwargs: `pin_memory=True`, `persistent_workers=(num_workers > 0)` (`:429-430`).

`get_dataloaders(args, supervised_override=False)` (`:565-603`) is the single entry point. It selects `t_type='contrastive' if is_bacp else 'supervised'` and `n_views = args.n_views if is_bacp else 1` (`:580-582`), and routes on `args.model_type` to the CV or text path, raising on anything else (`:603`). `supervised_override` exists so `BaCPTrainer.finetune()` can force the single-view supervised recipe while `self.is_bacp` is still true; it previously called `load_cv_dataloaders` directly, which raised `Unsupported dataset: sst2`, so BaCP + a language model + `--enable_finetune` had no working code path at all (`:568-573`, `bacp.py:346-352`).

## 2.8 Tier-0 truncation: `_LimitedLoader`

`project/training_utils.py:214-241`, with the `_maybe_limit` guard at `:244-247`.

```python
class _LimitedLoader:
    def __iter__(self):  return iter(itertools.islice(self.loader, self.n))   # :232-234
    def __len__(self):   return min(self.n, len(self.loader))                 # :236-237
    def __getattr__(self, name): return getattr(self.loader, name)            # :239-241
```

It is a wrapper with a **correct `__len__`** rather than an early `break` in the training loop, and the reason is stated at `:218-223`: `len(trainloader)` is load-bearing in three places - the scheduler's `total_steps`, the pruner's `total_steps`, and `train_batches` in the run record. An early `break` would leave all three describing a run that did not happen. `min(self.n, len(self.loader))` is the right form because a limit larger than the loader must not inflate the reported length (pinned by `project/tests/test_experiment_plumbing.py:66-68`); `__getattr__` delegation keeps `dataset`, `batch_size`, `collate_fn` and `drop_last` resolving through the wrapper (`:239-241`).

`_maybe_limit` is a no-op at `n <= 0` and passes `None` through unchanged (`:244-247`, pinned at `test_experiment_plumbing.py:88-92`), so tier 1+ pays nothing. Truncation is applied to all three loaders in `_initialize_data_loaders` (`:262-264`) and printed as `[SMOKE] loaders truncated: ... -- results are NOT valid measurements` (`:265-267`), and `limit_train_batches` / `limit_eval_batches` are written onto every run record, so a smoke run can never be misread as a measurement (`:224-226`).

BaCP's fine-tuning loaders are truncated separately at `project/bacp.py:363-365`, because `finetune()` builds them rather than `_initialize_data_loaders`. They previously escaped truncation entirely: a tier-0 BaCP cell ran its contrastive phase on 2 batches and then fine-tuned on the full training split, turning a 30-second smoke run into a 2.2-hour one (`bacp.py:354-361`).

The `smoke` protocol sets `limit_train_batches = limit_eval_batches = 2` (`manifest.py:115-116`; `:117-124` is the `num_workers` comment, not these keys), and the manifest treats the pair as a single coupled mechanism (`_COUPLED`, §5.1) so a rung may change both in one override.

## 2.9 Text pipeline

Salvaged from `dataset_utils_old.py`, which nothing imported - so the SST-2 and WikiText-2 halves of the submitted results table had **no reachable code path at all** (`dataset_factory.py:395-398`). `datasets` and `transformers` are imported lazily inside the loaders, not at module scope, because that cost about 4 s of import time on every run including every pytest session while being unused on the entire CV path (`:12-14`).

- **`get_tokenizer`** (`:459-468`) strips a `-mlm` suffix: `roberta-base-mlm` and `roberta-base` are the same checkpoint with different heads, and only the latter exists on the hub.
- **`TokenizedDataset`** (`:437-456`) yields exactly `{input_ids, attention_mask, labels}`, because `_handle_data_to_device` recognises a dict batch by its keys and the HF dataset's extra columns (`idx`, `sentence`, ...) would otherwise ride along.
- **SST-2** (`load_glue_dataloaders`, `:471-505`): `truncation=True, padding='max_length', max_length=128` (`:486-487`), `label` renamed to `labels`, torch format set (`:490-491`). GLUE test splits are unlabelled (`label == -1`), so **`testloader` is literally the same object as `valloader`** (`:505`). The comment at `:494-495` justifies this as what every paper in this space reports, CAP included - correct as far as it goes, but it means any early stopping on validation is model selection on the reported number. `patience=0` in the `east250` protocol makes this moot for CV; **no protocol has been defined for SST-2 yet**, so this needs to be settled before any text run.
- **WikiText-2** (`load_mlm_dataloaders`, `:508-548`): tokenize without truncation, then `group_texts` concatenates and chunks into `block_size` blocks (`:526-530`). `block_size` is capped defensively - `tokenizer.model_max_length` is 512 for RoBERTa but a sentinel $\approx 10^{30}$ for some tokenizers, which would make grouping allocate unboundedly (`:516-520`). `DataCollatorForLanguageModeling(mlm=True, mlm_probability=0.15)` supplies the `-100` label masking that both `Trainer._handle_metrics` and `BaCPTrainer._handle_metrics` branch on (`:75-77`, `:536-537`). The number of blocks after grouping is **not measured**. This loader is currently unreachable - defect D1, §8.7.
- No text augmentation exists, and that is a deliberate design position rather than a gap: in CAP the two "views" of a text example are the same tokens seen through two *different* models, which is exactly what the cross-model contrastive term compares, so `BaCPTrainer` duplicates the batch (`:590-593`, `bacp.py:685-697`).

---

# 3. Test parameters

## 3.1 The anchor cell

Everything in the ladder runs at one point in configuration space, declared once in `ANCHOR` (`project/experiments/manifest.py:57-63`) and inherited by every rung:

| field | value | source |
|---|---|---|
| `model_name` | `resnet34` | `manifest.py:58` |
| `model_type` | `cv` | `manifest.py:59` |
| `dataset_name` | `cifar10` | `manifest.py:60` |
| `num_classes` | 10 | `manifest.py:61` |
| `image_size` | 32 | `manifest.py:62` |

`LADDER_SPARSITY = 0.999` (`manifest.py:65`) is the single sparsity at which every attribution rung runs. The 40,000 / 10,000 / 10,000 split and the fixed `VAL_SPLIT_SEED = 1234` are described in §2.4; the split is a constant across all 65 tier-1 cells rather than a source of between-seed variance.

## 3.2 The `east250` protocol

`PROTOCOLS['east250']` (`manifest.py:92-106`). Every value is EAST's (arXiv:2411.13545v4, Table B.1 and §4) **except** `scheduler_type`, which is a declared deviation.

| parameter | value | from EAST? | set at |
|---|---|---|---|
| epochs | 250 | yes | `manifest.py:93` |
| batch size | 128 | yes | `manifest.py:94` |
| optimizer | SGD | yes | `manifest.py:95` |
| momentum | 0.9 | yes (EAST writes `SGD(0.9)`) | **hardcoded**, `project/training_utils.py:316` |
| learning rate | 0.1 | yes | `manifest.py:96` |
| weight decay | $1\times10^{-4}$ | yes (GraNet uses $5\times10^{-4}$) | `manifest.py:97` |
| LR schedule | cosine, per-step, `T_max = total_steps`, `eta_min = 1e-5` | **no - declared deviation** | `manifest.py:98`; `training_utils.py:343-350` |
| warmup | none (the cosine branch has no warmup; only `linear_with_warmup` does) | n/a | `training_utils.py:331-341` vs `:343-350` |
| patience | 0 = early stopping disabled | n/a (project decision) | `manifest.py:99`; guard at `project/trainer.py:377` |
| num_workers | 8 (fallback only; `pool.py` overrides - §4.2) | n/a | `manifest.py:105` |
| `delta_T` | 4000 optimizer **steps** | yes (RigL's interval in EAST's own table) | `resolve()`, `manifest.py:485` |
| sparsity scheduler | `cubic` | yes | `manifest.py:469` |
| `recovery_epochs` | 0 | n/a | `manifest.py:470` |
| **momentum** | **0.9** | yes | **hardcoded**, `training_utils.py:316`; **not a config key and not on `RunRecord`** - see *What is not parameterised* below |
| **fp16 AMP** | **on** | n/a | `enable_mixed_precision=True`, `trainer.py:44`, `bacp.py:55`; **not in `PROTOCOLS`** |
| gradient clipping | **none anywhere** | n/a | commented out at `training_utils.py:598` (§8.5) |

### The BaCP-side parameters - part of the protocol, not decoration

These are not in `PROTOCOLS`; `resolve()` applies them with `setdefault` to every `script == 'bacp'` rung (`manifest.py:520-538`), so they are as much a fixed part of every BaCP cell as the rows above. They belong in the paper's protocol table.

| parameter | value | set at | why it is load-bearing |
|---|---|---|---|
| $\tau$ (contrastive temperature) | **0.07** | `bacp.py:45` (dataclass default; no manifest override) | $1/\tau$ **scales the entire contrastive gradient** (§7.2.2), so $\tau$ and the learning rate are not independent knobs. Omitting it from the protocol table makes the reported lr uninterpretable on the BaCP arms |
| `contrastive_mode` | **`'cap'`** | `manifest.py:524`; default `bacp.py:82` | selects the CAP Eq. 1 functional over the legacy `SupConLoss + NTXentLoss` sum. The two are **different objectives**, not two implementations of one (§7.2.4) |
| `lambdas` $(\lambda_{PrC},\lambda_{FiC},\lambda_{SnC},\lambda_{CE})$ | **$(0.25,0.25,0.25,0.25)$** | `manifest.py:521`; default `bacp.py:248` | the rung overrides in §5.4-§5.6 are all deltas against this |
| `learnable_lambdas` | **`False`** | `bacp.py:74`; no manifest override | when true the four become raw `nn.Parameter`s with **no simplex projection** (§7.1, §8.8). Every ladder rung runs with it off |
| `num_snapshots` ($K$ for SnC) | **2** | `manifest.py:522`; default `bacp.py:92` | caps SnC's teacher count; the code previously allowed up to `epochs - 1` while the paper reported 2 |
| `n_views` | **2** | `manifest.py:523`; default `bacp.py:75`; forced to 1 when `is_bacp` is false (`training_utils.py:253`) | view count is entangled with the method, which is the entire reason rung C0b exists (§5.4) |
| `proj_mode` | **`'tied_frozen'`** | `manifest.py:532` | fairness requirement at this sparsity, not a preference (§1.4) |
| `distill_mode` / `lambda_kd` | **`'none'` / 0.0** | `manifest.py:537-538`; defaults `bacp.py:104-105` | cross-validated against each other at `bacp.py:167-181`, so a KD arm cannot be recorded with the term off |
| `kd_temperature` $T$ | **4.0** | `bacp.py:106`; no manifest override | the $T^2$ prefactor is a **balancing heuristic**, not an exact normaliser (§7.3), so $T$ changes the effective $\lambda_{kd}$ |
| `feature_distill_metric` | **`'cosine'`** | `bacp.py:107`; no manifest override | moot at the current call site, since both arguments arrive unit-norm and MSE $= \tfrac{2}{D}\times$ cosine exactly (§7.4). Record it anyway - it stops a cosine/MSE ablation being read as evidence about norm versus direction |
| BaCP finetune phase | 20 epochs, AdamW, lr $1\times10^{-4}$ (tier 0: 2 epochs) | `manifest.py:533-536` | a second optimizer and schedule on the BaCP arms only; `C-null` exists to prove this phase is not itself the effect (§5.4) |

Derived, not measured: `drop_last=True` on the train loader (`dataset_factory.py:372`) gives $\lfloor 40000/128 \rfloor = 312$ steps/epoch, so `total_steps` $= 312 \times 250 = 78{,}000$ (`training_utils.py:344`). The pruner freezes topology at `end_idx = round(0.8 \times 78{,}000) = 62{,}400` (`pruning_factory.py:160`), so `delta_T = 4000` yields **15** mask updates over a full run ($t = 4000, 8000, \dots, 60000$).

### EAST's internal contradiction on the LR schedule

EAST does not state one schedule. Its appendix Table B.1 gives `10x[75,150]` for the 250-epoch CIFAR runs. Its main text says the LR drops by $10\times$ at halfway and three-quarters of training, i.e. epochs **125 and 187**. Those are different schedules and **the paper does not say which produced its Table 1** (§6.4d). `_initialize_scheduler` implements only `linear_with_warmup` and `cosine` (`training_utils.py:329-353`), so neither EAST variant is reproducible here without new code; cosine is the closer of the two available options and is used widely in the same lineage. This is recorded on every run via the `scheduler_type` record field (`project/results.py:595`).

Consequence for the two uses of the protocol:

- **Attribution (the ladder).** Harmless. Every rung shares the schedule by construction - `resolve()` accumulates it from `PROTOCOLS` before any rung override (`manifest.py:465`) - so it cancels in every within-ladder delta.
- **Headline comparison against EAST's published 86.99 ± 0.32.** Not harmless. The substitution must be stated in the caption; the numbers are not a like-for-like reproduction.

### What is *not* parameterised

Three of the values above are fixed by the code rather than by `PROTOCOLS`, which is a different fact from their being unstated - they are in the tables above so the paper's protocol section can carry them; they are listed again here because a reader of the *record* cannot recover them.

- **Momentum is a literal**, not a config key (`training_utils.py:316`). It is therefore absent from `RunRecord` (`results.py:579-595` records `optimizer_type`, `learning_rate`, `weight_decay`, `scheduler_type`, `delta_T` - not momentum). A run under a different momentum would be indistinguishable in the record. This is the same defect class the weight-decay comment at `training_utils.py:305-308` describes as already fixed for `wd`.
- **fp16 AMP is on by default** - `enable_mixed_precision: bool = True` (`project/trainer.py:44`) - and is not in `PROTOCOLS`.
- **No gradient clipping exists anywhere.** The call is commented out at `training_utils.py:598`. Both facts are load-bearing for the open NaN defect (§8.1, §8.5).
- **Stale comment, `manifest.py:471`.** It reads "PROTOCOLS carries delta_T=4000", justifying `setdefault` over assignment. `PROTOCOLS` does **not** carry `delta_T` (`manifest.py:92-106`), so the `setdefault` at `manifest.py:485` always fires. The *behaviour* is correct (verified: `resolve('A5', tier=1)` yields `delta_T=4000`); only the comment is wrong. The second comment block at `manifest.py:476-478` gives the real reason (`baseline_script.py` has no `--delta_T` flag).

## 3.3 The `smoke` protocol (tier 0)

`PROTOCOLS['smoke']` (`manifest.py:108-126`):

| parameter | value | vs east250 |
|---|---|---|
| epochs | 2 | 250 |
| batch size | 32 | 128 |
| `limit_train_batches` | 2 | unset (full split) |
| `limit_eval_batches` | 2 | unset |
| num_workers | 0 | pool-derived |
| weight decay | **absent** | $1\times10^{-4}$ |

`delta_T` is **not** a key of `PROTOCOLS['smoke']` - it appears in neither protocol dict. It is set inside `resolve()` as `cfg.setdefault('delta_T', 1 if tier == 0 else 4000)` (`manifest.py:485`), which is why §3.2's table sources it to `resolve()` and not to `PROTOCOLS`. The tier-0 value is **1**.

Three independent reasons the numbers cannot be read as measurements:

1. **The loaders are truncated to two batches** (§2.8). A tier-0 "accuracy" is computed over $2 \times 32 = 64$ test images. `evaluate_exact` special-cases this so the dropped-sample invariant does not fire on every smoke record (`results.py:363-372`).
2. **The whole prune schedule is compressed into three optimizer steps.** Tier 0 has $\text{epochs} \times \text{limit\_train\_batches} = 4$ optimizer steps in total, so the cubic ramp's horizon is $T = \mathrm{round}(0.8 \times 4) = 3$ steps. Under the shipped `delta_T = 1` the mask updates **every** step and the ramp does reach its target - $0.999 \times (1-(1-3/3)^3) = 0.999$ at $t=3$ - so the label is attained, but by a three-step trajectory that has nothing in common with the 15-update, 62,400-step trajectory tier 1 runs (§3.2). Nothing about mask quality at tier 0 transfers.

   ⚠ **The 0.7030 figure belongs to the rejected setting, not the shipped one.** `manifest.py:480-484` says: "4000 is RigL's interval in EAST's own table. Tier 0 has only epochs\*limit_train_batches = 4 steps in total, so **that value** updates the mask exactly once, at t=1 of a 3-step cubic ramp -- every sparse rung then sits at $0.999\times(1-(1-1/3)^3) = 0.7030$ while the table calls it 99.9%." That is the manifest giving 0.7030 as its **reason for overriding `delta_T` to 1 at tier 0** - it describes what would happen at `delta_T = 4000`. An earlier version of this section attributed 0.7030 to the shipped `delta_T = 1` and called tier-0 sparsity "a lie by construction"; **the code does not support that reading** and the claim is withdrawn.
3. **The protocol silently changes the optimizer.** `smoke` omits `weight_decay`, so tier-0 runs fall back to `getattr(args, 'weight_decay', 5e-4)` (`training_utils.py:309`) - GraNet's value, not EAST's. Verified: `resolve('A5', tier=0)` returns a config with no `weight_decay` key at all. Tier 0 is therefore not even the same optimizer as tiers 1-3.

Tier 0 validates plumbing - that every rung resolves, launches, prunes, records and gates - and never appears in the paper (`TIERS[0]['why']`, `manifest.py:551-554`). Its cell keys are namespaced `smoke.*` precisely so a smoke record can never satisfy a real cell (§3.7).

## 3.4 Sparsity levels: two sets, for two questions

| constant | value | used for |
|---|---|---|
| `LADDER_SPARSITY` | 0.999 | every attribution rung, every tier (`manifest.py:65`) |
| `HEADLINE_SPARSITIES` | (0.99, 0.999, 0.9995, 0.9999) | the comparison sweep, winning config only (`manifest.py:69`) |

They differ because attribution and comparison have different statistical requirements (`manifest.py:51-56`). At 99.9%, ~21,265 weights survive against a 5,120-weight classifier: severe without being degenerate, and the outcome distribution is still approximately Gaussian. At 99.99% only ~2,126 survive, the outcome distribution becomes a **mixture** (some masks are viable, some are not) rather than a Gaussian, and resolving a 1-3 point component effect would need on the order of **100 runs per arm**. The headline table sweeps the full range because a comparison only needs the two arms to be measurable at each point; the ladder does not, because attribution needs a difference resolved against a noise floor.

**Open defect: the headline sweep is declared but not scheduled** (§8.8). `HEADLINE_SPARSITIES` is defined at `manifest.py:69` and referenced nowhere else. `cells()` accepts a `sparsity=` argument (`manifest.py:591`, `:608`) but no caller passes it - `pool.py:182` calls `M.cells(tier, rungs=rungs)`. `TIERS[3]['why']` claims tier 3 is "Everything at 8 seeds, plus the headline sparsity sweep" (`manifest.py:580`), but tier 3 resolves every cell at `LADDER_SPARSITY`.

## 3.5 Surviving weights at each sparsity

Denominator and parameter inventory: **§1.3.2** (authoritative). The main path prunes the task head, so the denominator is **21,265,088** and `target_params = int((1-s) * 21,265,088)`.

| target sparsity | surviving weights | dense classifier (5,120) as % of the surviving budget | projection head (65,536) as % of it |
|---|---:|---:|---:|
| 99% | 212,650 | 2.4% | 31% |
| **99.9%** (ladder) | **21,265** | **24.1%** | **308%** |
| 99.95% | 10,632 | 48.2% | 616% |
| 99.99% | 2,126 | 240.8% | 3,083% |

Two design decisions follow directly from this table, not from preference:

- **The A1/A2/A3/A4 2×2 is not optional.** At 99.9% a *dense* classifier is 24% of the entire surviving weight budget. Collapsing {uniform, ERK} × {head pruned, head dense} into one rung would permanently confound "ERK helps" with "ERK protects the classifier" (`manifest.py:228-239`, §5.2).
- **`proj_mode = 'tied_frozen'` is a fairness requirement**, not a preference (§1.4).

The 99.99% column additionally inherits the ERK allocator's fidelity caveat (§7.7, flag F8): at that target `conv1` receives roughly 2.7 expected surviving weights, and because masks are drawn Bernoulli per layer a completely dead stem is a realistic sampling outcome. Be sceptical of that column independently of the statistical argument in §3.4.

## 3.6 Primary endpoint

| | quantity | definition | where |
|---|---|---|---|
| **Primary** | `test_acc_exact_pct` | sample-weighted top-1 on the **test** split, `correct/total`, evaluated **once** on the final checkpoint | `PRIMARY`, `ladder.py:45`; computed at `results.py:296-384` |
| **Secondary** | `val_final5` | mean of the last five recorded **validation** accuracies | `_endpoint_from_detail`, `ladder.py:50-67` |

**Neither is best-epoch.** Best-epoch is the maximum of a noisy sequence: it is upward-biased (the bias grows with the number of epochs and with $\sigma$) and higher-variance than a mean, so it inflates every arm unequally depending on how noisy that arm happens to be - which at 99.9% sparsity differs systematically between rungs (estimator citation: R21, §6.2). `patience = 0` (`manifest.py:99`) disables early stopping for the same reason: stopping on a val criterion and then reporting test is defensible, but the project's stance is stronger - no selection at all, so there is no selection to argue about. There is **no test-set model selection** anywhere on the reporting path.

`evaluate_exact` also fixes two arithmetic errors in the older path (`results.py:298-305`): the reported metric was a batch-mean-of-batch-means, which diverges from the sample mean under a partial final batch, and perplexity was `mean(exp(batch_loss))`, upward-biased by Jensen (§7.13). It reports `eval_samples_dropped` so a silent `drop_last` loss on the test loader is visible in the record.

Four flags on the secondary endpoint, all from code reading, none measured:

1. **"Mean top-1 over the final 5 epochs" is the *secondary* in code, not the primary.** The docstring at `ladder.py:22-27` states this correctly; anything downstream that treats `val_final5` as the headline disagrees with `PRIMARY` at `ladder.py:45`.
2. **The "final five" fallback degrades silently.** With fewer than five values `_endpoint_from_detail` averages whatever exists (`ladder.py:64-65`) and returns it under the same column name. At tier 0 that is a 2-epoch mean.
3. **The key-preference chain can return a *training* metric.** `_endpoint_from_detail` tries `finetuning_acc`, then `accuracies`, then `training_acc` (`ladder.py:53`). `training_acc` is populated from `metrics.get('Training Accuracy')` (`project/bacp.py:893`). A record missing the first two therefore yields a train-accuracy mean labelled `val_final5`.
4. **The dict branch mis-orders sparse runs.** For any run with `target_sparsity` set, `accuracies` is a dict keyed by `round(current_sparsity, 4)` (`trainer.py:344-347`, `training_utils.py:720-722`). `_endpoint_from_detail` flattens it with `sorted(hist, key=str)` (`ladder.py:57-58`) - a *string* sort over JSON-serialized float keys. Since `"0.9988" < "0.999"` lexically but 0.9988 occurs *later* in time (the observed drift at divergence, §8.1), the concatenation is not chronological and "the last five" need not be the last five epochs.

## 3.7 Seeds, tiers, and cell-key scoping

`cells()` takes `n = min(rung.n_seeds, len(TIERS[tier]['seeds']))` (`manifest.py:607`), so a rung's declared seed count is an upper bound the tier can cut. Cell counts and costs: §5.8 and §4.4.

| tier | name | seeds | rungs | cells | planned U |
|---|---|---|---|---|---|
| 0 | smoke | (1,) | all 22 | 22 | 31.5 |
| 1 | spine | (1..5) | A0, A2-A5, C-null, C0, C0b, C1-C3, D1, D2 | 65 | 88.5 |
| 2 | attribution | (1..5) | all | 108 | 155.5 |
| 3 | full | (1..8) | all | 126 | 180.7 |

(`TIERS`, `manifest.py:548-584`; totals computed from `cells()`.)

**Tier scoping.** The cell key is `f'{scope}.{rung_id}.{model}.{dataset}.s{sparsity}.seed{seed}'` with `scope = 'smoke' if tier == 0 else 'east250'` (`manifest.py:618-620`). Examples: `smoke.A0.resnet34.cifar10.sdense.seed1` vs `east250.C-null.resnet34.cifar10.s0.999.seed1`. The key names the **experiment**, not the tier that scheduled it, so:

- tiers 1-3 share keys, and running them in sequence is **incremental** - a cell is complete iff a record carrying its key exists (`runner.completed_keys`, consumed at `pool.py:190`), so tier 2 after tier 1 runs only the 43 new cells rather than repeating all 65. Embedding the tier cost 244 U of duplicated work across the three tiers (`manifest.py:611-617`);
- tier 0 shares **none** of them, so a smoke record can never satisfy a real cell.

Both properties are asserted at import: `validate()` checks $K_0 \cap K_1 = \varnothing$ and $K_1 \subseteq K_3$ (`manifest.py:786-798`).

**Seed-coverage caveat at tier 1.** `A0` declares `n_seeds=8` so that every sparse rung's paired dense checkpoint exists at matching seed (`manifest.py:186-194`; enforced at `manifest.py:802-812`). Tier 1 supplies only 5 seeds, so the `min()` truncates *every* rung to 5 uniformly - the pairing holds, but the 8-seed dense reserve only materialises at tier 3. The `A0` note at `manifest.py:186-194` is also **truncated mid-sentence** in the source ("...enforces the coverage. the sparse rungs start from, and of the frozen teachers Stage C and D use.") and should be repaired before it is quoted anywhere.

**Measured noise floor.** From the archived first tier-1 attempt: A0 dense, 5 seeds from scratch, per-seed 91.6535, 91.5447, 91.5843, 91.2876, 92.1064; zero NaN. Recomputed from those five values:

| quantity | value |
|---|---:|
| mean | **91.6353** |
| sample sd, $ddof=1$ (= $\hat\sigma$) | **0.2975** |
| population sd, $ddof=0$ | 0.2661 |
| range | 0.8188 |
| 95% CI half-width, $t_{.975,4}\hat\sigma/\sqrt5$ | 0.369 |

So the headline is **91.64 ± 0.30** and $\hat{\sigma} = 0.30$ points, against the $\sigma = 1.5$ the plan's power table assumed; the consequences for detectability are in §7.12. ⚠ **An earlier figure of `91.68 ± 0.44` circulated and is not recoverable from these five values under any of `ddof=0`, `ddof=1`, range or CI half-width.** It is superseded everywhere in this document, and anything computed from it - the MDE columns, the $\chi^2$ interval - is recomputed at $\hat\sigma = 0.2975$.

⚠ **The "G0b threshold" is not a fact about the codebase.** The 2.0-point figure is a working number from this session's check and **is written in no file** (§5.9, §8.8): `noise_floor()` computes the statistic and nothing compares it to anything. Under the document's own legend it is neither MEASURED nor PLANNED. What can be said is the measurement: $\hat\sigma = 0.30$ on the dense arm, with the interval below. `sigma_chi2_interval` (`ladder.py:143-160`) must be printed alongside, because an $n=5$ estimate of $\sigma$ has a wide 95% interval and must not be quoted as *the* noise floor.

---

# 4. Hardware

## 4.1 The machine

| | |
|---|---|
| provider | Vast.ai rented instance |
| GPU | 4 x NVIDIA GeForce RTX 5070 Ti, 15 GB VRAM each |
| compute capability | sm_120 (Blackwell) |
| host CPU | 256 vCPU |
| disk | ~150 GB |
| software | torch 2.11.0+cu128, CUDA 12.8 |
| cost | approx **$1.603/hour** |

**The torch pin is deliberately not applied on this box.** Blackwell is sm_120 and needs torch >= 2.7 with cu128; `requirements.txt` pins `torch==2.5.1`, which predates it. `run_remote.sh:71-81` probes `torch.cuda.is_available()` first and, when the image's torch works, installs only the packages *around* it (`transformers`, `datasets`, `fvcore`, `pandas`, `scipy`, `pytest`, `matplotlib`, `tqdm`) and **creates no venv** - installing the pin would replace a functioning setup with one that raises "no kernel image is available for execution on the device". `PY` is resolved once at `run_remote.sh:42-46` so `start` and `status` never reach for a venv that `setup` deliberately did not create. Note that `scipy` is installed here by hand; it is **not** in `requirements.txt`, which is what makes the statistics fallback path in §7.12 live rather than hypothetical on a box set up any other way.

**Results survive a pod stop, not a terminate.** `run_remote.sh:50-53` sets `BACP_RESULTS_DIR=/workspace/bacp_results` whenever `/workspace` exists. (The comment at `run_remote.sh:50` calls `/workspace` "RunPod's persistent container volume"; this box is Vast.ai. The guard is a plain `[ -d /workspace ]`, so the behaviour is right and only the attribution is stale.) The two artefacts are treated differently by size and by what losing them costs: records are ~1.5 MB and are the *entire* result set plus all resume state; checkpoints are ~10 GB and are regenerable for ~1.5 GPU-hours. `snapshot` therefore tars `runs/` only (`run_remote.sh:153-155`), and `start` launches a second tmux session that snapshots every 300 s (`run_remote.sh:139-141`).

**A pytest gate runs before any GPU is spent** (`run_remote.sh:114-115`).

The first tier-1 attempt was stopped and its records archived to `/workspace/ARCHIVE_results_nan_20260819`.

### Determinism configuration

§2.7's reproducibility claims are about the *data* path. This is the rest of it, and it belongs beside the machine because it is the machine that decides what determinism costs.

| knob | setting | where | recorded per run? |
|---|---|---|---|
| `torch.manual_seed` / `cuda.manual_seed_all` / `random` / `numpy` | all seeded from one `--seed` (default 42) | `set_seed`, `utils.py:9-18`, called from `__post_init__` (`bacp.py:198`) so the recorded seed is the one active when models were built and the val split drawn | yes - `seed` on every record |
| `torch.backends.cudnn.deterministic` | **`True`** | `utils.py:16` - set unconditionally inside `set_seed` | yes - `cudnn_deterministic`, `results.py:210` |
| `torch.backends.cudnn.benchmark` | **`False`** | `utils.py:17` - set unconditionally inside `set_seed`. This forgoes cuDNN's autotuner, so it is also a throughput decision, taken silently | yes - `cudnn_benchmark`, `results.py:211` |
| `torch.use_deterministic_algorithms` | **off by default**; `--deterministic` turns it on and also sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` | `bacp.py:199-201`, `trainer.py:95`, flag at `scripting_utils.py:127-129`, `:218`, `:367`; rationale at `bacp.py:157-159` ("hard-errors on ops with no deterministic kernel and costs throughput") | yes - `deterministic_algorithms`, `results.py:212` |
| per-loader generators + `_seed_worker` | on | `dataset_factory.py:400-411`, `:424-432` (§2.7) | via `seed` / `val_split_seed` |
| **TF32 policy** | ⚠ **never set** | no `allow_tf32` and no `set_float32_matmul_precision` call exists anywhere in the repository, so matmul and conv precision are whatever the installed torch defaults to on Blackwell - a version-dependent property nothing pins | **no** |
| **NVIDIA driver version** | ⚠ **not captured** | `results.py:205-208` records `cuda_version`, `cudnn_version`, `gpu_name` and `gpu_count`; the driver is not among them | **no** |

No rung sets `--deterministic`, so the ladder runs on the seeded-but-not-forced path: bitwise reproducibility is **not** claimed, run-to-run variation from non-deterministic kernels sits inside the measured $\hat\sigma$ (§3.7) rather than being excluded from it, and that is the correct reading of the error bar. Two gaps to close before the paper's reproducibility statement is written: pin a TF32 policy explicitly rather than inheriting one, and record the driver version alongside `cuda_version`.

**Where the 72.9 s dataloader measurement was taken.** The `num_workers=8` finding behind the smoke protocol (§2.1, `manifest.py:117-124`) is attributed in the source to "this platform" and reasons about Windows process spawning. It was **not** measured on the Vast.ai box described above - that box is Linux, where `os.sched_getaffinity` is available and the whole `pool.py` worker derivation applies (§4.2). It comes from the **local Windows development checkout**; the source does not specify that machine further, and no CPU, RAM or disk figure for it is recorded anywhere. The conclusion (0 workers for a 2-batch CPU loader) is protocol-independent, but the 72.9 s figure itself must not be quoted as a property of the run box.

## 4.2 Scheduling

`project/experiments/pool.py` runs cells across GPUs. Three mechanisms, none optional.

**The dense rung is a hard barrier.** Every sparse cell resolves its starting weights from a dense record *matching its seed*, at execution time. `run_pool` partitions `todo` into `dense` (`script == 'baseline'`) and `sparse`, and runs them as two sequential phases (`pool.py:194-204`). If any dense cell fails the pool stops immediately rather than launching sparse cells that would all die on a missing checkpoint (`pool.py:225-228`). This is not hypothetical: launching everything at once is how **21 of 22 cells failed in the first tier-0 run** (`pool.py:16-19`).

**Claiming is atomic.** `run_grid`'s resume check reads run records - correct for resuming, racy for concurrency: two workers listing cells at the same instant both see a cell as incomplete and both run it, wasting a GPU-hour and writing two records for one cell. `claim()` uses `os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)` (`pool.py:71`), which is atomic on POSIX and Windows alike - exactly one caller creates the file. No lock server, no database, works the same on a rented box's shared volume as locally. Claims carry `{pid, worker, at}` and are reclaimed after `stale_after=7200` s (`pool.py:60`, `:78-89`) so a killed process or a reclaimed spot instance does not strand a cell. A **failed** cell deliberately keeps its claim (`pool.py:159-162`) so a retry loop does not spin on it; deleting its record is how you ask for another attempt.

**Concurrency is 2 cells per GPU, and the binding constraint is SMs, not memory.** MEASURED: per-cell GPU memory is only ~**815 MB** of 15 GB, so four cells would fit trivially. Utilisation is what decides it - one cell per GPU leaves the card at **53%**; two takes it to **98%**. Beyond that the return is small and the host CPU (dataloader workers) becomes the limit. Note the `pool.py:25-28` docstring is stale: it says "roughly 3 GB of 24 GB", written for a different card, against the measured 815 MB of 15 GB here. The conclusion (pack 2-3, measure before raising) survives; the numbers do not.

**Dataloader workers are derived, not inherited.**

```
usable_cores()             = len(os.sched_getaffinity(0))          # pool.py:121
workers_per_cell(c, cores) = max(2, cores // max(c,1) - 2)         # pool.py:134
```

`os.sched_getaffinity(0)` and not `os.cpu_count()` because inside a container `cpu_count()` reports the **host's** cores and ignores cgroup pinning - 256 on this box. `TrainingArguments.num_workers` defaults to `os.cpu_count()` (`project/trainer.py:46`), so without this override each cell would have spawned 256 workers; at 8 concurrent cells that is **1,024 processes on 256 cores** (`pool.py:126-134`, and the same trap documented at `manifest.py:101-105`). With 256 usable cores at 8 concurrent slots the derived value is $256/8 - 2 = \mathbf{30}$ workers per cell. `run_pool` computes it *before* anything is scheduled and stamps it onto every cell's config (`pool.py:179-184`), so the value is uniform across the sweep and lands in every record.

`run_remote.sh:54` additionally pins `OMP_NUM_THREADS=4`.

**Operational caveat.** `run_remote.sh:128` defaults `PERGPU` to **1**, not 2. The measured 8-slot configuration requires `BACP_PER_GPU=2` in the environment; without it `start` launches 4 slots and roughly half the throughput below.

## 4.3 Measured throughput

MEASURED at 2 concurrent cells per GPU, 4 GPUs = 8 slots. **1 U = one 250-epoch full-length run**, anchored to the dense cell at 2,050 s of one slot.

| cell class | s/epoch | epochs | wall-clock | in U |
|---|---|---|---|---|
| dense (A0, `baseline` script) | 8.2 | 250 | 2,050 s = **34 min** | 1.00 (definition) |
| sparse CE-only (C0, `pruning` script, 0.999) | 7.8 | 250 | 1,950 s = 32.5 min | 0.95 |
| BaCP (C-null, one view, no teachers) | 16.3 | 250 + 20 finetune | 4,401 s = 73.4 min | **2.15** |

Aggregate across 8 slots: **14.0 U/hour**, i.e. **$0.1145 per U** at $1.603/hour.

**The manifest's `cost_u` understates BaCP by 1.95x.** `C-null` is charged `cost_u=1.1` (`manifest.py:307`) against a measured 2.15 U. Every BaCP rung carries the same class of error. `budget()`'s own caveat (`manifest.py:710-714`) warns that projections use the *median* measured epoch time across all recorded runs, which understates BaCP rungs and overstates dense ones - the measurement above confirms the direction and quantifies it for the one BaCP config actually run. The loop that was supposed to break the median out per script is a no-op (`for script in ('pruning', 'bacp', 'dense'): pass`, `manifest.py:679-680`) - defect §8.8.

Only `C-null` has been measured on the BaCP path, and it is the **cheapest** BaCP config in the ladder: one view, zero teachers. C0b (two views), C1/C2 (a frozen dense teacher forward), C3, D1 (two snapshot teachers) and D2 (three teachers) are all **not measured**.

## 4.4 Budget

Planned, from the manifest's own `cost_u` fields:

| tier | cells | planned U | planned hours @14 U/h | planned cost |
|---|---|---|---|---|
| 0 smoke | 22 | 31.5 | - (CPU, minutes) | - |
| 1 spine | 65 | 88.5 | 6.3 | $10.13 |
| 2 attribution | 108 | 155.5 | 11.1 | $17.80 |
| 3 full | 126 | 180.7 | 12.9 | $20.69 |

Two corrections exist, and they answer different questions. **Both are reported because neither is a measurement of the rungs that have not run.**

**(a) The floor** - charge every BaCP cell at C-null's measured rate, which is known to be too cheap for every BaCP rung except C-null itself:

| class | cells | s each | total |
|---|---|---|---|
| dense A0 | 5 | 2,050 | 10,250 s |
| sparse pruning (A2, A3, A4, A5, C0) | 25 | 1,950 | 48,750 s |
| BaCP (C-null, C0b, C1, C2, C3, D1, D2) at C-null's rate | 35 | 4,401 | 154,035 s |
| **total slot-time** | 65 | | **213,035 s = 59.2 slot-hours** |

= **103.9 U**, **7.4 wall-hours** on 8 slots, **~$11.9** for tier 1. That is 17% above the plan's 88.5 U and it is a *lower bound*.

**(b) The scaled estimate** - keep the manifest's relative ordering between BaCP rungs and multiply every `bacp` cell's `cost_u` by 1.95 and every `pruning` cell's by 0.95:

| tier | cells | nominal U | corrected U | wall clock @14.0 U/h | $ @1.603/h |
|---|---|---|---|---|---|
| 1 | 65 | 88.5 | 142.9 | 10.2 h | ~$16 |
| 2 (cumulative) | 108 | 155.5 | 250.5 | 17.9 h | ~$29 |
| 3 (cumulative) | 126 | 180.7 | 293.8 | 21.0 h | ~$34 |

Tiers share cell keys (§3.7), so these are cumulative: tier 2 after tier 1 is ~7.7 h incremental; tier 3 after tier 2 is ~3.1 h. The true tier-1 figure lies between 103.9 U and 142.9 U and cannot be narrowed without timing C0b through D2.

## 4.5 Every timing in §4.3 and §4.4 is now stale

The stem defect (§1.3.3) means every figure above was measured against a network doing roughly a quarter of the arithmetic the corrected one will do: **MACs multiply by 3.98** when `maxpool` becomes `Identity`, while **zero parameters and zero `state_dict` keys change**.

- Unaffected: §1.3.2's inventory, §3.5's surviving-weight table, the ERK densities (§7.7), the 24% / 308% budget arguments, and every checkpoint-compatibility property.
- Stale and requiring re-measurement: the 8.2 / 7.8 / 16.3 s/epoch figures, the 14.0 U/hour aggregate, the $0.1145/U, the 53%/98% utilisation split that justified 2 cells per GPU, and the derived `num_workers=30`.

**All of §4.3 and §4.4 must be re-measured after the stem fix, before any tier is scheduled from them.** The concurrency choice is the one most likely to move: at ~4x the compute per cell, one cell may already saturate the SMs, in which case 2/GPU stops paying and `workers_per_cell` doubles to 62.

---

# 5. Experiments

Everything in this section is declared as data in two files and executed from them: `project/experiments/manifest.py` (the plan) and `project/experiments/ladder.py` (the read-out). Nothing here is a command line typed by hand. `manifest.py:913` calls `validate(strict=True)` at import, so a plan that is not self-consistent fails on `import manifest`, not six hours into a sweep.

**Status of the numbers in this section.** Costs and throughput are MEASURED (§4.3). The only *scientific* result measured so far is the Stage A dense rung `A0` (5 seeds, $91.64 \pm 0.30$ top-1). Every sparse cell attempted so far diverged to NaN (§8.1), and the stem decision (§1.3.3) invalidates even `A0` as a final number. **Treat every accuracy in this section as PLANNED unless explicitly labelled MEASURED.**

## 5.1 The design principle: one rung, one change

The ladder's single structural rule is that **a rung's `overrides` dict is the entire difference from the rung it inherits from** (`manifest.py:132-162`). A rung does not restate shared settings at all, so it cannot silently disagree with the rung below it. `resolve()` (`manifest.py:455`) walks the `inherits` chain root-first and applies each rung's overrides in order; `_chain()` (`manifest.py:442`) refuses cycles.

The consequence that matters for the statistics: **rung $k$'s "before" arm is literally rung $k-1$'s "after" arm.** There is no separate control run to drift out of protocol with the treatment run, and `aggregate()` (`ladder.py:295`) computes each row's $\Delta$ against the resolved parent, walking up to the nearest ancestor actually scheduled in the tier when the declared parent is not (`ladder.py:337-341`) - recording `delta_vs` and `delta_indirect` so an indirect comparator is visible rather than assumed.

This is the discipline the submitted paper's ablation lacked. `manifest.py:1-18` states the failure precisely: that ablation zeroed one $\lambda$ of four and left the other three at $0.25$, so "no PrC" was simultaneously a removal and a 25% downweighting of the whole contrastive signal (§5.6).

### The `bundle_ok` mechanism

`validate()` (`manifest.py:761-768`) refuses any rung whose `overrides` carries more than one key unless one of two escapes applies:

1. The key set is a subset of a declared coupled set in `_COUPLED` (`manifest.py:167-172`) - two flags that are one mechanism:

| coupled set | used by | why it is one mechanism |
|---|---|---|
| `{dyrelu_en, dyrelu_phasing_en}` | B1 | you cannot phase an activation that is off |
| `{limit_train_batches, limit_eval_batches}` | tier-0 protocol only, no rung | loader truncation is one switch |
| `{lambdas, num_snapshots}` | D1 | a snapshot weight with zero snapshots is not a setting |
| `{distill_mode, lambda_kd}` | C1 | a distillation mode with zero weight is not a mode |

2. The rung carries a written `bundle_ok` sentence. It is deliberately a **sentence and not a boolean** (`manifest.py:155-158`): the check is being made to say why.

**Exactly three rungs carry a `bundle_ok`.** Quoted verbatim:

**A1** (`manifest.py:203-207`) - the entry into sparsity:
> "Entry rung. Going from dense to sparse is irreducibly a bundle -- you cannot name a sparsity without naming a criterion, an allocation and a scope. A0->A1 is therefore read as 'what does sparsity cost', never as a single-mechanism contrast; the single-mechanism contrasts are A1->A2, A1->A3, A3->A4."

**C-null** (`manifest.py:308-311`) - the gate rung:
> "All three flags say the same thing -- turn BaCP off. The rung exists precisely to hold the code path fixed while removing every BaCP-specific ingredient, so it cannot be split."

**C3** (`manifest.py:364-369`) - the paper's central contrast:
> "One mechanism, three flags: the objective form changes from regression-onto-the-teacher to contrast-against-the-teacher. The TEACHER IS HELD FIXED (model_ft in both C2 and C3), which is the whole point -- swapping teacher and form together, as the pre-registered SnC-first order would have done, is the same two-changes-one-label error the submitted paper made."

`test_each_rung_changes_at_most_one_mechanism` (`project/tests/test_manifest.py:52`) pins this rule against regression.

### Where the ladder runs

| item | value | source |
|---|---|---|
| model / dataset | ResNet-34 / CIFAR-10, 32x32, 10 classes | `manifest.py:57-63` |
| ladder sparsity | $0.999$ | `manifest.py:65` |
| headline sweep | $(0.99,\ 0.999,\ 0.9995,\ 0.9999)$ - declared, not scheduled (§3.4) | `manifest.py:69` |
| protocol | 250 epochs, bs 128, SGD(0.9), lr 0.1, wd 1e-4, cosine, no early stopping (§3.2) | `manifest.py:92-106` |
| mask-update interval | $\Delta T = 4000$ steps (tier 0: 1) | `manifest.py:485` |
| external comparators | RigL $85.71 \pm 0.23$, EAST $86.99 \pm 0.32$ (§6.5) | `manifest.py:46-49` |
| primary endpoint | `test_acc_exact_pct` (§3.6) | `ladder.py:45`, `ladder.py:22-27` |

One **declared deviation**: `scheduler_type='cosine'` (§3.2, §6.4d). Every rung shares cosine by construction, so attribution *within* the ladder is unaffected; the headline comparison against EAST's published numbers must state the substitution rather than imply a match.

## 5.2 Stage A - the sparse substrate

Unpaired (`paired=False` on every rung). The mask lottery dominates run-to-run variance here, so seed correlation between arms is low and pairing buys little (`manifest.py:177-179`); `aggregate()` routes these rows to `unpaired_delta`, a Welch difference of means (§7.12).

| id | changes | exact `overrides` | answers | script | inherits | n | cost_u | **priority** | gate |
|---|---|---|---|---|---|---|---|---|---|
| **A0** | no pruning at all | `{}` | the dense ceiling; supplies every sparse rung's starting checkpoint | `baseline` | - | 8 | 1.0 | `essential` (default) | - |
| **A1** | uniform budget + magnitude + head in scope | `{'pruning_type':'local_magnitude', 'layerwise_alloc':'uniform', 'prune_task_head':True}` | what does sparsity cost at all | `pruning` | A0 | 3 | 1.0 | **optional** (`:202`) | - |
| **A2** | classifier removed from prunable scope | `{'prune_task_head': False}` | does protecting the head help, under uniform | `pruning` | A1 | 5 | 1.0 | `essential` (default) | - |
| **A3** | uniform -> ERK | `{'layerwise_alloc': 'erk'}` | does the allocation rule help, head pruned | `pruning` | A1 | 5 | 1.0 | `essential` (default) | - |
| **A4** | both, to close the 2x2 | `{'prune_task_head': False}` | does protecting the head help, under ERK | `pruning` | A3 | 5 | 1.0 | `essential` (default) | - |
| **A5** | static mask -> RigL drop-and-regrow | `{'pruning_type': 'rigl'}` | does dynamic topology add anything beyond allocation | `pruning` | A3 | 5 | 1.0 | `essential` (default) | - |
| **A6** | RigL schedule -> EAST cyclic | `{'pruning_type': 'east'}` | does the cyclic schedule add anything beyond RigL | `pruning` | A5 | 5 | 1.0 | **high** (`:261`) | - |

**`priority` is what decides what gets cut**, so it belongs in this table and not only in prose. The dataclass default is `'essential'` (`manifest.py:150`), and two Stage A rungs override it: **A1 is `optional`** - the manifest's reason is coverage, not power ("Optional only because A2-A4 pin the 2x2 corners that matter; if budget is tight this is the first Stage A rung to cut", `manifest.py:208-209`; see §5.10) - and **A6 is `high`**, the only Stage A rung marked so, because it is the EAST-cyclic arm the headline comparison needs. Their tier-1 exclusion (§3.7, §5.8) is a *tier* decision and is separate: tier 1 schedules A0, A2-A5, so A1 and A6 are absent from the spine whatever their priority says. Stage B's table (§5.3) already carries the column.

`A0` carries `n_seeds=8` for a structural reason, not a statistical one (`manifest.py:186-194`): `attach_checkpoint` pairs each sparse run with the **dense run of the same seed**. With only three dense records, seeds 4-8 of every sparse rung silently fell back to one shared checkpoint, so runs advertised as independent shared an initialisation and the paired designs above them were paired only from the pruning step onward. `validate()` now enforces coverage (`manifest.py:800-812`) and `test_sparse_runs_chain_to_a_checkpoint_of_the_same_seed` (`test_experiment_invariants.py:468`) enforces it against the records. The fallback path itself remains a defect (§8.4).

### The deliberate 2x2

A1/A2/A3/A4 form a full factorial on $\{\text{uniform}, \text{ERK}\} \times \{\text{head pruned}, \text{head dense}\}$ (`manifest.py:228-239`):

|  | head **pruned** | head **dense** |
|---|---|---|
| **uniform** | A1 | A2 |
| **ERK** | A3 | A4 |

Collapsing this into a single "we use ERK" rung would **permanently confound "ERK helps" with "ERK protects the classifier."** ERK assigns density by layer mass, and the classifier is a small, dense-ish layer under that rule - so an ERK arm run head-pruned already gives the head preferential treatment relative to uniform. At $0.999$ **the classifier is 24% of the entire surviving weight budget** (§3.5), so the confound is not hypothetical: a single collapsed arm could show a multi-point gain that is entirely head protection and nothing to do with the allocation rule elsewhere in the network. Once collapsed, no downstream statistic can separate them again, because the two factors were never varied independently.

One deliberate divergence from RigL is recorded rather than hidden (`manifest.py:235-239`, §6.6): RigL's Uniform distribution keeps the **first** layer dense; ours does not. Baking that carve-out in would put a third difference inside the uniform arms and destroy the 2x2. RigL applies the carve-out only to Uniform, not to ERK, so A3-A6 are unaffected.

### Why the main path runs through A3 -> A5, not A4

`A5` inherits **A3** (ERK, head **pruned**), not A4. The justification is comparability, stated at `manifest.py:246-255`:

> "RigL and GraNet both prune the final classifier by default -- RigL assigns ResNet-50 fc1000 an ERK sparsity of 0.957 and its reference implementation defaults `prune_last_layer=True`; GraNet registers every 2-D and 4-D tensor into the mask set with no classifier carve-out. Inheriting the head-dense corner would have made this rung, and therefore the whole substrate that Stage C is built on, incomparable to every published number we cite."

Because `BEST_SUBSTRATE` (`manifest.py:437`, default `'A5'`) is the root of the *entire* Stage C and D chain, this is not a local choice about one rung. Had A4 been on the main path, every BaCP number in the paper would sit on a substrate whose classifier is dense while RigL's, GraNet's and EAST's are not - and the headline table's comparison against $85.71$ / $86.99$ would be against a different problem. A2 and A4 remain corners of the 2x2 measuring **what the alternative convention buys**, not adopting it. The sparsity denominator this implies (21,265,088) is §1.3.2; the published-code evidence is §6.6.

`validate()` enforces that Stage C/D never drift off the substrate: for every substrate-rooted rung it asserts `pruning_type`, `layerwise_alloc` and `prune_task_head` match `resolve(BEST_SUBSTRATE)` unless the rung *declares* the change (`manifest.py:853-868`). Verified in this session: `resolve('C-null', tier=1)` carries `pruning_type='rigl'`, `layerwise_alloc='erk'`, `prune_task_head=True`, `target_sparsity=0.999`, `delta_T=4000`.

**ERK fidelity caveat** (affects A3-A6 and everything downstream): the code cites ERK and implements Erdos-Renyi over the unrolled matrix. Full statement, arithmetic and prescription: §7.7 and flag F8 (§6.3).

## 5.3 Stage B - EAST's architecture knobs

Paired (`paired=True`, the dataclass default at `manifest.py:149`). Same substrate, same mask-trajectory family; the intervention is architectural, so seeds correlate and pairing pays.

Stage B is **designed so that a null is publishable** (`manifest.py:266-269`): a pre-registered $\pm 1.0$pp equivalence bound with TOST reported alongside the difference CI (§7.12). "We can reject differences larger than 1.5pp" is a result; "$p = 0.43$" is not.

| id | changes | exact `overrides` | answers | script | inherits | n | cost_u | priority |
|---|---|---|---|---|---|---|---|---|
| **B1** | ReLU -> DyReLU | `{'dyrelu_en': True, 'dyrelu_phasing_en': False}` | does a dynamic activation help a 99.9%-sparse net | `pruning` | A6 | 5 | 1.0 | optional |
| **B2** | DyReLU phased out over training | `{'dyrelu_phasing_en': True}` | does phasing beat holding it on | `pruning` | B1 | 5 | 1.0 | optional |
| **B3** | residual blocks share weights | `{'weight_sharing_en': True}` | does parameter sharing survive extreme sparsity | `pruning` | B2 | 5 | 1.0 | optional |

B3 rewrites the sparsity target against the **unique** parameter count (§1.6), which is why `sparsity_requested` and `sparsity_target_adjusted` are separate fields on the record (`manifest.py:288-290`). `test_sparsity_target_was_attained` (`test_experiment_invariants.py:180`) compares against the adjusted value when present, and `test_architecture_is_stable` (`:280`) excludes sharing runs by design so the change shows up as a recorded field rather than as unexplained architecture drift.

All three are `priority='optional'` and Stage B is **absent from tier 1** entirely (`manifest.py:565-567`). If budget is short, Stage B is what gets cut. B1 additionally carries defect M5 (§8.6): 3,181,168 DyReLU parameters stay in the forward pass for the whole run yet sit outside the sparsity denominator.

## 5.4 Stage C - the scientific heart

Every Stage C rung shares the same dense teacher, the same number of optimisation steps, and matched forward-pass FLOPs. This is the block the submitted paper had **no analogue of at all** (`manifest.py:296-301`): it contained no CE-only arm - "no CE" ($\lambda_{CE} = 0$) is not "CE only" - so nothing separated the contrastive objective from the recipe wrapped around it: two views, AutoAugment, the SGD->AdamW phase swap, the projection head. Table 1's I.P. column runs a *different recipe*, not merely a different loss.

The $\lambda$ vector throughout is $(\lambda_{PrC}, \lambda_{FiC}, \lambda_{SnC}, \lambda_{CE})$; the code's index mapping is §7.1.

| id | changes | exact `overrides` | answers | script | inherits | n | cost_u | gate |
|---|---|---|---|---|---|---|---|---|
| **C-null** | BaCP path with every BaCP-specific ingredient off | `{'lambdas': (0.0,0.0,0.0,1.0), 'num_snapshots': 0, 'n_views': 1}` | is the BaCP *code path* neutral | `bacp` | `BEST_SUBSTRATE` | 5 | 1.1 | **G1** |
| **C0** | plain supervised training on the winning substrate | `{}` | the reference every later rung is measured against | `pruning` | `BEST_SUBSTRATE` | 8 | 1.0 | - |
| **C0b** | two augmented views per step, objective unchanged | `{'n_views': 2}` | is a gain contrast, or just more augmentation | `bacp` | C-null | 8 | 1.4 | - |
| **C1** | dense-teacher logit KD added | `{'distill_mode': 'kl', 'lambda_kd': 1.0}` | does *a teacher* help at all | `bacp` | C0b | 8 | 1.6 | **G3** |
| **C2** | logit KD -> feature-level matching | `{'distill_mode': 'feature'}` | does the *level* of injection matter | `bacp` | C1 | 8 | 1.6 | **G3** |
| **C3** | feature regression -> contrastive form (FiC) | `{'lambdas': (0.0,0.5,0.0,0.5), 'distill_mode': 'none', 'lambda_kd': 0.0}` | does the *contrastive form* matter | `bacp` | C2 | 8 | 1.8 | **G4** |

### The decomposition

C1 -> C2 -> C3 reads, in order (`manifest.py:371-374`):

$$\text{a teacher exists} \;\longrightarrow\; \text{features matter} \;\longrightarrow\; \text{the contrastive form matters}$$

Each arrow is one step. The last one is the tightest: **cosine feature matching is exactly the positive-pair term of InfoNCE with the denominator deleted** (derivation and its citation limits: §7.5), so C2 -> C3 is intended to isolate one thing - the presence of negatives - with the teacher (`model_ft`) held fixed on both sides. `test_c2_to_c3_holds_the_teacher_fixed` (`test_manifest.py:115`) pins the teacher. ⚠ **Under the shipped configuration the rung does not isolate one thing** - C3 also switches on label-defined positives. Open, §8.3.

C1 is the control the submitted paper most needed and did not have (`manifest.py:345-348`): three frozen dense teachers sit on the BaCP side of its Table 1 and **zero** on the I.P. side, so the cheapest explanation of every gain there is "a dense teacher helps" - nothing to do with contrast. C1 is what can refuse that reading.

### C-null is the cheapest, highest-information run in the plan

C-null and C0 run the *same objective* through *different code*: C0 through `pruning_script`, C-null through `BaCPTrainer` with one view and no contrastive terms. Any gap between them is an artefact of the BaCP path - the collate, the projection head, the optimizer swap, the finetune phase - and **every downstream number would be measuring that artefact rather than the method** (`manifest.py:311-317`). It runs first.

Two silent bugs in `resolve()` were caught by asserting that a resolved config contains what the ladder says it should (`manifest.py:493-510`):

1. A chain root's own overrides were skipped, so **C-null resolved to full BaCP** - the gate meant to catch implementation artefacts would have *been* the artefact.
2. The `BEST_SUBSTRATE` prefix was applied only to rungs whose own `inherits` was `None`, never to descendants. C-null roots all of Stage C and D, so C0b, C1, C2, C3, D1, D2 and all four D4 arms resolved with **no `pruning_type` at all** - and since `_initialize_pruner` computes `prune = (pruning_type and target_sparsity)`, they would have trained **dense** while the table labelled them 99.9%-sparse. Nothing would have raised.

`validate()` now asserts every non-dense rung resolves with both a pruner and a target (`manifest.py:838-851`) - described in the file as "the single highest-value assertion." `test_c_null_is_not_bacp` (`test_manifest.py:87`) states it separately "because it is the gate."

### Why C0b is not optional

**BaCP consumes two views per step by construction.** View count is therefore entangled with the method: it is not a hyperparameter that happens to differ, it is part of what BaCP *is*. Without C0b, a measured gain at C3 or D2 admits two explanations - "contrast helps" and "two augmented views per step help" - and they are **not separable after the fact**, because no arm in the design ever varies view count alone (`manifest.py:335-337`). C0b is that arm: `{'n_views': 2}`, objective unchanged, everything else inherited from C-null. It costs 1.4 U and it is the difference between a claim about contrastive learning and a claim about augmentation.

`proj_mode='tied_frozen'` is set as a default for all `bacp` rungs (`manifest.py:526-532`) and is a fairness requirement at this sparsity, not a preference (§1.4). `test_projection_head_is_frozen_on_every_bacp_rung` (`test_manifest.py:168`) enforces it.

## 5.5 Stage D - BaCP's teachers, forward

| id | changes | exact `overrides` | answers | script | inherits | n | cost_u | gate |
|---|---|---|---|---|---|---|---|---|
| **D1** | snapshot teachers added | `{'lambdas': (0.0, 1/3, 1/3, 1/3), 'num_snapshots': 2}` | does SnC add anything on top of FiC | `bacp` | C3 | 5 | 2.0 | **G5** |
| **D2** | pretrained-model teacher added = **full BaCP** | `{'lambdas': (0.25, 0.25, 0.25, 0.25)}` | does PrC add anything on top of FiC+SnC | `bacp` | D1 | 5 | 2.2 | **G5** |

D2 **is** full BaCP (`manifest.py:400-402`): "Everything above it is the path there, and every rung below it is a competing explanation that has been ruled out or not." `test_full_bacp_is_the_last_forward_rung` (`test_manifest.py:95`) pins that.

### The teacher-order correction

The forward order is **FiC -> SnC -> PrC**. This is **not** the pre-registered order, which was SnC -> PrC -> FiC. It was changed **before any run**, and the reason is recorded at `manifest.py:378-387`:

Stage C ends on a contrastive term against the **fine-tuned** teacher (FiC). Under the original order, Stage D would have begun with SnC - which means the C2 -> C3 step would have had to swap from `model_ft`-feature-regression to `snapshot`-contrast, **changing the teacher AND the objective form in one step**. That is precisely the two-changes-one-label error the submitted paper made, sitting on the paper's central claim. Holding the teacher fixed across C2/C3 is what makes "the contrastive form matters" a measurable claim rather than a confounded one.

The cost of the fix is stated honestly rather than hidden: a forward ladder measures each component **conditional on the prefix already installed**, so the order is part of the claim. It is fixed in advance here, and the D4 backward pass is what detects order-sensitivity.

## 5.6 Stage D4 - backward leave-one-out, with renormalisation

Generated rather than hand-written (`manifest.py:411-429`) so that each arm is guaranteed to differ from full BaCP in exactly one $\lambda$.

| id | removes | exact `overrides` | inherits | n | cost_u | priority |
|---|---|---|---|---|---|---|
| **D4-noPrC** | PrC | `{'lambdas': (0.0, 1/3, 1/3, 1/3)}` | D2 | 5 | 2.2 | high |
| **D4-noFiC** | FiC | `{'lambdas': (1/3, 0.0, 1/3, 1/3)}` | D2 | 5 | 2.2 | high |
| **D4-noSnC** | SnC | `{'lambdas': (1/3, 1/3, 0.0, 1/3)}` | D2 | 5 | 2.2 | high |
| **D4-noCE** | CE | `{'lambdas': (1/3, 1/3, 1/3, 0.0)}` | D2 | 5 | 2.2 | high |

### Why renormalisation to $1/3$ is the whole point

The submitted paper's Table 6 zeroed one of four $0.25$ weights and left the other three alone. Write the objective as $\mathcal{L} = \sum_i \lambda_i \mathcal{L}_i$. Under the old scheme, "no PrC" moved from

$$(0.25,\,0.25,\,0.25,\,0.25),\quad \textstyle\sum \lambda = 1.0 \qquad\longrightarrow\qquad (0,\,0.25,\,0.25,\,0.25),\quad \textstyle\sum\lambda = 0.75$$

That is **two interventions under one label**: a component was removed *and* the entire remaining objective was globally downweighted by 25% relative to the effective learning rate on that loss. A drop in accuracy is then unattributable - it could be the missing teacher or it could be that the whole contrastive signal got quieter. Renormalising to

$$(0,\ 1/3,\ 1/3,\ 1/3),\quad \textstyle\sum\lambda = 1.0$$

holds total objective mass fixed and leaves removal as the only change. `test_leave_one_out_arms_renormalise` (`test_manifest.py:134`) enforces it. The same reasoning is why $\lambda_{kd}$ sits outside the simplex (§7.1).

### An open degeneracy in D4

Verified by resolving both configs: **`resolve('D4-noPrC', tier=1)` and `resolve('D1', tier=1)` differ in exactly one key - `experiment_type` (`'ladder-D4'` vs `'ladder-D'`).** Every substantive setting is identical, because $(0, 1/3, 1/3, 1/3)$ with `num_snapshots=2` is exactly what D1 already runs. Three consequences and the prescription: §8.8.

## 5.7 Forward (sufficiency) vs backward (necessity)

Forward measures a component's effect **at the all-others-off corner** - sufficiency. Backward measures it **at the all-others-on corner** - necessity. They agree only under strict additivity, and their gap *is* the aggregate interaction (`manifest.py:424-429`), the highest-information-per-run quantity in the design.

| | backward large | backward $\approx 0$ |
|---|---|---|
| **forward large** | **additive** - the component carries real, non-substitutable signal at both corners; this is what a headline claim needs | **redundant** - the component works, but something already installed does the same job, so removing it costs nothing at the full corner. *Redundant is not useless* |
| **forward $\approx 0$** | **synergistic** - worthless alone, load-bearing in combination; only visible because both passes were run | **inert** - no effect at either corner; the component should be removed from the method |

A leave-one-out star alone (what the submitted paper ran) cannot distinguish **redundant** from **inert**, and cannot see **synergistic** at all: it only ever observes the all-on corner.

### The degrees-of-freedom argument

Let $p$ be the number of components. A pure forward ladder observes the corners

$$\varnothing,\quad \{1\},\quad \{1,2\},\quad \dots,\quad \{1..p\}$$

which is $p+1$ configurations, giving $p$ contrasts. Encode the design matrix with one column per main effect: it is **lower-triangular** - configuration $k$ has ones in columns $1..k$ and zeros after. A $p \times p$ triangular matrix with nonzero diagonal is full rank, so the $p$ main effects are exactly identified and **the residual degrees of freedom are zero**. There is no column left over to load an interaction onto, and no residual to test one against. This is not a power problem that more seeds fix: **no interaction is estimable even in principle**. `manifest.py:15-17` makes the identical argument about the backward star ("with $p$ arms and $p$ main effects it has zero residual degrees of freedom").

Running both passes gives $2p$ contrasts over the same $p$ main effects - $p$ residual degrees of freedom, which is precisely the forward/backward gap column. That is why the ladder runs both and why the backward pass is a second sweep over components that survive the forward one, rather than the primary design (`manifest.py:17-18`). Note that the D4-noPrC degeneracy (§8.8) makes the PrC row of that column **zero by construction rather than by measurement**.

## 5.8 Tiers and cells

A **cell** is one rung at one seed (`cells()`, `manifest.py:591`). Its `key` is written into `BACP_EXPERIMENT_GROUP` and copied onto the record by `training_utils._finalize_run`, which is the entire idempotence mechanism: a cell is complete iff a record exists carrying its key, so resume needs no bookkeeping file to fall out of sync. Key construction and the tier-scoping argument: §3.7.

| tier | name | rungs | seeds | purpose |
|---|---|---|---|---|
| 0 | smoke | all 22 | (1,) | every rung, every code path, on CPU, in minutes. Numbers are meaningless **by construction** and never appear in the paper (§3.3, `manifest.py:549-557`) |
| 1 | spine | A0, A2, A3, A4, A5, C-null, C0, C0b, C1, C2, C3, D1, D2 | 1-5 | the minimum set that makes the paper defensible: the G1 gate, the Stage A 2x2, Stage C in full |
| 2 | attribution | all | 1-5 | adds the backward pass and the rungs A/B left out, completing the forward/backward gap column |
| 3 | full | all | 1-8 | everything at 8 seeds where declared, plus the headline sparsity sweep (declared but unscheduled, §3.4) |

`cells()` caps seeds at `min(rung.n_seeds, len(spec['seeds']))` (`manifest.py:607`), so **the 8-seed declarations on A0/C0/C0b/C1/C2/C3 only materialise at tier 3**; at tiers 1 and 2 every rung runs 5 seeds.

### Cell counts and nominal U (computed from `cells()`)

| tier | runs | total U | A | B | C | D | D4 |
|---|---|---|---|---|---|---|---|
| 0 | 22 | 31.5 | 7 runs / 7.0 U | 3 / 3.0 | 6 / 8.5 | 2 / 4.2 | 4 / 8.8 |
| 1 | 65 | 88.5 | 25 / 25.0 | - | 30 / 42.5 | 10 / 21.0 | - |
| 2 | 108 | 155.5 | 33 / 33.0 | 15 / 15.0 | 30 / 42.5 | 10 / 21.0 | 20 / 44.0 |
| 3 | 126 | 180.7 | 36 / 36.0 | 15 / 15.0 | 45 / 64.7 | 10 / 21.0 | 20 / 44.0 |

Measured throughput, the 1.95x `cost_u` correction for BaCP cells, and both corrected budget estimates are in **§4.3-§4.4**. The caveat that must not be dropped: only **C-null** has been timed on the BaCP path and it is the *cheapest* BaCP rung, so the relative ordering among C0b, C1, C2, C3, D1 and D2 is the manifest's assumption, **not a measurement**.

## 5.9 The gates

Gates are hard stops, not caveats to write around. `evaluate_gates()` (`ladder.py:368`) returns `'pass'`, `'STOP'` or `'pending'` per gate; `report()` prints them (`ladder.py:649-661`) and `run_ladder.py:120` halts on any STOP.

**Honest implementation status.** A grep across the whole repository finds gate identifiers **G1, G3, G4 and G5 only**. The `gate` field on `Rung` is populated on exactly six rungs: C-null->G1, C1->G3, C2->G3, C3->G4, D1->G5, D2->G5. **G0, G0b, G2 and G6 have no definition anywhere in the codebase** - no branch in `evaluate_gates`, no threshold constant, no rung declaring them.

| gate | question | fires when | action on fire | implemented in `evaluate_gates`? | otherwise covered by |
|---|---|---|---|---|---|
| **G0** | does every rung run end-to-end before GPU money is spent | trigger **not stated in any source file**. The duty is discharged by tier 0 (`manifest.py:549-557`) plus `validate(strict=True)` at import (`manifest.py:913`) | not stated | **No** | `test_manifest.py` in full; `test_smoke_and_real_runs_are_not_mixed` (`test_experiment_invariants.py:297`); `test_tier_zero_truncates_the_loaders` (`test_manifest.py:203`) |
| **G0b** | is the noise floor small enough to resolve component effects | $\hat\sigma$ of the dense rung exceeds **2.0 points** - but the 2.0 is a working figure carried in this document alone: **it is written in no file**, and is therefore neither MEASURED nor PLANNED under the legend at the top of this document | fix variance before attempting to measure 1-point effects | **No** - `noise_floor()` (`ladder.py:356`) computes the statistic; nothing compares it to a threshold | `test_each_rung_has_enough_seeds` (`:334`) protects $n$, not $\hat\sigma$. **MEASURED: $\hat\sigma = 0.2975$ on A0, 5 seeds (§3.7)** - but there is no threshold in any file to pass |
| **G1** | does the BaCP code path reproduce plain CE | $\lvert \bar{x}_{\text{C-null}} - \bar{x}_{\text{C0}} \rvert > 0.5$pp, with $\ge 3$ seeds each | STOP. "Implementation bug, not a result" - every downstream number would measure the artefact | **Yes**, `ladder.py:408-424` | `test_g1_*` (`test_ladder_stats.py:184`, `:189`) |
| **G2** | *purpose not stated in any source file* | **not defined** | **not defined** | **No** | nothing. The nearest duty in the repo is the substrate-parity comparison against RigL $85.71\pm0.23$ / EAST $86.99\pm0.32$ named at `manifest.py:46-49`, which no code checks |
| **G3** | does *any* dense-teacher signal help, against the no-teacher control | the better of $\Delta(\text{C1} - \text{C0b})$ and $\Delta(\text{C2} - \text{C0b})$ has a 95% CI **upper** bound below $+0.5$pp | STOP: "No dense-teacher signal helps in this regime, so BaCP has no mechanism to exploit" | **Yes**, `ladder.py:426-455` | `test_g3_*` (`test_ladder_stats.py:151-181`) |
| **G4** | is the contrastive *form* distinguishable from feature regression against the same teacher | the C3 - C2 paired CI **contains zero** | STOP: "that is the paper's central claim" | **Yes**, `ladder.py:457-471` | `test_g4_*` (`test_ladder_stats.py:109-147`) |
| **G5** | are the individual teachers distinguishable | **both** +SnC (D1) and +PrC (D2) deltas have CIs containing zero | STOP: "Do not tune (Stage D-prime) a method whose own components are indistinguishable from each other" | **Yes**, `ladder.py:473-485` | no dedicated test in `test_ladder_stats.py` |

There is also **no ordering enforcement** between gates: `report()` evaluates all four implemented gates unconditionally and reports every STOP, rather than short-circuiting at the first. In practice `run_ladder.py` halts the sweep on any STOP, so the effect is the same, but a G4 STOP will still be printed alongside a G1 STOP that invalidates it.

### The two gate bugs that were found and fixed

Both are documented in `evaluate_gates`'s own docstring (`ladder.py:376-392`) and both would have let a null through.

**G4 was a non-inferiority test wearing an indistinguishability message.** It was written as `lo > -0.5`. A *perfect null* - C3 scoring exactly what C2 scores on every seed - gives a paired CI of $[0.00, 0.00]$, and $0.00 > -0.5$ is true, so the gate returned the **highest possible PASS** for the exact case it was built to catch. The gate guarding the paper's central claim could not fire. It now STOPs when the interval contains zero and passes only when the whole interval is above zero (`ladder.py:463-471`). `test_g4_stops_on_a_perfect_null` (`test_ladder_stats.py:109`) pins the fixed behaviour.

**G3 read the wrong comparator.** It took C2's ladder delta, which `aggregate()` computes against C2's declared parent **C1** - and C1 and C2 deliberately hold the teacher fixed, so that increment is *feature-KD vs logit-KD*, not *teacher vs no-teacher*. The worked example in the docstring: with C0b = 87.4, C1 = 84.4, C2 = 87.4, the true best teacher effect is $+0.00$pp and G3 printed `pass ... +3.00pp`. It now forms both teacher contrasts explicitly against C0b via `paired_delta` (`ladder.py:438-441`). This is why `per_seed_scores` was factored out of `aggregate` and shared with `evaluate_gates` (`ladder.py:277-292`) - so a gate can construct any contrast it needs rather than being limited to the ladder's incremental deltas - and why G3 returns `'pending'` rather than falling back when `scores` is not supplied (`ladder.py:429-436`).

**The gate bug that is still OPEN: G1 has no collapse floor.** See §8.2. It is not hypothetical - it is the state the first tier-1 attempt was actually in.

## 5.10 Statistical design

Formulas, critical values and the MDE table live in **§7.12**. This section states the design decisions.

### Paired vs unpaired, per stage

| stage | design | estimator | why |
|---|---|---|---|
| A | **unpaired** | `unpaired_delta`, Welch (`ladder.py:225`) | the mask lottery dominates run-to-run variance, so seed correlation between arms is low and pairing buys little (`manifest.py:177-179`) |
| B, C, D, D4 | **paired on seed** | `paired_delta` (`ladder.py:201`) | same substrate, same mask-trajectory family; the intervention is architectural or objective-level, so seeds correlate and pairing pays |

`paired_delta` takes the **intersection** of seeds present in both arms rather than assuming alignment, and reports `n_pairs` so a reader can see when pairing silently degraded (`ladder.py:207-211`). `test_paired_delta_on_disjoint_seeds_is_empty_not_wrong` (`test_ladder_stats.py:242`) pins the disjoint case.

### $n=3$ is not a sample size

At $n=3$ paired the MDE is about $3.1\,\sigma_d$ (`ladder.py:22-24`). A ladder reporting 1-point component effects at $n=3$ is reporting noise. Enforced two ways: `test_each_rung_has_enough_seeds` (`test_experiment_invariants.py:334`) fails any rung below 3 seeds, and `evaluate_gates(..., min_n=3)` returns `'pending'` rather than a verdict below the floor (`ladder.py:412-415`, `:461-462`). The manifest defaults to 5 and to 8 on the Stage C rungs carrying the claim. **A1 sits exactly at the floor at $n=3$** (`manifest.py:202`). Its `priority='optional'` is **not** given for that reason: the manifest's own note (`manifest.py:208-209`) reads "The floor. Optional only because A2-A4 pin the 2x2 corners that matter; if budget is tight this is the first Stage A rung to cut." That is a budget-and-design reason, stated as the *only* one. The two facts sit next to each other and should not be joined - a rung at the power floor that is also the first cut is worth flagging twice over, but the manifest attributes the priority to coverage, not to power.

### Multiplicity: fixed-sequence testing, no $\alpha$ penalty

The ladder is tested as a **fixed sequence** at $\alpha = 0.05$: rung $k$ is tested only if rung $k-1$ was significant, and testing stops at the first non-significant step. A fixed-sequence (hierarchical) procedure controls the family-wise error rate at $\alpha$ **without any per-test penalty**, because the ordering is pre-registered rather than chosen after seeing the data (R23, §6.2). This is what the gates operationalise: G1 -> G3 -> G4 -> G5 is the sequence, and a STOP halts the ladder by design (`ladder.py:656-661`).

The design's discipline is what buys this: the order must be fixed in advance, which is exactly why the FiC -> SnC -> PrC re-ordering (§5.5) had to happen **before any run** and is recorded in the file rather than in a lab notebook.

### The noise floor and the bolding rule

`noise_floor(agg)` is the **median within-rung standard deviation** (`ladder.py:356-363`) - the resolution limit of the table. Three rules are enforced in code rather than left to discipline (`ladder.py:7-19`):

1. **The floor is printed in every caption**, and no delta smaller than it is bolded or starred. `to_latex` bolds only when $\lvert\Delta\rvert > \text{floor}$ (`ladder.py:578-583`); `to_text` marks `*` above the floor and `~` below, with the legend "a `~` is NOT a result" (`ladder.py:533-546`). The submitted paper's ablation deltas were all under 0.4 points with **no error bars at all**.
2. **Missing is visible.** A planned rung with no record renders `--`; a crashed run renders `--` *and* appears in the failures list. `_fmt` (`ladder.py:491`) maps `None`, `NaN` and pandas `NA` all to the dash, because a LaTeX table containing `nan` claims a number was computed and came out undefined - a different and worse claim than "not run."
3. **$n \le 5$ gets no stars.** With five numbers, the table prints the five numbers (`values` column, `ladder.py:324`).

### A null must be publishable

TOST (`tost`, `ladder.py:261`) declares equivalence when the 90% CI on the difference lies strictly inside a pre-registered $\pm 1.0$pp bound. Stage B is designed around it (`manifest.py:266-269`).

## 5.11 Experiment-level invariants

`project/tests/test_experiment_invariants.py` runs against `results/runs/*.json`, not against code. Its own framing (`:1-20`): the rest of the suite would pass on a method that has no effect whatsoever. These ask a different question - *are the numbers a table would print comparable to each other and to the literature?* Marked `results`, excluded from the default run, invoked as `pytest -m results -s`.

Fixtures narrow the scope deliberately: `records` (everything on disk) -> `ladder_records` (carries an `experiment_group` and status `ok`) -> `real_records` (excludes tier-0 smoke). Ad-hoc runs and test fixtures also write records and must not be held to comparability invariants.

| test (`:line`) | scope | what it protects |
|---|---|---|
| `test_schema_version_matches` (`:83`) | all | a record predating the current schema may be missing fields a table reads |
| `test_every_run_records_its_seed` (`:92`) | ladder | the seed used to be popped from args before the dataclass was built, so **no published number was reproducible after the fact** |
| `test_provenance_is_recoverable` (`:100`) | ladder | a HEAD sha does not identify the code that ran when the tree is dirty; a dirty run must carry `git_patch_sha1`, and every run a `code_fingerprint` |
| `test_all_comparable_runs_share_one_code_fingerprint` (`:112`) | real | two arms built from different source cannot be compared. Names the split rather than tolerating it - mid-sweep code changes are the most common way a ladder quietly stops being a ladder |
| `test_no_evaluation_samples_were_dropped` (`:131`) | ladder | the `drop_last` defect, §2.7. Every headline number in the submitted paper carries this error |
| `test_metrics_are_in_range` (`:146`) | ladder | accuracies in $[0,100]$, sparsities in $[0,1]$, and **NaN is an explicit failure** |
| `test_exact_and_legacy_accuracy_agree` (`:159`) | ladder | >2pp between `evaluate_exact` and `evaluate` means the two are reading different loaders |
| `test_sparsity_target_was_attained` (`:180`) | ladder | requested vs achieved mask sparsity within 0.01 - 0.05 for EAST, whose cyclic schedule oscillates density by design. Compares against `sparsity_target_adjusted` when weight sharing rewrote the target |
| `test_value_sparsity_never_understates_mask_sparsity` (`:201`) | ladder | RigL/EAST zero-initialise regrown connections, so a structurally *active* weight reads as numerically zero and value-sparsity **overstates** - measured 0.9277 against a true 0.8976 (§7.11). Overstating is the flattering direction, so the inequality must hold in exactly one direction |
| `test_mask_covers_the_prunable_set` (`:217`) | ladder | if the mask spans a different parameter set than `layer_check` calls prunable, the reported sparsity describes neither |
| `test_active_parameter_count_matches_the_sparsity` (`:225`) | ladder | $1 - \text{active}/n$ must reproduce the reported sparsity to 0.02 |
| `test_field_is_single_valued_across_the_whole_ladder` (`:252`) | real | `scope_key`, `val_split_seed`, `model_name`, `dataset_name`. **Three sparsity denominators give three numbers for one checkpoint** (§7.11); if `scope_key` varies, the column headed "sparsity" means different things in different rows |
| `test_protocol_is_identical_across_the_ladder` (`:266`) | real | `epochs`, `batch_size`, `learning_rate`, `optimizer_type`, `scheduler_type`. Turns "at matched protocol" from a sentence into a fact - drift here confounds every delta with a training-budget difference |
| `test_architecture_is_stable` (`:280`) | real | same model name must mean the same `total_params`. Weight-sharing runs excluded, which is the point: `shared_params_unique` exists so that change is visible rather than reading as architecture drift |
| `test_smoke_and_real_runs_are_not_mixed` (`:297`) | ladder | a tier-0 row is not a measurement; if both share a table, one is being read as something it is not |
| `test_sparse_runs_are_chained_to_a_dense_checkpoint` (`:307`) | real | a sparse run with no `trained_weights` started from random or ImageNet weights and is not comparable to the rungs that did |
| `test_repeated_seeds_within_a_rung_are_not_duplicates` (`:318`) | real | two records for one (rung, seed) means the aggregate averages a number with itself and reports a standard deviation that is too small |
| `test_each_rung_has_enough_seeds` (`:334`) | real | the $n=3$ floor, with the MDE argument in the docstring |
| `test_dense_baseline_is_within_reach_of_published` (`:352`) | real | the attack it answers: dense baselines in the submitted paper sit **1.7 to 4.9 points below** published figures for the same architecture and dataset, so every "matches or exceeds dense" claim reads as "exceeds our undertrained dense". Reference values live in `results/reference/dense_reference.json`, not in code, so a tolerance change shows up as a data diff. Skips, not fails, when absent - which is the anchor's case (§6.5) |
| `test_run_status_ledger` (`:390`) | all | **not an assertion - a dashboard.** `pytest -m results -s` prints record counts, failures with their error strings, a smoke warning, and seeds-per-rung with an under-seeded flag |
| `test_checkpoint_paths_are_absolute` (`:435`) | ladder | a path in a record must not depend on who reads it or from where |
| `test_recorded_checkpoints_exist_on_disk` (`:453`) | ladder | a record pointing at a moved checkpoint is worse than no record: every downstream rung chains off it |
| `test_sparse_runs_chain_to_a_checkpoint_of_the_same_seed` (`:468`) | real | **pairing must hold all the way down to initialisation.** The runner permits a fallback to another seed's checkpoint with a warning; when that happens the rung is no longer seed-paired and every paired CI computed from it is overstated (§8.4). The warning is easy to miss in a 65-cell sweep; this is not |

The last three exist because of a real tier-0 failure (`:421-433`): the dense rung succeeded and its checkpoint sat on disk while **all 21 sparse rungs failed to find it**. `save_path` was recorded relative to the CWD, the runner launches each script with `cwd=project/scripts`, and `resolve_checkpoint` evaluated that path from the repo root. Nothing raised at record time; the symptom appeared one cell later as a missing baseline. The unit suite could not have caught it - the defect exists only in the relationship between two processes' working directories.

## 5.12 Execution

`run_grid` (`project/experiments/runner.py:276`) orders cells by stage, then rung, then seed, "which is also the dependency order: dense before sparse, and the gate rungs (C-null, C0) before anything that is measured against them" (`runner.py:281-283`). Resume is by completed cell key, so re-invoking the same tier runs only what is missing. `stop_on_fail` is available but off by default; a cell that exits 0 without writing a record is reported as `ok_no_record` rather than counted as success.

`attach_checkpoint` (`runner.py:145`) is what binds a sparse cell to its dense predecessor; its provenance gap is §8.4.

`run_ladder.py` is the single entry point: run the grid, aggregate the records, write `.tex`/`.csv`/`.txt` under `results/tables/`, evaluate the gates, and halt on a STOP. Every table carries a provenance header (`ladder.py:618`) with UTC timestamp, git sha + branch + dirty flag, code fingerprint, and planned/ok/failed/missing counts - plus a loud banner if any row is a tier-0 smoke record.

---

# 6. Citations

## 6.1 How to read this section

The organising rule from `docs/citations.md` is that **a wrong citation is worse than none**. A reviewer who checks one appendix and finds it says the opposite of what was claimed discounts the rest of the paper. **§6.3 matters more than §6.2.**

**How many mis-citation flags there are, and why the answer is not one number.** `docs/citations.md` is internally inconsistent on its own count: its opening paragraph says "**Three** such cases were found and are flagged below", while its closing *Summary of mis-citation flags* table lists **seven** rows. The seven-row table is the one that enumerates them and is the one reproduced as §6.3 below, so **seven** is the number this document works from - but the discrepancy is in the dossier and should be resolved there rather than propagated silently. An eighth flag (F8, ERK fidelity) is added here from the current session's parameter arithmetic and has no counterpart in the dossier.

Status legend: **OK** = verified against the primary source, as recorded in `docs/citations.md`; **RISK** = mis-citation risk; **CHOICE** = design choice with no canonical source; **REJECTED** = checked and rejected as a source; **UNVERIFIED** = cited in this document but **absent from `docs/citations.md`**, so no check against the primary source has been recorded by this project. **OK** is a claim about the dossier, not a general assertion of correctness, and a reference the dossier does not mention cannot carry it.

**Two disciplines this table holds itself to.**

1. Where the dossier does not record an arXiv identifier, this table says *not stated in the source* rather than supplying one from memory.
2. The same applies to author lists and titles. An entry whose bibliographic detail is **fuller than the dossier's own record** is marked **°** - the facts may well be right, but they were not read off the dossier and must be checked against the primary source before they enter a `.bib`.

## 6.2 Complete reference table

| # | Authors | Title | Venue / year | arXiv | Cited **for** | Status |
|---|---|---|---|---|---|---|
| R1 | Sun, Liu, Bair, Kolter | A Simple and Effective Pruning Approach for Large Language Models (WANDA) | ICLR 2024, Vienna | 2306.11695 | The importance metric $S_{ij}=\lvert W_{ij}\rvert\cdot\lVert X_j\rVert_2$ **only**. Not the comparison group (F1); not the Conv2d convention. | OK metric / RISK grouping |
| R2 | Mocanu, Mocanu, Stone, Nguyen, Gibescu, Liotta | Scalable training of ANNs with adaptive sparse connectivity inspired by network science (SET) | Nature Communications 9, 2383 (2018) | not stated in the source | Original Erdos-Renyi layer-wise allocation; also the SET column of EAST Table 1. | OK |
| R3 | Evci, Gale, Menick, Castro, Elsen | Rigging the Lottery: Making All Tickets Winners (RigL) | ICML 2020 | 1911.11134 | ERK kernel extension; drop-and-regrow; **zero-initialised** regrowth and its stated reason; cosine drop-fraction anneal; the `prune_last_layer=True` / `fc1000` ERK-0.957 convention (§6.6). | OK |
| R4 | Hinton, Vinyals, Dean | Distilling the Knowledge in a Neural Network | NIPS 2014 Deep Learning Workshop | 1503.02531 | Soft targets and the $T^2$ rescaling **only**. Not the KL form or its direction (F2). | OK $T^2$ / RISK KL |
| R5 | Romero, Ballas, Ebrahimi Kahou, Chassang, Gatta, Bengio | FitNets: Hints for Thin Deep Nets | ICLR 2015 | 1412.6550 | Squared-Euclidean hint loss on *intermediate* activations through a *learned regressor*. Cite only for that. Not for cosine-on-penultimate (F3). | OK narrow / RISK |
| R6 | Tung, Mori | Similarity-Preserving Knowledge Distillation (SP) | ICCV 2019 | not stated in the source | Listed as a **rejected** alternative for F3: squared Frobenius norm on $b\times b$ row-wise similarity matrices - relational. | REJECTED for F3 |
| R7 | Park, Kim, Lu, Cho **°** | Relational Knowledge Distillation (RKD) | CVPR 2019 | not stated in the source | Rejected alternative for F3: distance and angle *between samples* - relational. | REJECTED for F3 |
| R8 | Passalis, Tefas | Probabilistic Knowledge Transfer (PKT) | not stated in the source | not stated in the source | Rejected alternative for F3: cosine-**kernel** probability distribution matched by KL, not feature-to-feature cosine. | REJECTED for F3 |
| R9 | Tian, Krishnan, Isola | Contrastive Representation Distillation (CRD) | ICLR 2020 | 1910.10699 - **cite v3** ("Typo fixed in the newest version") | Table 2 cross-architecture benchmark (vgg13 to MobileNetV2: AT 59.40, NST 58.16, vanilla 64.6, CRD 69.73). **Not** a mechanism attribution (F4). | OK tables / RISK mechanism |
| R10 | Zhu, Gupta | To prune, or not to prune: exploring the efficacy of pruning for model compression | ICLR 2018 Workshop track | 1710.01878 | The cubic sparsity ramp. The source defines it over **training steps**; our monotone pruners apply it epoch-wise (§7.8). | OK with stated deviation |
| R11 | Wang, Isola | Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere | ICML 2020 | 2005.10242 | Theorem 1's alignment/uniformity decomposition and the identity $\lVert u-v\rVert^2=2-2u^\top v$ on unit vectors. **Asymptotic**; never mentions distillation (F5). | OK decomposition / RISK scope |
| R12 | Grill et al. | Bootstrap Your Own Latent (BYOL) | NeurIPS 2020 | 2006.07733 | Eq. 6-7: InfoNCE with a $\beta$-weighted log-sum-exp denominator and cosine similarity, and the authors' own sentence recovering BYOL at $\beta=0$. Table 5(b) bottom row: $\beta=0$, no predictor, no target network gives **0.1%** ImageNet top-1 vs SimCLR **69.4%** at $\beta=1$. | OK (closest published statement) |
| R13 | Chen, He **°** | Exploring Simple Siamese Representation Learning (SimSiam) **°** | not stated in the source | not stated in the source | **Do not cite** for the denominator-deletion framing: "InfoNCE" appears **zero** times; the paper frames itself only as "SimCLR without negative pairs". | REJECTED |
| R14 | Li, Durrant, Markovic, **Huang, Kundu, Chen**, Yin, Leontidis | Pushing the Limits of Sparsity: A Bag of Tricks for Extreme Pruning (EAST) | **TMLR, volume 2025** - record `journals/tmlr/LiDMHKCYL25` | 2411.13545 - **cite v4 for numbers** | **Verified in the dossier (§9):** all external sparse baselines (§6.5); the 250-epoch / wd 1e-4 / ERK / $\Delta T{=}4000$ protocol; the author-list and version hazards; the two things the paper does **not** state. **Not covered by the dossier:** DyReLU phasing and weight sharing - EAST is the mechanisms' origin, but `docs/citations.md` verifies neither against the paper, so §7.14's phasing formula and §1.6's sharing carry **no checked citation** and must not be attributed to R14 on this table's authority. | OK with hazards (F6, §6.4) for the verified half; **UNVERIFIED** for DyReLU phasing and weight sharing |
| R15 | Xu, Luo, Wang, Chang, Huang, Huang, Huang | From Dense to Sparse: Contrastive Pruning for Better Pre-trained Language Model Compression (CAP) | AAAI 2022, vol. 36, pp. 11547-11555 | 2112.07198 | **The direct antecedent.** PrC/FiC/SnC decomposition, Eq. 1, Table 1's positive sets, the memory bank ($N=4096$), the head-free `[CLS]` contrast. See §6.7. | OK |
| R16 | Khosla, Teterwak, Wang, Sarna, Tian, Isola, Maschinot, Liu, Krishnan **°** | Supervised Contrastive Learning (SupCon) | NeurIPS 2020 | 2004.11362 | $L_{out}$ vs $L_{in}$ and $L_{out}$'s superiority; §3's observation that SupCon reduces to NT-Xent at $\lvert P(i)\rvert=1$. | OK |
| R17 | Chen, Kornblith, Norouzi, Hinton | A Simple Framework for Contrastive Learning of Visual Representations (SimCLR) | ICML 2020 | 2002.05709 | NT-Xent. | OK |
| R18 | Blalock, Gonzalez Ortiz, Frankle, Guttag | What is the State of Neural Network Pruning? | MLSys 2020 | 2003.03033 | Methodology: pruning papers are not mutually comparable - protocol drift, missing baselines. | OK |
| R19 | Bouthillier et al. | Accounting for Variance in Machine Learning Benchmarks | MLSys 2021 | 2103.03098 | Seed variance; one seed is insufficient. | OK |
| R20 | Picard | `torch.manual_seed(3407)` is all you need | - | 2109.08203 | Measured seed spread on CIFAR-10. | OK |
| R21 | Dodge, Gururangan, Card, Schwartz, Smith | Show Your Work | EMNLP 2019 | 1909.03004 | The expected-max-vs-budget **estimator**. RISK: their case is max over a *hyperparameter search*; ours is best-epoch-of-a-single-run. Adjacent, not identical - argue in our own terms. | OK estimator / RISK case |
| R22 | Schuirmann (1987); Lakens (2017), *Social Psychological and Personality Science* | TOST equivalence testing | - | - | Makes a null result publishable; Stage B is designed around it. | OK |
| R23 | Westfall, Krishen (2001); Maurer, Hothorn, Lehmacher (1995) | Fixed-sequence testing | - | - | Testing the ladder in a pre-registered order at full $\alpha$ with no multiplicity correction. | OK |
| R24 | Tanaka, Kunin, Yamins, Ganguli | Pruning neural networks without any data by iteratively conserving synaptic flow (SynFlow) | NeurIPS 2020 | 2006.05467 | Layer collapse, maximal critical compression, iterative-vs-single-shot. Relevant to any layer-protection decision at 99.9%+; also the SynFlow column of EAST Table 1. | OK |
| R25 | Liu, Chen, Chen, Atashgahi, Yin, Kou, Shen, Pechenizkiy, Wang, Mocanu | Sparse Training via Boosting Pruning Plasticity with Neuroregeneration (GraNet) | NeurIPS 2021 | 2106.10404 | Published **dense** baselines (§6.5); the "every 2-D/4-D tensor is prunable" convention (§6.6). | OK |
| R26 | GraSP - authors not stated in the source | not stated in the source | not stated in the source | not stated in the source | Independent cross-check of GraNet's 24 shared sparse cells. RISK: disagrees with GraNet on VGG-19 (94.23 / 74.16), gives no std devs, and its ResNet-32 rows are **2x-width**. | RISK |
| R27 | He, Fan, Wu, Xie, Girshick **°** | Momentum Contrast (MoCo) **°** | CVPR 2020 **°** | not recorded in the dossier | Contrast only: MoCo's queue tolerates key-encoder drift; CAP's bank is exact because its teachers are frozen. | **UNVERIFIED** - "MoCo" does not occur in `docs/citations.md` |
| R28 | Dasgupta, Gupta **°** | An elementary proof of a theorem of Johnson and Lindenstrauss **°** | Random Structures & Algorithms 22(1), 60-65 (2003) **°** | not recorded in the dossier | $O(1/\sqrt d)$ inner-product distortion of a random projection to $d=128$ - why the never-trained `encoder_head` was **lossy rather than vacuous** (§7.14). ⚠ **Not** for "the head's distortion cancels on both sides of the loss" (§1.4): JL bounds one projection's distortion and says nothing about cancellation across two arms of a contrastive loss. That argument is the project's own and is derived, not cited. | **UNVERIFIED** - neither "Dasgupta" nor "Johnson" occurs in `docs/citations.md` |
| R29 | Chen, Dai, Liu, Chen, Yuan, Liu **°** | Dynamic ReLU **°** | ECCV 2020 **°** | not recorded in the dossier | DyReLU-B: $\mathrm{out}_c=\max_k(a_c^k x_c+b_c^k)$ with per-sample $(a,b)$ from a hyperfunction (§1.5, §7.14). | **UNVERIFIED** - neither "DyReLU" nor "Dynamic ReLU" occurs in `docs/citations.md`, and it underwrites §7.14's DyReLU-B formula |

### The gap R1-R29 left: architectures, datasets and one statistical method

R1-R29 was assembled around the *method* citations and carries **no entry at all** for any backbone this project builds, any dataset it registers, or the Welch-Satterthwaite approximation §7.12 computes. That is a real omission - a paper reporting ResNet-34 on CIFAR-10 must cite both - and it is filled here rather than left to the `.bib`. **Every row below is `UNVERIFIED`:** none of these appears anywhere in `docs/citations.md`, so nothing has been checked against a primary source by this project, and every bibliographic field is marked **°** for the same reason. arXiv identifiers are deliberately **not** supplied, per §6.1's first discipline.

| # | Authors | Title | Venue / year | arXiv | Cited **for** | Status |
|---|---|---|---|---|---|---|
| R30 | He, Zhang, Ren, Sun **°** | Deep Residual Learning for Image Recognition (ResNet) **°** | CVPR 2016 **°** | not recorded in the dossier | The anchor backbone: `ResNet(BasicBlock,[3,4,6,3])` = ResNet-34, and the `BasicBlock`/`Bottleneck` construction (§1.3, `models/resnet.py`). **Not** for the CIFAR stem - the 3x3 stride-1 adaptation is a convention, and defect M1 (§1.3.3) is ours | **UNVERIFIED** |
| R31 | Simonyan, Zisserman **°** | Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG) **°** | ICLR 2015 **°** | not recorded in the dossier | Configurations `"A"` (`vgg11`) and `"E"` (`vgg19`) (§1.7). Note GraNet's and GraSP's VGG-19 rows (§6.5) are a *different* CIFAR-adapted variant and are not interchangeable with these | **UNVERIFIED** |
| R32 | Zagoruyko, Komodakis **°** | Wide Residual Networks **°** | BMVC 2016 **°** | not recorded in the dossier | The WRN family that `WideResNet22` / `wrrn22` (`models/resnet.py:368`, `:414`) is presumably drawn from. ⚠ **Which published variant it corresponds to remains open** - see §1.2: the code passes `n_groups` into the `n_blocks` slot and hardcodes channels `[start, 96, 192, 384]`, and commit `3333f03` calls it "official WideResNet22" with no citation in the source. Citing R32 for it would assert a correspondence nobody has checked | **UNVERIFIED**, and the mapping is unresolved |
| R33 | Dosovitskiy et al. **°** | An Image is Worth 16x16 Words (ViT) **°** | ICLR 2021 **°** | not recorded in the dossier | The `vit-tiny` / `vit-small` entries (§1.1). Neither has been run (§1.7) | **UNVERIFIED** |
| R34 | Devlin, Chang, Lee, Toutanova **°** | BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding **°** | NAACL 2019 **°** | not recorded in the dossier | The masked-LM objective and the `[CLS]` pooling both text paths use (§1.3.4, §2.9) | **UNVERIFIED** |
| R35 | Sanh, Debut, Chaumond, Wolf **°** | DistilBERT, a distilled version of BERT **°** | NeurIPS 2019 EMC^2 workshop **°** | not recorded in the dossier | The `distilbert-base-uncased` and `-mlm` entries. Relevant twice over: it is itself a distillation method, and its tied embeddings drive the §7.11 warning | **UNVERIFIED** |
| R36 | Liu, Ott, Goyal, Du, Joshi, Chen, Levy, Lewis, Zettlemoyer, Stoyanov **°** | RoBERTa: A Robustly Optimized BERT Pretraining Approach **°** | 2019 **°** | not recorded in the dossier | The `roberta-base` and `-mlm` entries; its `<s>` token is the pooled feature (§1.3.4) | **UNVERIFIED** |
| R37 | Krizhevsky **°** | Learning Multiple Layers of Features from Tiny Images (CIFAR-10 / CIFAR-100) **°** | Technical report, 2009 **°** | not recorded in the dossier | **The anchor dataset** (§2.1, §2.2) and `cifar100` | **UNVERIFIED** |
| R38 | Netzer, Wang, Coates, Bissacco, Wu, Ng **°** | Reading Digits in Natural Images with Unsupervised Feature Learning (SVHN) **°** | NIPS 2011 workshop **°** | not recorded in the dossier | `svhn` (§2.2, never run) | **UNVERIFIED** |
| R39 | LeCun, Bottou, Bengio, Haffner **°** | Gradient-Based Learning Applied to Document Recognition (MNIST) **°** | Proceedings of the IEEE, 1998 **°** | not recorded in the dossier | `mnist` (§2.2, never run) | **UNVERIFIED** |
| R40 | Xiao, Rasul, Vollgraf **°** | Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms **°** | 2017 **°** | not recorded in the dossier | `fmnist` (§2.2, never run) | **UNVERIFIED** |
| R41 | Cohen, Afshar, Tapson, van Schaik **°** | EMNIST: an extension of MNIST to handwritten letters **°** | IJCNN 2017 **°** | not recorded in the dossier | `emnist`, `split='balanced'`, 47 classes (§2.2). Carries defect D2 (§8.7) | **UNVERIFIED** |
| R42 | Deng, Dong, Socher, Li, Li, Fei-Fei **°**; Russakovsky et al. **°** for ILSVRC | ImageNet: A Large-Scale Hierarchical Image Database **°**; ImageNet Large Scale Visual Recognition Challenge **°** | CVPR 2009 **°**; IJCV 2015 **°** | not recorded in the dossier | `imagenet` (§2.2), and the source of the `/dbfs` checkpoints that never resolve (§1.1). Carries defect D3 (§8.7) | **UNVERIFIED** |
| R43 | Socher, Perelygin, Wu, Chuang, Manning, Ng, Potts **°** | Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank (SST) **°** | EMNLP 2013 **°** | not recorded in the dossier | `sst2`'s underlying corpus (§2.2, §2.9) | **UNVERIFIED** |
| R44 | Wang, Singh, Michael, Hill, Levy, Bowman **°** | GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding **°** | ICLR 2019 **°** | not recorded in the dossier | The SST-2 **split** actually loaded, and the reason `testloader is valloader`: GLUE's test labels are withheld (§2.9) | **UNVERIFIED** |
| R45 | Merity, Xiong, Bradbury, Socher **°** | Pointer Sentinel Mixture Models (WikiText-2) **°** | ICLR 2017 **°** | not recorded in the dossier | `wikitext2` (§2.2, §2.9). Currently unreachable - defect D1 (§8.7) | **UNVERIFIED** |
| R46 | Satterthwaite (1946) **°**; Welch (1947) **°** | An Approximate Distribution of Estimates of Variance Components; The generalization of "Student's" problem when several different population variances are involved **°** | Biometrics Bulletin; Biometrika **°** | - | The **Welch-Satterthwaite** approximate degrees of freedom $\nu$ in `unpaired_delta` (§7.12), which Stage A's every interval depends on | **UNVERIFIED** |

Two further formulas in §7.12 - the $\chi^2$ interval on $\hat\sigma$ and the two-$t$ MDE - are standard results with **no single canonical source**, and none is recorded in the dossier or invented here; they are labelled as such at their use sites rather than given a row.

## 6.3 The mis-citation flags - the part that matters

| # | The tempting citation | Why it fails | What to do instead |
|---|---|---|---|
| **F1** | WANDA -> per-output-row ranking on CIFAR | WANDA's own **Appendix A** runs the identical comparison on image classifiers and reports the **opposite**: layer-wise is *slightly better* for both ConvNeXt-B and DeiT-B, and the paper says the per-output observation "might be unique to LLMs". That text is in the ICLR 2024 camera-ready, not a preprint artefact. Two further scope limits: its vision runs use 4096 calibration images at **70-80%** sparsity, nowhere near 99%+. | Cite for the metric $\lvert W\rvert\cdot\lVert X\rVert_2$ only. Recommended wording: *"WANDA reports the per-output comparison group as crucial for LLMs but explicitly finds it does not transfer to image classifiers; we adopt per-output ranking as a design choice, not on WANDA's authority."* The code makes it a recorded knob `--wanda_group {output,layer}` (`project/pruning_factory.py:489`) so the paper can report which was used. |
| **F2** | Hinton -> the KL form and its direction | Hinton et al. §2 defines the objective purely as **cross-entropy with soft targets**. "Kullback" and "relative entropy" appear **zero** times; "KL" appears only in §5.4, on ensembles of specialists. Separately, $1/T^2$ is a **high-temperature approximation**, not an identity: the exact gradient is $\partial C/\partial z_i=(1/T)(q_i-p_i)$ (Eq. 2), and the $1/T^2$ form needs Eq. 3's high-$T$ regime plus Eq. 4's per-case zero-meaned logits. | Cite Hinton for $T^2$ and soft targets. Cite **implementations** (RepDistiller, mmrazor, the PyTorch tutorial, HuggingFace) for `F.kl_div(log_softmax(s/T), softmax(t/T))` and for `batchmean`. Never write "*exactly* $1/T^2$", "at all temperatures", or present $T^2$ as an exact normaliser. |
| **F3** | FitNets -> cosine penultimate matching | FitNets is squared Euclidean, on **intermediate** activations, through a **learned regressor**, in unnormalised space; its second stage operates on softened logits. It contains no cosine similarity and no norm-mismatch discussion. The usual alternatives do not carry it either: SP and RKD are **relational** (sample-to-sample), PKT builds a cosine-**kernel** distribution matched by KL, CRD normalises for its NCE critic but argues from mutual information. | State the metric *and* its rationale as this paper's own design choice (§7.4). One candidate surfaced during the search had all three supporting claims refuted under adversarial vote and must not be cited on the strength of the dossier. |
| **F4** | CRD -> "the negatives cause the gain" | CRD's only ablation (Table 6) varies **two** things. The **sampling policy** is the larger and only quantified effect (+0.81% for their objective, +0.62% for InfoNCE). The **objective form** is nearly a wash: their bound wins 4 of 5 pairs, every gap $\le 0.33$ points, and it *loses* on resnet110/resnet32 (73.48 vs 73.53). The mutual-information story is motivation (§1, §3.1) and is never isolated by an experiment. L2 normalisation is stated once as an implementation detail and never ablated. | Cite the benchmark tables (Table 2), not a mechanism. Also state that Tables 1-2 are a cross-method benchmark, not a controlled ablation - methods differ in *which layers they act on*, which is exactly the confound rungs C1/C2/C3 remove. **Uncitable:** the $N\in\{16,\dots,16384\}$ sweep exists only as Figure 5(a), a plot; the sole published numeric fact is "the difference of error rate between N=4096 and N=16384 is less than 0.1%". No per-$N$ accuracies exist anywhere. |
| **F5** | Wang & Isola -> "cosine is InfoNCE without the denominator" | Their Theorem 1 is **asymptotic** (batch to infinity, $O(M^{-1/2})$ deviation), so it does not license "exactly" at finite batch size, and the paper is a pure self-supervised analysis: it never mentions distillation, teachers, students, hint losses or pruning. | Cite Wang & Isola for the **decomposition** and the identity $\lVert u-v\rVert^2=2-2u^\top v$. Present the distillation reading as a **one-line derivation of our own** (§7.5). BYOL Eq. 6-7 is the closer published statement - authors' own words, finite batch. Never write "as shown by [Wang & Isola]" for the distillation reading. |
| **F6** | EAST via the DBLP **CoRR** record | `journals/corr/abs-2411-13545` is **frozen at the five-author v1**. arXiv v1 and v2 list five authors (Li, Durrant, Markovic, Yin, Leontidis); v3 onward lists all eight. A `.bib` auto-generated from DBLP's CoRR entry is simply wrong. | Use the **TMLR** record `journals/tmlr/LiDMHKCYL25`, volume 2025. Cite **v4** for every number (§6.4). |
| **F7** | Any dense ResNet-34 / CIFAR-10 figure | **None is published.** GraNet covers ResNet-20/50 and VGG-19; EAST reports no dense row at all. | State it as a limitation. `results/reference/dense_reference.json` records `"source": "NOT FOUND"` so the parity gate **skips** that cell rather than silently passing it (§6.5). |
| **F8** *(new, this session)* | Citing Evci et al. for **ERK** while the code implements ER-on-the-unrolled-matrix | `calculate_erk_densities` (`project/pruning_factory.py:196-203`) uses layer mass $n_{in}k_h k_w + n_{out}$ - Erdos-Renyi over the **unrolled** weight matrix. Evci et al.'s ERK is $n^{l-1}+n^{l}+w^{l}+h^{l}$: the kernel dimensions are **added**, not multiplied into $n_{in}$. The two coincide for `Linear` layers and diverge for every convolution. Full arithmetic and the consequence for `conv1`: §7.7. | Either (a) implement Evci's mass and re-derive, or (b) keep the code and cite it as *"Erdos-Renyi allocation over the unrolled weight matrix (a variant of the ERK of Evci et al. 2020, which adds rather than multiplies the kernel extent)"*. Do **not** write "we use ERK [Evci et al. 2020]" unqualified. **Currently unfixed** (§8.3). |

## 6.4 The EAST citation hazard, in full

Three separate hazards attach to a single reference, and they compound.

**(a) Author list.** arXiv v1 and v2 carry **five** authors: Li, Durrant, Markovic, Yin, Leontidis. v3 onward carries **eight**: Li, Durrant, Markovic, Huang, Kundu, Chen, Yin, Leontidis. DBLP's CoRR record `journals/corr/abs-2411-13545` is still frozen at the five-author v1, so any `.bib` entry pulled from it omits three authors. **Use `journals/tmlr/LiDMHKCYL25`** - TMLR, volume 2025.

**(b) Version for the numbers.** v1/v3 report ResNet-34 / CIFAR-10 as **93.03 / 86.76 / 83.83 / 70.57** with *no standard deviations*. v4 reports **93.51 / 86.99 / 83.83 / 62.12** *with* them. The 99.99% cell moved by **8.45 points** between versions. Cite **v4**, and say so in the bibliography entry.

**(c) Two things the paper does not state,** which must not be asserted on its authority:

- whether training starts **from scratch or from ImageNet**;
- whether the **final classifier is in the pruned set** - the strings "classifier", "last layer", "final layer" and "output layer" have **zero occurrences** in v4.

Both have to be read off the ES2/ITOP code, not the paper.

**(d) An internal contradiction to disclose.** Appendix Table B.1 gives the LR schedule as `10x[75,150]`; the main text says $\times 10$ at halfway and three-quarters, i.e. epochs 125 and 187. The paper does not say which produced Table 1. Our declared substitution is cosine (§3.2).

**Two fidelity gaps this closed in our code.** `weight_decay` was hardcoded at GraNet's 5e-4 and is now a recorded field, set to EAST's **1e-4** for the anchor. And `delta_T` defaulted to **100**, updating masks **40x more often** than the baseline being reproduced (EAST/RigL use 4000) - a different algorithm, not a different setting.

## 6.5 Published baseline tables

### EAST v4, Table 1 - ResNet-34 / CIFAR-10, top-1, 3 runs

| Sparsity | RigL | EAST | SET | SynFlow |
|---|---|---|---|---|
| 99% | 92.92 ± 0.18 | 93.51 ± 0.13 | 93.09 ± 0.15 | 86.03 ± 0.71 |
| **99.9%** | **85.71 ± 0.23** | **86.99 ± 0.32** | 82.70 ± 0.91 | 61.61 ± 10.76 |
| 99.95% | 81.47 ± 0.32 | 83.83 ± 0.02 | 70.84 ± 1.51 | 56.33 ± 3.44 |
| 99.99% | 10.03 ± 0.19 | 62.12 ± 0.90 | 10.00 ± 0.00 | 10.00 ± 0.00 |

The pair the ladder anchors on - **RigL 85.71 ± 0.23 vs EAST 86.99 ± 0.32** - is confirmed, and it is the **99.9%** level. Note RigL **collapses to random** at 99.99% (10.03) while EAST holds 62.12; that column is a mixture, not a Gaussian, which is why the plan reports **median, IQR and collapse rate** there rather than mean ± std (§3.4, §7.12).

**Protocol** (v4, Table B.1 and §4): 250 epochs, batch 128, SGD momentum 0.9, LR 0.1, **weight decay 1e-4**, ERK, mask-update interval **4000** for RigL (1500 for SET), single A100, **3 runs** behind every ±.

### GraNet - published **dense** figures

| Model / dataset | Dense top-1 |
|---|---|
| ResNet-50 / CIFAR-10 | 94.75 ± 0.01 |
| ResNet-50 / CIFAR-100 | 78.23 ± 0.18 |
| VGG-19 / CIFAR-10 | 93.85 ± 0.05 |
| VGG-19 / CIFAR-100 | 73.43 ± 0.08 |

Recipe: 160 epochs, batch 128, SGD(0.9) LR 0.1, $\times 10$ at [80, 120], wd 5e-4. These are printed **identically in four separate GraNet tables** (1, 2, 6, 9), and all 24 shared sparse cells agree with GraSP's Table 2 - four-way internally consistent and externally cross-checked.

RISK: GraSP reports **94.23 / 74.16** for VGG-19, disagreeing with GraNet, and gives no standard deviations; its ResNet-32 rows are **2x-width** and must not be used as a ResNet-50 baseline.

RISK: GraNet's own CIFAR pipelines are internally inconsistent: CIFAR-10 pads with **reflect** before cropping, CIFAR-100 uses **zero** padding, GraSP uses zero for both. GraNet also trains CIFAR-10 on **45,000** images (a 0.1 validation split, against our 0.2 - §2.4) and uses **Nesterov** momentum, neither stated in the paper. If our dense numbers land below parity, check these three before concluding the recipe is wrong.

### The anchor cell has no published dense baseline - state this as a limitation

**There is no published dense ResNet-34 / CIFAR-10 figure.** That architecture/dataset pair is the ladder's **anchor cell**. GraNet covers ResNet-20/50 and VGG-19; EAST reports no dense row at all. Consequences that must appear in the paper:

1. The anchor's **dense parity cannot be checked against the literature**. Do not borrow a ResNet-50 or VGG-19 number as a stand-in.
2. `results/reference/dense_reference.json` records this explicitly as `"source": "NOT FOUND"`, so `test_dense_baseline_is_within_reach_of_published` (`project/tests/test_experiment_invariants.py:352-380`) **skips** that cell rather than silently passing it. The absence is machine-visible, not just prose.
3. Our own measurement stands alone: **A0 dense, 5 seeds, from scratch, 91.64 ± 0.30 top-1**. It is a MEASURED number with no published comparator, and must be labelled as such - not as "matching the literature" and not as "below the literature".

## 6.6 The sparsity-denominator citation

Both reference codebases **prune the final classifier by default**, and at 99.9%+ this changes the denominator by a large factor (§1.3.2, §3.5).

- **RigL** keeps bias and batch-norm dense under all distributions, and keeps the *first* layer dense **only under Uniform** - the ER and ERK bullets carry no such carve-out. It assigns ResNet-50's `fc1000` an ERK sparsity of **0.957**, and `resnet_model.py` defaults `prune_last_layer=True`. Cross-checked numerically: ER density for 2048 to 1000 is $(2048+1000)/(2048\times1000)=0.001488$; times $\varepsilon\approx28.9$ gives density 0.043, i.e. sparsity 0.957 - reproducing the published figure to three decimals, so it is not an extraction artefact.
- **GraNet** registers *every* 2-D and 4-D parameter tensor into the mask set with no name-based classifier exclusion; the only optional exclusion is the first conv, behind `--rm_first`.

**Consequence for the ladder:** rung A5 (`+ RigL`) inherits **A3** - ERK with the head **pruned** - not the head-dense corner (§5.2). Since Stage C is built on that substrate, the alternative arrangement would have made every downstream number incomparable to the published RigL/EAST figures the paper cites. The head-dense arms (A2, A4) remain as corners of the 2x2, measuring what the other convention buys rather than adopting it silently.

**CHOICE - one deliberate divergence.** RigL's *Uniform* distribution keeps the first layer dense; our `uniform` allocation does not. Baking it in would put a third difference inside the uniform arms and destroy the 2x2. RigL applies the carve-out only to Uniform, not ERK, so rungs A3-A6 - the comparable ones - are unaffected.

## 6.7 CAP: the direct antecedent, and the honest framing

**Reference:** Xu, Luo, Wang, Chang, Huang, Huang, Huang. *From Dense to Sparse: Contrastive Pruning for Better Pre-trained Language Model Compression.* AAAI 2022, vol. 36, pp. 11547-11555. arXiv:2112.07198.

**BaCP's module decomposition is CAP's.** The modules are literally **PrC, FiC and SnC**, and the objective is the same four-term sum. This is not a resemblance to be noted in passing; it is the antecedent.

| | CAP | BaCP (as originally implemented) |
|---|---|---|
| Modules | PrC, FiC, SnC | PrC, FiC, SnC - identical |
| Objective | $\lambda_1\mathcal{L}^{CE}+\lambda_2\mathcal{L}^{PrC}+\lambda_3\mathcal{L}^{SnC}+\lambda_4\mathcal{L}^{FiC}$ | same four terms, $\lambda$ indices permuted |
| Loss | one functional, two positive sets, **shared denominator** | two implementations summed, different denominators |
| Similarities | rectangular $B\times N$, student anchors x teacher candidates | square $2B\times 2B$ over $\mathrm{cat}[z_s,z_t]$ |
| Projection head | none (contrasts the `[CLS]` state) | `Linear(d,128)`, trainable, unpruned |
| Negatives | memory bank, $N=4096$ pre-encoded on CPU | in-batch only, $2B$ |
| Sparsity reported | 97% (PUBLISHED, R15) | 99% - **PUBLISHED (AAAI submission) and VOID** (§Status, §8.7 D7). The ladder anchors at **99.9%** (`LADDER_SPARSITY`, `manifest.py:65`), and no 99% number exists under the current codebase |
| Domain | **pre-trained language models only** | vision - CNNs run; **ViTs registered but never run** (§1.7) |

**Related Work must carry an explicit "Differences from CAP" paragraph.** This was reviewer objection **B7**, and honest framing is strictly stronger here than the alternative - a reviewer who discovers the overlap unaided discounts everything else. The paragraph should say, in this order:

1. CAP reports on **pre-trained language models only**; BaCP is CAP applied to vision.
2. The **module decomposition itself is not new** and is not claimed as a contribution.
3. **BaCP's claimed delta over CAP - every item of which is PLANNED, not established.** The list as it stood read: extension to vision backbones (CNNs and ViTs); the higher 99% sparsity regime against CAP's 97%; demonstration that the framework is pruning-criterion-agnostic across magnitude, SNIP-it and WANDA; and composition with EAST. **Nothing has been published (§Status), so none of it is a result yet.** Item by item, against what the rest of this document records:

   | claimed delta | status under the current codebase |
   |---|---|
   | extension to vision backbones, CNNs | **partly supported.** ResNet-34 is the only backbone exercised at scale, and only its dense rung completed; every sparse cell diverged (§8.1) |
   | extension to vision backbones, **ViTs** | **not supported.** Both ViT entries "have **not** been run under the current codebase" (§1.7). `_reject_dyrelu` also raises for them, so the DyReLU arms are unavailable there by construction (§1.1) |
   | the higher **99%** regime | **not supported.** The 99% figure is a void AAAI-submission number (table above). The ladder anchors at 99.9% and the headline sweep that would produce a 99% row is **declared but never scheduled** (§3.4, §8.8) |
   | **pruning-criterion-agnostic** across magnitude, SNIP-it and WANDA | **not supported.** "Only RigL has been exercised at scale. Magnitude, EAST, SNIP and WANDA are **untested** at these sparsities" (§7.9, §8.1) - and rungs A1-A4 and A6 all depend on them |
   | **composition with EAST** | **not supported.** "**Every EAST number produced before this fix is void**" (§7.10), and rung A6 (`pruning_type='east'`) has not been run |

   The honest form of the paragraph is therefore a statement of **what the ladder is designed to establish**, in the future tense, with the gates (§5.9) named as the conditions under which each item becomes claimable - not a list of deltas presented as delivered.
4. The **memory bank is a CAP component BaCP omitted**, not a BaCP contribution to claim. It slots directly into the rectangular formulation - $z_{\text{cand}}$ grows from $B$ rows to $B+4096$ with no change to the loss.

**Three further reporting corrections the paper owes** (from `docs/math.md` §7), independent of CAP framing:

- **SnC aggregation.** The paper writes $\mathcal{L}^{SnC}=\sum_{k=1}^{K}(\cdot)$; the code **averages** (`project/bacp.py:505`). Averaging is correct - it keeps the term's scale independent of $K$ so $\lambda_{SnC}$ needs no retuning as snapshots accumulate. **Fix the equation, not the code.**
- **$\lambda$ ordering.** `_combine_losses` maps $\lambda_1\to$ PrC, $\lambda_2\to$ FiC, $\lambda_3\to$ SnC, $\lambda_4\to$ CE (`project/bacp.py:776-779`). The plotting helper labelled its series `['CE','PrC','SnC','FiC']`, so **every published dynamic-lambda figure had three of four series mislabelled**. Now pinned by `test_lambda_to_loss_mapping` (`project/tests/test_losses.py:261`).
- **MLM perplexity.** RoBERTa and DistilBERT are **encoder-only**, not "decoder-only" as the appendix states, and pseudo-perplexity for masked language modelling is not the standard causal quantity - its computation needs defining explicitly (§7.13).

---

# 7. Mathematical formulas

Notation. $f_\theta$ is the sparse student, $\hat f$ a frozen teacher, $g$ the projection head, and $z=\mathrm{normalize}(g(f(x)))\in\mathbb{R}^{D}$ with $\lVert z\rVert_2=1$, so $z_i^\top\hat z_j$ is a cosine similarity. $\tau>0$ is the temperature, $B$ the batch size, $N$ the number of candidate embeddings, $C$ the number of classes.

Every formula below carries its `file:line`, and a citation **or** an explicit statement that it has none. Three kinds of entry appear, and they are labelled: a **published result** with its reference; a **CHOICE** - a definition or design decision with no canonical source, which the paper must derive rather than attribute; and an **UNVERIFIED** citation - a source named here but absent from `docs/citations.md`, so unchecked against the primary text by this project (§6.1). Where the code and the written mathematics disagree, the disagreement is stated rather than smoothed over.

The entries that carry no verified citation, collected so none is lost: $s_{\text{mask}}$ and $s_{\text{value}}$ (§7.11, CHOICE), the weight-sharing rescale $s_{\text{adj}}$ (§7.11, project-original), the Welch-Satterthwaite $\nu$ (§7.12, R46, UNVERIFIED), the $\chi^2$ interval on $\hat\sigma$ and the two-$t$ MDE (§7.12, standard results with no recorded source), DyReLU phasing and DyReLU-B (§7.14, R14 and R29, both UNVERIFIED for these uses), and the 4-D WANDA folding (§7.6, already labelled "our convention").

## 7.1 The BaCP objective and its four $\lambda$ terms

$$\mathcal{L}_{\mathrm{BaCP}} \;=\; \lambda_1\mathcal{L}^{\mathrm{PrC}} \;+\; \lambda_2\mathcal{L}^{\mathrm{FiC}} \;+\; \lambda_3\mathcal{L}^{\mathrm{SnC}} \;+\; \lambda_4\mathcal{L}^{\mathrm{CE}} \;+\; \lambda_{kd}\,\mathcal{L}^{\mathrm{KD}}$$

**Code:** `project/bacp.py:776-786`. The index mapping is $\lambda_1\to$ PrC (`:776`), $\lambda_2\to$ FiC (`:777`), $\lambda_3\to$ SnC (`:778`), $\lambda_4\to$ CE (`:779`), pinned by `project/tests/test_losses.py:261`. **Citation:** CAP Eq. 1 (R15) - same four terms, $\lambda$ indices permuted relative to the CAP paper (§6.7).

**SnC aggregation** is a mean, not a sum:

$$\mathcal{L}^{\mathrm{SnC}} \;=\; \frac{1}{K}\sum_{k=1}^{K}\mathcal{L}\!\left(z, \hat z^{(k)}\right)$$

`project/bacp.py:498-505` (`L_snc_raw /= len(self.ss_models_on_device)`), with $K$ capped by `num_snapshots`, default 2 (`bacp.py:92`, `:841`). This contradicts the submitted paper's $\sum_k$; the code is right (§6.7).

### The simplex constraint

The four task lambdas are **intended** to satisfy

$$\lambda_1+\lambda_2+\lambda_3+\lambda_4 = 1,\qquad \lambda_i\ge 0,$$

and every manifest override honours it: C-null $(0,0,0,1)$ (`manifest.py:305`), C3 $(0,\tfrac12,0,\tfrac12)$ (`:361`), D1 $(0,\tfrac13,\tfrac13,\tfrac13)$ (`:392`), D2 $(\tfrac14,\tfrac14,\tfrac14,\tfrac14)$ (`:398`), default $(\tfrac14,\tfrac14,\tfrac14,\tfrac14)$ (`bacp.py:248`, `manifest.py:521`).

⚠ **The simplex is a convention, not an enforced constraint.** `_initialize_lambdas` (`bacp.py:245-262`) unpacks the tuple with no projection, no softmax and no renormalisation. Under `learnable_lambdas=True` the four values become raw `nn.Parameter`s added to the optimizer (`bacp.py:251`, `:257`) and **nothing projects them back onto the simplex** - they are free to leave $[0,1]$ and their sum is unconstrained. Any claim that the learnable-$\lambda$ arm optimises *over the simplex* is unsupported by the code as written (§8.8).

### Why $\lambda_{kd}$ sits **outside** the simplex

`bacp.py:764-786`, docstring at `:768-774`. The reason is the ablation-design defect in the submitted paper (§5.6). Keeping $\lambda_{kd}$ outside means

$$\mathcal{L} = \underbrace{\textstyle\sum_{i=1}^{4}\lambda_i\mathcal{L}_i}_{\text{sums to a fixed scale}} \;+\; \lambda_{kd}\mathcal{L}^{\mathrm{KD}},$$

so turning distillation on does not silently rescale the contrastive terms and the rung C0b $\to$ C1 is **one** change. `lambda_kd` defaults to 0.0 (`bacp.py:105`) and is cross-validated against `distill_mode` at `:167-181` so a "KD arm" cannot be recorded with the term switched off.

## 7.2 The contrastive functional

### 7.2.1 The unified CAP form, as implemented

For anchor $i$ with positive set $P(i)$ drawn from the candidates:

$$\mathcal{L}_i \;=\; -\frac{1}{|P(i)|}\sum_{j\in P(i)}\log\frac{\exp\!\left(z_i^\top\hat z_j/\tau\right)}{\sum_{k=1}^{N}\exp\!\left(z_i^\top\hat z_k/\tau\right)},\qquad \mathcal{L}=\frac{1}{|\mathcal{A}|}\sum_{i\in\mathcal{A}}\mathcal{L}_i,\quad \mathcal{A}=\{i:|P(i)|>0\}$$

**Code:** `project/loss_functions.py:5-64`. Logits `:50`; log-softmax `:55`; per-anchor mean over positives `:63`; anchors with **no** positives **excluded** (not divided by a clamped 1) `:59-64`. **Citation:** CAP Eq. 1 (R15). This is the $L_{out}$ form (§7.2.3). $\tau$ default $0.07$ (`bacp.py:45`).

Each module instantiates this **twice over the same candidate set**, hence one shared denominator (CAP Table 1):

| Module | teacher $\hat f$ | unsupervised $P(i)$ | supervised $P(i)$ | code |
|---|---|---|---|---|
| PrC | pretrained $\phi_{pre}$ | $\{\hat z_i\}$ | $\{\hat z_j: y_j=y_i\}$ | `loss_functions.py:94-98` / `:102-103` |
| FiC | fine-tuned $\phi_{fine}$ | $\{\hat z_i\}$ | $\{\hat z_j: y_j=y_i\}$ | same |
| SnC | snapshots $\phi_{r'},\,r'<r$ | $\{\hat z_i\}$ | $\{\hat z_j: y_j=y_i\}$ | same |

$$\mathcal{L}^{\text{module}}=\mathcal{L}_{\text{unsup}}+\mathcal{L}_{\text{sup}}\qquad(\texttt{loss\_functions.py:90-105})$$

### 7.2.2 Rectangular, asymmetric anchor structure

The similarity matrix is $B\times N$ - **student anchors, teacher candidates** - not $2B\times 2B$:

$$S = \frac{Z_{\text{student}}\,\hat Z_{\text{teacher}}^{\top}}{\tau}\in\mathbb{R}^{B\times N},\qquad N \ne B \text{ permitted}$$

`loss_functions.py:50`. The superseded square construction $Z=\mathrm{cat}[z_s,z_t]$, $ZZ^\top$ carries four blocks

$$ZZ^\top=\begin{bmatrix} z_sz_s^\top & z_sz_t^\top\\ z_tz_s^\top & z_tz_t^\top\end{bmatrix}$$

of which only $z_sz_t^\top$ appears in CAP or in BaCP's own written equations. The other three are implementation artefacts: $z_sz_s^\top$ adds an unspecified within-student SimCLR term; $z_tz_t^\top$ contributes gradient-free constants to teacher-anchored denominators; $z_tz_s^\top$ makes teacher rows anchors, so averaging over all $2B$ rows **dilutes the loss by roughly $2\times$ and silently halves the effective $\lambda$ on every contrastive term**. (`docs/math.md` §3.1; `loss_functions.py:22-30`.)

**Gradient** (`docs/math.md` §2.1). With $s_{ik}=z_i^\top\hat z_k/\tau$ and $p_{ik}=\mathrm{softmax}_k(s_{ik})$:

$$\frac{\partial\mathcal{L}_i}{\partial z_i}=\frac{1}{\tau}\left[\sum_{k=1}^{N}p_{ik}\hat z_k-\frac{1}{|P(i)|}\sum_{j\in P(i)}\hat z_j\right]$$

predicted candidate centroid minus positive centroid. Two consequences: **$\hat z$ receives no gradient** (teacher forward under `torch.no_grad`), which is what makes this a distillation objective rather than a joint-embedding one; and **$1/\tau$ scales the entire gradient**, so $\tau$ and the learning rate are not independent knobs.

**Analytic fixed point** (`docs/math.md` §2.2, tested in `project/tests/test_cap_contrastive.py`): $z=\hat z=0\Rightarrow\mathcal{L}=\log N$ **exactly** - note $\log N$, not $\log(2N-1)$. There is no self-similarity to mask: the anchor's own index in the candidate set is a *legitimate* positive (the same input through a different model), unlike NT-Xent's degenerate self-match.

**Numerical stability.** $\mathcal{L}_i$ is invariant to a per-row constant shift, so the implementation subtracts $\max_k s_{ik}$, **detached** (`loss_functions.py:54`): it is a constant of the optimisation and letting gradient flow through it would add a spurious term.

### 7.2.3 SupCon $L_{out}$ vs $L_{in}$, and NT-Xent

**Citation:** Khosla et al. (R16) for the $L_{out}$/$L_{in}$ distinction; Chen et al. (R17) for NT-Xent.

$$\mathcal{L}^{sup}_{out}=\sum_{i\in I}\frac{-1}{|P(i)|}\sum_{p\in P(i)}\log\frac{\exp(z_i^\top z_p/\tau)}{\sum_{a\in A(i)}\exp(z_i^\top z_a/\tau)}$$

$$\mathcal{L}^{sup}_{in}=\sum_{i\in I}-\log\left\{\frac{1}{|P(i)|}\sum_{p\in P(i)}\frac{\exp(z_i^\top z_p/\tau)}{\sum_{a\in A(i)}\exp(z_i^\top z_a/\tau)}\right\}$$

By Jensen, $\mathcal{L}_{in}\le\mathcal{L}_{out}$; Khosla et al. establish $L_{out}$ as the superior formulation. **The implemented `SupConLoss` is the $L_{out}$ form**: `loss_functions.py:137-139` computes `sum(log_prob * mask_pos) / num_pos` - the sum over positives sits **outside** the log. $A(i)$ is all indices except self (`neg_mask = 1 - I`, `:126`, `:135`).

*Fidelity note:* Khosla's reference implementation multiplies by $\tau/\tau_{\text{base}}$; this implementation has no such prefactor, but `SupConLoss` is constructed with `temp = base_temp = tau` (`bacp.py:186`) so the prefactor would be $1$ and its absence is immaterial **at the current call site only**.

**NT-Xent** (`loss_functions.py:150-165`), over $Z=\mathrm{cat}[z_1,z_2]\in\mathbb{R}^{2B\times D}$ with $j(i)$ the paired view:

$$\mathcal{L}_{\text{NT-Xent}}=\frac{1}{2B}\sum_{i=1}^{2B}-\log\frac{\exp(z_i^\top z_{j(i)}/\tau)}{\sum_{k\ne i}\exp(z_i^\top z_k/\tau)}$$

implemented as a cross-entropy against target $j(i)$ (`:161-164`) with the diagonal masked to $-\infty$ (`:159`).

### 7.2.4 Why summing the two was self-defeating (the shared-denominator argument)

Labels are repeated across views, so $y_{i+B}=y_i$ **by construction** and the paired view is *always* in SupCon's positive set. NT-Xent's single positive is therefore a **strict subset** of SupCon's, and Khosla et al. §3 note SupCon reduces to NT-Xent at $|P(i)|=1$. Adding NT-Xent contributes **no new positive**; it re-weights one SupCon already had and **reclassifies every other same-class sample**:

| For a same-class, different-instance pair $(i,j)$, $y_j=y_i$, $j\ne i+B$ | treats $j$ as | gradient weight on that pair |
|---|---|---|
| SupCon | positive, **pull** | $1/\lvert P(i)\rvert$ |
| NT-Xent | negative, **push** | full softmax weight $p_{ij}$ |

At $B=512$ on CIFAR-10, $|P(i)|\approx 2B/C=102$, so **the push dominates the pull by roughly two orders of magnitude**. The composition converts supervised contrastive learning into instance discrimination that fights its own class structure. **That argument stands on the algebra above and on nothing else** - it is a statement about which pairs each term pulls and pushes, derived here, and it does not need a measurement.

⚠ **The one apparent confirmation is PUBLISHED-and-VOID and must not be used.** The submitted paper's Appendix Table 6 at 99% sparsity reads: supervised only **92.96**, self-supervised only 92.66, neither 92.63, both (as shipped) **92.32** - best alone, worst together, which is the ordering a dominated pull term predicts. But those are **PUBLISHED (AAAI submission) figures**, and every pre-fix ResNet-34 number, the AAAI submission's included, was produced under the `drop_last=True` eval defect and is **VOID** (§Status, §2.7, §8.7 D7). A void number cannot corroborate anything, and the four sit within 0.34 points of each other with **no error bars at all** - below any plausible noise floor (measured dense $\hat\sigma = 0.30$, §3.7), so even were they valid the ordering would not be resolvable. Quote them, if at all, as the motivating observation that prompted the algebra, explicitly labelled void; the **shared-denominator argument is analytic and needs no empirical support**. Re-measuring the crossover is what rungs C0b and C3 exist for.

*Empty-positive branch:* `SupConLoss` divides by `num_pos.clamp(min=1.0)` (`loss_functions.py:130`), which looks like a dilution bug but is **unreachable** in-batch since $|P(i)|\ge 1$ always. It becomes live only with a memory bank that may hold no example of a label - which is why `cap_contrastive_loss` **excludes** such anchors instead (`:59-64`).

## 7.3 `kd_kl_loss` - response distillation

$$\mathcal{L}_{KD} \;=\; T^{2}\cdot \mathrm{KL}\!\left(\sigma(z_t/T)\,\big\|\,\sigma(z_s/T)\right) \;=\; \frac{T^{2}}{B}\sum_{b=1}^{B}\sum_{c=1}^{C} \sigma_c(z_t^{(b)}/T)\,\log\frac{\sigma_c(z_t^{(b)}/T)}{\sigma_c(z_s^{(b)}/T)}$$

**Code:** `loss_functions.py:204-249`. Student log-softmax `:247`; teacher softmax, detached, `:248`; `F.kl_div(..., reduction='batchmean') * (T*T)` `:249`. $T$ default 4.0 (`bacp.py:106`). **Citation:** Hinton, Vinyals & Dean (R4) - **for $T^2$ and soft targets only** (F2).

**The $T^2$ factor.** Hinton et al., end of §2: soft-target gradients scale as $1/T^2$, so multiplying by $T^2$ keeps the relative contribution of soft and hard targets roughly unchanged as $T$ varies. Without it, raising the temperature quietly turns the term off - which would surface in the ladder as "KD does nothing", indistinguishable from a real null. ⚠ The $1/T^2$ relation is a **high-temperature approximation**: the exact gradient is $\partial C/\partial z_i=\tfrac{1}{T}(q_i-p_i)$ (Eq. 2). $T^2$ is a balancing heuristic, not an exact normaliser.

**Direction.** `F.kl_div(log_softmax(s/T), softmax(t/T))` computes $\mathrm{KL}(p_t\|q_s)$ - **forward** KL, teacher as reference. Gradient-identical to Hinton's cross-entropy-with-soft-targets because $H(p_t,q_s)=\mathrm{KL}(p_t\|q_s)+H(p_t)$ and $H(p_t)$ is constant in the student's parameters. **Cite implementations, not Hinton, for this form.**

**`batchmean`, not `mean`.** PyTorch's *default* is `'mean'`, which divides by `numel` $=B\times C$ and which PyTorch's own documentation notes does not return the true KL. The two differ by **exactly the class count** - 10 on CIFAR-10, 100 on CIFAR-100 - and that factor composes multiplicatively with $T^2$ and with $\lambda_{kd}$, silently detuning the KD/CE balance across datasets.

**3-D flattening.** `batchmean` divides by `input.size(0)` only, so a masked-LM tensor $[B,T,V]$ was summed over all $T$ token positions and divided by $B$ alone - the returned term was **exactly `seq_len` times** the per-example quantity (measured: $128.0\times$ at $T=128$). Fixed by reshaping to $[BT,V]$ first (`loss_functions.py:243-246`).

## 7.4 `feature_distill_loss` - cosine and MSE

$$\mathcal{L}^{\cos}_{\mathrm{feat}}=\frac{1}{B}\sum_{b=1}^{B}\left(1-\frac{z_s^{(b)\top}z_t^{(b)}}{\lVert z_s^{(b)}\rVert\,\lVert z_t^{(b)}\rVert}\right) \qquad\qquad \mathcal{L}^{\mathrm{mse}}_{\mathrm{feat}}=\frac{1}{BD}\sum_{b=1}^{B}\bigl\lVert z_s^{(b)}-z_t^{(b)}\bigr\rVert_2^{2}$$

**Code:** `loss_functions.py:252-289`; cosine `:286`, MSE `:288`; teacher detached `:284`. Default `mode='cosine'` at the signature (`:252`) and `feature_distill_metric='cosine'` at the config (`bacp.py:107`). **Citation:** FitNets (R5) for the squared-Euclidean *hint* idea **only**; there is **no valid source for cosine-on-penultimate** (F3).

**Why cosine is the default - the stated rationale.** Cosine is **scale-invariant**. A student at 0.1% density and a dense teacher have very different feature norms: with ~0.1% of connections surviving, the surviving activations' magnitude is not comparable to the dense teacher's, so a squared-Euclidean objective spends most of its gradient budget reconciling a **norm** the student cannot and should not match, rather than the **direction** that carries the representational content. Cosine removes the norm from the objective by construction. This is **BaCP's own design choice**; state it as such.

**CHOICE - but the two branches are not independent metrics at the current call site.** Every branch of `get_embeddings` ends in `F.normalize` (`model_factory.py:336`, `:338`), so both arguments reaching `feature_distill_loss` from `_distillation_loss` (`bacp.py:758-760`) are already unit-norm, and on unit vectors

$$\lVert z_s-z_t\rVert_2^{2}=2\,(1-\cos)\quad\text{exactly}\;\Longrightarrow\; \mathcal{L}^{\mathrm{mse}}_{\mathrm{feat}}=\frac{2}{D}\,\mathcal{L}^{\cos}_{\mathrm{feat}}$$

a monotone rescaling that changes only the effective $\lambda_{kd}$. Two consequences the paper must respect:

1. **Do not present a cosine/MSE ablation as evidence about norm versus direction.** Norm is a constant on both sides. Pinned by `test_mse_and_cosine_are_the_same_metric_on_normalised_features` (`project/tests/test_distillation.py:213`).
2. The norm-mismatch **rationale** for choosing cosine is currently **inoperative at the call site**, because normalisation has already removed the mismatch upstream. It becomes a real distinction only if unnormalised features are passed in, which nothing currently does. State the rationale as the reason for the design, and state that the implementation makes the choice moot - do not claim a measured benefit from it.

## 7.5 "Cosine matching is InfoNCE with the denominator deleted"

This underwrites the C2 $\to$ C3 rung (`manifest.py:358-374`), so precision matters more here than anywhere else in the section.

### The derivation (ours, exact algebra, no asymptotics)

For anchor $i$ with a single positive $\hat z_i$ over candidates $\{\hat z_k\}_{k=1}^{N}$, on unit-norm embeddings:

$$\mathcal{L}^{\mathrm{InfoNCE}}_i \;=\; -\log\frac{\exp(z_i^\top\hat z_i/\tau)}{\sum_{k}\exp(z_i^\top\hat z_k/\tau)} \;=\; \underbrace{-\frac{z_i^\top\hat z_i}{\tau}}_{\text{positive-pair term}} \;+\; \underbrace{\log\sum_{k}\exp\!\left(\frac{z_i^\top\hat z_k}{\tau}\right)}_{\text{denominator / uniformity}}$$

Delete the second term. What remains is $-\cos(z_i,\hat z_i)/\tau$, and

$$\mathcal{L}^{\cos}_{\mathrm{feat},i}=1-\cos(z_i,\hat z_i)=1+\tau\cdot\left(-\frac{\cos(z_i,\hat z_i)}{\tau}\right)$$

an **affine reparameterisation with positive slope $\tau$**. Minimising one minimises the other; the gradient directions are identical and the magnitudes differ by the constant factor $\tau$. This step is **exact algebra at finite batch size** and requires no limit. **It is our derivation - present it as a one-line derivation of our own.**

### What Wang & Isola (R11) do and do not support

| Claim | Supported? |
|---|---|
| The limiting normalised InfoNCE decomposes into an **alignment** term and a **uniformity** term (Theorem 1) | **Yes** - cite for this |
| The encoder maps onto the sphere, so the alignment loss is a distance on L2-normalised features, and (their appendix, in these words) it *is* cosine "up to a constant and a scaling" via $\lVert u-v\rVert^2=2-2u^\top v$ | **Yes** - cite for this identity |
| That the decomposition holds **exactly at finite batch size** | **No.** Theorem 1 is asymptotic: batch to infinity with $O(M^{-1/2})$ deviation |
| That any of this concerns **distillation, teachers, students, hint losses or pruning** | **No.** The paper is a pure self-supervised analysis and mentions none of these |

**Therefore: never write "as shown by [Wang & Isola]" for the distillation reading.**

### What BYOL (R12) does support - the closest published statement

BYOL Eq. 6 writes an InfoNCE whose log-sum-exp denominator is weighted by $\beta$; Eq. 7 defines the similarity as a normalised dot product (cosine); and the authors state **"We recover the BYOL loss ... with $\beta=0$"** - while BYOL's own Eq. 2 is $2-2\cos$. That **is** "the positive-pair term of InfoNCE with the denominator removed", asserted **by the authors**, **at finite batch size**. Cite BYOL Eq. 6-7 for the $\beta=0$ parameterisation.

### The single most useful citable number - and it is a warning

BYOL **Table 5(b), bottom row**: $\beta=0$ with **no predictor and no target network** gives **0.1%** ImageNet top-1, against SimCLR's **69.4%** at $\beta=1$. Published, controlled, same architecture, precisely "what happens when you delete the denominator".

⚠ **Do not over-read it.** BYOL's collapse is between a *trainable* online/target pair with no predictor. Rung C2 regresses onto a **frozen** teacher, which cannot drift, so the collapse mechanism does not transfer directly. Cite it as evidence that the denominator **does real work** - not as a prediction for what C2 will do.

### ⚠ Open: C2 $\to$ C3 does not currently isolate one thing

The manifest note (`manifest.py:370-374`) states that C2 $\to$ C3 "isolates ONE thing: the presence of negatives". Under the shipped configuration it does not. C3 turns on FiC via `CAPContrastiveLoss`, constructed at `bacp.py:189` with **both** defaults active - `supervised=True, unsupervised=True` (`loss_functions.py:78`). So:

- C3's **unsupervised** half **is** exactly C2's cosine term plus the log-sum-exp denominator - the clean isolation the rung claims;
- C3 **additionally** adds the **supervised** half, whose positive set is $\{\hat z_j:y_j=y_i\}$ - label-defined positives that C2 has no analogue of.

C2 $\to$ C3 therefore changes **two** things: the denominator, and the positive set. Options: run C3 with `supervised=False` for the strict two-way isolation, or insert a C3a rung between them. **Unresolved** (§8.3); the claim as currently worded is not supported by the configuration that will run.

## 7.6 WANDA importance

$$S_{ij} \;=\; \lvert W_{ij}\rvert \cdot \bigl\lVert X_{j}\bigr\rVert_{2}$$

where $i$ indexes the output row, $j$ the input feature, and $\lVert X_j\rVert_2$ is the L2 norm of input feature $j$ aggregated over a calibration set (WANDA: 128 sequences from C4). **Citation:** Sun, Liu, Bair, Kolter, ICLR 2024 (R1) - **for the metric only** (F1).

**Code:** `project/pruning_factory.py:637` (`scores = flat.abs() * scale`), with `scale` at `:626`.

*Fidelity note on the statistic.* `scaler_row` accumulates an **incremental mean of squared column norms** (`pruning_factory.py:550-561`): $\mathrm{prev}\leftarrow\mathrm{prev}\cdot\frac{n_{old}}{n_{old}+n_{new}}+\frac{1}{n_{old}+n_{new}}\sum_b x_{bj}^2$. Its square root is therefore the **root-mean-square** per input column, $\lVert X_j\rVert_2/\sqrt{n}$, not the raw L2 norm. Since $1/\sqrt n$ is a single positive scalar shared by every column of a layer, the ranking is unchanged **within both comparison groups** - the two forms are monotone-equivalent here. The accumulation is incremental precisely so the statistic does not depend on how many calibration batches happened to be available.

### The comparison group

$$\text{per-output row } i:\quad \text{drop the } k=\lfloor n_{in}\,s\rfloor \text{ smallest } S_{ij} \text{ within row } i$$

$$\text{layer-wise:}\quad \text{drop the } k=\lfloor \mathrm{numel}(S)\,s\rfloor \text{ smallest } S_{ij} \text{ across the whole layer}$$

`pruning_factory.py:639-649` (output) and `:650-660` (layer); knob `wanda_group` at `:489`, default `'output'`. **The choice is ours, not WANDA's** - see F1. Under per-output ranking every output neuron keeps the same fraction of its inputs; under layer-wise a row can lose an entire neuron, which is the risk per-output grouping exists to avoid.

### The 4-D convolution folding - our convention

WANDA's reference implementation (`lib/layerwrapper.py`) handles `nn.Linear` only; `WrappedGPT`'s reshape branch is gated on `isinstance(layer, nn.Linear)`. **There is no published convention for 4-D weights.** We fold the kernel into the input dimension:

$$W\in\mathbb{R}^{C_{out}\times C_{in}\times k_h\times k_w} \;\longrightarrow\; \tilde W\in\mathbb{R}^{C_{out}\times (C_{in}k_hk_w)}, \qquad X \;\xrightarrow{\ \mathrm{unfold}\ }\; \tilde X\in\mathbb{R}^{(BL)\times(C_{in}k_hk_w)}$$

so the scored matrix is exactly the $[\text{rows}\times\text{columns}]$ matrix WANDA scores for a Linear layer. `F.unfold` at `pruning_factory.py:542-545`; weight reshape at `:622`. Scoring per $(c_{in},k_h,k_w)$ position against a per-**channel** norm instead would ignore that different kernel positions see different input statistics. **State this in the paper; do not imply it came from the source.**

⚠ **Two failure modes the code announces rather than hides.** (1) With no calibration loader every norm falls back to 1, which makes this **exactly magnitude pruning** - printed as a warning at `:526-528`. (2) Grouped and depth-wise convolutions hit the fallback **deterministically**, because `F.unfold` is group-unaware and produces $C_{in}k_hk_w$ columns where the weight has only $(C_{in}/\text{groups})k_hk_w$; those layers are magnitude-pruned and named at `:663-669`. A ResNeXt-style backbone would otherwise report itself as WANDA throughout.

⚠ **Calibration is one-shot** (`pruning_factory.py:601-616`): it runs on the first mask update and never repeats, so under a gradual schedule the weights are re-read live while the activation norms stay frozen at the first update. A protocol decision, recorded as `calibration_step` and pinned by a test - but one the paper must state.

## 7.7 The ERK density solve - AUTHORITATIVE for the ERK numbers

### As implemented

For each prunable tensor $l$ with shape $(n_{out},n_{in},k_h,k_w)$ (or $(n_{out},n_{in})$), the code sets

$$m_l \;=\; \underbrace{n_{in}\,k_h\,k_w}_{\texttt{np.prod(param.shape[1:])}} \;+\; n_{out}, \qquad N_l=\mathrm{numel}(W_l)$$

`pruning_factory.py:198-203`. Densities are $d_l=\varepsilon\, m_l/N_l$ (`:239`) with $\varepsilon$ solved from the **global** budget

$$\sum_{l} d_l N_l = \varepsilon\sum_l m_l \;\stackrel{!}{=}\;\bigl\lfloor(1-s)\,N_{\text{total}}\bigr\rfloor \quad\Longrightarrow\quad \varepsilon=\frac{\text{remaining budget}}{\sum_{l\in U} m_l}$$

`:212`, `:226-234`. **Clipping / renormalisation loop** (`:220-255`): any $l$ with $d_l>1$ is fixed to $d_l=1$, removed from the unconstrained set $U$, its $N_l$ charged against the budget, and $\varepsilon$ re-solved - iterated until no violations remain (`:244-255`):

$$\varepsilon^{(r+1)}=\frac{\lfloor(1-s)N_{\text{total}}\rfloor-\sum_{l\notin U^{(r)}}N_l}{\sum_{l\in U^{(r)}}m_l}$$

**Verified:** the global parameter budget is held **exactly** - MEASURED `eff_sparsity` 0.999000 at target 0.999.

### As published

**Citation:** original Erdos-Renyi, Mocanu et al. 2018 (R2); the kernel extension **ERK**, Evci et al. 2020 (R3). Evci's ERK mass is

$$m_l^{\mathrm{ERK}} \;=\; n^{l-1}+n^{l}+w^{l}+h^{l}$$

kernel extents **added**. The two coincide for `Linear` layers ($k_h=k_w=1$) and **differ for every convolution**, where the code multiplies $n_{in}$ by $k_hk_w$ instead. This is **flag F8** (§6.3) and is currently unfixed (§8.3).

**Consequence at the ladder's operating point** (target 0.999, 3x3 CIFAR stem): `conv1` receives density **0.015352**, i.e. about **26.5 of its 1728** weights. At 0.9999 it is about **2.7** weights. Masks are drawn **Bernoulli** per layer (`pruning_factory.py:684` for RigL, `:751` for EAST), not by exact top-$k$, so the realised count is random with standard deviation $\sqrt{N_l d_l(1-d_l)}$ - at 0.9999 a **completely dead stem is a realistic sampling outcome**, not a pathological one.

**Also note:** `layerwise_alloc='uniform'` sets every layer to a flat $1-s$ (`pruning_factory.py:378-380`). The A1-A4 2x2 (§5.2) exists because "ERK helps" and "ERK protects the small classifier layer" are otherwise confounded: under ERK the classifier's ER mass is large relative to its parameter count, so it ends up near-dense whether or not it is nominally in the prunable scope.

## 7.8 The cubic sparsity ramp

$$s(t) \;=\; s_f\left(1-\left(1-\frac{t}{T}\right)^{3}\right),\qquad t=\text{current\_idx}+1,\quad T=\mathrm{round}(0.8\cdot T_{\text{final}})$$

**Code:** `pruning_factory.py:270-271`; $t$ at `:263`; $T$ at `:160`; clamp $s\leftarrow\min(s,s_f)$ at `:276`. **Citation:** Zhu & Gupta (R10). Their form is $s_t=s_f+(s_i-s_f)\bigl(1-\frac{t-t_0}{n\Delta t}\bigr)^3$, which reduces to the above at $s_i=0$, $t_0=0$.

The remaining $20\%$ of the schedule is a **recovery** phase at fixed $s_f$ (printed at `:161`). At the `east250` protocol this places the topology freeze at step 62,400 of 78,000 (§3.2).

⚠ **Deviation to state.** Zhu & Gupta define the schedule over **training steps**. Our implementation applies it **epoch-wise** for the monotone pruners and **step-wise** for the DST ones (`total_steps` kwarg, `:155-159`). That is a deviation and should be stated rather than assumed equivalent.

## 7.9 RigL drop-and-regrow

**Citation:** Evci, Gale, Menick, Castro, Elsen, ICML 2020 (R3). **Code:** `pruning_factory.py:672-719`.

**Drop fraction** (cosine anneal, `:314-316`, called at `:687`):

$$f(t)=\frac{\alpha}{2}\left(1+\cos\frac{\pi t}{T}\right),\qquad \alpha=0.3\ \text{(default, \texttt{:673})},\quad T=\text{end\_idx}$$

**Per layer,** with mask $m_l$ and $k_l=\lfloor f(t)\cdot\lVert m_l\rVert_0\rfloor$ (`:693`):

$$\mathcal{D}_l=\operatorname*{arg\,topk}_{j:\,m_{lj}=1}\bigl(-\lvert W_{lj}\rvert\bigr),\ \lvert\mathcal{D}_l\rvert=k_l \qquad\text{(drop smallest magnitude, \texttt{:700-703})}$$

$$\mathcal{G}_l=\operatorname*{arg\,topk}_{j:\,m_{lj}=0\ \text{after drop}}\bigl\lvert\nabla_{W_{lj}}\mathcal{L}\bigr\rvert,\ \lvert\mathcal{G}_l\rvert=k_l \qquad\text{(grow largest gradient, \texttt{:708-711})}$$

$$m_l \leftarrow (m_l \setminus \mathcal{D}_l)\cup\mathcal{G}_l,\qquad W_{lj}\leftarrow 0\ \ \forall j\in\mathcal{G}_l$$

**Zero-initialised regrowth** at `:716`. **Citation for the choice:** Evci et al.'s stated reason is that zero-valued new weights do not perturb the current function. Optimizer state for regrown weights is cleared by multiplying momentum / `exp_avg` / `exp_avg_sq` / `max_exp_avg_sq` by the mask (`:292-307`, called `:718-719`).

*Implementation note:* masking is applied by **value multiplication** (`apply_mask`, `:285-289`), not as a multiplicative mask inside the autograd graph, so $\nabla_{W_{lj}}\mathcal{L}$ remains defined and generally non-zero for inactive weights - which is what makes RigL's dense-gradient growth criterion available at all. The same fact is why value-sparsity overstates mask-sparsity (§7.11).

⚠ **Only RigL has been exercised at scale.** Magnitude, EAST, SNIP and WANDA are **untested at these sparsities**, which matters because rungs A1-A4 and A6 all depend on them (§8.1).

## 7.10 EAST's cyclic sparsity schedule

**Citation:** Li et al. 2024/2025 (R14). **Code:** `pruning_factory.py:722-859`.

$$s_{\text{target}}(t) \;=\; s_{\max}-\left(s_{\max}-s_{\min}\right)\cdot\tfrac12\left(1-\cos\frac{2\pi t}{T_c}\right),\qquad t\le T_c$$

`:776-778`. $T_c=\max(1,\lfloor \text{end\_idx}\cdot r_c\rfloor)$ with $r_c$ default $0.5$ (`:741`, `:732`); $s_{\min}$ default $0.05$ (`:738`).

**The phase is the whole point.** The cycle is **anchored at $s_{\max}$**: it starts and ends at $s_{\max}$ and dips to $s_{\min}$ at the half-cycle, relaxing the constraint mid-cycle to allow parameter exploration before re-tightening. The naive $s_{\min}+(s_{\max}-s_{\min})\cdot\tfrac12(1-\cos)$ starts at $s_{\min}$, and since `_init_sparse_masks` has just initialised at $s_{\max}$ that reads as "grow from 99.9% sparse to 5% sparse" at $t=0$ - abandoning the sparsity budget entirely.

⚠ **Dead code with the wrong phase.** `BasePruner.cyclic_scheduler` (`pruning_factory.py:318-320`) implements exactly the $s_{\min}$-anchored form that `EASTPruner` explicitly rejects, and is **never called** by anything (verified: no call sites). It should be deleted or corrected before anyone wires it up.

**Per-layer allocation preserves the ERK ratio** (`:802-808`):

$$a_l^{\text{target}} \;=\; a_l\cdot\frac{1-s_{\text{target}}}{1-s_{\text{curr}}},\qquad s_{\text{curr}}=1-\frac{\sum_l\lVert m_l\rVert_0}{\sum_l N_l}$$

with $k^{\text{prune}}_l=\max(0,\lfloor a_l-a_l^{\text{target}}\rfloor)$ when $s_{\text{target}}>s_{\text{curr}}$ and $k^{\text{grow}}_l=\max(0,\lfloor a_l^{\text{target}}-a_l\rfloor)$ otherwise.

**Past $T_c$** the level is held at $s_{\max}$ and the pruner only **rewires**, dropping and regrowing equal counts $k^{\text{prune}}_l=k^{\text{grow}}_l=\lfloor a_l\cdot p\rfloor$ with $p$ default $0.1$ (`:820`, `:731`) - the drop-and-regrow behaviour inherited from RigL, with regrown weights zero-initialised (`:853`) and optimizer state cleared (`:858-859`).

**Clamping is mandatory** (`:828-829`): $k$ is derived from a **global** sparsity ratio but spent **per-layer**, so a layer whose ERK density differs from the global average can otherwise be asked to prune more weights than it has active, or grow more than it has inactive - `torch.topk` then raises "selected index k out of range".

⚠ **Historical defect worth recording in the methods section:** `update_masks` previously carried a `current_idx % delta_T` gate, but `_step_pruning_step` already gates on `step % delta_T == 0` and `BasePruner.step` does `t = current_idx + 1` before calling it - so the inner gate saw `step+1` and never fired for any $\Delta T>1$. **EAST's masks never updated once**, staying frozen at their Bernoulli ERK initialisation for entire runs. Every EAST number produced before this fix is void.

## 7.11 Sparsity definitions

### The two measures

$$s_{\text{mask}} \;=\; 1-\frac{\sum_{l}\lVert m_l\rVert_0}{\sum_{l} N_l} \qquad(\texttt{pruning\_factory.py:899-910})$$

$$s_{\text{value}} \;=\; \frac{\sum_{l}\bigl\lvert\{j: W_{lj}=0\}\bigr\rvert}{\sum_{l} N_l} \qquad(\texttt{pruning\_factory.py:878-896})$$

**CHOICE - both are definitions, not results, and neither is cited.** They are the two obvious readings of "sparsity" and no source in `docs/citations.md` defines either; what *is* citable is the denominator convention that fills in $\sum_l N_l$ (RigL R3, GraNet R25 - §6.6). Which of the two a paper reports is a reporting decision, and the inequality between them below is derived here, not quoted.

`report_sparsity` returns $s_{\text{mask}}$ when a pruner owns masks and falls back to $s_{\text{value}}$ otherwise (`:913-916`). Both are recorded side by side, with their gap, by `results.measure_sparsity` (`project/results.py:272-292`).

### Why $s_{\text{value}}\ge s_{\text{mask}}$ under dynamic sparse training

`apply_mask` multiplies weights by the mask (`pruning_factory.py:285-289`), so $\{j:m_{lj}=0\}\subseteq\{j:W_{lj}=0\}$ **always**. Hence

$$s_{\text{value}}\;\ge\;s_{\text{mask}},\qquad\text{equality iff no structurally active weight is exactly zero.}$$

Under DST that condition fails by construction: RigL and EAST **zero-initialise regrown connections** (`:716`, `:853`), so a freshly grown weight is structurally active and numerically zero. **MEASURED on a tiny ResNet: mask 0.8976 vs value 0.9277** - the value-based measure **overstates** sparsity by ~3 points. Overstating is the *flattering* direction, which is exactly why it must be said, and why the invariant is one-sided (§5.11). For monotone pruning (magnitude, SNIP) nothing regrows and the two agree.

### The three denominators

Governed by one module-level setting, `set_prunable_scope` (`pruning_factory.py:62-97`), so the pruner, `check_model_sparsity` and `check_sparsity_distribution` cannot disagree about which weights are being described. Anchor counts for the first two rows: §1.3.2.

| Scope | Denominator $\sum_l N_l$ covers | Flag |
|---|---|---|
| **backbone** (module default) | 2-D+ trainable tensors, minus DyReLU hyperfunctions, minus `encoder_head`, minus task heads, minus embeddings | `_PRUNE_TASK_HEAD=False`, `_PRUNE_EMBEDDINGS=False` (`:58-59`) |
| **+ task head** (the ladder's main path) | adds `cls_head` / `classifier` / `vocab_projector` / `vocab_transform` / `lm_head` (`:40-45`) | `prune_task_head=True` |
| **+ embeddings** | additionally adds token and position embeddings (`:35`) | `prune_embeddings=True` |

**Always excluded, in every scope** (`layer_check`, `:127-142`): any tensor with $\dim\le 1$ (norms and biases), any frozen tensor, DyReLU hyperfunctions (`:23`), and the contrastive projection head (`:27` - not part of the task model; discarded after BaCP training).

**Why the choice is not cosmetic.** A dense `Linear(2048,100)` head is 204,800 weights; at 99% sparsity on ResNet-50 the *entire* surviving budget is roughly 250,000. For DistilBERT the prunable set is **63.3%** of the model with embeddings excluded and **99.7%** with them included, so the same reported fraction describes wildly different objects. `count_parameters` (`results.py:225-269`) therefore records absolute counts - `total_params`, `prunable_params`, `nonprunable_params`, `active_prunable_params`, `mask_total`, and a `mask_covers_prunable` consistency flag - not just the fraction.

⚠ **Embeddings are dangerous for transformers:** DistilBERT and RoBERTa **tie** `word_embeddings` to the output projection, so pruning the embedding matrix also prunes the LM head. Detected by pointer identity in `_has_tied_embeddings` (`:99-114`) and warned about at `:93-96`.

### Weight sharing changes the denominator again

$$s_{\text{adj}} \;=\; 1-\frac{(1-s)\,n_{\text{total}}}{n_{\text{unique}}}$$

**CHOICE - this rescale is project-original.** It has no source in `docs/citations.md` and no counterpart in EAST (R14), which is cited for weight sharing as a *mechanism* and not for any sparsity-accounting rule. The requirement it encodes - that the shared arm survive with the same **absolute** number of weights as the unshared arm, so the two are comparable at a stated sparsity - is this project's, and the paper must derive it rather than attribute it. `training_utils.py:357-390` (`keep_w = total*(1-s)`, `adjusted = 1 - keep_w/unique`). Well defined only when $s>1-n_{\text{unique}}/n_{\text{total}}$; below that threshold the requested weight budget exceeds the unique weights available and $s_{\text{adj}}<0$, which raises rather than silently handing a negative sparsity to the pruner (`:386+`). Both `requested_sparsity` and `adjusted_sparsity` are recorded (`results.py:902`). Measured values on the anchor: §1.6. Sharing itself: $W_i=\sigma_i W_{R-1}$ with a per-block scalar gain (`weight_sharing.py:33-43`); ⚠ the gains $\sigma_i$ **must be exempt from weight decay** - under the default $5\times10^{-4}$ they decay toward zero and progressively silence the shared blocks.

## 7.12 The statistics

All in `project/experiments/ladder.py`. Primary endpoint and why it is not best-epoch: §3.6. Design decisions (which stage is paired, the $n=3$ floor, fixed-sequence multiplicity, the bolding rule): §5.10.

### Paired delta - Stages B, C, D

Over $S=\mathrm{seeds}(A)\cap\mathrm{seeds}(B)$ (intersection taken explicitly, so degraded pairing is visible as `n_pairs`):

$$d_s=a_s-b_s,\qquad \bar d=\frac1n\sum_{s\in S}d_s,\qquad s_d=\sqrt{\frac{1}{n-1}\sum_{s\in S}(d_s-\bar d)^2}$$

$$\mathrm{CI}_{95} = \bar d \;\pm\; t_{0.975,\,n-1}\cdot\frac{s_d}{\sqrt n}$$

`:201-222`; sample sd with $ddof=1$ at `:134-140`; half-width at `:217`.

### Welch difference of means - Stage A

Used where the **mask lottery** dominates and seeds do not meaningfully pair across arms:

$$\mathrm{SE}=\sqrt{\frac{s_a^2}{n_a}+\frac{s_b^2}{n_b}},\qquad \nu=\frac{\left(\frac{s_a^2}{n_a}+\frac{s_b^2}{n_b}\right)^{2}}{\frac{(s_a^2/n_a)^2}{n_a-1}+\frac{(s_b^2/n_b)^2}{n_b-1}},\qquad \mathrm{CI}=(\bar a-\bar b)\pm t_{1-\alpha/2,\,\nu}\,\mathrm{SE}$$

`:238`, `:248-250`, `:253`. **Citation:** the approximate degrees of freedom $\nu$ is the **Welch-Satterthwaite** equation - Satterthwaite (1946), Welch (1947) - **R46, §6.2**. It is a named published result and had no entry in this document's reference table until now; TOST (R22) and fixed-sequence testing (R23) were cited while it was not, which was an oversight rather than a convention. It is **not recorded in `docs/citations.md`** either, so R46 carries status UNVERIFIED. Degenerate corner handled explicitly: if $\mathrm{SE}=0$ (both arms constant) the Welch $\nu$ is $0/0$ and used to abort `report()` before any table was written; it now returns `ci=(None,None)` with a note, because zero within-arm variance is **not** evidence of a zero-width interval (`:239-247`). `test_unpaired_delta_survives_zero_variance_in_both_arms` (`test_ladder_stats.py:229`) pins it.

### The scipy fallback path

⚠ **`scipy` is in `requirements-dev.txt` but not `requirements.txt`** (§8.8), so unless it is installed by hand - as `run_remote.sh` happens to do on the current box (§4.1) - the hard-coded $t$ table `_T95` (`:171-177`) is the **live** path. It is complete for $df\in[1,30]$ and falls back to $1.960$ above; `_T_SOURCE` (`:181`, `:187`, `:197`) records which path produced each critical value so a fallback interval is never mistaken for an exact one. The previous table omitted $df=1,6,8$, so $n=2,7,9$ silently fell through to a default of $2.0$ instead of $12.706$, $2.447$, $2.306$ - a paired interval up to **6.35x too narrow**, printed with no indication anything had gone wrong. $n=7$ is reachable in the shipped manifest (C3 declares 8 seeds; one crash gives 7). `test_t_critical_values_are_right_at_every_small_n` (`test_ladder_stats.py:200`) pins it.

`unpaired_delta` has **no such protection**: on the scipy-less path it uses a bare $t = 2.0$ for the Welch statistic (`:251-255`), records nothing in `_T_SOURCE`, and Welch df in Stage A can easily be near 4-8, where the true critical values are $2.78$ and $2.31$. **Stage A's intervals would print up to ~28% too narrow with no warning.** `sigma_chi2_interval` returns `(None, None)` without scipy (`:159-160`), so the $\hat\sigma$ intervals silently vanish from the table.

### $\chi^2$ interval on $\hat\sigma$ itself

$$\left[\;\sqrt{\frac{(n-1)\hat\sigma^{2}}{\chi^{2}_{1-a,\,n-1}}}\;,\;\; \sqrt{\frac{(n-1)\hat\sigma^{2}}{\chi^{2}_{a,\,n-1}}}\;\right],\qquad a=\frac{1-\text{conf}}{2}$$

`:143-160`. **Citation:** the $\chi^2$ interval on the variance of a normal sample is a standard textbook result; **no source for it is recorded in `docs/citations.md`**, and none is supplied here from memory (§6.1). Printed **wherever $\hat\sigma$ is**, because an $n=5$ estimate of $\sigma$ has a 95% interval running roughly **0.6x to 2.9x** the point estimate (docstring `:146-149`). *Derived here from the implemented formula* (not measured): at $n=5$ the multipliers are $\sqrt{4/\chi^2_{0.975,4}}\approx 0.599$ and $\sqrt{4/\chi^2_{0.025,4}}\approx 2.874$. Applied to the **MEASURED** dense $\hat\sigma=0.2975$ (§3.7): **95% CI $\approx [0.18,\ 0.85]$ points**. The point estimate is MEASURED; the interval is DERIVED, and no run has printed it yet. Quoting a bare $\hat\sigma$ from five runs as though it were the noise floor is how an underpowered design gets declared adequately powered.

### TOST - so a null is publishable

$$\text{equivalent within }\pm\Delta \iff \mathrm{CI}_{1-2\alpha}(\bar d)\subset(-\Delta,\,+\Delta)$$

`:261-272`; default $\Delta=1.0$ point, $\text{conf}=0.90$ (i.e. $\alpha=0.05$ one-sided each side). **Citation:** Schuirmann 1987; Lakens 2017 (R22).

### Minimum detectable effect

$$\mathrm{MDE} \;=\; \left(t_{1-\alpha/2,\;n-1}+t_{1-\beta,\;n-1}\right)\cdot\frac{\sigma_d}{\sqrt n}$$

**Citation:** the two-$t$ power formula for a paired mean is a standard result with no single canonical source; **it is not recorded in `docs/citations.md`** and none is supplied here from memory (§6.1). The docstring at `:20` states "at $n=3$ paired, $t(.975,2)=4.303$ and the minimum detectable effect is ~3.1 $\sigma_d$" - and $(4.303+1.061)/\sqrt3=3.10$, so the source's implied power target is **80%**, although the source does not state it.

| $n$ | $t_{.975,\,n-1}$ | $t_{.80,\,n-1}$ | multiplier $(t_{.975}+t_{.80})/\sqrt n$ | MDE at $\sigma_d=1.5$ (the plan's assumption) | MDE at $\sigma_d=0.2975$ (MEASURED dense $\hat\sigma$, §3.7) |
|---|---|---|---|---|---|
| 3 | 4.303 | 1.061 | 3.10 | 4.6 pp | 0.92 pp |
| 5 | 2.776 | 0.941 | 1.66 | **~2.5 pp** (as planned) | **~0.49 pp** (as measured) |
| 8 | 2.365 | 0.896 | 1.15 | 1.7 pp | 0.34 pp |

⚠ The last column previously read 1.4 / 0.7 / 0.51 pp, computed from the superseded $\hat\sigma = 0.44$. It is recomputed here at $\hat\sigma = 0.2975$ (§3.7).

The $n=5$ row is the headline consequence of the measured noise floor: a 1-point component effect moves from undetectable to comfortably detectable, which is the difference between the ladder answering its question and not.

*Provenance note:* an earlier power table circulated with an added $\rho$ column whose $t_{.80}$ entries (0.817 / 0.741 / 0.711 at $n=3/5/8$) do not match the $t$ distribution ($t_{.80,2}=1.061$, $t_{.80,4}=0.941$, $t_{.80,7}=0.896$). **The table above supersedes it.** The pairing generalisation is $\sigma_d = \sigma\sqrt{2(1-\rho)}$, with $\rho$ the between-arm seed correlation ($\rho=0$ for unpaired Stage A). **$\rho$ has not been measured**; it becomes estimable as soon as two arms share seeds.

⚠ **Three caveats that must travel with this table.**

1. $\hat\sigma=0.2975$ is the **dense** A0 arm. Sparse-rung variance at $0.999$ is **not measured** - every sparse cell attempted so far diverged (§8.1) - and there is no reason to expect the mask lottery to be as quiet as dense training. Read the $\sigma_d=0.2975$ column as a best case. It is also an $n=5$ estimate whose own 95% interval runs $[0.18,\ 0.85]$, so the MDE column inherits that width.
2. $\sigma_d$ for a paired difference equals $\hat\sigma$ only at a particular pairing correlation; the substitution $\sigma_d\approx\hat\sigma$ is an assumption.
3. In the collapse regime (99.99%, and currently 99.9% - §8.1) the distribution is a **mixture, not a Gaussian**, and no $t$-based MDE applies; report **median, IQR and collapse rate** there.

## 7.13 Perplexity, and the Jensen bias

**What is currently reported:**

$$\mathrm{PPL}_{\text{reported}} \;=\; \frac{1}{B}\sum_{b=1}^{B}\exp\!\bigl(\ell_b\bigr)$$

per-batch exponential at `project/trainer.py:336`, accumulated at `:280` and divided by the batch count at `:286` (mirrored for the BaCP fine-tune phase at `bacp.py:825`, `:637`).

Since $\exp$ is strictly convex, **Jensen's inequality** gives

$$\frac{1}{B}\sum_b \exp(\ell_b)\;\ge\;\exp\!\left(\frac1B\sum_b \ell_b\right),$$

with equality **iff every $\ell_b$ is identical**. The reported figure is therefore **biased upward**, and the bias grows with the across-batch variance of the loss - to second order, $\mathbb{E}[e^{\ell}]\approx e^{\bar\ell}\bigl(1+\tfrac12\mathrm{Var}(\ell)\bigr)$. A noisier model is penalised twice.

**What fixes it** (`results.evaluate_exact`, `results.py:296-385`):

$$\mathrm{PPL}_{\text{exact}} \;=\; \exp\!\left(\frac{\sum_{b}\sum_{i\in b}\ell_i}{\sum_{b}\lvert b\rvert}\right)$$

Cross-entropy accumulated with `reduction='sum'` (`:343`), token/sample count accumulated separately (`:345`), single exponential at the end (`:376-380`). Three further things this repairs:

1. **Accuracy** becomes $\text{correct}/\text{total}$ rather than a batch-mean of batch-means - equivalent only when every batch is the same size, which a partial final batch breaks (`:344`).
2. **MLM token weighting**: positions with `label == -100` are excluded before the sum, so the denominator is the number of *scored* tokens, not the number of sequences (`:332-338`).
3. **Overflow**: `safe_exp` (`:388-402`) returns `inf` rather than raising, since `math.exp` overflows above $\approx 709$ and a diverged model easily exceeds that. Perplexity is a reporting convenience and must never destroy a completed run's record.

⚠ `trainer.py:290` returns `perplexity` as `None` when the average is $\le 1.0$ - a reporting guard, not a statistic; do not let it appear in a table as a missing value without explanation.

⚠ **Pseudo-perplexity for masked language modelling is not the standard causal quantity** and needs defining explicitly wherever it is reported. Relatedly, RoBERTa and DistilBERT are **encoder-only**, not "decoder-only" as the submitted appendix states (`docs/math.md` §7.3).

## 7.14 Auxiliary formulas present in the code

| Quantity | Formula | Code | Citation | Note |
|---|---|---|---|---|
| DyReLU phasing | $\mathrm{out}=\beta_t\,\mathrm{DyReLU}(x)+(1-\beta_t)\,\mathrm{ReLU}(x)$, $\ \beta_t=\mathrm{clip}\!\left(1-\frac{t-t_{\text{start}}}{t_{\text{end}}-t_{\text{start}}},0,1\right)$ | `dyrelu_adapter.py:74-104` | EAST (R14) - **UNVERIFIED for this use.** `docs/citations.md` §9 verifies EAST for the baseline table, the protocol, the author-list and version hazards, and the two things the paper does not state. **DyReLU phasing is not among them**, so this row has no checked citation (§6.2, R14) | ⚠ `docs/math.md` §6.1 says the module is "**exactly** ReLU" at initialisation $a=[1,0]$, $b=[0,0]$. That holds only if the hyperfunction's sigmoid emits exactly $0.5$ so that $\theta=0$ after the $2\sigma(\cdot)-1$ rescale (`:34`). At random init it does not, so the anneal introduces a **small** discontinuity, not none. Narrow the claim. Placement and parameter counts: §1.5. |
| DyReLU-B | $\mathrm{out}_c=\max_k\left(a_c^k x_c+b_c^k\right)$, $(a,b)$ per-sample from a hyperfunction | `dyrelu_adapter.py:45-59` | Chen et al., ECCV 2020 (**R29** - **UNVERIFIED**: neither "DyReLU" nor "Dynamic ReLU" occurs in `docs/citations.md`, so the formula's attribution is unchecked) | Hyperfunction parameters are excluded from the pruning mask (`pruning_factory.py:23`), correct **only** because phasing removes them from the forward pass - at $\beta=0$ the adapter only *bypasses* the submodule, so it remains in the state dict, and a 2048-channel DyReLU-B hyperfunction is $\approx 5.2$M parameters. **Any reported parameter count or checkpoint size must account for this separately from the sparsity figure.** Defect M5, §8.6. |
| Projection-head degeneracy | $\min_{\theta,g}\mathcal{L}^{PrC}\bigl(g(f_\theta(x)),\hat g(\hat f(x))\bigr)$ admits solutions with $\theta$ unchanged | `model_factory.py:312-338` | Dasgupta & Gupta (**R28** - **UNVERIFIED**, absent from `docs/citations.md`), for the JL bound **only in its narrow use**: why a never-trained `encoder_head` is lossy rather than vacuous. It does **not** support "the distortion cancels on both sides of the loss" - that follows from student and teachers sharing one fixed map, and is derived in §1.4, not cited | The argument and the 308% figure: §1.4. Verified by `test_projection_head_cannot_absorb_the_objective` (`project/tests/test_bacp_loop.py:167`). |
| CIFAR stem MACs | `maxpool` to `Identity` multiplies MACs by **3.98** ($0.2912\to1.1594$ GMAC at $32\times32$) and changes **zero** parameters and **zero** state-dict keys | `model_factory.py:181-183`; `models/resnet.py:174` | - | Full statement, spatial trace and decision: §1.3.3. Cost consequences: §4.5. |

---

# 8. Open defects

Everything here is **open**. Nothing in this section is described as diagnosed, and nothing in it has been published. Items are listed as defects even where the intended behaviour is documented elsewhere in this file - the intent is not the state.

## 8.1 NaN divergence in sparse training at 0.999 - CAUSE NOT ESTABLISHED

The blocking defect. MEASURED on the archived first tier-1 attempt (`/workspace/ARCHIVE_results_nan_20260819`):

| arm | seeds NaN | onset |
|---|---|---|
| A0 (dense) | 0 of 5 | - |
| C-null (BaCP path, 0.999) | **5 of 5** | ~epoch 10 |
| C0 (CE only, 0.999) | **2 of 3 completed** | ~epoch 64 |

Diverged runs sit at exactly $10.0969\%$ - chance on CIFAR-10 - for the remainder of training, with loss printed as `nan`, and reported sparsity drifts $0.9990 \to 0.9988$ at the moment of divergence.

**Contributing facts established. None of them has been shown to be the cause.**

| fact | location |
|---|---|
| **No gradient clipping exists anywhere.** The call is commented out. | `project/training_utils.py:598` (§8.5) |
| fp16 AMP is **on by default** with a `GradScaler`, and is not a protocol field | `project/trainer.py:44` (§3.2) |
| **The pruner runs between `scaler.unscale_()` and `scaler.step()`** | `project/training_utils.py:505`, `:513-517` |
| lr 0.1 with **no warmup** on the cosine branch | `manifest.py:96`, `training_utils.py:343-350` |
| the sparsity drift $0.9990 \to 0.9988$ coincides with divergence, i.e. the mask changes at the moment the loss dies | recorded in the archived records |
| **Only RigL has ever been exercised at scale.** Magnitude, EAST, SNIP and WANDA are untested at these sparsities - and rungs A1-A4 and A6 all depend on them | §7.9 |

Consequences that hold until this is resolved: **no sparse-arm variance has been measured**, so the $\sigma_d = 0.2975$ column of the MDE table (§7.12) is a dense-arm best case; and in a collapse regime the outcome distribution is a mixture rather than a Gaussian, so $t$-based intervals do not apply and median / IQR / collapse-rate reporting is required (§3.4, §7.12).

Tracked as task #20.

## 8.2 The gates have no collapse floor

`evaluate_gates` G1 tests only $\lvert \bar{x}_{\text{C-null}} - \bar{x}_{\text{C0}}\rvert \le 0.5$ (`project/experiments/ladder.py:416`). **With both arms diverged to NaN and sitting at exactly $10.0969\%$ it computes $\lvert 0.00 \rvert \le 0.5$ and reports PASS on two dead runs.** The gate that exists to prove the BaCP code path reproduces plain CE cannot distinguish "identical" from "identically broken". This is not hypothetical - it is the state the first tier-1 attempt was actually in.

The same structural hole exists in **G3, G4 and G5**: all three are difference-based (`ladder.py:426-455`, `:457-471`, `:473-485`) and none checks that either arm is above chance.

**Prescription, not implemented:** an absolute floor - both arms must exceed $100/\text{num\_classes} + k\hat\sigma$ - plus a NaN-loss check read off the record. Note `test_metrics_are_in_range` (`test_experiment_invariants.py:146`) already fails a NaN metric, but it runs only under `pytest -m results` and does not gate the sweep.

Tracked as task #21.

## 8.3 Fidelity gaps between the written claim and the shipped configuration

### (a) The ERK fidelity gap - flag F8

`calculate_erk_densities` (`project/pruning_factory.py:196-203`) uses layer mass $n_{in}k_hk_w + n_{out}$ - Erdos-Renyi over the **unrolled** weight matrix - while Evci et al.'s ERK is $n^{l-1}+n^{l}+w^{l}+h^{l}$, which **adds** the kernel extents. The two coincide for `Linear` and differ for every convolution. **The code cites ERK and implements ER-on-unrolled.**

At target 0.999 with the 3x3 CIFAR stem, `conv1` receives density **0.015352**, i.e. about **26.5 of its 1728** weights survive; at 0.9999 about **2.7**, where a completely dead stem is a realistic Bernoulli outcome. The **global** parameter budget is held exactly (MEASURED `eff_sparsity` 0.999000). Full derivation: §7.7. Citation prescription: §6.3 F8.

**Status: unfixed.** Either implement Evci's mass and re-derive, or keep the code and re-word every citation. Affects A3-A6 and, through `BEST_SUBSTRATE`, all of Stage C and D.

### (b) C2 -> C3 does not isolate one mechanism

The manifest asserts the step "isolates ONE thing: the presence of negatives" (`manifest.py:370-374`), and `bundle_ok` on C3 makes that claim in writing (§5.1). But C3 constructs `CAPContrastiveLoss` with both `supervised=True` and `unsupervised=True` (`project/bacp.py:189`, defaults at `project/loss_functions.py:78`), so the step adds **the log-sum-exp denominator** *and* **label-defined positives** that C2 has no analogue of. That is two changes under one label - the exact error the ladder exists to avoid, sitting on the paper's central claim and on gate G4.

**Status: unresolved.** Options: run C3 with `supervised=False`, or insert a C3a rung between C2 and C3. Derivation of what the clean contrast would be: §7.5.

## 8.4 `resolve_checkpoint` has no provenance guard

`resolve_checkpoint` (`project/results.py:1150-1200`) is what binds every sparse cell to its dense predecessor, via `attach_checkpoint` (`project/experiments/runner.py:145-180`). It filters candidate records on four fields only - `status == 'ok'`, `model_name`, `dataset_name`, `method` (`results.py:1167`) - plus `seed`, and **only when `seed is not None`** (`:1169`). It then returns the most recent surviving candidate by `ended_at` (`:1180`).

**What it does not check:**

- **`code_fingerprint`.** Every record carries one (`results.py:501`, written at `:853` and `:1067`), and the identical staleness check already exists elsewhere in the same file - `require_gate` refuses a gate whose fingerprint no longer matches the current code (`results.py:1248-1252`, "it stops a 40-hour result silently satisfying a gate after the code underneath it changed"). `resolve_checkpoint` performs no such comparison, so **a sparse run will happily start from a dense checkpoint produced by different source code**, which is exactly the condition `test_all_comparable_runs_share_one_code_fingerprint` (`test_experiment_invariants.py:112`) exists to catch *after the fact*.
- **`git_sha` / `git_patch_sha1`** (`results.py:495`), for the same reason.
- **Protocol fields.** A dense checkpoint trained under a different `epochs`, `learning_rate` or `scheduler_type` resolves as readily as the right one.
- **Seed, when the caller passes `None`.** `attach_checkpoint` deliberately retries with `seed=None` when the same-seed lookup fails, printing `'! no dense checkpoint for seed N; falling back to ...'` (`runner.py:163-171`). The pairing is then broken for that cell, and **every paired CI computed from that rung is overstated** - a warning line in a 65-cell sweep is the only signal at run time. `test_sparse_runs_chain_to_a_checkpoint_of_the_same_seed` (`test_experiment_invariants.py:468`) catches it afterwards, under `-m results`.

**Prescription, not implemented:** compare `code_fingerprint` (and optionally the protocol keys) inside `resolve_checkpoint`, refuse or loudly downgrade on mismatch, and make the seed fallback opt-in rather than automatic. The 4-root relative-path rehydration at `results.py:1183-1193` is a separate, deliberate legacy accommodation and is not the defect.

**Status: open.** This defect is silent by construction: nothing raises, and the symptom appears only as an unexplained delta one rung later.

## 8.5 No gradient clipping anywhere

`torch.nn.utils.clip_grad_norm_` appears exactly once in the repository and it is commented out:

```
project/training_utils.py:598
#     # torch.nn.utils.clip_grad_norm_(args.model.parameters(), max_norm=1.0)
```

There is no clipping on the baseline path, the pruning path or the BaCP path, and no `max_grad_norm` field on any config or on `RunRecord`. Combined with lr 0.1, no warmup, fp16 AMP on by default, and a pruner that mutates weights inside the `GradScaler` window, this is one of the established contributing facts to §8.1 - **but it has not been shown to be the cause, and re-enabling it is a hypothesis to test, not a fix to announce.**

## 8.6 Model-layer register

| # | defect | location | status |
|---|---|---|---|
| **M1** | **The CIFAR stem keeps its `maxpool`.** `conv1` is already 3x3 stride-1 at `image_size <= 64`, but `self.maxpool` is left in the forward pass, so a 32x32 input runs layer1@16 / layer2@8 / layer3@4 / **layer4@2** instead of 32/16/8/4. The final stage - 61.6% of the backbone - operates on four spatial positions. MACs are 3.98x too low. | `model_factory.py:181-183`; `models/resnet.py:174`, `:241` | **Fix decided, not applied.** `maxpool -> Identity` changes zero parameters and zero `state_dict` keys, so §1.3.2, §3.5 and §7.7 are unaffected and all checkpoints stay loadable. Invalidates the measured A0 as a final number and every timing in §4.3-§4.4 (§4.5). Full statement: §1.3.3. |
| **M2** | The bare substring `'classifier'` in `_EXCLUDE_TASK_HEAD` removes VGG's entire 119.5M-parameter MLP from the pruning denominator, not just the final layer. A "99% sparse VGG-11" would leave 92.8% of the network untouched and dense. | `pruning_factory.py:44` + `:139-142` | Open. Anchor unaffected; no VGG run performed. §1.7. |
| **M3** | `resnet34_wide` is **uncallable** - it sets `width_per_group = 128`, which `BasicBlock.__init__` rejects unconditionally, so the function raises `ValueError` on its own default arguments. VERIFIED by construction. | `models/resnet.py:273-280` vs `:17-18` | Open, dead code. Delete or fix `BasicBlock`'s guard. |
| **M4** | `wrrn22` builds (9,663,178 params) but cannot be wrapped: it exposes only `self.layers` and has no `fc`, `classifier`, `head` or `conv1`, so `_get_embedded_dim_from_model` and `remove_last_layer` both raise and `adapt_resnet_for_small_images` is a silent no-op. VERIFIED (all four attribute checks `False`). | `models/resnet.py:368-419`; `model_factory.py:116-136`, `:213-236` | Open, dead code. Registering it as-is would fail at construction. |
| **M5** | Under rung B1 (`dyrelu_en=True, dyrelu_phasing_en=False`) **3,181,168 DyReLU parameters stay in the forward pass for the whole run** yet are excluded from the sparsity denominator - 150x the 21,265 surviving backbone weights. The exclusion's stated justification (`pruning_factory.py:20-22`) holds only for the *phasing* variant B2. | `pruning_factory.py:22` + `manifest.py:274` | Open; code-reading finding, not from a run. B1 is `priority='optional'` and unrun. §1.5, §7.14. |
| **M6** | VGG at 32x32 collapses to 1x1 through five maxpools and is silently re-inflated to 7x7 by `AdaptiveAvgPool2d`, so `Linear(25088, 4096)` sees 49 copies of one 512-vector. No error is raised. `adapt_resnet_for_small_images` never runs for VGG. | `model_factory.py:264`; `vgg.py:21` | Open; no VGG run performed. §1.7. |
| **M7** | Private torchvision symbols (`_ovewrite_named_param`, `WeightsEnum`) imported at module scope, so a torchvision version bump breaks `import model_factory` wholesale rather than only VGG. | `models/vgg.py:5` | Open; version-coupling risk. |

## 8.7 Data-layer register

| id | severity | status | detail |
|---|---|---|---|
| **D1** | blocking for `wikitext2` | **CONFIRMED this session** | `load_mlm_dataloaders` (`dataset_factory.py:508-509`) takes no `seed` parameter, but `load_text_dataloaders` calls it with `seed=seed` (`:560`), giving `TypeError: unexpected keyword argument 'seed'`. Even past that, the nested `loader()` closure references `seed` at `:542`, which resolves as a module global that does not exist (verified by `symtable`: `is_global=True`, no module-level `seed` binding), giving `NameError`. **WikiText-2 is unreachable today.** The GLUE path is unaffected - `load_glue_dataloaders` declares `seed=0` at `:472`. |
| **D2** | wrong data, silent | flagged, not executed here | `torchvision.datasets.EMNIST.__init__(self, root, split, **kwargs)` absorbs `train` into `**kwargs`, so `'train' in sig.parameters` is `False` and `get_dataset_test_fn` (`:260-261`) never sets `train=False`. The inherited `MNIST` default `train=True` then applies, and the `split='test'` written at `:263` is overwritten by `split='balanced'` at `:264-265`. Net effect: **the EMNIST "test" loader returns the 112,800-image training split**, scored with the eval transform. Read off the code path; not executed, because torchvision is not importable in the local checkout environment. **Verify on the run box before any EMNIST run.** |
| **D3** | crash | flagged, not executed here | `get_dataset_test_fn` passes `download=True` unconditionally (`:258`). `torchvision.datasets.ImageNet` carries a `download` parameter that raises `RuntimeError` when it is `True` ("no longer publicly accessible"). If so, `imagenet` is registered and CLI-accepted but unreachable through this factory. Unverified here for the same reason as D2. |
| **D4** | uncontrolled arm difference | open, by design | The contrastive path builds no validation split (`:340-341`), so BaCP's contrastive stage trains on 50,000 CIFAR-10 images against the CE baseline's 40,000. Its fine-tuning stage does use the 40,000/10,000 split. Not a test leak; **is** a confound between arms and must be stated in the paper. §2.4. |
| **D5** | latent, wrong augmentation | open | The fallback branch (`:184-198`) applies `RandomHorizontalFlip()` to `mnist`, `fmnist` and `emnist`. Mirroring destroys digit and character identity; `dataset_utils_old.py:109-113` deliberately used `RandomRotation(10)` and no flip for exactly these three. Harmless today (only CIFAR-10 is run) but wrong the moment the ladder extends. |
| **D6** | memory | open | `deepcopy(valset)` at `:338` duplicates the full underlying training array in host memory (153.6 MB uint8 for CIFAR-10). Required for correctness (§2.4); not measured as a bottleneck at 8 concurrent cells. |
| **D7** | provenance | resolved, but must be reported | `drop_last=True` on val/test scored 9,728 of 10,000 CIFAR-10 test images with batch-size-dependent membership. Fixed at `:373-380`. **Every pre-fix ResNet-34 number, including all AAAI-submission numbers, was produced under it and is VOID.** §2.7. |

## 8.8 Plan, statistics and tooling register

| defect | location | status |
|---|---|---|
| **`D4-noPrC` is `D1`.** The two configs differ in exactly one key - `experiment_type` (`'ladder-D4'` vs `'ladder-D'`) - because $(0,1/3,1/3,1/3)$ with `num_snapshots=2` is what D1 already runs. Three consequences: (i) `validate()`'s duplicate-config detector (`manifest.py:871-885`) does not fire, because it hashes a signature that includes `experiment_type`, a label derived from `rung.stage` (`manifest.py:540`) with no effect on training; (ii) the PrC row of the forward-vs-backward table is **zero by construction, not by measurement**, since $D2-D1$ and $D2-\text{D4-noPrC}$ are the same contrast with opposite sign, so nothing about PrC's interaction structure is learnable from it; (iii) 5 cells x 2.2 U at tiers 2 and 3 of duplicated compute. The other three D4 arms have no forward twin and their gaps are informative. | `manifest.py:411-429` | Open. Clean fix: alias D4-noPrC's records to D1's cell key, or drop the arm and report the D1-D2 contrast in both columns. **Not done.** §5.6, §5.7. |
| **The headline sparsity sweep is declared but never scheduled.** `HEADLINE_SPARSITIES` is defined and referenced nowhere else; `cells()` accepts `sparsity=` but no caller passes it (`pool.py:182`), and `TIERS[3]['why']` claims tier 3 includes the sweep while tier 3 resolves every cell at `LADDER_SPARSITY`. | `manifest.py:69`, `:591`, `:608`, `:580` | Open. Needs a driver before it can run. §3.4. |
| **`budget()`'s per-script timing loop is a no-op** - `for script in ('pruning', 'bacp', 'dense'): pass` - so every cost projection uses one median across dense, sparse and BaCP cells, which its own caveat string admits "understates BaCP rungs and overstates dense ones". | `manifest.py:679-680`, caveat `:710-714` | Open. Measured correction factor for the one BaCP config timed: 1.95x (§4.3). |
| **Stage A's Welch intervals use a hardcoded $t = 2.0$ when scipy is absent**, with nothing recorded in `_T_SOURCE`. At Welch df near 4-8 the true critical values are 2.78 and 2.31, so intervals print up to **~28% too narrow with no warning**. `sigma_chi2_interval` silently returns `(None, None)` on the same path. `scipy` is in `requirements-dev.txt` but **not** `requirements.txt`. | `ladder.py:251-255`, `:159-160` | Open. The paired path is protected by `_T95` (`:171-177`) and `_T_SOURCE`; the unpaired path is not. §7.12. |
| **The learnable-$\lambda$ arm is not constrained to the simplex.** `_initialize_lambdas` unpacks the tuple with no projection, softmax or renormalisation; under `learnable_lambdas=True` the four values become raw `nn.Parameter`s free to leave $[0,1]$ with an unconstrained sum. | `bacp.py:245-262`, `:251`, `:257` | Open. Any claim that this arm optimises over the simplex is unsupported by the code. §7.1. |
| **Momentum is a literal, not a recorded field.** Hardcoded 0.9 at `training_utils.py:316` and absent from `RunRecord` (`results.py:579-595`), so a run under different momentum is indistinguishable in the record. fp16 AMP (`trainer.py:44`) is likewise on by default and absent from `PROTOCOLS`. | `training_utils.py:316`; `trainer.py:44` | Open. Same defect class as the weight-decay hardcoding that was already fixed. §3.2. |
| **`BasePruner.cyclic_scheduler` is dead code with the wrong phase** - it implements the $s_{\min}$-anchored cosine that `EASTPruner` explicitly rejects, and has no call sites. | `pruning_factory.py:318-320` | Open. Delete or correct before anyone wires it up. §7.10. |
| **G0, G0b, G2 and G6 have no definition anywhere in the codebase.** G0b's 2.0-point threshold in particular is written in no file; `noise_floor()` computes the statistic and nothing compares it to anything. | `ladder.py:368-485` (four gates only) | Open. §5.9. |
| **No gate ordering enforcement.** `report()` evaluates all implemented gates unconditionally, so a G4 STOP prints alongside a G1 STOP that invalidates it. | `ladder.py:649-661` | Open, low severity - `run_ladder.py:120` halts on any STOP. §5.9. |
| **Stale comments** that will mislead a reader: `manifest.py:471` claims `PROTOCOLS` carries `delta_T` (it does not; behaviour is correct); `pool.py:25-28` quotes "3 GB of 24 GB" against a measured 815 MB of 15 GB; `run_remote.sh:50` attributes `/workspace` to RunPod on a Vast.ai box; `A0`'s `note` at `manifest.py:192-194` is truncated mid-sentence and must be repaired before it is quoted in any caption; `model_factory.py:160-163` asserts EAST trains CIFAR from scratch, which EAST's paper does not state (§1.1, §6.4c) and which must not be asserted on its authority. | as listed | Open, documentation only. |
| **No TF32 policy is set anywhere**, and the **NVIDIA driver version is not recorded**. `results.py:205-212` captures torch/CUDA/cuDNN versions, GPU name and count, both cuDNN determinism flags and `deterministic_algorithms`; matmul/conv TF32 behaviour is inherited from whatever the installed torch defaults to on Blackwell and appears in no record. | `results.py:205-212`; no `allow_tf32` or `set_float32_matmul_precision` call exists in the repository | Open. Pin a policy explicitly and record the driver before the reproducibility statement is written. §4.1. |
| **`docs/citations.md` contradicts itself on its own flag count** - the intro says "Three such cases were found", the closing summary table lists seven. §6.1 works from the seven-row table and says so, but the dossier should be fixed at the source. | `docs/citations.md:9` vs its *Summary of mis-citation flags* table | Open, documentation only. §6.1. |
| **Sixteen references are cited by this document and appear nowhere in `docs/citations.md`** - R27 (MoCo), R28 (JL), R29 (Dynamic ReLU), and R30-R46, the architectures, datasets and Welch-Satterthwaite added in this revision. They now carry status `UNVERIFIED` rather than `OK`, which is accurate but is not the same as having been checked. R28 and R29 are the pressing ones: they underwrite §1.4's fairness argument and §7.14's DyReLU-B formula respectively. | §6.2 | Open. Verify against the primary sources and fold them into the dossier before any `.bib` is generated. |
| **The headline `A0` figure `91.68 ± 0.44` did not reconcile with its own per-seed data** and was propagated into eight sections plus the MDE table and the $\chi^2$ interval. Corrected throughout to mean 91.6353, $\hat\sigma$ 0.2975 (§3.7). The lesson is mechanical, not clerical: the number was quoted rather than recomputed, and nothing in the pipeline recomputes a summary statistic from the record set at table-build time. | §3.7 | Corrected in this document. `ladder.aggregate()` computes these from records and should be the only source of any figure that appears in a caption. |
| **`run_remote.sh` defaults `PERGPU` to 1**, not the measured 2, so `start` launches 4 slots and roughly half the throughput unless `BACP_PER_GPU=2` is set. | `run_remote.sh:128` | Open, operational. §4.2. |
| **No protocol is defined for SST-2**, whose `testloader` is literally its `valloader` (GLUE test splits are unlabelled). Any early stopping on validation would be model selection on the reported number. | `dataset_factory.py:505`, `:494-497` | Open. Settle before any text run. §2.9. |
