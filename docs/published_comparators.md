# Published numbers we can be measured against

Every row is for a model **this project builds**. Fetched from the papers'
own results tables on 2026-08-22 and quoted exactly. Rows marked ✅ are a
direct match for our (model, dataset, criterion, sparsity) and are the ones a
reviewer will hold us to.

Sparsity convention: fraction of weights REMOVED.

---

## 1. ResNet-50 / CIFAR-10 — GraNet Table 2

Recipe: 160 epochs · batch 128 · SGD(0.9) LR 0.1 · ×10 at [80,120] · wd 5e-4.
Source: Liu et al., *Sparse Training via Boosting Pruning Plasticity with
Neuroregeneration*, NeurIPS 2021, arXiv:2106.10404.

| method | dense | 95% |
|---|---|---|
| dense baseline | **94.75 ± 0.01** | — |
| **GMP** (gradual magnitude pruning) | | **94.52 ± 0.08** ✅ |
| SNIP | | 90.86 |
| GraSP | | 91.32 |
| SynFlow | | 91.22 |
| RigL | | 93.86 ± 0.25 |
| GraNet (s_i=0.5) | | 94.38 ± 0.28 |
| GraNet (s_i=0) | | 94.44 ± 0.01 |

**✅ GMP @ 95% = 94.52 ± 0.08 is our exact cell**: ResNet-50, CIFAR-10,
gradual magnitude pruning, 95% sparsity. It is the single most important
number in this file — our I.P. magnitude baseline should land near it under a
matched protocol, and if it does, every BaCP gain measured against that
baseline is credible.

## 2. ResNet-50 / CIFAR-100 — GraNet Table 2

| method | dense | 90% |
|---|---|---|
| dense baseline | **78.23 ± 0.18** | — |
| GMP | | 76.91 ± 0.23 |
| SNIP | | 73.14 |
| GraSP | | 73.28 |
| SynFlow | | 73.37 |
| RigL | | 76.50 ± 0.33 |
| GraNet (s_i=0) | | 77.29 ± 0.45 |

## 3. VGG-19 / CIFAR-10

GraNet Table 2 (dense 93.85 ± 0.05) and GraSP Table 2 (dense 94.23 — the two
papers disagree on the dense baseline; see the caveat below).

| method | 90% | 95% | 98% | source |
|---|---|---|---|---|
| GMP | 93.59 ± 0.10 | — | — | GraNet |
| SNIP | 93.63 | 93.43 ± 0.20 | 92.05 ± 0.28 | GraNet / GraSP |
| GraSP | 93.30 | 93.04 ± 0.18 | 92.19 ± 0.12 | GraNet / GraSP |
| SynFlow | 93.35 | — | — | GraNet |
| SET | 92.46 | — | — | GraNet |
| RigL | 93.38 ± 0.11 | — | — | GraNet |
| GraNet (s_i=0) | 93.80 ± 0.10 | — | — | GraNet |
| OBD | 93.74 | 93.58 | 93.49 | GraSP |
| LT (lottery ticket) | 93.51 | 92.92 | 92.34 | GraSP |

## 4. VGG-19 / CIFAR-100

| method | 90% | 95% | 98% | source |
|---|---|---|---|---|
| GMP | — | — | 72.07 ± 0.37 | GraNet |
| SNIP | 72.84 ± 0.22 | 71.83 ± 0.23 | 58.46 | GraNet / GraSP |
| GraSP | 71.95 ± 0.18 | 71.23 ± 0.12 | 68.90 | GraNet / GraSP |
| SynFlow | — | — | 70.94 | GraNet |
| RigL | — | — | 69.82 ± 0.09 | GraNet |
| GraNet (s_i=0) | — | — | 72.35 ± 0.26 | GraNet |
| OBD | 73.83 | 71.98 | 67.79 | GraSP |
| LT | 72.78 | 71.44 | 68.95 | GraSP |

## 5. ResNet-34 / CIFAR-10 — EAST Table 1

Dynamic sparse training only, at extreme sparsity. Recipe: 250 epochs · batch
128 · SGD(0.9) LR 0.1 · wd 1e-4 · ERK · 3 runs.
Source: Li et al., *Pushing the Limits of Sparsity*, TMLR 2025,
arXiv:2411.13545 v4.

| sparsity | RigL | EAST | SET | SynFlow |
|---|---|---|---|---|
| 99% | 92.92 ± 0.18 | 93.51 ± 0.13 | 93.09 ± 0.15 | 86.03 ± 0.71 |
| 99.9% | 85.71 ± 0.23 | 86.99 ± 0.32 | 82.70 ± 0.91 | 61.61 ± 10.76 |
| 99.95% | 81.47 ± 0.32 | 83.83 ± 0.02 | 70.84 ± 1.51 | 56.33 ± 3.44 |
| 99.99% | 10.03 ± 0.19 | 62.12 ± 0.90 | 10.00 ± 0.00 | 10.00 ± 0.00 |

No magnitude/static row; no dense row. Not a comparator for a static paper
except as context for what extreme sparsity costs.

---

## What we have NO published comparator for

- **ResNet-34 / CIFAR-10 static magnitude** at any sparsity
- **VGG-11** anywhere
- **ResNet-101** anywhere
- **ViT-Tiny / ViT-Small** on CIFAR-10 at any sparsity
- **DistilBERT / RoBERTa** on SST-2 at any sparsity
- Any model at **0.97 or 0.99** except EAST's dynamic ResNet-34

Numbers for these must be reported standalone, explicitly labelled as having
no published comparator.

## Caveats that must travel with these numbers

1. **The two papers disagree on VGG-19/CIFAR-10 dense**: GraNet 93.85 ± 0.05,
   GraSP 94.23 (no std). Do not mix their rows without saying so.
2. **GraSP's ResNet rows are ResNet-32, not ResNet-50** — 2× width in some
   configurations. They must not be used as a ResNet-50 comparator.
3. **GraNet's CIFAR pipeline is internally inconsistent** and undocumented in
   the paper: CIFAR-10 pads with *reflect*, CIFAR-100 with zeros; it trains
   CIFAR-10 on **45,000** images (0.1 validation split) and uses **Nesterov**
   momentum. If our dense lands below parity, check these three first.
4. **Sparsity grids differ from ours.** The literature evaluates
   **90 / 95 / 98**; we run 95 / 97 / 99. Only **95%** overlaps. To be
   comparable at more than one point, our grid should include 90 and 98.
5. **GMP is our magnitude criterion under a different name** — "gradual
   magnitude pruning", Zhu & Gupta's cubic schedule applied to |W|. That is
   what our `magnitude` pruner implements, which is why row 1 is a true match.
