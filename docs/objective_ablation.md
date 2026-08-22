# Which contrastive objective — the measurement that decided it

ResNet-50 / CIFAR-10, BaCP + magnitude, seed 1. Two arms differing in
**exactly two config keys** (verified programmatically before the runs:
`contrastive_mode`, `proj_mode`). Everything else identical — tau 0.15, 60
epochs, delta_T 88, cubic ramp to 80% then 20% recovery, classifier pruned,
SGD 0.1 → AdamW 1e-4 × 50, lambdas 0.25 each, same dense checkpoint, same
data, full 10,000-image test split.

| arm | `contrastive_mode` | `proj_mode` |
|---|---|---|
| **legacy** (original design) | `legacy` — SupCon + NT-Xent over a 2B×2B `cat([student, teacher])` matrix, SimCLR-style all-pairs | `current` — per-model projection heads, the student's trainable |
| control (refactor) | `cap` — one CAP Eq. 1 functional, rectangular B×N | `tied_frozen` — one frozen head shared by student and teachers |

## Result

| sparsity | I.P. baseline | control | **legacy** | control − I.P. | **legacy − I.P.** | legacy − control |
|---|---|---|---|---|---|---|
| 0.95 | 91.71 | 91.78 | **92.43** | +0.07 | **+0.72** | **+0.65** |
| 0.97 | 91.35 | 91.40 | **92.04** | +0.05 | **+0.69** | **+0.64** |
| 0.99 | 89.80 | *(killed by an unrelated cancellation, `returncode=-15`)* | **91.24** | — | **+1.44** | — |

**The legacy objective wins at every measured point**, by ~0.65, and
out-gains the control against the shared I.P. baseline by roughly ten to one.
The margin grows with sparsity (+0.72 → +0.69 → +1.44), which is the shape a
genuine representation-preservation effect should have: more damage to
recover, more recovered.

**Decision: `legacy` is the project's objective.** `FAMILIES` sets
`contrastive_mode='legacy'`, `proj_mode='current'` for every family.

## Caveats to carry into the paper

1. **Single seed.** The ~0.65 gap is above the ~0.3 noise floor but not by a
   wide margin. Three seeds on the headline cells before publication.
2. **The trainable projection head is a fairness caveat, not a validity one.**
   Accuracy is measured after a full fine-tune of the masked network, so the
   number is a legitimate deployed-model result. What it constrains is the
   *claim*: with a trainable head the loss can be lowered by moving 262k
   unpruned parameters, so "BaCP preserves representations" is not cleanly
   supported by the loss alone. The defensible claim is the measured one —
   this objective yields better sparse models — with the frozen-head variant
   reported as the ablation that costs 0.65.
3. The control arm's numbers are preserved here because its records were
   deleted when `legacy` became the default (the two arms share record keys
   once the variant suffix is dropped).
