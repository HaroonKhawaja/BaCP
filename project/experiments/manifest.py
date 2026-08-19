"""The ablation ladder, declared as data.

One rung changes one thing and inherits everything else from the rung below it.
That property is enforced *by construction* here -- `resolve()` walks the
`inherits` chain and applies each rung's `overrides` in order -- rather than by
hand-copying command lines, which is how the submitted paper's ablation ended up
changing two things per arm (zeroing one lambda while leaving the other three at
0.25, so "no PrC" was simultaneously a removal and a 25% downweighting of the
whole contrastive signal).

The ladder answers a different question from the submitted paper's ablation. That
one was a leave-one-out star around full BaCP: four arms, all backward, measuring
*necessity at the all-on corner* and nothing else. It cannot localise where
accuracy starts to be lost on the way up to BaCP, it cannot distinguish a
redundant component from an inert one, and with p arms and p main effects it has
zero residual degrees of freedom, so no interaction is estimable even in
principle. Here the forward ladder is primary and the leave-one-out pass is a
second sweep run only over components that survive it.

Reading order:
    ANCHOR / PROTOCOLS  -- where the ladder runs and under what recipe
    LADDER              -- the rungs
    resolve()           -- rung id -> a complete, flat config
    cells()             -- rungs x seeds -> the units of work the runner executes
    budget()            -- measured wall-clock -> GPU-hour projection

Nothing here imports torch or touches a GPU; the module is a plan, and it is
validated at import against the real registries so a nonexistent pruner or model
fails now rather than six hours into a sweep.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))


# --- where the ladder runs -------------------------------------------------

# ResNet-34 / CIFAR-10 at 99.9% sparsity. This is EAST's own published setting
# (Li et al. 2024, arXiv 2411.13545), so every rung is comparable against a real
# external baseline -- RigL 85.71 +/- 0.23, EAST 86.99 +/- 0.32 -- rather than
# against nothing.
#
# 99.9% and not 99.99% because attribution and comparison have different
# statistical requirements. ~21,300 weights survive here against a 5,120-weight
# classifier: severe without being degenerate. At 99.99% only ~2,130 survive, the
# outcome distribution becomes a mixture rather than a Gaussian, and resolving a
# 1-3 point component effect would need on the order of 100 runs per arm. The
# headline table sweeps the full range; the ladder does not.
ANCHOR = {
    'model_name': 'resnet34',
    'model_type': 'cv',
    'dataset_name': 'cifar10',
    'num_classes': 10,
    'image_size': 32,
}

LADDER_SPARSITY = 0.999

# The headline comparison, run with the winning config only, matching EAST's
# published range.
HEADLINE_SPARSITIES = (0.99, 0.999, 0.9995, 0.9999)


# --- protocol --------------------------------------------------------------

# The GraSP -> RigL -> GraNet -> EAST lineage. Two declared deviations, both
# recorded on every run rather than left in prose:
#
#   scheduler_type='cosine'  A declared deviation, and worth stating precisely
#       because EAST CONTRADICTS ITSELF here: its appendix Table B.1 gives
#       '10x[75,150]' for the 250-epoch CIFAR runs, while its main text says the
#       LR drops by 10x at halfway and three-quarters of training -- i.e. epochs
#       125 and 187. Those are different schedules, and the paper does not say
#       which produced Table 1. _initialize_scheduler implements
#       linear_with_warmup and cosine only; cosine is used widely in the same
#       literature and is the closer of the two available options. What matters
#       for attribution is that every rung shares it, which cells() guarantees by
#       construction -- but the headline comparison against EAST's published
#       numbers must state the substitution rather than imply a match.
PROTOCOLS = {
    # EAST's published CIFAR recipe, verified against the TMLR 2025 version
    # (arXiv:2411.13545v4, Table B.1 and Section 4). Every value here is theirs
    # except `scheduler_type`, which is a declared deviation -- see below.
    'east250': {
        'epochs': 250,
        'batch_size': 128,
        'optimizer_type': 'sgd',          # momentum 0.9, written 'SGD(0.9)'
        'learning_rate': 0.1,
        'weight_decay': 1e-4,             # EAST's value. GraNet uses 5e-4.
        'scheduler_type': 'cosine',
        'patience': 0,          # no early stopping: it is test-set model selection
        # Fallback for a single-cell run. pool.py overrides this with
        # cores/concurrency, which is the only correct value when several cells
        # share a machine. Left explicit rather than inheriting os.cpu_count(),
        # which reports the HOST's cores inside a container -- 256 on the rented
        # box, so each cell would have spawned 256 dataloader workers.
        'num_workers': 8,
    },
    # Tier-0 smoke. Same code path, two batches of it.
    'smoke': {
        'epochs': 2,
        'batch_size': 32,
        'optimizer_type': 'sgd',
        'learning_rate': 0.1,
        'scheduler_type': 'cosine',
        'patience': 0,
        'limit_train_batches': 2,
        'limit_eval_batches': 2,
        # MEASURED: a CIFAR-10 DataLoader with num_workers=8 takes 72.9 s to
        # yield its first 2 batches on this platform; with 0 it takes 0.0 s.
        # Windows spawns each worker as a full process that re-imports torch and
        # receives a pickled copy of the dataset, and that cost is paid three
        # times per cell (train, val, test). Workers exist to prefetch while an
        # accelerator computes; a 2-batch CPU loader has nothing to prefetch and
        # no accelerator, so they were ~85% of tier-0 wall clock and bought
        # nothing. Tier 1+ keeps the default, where prefetching is real work.
        'num_workers': 0,
    },
}


# --- rungs -----------------------------------------------------------------

@dataclass(frozen=True)
class Rung:
    """One step of the ladder.

    `overrides` is the *entire* difference from the rung named in `inherits`.
    Keeping it minimal is the whole discipline: a rung whose overrides carry two
    unrelated keys is two experiments wearing one label, and no amount of
    downstream statistics can separate them again.
    """
    id: str
    stage: str
    label: str
    changes: str                       # the one thing, in words, for the caption
    script: str                        # 'baseline' | 'pruning' | 'bacp'
    inherits: str | None
    overrides: dict = field(default_factory=dict)
    paired: bool = True
    n_seeds: int = 5
    priority: str = 'essential'        # essential | high | optional
    cost_u: float = 1.0                # 1 U = one 250-epoch sparse run
    gate: str | None = None            # the gate this rung feeds, if any
    note: str = ''
    # A written justification for changing more than one thing. The validator
    # refuses multi-key overrides without one. It is deliberately a sentence and
    # not a boolean: the only rungs that should ever carry it are ones where the
    # bundle is genuinely irreducible, and being made to say why is the check.
    bundle_ok: str = ''

    @property
    def is_dense(self) -> bool:
        return self.script == 'baseline'


# Keys a rung is allowed to change together, because they are one mechanism
# expressed as two flags. Anything not listed here must be a single-key override.
_COUPLED = [
    {'dyrelu_en', 'dyrelu_phasing_en'},
    {'limit_train_batches', 'limit_eval_batches'},
    {'lambdas', 'num_snapshots'},
    {'distill_mode', 'lambda_kd'},
]


LADDER: list[Rung] = [

    # ---- Stage A: the sparse substrate ------------------------------------
    # Unpaired. The mask lottery dominates run-to-run variance here, so seed
    # correlation between arms is low and pairing buys little.

    Rung('A0', 'A', 'Dense',
         changes='no pruning at all',
         script='baseline', inherits=None,
         overrides={},
         paired=False, n_seeds=8, cost_u=1.0,
         note='The ceiling. n_seeds is 8, not 3, because it must cover every seed '
              'any sparse rung uses: attach_checkpoint pairs each sparse run with '
              'the DENSE run of the same seed, and with only three dense records '
              'seeds 4-8 of every sparse rung silently fell back to one shared '
              'checkpoint -- so runs advertised as independent shared an '
              'initialisation, and the paired designs above them were paired only '
              'from the pruning step onward. validate() now enforces the coverage. '
              'the sparse rungs start from, and of the frozen teachers Stage C '
              'and D use.'),

    Rung('A1', 'A', 'Uniform magnitude, head pruned',
         changes='uniform layer-wise budget; classifier in the prunable scope',
         script='pruning', inherits='A0',
         overrides={'pruning_type': 'local_magnitude',
                    'layerwise_alloc': 'uniform',
                    'prune_task_head': True},
         paired=False, n_seeds=3, priority='optional',
         bundle_ok='Entry rung. Going from dense to sparse is irreducibly a bundle '
                   '-- you cannot name a sparsity without naming a criterion, an '
                   'allocation and a scope. A0->A1 is therefore read as "what does '
                   'sparsity cost", never as a single-mechanism contrast; the '
                   'single-mechanism contrasts are A1->A2, A1->A3, A3->A4.',
         note='The floor. Optional only because A2-A4 pin the 2x2 corners that '
              'matter; if budget is tight this is the first Stage A rung to cut.'),

    Rung('A2', 'A', 'Uniform magnitude, head dense',
         changes='classifier removed from the prunable scope',
         script='pruning', inherits='A1',
         overrides={'prune_task_head': False},
         paired=False, n_seeds=5),

    Rung('A3', 'A', 'ERK, head pruned',
         changes='layer-wise budget switched from uniform to ERK',
         script='pruning', inherits='A1',
         overrides={'layerwise_alloc': 'erk'},
         paired=False, n_seeds=5),

    Rung('A4', 'A', 'ERK, head dense',
         changes='both, to close the 2x2',
         script='pruning', inherits='A3',
         overrides={'prune_task_head': False},
         paired=False, n_seeds=5,
         note='A1/A2/A3/A4 are a deliberate 2x2 on {uniform, ERK} x {head pruned, '
              'head dense}. Collapsing it into one rung would confound "ERK helps" '
              'with "ERK protects the classifier" permanently -- and at 99.9% the '
              'classifier is 24% of the entire surviving weight budget, so that '
              'confound is not a hypothetical. Note the main path runs through '
              'A3, the head-PRUNED corner, because that is what RigL and GraNet '
              'actually do; A2 and A4 measure what the alternative convention buys '
              'rather than adopting it. One deliberate divergence from RigL: its '
              'Uniform distribution keeps the FIRST layer dense, ours does not. '
              'Baking that in would put a third difference inside the uniform arms '
              'and destroy the 2x2; RigL applies the carve-out only to Uniform, not '
              'to ERK, so A3-A6 are unaffected.'),

    Rung('A5', 'A', '+ RigL drop-and-regrow',
         changes='static ERK mask replaced by dynamic topology',
         script='pruning', inherits='A3',
         overrides={'pruning_type': 'rigl'},
         paired=False, n_seeds=5,
         note='Answers whether dynamic topology adds anything beyond allocation. '
              'Inherits A3 (ERK, head PRUNED) and not A4 (head dense) on purpose: '
              'RigL and GraNet both prune the final classifier by default -- RigL '
              'assigns ResNet-50 fc1000 an ERK sparsity of 0.957 and its reference '
              'implementation defaults prune_last_layer=True; GraNet registers '
              'every 2-D and 4-D tensor into the mask set with no classifier '
              'carve-out. Inheriting the head-dense corner would have made this '
              'rung, and therefore the whole substrate that Stage C is built on, '
              'incomparable to every published number we cite. A4 remains a corner '
              'of the 2x2; it is not on the main path.'),

    Rung('A6', 'A', '+ EAST cyclic sparsity',
         changes="RigL's schedule replaced by EAST's cyclic one",
         script='pruning', inherits='A5',
         overrides={'pruning_type': 'east'},
         paired=False, n_seeds=5, priority='high'),

    # ---- Stage B: EAST's architecture knobs -------------------------------
    # Paired. Same substrate, same mask trajectory family; the intervention is
    # architectural, so seeds correlate and pairing pays.
    #
    # Designed so a null is publishable: pre-registered +/-1.0pp equivalence
    # bound, TOST reported alongside the difference CI. "We can reject
    # differences larger than 1.5pp" is a result; "p = 0.43" is not.

    Rung('B1', 'B', '+ DyReLU',
         changes='ReLU replaced by DyReLU',
         script='pruning', inherits='A6',
         overrides={'dyrelu_en': True, 'dyrelu_phasing_en': False},
         n_seeds=5, priority='optional'),

    Rung('B2', 'B', '+ DyReLU phasing',
         changes='DyReLU phased out over training instead of held on',
         script='pruning', inherits='B1',
         overrides={'dyrelu_phasing_en': True},
         n_seeds=5, priority='optional'),

    Rung('B3', 'B', '+ weight sharing',
         changes='residual blocks share weights',
         script='pruning', inherits='B2',
         overrides={'weight_sharing_en': True},
         n_seeds=5, priority='optional',
         note='Rewrites the sparsity target against the unique parameter count, '
              'which is why sparsity_requested and sparsity_target_adjusted are '
              'separate fields on the record.'),

    # ---- Stage C: distillation controls -----------------------------------
    # The scientific heart. Every rung shares the same dense teacher, the same
    # number of optimization steps, and matched forward-pass FLOPs.
    #
    # This is the block the submitted paper had no analogue of at all: it
    # contained no CE-only arm ("no CE" is not "CE only"), so nothing separated
    # the contrastive objective from the recipe wrapped around it -- two views,
    # AutoAugment, the SGD->AdamW phase swap, the projection head. Table 1's I.P.
    # column runs a different recipe, not merely a different loss.

    Rung('C-null', 'C', 'BaCP path, everything BaCP-specific off',
         changes='routed through BaCPTrainer with one view and no contrastive terms',
         script='bacp', inherits=None,          # config comes from BEST_SUBSTRATE
         overrides={'lambdas': (0.0, 0.0, 0.0, 1.0), 'num_snapshots': 0,
                    'n_views': 1},
         n_seeds=5, cost_u=1.1, gate='G1',
         bundle_ok='All three flags say the same thing -- turn BaCP off. The rung '
                   'exists precisely to hold the code path fixed while removing '
                   'every BaCP-specific ingredient, so it cannot be split.',
         note='GATE G1. Must reproduce C0 within +/-0.5pp. C0 runs the same '
              'objective through pruning_script; this runs it through '
              'BaCPTrainer. A gap between them is an artefact of the BaCP path -- '
              'the collate, the projection head, the optimizer swap, the '
              'finetune phase -- and every downstream number would be measuring '
              'that artefact rather than the method. Cheapest, highest-'
              'information run in the plan; run it first.'),

    Rung('C0', 'C', 'CE only',
         changes='plain supervised training on the winning substrate',
         script='pruning', inherits=None,       # config comes from BEST_SUBSTRATE
         overrides={},
         n_seeds=8, cost_u=1.0,
         note='The reference every later rung is measured against. The submitted '
              'paper had no arm like this at all -- "no CE" (lambda_CE = 0) is '
              'not "CE only", so nothing in it separated the contrastive '
              'objective from the recipe wrapped around it.'),

    Rung('C0b', 'C', "CE + BaCP's two-view augmentation",
         changes='two augmented views per step; objective unchanged',
         script='bacp', inherits='C-null',
         overrides={'n_views': 2},
         n_seeds=8, cost_u=1.4,
         note='Not optional. BaCP consumes two views per step, so view count is '
              'entangled with the method by construction; without this rung any '
              'measured gain is "contrast, or just more augmentation", and the '
              'two are not separable after the fact.'),

    Rung('C1', 'C', '+ KD (KL on logits)',
         changes='dense-teacher logit distillation added',
         script='bacp', inherits='C0b',
         overrides={'distill_mode': 'kl', 'lambda_kd': 1.0},
         n_seeds=8, cost_u=1.6, gate='G3',
         note='The control the submitted paper most needed and did not have. '
              'Three frozen dense teachers sit on the BaCP side of its Table 1 '
              'and zero on the I.P. side, so the cheapest explanation of every '
              'gain there is "a dense teacher helps" -- nothing to do with '
              'contrast. This rung is what can refuse that reading.'),

    Rung('C2', 'C', '+ feature distillation (same teacher)',
         changes='logit KD swapped for feature-level matching',
         script='bacp', inherits='C1',
         overrides={'distill_mode': 'feature'},
         n_seeds=8, cost_u=1.6, gate='G3',
         note='Same teacher (model_ft), same weight, same forward passes. Only '
              'the level at which the signal is injected changes.'),

    Rung('C3', 'C', '+ contrastive form (FiC, same teacher)',
         changes='feature regression swapped for the contrastive objective',
         script='bacp', inherits='C2',
         overrides={'lambdas': (0.0, 0.5, 0.0, 0.5),
                    'distill_mode': 'none', 'lambda_kd': 0.0},
         n_seeds=8, cost_u=1.8, gate='G4',
         bundle_ok='One mechanism, three flags: the objective form changes from '
                   'regression-onto-the-teacher to contrast-against-the-teacher. '
                   'The TEACHER IS HELD FIXED (model_ft in both C2 and C3), which '
                   'is the whole point -- swapping teacher and form together, as '
                   'the pre-registered SnC-first order would have done, is the '
                   'same two-changes-one-label error the submitted paper made.',
         note='GATE G4, and the single most important row in the paper. Cosine '
              'feature matching is exactly the positive-pair term of InfoNCE with '
              'the denominator deleted, so C2 -> C3 isolates ONE thing: the '
              'presence of negatives. C1 -> C2 -> C3 reads as: a teacher exists '
              '-> features matter -> the contrastive form matters.'),

    # ---- Stage D: BaCP internals ------------------------------------------
    # Forward over teachers, in the order FiC -> SnC -> PrC.
    #
    # This order is NOT the one originally pre-registered (SnC -> PrC -> FiC). It
    # was changed before any run, for a design reason worth stating: Stage C ends
    # on a contrastive term against the fine-tuned teacher, so starting Stage D
    # with SnC would have made C2 -> C3 change the teacher AND the objective form
    # in one step. Holding the teacher fixed across C2/C3 is what makes "the
    # contrastive form matters" a measurable claim rather than a confounded one.
    # A forward ladder measures each component conditional on the prefix already
    # installed, so the order is part of the claim and is fixed here in advance;
    # the D4 backward pass is what detects order-sensitivity.

    Rung('D1', 'D', '+ SnC (self-snapshots)',
         changes='snapshot teachers added to the contrastive term',
         script='bacp', inherits='C3',
         overrides={'lambdas': (0.0, 1 / 3, 1 / 3, 1 / 3), 'num_snapshots': 2},
         n_seeds=5, cost_u=2.0, gate='G5'),

    Rung('D2', 'D', '+ PrC (pretrained teacher) = full BaCP',
         changes='the pretrained-model teacher added',
         script='bacp', inherits='D1',
         overrides={'lambdas': (0.25, 0.25, 0.25, 0.25)},
         n_seeds=5, cost_u=2.2, gate='G5',
         note='This rung IS full BaCP. Everything above it is the path there, and '
              'every rung below it is a competing explanation that has been ruled '
              'out or not.'),
]


# Backward leave-one-out, generated rather than hand-written so each arm is
# guaranteed to differ from full BaCP in exactly one lambda -- and, unlike the
# submitted paper's Table 6, the remaining weights are RENORMALISED. Zeroing one
# of four 0.25 weights and leaving the rest alone changes two things at once: it
# removes a term and downweights the total objective by 25%.
_LOO = [
    ('D4-noPrC', 'PrC', (0.0, 1 / 3, 1 / 3, 1 / 3)),
    ('D4-noFiC', 'FiC', (1 / 3, 0.0, 1 / 3, 1 / 3)),
    ('D4-noSnC', 'SnC', (1 / 3, 1 / 3, 0.0, 1 / 3)),
    ('D4-noCE', 'CE', (1 / 3, 1 / 3, 1 / 3, 0.0)),
]
for _id, _who, _lam in _LOO:
    LADDER.append(Rung(
        _id, 'D4', f'Full BaCP without {_who}',
        changes=f'{_who} removed; remaining three weights renormalised to 1/3',
        script='bacp', inherits='D2',
        overrides={'lambdas': _lam},
        n_seeds=5, cost_u=2.2, priority='high',
        note='Backward pass. Forward measures sufficiency (effect at the '
             'all-others-off corner); backward measures necessity (effect at the '
             'all-others-on corner). They agree only under strict additivity, and '
             'their gap is the aggregate interaction -- the highest-information-'
             'per-run quantity in the whole design. Large forward + ~0 backward '
             'means redundant, not useless.'))

BY_ID = {r.id: r for r in LADDER}


# The rung whose config Stage C starts from. Resolved at plan time from Stage A's
# results when they exist; until then it is A5, the config Stage A is expected to
# end on. Overridable so a Stage-A surprise does not require editing this file.
BEST_SUBSTRATE = os.environ.get('BACP_BEST_SUBSTRATE', 'A5')


# --- resolution ------------------------------------------------------------

def _chain(rung_id: str) -> list[Rung]:
    """The inheritance chain, root first."""
    out, seen = [], set()
    node = BY_ID[rung_id]
    while node is not None:
        if node.id in seen:
            raise ValueError(f'cycle in the ladder at {node.id}')
        seen.add(node.id)
        out.append(node)
        node = BY_ID[node.inherits] if node.inherits else None
    return list(reversed(out))


def resolve(rung_id: str, *, tier: int | None = None, sparsity: float | None = None) -> dict:
    """A rung id -> the complete flat config that rung runs under.

    The accumulation is the point: a rung cannot silently disagree with the rung
    below it, because it does not restate the shared settings at all.
    """
    tier = TIER if tier is None else tier
    rung = BY_ID[rung_id]

    cfg = dict(ANCHOR)
    cfg.update(PROTOCOLS['smoke' if tier == 0 else 'east250'])

    if not rung.is_dense:
        cfg['target_sparsity'] = LADDER_SPARSITY if sparsity is None else sparsity
        cfg['sparsity_scheduler'] = 'cubic'
        cfg['recovery_epochs'] = 0
        # setdefault, NOT assignment: PROTOCOLS carries delta_T=4000, which is
        # RigL's mask-update interval in EAST's own table. A plain assignment
        # here silently overrode the protocol with 100 -- updating masks 40x
        # more often than the baseline being reproduced, which is a different
        # algorithm rather than a different setting.
        # Mask-update interval in optimizer STEPS. Set here rather than in
        # PROTOCOLS because baseline_script.py has no --delta_T flag, so a
        # protocol-level value would be handed to the dense rung and rejected.
        #
        # 4000 is RigL's interval in EAST's own table. Tier 0 has only
        # epochs*limit_train_batches = 4 steps in total, so that value updates
        # the mask exactly once, at t=1 of a 3-step cubic ramp -- every sparse
        # rung then sits at 0.999*(1-(1-1/3)^3) = 0.7030 while the table calls
        # it 99.9%.
        cfg.setdefault('delta_T', 1 if tier == 0 else 4000)

    # Rungs whose `inherits` is None but which are not the dense root take their
    # substrate from BEST_SUBSTRATE. Written this way rather than by hardcoding
    # 'A5' so that when Stage A finishes, one env var re-points all of Stage C at
    # whatever actually won.
    # Resolve from the ROOT of this rung's chain, not from the rung itself.
    #
    # Two bugs lived here, both silent, both caught only by asserting that a
    # resolved config contains what the ladder says it should:
    #
    #   1. The rung's own overrides were skipped for chain roots, so C-null --
    #      the G1 gate, whose entire purpose is to run the BaCP path with every
    #      BaCP-specific ingredient off -- resolved to full BaCP. The gate meant
    #      to catch implementation artefacts would have BEEN the artefact.
    #
    #   2. The BEST_SUBSTRATE prefix was applied only to rungs whose OWN inherits
    #      was None, never to their descendants. C-null roots the entire Stage C
    #      and D chain, so C0b, C1, C2, C3, D1, D2 and all four D4 arms resolved
    #      with no pruning_type at all -- and since _initialize_pruner computes
    #      `prune = (pruning_type and target_sparsity)`, they would have trained
    #      DENSE while the table labelled them 99.9%-sparse. Nothing would have
    #      raised.
    #
    # validate() now asserts every non-dense rung resolves with a pruner, which
    # is the check that catches this whole family.
    chain = _chain(rung_id)
    if chain[0].script != 'baseline':
        # A substrate-rooted chain (Stage C and D hang off BEST_SUBSTRATE rather
        # than off the dense rung), so the substrate goes on underneath.
        for parent in _chain(BEST_SUBSTRATE):
            cfg.update(parent.overrides)
    for node in chain:
        cfg.update(node.overrides)

    if rung.script == 'bacp':
        cfg.setdefault('lambdas', (0.25, 0.25, 0.25, 0.25))
        cfg.setdefault('num_snapshots', 2)
        cfg.setdefault('n_views', 2)
        cfg.setdefault('contrastive_mode', 'cap')
        # tied_frozen is a FAIRNESS requirement at this sparsity, not a
        # preference. The projection head is Linear(512, 128) = 65,536 weights.
        # At 99.9% only ~21,300 weights survive in the entire network, so a
        # trainable projection head would be 308% of the surviving budget --
        # sitting outside the sparsity denominator, during training, in one arm
        # of a comparison and not the other. Frozen, it contributes no trainable
        # capacity and the arms are comparable.
        cfg.setdefault('proj_mode', 'tied_frozen')
        cfg.setdefault('enable_finetune', True)
        cfg.setdefault('optimizer_type_ft', 'adamw')
        cfg.setdefault('learning_rate_ft', 1e-4)
        cfg.setdefault('epochs_ft', 20 if tier else 2)
        cfg.setdefault('distill_mode', 'none')
        cfg.setdefault('lambda_kd', 0.0)

    cfg['experiment_type'] = f'ladder-{rung.stage}'
    return cfg


# --- tiers -----------------------------------------------------------------

TIER = int(os.environ.get('BACP_TIER', '0'))

TIERS = {
    0: {
        'name': 'smoke',
        'why': 'Every rung, every code path, on CPU, in minutes. Numbers are '
               'meaningless BY CONSTRUCTION -- the loaders are truncated to two '
               'batches. This tier validates plumbing and never, under any '
               'circumstance, appears in the paper.',
        'rungs': [r.id for r in LADDER],
        'seeds': (1,),
    },
    1: {
        'name': 'spine',
        'why': 'The minimum set that makes the paper defensible: the G1 gate, the '
               'Stage A 2x2, and Stage C in full. Deliberately one cell deep and '
               'many methods wide -- a reviewer attacks the claim, not the '
               'breadth, so CE-only + KD + feature-KD at one measurable sparsity '
               'is worth more than one seed across twelve cells.',
        'rungs': ['A0', 'A2', 'A3', 'A4', 'A5',
                  'C-null', 'C0', 'C0b', 'C1', 'C2', 'C3',
                  'D1', 'D2'],
        'seeds': (1, 2, 3, 4, 5),
    },
    2: {
        'name': 'attribution',
        'why': 'Adds necessity (the backward leave-one-out pass) and the rungs '
               'Stage A and B left out, completing the forward/backward gap '
               'column.',
        'rungs': [r.id for r in LADDER],
        'seeds': (1, 2, 3, 4, 5),
    },
    3: {
        'name': 'full',
        'why': 'Everything at 8 seeds, plus the headline sparsity sweep.',
        'rungs': [r.id for r in LADDER],
        'seeds': (1, 2, 3, 4, 5, 6, 7, 8),
    },
}


def _safe(text) -> str:
    return ''.join(c if (c.isalnum() or c in '.-') else '-' for c in str(text))


def cells(tier: int | None = None, *, rungs=None, sparsity=None) -> list[dict]:
    """The units of work: one rung at one seed.

    `key` is what the runner writes into BACP_EXPERIMENT_GROUP, which
    training_utils._finalize_run copies onto the record. That is the whole
    idempotence mechanism -- a cell is complete iff a record exists carrying its
    key, so resume needs no separate bookkeeping file to fall out of sync.
    """
    tier = TIER if tier is None else tier
    spec = TIERS[tier]
    wanted = list(rungs) if rungs is not None else spec['rungs']
    seeds = spec['seeds']

    out = []
    for rung_id in wanted:
        rung = BY_ID[rung_id]
        n = min(rung.n_seeds, len(seeds))
        cfg = resolve(rung_id, tier=tier, sparsity=sparsity)
        for seed in seeds[:n]:
            # The key identifies the EXPERIMENT, not the tier that scheduled
            # it. Tiers 1-3 share the east250 protocol and differ only in which
            # rungs and how many seeds they cover, so the same (rung, seed) is
            # literally the same run -- embedding the tier made tier 2 repeat
            # all 65 of tier 1's runs, 244 U of duplicated work across the three
            # tiers. Tier 0 keeps its own namespace because the smoke protocol
            # is a genuinely different experiment (2 epochs, truncated loaders)
            # and must never satisfy a tier-1 cell.
            scope = 'smoke' if tier == 0 else 'east250'
            key = _safe(f'{scope}.{rung_id}.{cfg["model_name"]}.{cfg["dataset_name"]}'
                        f'.s{cfg.get("target_sparsity", "dense")}.seed{seed}')
            out.append({
                'key': key,
                'tier': tier,
                'rung': rung_id,
                'stage': rung.stage,
                'label': rung.label,
                'changes': rung.changes,
                'script': rung.script,
                'inherits': rung.inherits,
                'priority': rung.priority,
                'gate': rung.gate,
                'paired': rung.paired,
                'cost_u': rung.cost_u,
                'seed': seed,
                'model_name': cfg['model_name'],
                'dataset_name': cfg['dataset_name'],
                'sparsity': cfg.get('target_sparsity'),
                'config': cfg,
            })
    return out


def to_dataframe(tier: int | None = None):
    import pandas as pd
    rows = []
    for c in cells(tier):
        row = {k: v for k, v in c.items() if k != 'config'}
        row.update({f'cfg_{k}': v for k, v in c['config'].items()})
        rows.append(row)
    return pd.DataFrame(rows)


def write_manifest_csv(path=None, tier: int | None = None) -> Path:
    from results import results_dir
    path = Path(path) if path else results_dir() / 'manifest.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    to_dataframe(tier).to_csv(path, index=False)
    return path


# --- budget ----------------------------------------------------------------

def budget(tier: int | None = None, *, s_per_epoch=None, root=None) -> dict:
    """Projected cost, derived from measured per-epoch wall-clock where available.

    Reads `s_per_epoch_mean` out of the existing records rather than assuming a
    throughput number, so once tier 0 has run once the projection for tiers 1-3 is
    arithmetic on real timings instead of a guess. Falls back to cost_u multiples
    of a caller-supplied `s_per_epoch` when no records exist.
    """
    tier = TIER if tier is None else tier
    measured = {}
    if s_per_epoch is None:
        try:
            from results import load_runs
            df = load_runs(root=root)
            if not df.empty and 's_per_epoch_mean' in df:
                sub = df.dropna(subset=['s_per_epoch_mean'])
                for script in ('pruning', 'bacp', 'dense'):
                    pass
                measured['overall_median_s_per_epoch'] = float(
                    sub['s_per_epoch_mean'].median()) if len(sub) else None
                s_per_epoch = measured['overall_median_s_per_epoch']
        except Exception as exc:                                   # noqa: BLE001
            measured['error'] = f'{type(exc).__name__}: {exc}'

    grid = cells(tier)
    total_u = sum(c['cost_u'] for c in grid)
    by_stage = {}
    for c in grid:
        s = by_stage.setdefault(c['stage'], {'runs': 0, 'u': 0.0})
        s['runs'] += 1
        s['u'] += c['cost_u']

    epochs = PROTOCOLS['smoke' if tier == 0 else 'east250']['epochs']
    hours = None
    if s_per_epoch:
        hours = round(total_u * epochs * s_per_epoch / 3600.0, 2)

    return {
        'tier': tier,
        'tier_name': TIERS[tier]['name'],
        'n_runs': len(grid),
        'total_u': round(total_u, 1),
        'epochs_per_run': epochs,
        's_per_epoch_used': s_per_epoch,
        'projected_hours': hours,
        'by_stage': by_stage,
        'measured': measured,
        'caveat': ('1 U = one full-length sparse run at this tier. BaCP rungs are '
                   'charged 1.4-2.2 U for their extra teacher forwards and second '
                   'view. Projections use the MEDIAN measured epoch time across '
                   'all recorded runs, which understates BaCP rungs and overstates '
                   'dense ones; treat it as an order of magnitude, not a quote.'),
    }


# --- plan-time validation --------------------------------------------------

def validate(strict=True) -> list[str]:
    """Fail at plan time, not at 3am.

    Every pruner the ladder names must exist in PRUNER_REGISTRY, every model in
    MODELS, every dataset in the factory, and every rung must change at most one
    mechanism. Import-time by default: a manifest that references a pruner nobody
    implemented is worse than no manifest, because it looks like a plan.
    """
    problems = []

    try:
        from pruning_factory import PRUNER_REGISTRY
        from model_factory import MODELS
        from dataset_factory import CV_DATASETS, TEXT_DATASETS, DATASET_NUM_CLASSES
    except Exception as exc:                                       # noqa: BLE001
        problems.append(f'cannot import the registries to validate against: {exc}')
        if strict:
            raise RuntimeError(problems[0]) from exc
        return problems

    if ANCHOR['model_name'] not in MODELS:
        problems.append(f"anchor model {ANCHOR['model_name']!r} is not in MODELS")
    if ANCHOR['dataset_name'] not in CV_DATASETS and ANCHOR['dataset_name'] not in TEXT_DATASETS:
        problems.append(f"anchor dataset {ANCHOR['dataset_name']!r} is not registered")
    expected = DATASET_NUM_CLASSES.get(ANCHOR['dataset_name'])
    if expected is not None and expected != ANCHOR['num_classes']:
        problems.append(f"anchor num_classes {ANCHOR['num_classes']} != "
                        f"DATASET_NUM_CLASSES {expected}")

    for rung in LADDER:
        if rung.inherits and rung.inherits not in BY_ID:
            problems.append(f'{rung.id} inherits from unknown rung {rung.inherits!r}')
        if rung.script not in ('baseline', 'pruning', 'bacp'):
            problems.append(f'{rung.id} names unknown script {rung.script!r}')

        pruner = rung.overrides.get('pruning_type')
        if pruner and pruner not in PRUNER_REGISTRY:
            problems.append(
                f'{rung.id} needs pruner {pruner!r}, which is not in '
                f'PRUNER_REGISTRY {sorted(PRUNER_REGISTRY)}')

        keys = set(rung.overrides)
        if len(keys) > 1 and not rung.bundle_ok and not any(keys <= c for c in _COUPLED):
            # A2/A3/A4 and the LOO arms each change one key; anything else that
            # changes several must be a declared coupled pair, or the rung is two
            # experiments sharing one label.
            problems.append(
                f'{rung.id} changes {sorted(keys)} -- more than one mechanism, and '
                f'not a declared coupled set. Split it into two rungs.')

    # Every rung must be reachable and every chain must terminate.
    for rung in LADDER:
        try:
            _chain(rung.id)
        except ValueError as exc:
            problems.append(str(exc))

    for tier, spec in TIERS.items():
        for rung_id in spec['rungs']:
            if rung_id not in BY_ID:
                problems.append(f'tier {tier} names unknown rung {rung_id!r}')

    if BEST_SUBSTRATE not in BY_ID:
        problems.append(f'BEST_SUBSTRATE={BEST_SUBSTRATE!r} is not a rung')

    # Tiers 1-3 must SHARE cells, so that running them in sequence is
    # incremental rather than repeating work; and tier 0 must share none of
    # them, because its protocol differs and its records are not measurements.
    k0 = {c['key'] for c in cells(0)}
    k1 = {c['key'] for c in cells(1)}
    k3 = {c['key'] for c in cells(3)}
    if k0 & k1:
        problems.append(
            f'tier 0 shares {len(k0 & k1)} cell key(s) with tier 1; a smoke '
            f'record would satisfy a real cell')
    if not k1 <= k3:
        problems.append(
            f'tier 1 is not a subset of tier 3 ({len(k1 - k3)} key(s) outside), '
            f'so running tier 3 after tier 1 would repeat work')

    # Seed coverage. Every sparse rung resolves its starting checkpoint from the
    # dense rung of the SAME seed; if the dense rung runs fewer seeds, the extra
    # sparse seeds quietly share one initialisation and the pairing is fiction.
    dense = [r for r in LADDER if r.is_dense]
    if dense:
        dense_seeds = max(r.n_seeds for r in dense)
        greedy = [r.id for r in LADDER if not r.is_dense and r.n_seeds > dense_seeds]
        if greedy:
            problems.append(
                f'rungs {greedy} request more seeds than the dense rung provides '
                f'({dense_seeds}).每 sparse run pairs with the dense run of the '
                f'same seed, so the surplus seeds would share one checkpoint.'
                .replace('每', 'Each'))

    # Every rung's resolved config must actually contain what the rung says it
    # changes. This is the assertion that would have caught C-null resolving to
    # full BaCP, and it is cheap enough to run at import on every process.
    for rung in LADDER:
        try:
            cfg = resolve(rung.id, tier=1)
        except Exception as exc:                                   # noqa: BLE001
            problems.append(f'{rung.id} does not resolve: {type(exc).__name__}: {exc}')
            continue
        for key, want in rung.overrides.items():
            got = cfg.get(key)
            same = (tuple(got) == tuple(want)
                    if isinstance(want, (list, tuple)) and isinstance(got, (list, tuple))
                    else got == want)
            if not same:
                problems.append(
                    f'{rung.id} declares {key}={want!r} but resolves to {got!r} -- '
                    f'the rung is not running the experiment it describes.')

    # Every non-dense rung must actually get a pruner and a target. Without this,
    # a rung can resolve with pruning_type=None, train dense, and be reported at
    # 99.9% sparsity -- _initialize_pruner computes prune = (pruning_type and
    # target_sparsity) and silently sets args.pruner = None when either is
    # missing. This is the single highest-value assertion in the file.
    for rung in LADDER:
        if rung.is_dense:
            continue
        try:
            cfg = resolve(rung.id, tier=1)
        except Exception:                                          # noqa: BLE001
            continue
        if not cfg.get('pruning_type'):
            problems.append(
                f'{rung.id} is a sparse rung but resolves with no pruning_type -- '
                f'it would train DENSE while the table reports it at '
                f'{cfg.get("target_sparsity")} sparsity.')
        if not cfg.get('target_sparsity'):
            problems.append(f'{rung.id} resolves with no target_sparsity')

    # Every rung on the substrate-rooted path must agree with BEST_SUBSTRATE on
    # the substrate itself. If Stage C drifts off the winning substrate, its
    # deltas are measured against a different network than Stage A established.
    base = resolve(BEST_SUBSTRATE, tier=1)
    for rung in LADDER:
        if rung.is_dense or _chain(rung.id)[0].script == 'baseline':
            continue
        cfg = resolve(rung.id, tier=1)
        for key in ('pruning_type', 'layerwise_alloc', 'prune_task_head'):
            if key in rung.overrides:
                continue                       # a deliberate, declared change
            if cfg.get(key) != base.get(key):
                problems.append(
                    f'{rung.id} has {key}={cfg.get(key)!r} but the substrate '
                    f'({BEST_SUBSTRATE}) has {base.get(key)!r}, and the rung does '
                    f'not declare that change.')

    # And no two rungs may resolve to the same config: a duplicate means one of
    # them is not testing what its label claims, and the ladder would report a
    # difference of exactly zero as though it were a measurement.
    seen = {}
    for rung in LADDER:
        try:
            cfg = resolve(rung.id, tier=1)
        except Exception:                                          # noqa: BLE001
            continue
        sig = (rung.script,) + tuple(sorted(
            (k, tuple(v) if isinstance(v, (list, tuple)) else v) for k, v in cfg.items()))
        if sig in seen:
            problems.append(
                f'{rung.id} resolves to exactly the same config as {seen[sig]}. '
                f'One of the two is mislabelled.')
        seen[sig] = rung.id

    if problems and strict:
        raise RuntimeError('manifest is not self-consistent:\n  - '
                           + '\n  - '.join(problems))
    return problems


def summary(tier: int | None = None) -> str:
    tier = TIER if tier is None else tier
    grid = cells(tier)
    lines = [
        f'Tier {tier} ({TIERS[tier]["name"]}): {len(grid)} runs over '
        f'{len({c["rung"] for c in grid})} rungs',
        '',
        f'{"rung":<10} {"stage":<6} {"seeds":<6} {"script":<9} {"gate":<5} label',
        '-' * 88,
    ]
    seen = {}
    for c in grid:
        seen.setdefault(c['rung'], []).append(c['seed'])
    for rung_id in dict.fromkeys(c['rung'] for c in grid):
        r = BY_ID[rung_id]
        lines.append(f'{r.id:<10} {r.stage:<6} {len(seen[rung_id]):<6} '
                     f'{r.script:<9} {r.gate or "-":<5} {r.label}')
    return '\n'.join(lines)


validate(strict=True)


if __name__ == '__main__':
    print(summary())
    print()
    import json
    print(json.dumps(budget(), indent=2, default=str))
