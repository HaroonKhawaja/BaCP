"""L10 -- the ablation ladder's design, checked as code.

Unit tests below this level prove the code does what it was written to do. These
prove something categorically different and more important: that the *experiment*
measures what its labels claim. A ladder can be implemented perfectly and still
be worthless if two rungs differ in two things, or if the gate rung is secretly
the full method -- which it was, until `resolve()` was fixed to apply a
root-less rung's own overrides.

Everything here runs in milliseconds and touches no GPU, because the manifest is
a plan rather than an execution.
"""

import sys
from pathlib import Path

import pytest

_EXPERIMENTS = Path(__file__).resolve().parents[1] / 'experiments'
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

import manifest as M                                              # noqa: E402
import runner as R                                                # noqa: E402


# --- self-consistency ------------------------------------------------------

def test_manifest_validates():
    assert M.validate(strict=False) == []


def test_every_named_pruner_exists():
    """A rung naming a pruner nobody implemented is worse than no rung: it looks
    like a plan. This is why validate() runs at import."""
    from pruning_factory import PRUNER_REGISTRY
    for rung in M.LADDER:
        pruner = rung.overrides.get('pruning_type')
        if pruner:
            assert pruner in PRUNER_REGISTRY, f'{rung.id} names {pruner!r}'


def test_anchor_model_and_dataset_are_registered():
    from dataset_factory import DATASET_NUM_CLASSES
    from model_factory import MODELS
    assert M.ANCHOR['model_name'] in MODELS
    assert DATASET_NUM_CLASSES[M.ANCHOR['dataset_name']] == M.ANCHOR['num_classes']


# --- one rung, one mechanism -----------------------------------------------

def test_each_rung_changes_at_most_one_mechanism():
    """The discipline the whole design rests on.

    A rung whose overrides carry two unrelated keys is two experiments wearing
    one label, and no downstream statistics can separate them. Multi-key rungs
    are allowed only via a declared coupled set or a written `bundle_ok`
    justification -- being made to say why is the check.
    """
    for rung in M.LADDER:
        keys = set(rung.overrides)
        if len(keys) <= 1 or rung.bundle_ok:
            continue
        assert any(keys <= c for c in M._COUPLED), (
            f'{rung.id} changes {sorted(keys)} with no justification')


def test_rungs_resolve_to_what_they_declare():
    """Regression guard for the worst bug this file could have.

    `resolve()` used to skip a root-less rung's own overrides, so C-null -- the
    G1 gate, whose entire purpose is to run the BaCP path with every
    BaCP-specific ingredient off -- resolved to full BaCP. The gate designed to
    detect implementation artefacts would have BEEN the artefact, and nothing
    downstream could have noticed.
    """
    for rung in M.LADDER:
        cfg = M.resolve(rung.id, tier=1)
        for key, want in rung.overrides.items():
            got = cfg.get(key)
            if isinstance(want, (list, tuple)):
                assert tuple(got) == tuple(want), f'{rung.id}.{key}'
            else:
                assert got == want, f'{rung.id}.{key}: want {want!r}, got {got!r}'


def test_c_null_is_not_bacp():
    """Stated separately from the generic check because it is the gate."""
    cfg = M.resolve('C-null', tier=1)
    assert tuple(cfg['lambdas']) == (0.0, 0.0, 0.0, 1.0)
    assert cfg['num_snapshots'] == 0
    assert cfg['n_views'] == 1


def test_full_bacp_is_the_last_forward_rung():
    cfg = M.resolve('D2', tier=1)
    assert tuple(cfg['lambdas']) == (0.25, 0.25, 0.25, 0.25)
    assert cfg['num_snapshots'] == 2
    assert cfg['distill_mode'] == 'none'


def test_no_two_rungs_resolve_identically():
    """A duplicate means one rung is mislabelled, and the ladder would report a
    difference of exactly zero as though it were a measurement."""
    seen = {}
    for rung in M.LADDER:
        cfg = M.resolve(rung.id, tier=1)
        sig = (rung.script,) + tuple(sorted(
            (k, tuple(v) if isinstance(v, (list, tuple)) else v)
            for k, v in cfg.items()))
        assert sig not in seen, f'{rung.id} == {seen.get(sig)}'
        seen[sig] = rung.id


def test_c2_to_c3_holds_the_teacher_fixed():
    """The single most important contrast in the paper.

    C2 regresses onto the fine-tuned dense teacher's features; C3 contrasts
    against the SAME teacher. Only the objective's form changes, which is what
    makes "the contrastive form matters" measurable. The originally pre-registered
    SnC-first order would have swapped teacher and form together -- the same
    two-changes-one-label error the submitted paper made.
    """
    c2, c3 = M.resolve('C2', tier=1), M.resolve('C3', tier=1)
    # C3 turns on the FiC term (index 1 = FiC) and no other contrastive term.
    assert tuple(c3['lambdas'])[1] > 0
    assert tuple(c3['lambdas'])[0] == 0.0, 'C3 must not enable PrC'
    assert tuple(c3['lambdas'])[2] == 0.0, 'C3 must not enable SnC'
    assert c3['num_snapshots'] == 0, 'C3 must not have snapshot teachers'
    # ...and C2's teacher was already the fine-tuned one, by construction.
    assert c2['distill_mode'] == 'feature'


def test_leave_one_out_arms_renormalise():
    """Zeroing one of four 0.25 weights and leaving the rest alone -- what the
    submitted paper's Table 6 did -- is a removal AND a 25% global downweighting.
    The backward arms must sum to the same total as full BaCP."""
    full = sum(M.resolve('D2', tier=1)['lambdas'])
    for rung in M.LADDER:
        if not rung.id.startswith('D4-'):
            continue
        lam = M.resolve(rung.id, tier=1)['lambdas']
        assert sum(lam) == pytest.approx(full), f'{rung.id} sums to {sum(lam)}'
        assert sum(1 for x in lam if x == 0.0) == 1, f'{rung.id} zeroes != 1 term'


# --- protocol comparability ------------------------------------------------

@pytest.mark.parametrize('field', ['epochs', 'batch_size', 'learning_rate',
                                   'optimizer_type', 'scheduler_type',
                                   'image_size', 'model_name', 'dataset_name'])
def test_protocol_is_identical_across_every_rung(field):
    """Converts "at matched protocol" from prose into a fact.

    If any rung's epochs or learning rate drift, every delta in the table is
    confounded with a training-budget difference and the ladder means nothing.
    """
    values = {M.resolve(r.id, tier=1).get(field) for r in M.LADDER}
    assert len(values) == 1, f'{field} varies across rungs: {values}'


def test_sparse_rungs_share_one_sparsity_target():
    targets = {M.resolve(r.id, tier=1).get('target_sparsity')
               for r in M.LADDER if not r.is_dense}
    assert targets == {M.LADDER_SPARSITY}


def test_projection_head_is_frozen_on_every_bacp_rung():
    """A fairness requirement at this sparsity, not a preference.

    The projection head is Linear(512, 128) = 65,536 weights. At 99.9% only
    ~21,300 weights survive in the whole network, so a *trainable* head would be
    308% of the surviving budget -- sitting outside the sparsity denominator,
    during training, in one arm of a comparison and not the other.
    """
    for rung in M.LADDER:
        if rung.script != 'bacp':
            continue
        assert M.resolve(rung.id, tier=1)['proj_mode'] == 'tied_frozen', rung.id


def test_dense_rung_has_no_sparsity():
    cfg = M.resolve('A0', tier=1)
    assert 'target_sparsity' not in cfg
    assert 'pruning_type' not in cfg


# --- cells -----------------------------------------------------------------

def test_cell_keys_are_unique_and_filesystem_safe():
    keys = [c['key'] for c in M.cells(0)]
    assert len(keys) == len(set(keys))
    for key in keys:
        assert all(ch.isalnum() or ch in '.-' for ch in key), key


def test_every_tier_names_only_real_rungs():
    for tier, spec in M.TIERS.items():
        for rung_id in spec['rungs']:
            assert rung_id in M.BY_ID, f'tier {tier}: {rung_id}'


def test_tier_zero_truncates_the_loaders():
    """Tier 0 must be unmistakably a smoke run, or its numbers will eventually be
    read as results."""
    cfg = M.resolve('A0', tier=0)
    assert cfg['limit_train_batches'] > 0
    assert cfg['limit_eval_batches'] > 0


def test_tier_one_does_not_truncate():
    cfg = M.resolve('A0', tier=1)
    assert not cfg.get('limit_train_batches')
    assert not cfg.get('limit_eval_batches')


# --- runner ----------------------------------------------------------------

@pytest.mark.parametrize('tier', sorted(M.TIERS))
def test_every_config_key_is_a_real_cli_flag(tier):
    """The check that stops a sweep silently running a different experiment.

    argparse would not reject an unknown key -- the runner simply would not emit
    it, the script would use its default, and the record would still look clean.
    """
    for cell in M.cells(tier):
        spec = R.parser_spec(cell['script'])
        unknown = sorted(set(cell['config']) - set(spec))
        assert not unknown, f'{cell["rung"]} ({cell["script"]}): {unknown}'


def test_build_command_emits_store_true_flags_only_when_true():
    cell = dict(M.cells(0)[0])
    cell['config'] = dict(cell['config'], prune_task_head=False)
    assert '--prune_task_head' not in R.build_command(cell)
    cell['config'] = dict(cell['config'], prune_task_head=True)
    assert '--prune_task_head' in R.build_command(cell)


def test_build_command_rejects_unknown_keys():
    cell = dict(M.cells(0)[0])
    cell['config'] = dict(cell['config'], not_a_real_flag=1)
    with pytest.raises(KeyError, match='not_a_real_flag'):
        R.build_command(cell)


def test_build_command_expands_the_lambda_vector():
    cell = next(c for c in M.cells(0) if c['script'] == 'bacp')
    argv = R.build_command(cell)
    i = argv.index('--lambdas')
    assert len(argv[i + 1:i + 5]) == 4
    assert all(not a.startswith('--') for a in argv[i + 1:i + 5])


def test_seed_reaches_the_command_line():
    """It used to be popped from args before the dataclass was built, so it was
    never recorded anywhere and no run was reproducible after the fact."""
    for cell in M.cells(0)[:5]:
        argv = R.build_command(cell)
        assert '--seed' in argv
        assert argv[argv.index('--seed') + 1] == str(cell['seed'])


def test_budget_reports_per_stage_costs():
    b = M.budget(1)
    assert b['n_runs'] > 0
    assert set(b['by_stage']) <= {'A', 'B', 'C', 'D', 'D4'}
    assert b['total_u'] > 0
