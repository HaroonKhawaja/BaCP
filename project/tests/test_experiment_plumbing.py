"""L12 -- the plumbing the audit found had no tests at all.

Each of these covers something where a deliberate mutation of production code
left the whole suite green:

  `_LimitedLoader.__len__`  -- returning the untruncated length leaves tier-0
      smoke running 2 batches while the scheduler and the pruner are told there
      are hundreds, so `total_steps` is wrong, `end_idx` is wrong, and the pruner
      never reaches its target. Undetected.
  `layerwise_alloc='uniform'` -- never constructed by any test. Deleting the
      branch entirely, so uniform silently became ERK, was undetected. That is
      the floor arm of the Stage A 2x2.
  `attach_checkpoint`       -- zero tests, and the manifest's own seed counts
      made the advertised same-seed pairing unsatisfiable.
  `completed_keys` / resume -- widening the status filter would make every
      crashed cell permanently skipped, and nothing would notice.
  `code_fingerprint`        -- the only probe wrote into `project/tests/`, the
      one directory the fingerprint is defined to ignore, so only the negative
      direction was covered.
"""

import json
import sys
from pathlib import Path

import pytest
import torch

_PROJECT = Path(__file__).resolve().parents[1]
for _p in (str(_PROJECT),):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# --- _LimitedLoader --------------------------------------------------------

class _FakeLoader:
    def __init__(self, n, batch=4):
        self.batches = [(torch.zeros(batch, 3), torch.zeros(batch, dtype=torch.long))
                        for _ in range(n)]
        self.dataset = list(range(n * batch))
        self.batch_size = batch

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def test_limited_loader_reports_the_truncated_length():
    """len() is load-bearing in three places -- the scheduler's total_steps, the
    pruner's total_steps, and `train_batches` on the record. Returning the
    untruncated length would leave all three describing a run that did not
    happen, and the pruner would never reach its target because its end_idx sits
    far beyond the steps that actually execute."""
    from training_utils import _LimitedLoader

    inner = _FakeLoader(100)
    limited = _LimitedLoader(inner, 2)
    assert len(limited) == 2
    assert len(list(limited)) == 2, 'iteration must stop at the limit too'
    assert len(limited) != len(inner), 'the truncation is not visible in len()'


def test_limited_loader_never_reports_more_than_the_inner_loader():
    from training_utils import _LimitedLoader
    assert len(_LimitedLoader(_FakeLoader(3), 10)) == 3


def test_limited_loader_delegates_other_attributes():
    from training_utils import _LimitedLoader
    inner = _FakeLoader(10)
    limited = _LimitedLoader(inner, 2)
    assert limited.dataset is inner.dataset
    assert limited.batch_size == inner.batch_size


def test_limited_loader_is_reiterable():
    """The training loop iterates once per epoch; a one-shot iterator would make
    every epoch after the first see zero batches."""
    from training_utils import _LimitedLoader
    limited = _LimitedLoader(_FakeLoader(10), 3)
    assert len(list(limited)) == 3
    assert len(list(limited)) == 3


def test_maybe_limit_is_a_no_op_at_zero():
    from training_utils import _maybe_limit
    inner = _FakeLoader(5)
    assert _maybe_limit(inner, 0) is inner
    assert _maybe_limit(None, 3) is None


# --- uniform vs ERK layer-wise allocation ----------------------------------

@pytest.fixture
def _scope():
    from pruning_factory import set_prunable_scope
    set_prunable_scope(prune_task_head=True, prune_embeddings=False, model=None)
    yield
    set_prunable_scope(prune_task_head=False, prune_embeddings=False, model=None)


def _alloc_net():
    import torch.nn as nn
    torch.manual_seed(0)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.wide = nn.Conv2d(3, 32, 3, padding=1)     # 864 weights
            self.narrow = nn.Linear(32, 4)                 # 128 weights
        def forward(self, x):
            return self.narrow(self.wide(x).mean(dim=(2, 3)))

    return Net()


def test_uniform_allocation_gives_every_layer_the_same_density(_scope):
    """The floor arm of the Stage A 2x2. Deleting this branch so uniform silently
    became ERK was undetected by the entire suite -- and it would have collapsed
    the 2x2 into a 1x2 without changing a single label."""
    from pruning_factory import LocalMagnitudePrune

    p = LocalMagnitudePrune(_alloc_net(), 2, 0.9, scheduler_type='one_shot',
                            delta_T=1, total_steps=4, layerwise_alloc='uniform')
    densities = set(round(d, 6) for d in p.erk_densities.values())
    assert densities == {round(1 - 0.9, 6)}, (
        f'uniform must give every layer density 0.1; got {densities}')


def test_erk_allocation_differentiates_wide_from_narrow_layers(_scope):
    """ERK's whole point: a wide layer gets lower density than a narrow one."""
    from pruning_factory import LocalMagnitudePrune

    p = LocalMagnitudePrune(_alloc_net(), 2, 0.9, scheduler_type='one_shot',
                            delta_T=1, total_steps=4, layerwise_alloc='erk')
    d = p.erk_densities
    assert d['wide.weight'] < d['narrow.weight'], (
        f'ERK gave the wide layer {d["wide.weight"]:.4f} and the narrow one '
        f'{d["narrow.weight"]:.4f}; they should differ in that direction')


def test_uniform_and_erk_produce_different_masks(_scope):
    from pruning_factory import LocalMagnitudePrune

    masks = {}
    for alloc in ('uniform', 'erk'):
        p = LocalMagnitudePrune(_alloc_net(), 2, 0.9, scheduler_type='one_shot',
                                delta_T=1, total_steps=4, layerwise_alloc=alloc)
        p.step(3)
        masks[alloc] = {k: v.clone() for k, v in p.masks.items()}

    assert not all(torch.equal(masks['uniform'][k], masks['erk'][k])
                   for k in masks['uniform']), (
        'uniform and ERK gave identical masks, so the knob does nothing')


def test_unknown_allocation_raises(_scope):
    from pruning_factory import LocalMagnitudePrune
    with pytest.raises(ValueError, match='layerwise_alloc'):
        LocalMagnitudePrune(_alloc_net(), 2, 0.9, scheduler_type='one_shot',
                            delta_T=1, total_steps=4, layerwise_alloc='global')


# --- runner: resume and checkpoint resolution -------------------------------

def _write_record(root, **fields):
    """A minimal record on disk, enough for completed_keys / resolve_checkpoint."""
    import results as R
    payload = {'schema_version': R.SCHEMA_VERSION, 'status': 'ok', 'phase': 'dense',
               'method': 'dense', 'ended_at': '2026-01-01T00:00:00+00:00'}
    payload.update(fields)
    (root / 'runs').mkdir(parents=True, exist_ok=True)
    path = root / 'runs' / f"{payload['record_id']}.json"
    path.write_text(json.dumps(payload), encoding='utf-8')
    return path


def test_completed_keys_counts_only_successful_runs(tmp_path):
    """A crashed cell must stay eligible for a retry. Widening this filter to
    include 'failed' would make every crash permanently skipped, and the grid
    would report itself complete while missing those cells."""
    import runner as Rn

    root = tmp_path / 'results'
    _write_record(root, record_id='a', experiment_group='cell-a', status='ok')
    _write_record(root, record_id='b', experiment_group='cell-b', status='failed')
    _write_record(root, record_id='c', experiment_group=None, status='ok')

    keys = Rn.completed_keys(root)
    assert keys == {'cell-a'}, f'expected only the ok cell, got {keys}'


def test_attach_checkpoint_prefers_the_dense_run_of_the_same_seed(tmp_path):
    """The pairing the whole design rests on: a sparse run starts from the dense
    run of its OWN seed, so a paired comparison is paired all the way down to
    initialisation rather than only from the pruning step onward."""
    import runner as Rn

    root = tmp_path / 'results'
    for seed in (1, 2):
        ckpt = tmp_path / f'dense_seed{seed}.pt'
        ckpt.write_bytes(b'x')
        _write_record(root, record_id=f'dense{seed}', model_name='resnet34',
                      dataset_name='cifar10', seed=seed, save_path=str(ckpt))

    cell = {'script': 'pruning', 'seed': 2, 'model_name': 'resnet34',
            'dataset_name': 'cifar10', 'config': {}}
    out = Rn.attach_checkpoint(cell, root=root)
    assert out['config']['trained_weights'].endswith('dense_seed2.pt')


def test_attach_checkpoint_falls_back_loudly_when_the_seed_is_missing(tmp_path, capsys):
    """Falling back silently is how runs advertised as independent end up sharing
    an initialisation."""
    import runner as Rn

    root = tmp_path / 'results'
    ckpt = tmp_path / 'dense_seed1.pt'
    ckpt.write_bytes(b'x')
    _write_record(root, record_id='dense1', model_name='resnet34',
                  dataset_name='cifar10', seed=1, save_path=str(ckpt))

    cell = {'script': 'pruning', 'seed': 7, 'model_name': 'resnet34',
            'dataset_name': 'cifar10', 'config': {}}
    out = Rn.attach_checkpoint(cell, root=root)
    assert out['config']['trained_weights'].endswith('dense_seed1.pt')
    assert 'pairing' in capsys.readouterr().out.lower()


def test_attach_checkpoint_is_a_no_op_for_the_dense_rung(tmp_path):
    import runner as Rn
    cell = {'script': 'baseline', 'seed': 1, 'model_name': 'resnet34',
            'dataset_name': 'cifar10', 'config': {}}
    assert Rn.attach_checkpoint(cell, root=tmp_path / 'results') is cell


def test_a_missing_checkpoint_becomes_a_failure_row_not_an_exception(tmp_path):
    """This is the DEFAULT first run: every sparse cell needs a dense checkpoint
    that only rung A0 produces. It used to raise out of run_cell, past run_grid
    and past run_ladder.main -- aborting the whole sweep with no failure record,
    no summary, no table, and not even a log file."""
    import results as R
    import runner as Rn

    root = tmp_path / 'results'
    root.mkdir(parents=True, exist_ok=True)
    cell = {'key': 'cell-x', 'script': 'pruning', 'rung': 'A1', 'seed': 1,
            'label': 'x', 'changes': 'x', 'model_name': 'resnet34',
            'dataset_name': 'cifar10', 'sparsity': 0.999, 'config': {}}

    res = Rn.run_cell(cell, root=root, echo=False)
    assert res['status'] == 'failed'
    groups = {p.get('experiment_group') for p in R.iter_records(root)}
    assert 'cell-x' in groups, 'a scheduling failure must still be a row'


# --- code fingerprint, positive direction -----------------------------------

def test_fingerprint_changes_when_production_source_changes(tmp_path, monkeypatch):
    """The existing test only ever wrote its probe into project/tests/, which the
    fingerprint is defined to skip -- so it covered the exclusion rule and not the
    detection. Gate staleness depends entirely on the positive direction.
    """
    import results as R

    fake = tmp_path / 'repo'
    (fake / 'project').mkdir(parents=True)
    (fake / '.git').mkdir()
    src = fake / 'project' / 'thing.py'
    src.write_text('x = 1\n', encoding='utf-8')

    before = R.code_fingerprint(fake)
    src.write_text('x = 2\n', encoding='utf-8')
    after = R.code_fingerprint(fake)
    assert before != after, 'editing a production source file must move the digest'

    (fake / 'project' / 'tests').mkdir()
    (fake / 'project' / 'tests' / 't.py').write_text('y = 3\n', encoding='utf-8')
    assert R.code_fingerprint(fake) == after, 'tests/ must stay excluded'
