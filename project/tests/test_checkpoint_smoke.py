"""A smoke checkpoint must never seed a real run.

The hazard is silent by construction. `00_smoke_test_all` trains every model for
2 epochs on 2 batches and writes an ordinary `status: ok` dense record, and the
notebook README then tells people to run "any cell, in any order". Before this
guard, resolve_checkpoint filtered on status/model/dataset/method/seed and
nothing else, so the newest matching record won -- and right after a smoke pass
that is the 2-batch one. The sparse run that follows does not fail: it loads the
weights, trains, converges to something, and records a number that looks like a
result. That is worse than an error.

The README also asserted the guard already existed ("Smoke records carry a
.smoke suffix and never satisfy real cells"). It did not.
"""

import json

import pytest

from results import resolve_checkpoint


def _write_record(root, record_id, ckpt, **fields):
    """A minimal dense record plus the checkpoint file it points at.

    Same shape as test_experiment_plumbing._write_record -- enough of the schema
    for resolve_checkpoint, not a real record.
    """
    ckpt.write_bytes(b'weights')
    payload = {'schema_version': 1, 'status': 'ok', 'phase': 'dense',
               'method': 'dense', 'model_name': 'resnet50',
               'dataset_name': 'cifar10', 'seed': 42,
               'ended_at': '2026-01-01T00:00:00+00:00',
               'record_id': record_id, 'save_path': str(ckpt)}
    payload.update(fields)
    (root / 'runs').mkdir(parents=True, exist_ok=True)
    (root / 'runs' / f'{record_id}.json').write_text(json.dumps(payload),
                                                     encoding='utf-8')
    return payload


def test_resolve_checkpoint_skips_the_smoke_run(tmp_path):
    """The real baseline wins even when the smoke record is the newer one.

    The smoke record is deliberately given the later `ended_at`: candidates are
    sorted newest-first, so this is the ordering that actually occurs in the
    field -- you smoke-test the machine, then start the real sweep.
    """
    root = tmp_path / 'results'
    _write_record(root, 'real__dense', tmp_path / 'real.pt',
                  experiment_group='static.dense.resnet50.cifar10.dense.seed42',
                  is_smoke=False, ended_at='2026-01-01T00:00:00+00:00')
    _write_record(root, 'smoke__dense', tmp_path / 'smoke.pt',
                  experiment_group='static.dense.resnet50.cifar10.dense.seed42.smoke',
                  is_smoke=True, limit_train_batches=2, limit_eval_batches=2,
                  ended_at='2026-06-01T00:00:00+00:00')

    found = resolve_checkpoint('resnet50', 'cifar10', seed=42, root=root)
    assert found == str((tmp_path / 'real.pt').resolve())


def test_resolve_checkpoint_raises_when_only_a_smoke_run_exists(tmp_path):
    """No checkpoint is the correct answer here.

    Returning the smoke one is the failure this whole file exists to prevent:
    the run proceeds and the wrongness never surfaces.
    """
    root = tmp_path / 'results'
    _write_record(root, 'smoke__dense', tmp_path / 'smoke.pt',
                  experiment_group='static.dense.resnet50.cifar10.dense.seed42.smoke',
                  is_smoke=True, limit_train_batches=2, limit_eval_batches=2)

    with pytest.raises(FileNotFoundError, match='smoke record'):
        resolve_checkpoint('resnet50', 'cifar10', seed=42, root=root)


def test_the_key_suffix_alone_marks_a_record_as_smoke(tmp_path):
    """Either signal is sufficient, because they are set by different layers.

    `is_smoke` comes from the loader limits at record time; the `.smoke` suffix
    is appended by the notebook cell builder. A record carrying only the suffix
    must still be refused.
    """
    root = tmp_path / 'results'
    _write_record(root, 'suffix_only__dense', tmp_path / 'smoke.pt',
                  experiment_group='static.dense.resnet50.cifar10.dense.seed42.smoke')

    with pytest.raises(FileNotFoundError):
        resolve_checkpoint('resnet50', 'cifar10', seed=42, root=root)


def test_the_is_smoke_flag_alone_marks_a_record_as_smoke(tmp_path):
    """The other direction: a record with no experiment_group at all.

    Scripts run by hand outside the notebooks record no group, so the suffix
    check cannot be the only one.
    """
    root = tmp_path / 'results'
    _write_record(root, 'flag_only__dense', tmp_path / 'smoke.pt',
                  experiment_group=None, is_smoke=True, limit_train_batches=2)

    with pytest.raises(FileNotFoundError):
        resolve_checkpoint('resnet50', 'cifar10', seed=42, root=root)


def test_the_error_names_the_notebook_that_produces_the_baseline(tmp_path):
    """The old message pointed at 02_dense_baselines, which does not exist."""
    root = tmp_path / 'results'
    (root / 'runs').mkdir(parents=True)

    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_checkpoint('resnet50', 'cifar10', root=root)

    message = str(excinfo.value)
    assert '02_dense_baselines' not in message
    assert 'project/test_notebooks/<family>/resnet50/resnet50.ipynb' in message

def test_a_smoke_cell_may_still_chain_from_a_smoke_checkpoint(tmp_path):
    """The guard is directional, not absolute.

    00_smoke_test_all builds every cell with smoke=True, and its prune and bacp
    cells resolve the smoke dense checkpoint. An unconditional guard would leave
    that notebook able to check only 1 of the 3 pipelines it exists to check --
    a regression dressed up as safety.
    """
    root = tmp_path / 'results'
    _write_record(root, 'smoke__dense', tmp_path / 'smoke.pt',
                  experiment_group='static.dense.resnet50.cifar10.dense.seed42.smoke',
                  is_smoke=True, limit_train_batches=2, limit_eval_batches=2)

    found = resolve_checkpoint('resnet50', 'cifar10', seed=42, root=root,
                               allow_smoke=True)
    assert found == str((tmp_path / 'smoke.pt').resolve())


def test_attach_checkpoint_allows_smoke_only_for_a_smoke_cell(tmp_path):
    """The decision is taken from the cell's own key, not from a global flag."""
    import runner as R

    root = tmp_path / 'results'
    _write_record(root, 'smoke__dense', tmp_path / 'smoke.pt',
                  experiment_group='static.dense.resnet50.cifar10.dense.seed42.smoke',
                  is_smoke=True, limit_train_batches=2, limit_eval_batches=2)

    def _cell(key):
        return {'key': key, 'script': 'pruning', 'seed': 42,
                'model_name': 'resnet50', 'dataset_name': 'cifar10',
                'config': {}}

    smoke = R.attach_checkpoint(
        _cell('static.prune.resnet50.cifar10.s0.95.magnitude.seed42.smoke'),
        root=root)
    assert smoke['config']['trained_weights'] == str((tmp_path / 'smoke.pt').resolve())

    with pytest.raises(FileNotFoundError, match='smoke record'):
        R.attach_checkpoint(
            _cell('static.prune.resnet50.cifar10.s0.95.magnitude.seed42'),
            root=root)
