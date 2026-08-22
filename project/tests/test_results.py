"""L10 -- the results record layer.

Distinct from the experiment invariants (which run against a populated
results/runs.csv): this tests the *writer*, on synthetic runs, in a tmp_path.
"""

import json
from pathlib import Path

import pytest
import torch

import results
from results import (
    RunRecord,
    canonical_method,
    capture_environment,
    capture_git,
    code_fingerprint,
    count_parameters,
    measure_sparsity,
    normalise_metrics,
    record_failure,
    record_run,
    rebuild_index,
    register_gate,
    require_gate,
    resolve_checkpoint,
)

PRUNE = dict(pruning_type="magnitude", target_sparsity=0.5, sparsity_scheduler="cubic")


@pytest.fixture
def rroot(tmp_path):
    """An isolated results tree, so tests never touch the real one."""
    return tmp_path / "results"


# --- provenance -----------------------------------------------------------

def test_capture_git_returns_a_sha_and_dirty_flag():
    g = capture_git()
    assert g["git_sha"] and len(g["git_sha"]) == 40
    assert isinstance(g["git_dirty"], bool)
    # A tree with MODIFIED TRACKED files must carry a patch hash: a HEAD sha
    # alone does not identify the code that ran. Untracked files are excluded
    # -- `git diff HEAD` cannot represent them, so there is no patch to hash,
    # and git_untracked records them instead.
    tracked_dirty = g["git_n_dirty"] - g["git_untracked"]
    if tracked_dirty > 0:
        assert g["git_patch_sha1"], "modified tracked files but no patch hash"


def test_code_fingerprint_is_stable_and_excludes_tests(tmp_path):
    first = code_fingerprint()
    assert first == code_fingerprint()
    assert first != "unknown"

    # Editing a test must not invalidate a gate; editing production code must.
    project = results.repo_root() / "project"
    scratch = project / "tests" / "_fingerprint_probe.py"
    scratch.write_text("# transient\n", encoding="utf-8")
    try:
        assert code_fingerprint() == first, "tests/ should be outside the fingerprint"
    finally:
        scratch.unlink()


def test_capture_environment_records_torch_and_device():
    env = capture_environment()
    assert env["torch_version"] and env["python_version"]
    assert env["gpu_count"] == 0 or env["gpu_name"]
    assert isinstance(env["deterministic_algorithms"], bool)


# --- metric normalisation -------------------------------------------------

def test_normalise_metrics_reconciles_both_trainers():
    """The two evaluate() methods return the same quantities under different keys."""
    baseline = normalise_metrics({"accuracy": 91.2, "val_loss": 0.3, "perplexity": 1.4})
    bacp = normalise_metrics({"Finetuning Accuracy": 91.2,
                              "Finetuning Val Loss": 0.3,
                              "Finetuning Perplexity": 1.4})
    assert baseline == bacp
    assert baseline["test_acc_pct"] == 91.2


def test_normalise_metrics_preserves_unknown_keys():
    """A future addition to evaluate() must not vanish."""
    out = normalise_metrics({"Some New Metric": 5})
    assert out["metric_some_new_metric"] == 5


def test_normalise_metrics_separates_sparsity():
    out = normalise_metrics({"sparsity": 0.9})
    assert out["_sparsity_from_metrics"] == 0.9
    assert "sparsity_reported" not in out


# --- measurement ----------------------------------------------------------

def test_count_parameters_splits_prunable_from_total(tiny_wrapper):
    counts = count_parameters(tiny_wrapper)
    assert counts["total_params"] > counts["prunable_params"] > 0
    assert (counts["prunable_params"] + counts["nonprunable_params"]
            == counts["total_params"])
    assert counts["layer_sparsity"]


def test_count_parameters_reports_mask_coverage(tiny_wrapper, pruner_kwargs):
    from pruning_factory import get_pruner

    pruner = get_pruner(method_name="magnitude", model=tiny_wrapper, epochs=2,
                        sparsity=0.5, **pruner_kwargs(total_steps=10))
    counts = count_parameters(tiny_wrapper, pruner)
    assert counts["mask_covers_prunable"] is True
    assert counts["mask_total"] == counts["prunable_params"]


def test_measure_sparsity_keeps_both_measures(tiny_wrapper, pruner_kwargs):
    from pruning_factory import get_pruner

    pruner = get_pruner(method_name="magnitude", model=tiny_wrapper, epochs=2,
                        sparsity=0.5, **pruner_kwargs(total_steps=10))
    for idx in range(pruner.end_idx):
        pruner.step(idx, optimizer=None)

    s = measure_sparsity(tiny_wrapper, pruner)
    assert s["sparsity_mask"] is not None
    assert s["sparsity_reported"] == pytest.approx(s["sparsity_mask"])
    assert s["sparsity_gap"] == pytest.approx(s["sparsity_value"] - s["sparsity_mask"])


def test_measure_sparsity_without_a_pruner_falls_back_to_values(tiny_wrapper):
    s = measure_sparsity(tiny_wrapper, None)
    assert s["sparsity_mask"] is None
    assert s["sparsity_reported"] == pytest.approx(s["sparsity_value"])


# --- config access traps -------------------------------------------------

def test_cfg_prefers_the_construction_snapshot():
    """finetune() overwrites learning_rate with learning_rate_ft.

    Reading it live after a completed BaCP run would record the fine-tuning value as
    though it had been the contrastive one, silently mislabelling the protocol.
    """
    from types import SimpleNamespace

    trainer = SimpleNamespace(learning_rate=0.005,
                              logger_params={"learning_rate": 0.1})
    assert results._cfg(trainer, "learning_rate") == 0.1


def test_canonical_method_labels_dense_bacp_and_ce_only():
    from types import SimpleNamespace

    dense = SimpleNamespace(logger_params={"pruning_type": None, "target_sparsity": None})
    assert canonical_method(dense) == "dense"

    plain = SimpleNamespace(logger_params={"pruning_type": "magnitude",
                                           "target_sparsity": 0.9})
    assert canonical_method(plain) == "magnitude"

    bacp = SimpleNamespace(lambda_history={}, logger_params={
        "pruning_type": "magnitude", "target_sparsity": 0.9,
        "lambdas": [0.25, 0.25, 0.25, 0.25]})
    assert canonical_method(bacp) == "bacp+magnitude"

    # A CE-only run is a different method, not a variant -- otherwise the control
    # would be averaged into the thing it is controlling for.
    ce = SimpleNamespace(lambda_history={}, logger_params={
        "pruning_type": "magnitude", "target_sparsity": 0.9,
        "lambdas": [0.0, 0.0, 0.0, 1.0]})
    assert canonical_method(ce) == "ce_only+magnitude"


def test_canonical_method_is_not_fooled_by_finetune_flipping_is_bacp():
    """finetune() sets is_bacp=False, so reading it would mislabel the phase."""
    from types import SimpleNamespace

    trainer = SimpleNamespace(lambda_history={}, is_bacp=False, logger_params={
        "pruning_type": "east", "target_sparsity": 0.99})
    assert canonical_method(trainer) == "bacp+east"


# --- end to end -----------------------------------------------------------

def _train_and_record(initialized_args, rroot, **over):
    from trainer import Trainer

    kwargs = dict(epochs=2, recovery_epochs=0, seed=7)
    kwargs.update(over)
    trainer = Trainer(initialized_args("baseline", **kwargs))
    trainer.train()
    metrics = trainer.evaluate(load=False)
    rec = record_run(trainer, metrics, phase="dense", experiment_group="test",
                     root=rroot, strict=True)
    return trainer, rec


@pytest.mark.slow
def test_record_run_writes_a_complete_record(initialized_args, rroot):
    trainer, rec = _train_and_record(initialized_args, rroot, **PRUNE)

    assert rec is not None
    path = rroot / "runs" / f"{rec.record_id}.json"
    assert path.exists()

    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("seed", "git_sha", "code_fingerprint", "total_params",
                "prunable_params", "sparsity_reported", "duration_s",
                "torch_version", "method", "scope_key"):
        assert payload[key] is not None, f"{key} was not recorded"

    assert payload["seed"] == 7
    assert payload["seed_source"] == "args"
    assert payload["method"] == "magnitude"
    assert payload["scope_key"] == "backbone"
    assert payload["epochs_completed"] == 2
    assert payload["s_per_epoch_mean"] > 0
    assert payload["detail"]["epoch_times_s"]


@pytest.mark.slow
def test_record_run_computes_exact_metrics(initialized_args, rroot):
    """The exact metrics must be present and must not silently equal the batch means.

    Reported accuracy is a batch-mean of batch-means; with drop_last=False on eval
    the final batch is short, so the two differ unless the dataset divides evenly.
    """
    _, rec = _train_and_record(initialized_args, rroot)
    assert rec.test_acc_exact_pct is not None
    assert rec.eval_samples_seen and rec.eval_samples_seen > 0
    assert rec.eval_samples_dropped == 0, "eval must score every sample"
    assert rec.metrics_are_batch_means is True


@pytest.mark.slow
def test_rebuild_index_and_load_runs_roundtrip(initialized_args, rroot):
    """One training run produces TWO records, and both must survive the roundtrip.

    trainer.train() ends in _finalize_run, which writes its own 'prune' record --
    that is the production behaviour, and the whole point of _finalize_run. The
    explicit record_run below adds a second, 'dense'-phase record.

    This assertion used to read `n == 1`, and it passed only because the
    trainer's automatic record was escaping into the REAL results/ directory
    while the test inspected an isolated tmp one. 71 such fixture records had
    accumulated there before conftest's _isolate_results_dir closed the leak.
    Two records is the correct expectation; one was the symptom.
    """
    _train_and_record(initialized_args, rroot, **PRUNE)
    n = rebuild_index(root=rroot)
    assert n == 2, "expected the trainer's own record plus the explicit one"

    df = load = results.load_runs(root=rroot)
    assert len(load) == 2
    # load_runs dedupes on (run_id, phase) and ranks phases so the pruning record
    # sorts ahead of the standalone eval one.
    assert set(load["phase"]) == {"prune", "dense"}
    assert set(load["method"]) == {"magnitude"}
    # Priority columns come first, so runs.csv reads identity -> config -> result.
    assert list(df.columns)[:4] == ["record_id", "run_id", "phase", "status"]


def test_record_run_tolerates_a_nearly_empty_trainer(rroot):
    """A bare namespace should still produce a (thin) record rather than crash.

    Every field is read through getattr with a default, so a partially-built trainer
    degrades to a sparse record instead of losing the run entirely.
    """
    from types import SimpleNamespace

    rec = record_run(SimpleNamespace(), {}, root=rroot, strict=False)
    assert rec is not None
    assert rec.method == 'dense'
    assert rec.seed_source == 'unknown'


def test_record_run_swallows_an_exception_rather_than_killing_the_run(rroot):
    """The strict=False safety net. A results bug must never destroy a finished run.

    Simulated with a property that raises on access, which is the shape of a real
    failure here: something the recorder reads turns out not to behave.
    """
    class Hostile:
        @property
        def logger_params(self):
            raise RuntimeError('boom')

    assert record_run(Hostile(), {}, root=rroot, strict=False) is None

    with pytest.raises(RuntimeError, match='boom'):
        record_run(Hostile(), {}, root=rroot, strict=True)


@pytest.mark.slow
def test_resolve_checkpoint_finds_the_dense_run(initialized_args, rroot):
    """Pruning runs ask for their baseline by name instead of a hardcoded path."""
    trainer, rec = _train_and_record(initialized_args, rroot)
    assert rec.method == "dense"

    found = resolve_checkpoint(rec.model_name, rec.dataset_name, root=rroot)
    assert Path(found).is_file()
    assert found == rec.save_path


def test_resolve_checkpoint_error_names_the_fix(rroot):
    with pytest.raises(FileNotFoundError, match="02_dense_baselines"):
        resolve_checkpoint("resnet50", "cifar100", root=rroot)


# --- failures are rows, not silences -------------------------------------

def test_record_failure_writes_a_row(rroot):
    """A crashed run must be visible. Otherwise it is indistinguishable from a cell
    that was never scheduled, and a table shows a blank either way."""
    cell = {"key": "resnet50_cifar100_east_s0.99_seed42", "model_name": "resnet50",
            "dataset_name": "cifar100", "method": "east", "seed": 42,
            "sparsity": 0.99, "cmd": ["python", "x.py"]}
    rec = record_failure(cell, returncode=1, log_tail="CUDA out of memory",
                         experiment_group="04_main_results", root=rroot)

    assert rec.status == "failed"
    payload = json.loads((rroot / "runs" / f"{rec.record_id}.json").read_text())
    assert payload["error"] == "returncode=1"
    assert "out of memory" in payload["detail"]["log_tail"]

    # load_runs filters to ok by default, so a failure never pollutes a table...
    assert results.load_runs(root=rroot).empty
    # ...but is retrievable when you go looking.
    assert len(results.load_runs(root=rroot, status=("ok", "failed"))) == 1


# --- gates ----------------------------------------------------------------

def test_gate_roundtrip(rroot):
    register_gate("demo", [("thing works", True, "42 rows")], root=rroot)
    assert require_gate("demo", root=rroot)["ok"] is True


def test_require_gate_raises_when_absent(rroot):
    with pytest.raises(RuntimeError, match="has not run"):
        require_gate("never_ran", root=rroot)


def test_require_gate_raises_when_failed(rroot):
    register_gate("bad", [("parity", False, "delta=-4.9")], root=rroot)
    with pytest.raises(RuntimeError, match="failed"):
        require_gate("bad", root=rroot)


def test_require_gate_detects_stale_code(rroot):
    """The staleness check is the point of the gate chain.

    It stops a long-running result from silently satisfying a downstream gate after
    the code underneath it changed.
    """
    payload = register_gate("stale", [("ok", True, "")], root=rroot)
    path = rroot / "gates" / "stale.json"
    payload["code_fingerprint"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="stale"):
        require_gate("stale", root=rroot)
    assert require_gate("stale", allow_stale=True, root=rroot)["ok"] is True


# --- checkpoint paths must be absolute -------------------------------------
#
# A tier-0 sweep failed all 21 sparse rungs against a dense checkpoint that was
# sitting on disk the whole time: save_path is built relative to the CWD
# ('./research/bacp/...'), the runner launches each script with
# cwd=project/scripts, and resolve_checkpoint then evaluated that relative path
# from the repo root. Nothing raised at record time -- the failure surfaced 21
# cells later as FileNotFoundError, and looked like a missing baseline rather
# than a path bug.

def test_recorded_save_path_is_absolute(initialized_args, rroot, tmp_path, monkeypatch):
    """The record must not depend on the CWD of whoever reads it later."""
    from trainer import Trainer

    monkeypatch.chdir(tmp_path)
    trainer = Trainer(initialized_args("baseline", epochs=1, recovery_epochs=0, seed=7))
    trainer.train()
    rec = record_run(trainer, trainer.evaluate(load=False), phase="dense",
                     experiment_group="abs-path", root=rroot, strict=True)

    assert rec.save_path, "no save_path recorded"
    assert Path(rec.save_path).is_absolute(), (
        f"save_path {rec.save_path!r} is relative -- it only resolves from the "
        f"directory the training script happened to run in")


def test_resolve_checkpoint_rehydrates_a_relative_legacy_path(rroot, tmp_path):
    """Records written before the fix must still resolve.

    Re-running a dense baseline to repair a path bug costs a full dataset
    download, so resolve_checkpoint tries the known launch directories rather
    than failing on a record that predates the change.
    """
    import json

    from results import _atomic_write_json, resolve_checkpoint

    ckpt_dir = tmp_path / "research" / "bacp"
    ckpt_dir.mkdir(parents=True)
    ckpt = ckpt_dir / "m.pt"
    ckpt.write_bytes(b"weights")

    (rroot / "runs").mkdir(parents=True, exist_ok=True)
    _atomic_write_json(rroot / "runs" / "legacy__dense.json", {
        "record_id": "legacy__dense", "status": "ok", "method": "dense",
        "model_name": "m", "dataset_name": "d", "seed": 1,
        "ended_at": "2026-01-01T00:00:00+00:00",
        # relative, exactly as the old writer stored it
        "save_path": str(Path("research") / "bacp" / "m.pt"),
    })

    import os
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        got = resolve_checkpoint("m", "d", seed=1, method="dense", root=rroot)
    finally:
        os.chdir(cwd)

    assert Path(got).is_absolute()
    assert Path(got).is_file()
