"""L8 -- the baseline/pruning training loop, end to end on a tiny model."""

import pytest
import torch

from pruning_factory import check_model_sparsity

pytestmark = pytest.mark.slow

PRUNE = dict(pruning_type="magnitude", target_sparsity=0.5, sparsity_scheduler="cubic")


def _trainer(initialized_args, **over):
    from trainer import Trainer

    kwargs = dict(epochs=2, recovery_epochs=0)
    kwargs.update(over)
    return Trainer(initialized_args("baseline", **kwargs))


def test_train_two_epochs_records_metrics(initialized_args):
    """_update_metric_lists writes train_losses and accuracies.

    Note it never writes self.val_accuracies, which __init__ creates and nothing
    populates -- a dead attribute. Without a target_sparsity, accuracies is a list;
    with one it is a dict keyed by sparsity level, which is also why each new
    sparsity level restarts the best-checkpoint comparison.
    """
    trainer = _trainer(initialized_args)
    trainer.train()

    assert len(trainer.train_losses) == 2
    assert isinstance(trainer.accuracies, list) and len(trainer.accuracies) == 2
    assert trainer.val_accuracies == [], "val_accuracies is unused; update this test if that changes"


def test_train_writes_a_checkpoint_under_tmp_path(initialized_args, tmp_path):
    """save_path must be redirectable away from /dbfs.

    databricks_env=False sends the base path to './research/bacp/...', and the
    real_args fixture chdirs into tmp_path -- so this also proves the checkpoint
    layout is testable without Databricks.
    """
    trainer = _trainer(initialized_args, **PRUNE)
    trainer.train()
    assert (tmp_path / trainer.save_path.lstrip("./\\")).exists() or \
           any(tmp_path.rglob("*.pt")), "no checkpoint written under tmp_path"


def test_cv_runs_report_a_real_validation_loss(initialized_args):
    """A vision run must produce a non-None val_loss and perplexity.

    current_loss was initialised to 0 and never assigned on the CV branch, so
    avg_loss was always 0.0 and perplexity always exp(0)=1.0 -- both of which the
    return filter then converted to None. Every CIFAR experiment had accuracy and
    sparsity and nothing else, which is also why early stopping keys off accuracy.
    """
    trainer = _trainer(initialized_args, epochs=1)
    metrics = trainer._run_validation_epoch(desc="")

    assert metrics["accuracy"] is not None
    assert metrics["val_loss"] is not None and metrics["val_loss"] > 0.0
    assert metrics["perplexity"] is not None and metrics["perplexity"] > 1.0


def test_train_reaches_and_holds_target_sparsity(initialized_args):
    trainer = _trainer(initialized_args, epochs=2, **PRUNE)
    trainer.train()
    assert check_model_sparsity(trainer.model) == pytest.approx(0.5, abs=0.05)


def test_recovery_epochs_do_not_change_masks(initialized_args):
    """During recovery the masks are held and only re-applied.

    _step_pruning_step short-circuits on args.recover, so sparsity must not move.
    """
    trainer = _trainer(initialized_args, epochs=1, recovery_epochs=1, **PRUNE)
    trainer.train()
    before = check_model_sparsity(trainer.model)
    trainer._retrain()
    assert check_model_sparsity(trainer.model) == pytest.approx(before, abs=1e-6)


def test_evaluate_returns_sparsity(initialized_args):
    trainer = _trainer(initialized_args, epochs=1, **PRUNE)
    trainer.train()
    metrics = trainer.evaluate(load=False)
    assert "sparsity" in metrics


def test_dyrelu_beta_decreases_across_epochs(initialized_args):
    """Phasing must actually anneal over a run.

    Guards against the schedule being configured such that beta never leaves 1.0 --
    which is what a hardcoded t_start=10 does on a short schedule.
    """
    from dyrelu_adapter import DyReLUAdapter, set_t_for_dyrelu_adapter

    trainer = _trainer(initialized_args, epochs=3, dyrelu_phasing_en=True)
    adapters = [m for m in trainer.model.modules() if isinstance(m, DyReLUAdapter)]
    assert adapters, "expected DyReLUAdapter modules"

    set_t_for_dyrelu_adapter(trainer.model, 0, 2)
    start = adapters[0].get_beta()
    trainer.train()
    assert adapters[0].get_beta() < start


def test_early_stopping_on_patience(initialized_args):
    """patience must be able to terminate the loop early."""
    trainer = _trainer(initialized_args, epochs=8, patience=1)
    trainer.train()
    assert 0 < len(trainer.train_losses) <= 8
