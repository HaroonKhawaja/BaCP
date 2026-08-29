"""L7 -- the eight initialisers, and one optimizer step.

test_initialize_pruner_reads_only_declared_fields is the highest-value test in the
suite: it is parametrized over both dataclasses and would have caught, in
milliseconds, the field drift that made every BaCP run raise AttributeError inside
__post_init__.
"""

from dataclasses import fields

import pytest
import torch

import training_utils
from training_utils import (
    _initialize_all,
    _initialize_data_loaders,
    _initialize_models,
    _initialize_optimizer,
    _initialize_pruner,
    _initialize_scheduler,
    _optimizer_step,
    _step_pruning_step,
)

PRUNE = dict(pruning_type="magnitude", target_sparsity=0.5, sparsity_scheduler="cubic")


def _ready(fake_args, patch_models, patch_dataloaders, **over):
    """An args object carried far enough that the pruner can be initialised."""
    args = fake_args(**over)
    _initialize_models(args)
    _initialize_data_loaders(args)
    _initialize_optimizer(args)
    return args


# --- Optimizer ----------------------------------------------------------

def test_initialize_optimizer_sgd(fake_args, patch_models, patch_dataloaders):
    args = _ready(fake_args, patch_models, patch_dataloaders, optimizer_type="sgd")
    assert isinstance(args.optimizer, torch.optim.SGD)
    assert args.optimizer.param_groups[0]["lr"] == pytest.approx(0.1)


def test_initialize_optimizer_adamw(fake_args, patch_models, patch_dataloaders):
    args = _ready(fake_args, patch_models, patch_dataloaders,
                  optimizer_type="adamw", learning_rate=1e-4)
    assert isinstance(args.optimizer, torch.optim.AdamW)


def test_initialize_optimizer_unknown_raises(fake_args, patch_models, patch_dataloaders):
    args = fake_args(optimizer_type="rmsprop")
    _initialize_models(args)
    with pytest.raises(ValueError, match="Unknown optimizer"):
        _initialize_optimizer(args)


# --- Scheduler ----------------------------------------------------------

def test_initialize_scheduler_none(fake_args, patch_models, patch_dataloaders):
    args = _ready(fake_args, patch_models, patch_dataloaders, scheduler_type=None)
    _initialize_scheduler(args)
    assert args.scheduler is None


def test_initialize_scheduler_cosine(fake_args, patch_models, patch_dataloaders):
    args = _ready(fake_args, patch_models, patch_dataloaders, scheduler_type="cosine")
    _initialize_scheduler(args)
    assert args.scheduler is not None
    assert args.total_steps == args.epochs * len(args.trainloader)


def test_initialize_scheduler_linear_with_warmup(fake_args, patch_models, patch_dataloaders):
    args = _ready(fake_args, patch_models, patch_dataloaders,
                  scheduler_type="linear_with_warmup")
    _initialize_scheduler(args)
    assert args.warmup_steps == int(args.total_steps * 0.1)


# --- Pruner -------------------------------------------------------------

@pytest.mark.parametrize("kind", ["baseline", "bacp"])
def test_initialize_pruner_reads_only_declared_fields(kind, real_args):
    """_initialize_pruner must only touch fields both dataclasses declare.

    It reads args.weight_sharing_en and args.delta_T unconditionally. Both were
    declared on TrainingArguments only, so every BaCP run raised AttributeError
    here during __post_init__ -- before a batch was loaded. The cause was the
    eight-call initialisation sequence being duplicated across the two dataclasses
    and drifting; that duplication is now a single _initialize_all.
    """
    args = real_args(kind, trained_weights="does-not-exist.pt", **PRUNE)
    _initialize_models(args)
    _initialize_data_loaders(args)
    _initialize_optimizer(args)
    _initialize_pruner(args)
    assert args.pruner is not None


@pytest.mark.parametrize("kind", ["baseline", "bacp"])
def test_initialize_all_completes_for_both_dataclasses(kind, real_args):
    args = real_args(kind, trained_weights="does-not-exist.pt", **PRUNE)
    _initialize_all(args)
    for attr in ("model", "trainloader", "optimizer", "pruner", "save_path", "logger_params"):
        assert hasattr(args, attr), f"missing {attr}"


def test_pruner_is_none_without_pruning_type(fake_args, patch_models, patch_dataloaders):
    args = _ready(fake_args, patch_models, patch_dataloaders)
    _initialize_pruner(args)
    assert args.pruner is None


def test_pruner_honours_a_supplied_module(fake_args, patch_models, patch_dataloaders, spy_pruner):
    sentinel = spy_pruner()
    args = _ready(fake_args, patch_models, patch_dataloaders,
                  pruning_module=sentinel, **PRUNE)
    _initialize_pruner(args)
    assert args.pruner is sentinel


def test_initializer_dependency_order_is_enforced(fake_args):
    """The sequence order is load-bearing, and violating it must fail loudly.

    Documents the graph rather than leaving it as tribal knowledge: the optimizer
    needs args.model, so it cannot run before _initialize_models.
    """
    args = fake_args()
    if hasattr(args, "model"):
        del args.model
    with pytest.raises(AttributeError):
        _initialize_optimizer(args)


# --- One optimizer step -------------------------------------------------

def _step_once(args, batch=None):
    from training_utils import _handle_data_to_device

    batch = batch or next(iter(args.trainloader))
    data, labels = _handle_data_to_device(args, batch)
    logits = args.model(data)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    _optimizer_step(args, loss, global_step=1)
    return loss


def test_optimizer_step_reduces_loss(fake_args, patch_models, patch_dataloaders):
    args = _ready(fake_args, patch_models, patch_dataloaders)
    _initialize_scheduler(args)
    _initialize_pruner(args)

    batch = next(iter(args.trainloader))
    first = _step_once(args, batch).item()
    for _ in range(6):
        last = _step_once(args, batch).item()
    assert last < first


def test_optimizer_step_reapplies_the_mask(fake_args, patch_models, patch_dataloaders):
    """The core sparse-training invariant: pruned weights stay exactly 0.0.

    optimizer.step() has no knowledge of the mask, so without a re-application
    afterwards momentum and weight decay would immediately revive pruned weights.
    Nothing else in the repository checked this.
    """
    args = _ready(fake_args, patch_models, patch_dataloaders, **PRUNE)
    _initialize_scheduler(args)
    _initialize_pruner(args)

    args.pruner.step(args.pruner.end_idx, optimizer=args.optimizer)
    zeroed = {n: (p == 0) for n, p in args.pruner.prunable_params.items()}
    assert any(z.any() for z in zeroed.values())

    _step_once(args)

    for name, param in args.pruner.prunable_params.items():
        assert (param[zeroed[name]] == 0).all(), f"{name} revived after optimizer.step()"


def test_optimizer_step_without_pruner_is_safe(fake_args, patch_models, patch_dataloaders):
    args = _ready(fake_args, patch_models, patch_dataloaders)
    _initialize_scheduler(args)
    args.pruner = None
    _step_once(args)


def test_optimizer_step_advances_the_scheduler_once(fake_args, patch_models, patch_dataloaders):
    args = _ready(fake_args, patch_models, patch_dataloaders, scheduler_type="cosine")
    _initialize_scheduler(args)
    args.pruner = None
    before = args.scheduler.last_epoch
    _step_once(args)
    assert args.scheduler.last_epoch == before + 1


def test_optimizer_step_with_mixed_precision_on_cpu(fake_args, patch_models, patch_dataloaders):
    """GradScaler on CPU warns once and self-disables, so the AMP path is locally
    exercisable: unscale_/step/update become no-op passthroughs."""
    from torch.amp import GradScaler

    args = _ready(fake_args, patch_models, patch_dataloaders,
                  enable_mixed_precision=True)
    _initialize_scheduler(args)
    args.pruner = None
    args.scaler = GradScaler()

    before = next(iter(args.model.parameters())).clone()
    _step_once(args)
    assert not torch.equal(before, next(iter(args.model.parameters())))


# --- Pruning gate -------------------------------------------------------

def test_step_pruning_gating(fake_args, spy_pruner, tiny_wrapper):
    """delta_T, end_idx and recover gate independently."""
    pruner = spy_pruner(end_idx=100)
    args = fake_args(delta_T=10, **PRUNE)
    args.pruner = pruner
    args.recover = False
    # _step_pruning_step forwards args.optimizer so the pruner can mask momentum,
    # and re-reads args.model to refresh the cached sparsity after a mask update.
    args.optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(2))], lr=0.1)
    args.model = tiny_wrapper

    for step in (0, 5, 10, 20, 101, 110):
        _step_pruning_step(args, step)

    called = [idx for idx, _ in pruner.step_calls]
    assert called == [0, 10, 20], f"unexpected step schedule: {called}"

    args.recover = True
    _step_pruning_step(args, 30)
    assert [idx for idx, _ in pruner.step_calls] == [0, 10, 20]


def test_step_pruning_refreshes_the_cached_sparsity(fake_args, patch_models, patch_dataloaders):
    """args.current_sparsity must track the model, not stay at its initial value.

    _get_sparsity_key reads this attribute to key the accuracy history and to label
    the final metrics. Its only writer used to be _epoch_pruning_step; when that was
    removed for driving the pruner on an epoch clock against a step-indexed end_idx,
    the attribute silently froze at the starting sparsity -- so an iteratively pruned
    run would report the sparsity it began at.
    """
    args = _ready(fake_args, patch_models, patch_dataloaders, delta_T=1, **PRUNE)
    _initialize_scheduler(args)
    _initialize_pruner(args)
    args.current_sparsity = 0.0

    _step_pruning_step(args, args.pruner.end_idx)

    from pruning_factory import check_model_sparsity
    assert args.current_sparsity > 0.0
    assert args.current_sparsity == pytest.approx(check_model_sparsity(args.model), abs=1e-6)


def test_step_pruning_passes_the_optimizer(fake_args, spy_pruner, tiny_wrapper):
    """The optimizer must reach the pruner so momentum can be masked."""
    pruner = spy_pruner(end_idx=100)
    args = fake_args(delta_T=1, **PRUNE)
    args.pruner = pruner
    args.recover = False
    args.optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(2))], lr=0.1)
    args.model = tiny_wrapper

    _step_pruning_step(args, 1)
    assert pruner.step_calls[-1][1] is True


# --- Contract guard -----------------------------------------------------

def test_fake_args_covers_the_dataclass_fields(fake_args):
    """The test stand-in must remain a superset of the real dataclass fields.

    Generated from the source of truth so the contract cannot silently drift.
    """
    from bacp import BaCPTrainingArguments
    from trainer import TrainingArguments

    for cls, is_bacp in ((TrainingArguments, False), (BaCPTrainingArguments, True)):
        declared = {f.name for f in fields(cls)} - {"criterion", "auto_init"}
        provided = set(vars(fake_args(is_bacp=is_bacp)))
        missing = declared - provided
        assert not missing, f"{cls.__name__} fields absent from fake_args: {sorted(missing)}"
