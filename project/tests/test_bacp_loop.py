"""L9 -- the full BaCP loop, end to end, on a tiny model.

Marked slow. These are the tests that prove BaCP is executable at all: on `main`
the training path hit eight distinct crashes in sequence, the first of them
inside BaCPTrainingArguments.__post_init__ before a single batch was loaded.
"""

import pytest
import torch

pytestmark = pytest.mark.slow

PRUNE = dict(pruning_type="magnitude", target_sparsity=0.5, sparsity_scheduler="cubic")


def _trainer(initialized_args, **overrides):
    from bacp import BaCPTrainer

    kwargs = dict(epochs=2, recovery_epochs=0, enable_finetune=False,
                  # A string path that does not exist. _initialize_models calls
                  # load_weights(model_ft, trained_weights) unconditionally, and
                  # None would blow up in os.path.exists(). A missing path is the
                  # realistic case and exercises the real early-return branch.
                  trained_weights="does-not-exist.pt")
    kwargs.update(overrides)
    return BaCPTrainer(initialized_args("bacp", **kwargs))


def test_bacp_args_construct_with_all_eight_initializers(initialized_args):
    """BaCPTrainingArguments survives the full initialisation sequence.

    Bug 2: _initialize_pruner reads args.weight_sharing_en and args.delta_T, which
    were declared only on TrainingArguments. Every BaCP run therefore raised
    AttributeError here, before training.
    """
    args = initialized_args("bacp", trained_weights="does-not-exist.pt", **PRUNE)
    assert args.model is not None
    assert args.model_pt is not None and args.model_ft is not None
    assert args.trainloader is not None
    assert args.optimizer is not None
    assert args.pruner is not None


def test_bacp_trainer_constructs(initialized_args):
    """BaCPTrainer.__init__ completes.

    Bug 13: __init__ called self._initialize_dyrelu_phasing(), but that is a
    module-level function in training_utils pulled in by a star import, not a
    method -- so this raised AttributeError. It was also a duplicate; the schedule
    is already configured during __post_init__.
    """
    trainer = _trainer(initialized_args, **PRUNE)
    assert trainer.model.training
    assert not trainer.model_pt.training and not trainer.model_ft.training
    assert trainer.snapshots == []


def test_one_train_step_produces_all_four_losses(initialized_args):
    """A single epoch runs and reports CE, PrC, FiC and SnC.

    SnC is exactly 0.0 on the first epoch because no snapshots exist yet -- the
    snapshot list is only appended to at the end of an epoch.
    """
    trainer = _trainer(initialized_args, epochs=1, **PRUNE)
    metrics = trainer._run_train_epoch(0, desc="")

    for key in ("Total Loss", "CE Loss", "PrC Loss", "FiC Loss", "SnC Loss"):
        assert key in metrics, f"missing {key}"
        assert metrics[key] == metrics[key], f"{key} is NaN"
    assert metrics["SnC Loss"] == 0.0
    assert metrics["CE Loss"] > 0.0


def test_bacp_train_two_epochs_end_to_end(initialized_args):
    """The headline gate: two epochs of real BaCP training, no crash.

    Exercises the replacement for _handle_optimizer_and_pruning (undefined), the
    snapshot lifecycle, and the checkpoint write.
    """
    trainer = _trainer(initialized_args, epochs=2, **PRUNE)
    trainer.train()

    from pruning_factory import check_model_sparsity
    sparsity = check_model_sparsity(trainer.model)
    assert sparsity == pytest.approx(0.5, abs=0.05), (
        f"expected ~0.5 sparsity after the schedule, got {sparsity}"
    )
    assert len(trainer.snapshots) >= 1


def test_snapshots_are_cpu_eval_and_independent(initialized_args):
    """A snapshot must be a frozen copy, not a live alias of the model.

    If deepcopy were ever replaced with a reference, SnC would contrast the model
    against itself and collapse to a constant.
    """
    # epochs=2, not 1: _handle_snapshot_creation caps at epochs - 1, so a
    # single-epoch run legitimately produces zero snapshots.
    trainer = _trainer(initialized_args, epochs=2, **PRUNE)
    trainer._handle_snapshot_creation()
    snap = trainer.snapshots[0]

    assert not snap.training
    assert all(p.device.type == "cpu" for p in snap.parameters())

    target = next(iter(trainer.model.model.layer4.parameters()))
    before = next(iter(snap.model.layer4.parameters())).clone()
    with torch.no_grad():
        target.add_(1.0)
    after = next(iter(snap.model.layer4.parameters()))
    assert torch.equal(before, after), "snapshot tracked the live model"


def test_snapshots_cap_at_epochs_minus_one(initialized_args):
    """_handle_snapshot_creation stops at epochs - 1 snapshots.

    Worth pinning because every snapshot is an extra forward pass per batch and
    is moved onto the GPU, so the cap is what bounds both cost and memory. Note
    the paper reports 2 snapshots while the code permits up to epochs - 1.
    """
    trainer = _trainer(initialized_args, epochs=3, **PRUNE)
    for _ in range(10):
        trainer._handle_snapshot_creation()
    assert len(trainer.snapshots) == trainer.epochs - 1


def _contrastive_backward(trainer):
    """Run PrC only (lambda_CE = 0) and return the gradients it produced."""
    from training_utils import _handle_data_to_device

    batch = next(iter(trainer.trainloader))
    data, labels = _handle_data_to_device(trainer, batch)
    data1, data2 = data

    trainer.model.zero_grad(set_to_none=True)
    curr_emb, _ = trainer._get_embeddings_and_logits(data1)
    with torch.no_grad():
        pt_emb = trainer.model_pt(data2, return_emb=True)

    trainer._calculate_contrastive_loss(curr_emb, pt_emb, labels).backward()

    def total(module):
        return sum(p.grad.abs().sum().item()
                   for p in module.parameters() if p.grad is not None)

    return total(trainer.model.model.layer4), total(trainer.model.encoder_head)


@pytest.mark.parametrize("proj_mode,expect_backbone_grad", [
    ("tied_frozen", True),
    ("none", True),
    ("current", True),
])
def test_contrastive_gradient_reaches_the_backbone(initialized_args, proj_mode,
                                                  expect_backbone_grad):
    """The contrastive objective must move the weights being pruned.

    All three modes do propagate *some* gradient into the backbone -- the point of
    the comparison is test_projection_head_cannot_absorb_the_objective below.
    """
    trainer = _trainer(initialized_args, epochs=1, proj_mode=proj_mode,
                       lambdas=[1.0, 0.0, 0.0, 0.0], **PRUNE)  # PrC only
    backbone, _ = _contrastive_backward(trainer)
    assert (backbone > 0) is expect_backbone_grad


def test_projection_head_cannot_absorb_the_objective(initialized_args):
    """tied_frozen must leave the head with no gradient at all.

    This is the mechanism behind the paper's ablation. encoder_head is excluded
    from pruning by layer_check and, on a real ResNet-50, is a Linear(2048, 128)
    holding ~262k dense parameters. Because all three contrastive losses are
    computed only on its output and the teacher branch is frozen, the student could
    satisfy PrC/FiC/SnC by moving the head and never touching the backbone. That is
    why removing PrC, SnC or CE each *improved* 99%-sparsity accuracy, and why
    learnable lambdas drove the three contrastive weights to ~0.003.

    Under tied_frozen the head is one shared frozen matrix, so the only remaining
    route to a lower loss is the backbone.
    """
    tied = _trainer(initialized_args, epochs=1, proj_mode="tied_frozen",
                    lambdas=[1.0, 0.0, 0.0, 0.0], **PRUNE)
    tied_backbone, tied_head = _contrastive_backward(tied)

    current = _trainer(initialized_args, epochs=1, proj_mode="current",
                       lambdas=[1.0, 0.0, 0.0, 0.0], **PRUNE)
    _, current_head = _contrastive_backward(current)

    assert tied_head == 0.0, "the tied head must be frozen"
    assert tied_backbone > 0.0, "the backbone must still receive gradient"
    assert current_head > 0.0, (
        "under proj_mode='current' the head is trainable and absorbs gradient -- "
        "this is the behaviour being replaced"
    )


def test_tied_head_is_one_shared_module(initialized_args):
    args = initialized_args("bacp", trained_weights="does-not-exist.pt",
                            proj_mode="tied_frozen", **PRUNE)
    assert args.model.encoder_head is args.model_pt.encoder_head
    assert args.model.encoder_head is args.model_ft.encoder_head
    assert all(not p.requires_grad for p in args.model.encoder_head.parameters())


def test_proj_mode_none_bypasses_the_head(initialized_args):
    """proj_mode='none' contrasts pooled features, so embeddings are backbone-wide."""
    args = initialized_args("bacp", trained_weights="does-not-exist.pt",
                            proj_mode="none", **PRUNE)
    x = torch.randn(2, 3, 8, 8)
    emb = args.model(x, return_emb=True)
    assert emb.shape[1] == args.model.embedded_dim
    assert emb.shape[1] != args.model.num_out_features


def test_legacy_contrastive_mode_still_runs(initialized_args):
    """contrastive_mode='legacy' must reproduce the original code path.

    Required for reporting a before/after rather than silently replacing the
    results table.
    """
    trainer = _trainer(initialized_args, epochs=1, contrastive_mode="legacy",
                       proj_mode="current", **PRUNE)
    metrics = trainer._run_train_epoch(0, desc="")
    assert metrics["PrC Loss"] > 0.0


def test_finetune_switches_to_the_ft_optimizer(initialized_args):
    """learning_rate_ft must actually take effect.

    finetune() built the optimizer on a throwaway SimpleNamespace, so
    self.optimizer was never replaced and every fine-tuning epoch ever run used the
    contrastive learning rate.
    """
    trainer = _trainer(initialized_args, epochs=1, enable_finetune=True,
                       epochs_ft=1, learning_rate_ft=0.005,
                       optimizer_type_ft="sgd", **PRUNE)
    trainer.save_path = "does-not-exist.pt"
    trainer.finetune()
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.005)


def test_evaluate_without_finetune_falls_back_to_test_loader(initialized_args):
    """evaluate() must work when finetune() never ran.

    --enable_finetune is a store_true flag whose False default overrides the
    dataclass default of True, so the plain CLI invocation skips finetune() and then
    hit AttributeError on the unassigned ft_testloader.
    """
    trainer = _trainer(initialized_args, epochs=1, enable_finetune=False, **PRUNE)
    assert trainer.ft_testloader is None
    metrics = trainer.evaluate()
    assert "sparsity" in metrics


def test_snapshot_cap_respects_num_snapshots(initialized_args):
    trainer = _trainer(initialized_args, epochs=6, num_snapshots=2, **PRUNE)
    for _ in range(10):
        trainer._handle_snapshot_creation()
    assert len(trainer.snapshots) == 2


def test_get_pruner_uses_registry_and_masks_only_prunable_params(initialized_args):
    """get_pruner rebuilds a pruner and recovers masks from the sparse weights.

    Three separate defects, none of which could ever have run: the registry name
    (PRUNING_REGISTRY vs PRUNER_REGISTRY), a 4th positional argument into a
    **kwargs-only signature, and a mask dict spanning every named_parameter --
    which would permanently freeze any weight incidentally equal to 0.0, such as
    a BatchNorm bias.
    """
    from pruning_factory import layer_check

    trainer = _trainer(initialized_args, epochs=1, **PRUNE)
    trainer.save_path = "does-not-exist.pt"  # load_weights early-returns
    pruner = trainer.get_pruner()

    expected = {name for name, param in trainer.model.named_parameters()
                if layer_check(name, param)}
    assert set(pruner.masks) == expected

    excluded = [n for n in pruner.masks if "encoder_head" in n or "bn" in n]
    assert not excluded, f"non-prunable params ended up masked: {excluded}"
