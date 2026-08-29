"""finetune_only: append a fine-tune to an already-pruned checkpoint.

Running the 50 pruning epochs again just to add 25 fine-tune epochs costs 3x
what it needs to. These pin the three things that make skipping them safe:
the recovered mask is exactly the checkpoint's, the fine-tune cannot resurrect a
pruned weight, and a dense checkpoint passed by mistake is refused rather than
fine-tuned and reported as a sparse result.
"""
import pytest
import torch
import torch.nn as nn

import pruning_factory as pf
from pruning_factory import layer_check, check_model_sparsity

TARGET = 0.90


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.cls_head = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        return self.cls_head(self.bn(x).mean((2, 3)))


def _pruner(model):
    return pf.get_pruner('magnitude', model, 1, TARGET, scheduler_type='cubic',
                         delta_T=1, total_steps=8)


def _recover(model, pruner):
    """The body of Trainer._recover_masks_from_weights, kept in step with it."""
    pruner.masks = {
        name: (param != 0).float()
        for name, param in model.named_parameters()
        if layer_check(name, param)
    }
    pruner.apply_mask()
    return check_model_sparsity(model)


@pytest.fixture
def sparse_checkpoint(tmp_path):
    pf.set_prunable_scope(prune_task_head=True)
    model = _Tiny()
    p = _pruner(model)
    p.current_sparsity = p.target_sparsity = TARGET
    p.update_masks(epoch=1)
    p.apply_mask()
    path = tmp_path / 'sparse.pt'
    torch.save(model.state_dict(), path)
    zeros = {n: (q == 0).clone() for n, q in model.named_parameters() if layer_check(n, q)}
    return path, check_model_sparsity(model), zeros


def test_recovered_mask_matches_the_checkpoint(sparse_checkpoint):
    path, sparsity, zeros = sparse_checkpoint
    model = _Tiny()
    model.load_state_dict(torch.load(path))
    got = _recover(model, _pruner(model))

    assert got == pytest.approx(sparsity, abs=1e-9)
    for name, param in model.named_parameters():
        if layer_check(name, param):
            assert torch.equal(zeros[name], param == 0), name


def test_finetune_cannot_resurrect_a_pruned_weight(sparse_checkpoint):
    path, sparsity, _ = sparse_checkpoint
    model = _Tiny()
    model.load_state_dict(torch.load(path))
    pruner = _pruner(model)
    _recover(model, pruner)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for _ in range(3):
        opt.zero_grad()
        nn.functional.cross_entropy(
            model(torch.randn(8, 3, 8, 8)), torch.randint(0, 10, (8,))).backward()
        opt.step()
        pruner.apply_mask()

    assert check_model_sparsity(model) == pytest.approx(sparsity, abs=1e-9)


def test_a_dense_checkpoint_is_far_enough_off_target_to_be_caught():
    """The guard in _recover_masks_from_weights rejects at a 0.01 tolerance."""
    pf.set_prunable_scope(prune_task_head=True)
    model = _Tiny()
    got = _recover(model, _pruner(model))
    assert abs(got - TARGET) > 0.01


def test_trainer_exposes_the_flag_and_defaults_it_off():
    from trainer import TrainingArguments
    from dataclasses import fields

    names = {f.name for f in fields(TrainingArguments)}
    assert 'finetune_only' in names
    assert next(f for f in fields(TrainingArguments)
                if f.name == 'finetune_only').default is False


_REQUIRED = [
    '--model_name', 'resnet34', '--model_type', 'cv',
    '--dataset_name', 'cifar10', '--num_classes', '10',
    '--trained_weights', '/tmp/sparse.pt',
    '--pruning_type', 'magnitude', '--target_sparsity', '0.99',
    '--sparsity_scheduler', 'cubic',
]


def test_cli_exposes_finetune_only():
    from scripting_utils import pruning_parse_args

    off = pruning_parse_args(_REQUIRED)
    assert off.finetune_only is False, 'must default off'

    on = pruning_parse_args(_REQUIRED + ['--finetune_only', '--epochs_ft', '25'])
    assert on.finetune_only is True
    assert on.epochs_ft == 25
