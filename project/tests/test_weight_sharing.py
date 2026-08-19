"""L6b -- EAST weight sharing (EAST component 2).

Promotes the notebook's `weight_sharing_test` -- which compared id(module.weight)
and *printed* a check mark -- into real assertions, and pins the checkpoint
ordering hazard that the printed version could not have caught.

Reference: Li et al. 2024, arXiv 2411.13545.
"""

import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace

from pruning_factory import layer_check
from training_utils import _initialize_optimizer, _no_decay_param_groups
from utils import load_weights
from weight_sharing import apply_weight_sharing_resnet

R = 2  # apply_weight_sharing_resnet default: master is block[R-1], blocks R.. share it


@pytest.mark.parametrize("layer_name", ["layer1", "layer2", "layer3", "layer4"])
def test_shared_blocks_alias_the_same_parameter_object(tiny_wrapper_shared, layer_name):
    """block[i].conv1.weight IS block[R-1].conv1.weight for i > R-1.

    Identity, not equality: sharing replaces the Parameter object so a single
    tensor backs several blocks. The tiny model has 3 blocks per layer precisely so
    this is non-vacuous -- with 1 block per layer the function short-circuits on
    `num_blocks <= R` and every assertion here would pass while testing nothing.
    """
    layer = getattr(tiny_wrapper_shared.model, layer_name)
    master = layer[R - 1]

    for i in range(R, len(layer)):
        assert layer[i].conv1.weight is master.conv1.weight
        assert layer[i].conv2.weight is master.conv2.weight

    # Block 0 is never shared: it carries the stride/channel change.
    assert layer[0].conv1.weight is not master.conv1.weight


def test_named_parameters_dedupes_aliases(tiny_wrapper_shared):
    """parameters() yields each shared tensor once, so counts are not inflated."""
    model = tiny_wrapper_shared.model
    assert model.unique_params < model.total_params

    ids = [id(p) for p in tiny_wrapper_shared.parameters()]
    assert len(ids) == len(set(ids))


def test_state_dict_does_not_dedupe(tiny_wrapper_shared):
    """state_dict() keeps every name, even when several map to one tensor.

    This asymmetry with named_parameters() is the root of the loading hazard below.
    """
    layer1 = tiny_wrapper_shared.model.layer1
    keys = tiny_wrapper_shared.state_dict()
    assert "model.layer1.1.conv1.weight" in keys
    assert "model.layer1.2.conv1.weight" in keys
    assert layer1[2].conv1.weight is layer1[1].conv1.weight


def test_loading_after_sharing_is_last_write_wins(tmp_path, tiny_wrapper):
    """Demonstrates the hazard: share-then-load silently discards the master value.

    Because state_dict() does not dedupe, load_state_dict writes every aliased key
    into the same storage in turn, so the shared weight ends up holding whichever
    block came last in iteration order rather than the master's.
    """
    layer1 = tiny_wrapper.model.layer1
    with torch.no_grad():
        layer1[1].conv1.weight.fill_(1.0)
        layer1[2].conv1.weight.fill_(2.0)

    path = tmp_path / "unshared.pt"
    torch.save(tiny_wrapper.state_dict(), path)

    apply_weight_sharing_resnet(tiny_wrapper)
    assert load_weights(tiny_wrapper, str(path)) is True

    master = layer1[R - 1].conv1.weight
    assert torch.allclose(master, torch.full_like(master, 2.0)), (
        "expected block 2's value to have overwritten the master"
    )


def test_loading_before_sharing_preserves_master(tmp_path, tiny_wrapper):
    """The fix: load first, then share. _initialize_models now does this."""
    layer1 = tiny_wrapper.model.layer1
    with torch.no_grad():
        layer1[1].conv1.weight.fill_(1.0)
        layer1[2].conv1.weight.fill_(2.0)

    path = tmp_path / "unshared.pt"
    torch.save(tiny_wrapper.state_dict(), path)

    assert load_weights(tiny_wrapper, str(path)) is True
    apply_weight_sharing_resnet(tiny_wrapper)

    master = layer1[R - 1].conv1.weight
    assert torch.allclose(master, torch.full_like(master, 1.0))


def test_scaler_is_registered_and_receives_gradient(tiny_wrapper_shared):
    layer1 = tiny_wrapper_shared.model.layer1
    shared_block = layer1[R]
    assert isinstance(shared_block.conv1.scaler, nn.Parameter)

    x = torch.randn(2, 3, 8, 8)
    y = torch.randint(0, 4, (2,))
    tiny_wrapper_shared.zero_grad()
    torch.nn.functional.cross_entropy(tiny_wrapper_shared(x), y).backward()

    assert shared_block.conv1.scaler.grad is not None


def test_scaler_zero_zeroes_the_shared_block_contribution(tiny_wrapper_shared):
    """The scaler genuinely gates the shared conv."""
    conv = tiny_wrapper_shared.model.layer1[R].conv1
    x = torch.randn(2, conv.in_channels, 8, 8)

    with torch.no_grad():
        conv.scaler.fill_(0.0)
        out = conv(x)
    assert torch.allclose(out, torch.zeros_like(out))


def test_scaler_is_exempt_from_weight_decay(tiny_wrapper_shared):
    """Scale parameters must land in a zero-weight-decay group.

    Under the default weight_decay=5e-4 the scalers decay toward zero over
    training, progressively silencing the shared blocks -- a slow degradation that
    looks like the method simply not working.
    """
    groups = _no_decay_param_groups(tiny_wrapper_shared, 5e-4)
    assert len(groups) == 2

    no_decay = groups[1]
    assert no_decay["weight_decay"] == 0.0

    scaler_ids = {id(m.scaler) for m in tiny_wrapper_shared.modules()
                  if hasattr(m, "scaler")}
    assert scaler_ids
    assert scaler_ids <= {id(p) for p in no_decay["params"]}


def test_initialize_optimizer_uses_the_no_decay_split(tiny_wrapper_shared):
    args = SimpleNamespace(model=tiny_wrapper_shared, optimizer_type="sgd",
                           learning_rate=0.1)
    _initialize_optimizer(args)
    decays = [g["weight_decay"] for g in args.optimizer.param_groups]
    assert 0.0 in decays and 5e-4 in decays


def test_gradients_accumulate_into_the_master(tiny_wrapper_shared):
    """A shared weight used in N places accumulates gradient from all N."""
    layer1 = tiny_wrapper_shared.model.layer1
    master = layer1[R - 1].conv1.weight

    x = torch.randn(2, 3, 8, 8)
    y = torch.randint(0, 4, (2,))
    tiny_wrapper_shared.zero_grad()
    torch.nn.functional.cross_entropy(tiny_wrapper_shared(x), y).backward()

    assert master.grad is not None and master.grad.abs().sum() > 0


def test_sharing_is_a_noop_when_blocks_do_not_exceed_R(tiny_wrapper):
    """A layer with <= R blocks is skipped, so nothing is aliased."""
    short = nn.Sequential(*[tiny_wrapper.model.layer1[i] for i in range(2)])
    tiny_wrapper.model.layer1 = short
    apply_weight_sharing_resnet(tiny_wrapper)
    assert short[0].conv1.weight is not short[1].conv1.weight


def test_scaler_is_not_prunable(tiny_wrapper_shared):
    """A 0-dim scale parameter must not enter the pruning mask."""
    offenders = [n for n, p in tiny_wrapper_shared.named_parameters()
                 if n.endswith("scaler") and layer_check(n, p)]
    assert not offenders


def test_shared_model_survives_deepcopy(tiny_wrapper_shared):
    """BaCP snapshots deepcopy the model, so sharing must survive a copy.

    The per-instance forward override is a types.MethodType bound to a function
    defined inside apply_weight_sharing_resnet. copy has a MethodType handler so
    deepcopy works, but torch.save(model) -- pickling the whole module rather than
    its state_dict -- cannot serialise a local function. Save state dicts only.
    """
    from copy import deepcopy

    clone = deepcopy(tiny_wrapper_shared)
    layer1 = clone.model.layer1
    assert layer1[R].conv1.weight is layer1[R - 1].conv1.weight

    x = torch.randn(2, 3, 8, 8)
    assert clone(x).shape == (2, 4)
