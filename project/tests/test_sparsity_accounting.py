"""L5 -- sparsity accounting.

The notebooks eyeballed printed sparsity tables. These are the same checks as
assertions, plus the weight-sharing arithmetic that decides what a reported
sparsity figure actually means.
"""

import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace

from pruning_factory import check_model_sparsity, get_pruner, layer_check
from training_utils import _calculate_adjusted_sparsity


def test_mask_sparsity_diverges_from_value_sparsity_under_regrowth(tiny_wrapper, pruner_kwargs):
    """The measurement that matters for dynamic sparse training.

    RigL and EAST zero-initialise regrown connections, so a freshly grown weight is
    structurally *active* but numerically zero. check_model_sparsity counts exact
    zeros and therefore counts it as pruned -- overstating sparsity, which is the
    flattering direction.

    Measured here: the mask holds at the target while the value-based figure drifts
    roughly 3 points above it. For monotone pruning (magnitude, SNIP) nothing regrows
    and the two agree, which is why this never mattered before the switch to dynamic
    pruning.
    """
    from pruning_factory import check_mask_sparsity, report_sparsity

    def grads():
        x = torch.randn(4, 3, 8, 8)
        y = torch.randint(0, 4, (4,))
        tiny_wrapper.zero_grad()
        torch.nn.functional.cross_entropy(tiny_wrapper(x), y).backward()

    grads()
    pruner = get_pruner(method_name="rigl", model=tiny_wrapper, epochs=4, sparsity=0.9,
                        **pruner_kwargs(scheduler_type="f_decay", total_steps=40, alpha=0.3))

    for idx in range(1, 8):
        grads()
        pruner.step(idx, optimizer=None)

    by_mask = check_mask_sparsity(pruner)
    by_value = check_model_sparsity(tiny_wrapper)

    assert by_mask == pytest.approx(0.9, abs=0.02), "RigL must preserve the active count"
    assert by_value > by_mask, "zero-init regrowth should inflate the value-based figure"

    # report_sparsity prefers the mask whenever a pruner owns one.
    assert report_sparsity(tiny_wrapper, pruner) == pytest.approx(by_mask)
    assert report_sparsity(tiny_wrapper, None) == pytest.approx(by_value)


def test_mask_and_value_sparsity_agree_for_monotone_pruning(tiny_wrapper, pruner_kwargs):
    """Magnitude pruning never regrows, so both measures coincide."""
    from pruning_factory import check_mask_sparsity

    pruner = get_pruner(method_name="magnitude", model=tiny_wrapper, epochs=2,
                        sparsity=0.5, **pruner_kwargs(total_steps=10))
    for idx in range(pruner.end_idx):
        pruner.step(idx, optimizer=None)

    assert check_mask_sparsity(pruner) == pytest.approx(check_model_sparsity(tiny_wrapper),
                                                       abs=0.01)


def test_check_mask_sparsity_returns_none_without_a_pruner():
    from pruning_factory import check_mask_sparsity
    assert check_mask_sparsity(None) is None


def test_check_model_sparsity_hand_counted():
    """Two 4-element layers, three weights zeroed -> 3/8."""
    model = nn.Sequential(nn.Linear(4, 1, bias=False), nn.Linear(4, 1, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[0.0, 0.0, 1.0, 2.0]]))
        model[1].weight.copy_(torch.tensor([[0.0, 1.0, 2.0, 3.0]]))
    assert check_model_sparsity(model) == pytest.approx(3 / 8)


def test_check_model_sparsity_ignores_excluded_layers(tiny_wrapper):
    """Zeroing all of encoder_head must not move the reported sparsity.

    encoder_head is excluded by layer_check, so it contributes to neither the
    numerator nor the denominator. Worth stating plainly: reported sparsity is
    sparsity *of the prunable backbone*, not of the whole module -- and the
    contrastive projection head lives entirely outside it.
    """
    before = check_model_sparsity(tiny_wrapper)
    with torch.no_grad():
        tiny_wrapper.encoder_head.weight.zero_()
        tiny_wrapper.encoder_head.bias.zero_()
    assert check_model_sparsity(tiny_wrapper) == pytest.approx(before)


def test_sparsity_after_full_schedule_matches_target(tiny_wrapper, pruner_kwargs):
    for target in (0.5, 0.9):
        wrapper = tiny_wrapper
        pruner = get_pruner(method_name="magnitude", model=wrapper, epochs=2,
                            sparsity=target, **pruner_kwargs(total_steps=10))
        for idx in range(pruner.end_idx):
            pruner.step(idx, optimizer=None)
        assert check_model_sparsity(wrapper) == pytest.approx(target, abs=0.02)


def test_erk_densities_sum_to_the_parameter_budget(tiny_wrapper, pruner_kwargs):
    """Sum(density_i * numel_i) must equal (1 - target) * total, with every
    density a valid probability."""
    target = 0.9
    pruner = get_pruner(method_name="east", model=tiny_wrapper, epochs=2,
                        sparsity=target,
                        **pruner_kwargs(scheduler_type="cyclic", total_steps=40))
    densities = pruner.erk_densities

    prunable = {n: p for n, p in tiny_wrapper.named_parameters() if layer_check(n, p)}
    total = sum(p.numel() for p in prunable.values())
    kept = sum(densities[n] * prunable[n].numel() for n in densities)

    assert all(0.0 < d <= 1.0 for d in densities.values())
    assert kept == pytest.approx((1.0 - target) * total, rel=0.05)


# --- Weight-sharing arithmetic -------------------------------------------

def _shared_args(total, unique, target):
    model = SimpleNamespace(total_params=total, unique_params=unique)
    return SimpleNamespace(model=model, target_sparsity=target)


def test_calculate_adjusted_sparsity_math():
    """total=100, unique=50, target=0.9 -> 0.8.

    Keeping 10 of 100 nominal weights means keeping 10 of the 50 unique ones, so
    the unique set is pruned to 80% rather than 90%. This rescaling is what makes
    a shared model comparable to an unshared one at the same headline sparsity,
    and it is implemented correctly.
    """
    assert _adjusted(100, 50, 0.9) == pytest.approx(0.8)


def _adjusted(total, unique, target):
    return _calculate_adjusted_sparsity(_shared_args(total, unique, target))


def test_calculate_adjusted_sparsity_is_identity_without_sharing():
    assert _adjusted(100, 100, 0.9) == pytest.approx(0.9)


def test_calculate_adjusted_sparsity_rejects_unreachable_target():
    """An unreachable target used to silently produce a negative sparsity.

    total=100, unique=50, target=0.3 wants to keep 70 weights out of a unique set
    of 50, giving an adjusted target of -0.4 -- which was then handed straight to
    the pruner.
    """
    with pytest.raises(ValueError, match="unreachable"):
        _adjusted(100, 50, 0.3)


def test_calculate_adjusted_sparsity_without_sharing_metadata_errors_clearly():
    """A model lacking total_params/unique_params must name the cause.

    Reachable through BaCP: weight sharing was only ever applied on the non-BaCP
    branch, so `--weight_sharing_en` on a BaCP run set the flag without ever
    running apply_weight_sharing_resnet, and this produced a bare AttributeError.
    """
    args = SimpleNamespace(model=nn.Linear(4, 4), target_sparsity=0.9)
    with pytest.raises(RuntimeError, match="apply_weight_sharing_resnet"):
        _calculate_adjusted_sparsity(args)


def test_sparsity_denominator_counts_shared_params_once(tiny_wrapper_shared):
    """named_parameters() de-duplicates aliased tensors; state_dict() does not.

    Weight sharing makes several blocks' conv weights the *same* Parameter object.
    check_model_sparsity iterates named_parameters(), which de-duplicates by
    default, so a shared weight is counted once -- which is the correct
    denominator. state_dict() keeps every name, which is a separate hazard when
    loading checkpoints.
    """
    model = tiny_wrapper_shared.model
    assert model.unique_params < model.total_params

    param_ids = [id(p) for p in tiny_wrapper_shared.parameters()]
    assert len(param_ids) == len(set(param_ids)), "parameters() yielded a duplicate"

    state_keys = list(tiny_wrapper_shared.state_dict())
    assert len(state_keys) == len(set(state_keys))
