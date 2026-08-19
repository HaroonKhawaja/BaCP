"""L6a -- DyReLU phasing (EAST component 1).

EAST's first trick: run DyReLU early so the network explores a richer activation
family, then anneal back to plain ReLU so the extra hyperfunction parameters are
gone by the end of training (Li et al. 2024, arXiv 2411.13545; DyReLU from
Chen et al. 2020, ECCV).

The anneal is a scalar beta in [0, 1]:  out = beta*DyReLU(x) + (1-beta)*ReLU(x).
Everything below pins beta's schedule and the two crash paths around it.
"""

import pytest
import torch
import torch.nn as nn

from dyrelu_adapter import DyReLUAdapter, DyReLUB, set_t_for_dyrelu_adapter, step_dyrelu_adapter


def _adapter(t_start=10, t_end=20, t=0, channels=8):
    a = DyReLUAdapter(channels)
    a.t_start, a.t_end, a.t = t_start, t_end, t
    return a


def test_beta_is_one_before_t_start():
    assert _adapter(t=0).get_beta() == 1.0
    assert _adapter(t=9).get_beta() == 1.0


def test_beta_is_zero_at_and_after_t_end():
    assert _adapter(t=20).get_beta() == 0.0
    assert _adapter(t=99).get_beta() == 0.0


def test_beta_is_half_at_midpoint():
    assert _adapter(t=15).get_beta() == pytest.approx(0.5)


def test_beta_decreases_monotonically_across_the_window():
    betas = [_adapter(t=t).get_beta() for t in range(0, 25)]
    assert betas == sorted(betas, reverse=True)
    assert betas[0] == 1.0 and betas[-1] == 0.0


def test_beta_with_zero_length_window_does_not_divide_by_zero():
    """t_start == t_end is a valid request for an instant switch.

    The interpolation divides by (t_end - t_start), which used to raise
    ZeroDivisionError for this configuration.
    """
    assert _adapter(t_start=5, t_end=5, t=5).get_beta() == 0.0
    assert _adapter(t_start=5, t_end=5, t=4).get_beta() == 1.0


def test_beta_unset_window_raises_a_clear_message():
    a = DyReLUAdapter(8)
    with pytest.raises(RuntimeError, match="set_t_for_dyrelu_adapter"):
        a.get_beta()


def test_step_increments_t_and_caches_beta():
    a = _adapter(t_start=0, t_end=4, t=0)
    for _ in range(2):
        a.step()
    assert a.t == 2
    assert a.current_beta == pytest.approx(0.5)


def test_adapter_equals_relu_at_beta_zero():
    a = _adapter(t=999)
    x = torch.randn(2, 8, 4, 4)
    assert torch.equal(a(x), torch.relu(x))


def test_adapter_equals_dyrelu_at_beta_one():
    a = _adapter(t=0)
    x = torch.randn(2, 8, 4, 4)
    assert torch.allclose(a(x), a.dyrelu(x), atol=1e-6)


def test_dyrelu_b_initialises_to_exact_relu():
    """DyReLU-B with a=[1,0], b=[0,0] is max(x, 0) -- i.e. ReLU, exactly.

    This is what makes phasing coherent: at initialisation the richer activation
    is indistinguishable from the one it will decay back into, so the anneal
    introduces no discontinuity. The hyperfunction output is zeroed here to
    isolate the initialisation constants from the learned modulation.
    """
    m = DyReLUB(8)
    with torch.no_grad():
        for p in m.hyperfunction.parameters():
            p.zero_()
        # A zeroed Linear feeds Sigmoid -> 0.5, and the module maps that to
        # 2*0.5 - 1 = 0, leaving exactly a_init / b_init.
    x = torch.randn(4, 8, 3, 3)
    assert torch.allclose(m(x), torch.relu(x), atol=1e-6)


def test_step_dyrelu_adapter_on_model_without_adapters_is_a_noop():
    """A model with no DyReLUAdapter must not crash the phasing step.

    beta_val stays None and the f-string format spec then raised TypeError.
    Reachable whenever dyrelu_phasing_en is True and dyrelu_en is also True:
    BasicBlock's if/elif picks DyReLUB, so no adapters exist to step.
    """
    step_dyrelu_adapter(nn.Sequential(nn.Linear(4, 4), nn.ReLU()))


def test_step_dyrelu_adapter_steps_every_adapter_once(tiny_wrapper_dyrelu):
    adapters = [m for m in tiny_wrapper_dyrelu.modules() if isinstance(m, DyReLUAdapter)]
    assert adapters, "tiny_wrapper_dyrelu should contain real DyReLUAdapter modules"

    set_t_for_dyrelu_adapter(tiny_wrapper_dyrelu, 0, 4)
    step_dyrelu_adapter(tiny_wrapper_dyrelu)
    assert all(a.t == 1 for a in adapters)


def test_set_t_reaches_every_adapter(tiny_wrapper_dyrelu):
    set_t_for_dyrelu_adapter(tiny_wrapper_dyrelu, 3, 7)
    adapters = [m for m in tiny_wrapper_dyrelu.modules() if isinstance(m, DyReLUAdapter)]
    assert all(a.t_start == 3 and a.t_end == 7 for a in adapters)


def test_dyrelu_params_are_not_prunable(tiny_wrapper_dyrelu):
    """The hyperfunction MLPs must be excluded from the pruning mask.

    Correct, because phasing removes them from the forward pass by the end of
    training -- but it also means they are invisible to every sparsity number the
    codebase reports. They are large (a 2048-channel DyReLU-B hyperfunction is
    ~5.2M parameters), and at beta=0 the adapter merely bypasses self.dyrelu
    rather than deleting it, so the modules stay in the state dict. Any reported
    parameter count or checkpoint size has to account for that separately.
    """
    from pruning_factory import layer_check

    offenders = [name for name, param in tiny_wrapper_dyrelu.named_parameters()
                 if layer_check(name, param) and ("hyperfunction" in name or "relu" in name)]
    assert not offenders, f"DyReLU params leaked into the prunable set: {offenders}"
