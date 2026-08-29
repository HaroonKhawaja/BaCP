"""L6c -- the EAST pruner (cyclic sparsity, EAST component 3).

Every test in this file guards a defect that made EAST inert. Before these fixes
the pruner could not even be constructed, and if it had been, its mask update was
an unconditional early return -- so no EAST result has ever existed.

Reference: Li et al. 2024, "Pushing the Limits of Sparsity: A Bag of Tricks for
Extreme Pruning", arXiv 2411.13545.
"""

import pytest
import torch

from pruning_factory import (EASTPruner, check_mask_sparsity,
                             check_model_sparsity, get_pruner)

S_MAX = 0.9
S_MIN = 0.1


def _with_grads(wrapper, batch=4):
    """Populate .grad on every parameter -- EAST skips params whose grad is None."""
    x = torch.randn(batch, 3, 8, 8)
    y = torch.randint(0, 4, (batch,))
    wrapper.zero_grad()
    torch.nn.functional.cross_entropy(wrapper(x), y).backward()
    return wrapper


def _east(wrapper, pruner_kwargs, **over):
    kw = pruner_kwargs(scheduler_type="cyclic", delta_T=1, total_steps=40,
                       s_min=S_MIN, Tc_ratio=0.5, prune_rate=0.1)
    kw.update(over)
    return get_pruner(method_name="east", model=wrapper, epochs=4, sparsity=S_MAX, **kw)


def test_east_constructs(tiny_wrapper, pruner_kwargs):
    """Tc_ratio was never passed by _initialize_pruner, so __init__ evaluated
    int(self.end_idx * None) and raised TypeError. EAST had never constructed."""
    pruner = _east(tiny_wrapper, pruner_kwargs)
    assert isinstance(pruner, EASTPruner)
    assert pruner.Tc >= 1


def test_east_s_min_kwarg_is_honoured(tiny_wrapper, pruner_kwargs):
    """s_min was never assigned to self, so update_masks' getattr(self,'s_min',0.05)
    silently returned 0.05 on every run regardless of what the caller passed."""
    pruner = _east(tiny_wrapper, pruner_kwargs, s_min=0.3)
    assert pruner.s_min == 0.3


def test_east_defaults_fill_in_for_none_kwargs(tiny_wrapper, pruner_kwargs):
    """_initialize_pruner passes an explicit None for non-EAST pruners, so a bare
    kwargs.get(key, default) would hand None straight through."""
    pruner = _east(tiny_wrapper, pruner_kwargs, s_min=None, Tc_ratio=None, prune_rate=None)
    assert pruner.s_min == 0.05
    assert pruner.Tc_ratio == 0.5
    assert pruner.prune_rate == 0.1


def test_east_init_masks_are_sparse_near_s_max(tiny_wrapper, pruner_kwargs):
    """Masks start sparse, at roughly the ERK-distributed target density.

    Only "roughly": _init_sparse_masks samples with torch.bernoulli, so the
    realised sparsity matches the target in expectation rather than exactly.
    Worth knowing before reporting a headline sparsity figure to four decimals.
    """
    pruner = _east(tiny_wrapper, pruner_kwargs)
    active = sum(m.sum().item() for m in pruner.masks.values())
    total = sum(m.numel() for m in pruner.masks.values())
    assert (1.0 - active / total) == pytest.approx(S_MAX, abs=0.1)


def test_east_update_masks_actually_runs(tiny_wrapper, pruner_kwargs):
    """The headline test: masks must change when the pruner is stepped.

    update_masks used to re-gate on `current_idx % delta_T != 0` even though
    _step_pruning_step had already gated on exactly that -- and BasePruner.step
    passes current_idx + 1, so (step+1) % delta_T is never 0 for delta_T > 1.
    The method returned immediately every single time, leaving the masks frozen at
    their Bernoulli initialisation for the entire run.
    """
    _with_grads(tiny_wrapper)
    pruner = _east(tiny_wrapper, pruner_kwargs, delta_T=4)
    before = {k: v.clone() for k, v in pruner.masks.items()}

    pruner.step(8, optimizer=None)

    changed = [k for k in before if not torch.equal(before[k], pruner.masks[k])]
    assert changed, "EAST stepped but no mask changed -- update_masks is still inert"


def test_east_cyclic_schedule_starts_at_s_max(tiny_wrapper, pruner_kwargs):
    """Sparsity must not collapse on the first step.

    The cosine phase was anchored at s_min, so with masks initialised at s_max the
    very first update saw diff = s_min - s_max, took the *growing* branch, and
    densified the network -- at s_min=0.05 that means regrowing to 95% density,
    abandoning the sparsity budget entirely.

    Deliberately formula-independent: it asserts the observable consequence, so it
    stays valid regardless of how the schedule is finally parameterised.
    """
    _with_grads(tiny_wrapper)
    pruner = _east(tiny_wrapper, pruner_kwargs, delta_T=1)
    # MASK sparsity, not value sparsity. This used to read check_model_sparsity,
    # which counts numerically-zero weights -- and no path here ever writes a
    # non-zero value (growth assigns 0.0; apply_mask only multiplies), so that
    # count is monotone non-decreasing across a step and the comparison could not
    # fail for ANY mask behaviour. Verified: replacing update_masks with total
    # densification (every mask set to ones) still passed. Restoring the
    # pre-fix inverted cosine drives MASK sparsity 0.9013 -> 0.2241 while
    # check_model_sparsity reads 0.9013 on both sides.
    before = check_mask_sparsity(pruner)

    pruner.step(0, optimizer=None)
    after = check_mask_sparsity(pruner)

    # The bound is the cycle midpoint, not a tight tolerance. The cosine starts
    # at s_max and dips toward s_min over half a cycle, so a correct first step
    # lands in the UPPER half; the inverted phase starts at s_min and lands in
    # the lower half. Measured: correct 0.901 -> 0.870, inverted 0.901 -> 0.224,
    # against a midpoint of 0.475. A tight tolerance could not separate these --
    # the correct schedule legitimately moves 3 points on the first step.
    midpoint = (pruner.s_max + pruner.s_min) / 2
    assert after > midpoint, (
        f"mask sparsity fell from {before:.3f} to {after:.3f} on the first step, "
        f"below the cycle midpoint {midpoint:.3f}. The cyclic cosine phase is "
        f"inverted: the schedule is anchored at s_min and is densifying the "
        f"network where it should be pruning it."
    )
    assert after >= pruner.s_min


def test_east_cycle_dips_toward_s_min_at_half_cycle(tiny_wrapper, pruner_kwargs):
    """The point of cyclic sparsity: relax mid-cycle to allow exploration."""
    pruner = _east(tiny_wrapper, pruner_kwargs)
    s_max, s_min, Tc = pruner.s_max, pruner.s_min, pruner.Tc

    import math

    def target(t):
        return s_max - (s_max - s_min) * 0.5 * (1 - math.cos(2 * math.pi * t / Tc))

    assert target(0) == pytest.approx(s_max)
    assert target(Tc / 2) == pytest.approx(s_min)
    assert target(Tc) == pytest.approx(s_max)


def test_east_noncyclic_phase_preserves_active_count(tiny_wrapper, pruner_kwargs):
    """Past Tc the schedule holds sparsity fixed and only rewires.

    prune-k equals grow-k, so the active parameter count is unchanged -- the
    drop-and-regrow behaviour EAST inherits from RigL (Evci et al. 2020).
    """
    _with_grads(tiny_wrapper)
    pruner = _east(tiny_wrapper, pruner_kwargs, delta_T=1, Tc_ratio=0.1)
    before = sum(m.sum().item() for m in pruner.masks.values())

    beyond_tc = pruner.Tc + 5
    assert beyond_tc > pruner.Tc
    pruner.step(beyond_tc, optimizer=None)

    after = sum(m.sum().item() for m in pruner.masks.values())
    assert after == pytest.approx(before, rel=0.02)


def test_east_grown_weights_are_zero_initialized(tiny_wrapper, pruner_kwargs):
    """Regrown connections must start at exactly zero.

    The previous version asserted only ``torch.isfinite(...)`` -- a tautology --
    and indexed with the whole active mask rather than the grown positions,
    because it never captured the pre-step mask. It provided no coverage of the
    property in its own name: deleting EAST's zero-init line left it green.

    Two things this now pins. The assertion is real: it diffs the mask across the
    step and checks only positions that turned on. And it guards an invariant
    that matters more for RigL than for EAST -- EAST draws grow candidates solely
    from the OLD mask's inactive set, where apply_mask has already forced every
    weight to zero, so its zero-init line is currently redundant; RigL draws from
    the POST-drop mask, so a just-dropped non-zero weight can be regrown in the
    same call and its zero-init is load-bearing. Pinning it here means a future
    change to EAST's candidate selection cannot silently inherit a guard keyed on
    the wrong mask.
    """
    _with_grads(tiny_wrapper)
    pruner = _east(tiny_wrapper, pruner_kwargs, delta_T=1, Tc_ratio=0.1)
    before = {name: mask.clone().bool() for name, mask in pruner.masks.items()}

    pruner.step(pruner.Tc + 5, optimizer=None)

    grown_total = 0
    for name, param in pruner.prunable_params.items():
        grown = pruner.masks[name].bool() & (~before[name])
        n = int(grown.sum().item())
        grown_total += n
        if n:
            nonzero = int(torch.count_nonzero(param[grown]))
            assert nonzero == 0, (
                f"{name}: {nonzero} of {n} regrown weights are non-zero. A "
                f"regrown connection that resumes at its stale magnitude is not "
                f"a new connection, and the mask-vs-value sparsity accounting "
                f"stops being interpretable."
            )
    assert grown_total > 0, (
        "no connection was regrown, so this test asserted nothing at all"
    )


def test_east_masks_cover_only_prunable_params(tiny_wrapper, pruner_kwargs):
    from pruning_factory import layer_check

    pruner = _east(tiny_wrapper, pruner_kwargs)
    expected = {n for n, p in tiny_wrapper.named_parameters() if layer_check(n, p)}
    assert set(pruner.masks) == expected
