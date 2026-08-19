"""L4 -- pruner mask mechanics. All five registry pruners, on a tiny model.

The single most important invariant in a sparse-training codebase is that a
pruned weight stays exactly zero after an optimizer step. Nothing in the repo
checked it before this file.
"""

import pytest
import torch
import torch.nn as nn

from pruning_factory import PRUNER_REGISTRY, get_pruner, layer_check

ALL_PRUNERS = sorted(PRUNER_REGISTRY)


def _grads(wrapper, batch=4):
    x = torch.randn(batch, 3, 8, 8)
    y = torch.randint(0, 4, (batch,))
    wrapper.zero_grad()
    torch.nn.functional.cross_entropy(wrapper(x), y).backward()
    return wrapper


def _build(name, wrapper, pruner_kwargs, sparsity=0.5, **over):
    kw = pruner_kwargs(**over)
    return get_pruner(method_name=name, model=wrapper, epochs=2, sparsity=sparsity, **kw)


# --- layer_check ---------------------------------------------------------

def test_layer_check_selects_the_expected_param_set(tiny_wrapper):
    """Pins exactly which parameters the pruners own.

    The backbone convolutions are in; the task head, the contrastive projection
    head, embeddings and the DyReLU hyperfunctions are all out -- matching the
    protocol stated in the paper's appendix D.4.

    Two exclusions are worth being explicit about. `cls_head` is now excluded, per
    "the fc layer was kept dense"; it used to be prunable, which contradicted the
    paper and is not a rounding error at extreme sparsity (a dense Linear(2048, 100)
    is 204,800 weights against a ~250,000 surviving budget at 99% on ResNet-50).
    And `encoder_head` is excluded because it is discarded after training -- which
    is exactly what let it absorb the contrastive objective; see
    test_projection_head_cannot_absorb_the_objective.
    """
    selected = {n for n, p in tiny_wrapper.named_parameters() if layer_check(n, p)}

    assert any("layer4" in n and "conv" in n for n in selected)
    assert not any("cls_head" in n for n in selected)
    assert not any("encoder_head" in n for n in selected)
    assert not any("hyperfunction" in n or ".relu" in n for n in selected)
    # BatchNorm weights/biases are 1-D and therefore excluded.
    assert not any(n.endswith("bn1.weight") for n in selected)


def test_prune_task_head_flag_changes_the_prunable_set(tiny_wrapper):
    """Dynamic sparse training conventionally prunes everything, including the head.

    Both conventions are defensible; they are not interchangeable. At 99% sparsity
    on ResNet-50/CIFAR-100 a dense Linear(2048, 100) is 204,800 weights against a
    surviving budget of roughly 250,000, so the flag moves the headline figure
    substantially. The paper must state which it reports.
    """
    from pruning_factory import set_prunable_scope

    def selected():
        return {n for n, p in tiny_wrapper.named_parameters() if layer_check(n, p)}

    set_prunable_scope(prune_task_head=False)
    assert not any("cls_head" in n for n in selected())

    set_prunable_scope(prune_task_head=True)
    assert any("cls_head" in n for n in selected())

    # The contrastive projection head stays out either way -- it is discarded after
    # BaCP training and is not part of the deployed model.
    assert not any("encoder_head" in n for n in selected())


def test_prune_embeddings_flag_changes_the_prunable_set():
    from pruning_factory import set_prunable_scope

    weight = nn.Parameter(torch.zeros(8, 8))
    name = "distilbert.embeddings.word_embeddings.weight"

    set_prunable_scope(prune_embeddings=False)
    assert not layer_check(name, weight)

    set_prunable_scope(prune_embeddings=True)
    assert layer_check(name, weight)


def test_prunable_scope_is_shared_by_pruner_and_reporting(tiny_wrapper, pruner_kwargs):
    """The pruner's mask keys and check_model_sparsity's denominator must match.

    This is why the scope is module-level state rather than a per-call argument: if
    the two disagreed, the reported sparsity would silently describe a different set
    of weights than the one actually pruned.
    """
    from pruning_factory import check_model_sparsity, set_prunable_scope

    set_prunable_scope(prune_task_head=True)
    pruner = _build("magnitude", tiny_wrapper, pruner_kwargs)

    counted = {n for n, p in tiny_wrapper.named_parameters() if layer_check(n, p)}
    assert set(pruner.masks) == counted
    assert any("cls_head" in n for n in pruner.masks)

    for idx in range(pruner.end_idx):
        pruner.step(idx, optimizer=None)
    assert check_model_sparsity(tiny_wrapper) == pytest.approx(0.5, abs=0.02)


def test_tied_embedding_detection():
    """set_prunable_scope warns when input and output embeddings share storage.

    DistilBERT and RoBERTa both tie them, so pruning embeddings there also prunes
    the LM head -- the single most consequential silent behaviour in the whole
    sparsity-accounting story.
    """
    from pruning_factory import _has_tied_embeddings

    emb = nn.Embedding(16, 8)
    tied = nn.Linear(8, 16, bias=False)
    tied.weight = emb.weight

    class Tied(nn.Module):
        def get_input_embeddings(self): return emb
        def get_output_embeddings(self): return tied

    class Untied(nn.Module):
        def get_input_embeddings(self): return nn.Embedding(16, 8)
        def get_output_embeddings(self): return nn.Linear(8, 16, bias=False)

    assert _has_tied_embeddings(Tied())
    assert not _has_tied_embeddings(Untied())
    assert not _has_tied_embeddings(nn.Linear(4, 4))   # no such methods at all


def test_layer_check_excludes_embeddings_and_task_heads():
    """Language-model exclusions, checked by name without needing a real model.

    Embeddings are non-negotiable for transformers: DistilBERT and RoBERTa tie
    word_embeddings to the output projection, so pruning that matrix prunes the LM
    head too. On DistilBERT it is 23.4M of 66M parameters, and including it made the
    prunable set 99.7% of the model.
    """
    weight = nn.Parameter(torch.zeros(8, 8))

    for name in ("distilbert.embeddings.word_embeddings.weight",
                 "distilbert.embeddings.position_embeddings.weight",
                 "vocab_transform.weight", "vocab_projector.weight",
                 "lm_head.dense.weight", "classifier.weight",
                 "pre_classifier.weight", "encoder_head.weight"):
        assert not layer_check(name, weight), f"{name} should not be prunable"

    for name in ("distilbert.transformer.layer.0.attention.q_lin.weight",
                 "roberta.encoder.layer.3.output.dense.weight",
                 "model.layer4.0.conv1.weight"):
        assert layer_check(name, weight), f"{name} should be prunable"


def test_layer_check_excludes_1d_and_frozen():
    assert not layer_check("x.bias", torch.zeros(4))
    frozen = nn.Parameter(torch.zeros(4, 4), requires_grad=False)
    assert not layer_check("x.weight", frozen)
    assert layer_check("x.weight", nn.Parameter(torch.zeros(4, 4)))
    assert not layer_check("x.weight", None)


# --- Construction contract ----------------------------------------------

@pytest.mark.parametrize("name", ALL_PRUNERS)
def test_all_registry_pruners_are_constructible(name, tiny_wrapper, pruner_kwargs):
    """Every registered pruner must build from one canonical kwargs dict.

    Two of the five could not: local_magnitude took scheduler_type positionally
    and forwarded a 4th positional argument into a **kwargs-only signature, and
    also called calculate_erk_densities() with no argument against a
    (self, model) signature. east never received Tc_ratio and so evaluated
    int(end_idx * None).
    """
    _grads(tiny_wrapper)
    pruner = _build(name, tiny_wrapper, pruner_kwargs)
    assert pruner.masks, f"{name} produced no masks"


def test_base_pruner_requires_scheduler_delta_and_total_steps(tiny_wrapper):
    with pytest.raises(ValueError, match="scheduler_type"):
        get_pruner(method_name="magnitude", model=tiny_wrapper, epochs=2, sparsity=0.5,
                   scheduler_type=None, delta_T=None, total_steps=None)


def test_mask_keys_equal_prunable_keys(tiny_wrapper, pruner_kwargs):
    pruner = _build("magnitude", tiny_wrapper, pruner_kwargs)
    assert set(pruner.masks) == set(pruner.prunable_params)


# --- Mask application ---------------------------------------------------

def test_apply_mask_zeroes_exactly_the_masked_weights(tiny_wrapper, pruner_kwargs):
    pruner = _build("magnitude", tiny_wrapper, pruner_kwargs)
    name, param = next(iter(pruner.prunable_params.items()))

    mask = torch.ones_like(param)
    mask.view(-1)[:3] = 0.0
    pruner.masks[name] = mask
    pruner.apply_mask()

    assert (param.view(-1)[:3] == 0).all()
    assert (param.view(-1)[3:] != 0).any()


def test_apply_mask_is_idempotent(tiny_wrapper, pruner_kwargs):
    pruner = _build("magnitude", tiny_wrapper, pruner_kwargs)
    pruner.step(1, optimizer=None)
    snapshot = {n: p.clone() for n, p in pruner.prunable_params.items()}
    pruner.apply_mask()
    for n, p in pruner.prunable_params.items():
        assert torch.equal(p, snapshot[n])


# --- Scoring ------------------------------------------------------------

def test_magnitude_keeps_the_largest_weights():
    """Hand-built case: [1,2,3,4] at 50% keeps {3,4}."""
    model = nn.Sequential(nn.Linear(4, 1, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))

    pruner = get_pruner(method_name="magnitude", model=model, epochs=1, sparsity=0.5,
                        scheduler_type="one_shot", delta_T=1, total_steps=1)
    pruner.step(0, optimizer=None)
    assert torch.equal(model[0].weight.view(-1), torch.tensor([0.0, 0.0, 3.0, 4.0]))


def test_global_threshold_is_global_not_per_layer():
    """Two layers with disjoint magnitude ranges: the small one is fully pruned.

    A per-layer threshold would keep half of each layer instead.
    """
    model = nn.Sequential(nn.Linear(4, 1, bias=False), nn.Linear(4, 1, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[0.01, 0.02, 0.03, 0.04]]))
        model[1].weight.copy_(torch.tensor([[10.0, 20.0, 30.0, 40.0]]))

    pruner = get_pruner(method_name="magnitude", model=model, epochs=1, sparsity=0.5,
                        scheduler_type="one_shot", delta_T=1, total_steps=1)
    pruner.step(0, optimizer=None)

    assert (model[0].weight == 0).all()
    assert (model[1].weight != 0).all()


def test_snip_scores_use_gradients(tiny_wrapper, pruner_kwargs):
    """SNIP scores |w * dL/dw|, so it must skip params with no grad rather than
    silently treating them as zero-importance."""
    pruner = _build("snip", tiny_wrapper, pruner_kwargs)
    tiny_wrapper.zero_grad(set_to_none=True)
    pruner.step(1, optimizer=None)  # all grads None -> no crash, no masking
    assert all((m == 1).all() for m in pruner.masks.values())

    _grads(tiny_wrapper)
    pruner.step(2, optimizer=None)
    assert any((m == 0).any() for m in pruner.masks.values())


def test_rigl_preserves_active_count(tiny_wrapper, pruner_kwargs):
    """Drop-k then grow-k leaves the active parameter count unchanged."""
    _grads(tiny_wrapper)
    pruner = _build("rigl", tiny_wrapper, pruner_kwargs, sparsity=0.5,
                    scheduler_type="f_decay")
    pruner.step(1, optimizer=None)
    before = sum(m.sum().item() for m in pruner.masks.values())
    _grads(tiny_wrapper)
    pruner.step(2, optimizer=None)
    after = sum(m.sum().item() for m in pruner.masks.values())
    assert after == pytest.approx(before, rel=0.05)


# --- Optimizer state ----------------------------------------------------

def test_reset_optimizer_states_masks_momentum(tiny_wrapper, pruner_kwargs):
    """Pruned coordinates must be cleared from the optimizer's momentum too.

    Otherwise a stale momentum buffer keeps nudging a weight that the mask has
    just zeroed, and apply_mask has to keep undoing it -- wasted work that also
    perturbs the surviving weights through the shared update.
    """
    pruner = _build("magnitude", tiny_wrapper, pruner_kwargs)
    opt = torch.optim.SGD(tiny_wrapper.parameters(), lr=0.1, momentum=0.9)
    _grads(tiny_wrapper)
    opt.step()

    name, param = next(iter(pruner.prunable_params.items()))
    mask = torch.ones_like(param)
    mask.view(-1)[:3] = 0.0
    pruner.masks[name] = mask
    pruner.optimizer = opt

    pruner.reset_optimizer_states(name, param)
    buf = opt.state[param]["momentum_buffer"]
    assert (buf.view(-1)[:3] == 0).all()


# --- Schedules ----------------------------------------------------------

@pytest.mark.parametrize("scheduler", ["linear", "cubic", "one_shot", "f_decay", "cyclic"])
def test_sparsity_schedule_never_exceeds_target(scheduler, tiny_wrapper, pruner_kwargs):
    pruner = _build("magnitude", tiny_wrapper, pruner_kwargs, sparsity=0.5,
                    scheduler_type=scheduler, total_steps=10)
    for idx in range(12):
        pruner.step(idx, optimizer=None)
        assert pruner.current_sparsity <= 0.5 + 1e-9


@pytest.mark.parametrize("scheduler", ["linear", "cubic"])
def test_sparsity_schedule_is_monotone_and_reaches_target(scheduler, tiny_wrapper, pruner_kwargs):
    pruner = _build("magnitude", tiny_wrapper, pruner_kwargs, sparsity=0.5,
                    scheduler_type=scheduler, total_steps=10)
    seen = []
    for idx in range(pruner.end_idx):
        pruner.step(idx, optimizer=None)
        seen.append(pruner.current_sparsity)

    assert seen == sorted(seen)
    assert seen[-1] == pytest.approx(0.5, abs=1e-6)


# --- Structural guard ---------------------------------------------------

def test_only_one_pruning_call_site_exists():
    """Pruning must be driven on a single clock.

    _epoch_pruning_step passed an *epoch* index while the pruner's end_idx and Tc
    are derived from total_steps -- a *step* index. With both live the same pruner
    was driven on two incompatible scales, and the epoch-indexed path also passed
    no optimizer, silently skipping reset_optimizer_states.
    """
    import training_utils

    assert not hasattr(training_utils, "_epoch_pruning_step"), (
        "_epoch_pruning_step is back; pruning should be step-driven only"
    )
    assert hasattr(training_utils, "_step_pruning_step")
