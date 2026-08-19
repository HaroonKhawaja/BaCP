"""L1c -- distillation controls (rungs C1/C2).

These are not part of BaCP. They exist so the claim "the contrastive objective
helps" can be distinguished from the far weaker "a dense teacher helps", which
is the only thing the submitted paper's evidence actually supports: three frozen
dense teachers sat on the BaCP side of every comparison in its Table 1 and zero
on the I.P. side.

Hinton et al. 2015 (KD) and Romero et al. 2015 (FitNets).
"""

import math

import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace

from bacp import BaCPTrainer
from loss_functions import feature_distill_loss, kd_kl_loss


# --- kd_kl_loss ------------------------------------------------------------

def test_self_distillation_is_zero():
    z = torch.randn(16, 10)
    assert float(kd_kl_loss(z, z, T=4.0)) == pytest.approx(0.0, abs=1e-5)


def test_kd_is_non_negative():
    """KL divergence is non-negative; a negative value means the log_softmax and
    softmax arguments were swapped, which trains the student *away* from the
    teacher."""
    torch.manual_seed(0)
    for _ in range(20):
        s, t = torch.randn(8, 12), torch.randn(8, 12)
        assert float(kd_kl_loss(s, t, T=2.0)) >= -1e-6


def test_temperature_squared_keeps_the_gradient_scale():
    """Without the T^2 factor the KD gradient shrinks as 1/T^2.

    That matters for the ladder specifically: raising T would quietly turn the
    term off, and the rung would report "KD does nothing" -- indistinguishable
    from a real null. Hinton et al. sec. 2.1.

    An earlier version of this test assigned the student logits twice, so the
    first tensor (and the gradient measured from it) was dead and the comparison
    was between two independently-seeded draws rather than between temperatures.
    """
    torch.manual_seed(0)
    t = torch.randn(32, 10)
    base = torch.randn(32, 10)

    grads = {}
    for T in (1.0, 8.0):
        s = base.clone().requires_grad_(True)     # same student every time
        kd_kl_loss(s, t, T=T).backward()
        grads[T] = s.grad.abs().mean().item()

    ratio = grads[8.0] / grads[1.0]
    # With the T^2 compensation the two stay within an order of magnitude.
    # Without it this ratio is ~1/64.
    assert ratio > 0.1, (
        f'gradient at T=8 is {ratio:.4f}x the T=1 gradient -- the T^2 '
        f'compensation is missing, so temperature silently disables KD')


def test_masked_lm_logits_are_normalised_per_token_not_per_sequence():
    """A [B, T, V] tensor must give the same loss as its [B*T, V] flattening.

    `batchmean` divides by input.size(0) alone, so before the reshape an MLM
    batch was summed over all T token positions and divided by B -- the term was
    exactly seq_len times too large (measured: 128.0x at T=128). One lambda_kd
    shared across the vision and MLM paths would then mean two different things,
    and the MLM arm would be effectively running at a 128x larger KD weight.
    """
    torch.manual_seed(0)
    B, T, V = 4, 32, 50
    s3, t3 = torch.randn(B, T, V), torch.randn(B, T, V)

    got = float(kd_kl_loss(s3, t3, T=4.0))
    want = float(kd_kl_loss(s3.reshape(-1, V), t3.reshape(-1, V), T=4.0))
    assert got == pytest.approx(want, rel=1e-6), (
        f'3-D form gives {got:.6f}, flattened gives {want:.6f} -- ratio '
        f'{got / want:.1f}x (seq_len is {T})')


def test_uses_batchmean_not_mean():
    """`mean` divides by the number of logits, so the loss would scale with the
    class count and a lambda tuned on CIFAR-10 would be 10x off on CIFAR-100."""
    torch.manual_seed(0)
    import torch.nn.functional as F
    s, t = torch.randn(4, 7), torch.randn(4, 7)
    expected = F.kl_div(F.log_softmax(s / 3, -1), F.softmax(t / 3, -1),
                        reduction='batchmean') * 9.0
    assert float(kd_kl_loss(s, t, T=3.0)) == pytest.approx(float(expected), rel=1e-6)


def test_teacher_receives_no_gradient():
    s = torch.randn(8, 5, requires_grad=True)
    t = torch.randn(8, 5, requires_grad=True)
    kd_kl_loss(s, t, T=2.0).backward()
    assert t.grad is None, 'the teacher must be detached; it is frozen by construction'
    assert s.grad is not None


# --- feature_distill_loss --------------------------------------------------

def test_cosine_self_match_is_zero():
    z = torch.randn(16, 64)
    assert float(feature_distill_loss(z, z, 'cosine')) == pytest.approx(0.0, abs=1e-6)
    assert float(feature_distill_loss(z, z, 'mse')) == pytest.approx(0.0, abs=1e-6)


def test_cosine_is_scale_invariant_and_mse_is_not():
    """Why cosine is the default here.

    The student is pruned to 0.1% density while the teacher stays dense, so their
    feature norms differ by a large and drifting factor. Under MSE that norm gap
    dominates the loss and the rung would report a failure of *the metric* rather
    than of feature matching.
    """
    torch.manual_seed(0)
    z = torch.randn(16, 64)
    assert float(feature_distill_loss(z * 7.0, z, 'cosine')) == pytest.approx(0.0, abs=1e-5)
    assert float(feature_distill_loss(z * 7.0, z, 'mse')) > 1.0


def test_cosine_teacher_is_detached():
    s = torch.randn(8, 32, requires_grad=True)
    t = torch.randn(8, 32, requires_grad=True)
    feature_distill_loss(s, t, 'cosine').backward()
    assert t.grad is None
    assert s.grad is not None


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match='cosine'):
        feature_distill_loss(torch.randn(2, 3), torch.randn(2, 3), mode='l1')


# --- integration with the objective ---------------------------------------

def _combine(lambda_kd, kd, **lambdas):
    args = SimpleNamespace(lambda1=0.25, lambda2=0.25, lambda3=0.25, lambda4=0.25,
                           lambda_kd=lambda_kd, device='cpu', **lambdas)
    return BaCPTrainer._combine_losses(args, torch.tensor(2.0), torch.tensor(3.0),
                                       torch.tensor(5.0), torch.tensor(7.0), kd)


def test_lambda_kd_is_outside_the_four_lambda_simplex():
    """Turning distillation on must not rescale the contrastive terms.

    Structural fix for the flaw in the submitted paper's Table 6: zeroing one of
    four 0.25 weights while leaving the rest untouched made every "no X" arm both
    a removal and a 25% downweighting of the whole objective. Two interventions,
    one label -- and no downstream statistics can separate them again.
    """
    off = _combine(0.0, torch.tensor(11.0))
    on = _combine(0.5, torch.tensor(11.0))
    assert [float(x) for x in on[1:5]] == pytest.approx([float(x) for x in off[1:5]])
    assert float(on[0]) == pytest.approx(float(off[0]) + 11.0 * 0.5)


def test_distillation_off_by_default_costs_nothing():
    """With distill_mode='none' the helper must not run a teacher forward pass.

    If it did, every BaCP rung would silently carry the cost and the Stage C
    FLOP matching argument would be wrong in the direction that flatters BaCP.
    """
    calls = []

    class Teacher:
        def forward_all(self, x):
            calls.append(x)
            return torch.randn(2, 4), torch.randn(2, 3)

        def get_embeddings(self, f):
            return f

    trainer = SimpleNamespace(distill_mode='none', lambda_kd=0.0, device='cpu',
                              model_ft=Teacher(), kd_temperature=4.0,
                              feature_distill_metric='cosine')
    out = BaCPTrainer._distillation_loss(trainer, torch.randn(2, 4),
                                         torch.randn(2, 3), torch.randn(2, 1),
                                         torch.zeros(2, dtype=torch.long))
    assert float(out) == 0.0
    assert calls == [], 'a teacher forward ran even though distillation is off'


@pytest.mark.parametrize('mode', ['kl', 'feature', 'both'])
def test_distillation_modes_produce_a_positive_term(mode):
    torch.manual_seed(0)

    class Teacher:
        def forward_all(self, x):
            return torch.randn(8, 16), torch.randn(8, 5)

        def get_embeddings(self, f):
            return f

    trainer = SimpleNamespace(distill_mode=mode, lambda_kd=1.0, device='cpu',
                              model_ft=Teacher(), kd_temperature=4.0,
                              feature_distill_metric='cosine')
    out = BaCPTrainer._distillation_loss(trainer, torch.randn(8, 16),
                                         torch.randn(8, 5), torch.randn(8, 1),
                                         torch.zeros(8, dtype=torch.long))
    assert float(out) > 0.0
    assert math.isfinite(float(out))


def test_mse_and_cosine_are_the_same_metric_on_normalised_features():
    """They are not two metrics at BaCP's call site, and the paper must not
    present a cosine/MSE comparison as evidence about norm versus direction.

    Every branch of ClassificationAndEncoderNetwork.get_embeddings ends in
    F.normalize, so both arguments reaching feature_distill_loss are unit-norm --
    and on unit-norm vectors ||z_s - z_t||^2 = 2(1 - cos) exactly. The MSE branch
    is then (2/D) times the cosine branch: a monotone rescaling that changes only
    the effective lambda_kd. The distinction becomes real only if unnormalised
    features are passed in, which nothing currently does.
    """
    import torch.nn.functional as F
    torch.manual_seed(0)
    for D in (64, 128, 256):
        a = F.normalize(torch.randn(16, D), dim=-1)
        b = F.normalize(torch.randn(16, D), dim=-1)
        cos = float(feature_distill_loss(a, b, 'cosine'))
        mse = float(feature_distill_loss(a, b, 'mse'))
        assert mse == pytest.approx(2.0 / D * cos, rel=1e-5), (
            f'D={D}: mse {mse:.8f} != (2/{D}) x cosine {2.0 / D * cos:.8f}')


def test_distill_mode_both_differs_from_each_component():
    """The kl/feature/both parametrization above asserts only "> 0" for each, so
    'both' was never distinguished from 'kl' -- deleting 'both' from the feature
    branch left the whole suite green while silently turning 'both' into 'kl'."""
    torch.manual_seed(0)
    feats, logits = torch.randn(8, 16), torch.randn(8, 5)

    class Teacher:
        def forward_all(self, x):
            torch.manual_seed(1)
            return torch.randn(8, 16), torch.randn(8, 5)

        def get_embeddings(self, f):
            return f

    def loss(mode):
        trainer = SimpleNamespace(distill_mode=mode, lambda_kd=1.0, device='cpu',
                                  model_ft=Teacher(), kd_temperature=4.0,
                                  feature_distill_metric='cosine')
        return float(BaCPTrainer._distillation_loss(
            trainer, feats, logits, torch.randn(8, 1),
            torch.zeros(8, dtype=torch.long)))

    kl, feat, both = loss('kl'), loss('feature'), loss('both')
    assert both == pytest.approx(kl + feat, rel=1e-5), (
        f"'both' ({both:.4f}) must be kl ({kl:.4f}) + feature ({feat:.4f})")
    assert both > kl and both > feat, "'both' collapsed to one component"
