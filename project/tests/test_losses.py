"""L1 -- pure loss math. Imports only loss_functions, bacp and torch.

Every assertion here is either an analytic fixed point or a comparison against
a short reference implementation in this file. Nothing depends on a golden file
or on a lucky seed, so these tests stay meaningful when the losses are rewritten.

Reference: Khosla et al. 2020 (SupCon, NeurIPS 33); Chen et al. 2020 (SimCLR,
ICML); Xu et al. 2022 (CAP, AAAI 36) for the cross-model formulation these
losses are standing in for.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from bacp import BaCPTrainer
from loss_functions import NTXentLoss, SupConLoss

TAU = 0.5
DIM = 8


def _unit(batch, dim=DIM, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.nn.functional.normalize(torch.randn(batch, dim, generator=g), dim=1)


def _reference_supcon(z1, z2, labels, tau, eps=1e-8):
    """Loop-form SupCon matching the shipped semantics exactly.

    Deliberately mirrors two shipped choices rather than the paper:
      * the denominator runs over all non-self entries (Khosla's L_out), and
      * an anchor with no positives is divided by clamp(min=1) and so
        contributes 0 to the mean instead of being excluded.
    The second only matters if such an anchor can exist -- see
    test_supcon_empty_positive_branch_is_unreachable.
    """
    z = torch.cat([z1, z2], dim=0)
    lab = labels.view(-1).repeat(2)
    n = z.shape[0]
    sims = (z @ z.T) / tau

    per_anchor = []
    for i in range(n):
        row = sims[i]
        shifted = row - row.max()
        denom = sum(math.exp(shifted[k]) for k in range(n) if k != i)
        positives = [j for j in range(n) if j != i and lab[j] == lab[i]]
        total = sum(shifted[j].item() - math.log(denom + eps) for j in positives)
        per_anchor.append(-total / max(len(positives), 1))
    return sum(per_anchor) / n


# --- NT-Xent -------------------------------------------------------------

def test_ntxent_single_pair_is_exactly_zero():
    """With N=1 the loss is exactly 0, regardless of the embeddings.

    After the -inf diagonal mask each row has exactly one live logit, so the
    softmax over it is 1 and the cross-entropy is 0. Pins the diagonal masking:
    if the mask were dropped, the self-similarity would compete and the loss
    would be strictly positive.
    """
    loss = NTXentLoss(TAU, "cpu")(_unit(1, seed=1), _unit(1, seed=2))
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("n", [2, 4, 8])
def test_ntxent_zero_embeddings_is_log_2n_minus_1(n):
    """All-zero embeddings give exactly log(2N-1).

    Every off-diagonal logit is 0, so each row is uniform over its 2N-1 live
    entries and -log(1/(2N-1)) = log(2N-1).
    """
    z = torch.zeros(n, DIM)
    loss = NTXentLoss(TAU, "cpu")(z, z.clone())
    assert loss.item() == pytest.approx(math.log(2 * n - 1), abs=1e-5)


def test_ntxent_symmetric_in_arguments():
    """Swapping the arguments permutes the problem without changing it."""
    z1, z2 = _unit(6, seed=3), _unit(6, seed=4)
    criterion = NTXentLoss(TAU, "cpu")
    assert criterion(z1, z2).item() == pytest.approx(criterion(z2, z1).item(), abs=1e-6)


def test_ntxent_aligned_beats_independent():
    z = _unit(8, seed=5)
    criterion = NTXentLoss(TAU, "cpu")
    assert criterion(z, z.clone()).item() < criterion(z, _unit(8, seed=6)).item()


def test_ntxent_lower_temperature_sharpens_aligned_pairs():
    z = _unit(8, seed=7)
    cold = NTXentLoss(0.05, "cpu")(z, z.clone()).item()
    warm = NTXentLoss(1.0, "cpu")(z, z.clone()).item()
    assert cold < warm


# --- SupCon --------------------------------------------------------------

@pytest.mark.parametrize("n", [2, 4, 8])
def test_supcon_zero_embeddings_is_log_2n_minus_1(n):
    """Same analytic fixed point as NT-Xent, reached by a different path.

    Useful precisely because the two implementations normalise differently:
    at z=0 they must still agree.
    """
    z = torch.zeros(n, DIM)
    labels = torch.arange(n)
    loss = SupConLoss(TAU, TAU, "cpu")(z, z.clone(), labels)
    assert loss.item() == pytest.approx(math.log(2 * n - 1), abs=1e-5)


def test_supcon_matches_reference_implementation():
    z1, z2 = _unit(6, seed=8), _unit(6, seed=9)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    got = SupConLoss(TAU, TAU, "cpu")(z1, z2, labels).item()
    assert got == pytest.approx(_reference_supcon(z1, z2, labels, TAU), abs=1e-4)


def test_supcon_all_distinct_labels_has_one_positive_per_row():
    """Even with all-distinct labels every anchor has exactly one positive.

    ``labels.repeat(n_views, 1)`` makes anchor i and anchor i+B share a label by
    construction, so |P(i)| >= 1 always. This is why NT-Xent's single positive is
    a strict *subset* of SupCon's -- SupCon reduces to NT-Xent at |P(i)| = 1
    (Khosla et al. 2020, sec. 3).
    """
    n = 5
    z = _unit(n, seed=10)
    labels = torch.arange(n)
    lab = labels.view(-1, 1).repeat(2, 1)
    mask_pos = torch.eq(lab, lab.T).float() - torch.eye(2 * n)
    assert torch.equal(mask_pos.sum(1), torch.ones(2 * n))


def test_supcon_all_same_label_has_2n_minus_1_positives():
    n = 5
    labels = torch.zeros(n, dtype=torch.long)
    lab = labels.view(-1, 1).repeat(2, 1)
    mask_pos = torch.eq(lab, lab.T).float() - torch.eye(2 * n)
    assert torch.equal(mask_pos.sum(1), torch.full((2 * n,), 2.0 * n - 1))


def test_supcon_empty_positive_branch_is_unreachable():
    """``num_pos.clamp(min=1.0)`` cannot currently trigger.

    Records a real finding: because labels are repeated across views, no anchor
    can ever have an empty positive set, so the clamp is dead code and the
    "empty positives dilute the mean" concern does NOT apply to the shipped
    loss. It becomes live only once candidates come from a memory bank that may
    not contain a given label -- which is why the replacement loss needs an
    explicit exclude-rather-than-dilute guard.
    """
    for n in (1, 3, 7):
        for labels in (torch.arange(n), torch.zeros(n, dtype=torch.long)):
            lab = labels.view(-1, 1).repeat(2, 1)
            mask_pos = torch.eq(lab, lab.T).float() - torch.eye(2 * n)
            assert (mask_pos.sum(1) >= 1).all(), (
                "an anchor with no positives is reachable after all -- the "
                "clamp(min=1) dilution is then a live bug, not dead code"
            )


def test_supcon_base_temp_is_currently_unused():
    """``base_temp`` is stored and never applied.

    Khosla et al. scale the loss by (temp / base_temp); this implementation
    drops that factor. Harmless today only because BaCP constructs
    ``SupConLoss(tau, tau, device)``, making the ratio exactly 1.

    This test guards the legacy path. When the loss is replaced by the CAP
    formulation, decide the scaling explicitly there and drop the dead argument
    rather than "fixing" it here -- changing it rescales every contrastive term.
    """
    z1, z2 = _unit(4, seed=11), _unit(4, seed=12)
    labels = torch.tensor([0, 0, 1, 1])
    same = SupConLoss(TAU, TAU, "cpu")(z1, z2, labels).item()
    wildly_different = SupConLoss(TAU, 999.0, "cpu")(z1, z2, labels).item()
    assert same == pytest.approx(wildly_different, abs=1e-9)


# --- Gradient routing ----------------------------------------------------

@pytest.mark.parametrize("which", ["supcon", "ntxent"])
def test_grad_flows_to_first_arg_only(which):
    """Gradient reaches the student embedding, never the frozen teacher.

    Mirrors the production call: the teacher forward runs under torch.no_grad,
    so its embeddings arrive detached. If a rewrite accidentally made the
    teacher trainable, this is where it shows up.
    """
    z1 = _unit(4, seed=13).clone().requires_grad_(True)
    z2 = _unit(4, seed=14).clone().detach()
    labels = torch.tensor([0, 0, 1, 1])

    if which == "supcon":
        loss = SupConLoss(TAU, TAU, "cpu")(z1, z2, labels)
    else:
        loss = NTXentLoss(TAU, "cpu")(z1, z2)
    loss.backward()

    assert z1.grad is not None and torch.isfinite(z1.grad).all()
    assert z1.grad.abs().sum() > 0
    assert z2.grad is None


# --- Loss combination ----------------------------------------------------

def _combine(l1, l2, l3, l4, prc, snc, fic, ce, kd=None, lambda_kd=0.0):
    """Call the unbound method against a stand-in carrying just the weights."""
    args = SimpleNamespace(lambda1=l1, lambda2=l2, lambda3=l3, lambda4=l4,
                           lambda_kd=lambda_kd, device="cpu")
    return BaCPTrainer._combine_losses(args, prc, snc, fic, ce, kd)


def test_combine_losses_is_weighted_sum():
    total, prc, snc, fic, ce, kd = _combine(0.1, 0.2, 0.3, 0.4, 2.0, 3.0, 5.0, 7.0)
    assert prc == pytest.approx(2.0 * 0.1)
    assert fic == pytest.approx(5.0 * 0.2)
    assert snc == pytest.approx(3.0 * 0.3)
    assert ce == pytest.approx(7.0 * 0.4)
    assert kd == pytest.approx(0.0)
    assert total == pytest.approx(prc + snc + fic + ce)


def test_distillation_weight_is_outside_the_four_lambda_simplex():
    """lambda_kd must not rescale the contrastive terms.

    This is the structural fix for the flaw in the submitted paper's Table 6,
    where zeroing one of four 0.25 weights and leaving the other three alone made
    every "no X" arm simultaneously a removal and a 25% downweighting of the
    whole objective -- two interventions under one label. Here, turning
    distillation on changes exactly one thing.
    """
    without = _combine(0.25, 0.25, 0.25, 0.25, 2.0, 3.0, 5.0, 7.0)
    with_kd = _combine(0.25, 0.25, 0.25, 0.25, 2.0, 3.0, 5.0, 7.0,
                       kd=torch.tensor(11.0), lambda_kd=0.5)
    # The four weighted components are untouched...
    assert with_kd[1:5] == pytest.approx(without[1:5])
    # ...and the total moves by exactly the new term.
    assert float(with_kd[0]) == pytest.approx(float(without[0]) + 11.0 * 0.5)


def test_combine_losses_defaults_kd_to_zero_when_absent():
    """A five-argument call from older code must still sum to the same total."""
    with_none = _combine(0.25, 0.25, 0.25, 0.25, 2.0, 3.0, 5.0, 7.0, kd=None)
    assert float(with_none[5]) == pytest.approx(0.0)
    assert float(with_none[0]) == pytest.approx(0.25 * (2.0 + 3.0 + 5.0 + 7.0))


@pytest.mark.parametrize("index,expected_module,expected_value", [
    (1, "PrC", 2.0),
    (2, "FiC", 5.0),
    (3, "SnC", 3.0),
    (4, "CE", 7.0),
])
def test_lambda_to_loss_mapping(index, expected_module, expected_value):
    """Pins the true mapping: lambda1=PrC, lambda2=FiC, lambda3=SnC, lambda4=CE.

    ``utils.print_dynamic_lambdas_statistics`` labels its four series
    ['CE lambda', 'PrC lambda', 'SnC lambda', 'FiC lambda'], which is a
    different permutation -- so every published dynamic-lambda figure legend is
    mislabelled. Fix the plotting function, not this test.

    Note the argument order (prc, snc, fic, ce) differs from the return order
    (total, prc, snc, fic, ce); that mismatch is itself worth pinning.
    """
    lambdas = [1.0 if i == index else 0.0 for i in (1, 2, 3, 4)]
    total, *_ = _combine(*lambdas, prc=2.0, snc=3.0, fic=5.0, ce=7.0)
    assert total == pytest.approx(expected_value), (
        f"lambda{index} should scale {expected_module}"
    )
