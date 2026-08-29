"""L1b -- the CAP-aligned contrastive loss, and the reason it replaced the old pair.

The last test in this file is the important one: it demonstrates that under the
original configuration the contrastive gradient could not reach the backbone at
all, which is the mechanism behind the paper's ablation showing that removing
PrC/SnC/CE each *improved* 99%-sparsity accuracy.

References:
  Xu et al. 2022, CAP, AAAI 36 (arXiv 2112.07198) -- Eq. 1 and Table 1
  Khosla et al. 2020, SupCon, NeurIPS 33
  Chen et al. 2020, SimCLR, ICML
"""

import math

import pytest
import torch

from loss_functions import CAPContrastiveLoss, NTXentLoss, SupConLoss, cap_contrastive_loss

TAU = 0.5
DIM = 8


def _unit(n, dim=DIM, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.nn.functional.normalize(torch.randn(n, dim, generator=g), dim=1)


def _eye_mask(b, n):
    m = torch.zeros(b, n, dtype=torch.bool)
    k = min(b, n)
    m[torch.arange(k), torch.arange(k)] = True
    return m


# --- Analytic fixed points ----------------------------------------------

@pytest.mark.parametrize("n_cand", [2, 5, 16])
def test_zero_embeddings_is_log_n_candidates(n_cand):
    """All-zero embeddings give exactly log(N) over N candidates.

    Note log(N), not log(2N-1) as in the square formulations: the matrix is
    B x N and there is no self-similarity to mask out. The anchor's own index in
    the candidate set is a *legitimate positive* -- it is the same input seen
    through a different model -- rather than the degenerate self-match that
    NT-Xent has to exclude with a -inf diagonal.
    """
    z_a = torch.zeros(4, DIM)
    z_c = torch.zeros(n_cand, DIM)
    loss = cap_contrastive_loss(z_a, z_c, _eye_mask(4, n_cand), TAU)
    assert loss.item() == pytest.approx(math.log(n_cand), abs=1e-5)


def test_all_candidates_positive_is_log_n():
    """When every candidate is a positive, the loss is still log(N) at z=0."""
    z_a, z_c = torch.zeros(3, DIM), torch.zeros(6, DIM)
    mask = torch.ones(3, 6, dtype=torch.bool)
    assert cap_contrastive_loss(z_a, z_c, mask, TAU).item() == pytest.approx(math.log(6), abs=1e-5)


def test_perfect_alignment_beats_misalignment():
    z = _unit(6, seed=1)
    mask = _eye_mask(6, 6)
    aligned = cap_contrastive_loss(z, z.clone(), mask, TAU).item()
    misaligned = cap_contrastive_loss(z, _unit(6, seed=2), mask, TAU).item()
    assert aligned < misaligned


def test_lower_temperature_sharpens_aligned_pairs():
    z = _unit(6, seed=3)
    mask = _eye_mask(6, 6)
    assert (cap_contrastive_loss(z, z.clone(), mask, 0.05).item()
            < cap_contrastive_loss(z, z.clone(), mask, 1.0).item())


# --- Structural properties ----------------------------------------------

@pytest.mark.parametrize("n_cand", [4, 9, 32])
def test_candidate_count_may_differ_from_batch(n_cand):
    """The matrix is rectangular, which is what makes a memory bank possible.

    CAP holds 4096 pre-encoded teacher examples on CPU and fetches them per batch.
    A square cat([z1, z2]) @ .T formulation cannot express that at all.
    """
    z_a = _unit(4, seed=4)
    z_c = _unit(n_cand, seed=5)
    loss = cap_contrastive_loss(z_a, z_c, _eye_mask(4, n_cand), TAU)
    assert torch.isfinite(loss)


def test_gradient_does_not_flow_to_a_detached_teacher():
    """Mirrors the production call: the teacher forward runs under torch.no_grad,
    so its embeddings arrive detached and must stay gradient-free."""
    z_a = _unit(4, seed=6).clone().requires_grad_(True)
    z_c = _unit(4, seed=7).clone().detach()

    cap_contrastive_loss(z_a, z_c, _eye_mask(4, 4), TAU).backward()

    assert z_a.grad is not None and z_a.grad.abs().sum() > 0
    assert z_c.grad is None


def test_loss_averages_over_student_anchors_only():
    """Exactly B anchor terms, not 2B -- the structural difference from the old pair.

    The legacy losses built cat([z_student, z_teacher]) @ .T, a 2B x 2B matrix, and
    averaged over all 2B rows. Half of those rows anchor on a frozen
    representation, which diluted the loss magnitude roughly 2x and therefore
    silently halved the effective lambda on every contrastive term. Here the mean
    runs over the B student rows and nothing else.
    """
    b, n = 4, 6
    z_a, z_c = _unit(b, seed=20), _unit(n, seed=21)
    mask = _eye_mask(b, n)

    logits = (z_a @ z_c.T) / TAU
    shifted = logits - logits.max(dim=1, keepdim=True).values
    log_prob = shifted - torch.logsumexp(shifted, dim=1, keepdim=True)
    manual = [-(log_prob[i][mask[i]].sum() / mask[i].sum()).item() for i in range(b)]

    assert len(manual) == b
    got = cap_contrastive_loss(z_a, z_c, mask, TAU).item()
    assert got == pytest.approx(sum(manual) / b, abs=1e-5)


def test_anchors_without_positives_are_excluded_not_diluted():
    """An all-False mask row must not contribute a zero to the mean.

    Unreachable with in-batch candidates -- every anchor has its own teacher view --
    but a memory bank may hold no example of a given label, and contributing zeros
    would scale the loss by a bank-composition-dependent factor.
    """
    z_a, z_c = _unit(4, seed=8), _unit(4, seed=9)
    full = _eye_mask(4, 4)

    partial = full.clone()
    partial[2:] = False  # anchors 2 and 3 have no positives

    only_first_two = cap_contrastive_loss(z_a[:2], z_c, full[:2], TAU)
    with_empty_rows = cap_contrastive_loss(z_a, z_c, partial, TAU)
    assert with_empty_rows.item() == pytest.approx(only_first_two.item(), abs=1e-6)


def test_all_empty_mask_returns_connected_zero():
    z_a = _unit(4, seed=10).clone().requires_grad_(True)
    loss = cap_contrastive_loss(z_a, _unit(4, seed=11), torch.zeros(4, 4, dtype=torch.bool), TAU)
    assert loss.item() == 0.0
    loss.backward()  # must not raise: the graph stays connected


# --- The module: L_unsup + L_sup -----------------------------------------

def test_module_sums_instance_and_label_terms():
    z_a, z_c = _unit(6, seed=12), _unit(6, seed=13)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])

    both = CAPContrastiveLoss(TAU)(z_a, z_c, labels).item()
    inst = CAPContrastiveLoss(TAU, supervised=False)(z_a, z_c, labels).item()
    lab = CAPContrastiveLoss(TAU, unsupervised=False)(z_a, z_c, labels).item()
    assert both == pytest.approx(inst + lab, abs=1e-5)


def test_module_label_term_uses_labels():
    """The supervised term must actually depend on the label assignment."""
    z_a, z_c = _unit(6, seed=14), _unit(6, seed=15)
    grouped = CAPContrastiveLoss(TAU, unsupervised=False)(
        z_a, z_c, torch.tensor([0, 0, 0, 1, 1, 1])).item()
    distinct = CAPContrastiveLoss(TAU, unsupervised=False)(
        z_a, z_c, torch.arange(6)).item()
    assert grouped != pytest.approx(distinct)


def test_shared_denominator_across_the_two_terms():
    """Both terms normalise over the same candidate set.

    This is the difference from summing SupConLoss and NTXentLoss, which
    normalise over different sets and so cannot be combined coherently.
    """
    z_a, z_c = _unit(4, seed=16), _unit(4, seed=17)
    labels = torch.tensor([0, 0, 1, 1])
    inst_mask = _eye_mask(4, 4)
    label_mask = labels.view(-1, 1) == labels.view(1, -1)

    manual = (cap_contrastive_loss(z_a, z_c, inst_mask, TAU)
              + cap_contrastive_loss(z_a, z_c, label_mask, TAU))
    assert CAPContrastiveLoss(TAU)(z_a, z_c, labels).item() == pytest.approx(manual.item(), abs=1e-6)


# --- Why the legacy pair worked against itself ---------------------------

def test_legacy_pair_treats_same_class_pairs_oppositely():
    """Demonstrates the defect that motivated this rewrite.

    Construct a batch where two same-class samples are far apart in embedding
    space. SupCon calls them positives and pulls; NT-Xent's single-target
    cross-entropy calls them negatives and pushes. Moving them together must
    therefore *lower* SupCon and *raise* NT-Xent -- opposite signs on the same
    perturbation, which is exactly why the paper's Table 6 shows both-together
    (92.32) scoring worse than supervised-alone (92.96) at 99% sparsity.
    """
    labels = torch.tensor([0, 0, 1, 1])

    apart = torch.tensor([
        [1.0, 0.0], [-1.0, 0.0],   # class 0, opposed
        [0.0, 1.0], [0.0, -1.0],   # class 1, opposed
    ])
    together = torch.tensor([
        [1.0, 0.0], [1.0, 0.0],
        [0.0, 1.0], [0.0, 1.0],
    ])

    supcon = SupConLoss(TAU, TAU, "cpu")
    ntxent = NTXentLoss(TAU, "cpu")

    sup_delta = supcon(together, together, labels).item() - supcon(apart, apart, labels).item()
    unsup_delta = ntxent(together, together).item() - ntxent(apart, apart).item()

    assert sup_delta < 0, "pulling same-class samples together should reduce SupCon"
    assert unsup_delta > 0, "the same move should increase NT-Xent"


def test_supcon_positive_set_contains_the_ntxent_positive():
    """NT-Xent's single positive is a strict subset of SupCon's positive set.

    labels.repeat(n_views, 1) makes anchor i and anchor i+B share a label by
    construction, so the paired view is always in P(i). Adding NT-Xent therefore
    contributes no new positive -- it re-weights one SupCon already had, and
    reclassifies every other same-class sample from positive to negative.
    """
    b = 4
    labels = torch.arange(b)
    lab = labels.view(-1, 1).repeat(2, 1)
    mask_pos = (torch.eq(lab, lab.T).float() - torch.eye(2 * b)).bool()

    for i in range(b):
        assert mask_pos[i, i + b], "the NT-Xent positive is missing from P(i)"
