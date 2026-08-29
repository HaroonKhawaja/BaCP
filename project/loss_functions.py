import torch
import torch.nn as nn


def cap_contrastive_loss(z_anchor, z_cand, pos_mask, tau, eps=1e-8):
    """Cross-model contrastive loss, CAP Eq. 1 (Xu et al. 2022, AAAI 36).

        L_i = -1/|P(i)| * sum_{j in P(i)} log( exp(sim(z_i, zc_j)/tau)
                                               / sum_k exp(sim(z_i, zc_k)/tau) )

    Args:
        z_anchor: [B, D] L2-normalized student embeddings (require grad).
        z_cand:   [N, D] L2-normalized candidate embeddings from the frozen
                  teacher (in-batch, optionally plus a memory bank). N need not
                  equal B.
        pos_mask: [B, N] bool. Which candidates are positives for each anchor.
        tau:      temperature.

    Two structural properties, both deliberate and both differing from the
    SupConLoss/NTXentLoss pair this replaces:

    RECTANGULAR, NOT SQUARE. The similarity matrix is B x N -- anchors from the
    student, candidates from the teacher. The previous implementations built
    `z = cat([z_student, z_teacher])` and took `z @ z.T`, a 2B x 2B matrix with
    four blocks. Only the student-teacher block appears in CAP or in BaCP's own
    written equations; the other three were unintended. Their effects were:
    student-student added a SimCLR term nobody specified, teacher-teacher
    contributed constants with no gradient, and teacher-anchored rows diluted the
    mean over 2B anchors by ~2x, silently halving the effective lambda on every
    contrastive term.

    ONE FUNCTIONAL, CALLED TWICE. CAP composes the unsupervised and supervised
    terms as the same loss under two definitions of P(i), and because the
    denominator is a function of the logits alone, both calls normalize over
    the identical candidate set. The material difference from the legacy
    SupCon+NTXent sum is the anchor set, not the normalizer: the legacy pair
    anchors on all 2B rows of a square 2Bx2B matrix, so each contrastive
    term's mean runs over twice the anchors -- silently halving the effective
    lambda on every contrastive term relative to the equations as written.

    Anchors with no positives are EXCLUDED, not divided by a clamped 1. With
    in-batch candidates every anchor has a positive, so this cannot currently
    trigger -- but a memory bank may hold no example of a given label, and
    contributing a zero to the mean would then dilute the loss by a
    bank-composition-dependent factor.
    """
    logits = (z_anchor @ z_cand.T) / tau

    # Row-wise max subtraction for numerical stability. Detached: it is a constant
    # shift that cancels in the log-softmax, and must not carry gradient.
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

    pos_mask = pos_mask.to(log_prob.dtype)
    n_pos = pos_mask.sum(dim=1)
    has_pos = n_pos > 0
    if not bool(has_pos.any()):
        return z_anchor.sum() * 0.0  # keeps the graph connected

    per_anchor = -(log_prob * pos_mask).sum(dim=1) / n_pos.clamp(min=eps)
    return per_anchor[has_pos].mean()


class CAPContrastiveLoss(nn.Module):
    """PrC / FiC / SnC as CAP defines them: L_unsup + L_sup over a shared denominator.

    Positive sets follow CAP Table 1:
        unsupervised -> the same instance seen through the other model
        supervised   -> every candidate carrying the same label

    Reference: Xu et al. 2022, "From Dense to Sparse: Contrastive Pruning for
    Better Pre-trained Language Model Compression", AAAI 36, arXiv 2112.07198.
    """

    def __init__(self, tau, supervised=True, unsupervised=True):
        super().__init__()
        self.tau = tau
        self.supervised = supervised
        self.unsupervised = unsupervised

    def forward(self, z_anchor, z_cand, labels, cand_labels=None):
        if cand_labels is None:
            # In-batch candidates: row i of z_cand is the same sample as row i of
            # z_anchor, seen through the teacher.
            cand_labels = labels

        loss = z_anchor.sum() * 0.0

        if self.unsupervised:
            # P(i) = {the teacher's view of sample i}
            inst = torch.zeros(z_anchor.shape[0], z_cand.shape[0],
                               dtype=torch.bool, device=z_anchor.device)
            n = min(z_anchor.shape[0], z_cand.shape[0])
            inst[torch.arange(n), torch.arange(n)] = True
            loss = loss + cap_contrastive_loss(z_anchor, z_cand, inst, self.tau)

        if self.supervised:
            # P(i) = {teacher views of every sample sharing anchor i's label}
            label_mask = labels.view(-1, 1) == cand_labels.view(1, -1)
            loss = loss + cap_contrastive_loss(z_anchor, z_cand, label_mask, self.tau)

        return loss


class SupConLoss(nn.Module):
    """Supervised contrastive loss, L_out form (Khosla et al. 2020, Eq. 2).

    Reference: Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020,
    arXiv 2004.11362. P(i) = every other view in the batch sharing anchor i's
    label; denominator over A(i) = all-except-self; positives weighted
    1/|P(i)|; the mean over log-probabilities sits OUTSIDE the log (their
    L_out, which the paper argues outperforms L_in).

    Known deviations from their Eq. 2, deliberate and pinned by tests
    (test_losses.py):
      - the (temp / base_temp) scale factor is dropped; base_temp is stored
        and unused (test_supcon_base_temp_is_currently_unused);
      - an anchor with no positive is DILUTED via clamp(min=1.0) rather than
        excluded (the CAP path above excludes instead);
      - an eps inside log(denom + eps), absent from the paper.

    Kept for `contrastive_mode='legacy'`; the CAP path above is the default.
    """

    def __init__(self, temp, base_temp, device, n_views=2, eps=1e-8):
        super(SupConLoss, self).__init__()
        self.temp = temp
        self.base_temp = base_temp
        self.device = device
        self.n_views = n_views
        self.eps = eps
    
    def forward(self, z1, z2, labels):
        z = torch.cat([z1, z2], dim=0)
        N_total = z.shape[0]

        labels = labels.view(-1, 1).repeat(self.n_views, 1)

        mask_pos = torch.eq(labels, labels.T).float() 
        diag_mask = torch.eye(N_total, device=z.device)
        mask_pos = mask_pos - diag_mask
        neg_mask = 1 - diag_mask

        logits = torch.matmul(z, z.T) / self.temp

        num_pos = torch.sum(mask_pos, dim=1).clamp(min=1.0)

        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max

        denom = torch.sum(torch.exp(logits) * neg_mask, dim=1, keepdim=True)

        log_prob = (logits - torch.log(denom + self.eps)) * mask_pos
        mean_log_prob = torch.sum(log_prob, dim=1) / num_pos
        loss = torch.mean(-mean_log_prob)
        return loss
    
class NTXentLoss(nn.Module):
    """NT-Xent, the SimCLR objective (Chen et al. 2020, Eq. 1).

    Reference: Chen, Kornblith, Norouzi & Hinton, "A Simple Framework for
    Contrastive Learning of Visual Representations", ICML 2020,
    arXiv 2002.05709. For 2B views the positive of view i is its augmented
    counterpart i+B (mod 2B); every other non-self view is a negative, and
    self-similarity is masked to -inf before the softmax, matching their
    1[k != i] indicator.

    Kept for `contrastive_mode='legacy'`; the CAP path above is the default.
    """

    def __init__(self, temp, device, n_views=2, eps=1e-8):
        super(NTXentLoss, self).__init__()
        self.temp = temp
        self.device = device
        self.n_views = n_views
        self.eps = eps

    def forward(self, z1, z2):
        z = torch.cat([z1, z2], dim=0)
        N_total = z.shape[0]
        N = N_total // self.n_views

        mask = torch.eye(N_total, device=z.device).bool()

        logits = torch.matmul(z, z.T) / self.temp

        logits.masked_fill_(mask, float('-inf'))

        targets = torch.arange(N, device=z.device)
        targets = torch.cat([targets + N, targets], dim=0)

        loss = nn.CrossEntropyLoss()(logits, targets)
        return loss





# --- distillation controls -------------------------------------------------
#
# These exist to answer the objection that sinks the contrastive claim if it is
# left unanswered: BaCP puts three frozen dense teachers on one side of every
# comparison and zero on the other, so the cheapest explanation of any gain is
# "a dense teacher helps" -- nothing about contrast. Rung C1 injects the teacher
# at the logits, C2 at the features, C3 swaps in the contrastive form against the
# same teacher. Only C2 -> C3 isolates "contrastive" from "distillation", and it
# does so only because C1/C2/C3 hold the teacher fixed.
#
# CITATION DISCIPLINE, checked against the primary sources:
#
#   kd_kl_loss  -- Hinton, Vinyals & Dean (2015), arXiv 1503.02531, is the right
#       source for the soft-target idea and for the T^2 rescaling. It is NOT a
#       source for the KL form or its direction: the paper defines the objective
#       as cross-entropy with soft targets and never writes a KL divergence in
#       it at all ('Kullback' and 'relative entropy' appear zero times; 'KL'
#       appears only in Sec. 5.4, on ensembles). Cite implementations, not
#       Hinton, for `F.kl_div(log_softmax(s/T), softmax(t/T))`.
#
#   feature_distill_loss -- do NOT cite FitNets (Romero et al. 2015) for this.
#       FitNets regresses INTERMEDIATE activations through a LEARNED regressor
#       under a squared Euclidean loss in unnormalised space, and its second
#       stage operates on softened logits. It contains no cosine matching and no
#       discussion of teacher/student norm mismatch, so it supports neither the
#       metric nor the justification. Nor do the usual alternatives: SP (Tung &
#       Mori 2019) and RKD (Park et al. 2019) are RELATIONAL -- they match
#       sample-to-sample structure within a batch, not student feature to teacher
#       feature -- and PKT builds a cosine-kernel distribution matched by KL.
#       Cosine-on-penultimate-features here is BaCP's own design choice and the
#       paper should say so rather than attach a source that does not carry it.


def kd_kl_loss(student_logits, teacher_logits, T=4.0):
    r"""Hinton-style response distillation: T^2 * KL(teacher_T || student_T).

    .. math::
        \mathcal{L}_{KD} = T^2 \, \mathrm{KL}\!\left(
            \sigma(z_t/T) \,\|\, \sigma(z_s/T) \right)

    The T^2 factor is not cosmetic. Hinton et al. (2015), end of Sec. 2, note that
    the soft-target gradients scale as 1/T^2 and multiply by T^2 so that the
    relative contribution of the soft and hard targets stays roughly unchanged as
    T varies. Without it, raising the temperature quietly turns the term off --
    which would surface in the results as "KD does nothing", indistinguishable from
    a real null. Note the 1/T^2 relation is a HIGH-TEMPERATURE approximation (the
    exact gradient is (1/T)(q_i - p_i), Eq. 2); T^2 is a balancing heuristic, not
    an exact normaliser, and the paper should not claim otherwise.

    Direction: `F.kl_div(log_softmax(student/T), softmax(teacher/T))` computes
    KL(teacher || student) -- forward KL, teacher as the reference. That is the
    standard convention across the reference implementations (RepDistiller,
    mmrazor, the PyTorch tutorial), and it is gradient-identical to Hinton's
    cross-entropy-with-soft-targets because H(p_t, q_s) = KL(p_t || q_s) + H(p_t)
    and H(p_t) is constant in the student's parameters.

    `batchmean` and not `mean`: PyTorch's DEFAULT for `F.kl_div` is `mean`, which
    divides by numel (batch x classes) and which PyTorch's own documentation notes
    does not return the true KL. The two differ by exactly the class count -- 10
    on CIFAR-10, 100 on CIFAR-100 -- which composes multiplicatively with T^2 and
    with lambda_kd, silently detuning the KD/CE balance across datasets. Every
    reference implementation uses batchmean or hand-rolls it.
    """
    import torch.nn.functional as F

    # Flatten a 3-D masked-LM tensor to [B*T, V] FIRST. `batchmean` divides by
    # input.size(0) only, so on the MLM path a [B, T, V] tensor was summed over
    # all T token positions and divided by B alone -- the returned term was
    # exactly seq_len times the per-example quantity (measured: 128.0x at
    # T=128). One shared lambda_kd across the vision and MLM paths would then
    # mean two different things. _task_loss and _task_accuracy already branch on
    # dim()==3; this had not.
    if student_logits.dim() == 3:
        student_logits = student_logits.reshape(-1, student_logits.size(-1))
        teacher_logits = teacher_logits.reshape(-1, teacher_logits.size(-1))

    s = F.log_softmax(student_logits / T, dim=-1)
    t = F.softmax(teacher_logits.detach() / T, dim=-1)
    return F.kl_div(s, t, reduction='batchmean') * (T * T)


def feature_distill_loss(z_student, z_teacher, mode='cosine'):
    r"""Feature-level matching between student and teacher embeddings.

    'cosine' -- :math:`1 - \cos(z_s, z_t)`, averaged over the batch. Scale
        invariant, which is the property that matters when a 0.1%-density student
        is regressed onto a dense teacher.
    'mse'    -- squared Euclidean, the loss FitNets uses (though FitNets applies
        it to intermediate activations through a learned regressor, not to
        penultimate features -- see the citation note at the top of this section).

    **The two are not independent metrics at BaCP's call site.** Every branch of
    ``ClassificationAndEncoderNetwork.get_embeddings`` ends in ``F.normalize``,
    so both arguments arriving from ``_distillation_loss`` are unit-norm, and on
    unit-norm inputs :math:`\lVert z_s - z_t \rVert^2 = 2(1 - \cos)` exactly --
    the MSE branch is then :math:`(2/D)` times the cosine branch, a monotone
    rescaling that changes only the effective ``lambda_kd``. Choosing between
    them here is a learning-rate decision, not a metric comparison, and the
    paper must not present a cosine/MSE ablation as evidence about norm versus
    direction. The distinction becomes real only if unnormalised features are
    passed in, which nothing currently does.

    The deliberate parallel with the contrastive rung: cosine feature matching is
    the *positive-pair* term of an InfoNCE objective with the denominator
    removed, so C2 -> C3 isolates one thing -- the presence of negatives. That is
    the sharpest available statement of what "contrastive" contributes over
    "distillation", and as far as the citation search went it is NOT an argument
    anyone has published in this form. Treat it as this paper's framing and state
    it as such; the nearest published relative is the alignment/uniformity
    decomposition of InfoNCE, which is a different decomposition of a different
    object.
    """
    import torch.nn.functional as F
    z_teacher = z_teacher.detach()
    if mode == 'cosine':
        return (1.0 - F.cosine_similarity(z_student, z_teacher, dim=-1)).mean()
    if mode == 'mse':
        return F.mse_loss(z_student, z_teacher)
    raise ValueError(f"feature_distill_loss mode must be 'cosine' or 'mse', got {mode!r}")
