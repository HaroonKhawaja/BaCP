"""L4b -- WANDA pruner.

The point of these tests is narrow and specific: prove that WANDA is not
magnitude pruning wearing a different label. That is the failure mode that
matters, because the submitted paper reports a WANDA column produced by code
that was 133 commented-out lines containing a syntax error -- so whatever
produced those numbers, it was not this criterion.

Sun et al. 2023 (arXiv 2306.11695). Importance is |W_ij| * ||X_j||_2, ranked
within each output row.
"""

import pytest
import torch
import torch.nn as nn

from pruning_factory import PRUNER_REGISTRY, WandaPruner, set_prunable_scope


class TinyNet(nn.Module):
    """Conv + Linear, so both the 4-D and 2-D scoring paths are exercised."""

    def __init__(self, in_ch=3, width=8, classes=6):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, width, 3, padding=1)
        self.fc = nn.Linear(width * 4 * 4, classes)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = torch.nn.functional.adaptive_avg_pool2d(x, 4).flatten(1)
        return self.fc(x)


def _loader(skew_channel=None, n=64, batch=16, scale=100.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, 8, 8, generator=g)
    if skew_channel is not None:
        x[:, skew_channel] *= scale
    y = torch.zeros(n, dtype=torch.long)
    return [(x[i:i + batch], y[i:i + batch]) for i in range(0, n, batch)]


def _pruner(model, sparsity=0.5, loader=None, batches=4):
    return WandaPruner(model, 2, sparsity, scheduler_type='one_shot',
                       delta_T=1, total_steps=4,
                       calibration_loader=loader, calibration_batches=batches)


@pytest.fixture(autouse=True)
def _scope():
    set_prunable_scope(prune_task_head=True, prune_embeddings=False, model=None)
    yield
    set_prunable_scope(prune_task_head=False, prune_embeddings=False, model=None)


@pytest.fixture
def net():
    torch.manual_seed(0)
    return TinyNet()


def test_wanda_is_registered():
    """It was commented out of the registry, which is why --pruning_type wanda
    raised ValueError and the paper's WANDA column could not have come from it."""
    assert PRUNER_REGISTRY['wanda'] is WandaPruner


def test_ranking_is_per_output_row(net):
    """Every output neuron keeps the same number of inputs.

    This is the property that separates WANDA from a globally-thresholded
    criterion. Under a global threshold at high sparsity, whole neurons are
    deleted; WANDA holds each row's budget fixed.
    """
    p = _pruner(net, 0.5, _loader(skew_channel=0))
    p.step(3)
    mask = p.masks['fc.weight']
    keeps = mask.sum(dim=1)
    assert keeps.min() == keeps.max(), (
        f'per-row keep counts differ: {keeps.tolist()} -- ranking is not row-wise')


def test_activations_change_the_ranking(net):
    """Calibrated WANDA must disagree with magnitude on the same weights.

    If it agrees everywhere, the activation term is not reaching the score and
    the column is mislabelled.
    """
    calibrated = _pruner(TinyNet_from(net), 0.5, _loader(skew_channel=0))
    calibrated.step(3)
    plain = _pruner(TinyNet_from(net), 0.5, None)
    plain.step(3)

    agreement = (calibrated.masks['conv.weight'] == plain.masks['conv.weight'])
    assert agreement.float().mean() < 0.99, (
        'calibrated WANDA produced (almost) the same mask as magnitude pruning')


def test_high_activation_channel_is_protected(net):
    """The defining behaviour: weights reading a large input keep their weights.

    With input channel 0 amplified 100x, WANDA should retain far more of the
    conv weights that read channel 0 than magnitude does -- and pay for it out of
    the other two channels.
    """
    w = _pruner(TinyNet_from(net), 0.5, _loader(skew_channel=0))
    w.step(3)
    m = _pruner(TinyNet_from(net), 0.5, None)
    m.step(3)

    wanda_ch0 = w.masks['conv.weight'].sum(dim=(2, 3))[:, 0].sum()
    magn_ch0 = m.masks['conv.weight'].sum(dim=(2, 3))[:, 0].sum()
    assert wanda_ch0 > magn_ch0, (
        f'WANDA kept {wanda_ch0} weights on the amplified channel vs magnitude '
        f'{magn_ch0}; the activation norm is not being applied')


def test_uncalibrated_wanda_warns(net, capsys):
    """Falling back to magnitude must be loud.

    A silent fallback is the single most dangerous failure this class can have:
    it puts a WANDA-labelled column into a table that contains magnitude numbers,
    and nothing downstream can detect it.
    """
    p = _pruner(net, 0.5, None)
    p.step(3)
    out = capsys.readouterr().out
    assert 'WANDA' in out and 'magnitude' in out.lower()


def test_calibration_statistic_is_batch_count_independent(net):
    """The accumulated norm is a mean, not a sum.

    A sum would make every score scale with however many calibration batches
    happened to be available, so a run with 4 batches and a run with 8 would
    rank differently for no methodological reason.
    """
    loader = _loader(skew_channel=0, n=128)
    few = _pruner(TinyNet_from(net), 0.5, loader, batches=2)
    few.calibrate()
    many = _pruner(TinyNet_from(net), 0.5, loader, batches=8)
    many.calibrate()

    a = few.scaler_row['conv.weight']
    b = many.scaler_row['conv.weight']
    ratio = (b / a.clamp(min=1e-12)).median()
    assert 0.5 < float(ratio) < 2.0, (
        f'norms scaled by {float(ratio):.2f}x with 4x the batches -- the '
        f'statistic is a sum, not a mean')


def test_mask_reaches_the_requested_sparsity(net):
    for target in (0.5, 0.9):
        p = _pruner(TinyNet_from(net), target, _loader(skew_channel=0))
        p.step(3)
        p.apply_mask()
        total = sum(m.numel() for m in p.masks.values())
        zeros = sum((m == 0).sum().item() for m in p.masks.values())
        got = zeros / total
        assert abs(got - target) < 0.05, f'target {target}, achieved {got:.3f}'


def test_conv_folding_matches_the_weight_layout(net):
    """The per-layer norm vector must have one entry per column of the folded
    weight matrix -- in*kh*kw for a conv, in_features for a linear.

    Getting this wrong does not raise; it silently falls through to the
    ones-vector branch and the layer is magnitude-pruned while the run reports
    itself as WANDA.
    """
    p = _pruner(net, 0.5, _loader(skew_channel=0))
    p.calibrate()
    for name, param in p.prunable_params.items():
        norms = p.scaler_row.get(name)
        assert norms is not None, f'{name} has no calibration statistic'
        expected = param.reshape(param.shape[0], -1).shape[1]
        assert norms.numel() == expected, (
            f'{name}: norm vector is {norms.numel()}, folded weight has '
            f'{expected} columns -- this layer would silently fall back to '
            f'magnitude')


def TinyNet_from(reference):
    """A fresh net with identical weights, so arms differ only in the criterion."""
    torch.manual_seed(0)
    net = TinyNet()
    net.load_state_dict(reference.state_dict())
    return net


# --- the criterion itself, pinned against a hand computation ---------------
#
# Everything above this line checks that WANDA is NOT magnitude pruning. That is
# a weaker property than it sounds: a criterion that ignores |W| entirely and
# ranks purely by activation norm satisfies every one of those tests. Measured:
# replacing `scores = flat.abs() * scale` with `scores = scale.expand_as(flat)`
# flips 27.8% of conv and 40.9% of fc mask elements and still passes 8/8. So does
# dropping the .sqrt(), which scores against ||X||^2 instead of ||X||.
#
# These tests pin the exact criterion instead.

class OneLinear(nn.Module):
    """A single Linear layer, so the score can be computed by hand."""

    def __init__(self, W):
        super().__init__()
        self.fc = nn.Linear(W.shape[1], W.shape[0], bias=False)
        with torch.no_grad():
            self.fc.weight.copy_(W)

    def forward(self, x):
        return self.fc(x)


def _hand_reference(W, X, sparsity):
    """S = |W_ij| * ||X_j||_2, ranked within each output row."""
    col_norm = X.pow(2).mean(dim=0).sqrt()          # RMS over calibration rows
    scores = W.abs() * col_norm.unsqueeze(0)
    k = int(W.shape[1] * sparsity)
    keep = torch.ones_like(W)
    if k >= 1:
        idx = torch.argsort(scores, dim=1)[:, :k]
        keep.scatter_(1, idx, 0.0)
    return keep, scores


def test_mask_matches_a_hand_computed_wanda_score():
    """The mask must equal |W| * ||X|| ranked per row, element for element.

    Weights and activations are chosen so that magnitude alone, activation alone,
    and |W|*||X||^2 each give a DIFFERENT answer from |W|*||X|| -- so this single
    assertion pins the product and its exponent at once.
    """
    W = torch.tensor([
        [4.0, 1.0, 0.5, 3.0],
        [0.2, 5.0, 2.0, 0.1],
    ])
    # Column RMS ~ [1, 4, 8, 2]: column 0 is a big weight on a quiet input,
    # column 2 a small weight on a loud one -- exactly where the criteria differ.
    X = torch.zeros(16, 4)
    X[:, 0] = 1.0
    X[:, 1] = 4.0
    X[:, 2] = 8.0
    X[:, 3] = 2.0

    model = OneLinear(W)
    loader = [(X[:8], torch.zeros(8, dtype=torch.long)),
              (X[8:], torch.zeros(8, dtype=torch.long))]
    p = _pruner(model, 0.5, loader, batches=2)
    p.step(3)

    expected, scores = _hand_reference(W, X, 0.5)
    got = p.masks['fc.weight']
    assert torch.equal(got, expected), (
        f'mask does not match the hand-computed WANDA score.\n'
        f'scores=\n{scores}\nexpected=\n{expected}\ngot=\n{got}')

    # And confirm the fixture is discriminating: the three wrong criteria must
    # each disagree with the right one, or the assertion above proves nothing.
    col_norm = X.pow(2).mean(dim=0).sqrt()
    for label, wrong in (
        ('magnitude only', W.abs()),
        ('activation only', col_norm.unsqueeze(0).expand_as(W)),
        ('|W| * ||X||^2', W.abs() * col_norm.pow(2).unsqueeze(0)),
    ):
        k = int(W.shape[1] * 0.5)
        alt = torch.ones_like(W)
        alt.scatter_(1, torch.argsort(wrong, dim=1)[:, :k], 0.0)
        assert not torch.equal(alt, expected), (
            f'fixture is not discriminating: {label} gives the same mask')


def test_conv_columns_align_with_the_folded_weight_layout():
    """A per-CHANNEL statistic must not be mistaken for a per-column one.

    The score indexes columns of the [out, in*kh*kw] folded weight, so the
    statistic needs one entry per (in_channel, kh, kw) position -- not one per
    input channel. A per-channel vector has the right length only when kh*kw==1,
    and everywhere else it silently falls through to the ones-vector branch and
    magnitude-prunes the layer while the run reports itself as WANDA.
    """
    torch.manual_seed(0)
    net = TinyNet()
    p = _pruner(net, 0.5, _loader(skew_channel=0))
    p.calibrate()

    W = dict(net.named_parameters())['conv.weight']
    in_ch, kh, kw = W.shape[1], W.shape[2], W.shape[3]
    norms = p.scaler_row['conv.weight']
    assert norms.numel() == in_ch * kh * kw, (
        f'statistic has {norms.numel()} entries; the folded weight has '
        f'{in_ch * kh * kw} columns ({in_ch} channels x {kh}x{kw} kernel)')
    assert norms.numel() != in_ch, 'a per-channel vector would be indistinguishable'

    # Positions within one input channel must not all be identical: if they were,
    # a per-channel statistic would be numerically equivalent and the alignment
    # would be untestable.
    per_channel = norms.reshape(in_ch, kh * kw)
    spread = (per_channel.max(dim=1).values - per_channel.min(dim=1).values)
    assert float(spread.max()) > 0, (
        'kernel positions within a channel all carry the same statistic, so this '
        'test cannot distinguish per-position from per-channel')


def test_calibration_uses_every_batch_not_just_the_last():
    """The batch-count test above rules out a SUM. It does not rule out keeping
    only the final batch, which is also batch-count independent."""
    torch.manual_seed(0)
    net = TinyNet()

    early = torch.zeros(16, 3, 8, 8) + 5.0
    late = torch.zeros(16, 3, 8, 8) + 1.0
    y = torch.zeros(16, dtype=torch.long)

    p = _pruner(net, 0.5, [(early, y), (late, y)], batches=2)
    p.calibrate()
    both = float(p.scaler_row['conv.weight'].mean())

    q = _pruner(TinyNet_from(net), 0.5, [(late, y)], batches=1)
    q.calibrate()
    last_only = float(q.scaler_row['conv.weight'].mean())

    assert both > last_only * 1.5, (
        f'accumulated statistic {both:.3f} is indistinguishable from using only '
        f'the final batch ({last_only:.3f}); earlier batches are being discarded')


def test_layer_wise_group_is_available_and_differs_from_per_output():
    """WANDA's own Appendix A finds per-output does NOT beat layer-wise on image
    classifiers, so the grouping is a design choice here and both must work."""
    torch.manual_seed(0)
    net = TinyNet()
    loader = _loader(skew_channel=0)

    per_out = WandaPruner(TinyNet_from(net), 2, 0.5, scheduler_type='one_shot',
                          delta_T=1, total_steps=4, calibration_loader=loader,
                          calibration_batches=4, wanda_group='output')
    per_out.step(3)
    per_layer = WandaPruner(TinyNet_from(net), 2, 0.5, scheduler_type='one_shot',
                            delta_T=1, total_steps=4, calibration_loader=loader,
                            calibration_batches=4, wanda_group='layer')
    per_layer.step(3)

    keeps = per_out.masks['fc.weight'].sum(dim=1)
    assert keeps.min() == keeps.max(), 'per-output must equalise rows'

    keeps_l = per_layer.masks['fc.weight'].sum(dim=1)
    assert not torch.equal(per_out.masks['fc.weight'], per_layer.masks['fc.weight']), (
        'the two comparison groups produced identical masks, so the knob does '
        'nothing')
    assert keeps_l.min() != keeps_l.max(), (
        'a single layer-wide threshold should let rows differ')


def test_unknown_group_raises():
    with pytest.raises(ValueError, match='wanda_group'):
        WandaPruner(TinyNet(), 2, 0.5, scheduler_type='one_shot', delta_T=1,
                    total_steps=4, wanda_group='global')
