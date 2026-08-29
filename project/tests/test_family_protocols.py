"""FAMILIES protocol invariants that a silent edit could otherwise break.

These pin learning rates that were chosen by measurement, not preference. Each
one has a run behind it, and getting any of them wrong does not raise -- it
just produces a number that means something other than what the paper says it
means, which is the expensive kind of wrong.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'test_notebooks'))

from nb_common import FAMILIES                                     # noqa: E402


def test_mobilenet_arms_share_learning_rate():
    """MobileNetV2 is the one family where BOTH arms run SGD 0.1.

    It first inherited the ResNet recipe untuned, and at I.P. 0.01 its
    baseline collapses to chance under magnitude at 0.99 (10.02 +/- 0.03)
    while 0.1 reaches 78.63 +/- 3.51. Reporting the untuned arm turns a true
    +1.7 into a headline +70.3. BaCP was probed too and 0.1 is its optimum as
    well, so this is each arm's measured best rather than a handicap on one.

    It must live in FAMILIES rather than be passed per call site: a `variant=`
    suffix means "deviates from the family default", so tagging the CORRECT
    rate as a deviation splits one protocol across two record namespaces --
    which is exactly how magnitude I.P. ended up under `.lr0.1` keys while
    SNIP/WANDA I.P. sat under main keys.
    """
    fam = FAMILIES['mobilenet_v2']
    assert fam['prune']['learning_rate'] == 0.1, (
        'MobileNetV2 I.P. must run 0.1; 0.01 collapses it at high sparsity')
    assert fam['bacp']['learning_rate'] == 0.1
    assert fam['prune']['learning_rate'] == fam['bacp']['learning_rate'], (
        'the arms must be matched on this family, or the delta is an '
        'artefact of the learning rate rather than the objective')


@pytest.mark.parametrize('model', ['resnet34', 'resnet50', 'resnet101'])
def test_resnet_keeps_the_asymmetric_rates(model):
    """The ResNets keep I.P. 0.01 / BaCP 0.1 -- each arm's own tuned optimum.

    Guards against someone "harmonising" MobileNetV2's change across the grid,
    which would silently re-scope every headline number in the paper.
    """
    fam = FAMILIES[model]
    assert fam['prune']['learning_rate'] == 0.01
    assert fam['bacp']['learning_rate'] == 0.1


@pytest.mark.parametrize('model', ['vgg11', 'vgg19'])
def test_vgg_bacp_stays_at_the_probed_rate(model):
    """Both VGGs run BaCP at 0.05, fixed from the vgg19 probe.

    Zero BatchNorm layers, so they cannot absorb 0.1; see
    sec:experiments:vgglr.
    """
    assert FAMILIES[model]['bacp']['learning_rate'] == 0.05
    assert FAMILIES[model]['prune']['learning_rate'] == 0.01


def test_no_family_silently_disagrees_with_itself():
    """Every family defines both arms, so a missing key cannot fall back to a
    default that nobody chose."""
    for name, fam in FAMILIES.items():
        for arm in ('prune', 'bacp'):
            assert 'learning_rate' in fam[arm], f'{name}/{arm} has no rate'
            assert fam[arm]['learning_rate'] > 0, f'{name}/{arm} rate not positive'
