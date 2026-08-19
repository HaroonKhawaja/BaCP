"""L11 -- the ladder's statistics and its gates.

`evaluate_gates` had no test at all, and two of the four gates were wrong in the
same direction: they could not fire for the case they were built to catch.

  **G4** was written as `lo > -0.5` -- a non-inferiority test -- while its message
  claimed to detect indistinguishability. A perfect null (C3 scoring exactly what
  C2 scores on every seed) gave CI [0.00, 0.00] and the highest possible PASS. The
  gate guarding the paper's central claim scored "the contrastive form does
  literally nothing" as success and let the ladder proceed to Stage D.

  **G3** read C2's ladder delta, which `aggregate` computes against C2's declared
  parent C1 -- and C1/C2 deliberately hold the teacher fixed, so that increment is
  feature-KD vs logit-KD, not teacher vs no-teacher. With C0b=87.4, C1=84.4,
  C2=87.4 the true best teacher effect is +0.00pp and G3 printed "pass ...
  +3.00pp": the exact reviewer objection the rung exists to refuse, answered with
  a number that does not exist.

Every test here fabricates its own records, so nothing depends on a sweep having
run.
"""

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

_EXPERIMENTS = Path(__file__).resolve().parents[1] / 'experiments'
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

import ladder as L                                                # noqa: E402
import manifest as M                                              # noqa: E402


def _records_frame(values, tier=1, n=5):
    """A DataFrame shaped exactly like load_ladder's output.

    Built from M.cells(tier) rather than by hand so aggregate() is exercised on
    the real manifest -- including its inherits chain, which is what the
    nearest-scheduled-ancestor logic walks.
    """
    rows = []
    for cell in M.cells(tier):
        rung, seed = cell['rung'], cell['seed']
        have = rung in values and seed <= n
        rows.append({
            'key': cell['key'], 'rung': rung, 'stage': cell['stage'],
            'label': cell['label'], 'changes': cell['changes'],
            'inherits': cell['inherits'], 'script': cell['script'],
            'paired': cell['paired'], 'gate': cell['gate'], 'seed': seed,
            'sparsity': cell['sparsity'],
            'status': 'ok' if have else 'missing',
            L.PRIMARY: values.get(rung) if have else None,
            'val_final5': None, 'sparsity_mask': None, 'is_smoke': False,
            'record_id': cell['key'] if have else None,
        })
    return pd.DataFrame(rows)


def _agg(values, n=5, noise=0.0, per_seed=None):
    """Build an aggregate frame and per-seed scores from {rung: mean}.

    `per_seed` overrides a rung's values with an explicit {seed: value} map, so a
    NOISY null can be expressed -- the seeds must disagree in the DIFFERENCE, not
    merely each carry the same offset, or the paired sd is zero and the interval
    collapses to a point.
    """
    rows, scores = [], {}
    for rung, mu in values.items():
        spec = M.BY_ID[rung]
        if per_seed and rung in per_seed:
            seeds = dict(per_seed[rung])
        else:
            seeds = {s: mu + (noise * ((s % 3) - 1)) for s in range(1, n + 1)}
        scores[rung] = seeds
        vals = list(seeds.values())
        rows.append({
            'rung': rung, 'stage': spec.stage, 'label': spec.label,
            'changes': spec.changes, 'inherits': spec.inherits, 'gate': spec.gate,
            'paired': spec.paired, 'n_planned': n, 'n_ok': n, 'n_failed': 0,
            'n_missing': 0, 'mean': L._mean(vals), 'std': L._std(vals),
            'sigma_ci_lo': None, 'sigma_ci_hi': None,
            'seeds': sorted(seeds), 'values': vals,
            'delta': None, 'delta_ci_lo': None, 'delta_ci_hi': None,
            'delta_n': 0, 'delta_vs': spec.inherits, 'delta_paired': None,
            'delta_indirect': False, 'delta_note': '', 't_source': 'table',
        })
    agg = pd.DataFrame(rows)
    for i, r in agg.iterrows():
        parent = r['inherits']
        if parent in scores:
            d = L.paired_delta(scores[r['rung']], scores[parent])
            agg.at[i, 'delta'] = d['mean']
            agg.at[i, 'delta_n'] = d['n_pairs']
            agg.at[i, 'delta_ci_lo'] = d['ci'][0]
            agg.at[i, 'delta_ci_hi'] = d['ci'][1]
    return agg, scores


def _gate(name, agg, scores):
    return next(g for g in L.evaluate_gates(agg, scores=scores) if g[0] == name)


# --- G4: the paper's central claim -----------------------------------------

def test_g4_stops_on_a_perfect_null():
    """The regression that matters most in this file.

    C3 == C2 on every seed means the contrastive form did nothing. That must be
    a hard stop, not the highest-scoring pass.
    """
    agg, scores = _agg({'C2': 87.4, 'C3': 87.4})
    assert _gate('G4', agg, scores)[1] == 'STOP'


def test_g4_stops_on_a_noisy_null():
    """Per-seed differences scattered around zero: mean +0.02, CI straddling it."""
    agg, scores = _agg(
        {'C2': 87.4, 'C3': 87.4},
        per_seed={'C2': {1: 87.1, 2: 87.6, 3: 87.3, 4: 87.5, 5: 87.5},
                  'C3': {1: 87.5, 2: 87.2, 3: 87.6, 4: 87.2, 5: 87.4}})
    name, status, detail = _gate('G4', agg, scores)
    assert status == 'STOP', detail
    lo = agg[agg['rung'] == 'C3'].iloc[0]['delta_ci_lo']
    hi = agg[agg['rung'] == 'C3'].iloc[0]['delta_ci_hi']
    assert lo < 0 < hi, f'fixture is not actually a noisy null: [{lo}, {hi}]'


def test_g4_passes_only_when_the_interval_clears_zero():
    agg, scores = _agg({'C2': 87.4, 'C3': 89.0})
    assert _gate('G4', agg, scores)[1] == 'pass'


def test_g4_stops_when_c3_is_worse():
    agg, scores = _agg({'C2': 89.0, 'C3': 87.0})
    assert _gate('G4', agg, scores)[1] == 'STOP'


def test_g4_is_pending_below_the_seed_floor():
    """Two seeds cannot settle a 1-point effect; refusing to rule is the
    honest outcome, and it is distinct from ruling 'no'."""
    agg, scores = _agg({'C2': 87.4, 'C3': 89.0}, n=2)
    assert _gate('G4', agg, scores)[1] == 'pending'


# --- G3: does ANY dense-teacher signal help? -------------------------------

def test_g3_stops_when_no_teacher_arm_beats_the_no_teacher_control():
    """The audit's false pass, reproduced.

    C1 costs 3 points, C2 lands exactly on C0b. The best true teacher effect is
    zero. Reading C2's incremental delta against C1 instead reported +3.00pp.
    """
    agg, scores = _agg({'C0b': 87.4, 'C1': 84.4, 'C2': 87.4})
    name, status, detail = _gate('G3', agg, scores)
    assert status == 'STOP', detail
    assert '+0.00' in detail or '-3.00' in detail


def test_g3_passes_when_a_teacher_genuinely_helps():
    agg, scores = _agg({'C0b': 87.4, 'C1': 90.4, 'C2': 90.5})
    name, status, detail = _gate('G3', agg, scores)
    assert status == 'pass', detail
    assert 'C0b' in detail, 'the contrast must name the control it was measured against'


def test_g3_is_pending_when_an_arm_has_not_run():
    """Deciding from the surviving arm asserted a result about an arm with zero
    runs -- and then halted the ladder on it."""
    agg, scores = _agg({'C0b': 87.4, 'C1': 90.4})
    assert _gate('G3', agg, scores)[1] == 'pending'


def test_g3_is_pending_without_per_seed_scores():
    agg, _ = _agg({'C0b': 87.4, 'C1': 90.4, 'C2': 90.5})
    assert next(g for g in L.evaluate_gates(agg) if g[0] == 'G3')[1] == 'pending'


# --- G1 --------------------------------------------------------------------

def test_g1_stops_when_the_bacp_path_does_not_reproduce_ce():
    agg, scores = _agg({'C-null': 84.0, 'C0': 87.4})
    assert _gate('G1', agg, scores)[1] == 'STOP'


def test_g1_passes_within_tolerance():
    agg, scores = _agg({'C-null': 87.3, 'C0': 87.4})
    assert _gate('G1', agg, scores)[1] == 'pass'


# --- critical values -------------------------------------------------------

@pytest.mark.parametrize('n,expected', [
    (2, 12.706), (3, 4.303), (4, 3.182), (5, 2.776),
    (6, 2.571), (7, 2.447), (8, 2.365), (9, 2.306), (10, 2.262),
])
def test_t_critical_values_are_right_at_every_small_n(n, expected):
    """The fallback table omitted df 1, 6 and 8, so n = 2, 7 and 9 silently used
    2.0 -- a paired interval up to 6.35x too narrow, printed with no indication.
    scipy is in requirements-dev.txt but NOT requirements.txt, so on a GPU box
    the fallback is the live path."""
    assert L._t_crit(n) == pytest.approx(expected, rel=1e-3)


def test_t_critical_fallback_matches_scipy(monkeypatch):
    """Force the no-scipy path and compare against the real distribution."""
    scipy = pytest.importorskip('scipy')
    import builtins
    real_import = builtins.__import__

    def no_scipy(name, *a, **k):
        if name.startswith('scipy'):
            raise ImportError('blocked')
        return real_import(name, *a, **k)

    for n in range(2, 12):
        exact = float(scipy.stats.t.ppf(0.975, n - 1))
        monkeypatch.setattr(builtins, '__import__', no_scipy)
        fallback = L._t_crit(n)
        monkeypatch.undo()
        assert fallback == pytest.approx(exact, rel=2e-3), f'n={n}'


# --- degenerate inputs -----------------------------------------------------

def test_unpaired_delta_survives_zero_variance_in_both_arms():
    """This raised ZeroDivisionError out of report(), losing the whole table."""
    d = L.unpaired_delta({1: 10.0, 2: 10.0, 3: 10.0}, {1: 12.5, 2: 12.5, 3: 12.5})
    assert d['mean'] == pytest.approx(-2.5)
    assert d['ci'] == (None, None), 'a zero-width CI would overstate the evidence'
    assert 'variance' in d.get('note', '')


def test_paired_delta_reports_the_number_of_pairs_it_actually_used():
    d = L.paired_delta({1: 90.0, 2: 91.0, 3: None}, {1: 89.0, 2: 90.0, 3: 88.0})
    assert d['n_pairs'] == 2


def test_paired_delta_on_disjoint_seeds_is_empty_not_wrong():
    d = L.paired_delta({1: 90.0}, {2: 89.0})
    assert d['n_pairs'] == 0 and d['mean'] is None


# --- rendering -------------------------------------------------------------

@pytest.mark.parametrize('value', [None, float('nan')])
def test_missing_cells_render_as_a_dash_not_nan(value):
    """pandas turns None into NaN when a column is built, so a table promising
    '--' printed 'nan' -- which claims a number was computed and came out
    undefined, a different and worse statement than 'not run'."""
    assert L._fmt(value) == '--'


def test_latex_table_has_no_nan_and_balances():
    agg, _ = _agg({'A0': 93.0, 'C0': 87.0})
    agg.loc[len(agg)] = {c: None for c in agg.columns}
    agg.at[len(agg) - 1, 'rung'] = 'C3'
    agg.at[len(agg) - 1, 'stage'] = 'C'
    agg.at[len(agg) - 1, 'changes'] = 'never ran'
    agg.at[len(agg) - 1, 'n_ok'] = 0
    tex = L.to_latex(agg)
    assert 'nan' not in tex.lower()
    assert tex.count(r'\begin{tabular}') == tex.count(r'\end{tabular}') == 1
    assert '--' in tex


def test_noise_floor_ignores_single_seed_rungs():
    agg, _ = _agg({'A0': 93.0, 'C0': 87.0}, n=1)
    assert L.noise_floor(agg) is None


# --- delta comparators -----------------------------------------------------

def test_delta_falls_back_to_the_nearest_scheduled_ancestor():
    """In tier 1 -- the tier the paper is built from -- A2 and A3 declare A1 as
    their parent, and A1 is not in that tier. Comparing only against the declared
    parent left them with a blank delta indistinguishable from 'no effect'."""
    assert 'A1' not in M.TIERS[1]['rungs']
    assert M.BY_ID['A2'].inherits == 'A1'
    assert M.BY_ID['A3'].inherits == 'A1'

    # Exercise aggregate() itself, not the fixture: the ancestor walk lives there.
    df = _records_frame({'A0': 93.0, 'A2': 90.0, 'A3': 91.0, 'A4': 91.2}, tier=1)
    agg = L.aggregate(df, tier=1)

    a3 = agg[agg['rung'] == 'A3'].iloc[0]
    assert a3['delta'] is not None or a3['delta_note'], (
        'a blank delta must carry a reason, or it is indistinguishable from '
        '"no effect"')
    if a3['delta'] is not None:
        assert a3['delta_vs'] == 'A0', (
            f"A1 is not in tier 1, so A3 must fall back to the nearest scheduled "
            f"ancestor; got {a3['delta_vs']}")
        assert bool(a3['delta_indirect']) is True

    # A4's parent A3 IS scheduled, so it must be a direct comparison.
    a4 = agg[agg['rung'] == 'A4'].iloc[0]
    assert a4['delta_vs'] == 'A3'
    assert bool(a4['delta_indirect']) is False
