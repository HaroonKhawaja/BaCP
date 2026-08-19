"""L11 -- the analysis path, against fabricated records.

Two layers have to work and they fail independently:

  the EXECUTION path -- subprocesses, checkpoint chaining, record writing --
      which only a real sweep exercises, and

  the ANALYSIS path -- deltas, confidence intervals, the noise floor, the gates,
      the LaTeX -- which a real sweep at tier 0 exercises NOT AT ALL, because
      tier 0 runs a single seed and every gate therefore reads 'pending'.

So the gates could be arbitrarily wrong and a green tier-0 run would never say
so. That is not hypothetical: G4 was written as `lo > -0.5`, a non-inferiority
test, while its message claimed to detect indistinguishability -- a PERFECT NULL
scored the highest possible pass. The gate guarding the paper's central claim
could not fire for the case it existed to catch, and no amount of running the
pipeline would have revealed it.

These tests fabricate records instead, which makes two things possible that a
real run cannot cheaply give you:

  1. a pure null -- you cannot validate a hard stop by hoping your experiment
     fails, and
  2. seconds instead of GPU-hours.

Every test here runs against a temp results directory, so nothing touches the
real one.
"""

import json
import os
import zlib

import pytest

pytest.importorskip('pandas')

import ladder as L                                                # noqa: E402
import manifest as M                                              # noqa: E402


# A plausible ladder: sparsity costs a lot, the substrate claws some back,
# distillation helps a little, contrast helps a little more, and removing CE is
# catastrophic. Used as the "method works" fixture.
WORKING = {
    'A0': 93.0, 'A1': 78.0, 'A2': 80.5, 'A3': 82.0, 'A4': 83.5,
    'A5': 85.7, 'A6': 87.0, 'B1': 87.2, 'B2': 87.3, 'B3': 87.1,
    'C-null': 85.6, 'C0': 85.7, 'C0b': 86.0, 'C1': 87.4, 'C2': 87.6,
    'C3': 88.9, 'D1': 89.4, 'D2': 90.1,
    'D4-noPrC': 89.6, 'D4-noFiC': 89.3, 'D4-noSnC': 88.7, 'D4-noCE': 71.0,
}


def _jitter(rung, seed, spread):
    """Reproducible noise that depends on the SEED ONLY, not the rung.

    Two deliberate properties:

    zlib.crc32 rather than hash(). Python salts the hash of a str with
    PYTHONHASHSEED, so hash(('C3', 1)) differs between processes -- which made
    the null scenario draw fresh noise on every pytest invocation, and G4 then
    passed or failed at random. A flaky test of a hard stop is worse than none.

    Shared across rungs, so a seed's offset cancels in any paired difference.
    That is what a paired design with high seed-correlation looks like, and it
    makes a fabricated null EXACTLY null: with independent per-rung noise, two
    arms that are meant to be identical acquired a spurious offset of a few
    tenths, which is the same order as the gate thresholds being tested.
    Within-rung variance across seeds is still non-zero, so the noise floor
    remains estimable.
    """
    h = zlib.crc32(f'seed:{seed}'.encode())
    return ((h % 1000) / 1000.0 - 0.5) * spread


def _fabricate(root, tier, curve, jitter=0.6):
    """Write one record per manifest cell, scoring per `curve`.

    Deterministic jitter, so the within-rung std is non-zero (the noise floor is
    estimable) but the run reproduces exactly, in this process and every other.
    """
    (root / 'runs').mkdir(parents=True, exist_ok=True)
    for cell in M.cells(tier):
        rung = cell['rung']
        base = curve if isinstance(curve, (int, float)) else curve.get(rung, 85.0)
        val = base + _jitter(rung, cell['seed'], jitter)
        rec = {
            'schema_version': 1,
            'record_id': f'{rung}_seed{cell["seed"]}',
            'experiment_group': cell['key'], 'status': 'ok', 'is_smoke': True,
            'seed': cell['seed'], 'seed_source': 'args',
            'method': 'dense' if cell['script'] == 'baseline' else 'rigl',
            'model_name': cell['model_name'], 'dataset_name': cell['dataset_name'],
            'test_acc_exact_pct': round(val, 3),
            'code_fingerprint': 'f' * 12, 'git_sha': 'a' * 40,
            'ended_at': f'2026-01-01T00:00:{cell["seed"]:02d}+00:00',
            'eval_samples_dropped': 0, 'epochs': 2, 'batch_size': 32,
            'save_path': str(root / 'x.pt'), 'detail': {},
        }
        (root / 'runs' / f'{rec["record_id"]}.json').write_text(
            json.dumps(rec), encoding='utf-8')


@pytest.fixture
def report(tmp_path, monkeypatch):
    """Build a report from a fabricated record set in an isolated results dir."""
    def _make(curve, tier=1):
        monkeypatch.setenv('BACP_RESULTS_DIR', str(tmp_path))
        _fabricate(tmp_path, tier, curve)
        df = L.load_ladder(tier, root=tmp_path)
        agg = L.aggregate(df, tier=tier, root=tmp_path)
        scores = L.per_seed_scores(df)
        return agg, L.evaluate_gates(agg, scores=scores), df
    return _make


def _gate(gates, name):
    return next(g for g in gates if g[0] == name)


# --- the gates must STOP on a null -----------------------------------------

def test_gates_stop_on_a_pure_null(report):
    """Every rung scoring identically must halt the ladder.

    This is the test G4 failed: written as a non-inferiority bound, a perfect
    null returned CI [0.00, 0.00] and scored the highest possible PASS.
    """
    _, gates, _ = report(85.0)

    assert _gate(gates, 'G3')[1] == 'STOP', 'no teacher signal, yet G3 passed'
    assert _gate(gates, 'G4')[1] == 'STOP', (
        'the contrastive form is indistinguishable from feature regression, '
        'yet the gate guarding that exact claim passed')
    assert _gate(gates, 'G5')[1] == 'STOP'

    # G1 SHOULD pass on a null -- it checks that the BaCP code path reproduces
    # CE, which is precisely what a null demonstrates.
    assert _gate(gates, 'G1')[1] == 'pass'


def test_gates_pass_on_a_working_ladder(report):
    _, gates, _ = report(WORKING)
    for name in ('G1', 'G3', 'G4', 'G5'):
        assert _gate(gates, name)[1] == 'pass', _gate(gates, name)


# --- G3 must measure against the control, not the previous rung -------------

def test_g3_measures_against_the_no_teacher_control(report):
    """The exact false pass the audit found.

    G3 used to read C2's ladder delta, which aggregate() computes against C2's
    declared parent C1 -- and C1/C2 deliberately hold the teacher fixed, so that
    increment is feature-KD vs logit-KD, not teacher vs no-teacher. With
    C0b = C2 and C1 three points BELOW both, the true best teacher effect is
    zero and the incremental delta is +3.
    """
    curve = dict(WORKING, C0b=87.4, C1=84.4, C2=87.4)
    _, gates, _ = report(curve)
    assert _gate(gates, 'G3')[1] == 'STOP', (
        'G3 passed on a +3.00 increment between two arms that share a teacher, '
        'while the effect against the no-teacher control is zero')


def test_a_gate_with_a_missing_arm_is_pending_not_decided(report, tmp_path):
    """Deciding from whichever arm happens to exist asserts a result about an
    arm with zero runs."""
    monkeypatch_curve = dict(WORKING)
    _, gates, _ = report(monkeypatch_curve)
    for f in (tmp_path / 'runs').glob('C2_seed*.json'):
        f.unlink()
    df = L.load_ladder(1, root=tmp_path)
    gates = L.evaluate_gates(L.aggregate(df, tier=1, root=tmp_path),
                             scores=L.per_seed_scores(df))
    assert _gate(gates, 'G3')[1] == 'pending'


# --- the table -------------------------------------------------------------

def test_noise_floor_is_estimable_and_positive(report):
    agg, _, _ = report(WORKING)
    floor = L.noise_floor(agg)
    assert floor is not None and floor > 0


def test_latex_is_balanced_and_free_of_nan(report):
    """pandas turns None into NaN, and a LaTeX table containing 'nan' claims a
    number was computed and came out undefined -- a different and worse claim
    than 'not run'."""
    agg, _, df = report(WORKING)
    tex = L.to_latex(agg, provenance='test')
    assert tex.count('{') == tex.count('}')
    assert 'nan' not in tex.lower()
    assert r'\bottomrule' in tex


def test_deltas_below_the_noise_floor_are_not_bolded(report):
    """The single rule that forecloses the commonest objection to a 0.3-point
    'win'."""
    agg, _, _ = report(WORKING)
    floor = L.noise_floor(agg)
    tex = L.to_latex(agg, floor=floor)
    for r in agg.itertuples():
        if r.delta is None or (isinstance(r.delta, float) and r.delta != r.delta):
            continue
        if abs(float(r.delta)) < floor:
            assert r'\textbf{%+.2f}' % float(r.delta) not in tex, (
                f'{r.rung} delta {r.delta:+.2f} is inside the noise floor '
                f'{floor:.2f} but was bolded')


def test_missing_cells_render_as_a_dash(report, tmp_path):
    agg, _, _ = report(WORKING)
    for f in (tmp_path / 'runs').glob('D2_seed*.json'):
        f.unlink()
    df = L.load_ladder(1, root=tmp_path)
    agg = L.aggregate(df, tier=1, root=tmp_path)
    row = agg[agg['rung'] == 'D2'].iloc[0]
    assert row['n_ok'] == 0
    assert '--' in L.to_text(agg)
