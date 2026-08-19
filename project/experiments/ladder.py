"""Turn run records into the ladder table.

The table has one row per rung and answers, for each: what changed, what it
scored, and what that change bought relative to the rung below it. Everything
else here exists to stop that table saying more than the data supports.

Three rules are enforced in code rather than left to discipline:

  **The noise floor is printed in every caption**, and no delta smaller than it
  is ever bolded or starred. The submitted paper's ablation deltas were all under
  0.4 points with no error bars at all, which is how a table claims a result it
  cannot support.

  **Missing is visible.** A rung the manifest planned and that has no record
  renders as `--`, and a rung whose run crashed renders as `--` *and* appears in
  the failures list. A blank cell that could mean either is how a crashed run
  becomes an invisible one.

  **n <= 5 gets no stars.** With five numbers, the table prints the five numbers.
  At n=3 paired, t(.975, 2) = 4.303 and the minimum detectable effect is ~3.1
  sigma_d -- a p-value there is theatre.

Primary endpoint is `test_acc_exact_pct`: sample-weighted top-1 on the test set,
evaluated once on the final checkpoint. Not best-epoch, which is the maximum of a
noisy sequence and therefore both upward-biased and higher-variance, and not
anything selected on the test set. The final-5-epoch validation mean is carried
alongside as a lower-variance secondary signal, never as the headline.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import manifest as M                                              # noqa: E402

PRIMARY = 'test_acc_exact_pct'


# --- loading ---------------------------------------------------------------

def _endpoint_from_detail(payload) -> float | None:
    """Mean of the last five recorded validation accuracies, if there are five."""
    detail = payload.get('detail') or {}
    for key in ('finetuning_acc', 'accuracies', 'training_acc'):
        hist = detail.get(key)
        if not hist:
            continue
        if isinstance(hist, dict):
            values = [hist[k] for k in sorted(hist, key=str)]
        else:
            values = list(hist)
        values = [float(v) for v in values if isinstance(v, (int, float))]
        if len(values) >= 5:
            return sum(values[-5:]) / 5.0
        if values:
            return sum(values) / len(values)
    return None


def load_ladder(tier=None, root=None, *, include_smoke=True):
    """Join manifest cells to their records. One row per planned cell.

    Left-joined on the manifest, deliberately: a planned cell with no record has
    to appear as a row with a null score, or it vanishes from the table and the
    reader has no way to know it was ever intended.
    """
    import pandas as pd
    from results import iter_records

    by_group = {}
    for payload in iter_records(root):
        group = payload.get('experiment_group')
        if not group:
            continue
        # Later record wins for a given group; the BaCP path writes one record
        # per phase and the fine-tuned one is the terminal state of the run.
        prev = by_group.get(group)
        if prev is None or (payload.get('ended_at') or '') >= (prev.get('ended_at') or ''):
            by_group[group] = payload

    rows = []
    for cell in M.cells(tier):
        rec = by_group.get(cell['key'])
        row = {
            'key': cell['key'], 'rung': cell['rung'], 'stage': cell['stage'],
            'label': cell['label'], 'changes': cell['changes'],
            'inherits': cell['inherits'], 'script': cell['script'],
            'paired': cell['paired'], 'gate': cell['gate'],
            'seed': cell['seed'], 'sparsity': cell['sparsity'],
            'status': 'missing', PRIMARY: None, 'val_final5': None,
            'sparsity_mask': None, 'is_smoke': None, 'record_id': None,
        }
        if rec is not None:
            row.update({
                'status': rec.get('status', 'unknown'),
                PRIMARY: rec.get(PRIMARY),
                'val_final5': _endpoint_from_detail(rec),
                'sparsity_mask': rec.get('sparsity_mask'),
                'sparsity_value': rec.get('sparsity_value'),
                'is_smoke': rec.get('is_smoke'),
                'record_id': rec.get('record_id'),
                'scope_key': rec.get('scope_key'),
                'epochs': rec.get('epochs'),
                'batch_size': rec.get('batch_size'),
                'learning_rate': rec.get('learning_rate'),
                'code_fingerprint': rec.get('code_fingerprint'),
                's_per_epoch_mean': rec.get('s_per_epoch_mean'),
                'eval_samples_dropped': rec.get('eval_samples_dropped'),
            })
        rows.append(row)

    df = pd.DataFrame(rows)
    if not include_smoke and 'is_smoke' in df:
        df = df[df['is_smoke'] != True]                            # noqa: E712
    return df


# --- statistics ------------------------------------------------------------

def _mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else None


def _std(xs):
    """Sample standard deviation, ddof=1. Returns None for n < 2."""
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def sigma_chi2_interval(s, n, conf=0.95):
    """Confidence interval on a standard deviation itself.

    Printed wherever sigma-hat is, because an n=5 estimate of sigma has a 95%
    interval running roughly 0.6x to 2.9x the point estimate. Quoting a bare
    sigma-hat from five runs as though it were the noise floor is how an
    underpowered design gets declared adequately powered.
    """
    if s is None or n is None or n < 2:
        return (None, None)
    try:
        from scipy import stats
        a = (1 - conf) / 2
        lo = math.sqrt((n - 1) * s * s / stats.chi2.ppf(1 - a, n - 1))
        hi = math.sqrt((n - 1) * s * s / stats.chi2.ppf(a, n - 1))
        return (lo, hi)
    except Exception:                                              # noqa: BLE001
        return (None, None)


# Two-sided 95% critical values of Student's t, keyed by degrees of freedom.
# Complete for df 1..30 and NOT a sparse lookup with a 2.0 default: the previous
# table omitted df 1, 6 and 8, so n = 2, 7 and 9 silently fell through to 2.0
# instead of 12.706, 2.447 and 2.306 -- a paired interval up to 6.35x too narrow,
# printed with no indication anything had gone wrong. n=7 is reachable in the
# shipped manifest (C3 declares n_seeds=8; one crashed seed gives 7), and scipy
# is in requirements-dev.txt but NOT requirements.txt, so on a GPU box this
# fallback is the live path.
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
    27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}

# Which code path produced the most recent critical value. Recorded so a
# table-time fallback interval is never mistaken for an exact one.
_T_SOURCE = {'value': 'unused'}


def _t_crit(n, conf=0.95):
    try:
        from scipy import stats
        _T_SOURCE['value'] = 'scipy'
        return float(stats.t.ppf(1 - (1 - conf) / 2, n - 1))
    except Exception:                                              # noqa: BLE001
        pass
    if conf != 0.95:
        # The table is 95% only; anything else without scipy is not answerable,
        # and guessing here would silently mislabel an interval.
        _T_SOURCE['value'] = 'unavailable'
        return float('nan')
    df = max(int(n) - 1, 1)
    _T_SOURCE['value'] = 'table' if df in _T95 else 'table-asymptotic'
    return _T95.get(df, 1.960)


def paired_delta(after: dict, before: dict, conf=0.95):
    """Per-seed differences between two arms.

    Pairing on seed is only meaningful when both arms actually ran that seed, so
    the intersection is taken rather than assuming alignment. `n_pairs` is
    reported so a reader can see when pairing silently degraded.
    """
    seeds = sorted(set(after) & set(before))
    diffs = [after[s] - before[s] for s in seeds
             if after[s] is not None and before[s] is not None]
    n = len(diffs)
    if n == 0:
        return {'n_pairs': 0, 'mean': None, 'ci': (None, None), 'sd': None,
                'paired': True}
    mean = sum(diffs) / n
    sd = _std(diffs)
    half = (_t_crit(n, conf) * sd / math.sqrt(n)) if (sd is not None and n > 1) else None
    return {
        'n_pairs': n, 'mean': mean, 'sd': sd, 'paired': True,
        'ci': (mean - half, mean + half) if half is not None else (None, None),
        'diffs': diffs, 'seeds': seeds,
    }


def unpaired_delta(after: dict, before: dict, conf=0.95):
    """Welch difference of means. Used for Stage A, where the mask lottery
    dominates and seeds do not meaningfully pair across arms."""
    a = [v for v in after.values() if v is not None]
    b = [v for v in before.values() if v is not None]
    if not a or not b:
        return {'n_pairs': 0, 'mean': None, 'ci': (None, None), 'paired': False}
    ma, mb = _mean(a), _mean(b)
    sa, sb = _std(a), _std(b)
    mean = ma - mb
    if sa is None or sb is None:
        return {'n_pairs': min(len(a), len(b)), 'mean': mean, 'ci': (None, None),
                'paired': False, 'n_a': len(a), 'n_b': len(b)}
    se = math.sqrt(sa * sa / len(a) + sb * sb / len(b))
    if se == 0:
        # Both arms constant. The Welch df expression is then 0/0 and raises,
        # which used to abort report() before any table was written -- the whole
        # run lost to a degenerate corner. Zero within-arm variance is not
        # evidence of a zero-width interval either, so the CI is reported as
        # undefined rather than as a point.
        return {'n_pairs': min(len(a), len(b)), 'mean': mean, 'paired': False,
                'ci': (None, None), 'n_a': len(a), 'n_b': len(b), 'df': None,
                'note': 'zero within-arm variance in both arms; CI undefined'}
    df = ((sa ** 2 / len(a) + sb ** 2 / len(b)) ** 2
          / ((sa ** 2 / len(a)) ** 2 / (len(a) - 1)
             + (sb ** 2 / len(b)) ** 2 / (len(b) - 1)))
    try:
        from scipy import stats
        t = float(stats.t.ppf(1 - (1 - conf) / 2, df))
    except Exception:                                              # noqa: BLE001
        t = 2.0
    return {'n_pairs': min(len(a), len(b)), 'mean': mean, 'paired': False,
            'ci': (mean - t * se, mean + t * se), 'n_a': len(a), 'n_b': len(b),
            'df': df}


def tost(delta, bound=1.0, conf=0.90):
    """Two one-sided tests for equivalence within +/- `bound` points.

    Present so a null is publishable. "The 90% CI on the difference lies inside
    +/-1.0pp, so we can reject effects larger than that" is a finding; "p = 0.43"
    is the absence of one. Stage B is designed around this.
    """
    ci = delta.get('ci', (None, None))
    if ci[0] is None:
        return {'equivalent': None, 'bound': bound}
    return {'equivalent': bool(ci[0] > -bound and ci[1] < bound),
            'bound': bound, 'ci': ci}


# --- aggregation -----------------------------------------------------------

def per_seed_scores(df, endpoint=PRIMARY):
    """{rung: {seed: value}} over successful runs.

    Shared by aggregate() and evaluate_gates() so a gate can form an explicit
    contrast against any arm rather than being limited to the ladder's
    incremental deltas -- which is what made G3 report a feature-vs-logit
    increment as a teacher-vs-no-teacher effect.
    """
    import pandas as pd
    ok = df[df['status'] == 'ok']
    out = {}
    for rung, grp in ok.groupby('rung'):
        out[rung] = {int(r.seed): (None if pd.isna(getattr(r, endpoint))
                                   else float(getattr(r, endpoint)))
                     for r in grp.itertuples()}
    return out


def aggregate(df=None, tier=None, root=None, endpoint=PRIMARY):
    """One row per rung, with the delta against the rung it inherits from."""
    import pandas as pd
    df = load_ladder(tier, root) if df is None else df

    scores = per_seed_scores(df, endpoint=endpoint)

    rows = []
    order = {r.id: i for i, r in enumerate(M.LADDER)}
    for rung in sorted({c['rung'] for c in M.cells(tier)}, key=lambda r: order.get(r, 99)):
        spec = M.BY_ID[rung]
        planned = [c for c in M.cells(tier) if c['rung'] == rung]
        got = scores.get(rung, {})
        values = [v for v in got.values() if v is not None]

        failed = int((df[(df['rung'] == rung)]['status'] == 'failed').sum())
        missing = int((df[(df['rung'] == rung)]['status'] == 'missing').sum())

        s = _std(values)
        lo, hi = sigma_chi2_interval(s, len(values))
        row = {
            'rung': rung, 'stage': spec.stage, 'label': spec.label,
            'changes': spec.changes, 'inherits': spec.inherits,
            'gate': spec.gate, 'paired': spec.paired,
            'n_planned': len(planned), 'n_ok': len(values),
            'n_failed': failed, 'n_missing': missing,
            'mean': _mean(values), 'std': s,
            'sigma_ci_lo': lo, 'sigma_ci_hi': hi,
            'seeds': sorted(k for k, v in got.items() if v is not None),
            'values': [got[k] for k in sorted(got) if got[k] is not None],
            'delta': None, 'delta_ci_lo': None, 'delta_ci_hi': None,
            'delta_n': 0, 'delta_vs': spec.inherits, 'delta_paired': None,
            'delta_indirect': False, 'delta_note': '',
            't_source': _T_SOURCE['value'],
        }

        # Walk up to the nearest ancestor that is actually scheduled in this
        # tier. Comparing only against the declared parent silently produced a
        # blank delta for A2 and A3 in tier 1 -- the tier the paper is built
        # from -- because their parent A1 is not in that tier at all. A blank
        # that means "not scheduled" is indistinguishable from one that means
        # "no effect", so the comparator that was actually used is recorded.
        parent, hops = spec.inherits, 0
        while parent and parent not in scores:
            parent = M.BY_ID[parent].inherits if parent in M.BY_ID else None
            hops += 1
        if parent and values:
            fn = paired_delta if spec.paired else unpaired_delta
            d = fn(got, scores[parent])
            row.update({'delta': d['mean'], 'delta_n': d['n_pairs'],
                        'delta_ci_lo': d['ci'][0], 'delta_ci_hi': d['ci'][1],
                        'delta_paired': d['paired'], 'delta_vs': parent,
                        'delta_indirect': hops > 0,
                        'delta_note': d.get('note', '')})
        elif spec.inherits:
            row['delta_note'] = (f'no comparator: {spec.inherits} is not '
                                 f'scheduled in this tier')
        rows.append(row)
    return pd.DataFrame(rows)


def noise_floor(agg) -> float | None:
    """Median within-rung standard deviation. The resolution limit of the table."""
    stds = [s for s in agg['std'].tolist() if s is not None and not math.isnan(s)]
    if not stds:
        return None
    stds.sort()
    n = len(stds)
    return stds[n // 2] if n % 2 else (stds[n // 2 - 1] + stds[n // 2]) / 2


# --- gates -----------------------------------------------------------------

def evaluate_gates(agg, *, g1_tol=0.5, g3_min=0.5, min_n=3, scores=None):
    """The hard stops, evaluated from the aggregate.

    Each returns 'pass', 'STOP' or 'pending'. A STOP halts the ladder and sends
    the method back for a re-think -- it is not a caveat to write around.

    Two of these were wrong in ways that would have let a null through:

      G4 was written as `lo > -0.5`, a NON-INFERIORITY test, while its message
      claimed to detect indistinguishability. A perfect null -- C3 scoring
      exactly what C2 scores on every seed -- produced CI [0.00, 0.00] and the
      highest possible PASS. The gate guarding the paper's central claim could
      not fire for the case it was built to catch. It now STOPs when the
      interval contains zero and passes only when the whole interval is above
      zero.

      G3 read C2's ladder delta, which aggregate() computes against C2's
      declared parent C1 -- and C1 and C2 deliberately hold the teacher fixed,
      so that increment is feature-KD vs logit-KD, not teacher vs no-teacher.
      With C0b=87.4, C1=84.4, C2=87.4 the true best teacher effect is +0.00pp
      and G3 printed "pass ... +3.00pp". It now computes both teacher arms
      against the no-teacher control C0b explicitly.

    `scores` is the per-rung {seed: value} mapping from aggregate(); when it is
    not supplied the gates that need an explicit contrast return 'pending'
    rather than falling back to an incremental delta.
    """
    def get(rung, col='mean'):
        sub = agg[agg['rung'] == rung]
        if sub.empty:
            return None
        v = sub.iloc[0][col]
        return None if (v is None or (isinstance(v, float) and math.isnan(v))) else v

    def n_ok(rung):
        sub = agg[agg['rung'] == rung]
        return 0 if sub.empty else int(sub.iloc[0]['n_ok'])

    out = []

    # -- G1: does the BaCP code path reproduce plain CE? --------------------
    a, b = get('C-null'), get('C0')
    if a is None or b is None:
        out.append(('G1', 'pending', 'C-null and C0 must both have runs.'))
    elif min(n_ok('C-null'), n_ok('C0')) < min_n:
        out.append(('G1', 'pending',
                    f'need >= {min_n} seeds each; have C-null={n_ok("C-null")}, '
                    f'C0={n_ok("C0")}.'))
    elif abs(a - b) <= g1_tol:
        out.append(('G1', 'pass', f'C-null {a:.2f} vs C0 {b:.2f}, '
                                  f'|diff| {abs(a - b):.2f} <= {g1_tol}'))
    else:
        out.append(('G1', 'STOP', f'C-null {a:.2f} vs C0 {b:.2f} differ by '
                                  f'{abs(a - b):.2f}pp. The BaCP code path does not '
                                  f'reproduce plain CE, so every downstream number '
                                  f'measures that discrepancy rather than the '
                                  f'method. Implementation bug, not a result.'))

    # -- G3: does ANY dense-teacher signal help, against the no-teacher arm? -
    if scores is None:
        out.append(('G3', 'pending',
                    'needs per-seed scores; call evaluate_gates(agg, scores=...).'))
    elif not all(k in scores and scores[k] for k in ('C0b', 'C1', 'C2')):
        missing = [k for k in ('C0b', 'C1', 'C2')
                   if k not in scores or not scores[k]]
        out.append(('G3', 'pending', f'no runs yet for {missing}. Deciding this '
                                     f'gate from whichever arm happens to exist '
                                     f'would assert a result about an arm with '
                                     f'zero runs.'))
    else:
        # Both teacher arms measured against C0b, the no-teacher control --
        # NOT against each other, and not against their ladder parents.
        d1 = paired_delta(scores['C1'], scores['C0b'])
        d2 = paired_delta(scores['C2'], scores['C0b'])
        ups = [d['ci'][1] for d in (d1, d2) if d['ci'][1] is not None]
        if not ups:
            out.append(('G3', 'pending', 'no overlapping seeds against C0b.'))
        elif max(ups) < g3_min:
            out.append(('G3', 'STOP',
                        f'neither KD (vs C0b: {d1["mean"]:+.2f}) nor feature-KD '
                        f'({d2["mean"]:+.2f}) can reach +{g3_min}pp (best CI upper '
                        f'bound {max(ups):+.2f}). No dense-teacher signal helps in '
                        f'this regime, so BaCP has no mechanism to exploit.'))
        else:
            out.append(('G3', 'pass',
                        f'best teacher signal vs the no-teacher control C0b: '
                        f'KD {d1["mean"]:+.2f}, feature-KD {d2["mean"]:+.2f}, '
                        f'best CI upper bound {max(ups):+.2f}pp'))

    # -- G4: is the contrastive FORM distinguishable from feature regression? -
    lo3, hi3, n3 = get('C3', 'delta_ci_lo'), get('C3', 'delta_ci_hi'), n_ok('C3')
    if lo3 is None or hi3 is None:
        out.append(('G4', 'pending', 'needs the C3 delta against C2.'))
    elif n3 < min_n:
        out.append(('G4', 'pending', f'C3 has {n3} seed(s); need >= {min_n}.'))
    elif lo3 > 0:
        out.append(('G4', 'pass', f'C3 - C2 = [{lo3:+.2f}, {hi3:+.2f}], entirely '
                                  f'above zero'))
    else:
        out.append(('G4', 'STOP',
                    f'C3 - C2 = [{lo3:+.2f}, {hi3:+.2f}], which contains zero. The '
                    f'contrastive form is not distinguishable from plain feature '
                    f'distillation against the same teacher, and that is the '
                    f"paper's central claim."))

    # -- G5: are the individual teachers distinguishable? --------------------
    d1lo, d1hi = get('D1', 'delta_ci_lo'), get('D1', 'delta_ci_hi')
    d2lo, d2hi = get('D2', 'delta_ci_lo'), get('D2', 'delta_ci_hi')
    if None in (d1lo, d1hi, d2lo, d2hi):
        out.append(('G5', 'pending', 'needs D1 and D2 deltas.'))
    elif (d1lo <= 0 <= d1hi) and (d2lo <= 0 <= d2hi):
        out.append(('G5', 'STOP', 'both +SnC and +PrC have CIs containing zero. Do '
                                  'not tune (Stage D-prime) a method whose own '
                                  'components are indistinguishable from each '
                                  'other.'))
    else:
        out.append(('G5', 'pass', f'+SnC [{d1lo:+.2f}, {d1hi:+.2f}], '
                                  f'+PrC [{d2lo:+.2f}, {d2hi:+.2f}]'))
    return out


# --- rendering -------------------------------------------------------------

def _fmt(v, nd=2, dash='--'):
    """None, NaN and pandas' NA all render as the dash.

    pandas converts None to NaN when a column is built, so a table promising
    '--' for a missing cell printed 'nan' instead -- and a LaTeX table with
    'nan' in it is a table that says a number was computed and came out
    undefined, which is a different and worse claim than "not run".
    """
    if v is None:
        return dash
    try:
        if isinstance(v, float) and math.isnan(v):
            return dash
    except TypeError:
        pass
    try:
        import pandas as pd
        if pd.isna(v):
            return dash
    except Exception:                                              # noqa: BLE001
        pass
    try:
        return f'{float(v):.{nd}f}'
    except (TypeError, ValueError):
        return dash


def to_text(agg, *, floor=None, endpoint=PRIMARY) -> str:
    floor = noise_floor(agg) if floor is None else floor
    lines = []
    head = (f'{"rung":<10} {"n":>3}  {"mean":>7} {"+-sd":>7}   '
            f'{"delta":>8} {"95% CI":>18}  {"":>2} change')
    lines.append(head)
    lines.append('-' * len(head))
    stage = None
    for r in agg.itertuples():
        if r.stage != stage:
            stage = r.stage
            lines.append(f'-- stage {stage} ' + '-' * (len(head) - 11 - len(stage)))
        ci = ('' if _fmt(r.delta_ci_lo) == '--'
              else f'[{_fmt(r.delta_ci_lo)}, {_fmt(r.delta_ci_hi)}]')
        flag = ''
        if r.delta is not None and floor:
            flag = '*' if abs(r.delta) > floor else '~'
        miss = ''
        if r.n_failed:
            miss = f'  [{r.n_failed} FAILED]'
        elif r.n_ok < r.n_planned:
            miss = f'  [{r.n_planned - r.n_ok} missing]'
        lines.append(
            f'{r.rung:<10} {r.n_ok:>3}  {_fmt(r.mean):>7} {_fmt(r.std):>7}   '
            f'{_fmt(r.delta, dash=""):>8} {ci:>18}  {flag:>2} {r.changes}{miss}')
    lines.append('')
    if floor:
        lines.append(f'Noise floor (median within-rung sd, ddof=1): {floor:.2f} points. '
                     f"'*' marks |delta| above it, '~' below -- a '~' is NOT a result.")
    else:
        lines.append('Noise floor: not estimable (no rung has 2+ successful seeds).')
    lines.append(f'Endpoint: {endpoint}. No test-set model selection; no early stopping.')
    return '\n'.join(lines)


def to_latex(agg, *, caption=None, label='tab:ladder', floor=None,
             provenance='') -> str:
    """booktabs, hand-emitted.

    Not DataFrame.to_latex: it escapes pre-escaped strings a second time and
    leaves nowhere to hook \\cmidrule between stages, which is what makes a
    ladder table readable as a ladder.
    """
    floor = noise_floor(agg) if floor is None else floor
    out = ['% ' + line for line in provenance.splitlines() if line]
    out += [
        r'\begin{table}[t]', r'\centering', r'\small',
        r'\begin{tabular}{llrrrr}', r'\toprule',
        r'Rung & Change & $n$ & Top-1 (\%) & $\Delta$ & 95\% CI \\',
        r'\midrule',
    ]
    stage = None
    for r in agg.itertuples():
        if r.stage != stage:
            if stage is not None:
                out.append(r'\cmidrule(lr){1-6}')
            stage = r.stage
        m, sd = _fmt(r.mean), _fmt(r.std)
        mean = f'{m} $\\pm$ {sd}' if (m != '--' and sd != '--') else m
        # No bolding inside the noise floor. This single rule forecloses the most
        # common objection a reviewer raises against a 0.3-point "win".
        delta = '--'
        if _fmt(r.delta) != '--':
            delta = f'{float(r.delta):+.2f}'
            if floor and abs(float(r.delta)) > floor:
                delta = r'\textbf{' + delta + '}'
        ci = ('--' if _fmt(r.delta_ci_lo) == '--'
              else f'[{float(r.delta_ci_lo):+.2f}, {float(r.delta_ci_hi):+.2f}]')
        change = str(r.changes).replace('_', r'\_').replace('%', r'\%')
        out.append(f'{r.rung} & {change} & {r.n_ok} & {mean} & {delta} & {ci} \\\\')
    out += [r'\bottomrule', r'\end{tabular}']

    cap = caption or (
        'Ablation ladder on ResNet-34 / CIFAR-10 at 99.9\\% sparsity. Each rung '
        'changes exactly one thing relative to the rung above it and inherits '
        'everything else. $\\Delta$ is against the inherited rung; Stage A is '
        'unpaired (Welch), Stages B--D are seed-paired. ')
    if floor:
        cap += (f'Median within-rung standard deviation is {floor:.2f} points; '
                f'differences below that are not resolvable at these sample sizes '
                f'and are left unbolded.')
    out += [r'\caption{' + cap + '}', r'\label{' + label + '}', r'\end{table}']
    return '\n'.join(out)


def write_tables(agg, root=None, *, stem='table_ladder', provenance='') -> dict:
    from results import results_dir
    out_root = results_dir(root) / 'tables'
    out_root.mkdir(parents=True, exist_ok=True)

    csv_path = out_root / f'{stem}.csv'
    tex_path = out_root / f'{stem}.tex'
    txt_path = out_root / f'{stem}.txt'

    agg.to_csv(csv_path, index=False)
    tex_path.write_text(to_latex(agg, provenance=provenance), encoding='utf-8')
    txt_path.write_text(to_text(agg), encoding='utf-8')
    return {'csv': csv_path, 'tex': tex_path, 'txt': txt_path}


def provenance_header(df, root=None) -> str:
    from results import capture_git, code_fingerprint
    from datetime import datetime, timezone
    git = capture_git()
    n_smoke = int((df['is_smoke'] == True).sum()) if 'is_smoke' in df else 0  # noqa: E712
    lines = [
        f'generated  {datetime.now(timezone.utc).isoformat()}',
        f'git        {git.get("git_sha")} ({git.get("git_branch")})'
        f'{" DIRTY" if git.get("git_dirty") else ""}',
        f'fingerprint {code_fingerprint()[:16]}',
        f'rows       {len(df)} planned, {(df["status"] == "ok").sum()} ok, '
        f'{(df["status"] == "failed").sum()} failed, '
        f'{(df["status"] == "missing").sum()} missing',
    ]
    if n_smoke:
        lines.append(f'!! {n_smoke} of these rows are TIER-0 SMOKE runs on truncated '
                     f'loaders. They are not measurements and must not be published.')
    return '\n'.join(lines)


def report(tier=None, root=None, endpoint=PRIMARY, write=True) -> dict:
    """The one call the notebook makes. Prints the table, writes the files."""
    df = load_ladder(tier, root)
    agg = aggregate(df, tier=tier, root=root, endpoint=endpoint)
    scores = per_seed_scores(df, endpoint=endpoint)
    prov = provenance_header(df, root)

    print(prov)
    print()
    print(to_text(agg, endpoint=endpoint))
    print()
    print('Gates')
    print('-----')
    gates = evaluate_gates(agg, scores=scores)
    for name, status, detail in gates:
        mark = {'pass': 'ok  ', 'STOP': 'STOP', 'pending': '... '}[status]
        print(f'  [{mark}] {name}: {detail}')

    stops = [g for g in gates if g[1] == 'STOP']
    if stops:
        print()
        print('!! A hard stop has fired. The ladder halts here by design: the next '
              'stage would spend GPU-hours measuring something the controls have '
              'already ruled out.')

    paths = write_tables(agg, root, provenance=prov) if write else {}
    if paths:
        print()
        for kind, path in paths.items():
            print(f'  wrote {kind}: {path}')
    return {'df': df, 'agg': agg, 'gates': gates, 'paths': paths,
            'noise_floor': noise_floor(agg)}


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--tier', type=int, default=None)
    a = ap.parse_args()
    report(a.tier)
