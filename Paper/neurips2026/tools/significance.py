#!/usr/bin/env python3
"""Paired significance test over the matched grid.

    python Paper/neurips2026/tools/significance.py results/runs_grid

"BaCP leads 35 of 48 cells" is a weak way to state the result: 35/48 is not far
from a coin flip by eye, and a reader has no way to judge it. This computes the
test that answers that, from the same records the accuracy macros come from.

The unit of analysis is the CELL (model x criterion x sparsity), and within a
cell the arms are paired BY SEED -- both arms of a seed share an initialisation
and a dense checkpoint, so the difference is the quantity with the least
nuisance variance in it. The test is over the 48 per-cell mean differences.

Cells are not independent (four backbones x three criteria x four sparsities
share models and data), so this is a statement about the grid as run, not a
population claim. It is reported for what it is.

Values printed here are registered in the MEASURED block of
make_results_macros.py; re-run this after any change to the records and update
them together.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import re
import statistics as st
import sys

# static.<arm>.<model>.<dataset>.s<sparsity>.<criterion>.seed<n>
# Anchored: a trailing variant suffix (.lr0.1, .gradnorm) is a DIFFERENT
# experiment and must not be folded in as though it were another seed.
KEY = re.compile(r'^static\.(prune|bacp)\.([a-z0-9_]+)\.([a-z0-9]+)\.'
                 r's(0\.\d+)\.(magnitude|snip|wanda)\.seed(\d)$')


def load(records_dir: str, dataset: str = 'cifar10'):
    cells: dict = collections.defaultdict(dict)
    skipped = collections.Counter()
    for path in sorted(glob.glob(os.path.join(records_dir, '*.json'))):
        try:
            rec = json.loads(open(path, encoding='utf-8').read())
        except Exception:
            skipped['unreadable'] += 1
            continue
        m = KEY.match(rec.get('experiment_group') or '')
        if not m:
            skipped['not a grid cell'] += 1
            continue
        arm, model, ds, sp, crit, seed = m.groups()
        if ds != dataset:
            skipped['other dataset'] += 1
            continue
        if rec.get('status') != 'ok':
            skipped['not ok'] += 1
            continue
        acc = rec.get('test_acc_exact_pct')
        if acc is None:
            skipped['no exact accuracy'] += 1
            continue
        cells[(model, crit, sp)].setdefault(arm, {})[seed] = acc
    return cells, skipped


def sign_test(k: int, n: int) -> float:
    """Two-sided exact binomial test against p=0.5."""
    k = max(k, n - k)
    tail = sum(math.comb(n, i) for i in range(k, n + 1))
    return min(1.0, 2 * tail / (2 ** n))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('records', nargs='?', default='results/runs_grid')
    ap.add_argument('--dataset', default='cifar10')
    ap.add_argument('--min-seeds', type=int, default=3)
    args = ap.parse_args(argv)

    cells, skipped = load(args.records, args.dataset)

    deltas, rows = [], []
    for key, arms in sorted(cells.items()):
        ip, bp = arms.get('prune', {}), arms.get('bacp', {})
        seeds = sorted(set(ip) & set(bp))
        if len(seeds) < args.min_seeds:
            continue
        per_seed = [bp[s] - ip[s] for s in seeds]
        mu = st.fmean(per_seed)
        sd = st.stdev(per_seed) if len(per_seed) > 1 else 0.0
        deltas.append(mu)
        rows.append((key, mu, sd, abs(mu) > sd))

    n = len(deltas)
    if not n:
        print('no cells with enough paired seeds', file=sys.stderr)
        return 1
    pos = sum(1 for d in deltas if d > 0)
    clears = sum(1 for *_x, c in rows if c)

    print(f'records dir      : {args.records}')
    print(f'skipped          : {dict(skipped)}')
    print(f'cells (n>={args.min_seeds} paired) : {n}')
    print(f'BaCP ahead       : {pos} / {n}')
    print(f'mean delta       : {st.fmean(deltas):+.2f}')
    print(f'median delta     : {st.median(deltas):+.2f}')
    print(f'clears own spread: {clears} / {n}')
    print()
    p_sign = sign_test(pos, n)
    print(f'sign test (two-sided exact) : p = {p_sign:.3g}')
    try:
        from scipy.stats import wilcoxon
        w, p_w = wilcoxon(deltas)
        print(f'Wilcoxon signed-rank        : W = {w:.1f}, p = {p_w:.3g}')
    except ImportError:
        p_w = None
        print('Wilcoxon signed-rank        : scipy not installed')

    print()
    print('--- macro values for the MEASURED block ---')
    print(f'\\defresult{{summary.test.ncells}}{{{n}}}')
    print(f'\\defresult{{summary.test.npositive}}{{{pos}}}')
    print(f'\\defresult{{summary.test.clears}}{{{clears}}}')
    print(f'\\defresult{{summary.test.signp}}{{{p_sign:.4f}}}')
    if p_w is not None:
        print(f'\\defresult{{summary.test.wilcoxonp}}{{{p_w:.2g}}}')
        print(f'\\defresult{{summary.test.wilcoxonw}}{{{w:.0f}}}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
