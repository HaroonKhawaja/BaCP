"""Run the whole ablation ladder and print the table. This is the file to run.

    python project/experiments/run_ladder.py                # tier 0, CPU smoke
    python project/experiments/run_ladder.py --tier 1       # the real spine
    python project/experiments/run_ladder.py --report_only  # re-print the table

It is safe to run repeatedly and safe to interrupt. Completion is keyed off the
run records themselves, so a second invocation runs only what is missing, and
killing it mid-sweep loses at most the run in flight. To force one rung to
re-run, delete its record from `results/runs/` -- there is no other bookkeeping
to keep in sync.

What it does, in order:

  1. Validate the manifest against the live registries, and refuse to start if a
     rung names a pruner or model that does not exist. A sweep that discovers
     this at cell 40 has wasted the 39 runs before it.
  2. Write `results/manifest.csv` -- the intended grid, so a reader can tell
     planned-and-missing from never-planned.
  3. Execute every incomplete cell as its own subprocess.
  4. Aggregate the records into the ladder table, write .tex/.csv/.txt, and
     evaluate the hard-stop gates.

Tier 0 truncates the dataloaders to two batches. Its numbers are meaningless by
construction and every row it produces is stamped `is_smoke=True`; the table
prints a warning naming them. It exists to prove the pipeline end to end on CPU
so that the first GPU-hour goes into results rather than into debugging.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent), str(_HERE.parent / 'scripts')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ladder                                                      # noqa: E402
import manifest as M                                               # noqa: E402
import runner                                                      # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(
        description='Run the ablation ladder end to end and emit the table.')
    ap.add_argument('--tier', type=int, default=None,
                    help='0 = CPU smoke (default), 1 = the spine, 2/3 = wider. '
                         'Also settable with BACP_TIER.')
    ap.add_argument('--stages', nargs='*', default=None,
                    help='Restrict to stages, e.g. --stages C D')
    ap.add_argument('--rungs', nargs='*', default=None,
                    help='Restrict to named rungs, e.g. --rungs C-null C0')
    ap.add_argument('--seeds', type=int, nargs='*', default=None)
    ap.add_argument('--dry_run', action='store_true',
                    help='Print the command lines and stop.')
    ap.add_argument('--report_only', action='store_true',
                    help='Skip execution; aggregate whatever records exist.')
    ap.add_argument('--no_resume', action='store_true',
                    help='Re-run cells that already have records.')
    ap.add_argument('--stop_on_fail', action='store_true')
    ap.add_argument('--timeout', type=int, default=None,
                    help='Per-cell timeout in seconds.')
    a = ap.parse_args(argv)

    tier = M.TIER if a.tier is None else a.tier

    print('=' * 78)
    print(f'BaCP ablation ladder -- tier {tier} ({M.TIERS[tier]["name"]})')
    print('=' * 78)
    print(M.TIERS[tier]['why'])
    print()

    # 1. plan-time validation
    problems = M.validate(strict=False)
    if problems:
        print('Manifest is not self-consistent. Refusing to start:')
        for p in problems:
            print(f'  - {p}')
        return 2
    print(M.summary(tier))
    print()

    if tier == 0:
        print('!! TIER 0: dataloaders are truncated to two batches. Every number '
              'below is meaningless by construction. This tier validates the '
              'pipeline, never the method.')
        print()

    # 2. the intended grid, on disk
    if not a.report_only:
        path = M.write_manifest_csv(tier=tier)
        print(f'manifest -> {path}')

    # 3. execute
    if not a.report_only:
        t0 = time.perf_counter()
        summary = runner.run_grid(
            tier, rungs=a.rungs, stages=a.stages, seeds=a.seeds,
            dry_run=a.dry_run, resume=not a.no_resume,
            stop_on_fail=a.stop_on_fail, timeout=a.timeout)
        print(f'\nexecution took {round(time.perf_counter() - t0, 1)}s')
        if a.dry_run:
            return 0
        if summary['ok_no_record']:
            print(f"!! {summary['ok_no_record']} cell(s) exited 0 but wrote no "
                  f"record. They will be retried on the next run; check "
                  f"results/logs/ for what _finalize_run swallowed.")

    # 4. the table
    print()
    print('=' * 78)
    print('LADDER')
    print('=' * 78)
    out = ladder.report(tier)

    stops = [g for g in out['gates'] if g[1] == 'STOP']
    return 1 if stops else 0


if __name__ == '__main__':
    raise SystemExit(main())
