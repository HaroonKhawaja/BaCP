#!/usr/bin/env python3
"""Find the fastest way to run these experiments.

Runs the REAL training command for a handful of epochs under different
settings, times each epoch, and prints a table so you can see what actually
helps.

Why this file is safe to edit freely: it lives outside project/, and
code_fingerprint only hashes files under project/. So nothing here can change
a fingerprint, invalidate a run record, or affect the sweep. It also writes its
checkpoints into its own scratch folders, well away from results/.

    python bench_speed.py                  # compare settings on the dense rung
    python bench_speed.py --rung A3        # a sparse rung instead
    python bench_speed.py --rung C0        # a contrastive rung (the slow ones)
    python bench_speed.py --epochs 8       # time more epochs per setting

EDIT THE `SETTINGS` LIST BELOW. That is the whole point of the script.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

# True when this was pasted into a notebook cell rather than run as a file.
# Notebooks have no __file__, and their sys.argv belongs to the kernel, so both
# the repo lookup and the argument parsing have to behave differently.
IN_NOTEBOOK = '__file__' not in globals()


def find_repo() -> Path:
    """Locate the repo root, however this file is being run.

    Three ways, in order of reliability: the file's own location; the env var
    nb.setup() exports; then walking up from the working directory looking for
    project/, which is the same search run_ladder.ipynb does.
    """
    if not IN_NOTEBOOK:
        return Path(__file__).resolve().parent
    if os.environ.get('BACP_REPO'):
        return Path(os.environ['BACP_REPO'])
    here = Path.cwd()
    while not (here / 'project').exists() and here != here.parent:
        here = here.parent
    if not (here / 'project').exists():
        raise SystemExit('could not find the repo root. Either run nb.setup() '
                         'first, or set BACP_REPO to the repo path.')
    return here


def find_scratch() -> Path:
    """Somewhere fast to write checkpoints during the benchmark.

    Must NOT be the repo on Databricks: the repo is on a network mount, which
    is the very thing being measured against.
    """
    if os.environ.get('BENCH_SCRATCH'):
        return Path(os.environ['BENCH_SCRATCH'])
    if Path('/local_disk0').is_dir():          # Databricks node-local SSD
        return Path('/local_disk0/bench')
    import tempfile
    return Path(tempfile.gettempdir()) / 'bacp_bench'


REPO = find_repo()
SCRATCH = find_scratch()

# --------------------------------------------------------------------------
# THE THING YOU TUNE
#
# Each entry is one configuration to time. Add, remove and edit freely.
#
#   name      what to call it in the results table
#   workers   --num_workers, the dataloader processes per run
#   where     the working directory, which is what decides where the ~85 MB
#             per-epoch checkpoint lands. The training code builds its save
#             path as './research/bacp/...', relative to this. On Databricks,
#             the repo is on a network mount (/Workspace) and local disk is
#             /local_disk0 -- expect a large difference between the two.
#   extra     any additional command-line flags, e.g. ['--batch_size', '256']
# --------------------------------------------------------------------------

SETTINGS = [
    # The baseline: exactly what your sweep runs today.
    dict(name='as-is (repo dir, 10 workers)',
         workers=10, where=REPO / 'project' / 'scripts', extra=[]),

    # Same, but checkpoints go to local disk instead of the network mount.
    # If this is much faster, the per-epoch checkpoint write is your bottleneck.
    dict(name='local disk, 10 workers',
         workers=10, where=SCRATCH / 'a', extra=[]),

    # Fewer workers. With two runs sharing 24 cores you are at zero headroom;
    # fewer workers each may beat more.
    dict(name='local disk, 4 workers',
         workers=4, where=SCRATCH / 'b', extra=[]),

    # More workers, to check you are not simply starved of data.
    dict(name='local disk, 16 workers',
         workers=16, where=SCRATCH / 'c', extra=[]),

    # Bigger batches. Fewer optimizer steps per epoch means fewer of the
    # per-step GPU stalls, and this model is tiny next to the card.
    dict(name='local disk, 10 workers, batch 256',
         workers=10, where=SCRATCH / 'd', extra=['--batch_size', '256']),
]


# --------------------------------------------------------------------------
# machinery
# --------------------------------------------------------------------------

EPOCH_LINE = re.compile(r'Epoch \[(\d+)/(\d+)\]')


def real_command(rung: str, seed: int, tier: int) -> list[str]:
    """The exact argv the sweep would use for this cell.

    Built through the project's own manifest and runner rather than typed out
    here, so a benchmark can never drift from what actually runs.
    """
    for sub in ('project', 'project/experiments', 'project/scripts'):
        path = str(REPO / sub)
        if path not in sys.path:
            sys.path.insert(0, path)

    import manifest
    import runner

    matches = [c for c in manifest.cells(tier, rungs=[rung]) if c['seed'] == seed]
    if not matches:
        available = sorted({c['rung'] for c in manifest.cells(tier)})
        raise SystemExit(f'no cell for rung {rung} seed {seed} in tier {tier}.\n'
                         f'tier {tier} has: {", ".join(available)}')
    return runner.build_command(matches[0], python=sys.executable)


def set_flag(argv: list[str], flag: str, value: str) -> list[str]:
    """Replace a flag's value, or append the flag if it is not there."""
    argv = list(argv)
    if flag in argv:
        argv[argv.index(flag) + 1] = value
    else:
        argv += [flag, value]
    return argv


def time_one(argv, where: Path, epochs: int, gpu: int):
    """Run until `epochs` epochs have been logged, timing each one.

    Returns (startup_seconds, [epoch_seconds, ...]).

    The first epoch is timed from process start, so it carries CUDA init,
    imports, model build and the dataset check. That is reported separately as
    startup and excluded from the epoch average -- it is a per-run cost paid
    once, not a per-epoch cost, and leaving it in would flatter whichever
    setting happened to have a warm page cache.
    """
    where.mkdir(parents=True, exist_ok=True)
    env = {**os.environ,
           'CUDA_VISIBLE_DEVICES': str(gpu),
           'PYTHONUNBUFFERED': '1'}

    started = time.time()
    marks: list[float] = []

    proc = subprocess.Popen(argv, cwd=str(where), env=env, text=True,
                            bufsize=1, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    tail: list[str] = []
    try:
        for line in proc.stdout:
            tail.append(line.rstrip())
            del tail[:-15]
            if EPOCH_LINE.search(line):
                marks.append(time.time())
                done = len(marks)
                print(f'    epoch {done}/{epochs}', end='\r', flush=True)
                if done >= epochs:
                    break
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()

    print(' ' * 30, end='\r')

    if not marks:
        print('    !! no epoch lines seen. last output:')
        for line in tail:
            print(f'       {line}')
        return None, []

    startup = marks[0] - started
    per_epoch = [b - a for a, b in zip(marks, marks[1:])]
    return startup, per_epoch


def main(argv=None) -> None:
    """Run the comparison.

    From a notebook, pass the options as a list: main(['--rung', 'C0']).
    Left as None in a notebook it would read the kernel's own sys.argv and
    fail, so that case is defended below.
    """
    if argv is None and IN_NOTEBOOK:
        argv = []

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rung', default='A0', help='which rung to time (default A0, the dense one)')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--tier', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=6,
                    help='epochs to log per setting; the first is startup (default 6)')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--keep', action='store_true',
                    help='keep the scratch checkpoints instead of deleting them')
    args = ap.parse_args(argv)

    if args.epochs < 3:
        raise SystemExit('use at least 3 epochs: one is spent on startup and '
                         'a single timed epoch is not a measurement')

    base = real_command(args.rung, args.seed, args.tier)
    base = set_flag(base, '--epochs', str(args.epochs))

    print(f'rung {args.rung}, seed {args.seed}, tier {args.tier}')
    print(f'timing {args.epochs - 1} epochs per setting '
          f'(the first is counted as startup)\n')

    results = []
    for cfg in SETTINGS:
        print(f'>> {cfg["name"]}')
        argv = set_flag(base, '--num_workers', str(cfg['workers']))
        argv += cfg['extra']

        startup, per_epoch = time_one(argv, Path(cfg['where']), args.epochs, args.gpu)
        if not per_epoch:
            results.append((cfg['name'], None, None))
            continue

        median = statistics.median(per_epoch)
        results.append((cfg['name'], startup, median))
        spread = ', '.join(f'{x:.1f}' for x in per_epoch)
        print(f'   startup {startup:5.1f}s   epochs [{spread}]   median {median:.2f}s\n')

    # ---- the table -------------------------------------------------------
    ok = [r for r in results if r[2] is not None]
    print('\n' + '=' * 72)
    print(f'{"setting":<38} {"startup":>9} {"s/epoch":>9} {"vs first":>10}')
    print('-' * 72)
    reference = ok[0][2] if ok else None
    for name, startup, median in results:
        if median is None:
            print(f'{name:<38} {"FAILED":>9}')
            continue
        change = f'{(median / reference - 1) * 100:+.0f}%' if reference else '--'
        print(f'{name:<38} {startup:8.1f}s {median:8.2f}s {change:>10}')
    print('=' * 72)

    if ok:
        best_name, _, best = min(ok, key=lambda r: r[2])
        print(f'\nfastest: {best_name} at {best:.2f}s/epoch')
        if reference and best < reference:
            saved = (reference - best) * 250 * 88.5 / 3600
            print(f'across the whole tier-1 ladder that is roughly '
                  f'{saved:.0f} GPU-hours less than the current setup')
        print('\nNote: this times ONE run alone. Your sweep runs two at a time, '
              'so real per-run epochs will be slower than these -- but the '
              'ranking between settings is what matters here.')

    if not args.keep:
        for cfg in SETTINGS:
            where = Path(cfg['where'])
            if SCRATCH in where.parents or where == SCRATCH:
                shutil.rmtree(where, ignore_errors=True)


if __name__ == '__main__' and not IN_NOTEBOOK:
    main()
elif IN_NOTEBOOK:
    # Don't start a benchmark just because the cell was run. Pasting this in
    # is how you load it; starting it is a separate decision.
    print(f'loaded.  repo    {REPO}')
    print(f'         scratch {SCRATCH}')
    print("\nrun it with:  main()                    # dense rung, default settings")
    print("              main(['--rung', 'C0'])    # a contrastive rung")
