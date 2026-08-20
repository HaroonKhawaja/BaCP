"""Run ladder cells on every GPU at once.

A "cell" is one rung at one seed -- a single training run. Cells are
independent, so we run several at the same time, one per GPU.

Three rules keep that safe, and each has a function below:

  Dense cells go first. Every sparse cell starts from a dense checkpoint of its
  own seed, so the dense rung has to be finished before any sparse cell begins.
  See run_dense_cells().

  Seeds finish one at a time. All of seed 0, then all of seed 1, and so on.
  Stop the sweep halfway and you have whole seeds you can average, instead of a
  slice of every seed and nothing you can report. See run_sparse_cells().

  Two workers never take the same cell. A worker creates a claim file before it
  starts; whoever creates the file first owns the cell. See try_claim().

Stopping and restarting is safe. A cell counts as done once it has a run
record, so restarting picks up where you left off.

All settings are the constants directly below. Change them here, not on the
command line.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent), str(_HERE.parent / 'scripts')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import manifest as M                                              # noqa: E402
import runner as R                                                # noqa: E402


# --- settings --------------------------------------------------------------

# Dataloader worker processes for each cell. This is PER CELL, so the machine
# runs NUM_WORKERS x CELLS_PER_GPU x (number of GPUs) worker processes in total.
NUM_WORKERS = 24

# How many cells share one GPU. ResNet-34 at batch 128 on 32x32 inputs uses
# about 3 GB of 24 GB, so a card can hold more than one -- but every extra cell
# also multiplies the worker processes above.
CELLS_PER_GPU = 1

# A claim older than this is treated as dead and taken over. Covers a crashed
# worker or a reclaimed spot instance.
CLAIM_EXPIRES_AFTER = 2 * 60 * 60


# --- claims ----------------------------------------------------------------

def claim_path(key):
    """The claim file for one cell."""
    from results import results_dir
    folder = results_dir() / 'claims'
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f'{key}.claim'


def try_claim(key, worker):
    """Take exclusive ownership of a cell. True if we got it, False if taken.

    O_CREAT | O_EXCL is the whole mechanism: the filesystem guarantees that
    exactly one caller creates the file. No lock server and no database, and it
    behaves the same on a rented box's shared volume as it does locally.
    """
    path = claim_path(key)
    note = json.dumps({'pid': os.getpid(), 'worker': worker, 'at': time.time()})
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, 'w') as fh:
            fh.write(note)
        return True
    except FileExistsError:
        pass

    if claim_is_stale(path):
        print(f'  [pool] taking over a stale claim on {key}')
        release_claim(key)
        return try_claim(key, worker)
    return False


def claim_is_stale(path):
    """True if a claim file is older than CLAIM_EXPIRES_AFTER."""
    try:
        made_at = json.loads(path.read_text()).get('at', 0)
    except Exception:                                              # noqa: BLE001
        return True                       # unreadable claim: assume abandoned
    return time.time() - made_at > CLAIM_EXPIRES_AFTER


def release_claim(key):
    """Give up ownership of a cell so it can be run again."""
    try:
        claim_path(key).unlink()
    except OSError:
        pass


def clear_all_claims():
    """Delete every claim file. Returns how many. Use after a crash, and never
    while other workers are still running."""
    folder = claim_path('x').parent
    count = 0
    for f in folder.glob('*.claim'):
        try:
            f.unlink()
            count += 1
        except OSError:
            pass
    return count


# --- running one cell ------------------------------------------------------

def count_gpus():
    """How many GPUs are visible. 1 if torch cannot tell us."""
    try:
        import torch
        return max(torch.cuda.device_count(), 1)
    except Exception:                                              # noqa: BLE001
        return 1


def run_one_cell(cell, gpu, worker):
    """Claim a cell, train it on `gpu`, and return a result dict."""
    if not try_claim(cell['key'], worker):
        return {'key': cell['key'], 'status': 'claimed_elsewhere'}

    print(f'  [{worker}] start {cell["rung"]} seed{cell["seed"]}')
    try:
        result = R.run_cell(cell, echo=False,
                            env={'CUDA_VISIBLE_DEVICES': str(gpu)})
    except Exception as exc:                                       # noqa: BLE001
        release_claim(cell['key'])
        print(f'  [{worker}] ERROR {cell["rung"]}: {type(exc).__name__}: {exc}')
        return {'key': cell['key'], 'status': 'error', 'error': str(exc)}

    label = 'ok' if result['status'].startswith('ok') else result['status'].upper()
    print(f'  [{worker}] {label} {cell["rung"]} seed{cell["seed"]} '
          f'({result.get("elapsed_s", 0):.0f}s)')

    # A failed cell keeps its claim so a restart does not spin on it forever.
    # Delete its record when you want another attempt.
    if not result['status'] == 'failed':
        release_claim(cell['key'])
    return result


# --- running many cells ----------------------------------------------------

def run_cells_together(cells, gpus, label):
    """Run `cells` in parallel across `gpus` and wait for all of them.

    One thread per GPU slot. Each thread pulls the next cell off a shared queue
    until the queue is empty, so a slow cell does not hold up the others.
    """
    print(f'\n== {label}: {len(cells)} cell(s)')
    pending = queue.Queue()
    for cell in cells:
        pending.put(cell)

    results = []
    lock = threading.Lock()

    def work_until_empty(gpu, worker):
        while True:
            try:
                cell = pending.get_nowait()
            except queue.Empty:
                return
            result = run_one_cell(cell, gpu, worker)
            with lock:
                results.append(result)

    threads = []
    for gpu in gpus:
        for slot in range(CELLS_PER_GPU):
            worker = f'g{gpu}.{slot}'
            t = threading.Thread(target=work_until_empty, args=(gpu, worker),
                                 daemon=True)
            t.start()
            threads.append(t)
    for t in threads:
        t.join()
    return results


def run_dense_cells(cells, gpus):
    """Run every dense cell, on all GPUs at once.

    This is a barrier: sparse cells load a dense checkpoint of their own seed,
    so nothing sparse may start until this finishes. Running all seeds together
    here keeps the whole box busy; doing it seed by seed would idle every card
    but one at each barrier.
    """
    return run_cells_together(cells, gpus, 'dense (barrier, all seeds)')


def run_sparse_cells(cells, gpus):
    """Run every sparse cell, one seed at a time.

    Seed 0 finishes completely before seed 1 starts. If the sweep is cut short
    you are left with a few complete seeds -- which you can average and report
    -- rather than every seed part-finished, which you cannot.
    """
    results = []
    seeds = sorted({c['seed'] for c in cells})
    for position, seed in enumerate(seeds, start=1):
        this_seed = [c for c in cells if c['seed'] == seed]
        results += run_cells_together(
            this_seed, gpus, f'sparse seed {seed}  [{position} of {len(seeds)}]')
        print(f'-- seed {seed} finished. Safe to stop here.')
    return results


# --- the sweep -------------------------------------------------------------

def cells_to_run(tier, rungs=None, stages=None, seeds=None):
    """The cells of a tier that do not have a record yet, ready to run."""
    cells = M.cells(tier, rungs=rungs)
    if stages:
        cells = [c for c in cells if c['stage'] in set(stages)]
    if seeds is not None:
        cells = [c for c in cells if c['seed'] in set(seeds)]
    for c in cells:
        c['config'] = dict(c['config'], num_workers=NUM_WORKERS)

    already_done = R.completed_keys()
    todo = [c for c in cells if c['key'] not in already_done]
    return cells, todo


def count_statuses(results):
    """How many results are ok, and how many failed."""
    ok = sum(1 for r in results if r['status'].startswith('ok'))
    failed = sum(1 for r in results if r['status'] in ('failed', 'error'))
    return ok, failed


def run_pool(tier=None, gpus=None, rungs=None, stages=None, seeds=None):
    """Run a tier: dense cells first, then the sparse cells one seed at a time.

    Returns {'ok': n, 'failed': n, 'skipped': n, 'results': [...]} -- the shape
    the stage notebooks print.
    """
    gpus = list(range(gpus if gpus else count_gpus()))
    slots = len(gpus) * CELLS_PER_GPU

    all_cells, todo = cells_to_run(tier, rungs, stages, seeds)
    dense = [c for c in todo if c['script'] == 'baseline']
    sparse = [c for c in todo if c['script'] != 'baseline']
    skipped = len(all_cells) - len(todo)

    print(f'tier {M.TIER if tier is None else tier}: {len(all_cells)} cells, '
          f'{skipped} already done, {len(todo)} to run')
    print(f'  {len(gpus)} GPU(s) x {CELLS_PER_GPU} = {slots} cells at once, '
          f'{NUM_WORKERS} dataloader workers each')
    print(f'  order: all dense cells, then one seed at a time')

    def summarize(results):
        ok, failed = count_statuses(results)
        return {'ok': ok, 'failed': failed, 'skipped': skipped,
                'results': results}

    results = []
    if dense:
        results += run_dense_cells(dense, gpus)
        if count_statuses(results)[1]:
            print('!! a dense cell failed. Every sparse cell would then fail to '
                  'find its checkpoint, so stopping here.')
            return summarize(results)
    if sparse:
        results += run_sparse_cells(sparse, gpus)

    summary = summarize(results)
    print(f"\n{summary['ok']} ok, {summary['failed']} failed, {skipped} skipped")
    for r in results:
        if r['status'] in ('failed', 'error'):
            print(f"  FAILED {r['key']}: {r.get('error', r.get('returncode'))}")
    return summary


def main():
    ap = argparse.ArgumentParser(
        description='Run ladder cells on every GPU at once.')
    ap.add_argument('--tier', type=int, default=None)
    ap.add_argument('--gpus', type=int, default=None,
                    help='how many GPUs to use (default: all of them)')
    ap.add_argument('--rungs', nargs='*', default=None)
    ap.add_argument('--stages', nargs='*', default=None)
    ap.add_argument('--seeds', type=int, nargs='*', default=None)
    ap.add_argument('--clear-claims', action='store_true',
                    help='delete every claim before starting. Use after a '
                         'crash, never while other workers are live.')
    a = ap.parse_args()

    if a.clear_claims:
        print(f'cleared {clear_all_claims()} claim(s)')

    run_pool(a.tier, gpus=a.gpus, rungs=a.rungs, stages=a.stages, seeds=a.seeds)


if __name__ == '__main__':
    main()
