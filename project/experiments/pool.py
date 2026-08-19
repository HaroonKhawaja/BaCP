"""Run ladder cells across several GPUs concurrently.

`runner.run_grid` executes one cell at a time, which leaves seven of eight cards
idle. The cells are independent once the dense rung exists, so the work is
embarrassingly parallel -- but three things have to be got right, and none of
them is optional:

  **Claiming must be atomic.** run_grid's resume check reads the run records,
  which is correct for resuming and racy for concurrency: two workers listing
  cells at the same moment both see a cell as incomplete and both run it. That
  wastes a GPU-hour and writes two records for one cell. Claims here are
  `os.open(..., O_CREAT | O_EXCL)`, which is atomic on POSIX and Windows alike --
  exactly one worker can create a given file.

  **The dense rung is a barrier.** Every sparse cell resolves its starting
  weights from a dense record MATCHING ITS SEED, at execution time. Launch
  everything at once and the sparse cells race the dense ones and die on a
  missing checkpoint -- which is precisely how 21 of 22 cells failed in the first
  tier-0 run. So dense cells run first, to completion, and only then the rest.

  **A crashed worker must not hold its claim forever.** Claims carry a pid and a
  timestamp and are reclaimed after `stale_after` seconds, so a killed process --
  or a reclaimed spot instance -- does not strand a cell.

Concurrency is per GPU: `--per-gpu 2` runs two cells on each card. ResNet-34 at
batch 128 on 32x32 inputs uses roughly 3 GB of 24 GB and nowhere near all the
SMs, so packing 2-3 per card is usually a real throughput win. The limit is host
CPU -- every cell spawns dataloader workers -- so measure before raising it.
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


# --- claims ----------------------------------------------------------------

def _claims_dir(root=None) -> Path:
    from results import results_dir
    d = results_dir(root) / 'claims'
    d.mkdir(parents=True, exist_ok=True)
    return d


def claim(key, root=None, stale_after=7200, worker=''):
    """Try to take exclusive ownership of a cell. True if we got it.

    O_CREAT|O_EXCL is the whole mechanism: the filesystem guarantees that exactly
    one caller creates the file. No lock server, no database, and it works the
    same on the shared volume of a rented box as it does locally.
    """
    path = _claims_dir(root) / f'{key}.claim'
    payload = json.dumps({'pid': os.getpid(), 'worker': worker,
                          'at': time.time()}).encode()
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, 'wb') as fh:
            fh.write(payload)
        return True
    except FileExistsError:
        pass

    # Reclaim a stale claim: a killed worker, or a spot instance that went away.
    try:
        age = time.time() - json.loads(path.read_text()).get('at', 0)
    except Exception:                                              # noqa: BLE001
        age = float('inf')
    if age > stale_after:
        try:
            path.unlink()
            print(f'  [pool] reclaimed stale claim on {key} (age {age/60:.0f} min)')
            return claim(key, root, stale_after, worker)
        except OSError:
            pass
    return False


def release(key, root=None):
    try:
        (_claims_dir(root) / f'{key}.claim').unlink()
    except OSError:
        pass


def clear_claims(root=None) -> int:
    n = 0
    for f in _claims_dir(root).glob('*.claim'):
        try:
            f.unlink()
            n += 1
        except OSError:
            pass
    return n


# --- the pool --------------------------------------------------------------

def usable_cores():
    """Cores this process may actually run on.

    os.cpu_count() reports the HOST's cores inside a container and ignores cgroup
    pinning, so it over-reports badly on a rented box -- 256 where the container
    may have far fewer.
    """
    try:
        return len(os.sched_getaffinity(0))          # Linux, respects cgroups
    except AttributeError:
        return os.cpu_count() or 4


def workers_per_cell(concurrency, cores=None):
    """Dataloader workers for one cell, given how many cells share the machine.

    Leaves two cores per cell for the training process itself and the main
    thread. Without this each cell inherited os.cpu_count(): 256 workers per
    cell, 1,024 processes across four concurrent cells, on 256 cores.
    """
    cores = cores or usable_cores()
    return max(2, cores // max(concurrency, 1) - 2)


def _worker(name, gpu, work, results, root, timeout, echo):
    while True:
        try:
            cell = work.get_nowait()
        except queue.Empty:
            return
        if not claim(cell['key'], root=root, worker=name):
            results.append({'key': cell['key'], 'status': 'claimed_elsewhere'})
            work.task_done()
            continue
        try:
            if echo:
                print(f'  [{name}] -> {cell["rung"]} seed{cell["seed"]}')
            res = R.run_cell(cell, root=root, timeout=timeout, echo=False,
                             env={'CUDA_VISIBLE_DEVICES': str(gpu)})
            results.append(res)
            if echo:
                mark = 'ok' if res['status'].startswith('ok') else res['status'].upper()
                print(f'  [{name}] {mark} {cell["rung"]} seed{cell["seed"]} '
                      f'({res.get("elapsed_s", 0):.0f}s)')
            # A failed cell keeps its claim, so a retry loop does not spin on it.
            # Deleting its record is how you ask for another attempt.
            if res['status'] == 'failed':
                pass
            else:
                release(cell['key'], root=root)
        except Exception as exc:                                   # noqa: BLE001
            results.append({'key': cell['key'], 'status': 'error', 'error': str(exc)})
            release(cell['key'], root=root)
            if echo:
                print(f'  [{name}] ERROR {cell["rung"]}: {type(exc).__name__}: {exc}')
        finally:
            work.task_done()


def run_pool(tier=None, *, gpus=None, per_gpu=1, root=None, timeout=None,
             rungs=None, stages=None, seeds=None, echo=True):
    """Run a tier across GPUs. Dense cells first, then everything else."""
    gpus = list(range(gpus)) if isinstance(gpus, int) else (gpus or [0])

    # Size the dataloaders to the machine and the concurrency, before anything
    # is scheduled. Every cell inherits this.
    nw = workers_per_cell(len(gpus) * per_gpu)
    print(f'  {usable_cores()} usable core(s) -> num_workers={nw} per cell')

    grid = M.cells(tier, rungs=rungs)
    for c in grid:
        c['config'] = dict(c['config'], num_workers=nw)
    if stages:
        grid = [c for c in grid if c['stage'] in set(stages)]
    if seeds is not None:
        grid = [c for c in grid if c['seed'] in set(seeds)]

    done = R.completed_keys(root)
    todo = [c for c in grid if c['key'] not in done]

    # The barrier. Sparse cells resolve a dense checkpoint of the SAME SEED at
    # execution time, so they cannot start until the dense rung has finished.
    dense = [c for c in todo if c['script'] == 'baseline']
    sparse = [c for c in todo if c['script'] != 'baseline']

    print(f'tier {M.TIER if tier is None else tier}: {len(grid)} cells, '
          f'{len(grid) - len(todo)} already done, {len(todo)} to run')
    print(f'  {len(gpus)} GPU(s) x {per_gpu} concurrent = {len(gpus) * per_gpu} workers')

    summary = {'ok': 0, 'failed': 0, 'skipped': len(grid) - len(todo), 'results': []}

    for phase, cells_ in (('dense (barrier)', dense), ('sparse', sparse)):
        if not cells_:
            continue
        print(f'\n== {phase}: {len(cells_)} cell(s)')
        work = queue.Queue()
        for c in cells_:
            work.put(c)
        results, threads = [], []
        for gi, gpu in enumerate(gpus):
            for slot in range(per_gpu):
                name = f'g{gpu}.{slot}'
                t = threading.Thread(target=_worker, daemon=True, args=(
                    name, gpu, work, results, root, timeout, echo))
                t.start()
                threads.append(t)
        for t in threads:
            t.join()
        summary['results'].extend(results)
        summary['ok'] += sum(1 for r in results if r['status'].startswith('ok'))
        summary['failed'] += sum(1 for r in results
                                 if r['status'] in ('failed', 'error'))
        if phase.startswith('dense') and summary['failed']:
            print('!! the dense rung failed; every sparse cell would fail to '
                  'resolve its checkpoint. Stopping.')
            return summary

    print(f"\n{summary['ok']} ok, {summary['failed']} failed, "
          f"{summary['skipped']} skipped")
    for r in summary['results']:
        if r['status'] in ('failed', 'error'):
            print(f"  FAILED {r['key']}: {r.get('error', r.get('returncode'))}")
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--tier', type=int, default=None)
    ap.add_argument('--gpus', type=int, default=None,
                    help='number of GPUs (default: all visible)')
    ap.add_argument('--per-gpu', type=int, default=1,
                    help='concurrent cells per GPU. 2-3 usually pays here; the '
                         'limit is host CPU for dataloaders, so measure.')
    ap.add_argument('--rungs', nargs='*', default=None)
    ap.add_argument('--stages', nargs='*', default=None)
    ap.add_argument('--seeds', type=int, nargs='*', default=None)
    ap.add_argument('--timeout', type=int, default=None)
    ap.add_argument('--clear-claims', action='store_true',
                    help='drop every claim before starting. Use after a crash, '
                         'never while other workers are live.')
    a = ap.parse_args()

    if a.clear_claims:
        print(f'cleared {clear_claims()} claim(s)')

    n_gpus = a.gpus
    if n_gpus is None:
        try:
            import torch
            n_gpus = max(torch.cuda.device_count(), 1)
        except Exception:                                          # noqa: BLE001
            n_gpus = 1
    run_pool(a.tier, gpus=n_gpus, per_gpu=a.per_gpu, timeout=a.timeout,
             rungs=a.rungs, stages=a.stages, seeds=a.seeds)
