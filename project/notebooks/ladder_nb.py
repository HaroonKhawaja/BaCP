"""Shared helpers for the ladder notebooks.

This file lives under `project/notebooks/`, which `results.code_fingerprint`
excludes from its hash. That is deliberate and load-bearing: the notebooks are an
*interface onto* the experiment, not part of it, so editing them must never
invalidate a run record or split the fingerprint mid-sweep.

Everything here is thin over `manifest`, `runner` and `pool`. Nothing reimplements
training, and nothing special-cases the notebook path -- `watch()` runs the exact
argv that `pool.py` would run, so what you see in a cell is what the sweep does.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

# --- environment -----------------------------------------------------------

REPO: Path | None = None
PY: str = sys.executable


def setup(repo=None, results=None, quiet=False) -> dict:
    """Put the project on sys.path, set the env, and report what we're running on.

    Call this first in every notebook. Returns a dict of what it resolved.

    PYTHONUNBUFFERED is the one non-obvious line. Without it a child process
    buffers stdout and all 250 epochs arrive in a single lump when the run ends,
    which defeats the entire purpose of watching a run live.
    """
    global REPO, PY

    REPO = Path(repo or os.environ.get('BACP_REPO')
                or ('/workspace/BaCP' if Path('/workspace/BaCP').exists()
                    else Path(__file__).resolve().parents[2]))
    os.environ['BACP_REPO'] = str(REPO)

    default_results = ('/workspace/bacp_results' if Path('/workspace').is_dir()
                       else str(REPO / 'results'))
    os.environ['BACP_RESULTS_DIR'] = str(results or
                                         os.environ.get('BACP_RESULTS_DIR')
                                         or default_results)
    os.environ['PYTHONUNBUFFERED'] = '1'

    for sub in ('project', 'project/experiments', 'project/scripts'):
        p = str(REPO / sub)
        if p not in sys.path:
            sys.path.insert(0, p)

    PY = sys.executable
    info = {'repo': REPO, 'results': Path(os.environ['BACP_RESULTS_DIR']),
            'python': PY}

    try:
        import torch
        info['torch'] = torch.__version__
        info['cuda'] = torch.version.cuda
        info['gpus'] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if info['gpus']:
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['sm'] = ''.join(map(str, torch.cuda.get_device_capability(0)))
    except Exception as exc:                                       # noqa: BLE001
        info['torch_error'] = f'{type(exc).__name__}: {exc}'

    runs = info['results'] / 'runs'
    info['records'] = len(list(runs.glob('*.json'))) if runs.is_dir() else 0

    if not quiet:
        print(f'repo      {info["repo"]}')
        print(f'results   {info["results"]}')
        print(f'python    {info["python"]}')
        if 'torch_error' in info:
            print(f'torch     !! {info["torch_error"]}')
            print('          the kernel has no torch. In VS Code pick '
                  '/venv/main/bin/python as the kernel.')
        else:
            print(f'torch     {info["torch"]}  cuda {info["cuda"]}')
            if info['gpus']:
                print(f'gpus      {info["gpus"]} x {info["gpu_name"]}  sm_{info["sm"]}')
            else:
                print('gpus      !! NONE VISIBLE -- do not start a run')
        print(f'records   {info["records"]}')
    return info


def _mods():
    import manifest as M
    import runner as R
    return M, R


# --- selecting work --------------------------------------------------------

def pick(rung, seed=1, tier=1, *, suffix='', **overrides):
    """One cell, with optional config overrides.

    Any override makes the run a DIAGNOSTIC rather than a ladder measurement, so
    pass `suffix` to keep it out of the ladder's key space -- otherwise its record
    satisfies the real cell and the sweep will skip the real one.
    """
    M, R = _mods()
    matches = [c for c in M.cells(tier, rungs=[rung]) if c['seed'] == seed]
    if not matches:
        raise KeyError(f'rung {rung!r} seed {seed} is not in tier {tier}. '
                       f'tier {tier} has rungs '
                       f'{sorted({c["rung"] for c in M.cells(tier)})}')
    cell = matches[0]
    cell['config'].setdefault('num_workers', 8)
    for k, v in overrides.items():
        if v is not None:
            cell['config'][k] = v
    if suffix:
        cell['key'] += suffix
    elif overrides:
        print('!! overrides set with no suffix: this record will satisfy the '
              'real ladder cell. Pass suffix=".probe" unless that is intended.')
    return cell


def attach(cell, quiet=False):
    """Resolve the dense checkpoint of the SAME seed. Returns the cell either way."""
    _, R = _mods()
    try:
        cell = R.attach_checkpoint(cell)
        if not quiet:
            print(f'checkpoint  {cell["config"].get("trained_weights")}')
    except Exception as exc:                                       # noqa: BLE001
        if not quiet:
            print(f'checkpoint  NOT RESOLVED -- {type(exc).__name__}: {exc}')
            print('            expected for A0; every other rung needs A0 at '
                  'this seed first.')
    return cell


def show(cell):
    print(f'key       {cell["key"]}')
    print(f'rung      {cell["rung"]}  ({cell["changes"]})')
    print(f'script    {cell["script"]}   seed {cell["seed"]}')
    print('config:')
    for k, v in sorted(cell['config'].items()):
        print(f'  {k:24s} {v}')


# --- watching one run ------------------------------------------------------

_EPOCH = re.compile(r'Epoch \[(\d+)/(\d+)\]')


def _field(name, line):
    # \b keeps 'loss:' from matching inside 'val_loss:' (underscore is a word
    # char, so there is no boundary there), and IGNORECASE lets the same name
    # hit both trainer formats: 'loss:'/'accuracy:' from Trainer and
    # 'Total Loss:'/'Training Accuracy:' from BaCPTrainer.
    m = re.search(rf'\b{name}: (nan|-?[0-9.]+(?:[eE][-+]?\d+)?)', line,
                  re.IGNORECASE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return float('nan')


def watch(cell, gpu=0, echo_other=True):
    """Run one cell as a subprocess, streaming one line per epoch.

    Returns (history, first_nan_epoch). Interrupting the kernel terminates the
    child rather than orphaning it.
    """
    _, R = _mods()
    argv = R.build_command(cell, python=PY)
    env = {**os.environ,
           'CUDA_VISIBLE_DEVICES': str(gpu),
           'BACP_EXPERIMENT_GROUP': cell['key']}

    hist = {'epoch': [], 'acc': [], 'loss': [], 'sparsity': [], 'val_loss': []}
    first_nan, t0 = None, time.time()

    print(' '.join(argv[1:]), '\n')
    proc = subprocess.Popen(argv, cwd=str(REPO / 'project' / 'scripts'),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, env=env)
    try:
        for raw in proc.stdout:
            line = raw.rstrip()
            m = _EPOCH.search(line)
            if not m:
                if echo_other and line.strip():
                    print(line)
                continue
            ep, tot = int(m.group(1)), int(m.group(2))
            acc, loss, sp = (_field('accuracy', line), _field('loss', line),
                             _field('sparsity', line))
            hist['epoch'].append(ep)
            hist['acc'].append(acc)
            hist['loss'].append(loss)
            hist['sparsity'].append(sp)
            hist['val_loss'].append(_field('val_loss', line))

            if loss is not None and loss != loss and first_nan is None:
                first_nan = ep
                print(f'\n  *** LOSS WENT NaN AT EPOCH {ep}/{tot} ***\n')

            per = (time.time() - t0) / max(ep, 1)
            acc_s = '   None' if acc is None else f'{acc:7.4f}'
            print(f'epoch {ep:4d}/{tot}  acc {acc_s}  loss {loss}  '
                  f'sparsity {sp}  [{per:5.1f}s/ep  eta {per*(tot-ep)/60:5.1f}m]')
    except KeyboardInterrupt:
        proc.terminate()
        print('\ninterrupted; child terminated')
    finally:
        proc.wait()

    print(f'\nexit {proc.returncode} after {(time.time()-t0)/60:.1f} min'
          + (f'   FIRST NaN AT EPOCH {first_nan}' if first_nan else ''))
    return hist, first_nan


def plot(hist, first_nan=None, title=''):
    """Accuracy, finite loss, and sparsity. The dashed line marks the first NaN."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.4))
    ax[0].plot(hist['epoch'], hist['acc'], lw=1)
    ax[0].set_title('val accuracy (%)')
    ax[0].axhline(10.0, color='grey', ls=':', lw=1)      # chance on CIFAR-10
    finite = [(e, l) for e, l in zip(hist['epoch'], hist['loss'])
              if l is not None and l == l]
    ax[1].plot([e for e, _ in finite], [l for _, l in finite], lw=1)
    ax[1].set_title('train loss (finite only)')
    ax[2].plot(hist['epoch'], hist['sparsity'], lw=1)
    ax[2].set_title('sparsity')
    for a in ax:
        a.set_xlabel('epoch')
        if first_nan:
            a.axvline(first_nan, color='crimson', ls='--', lw=1)
    fig.suptitle(title, fontsize=9)
    plt.tight_layout()
    plt.show()


# --- running a whole stage -------------------------------------------------

def run_stage(rungs, tier=1, gpus=None, seeds=None):
    """Run a list of rungs across the GPUs, in process, printing as it goes.

    Dense cells run first as a hard barrier -- every sparse cell resolves a dense
    checkpoint of its own seed at execution time, so launching them together is
    how 21 of 22 cells failed on the first attempt. The sparse cells then run one
    seed at a time, so an interrupted sweep leaves whole seeds behind.

    Cells per GPU and dataloader workers are constants in `pool.py`; edit them
    there rather than passing them in.

    Idempotent: a cell counts as complete iff a record carrying its key exists,
    so re-running this skips what is done and picks up what is not.
    """
    import pool as P
    return P.run_pool(tier, gpus=gpus, rungs=list(rungs), seeds=seeds)


def progress(tier=1):
    """A live view of every log on disk: epochs done, last accuracy, NaN or not."""
    import pandas as pd
    logs = Path(os.environ['BACP_RESULTS_DIR']) / 'logs'
    rows = []
    for f in sorted(logs.glob('*.log')) if logs.is_dir() else []:
        text = f.read_text(errors='ignore')
        eps = _EPOCH.findall(text)
        if not eps:
            continue
        done, total = int(eps[-1][0]), int(eps[-1][1])
        last = [ln for ln in text.splitlines() if _EPOCH.search(ln)][-1]
        rows.append({'cell': f.stem, 'epoch': done, 'of': total,
                     'pct': round(100 * done / max(total, 1)),
                     'acc': _field('accuracy', last),
                     'loss': _field('loss', last),
                     'nan': 'nan' in text.lower()})
    return pd.DataFrame(rows).sort_values('cell') if rows else pd.DataFrame()


def report(tier=1):
    """The aggregate table, the gates, and the written .csv/.tex/.txt.

    Pass the tier explicitly. `ladder.report()` otherwise falls back to
    BACP_TIER, which defaults to 0, and tier 0 uses `smoke.*` keys that match
    none of the real records -- you get an empty table and no error.
    """
    import importlib
    import ladder
    importlib.reload(ladder)
    return ladder.report(tier)
