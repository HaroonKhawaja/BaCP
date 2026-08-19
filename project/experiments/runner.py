"""Execute manifest cells as subprocesses, idempotently.

Three properties matter here and each is deliberate:

  **Subprocess, not in-process.** Every run gets a fresh interpreter. This is not
  fastidiousness: `pruning_factory` keeps the prunable scope in module-level
  globals, `torch` caches CUDA allocator state, and HuggingFace caches models on
  the module. Running twenty configs in one kernel means config 7 can silently
  inherit config 6's prunable scope -- and since every sparsity number in the
  paper depends on that scope, the corruption would be invisible and total.

  **Idempotent via the record, not a ledger.** A cell is complete iff a run
  record exists carrying its key in `experiment_group`. There is no separate
  progress file to fall out of sync with reality, and deleting a record is
  exactly how you ask for that one cell to be re-run.

  **A failure is a row.** On non-zero exit the runner writes a `status='failed'`
  record. Without it a crashed cell is indistinguishable from one that was never
  scheduled, and the table shows a blank either way -- which is how a missing
  result becomes an invisible result.

The command line is constructed by introspecting the real argparse parsers rather
than by restating the CLI here, so a flag that does not exist fails loudly at
construction instead of being dropped by the shell.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
for _p in (str(_PROJECT), str(_PROJECT / 'scripts')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import manifest as M                                              # noqa: E402


SCRIPTS = {
    'baseline': _PROJECT / 'scripts' / 'baseline_script.py',
    'pruning': _PROJECT / 'scripts' / 'pruning_script.py',
    'bacp': _PROJECT / 'scripts' / 'bacp_script.py',
}

_PARSER_BUILDERS = {
    'baseline': 'build_baseline_parser',
    'pruning': 'build_pruning_parser',
    'bacp': 'build_bacp_parser',
}

_parser_cache: dict[str, dict] = {}


def parser_spec(script: str) -> dict:
    """{dest: {'opt': '--x', 'store_true': bool, 'nargs': n}} for one script."""
    if script in _parser_cache:
        return _parser_cache[script]
    import scripting_utils
    parser = getattr(scripting_utils, _PARSER_BUILDERS[script])()
    spec = {}
    for action in parser._actions:
        if not action.option_strings or action.dest == 'help':
            continue
        spec[action.dest] = {
            'opt': action.option_strings[0],
            'store_true': action.__class__.__name__ == '_StoreTrueAction',
            'nargs': action.nargs,
            'required': action.required,
        }
    _parser_cache[script] = spec
    return spec


def build_command(cell: dict, *, python=None, extra=None) -> list[str]:
    """Manifest config -> argv.

    Raises on any config key the target script's parser does not define. That is
    the desired behaviour: a silently-dropped flag means the run executed a
    different experiment than the manifest says it did, and the record would
    still look clean.
    """
    script = cell['script']
    spec = parser_spec(script)
    cfg = dict(cell['config'])
    cfg['seed'] = cell['seed']

    unknown = sorted(set(cfg) - set(spec))
    if unknown:
        raise KeyError(
            f'manifest rung {cell["rung"]!r} sets {unknown}, which '
            f'{SCRIPTS[script].name} does not accept. Either add the argument to '
            f'the parser or remove it from the rung -- do not let it be dropped.')

    argv = [python or sys.executable, str(SCRIPTS[script])]
    for key, value in cfg.items():
        meta = spec[key]
        if meta['store_true']:
            if value:
                argv.append(meta['opt'])
            continue
        if value is None:
            continue
        argv.append(meta['opt'])
        if isinstance(value, (list, tuple)):
            argv.extend(str(v) for v in value)
        else:
            argv.append(str(value))
    if extra:
        argv.extend(extra)
    return argv


# --- completion ------------------------------------------------------------

def completed_keys(root=None, *, statuses=('ok',)) -> set[str]:
    """Keys of cells that already have a record.

    Reads the JSON records directly rather than runs.csv so a stale index cannot
    cause a finished six-hour run to be repeated.
    """
    from results import iter_records
    out = set()
    for payload in iter_records(root):
        if payload.get('status') in statuses and payload.get('experiment_group'):
            out.add(payload['experiment_group'])
    return out


def is_complete(cell, done=None, root=None) -> bool:
    done = completed_keys(root) if done is None else done
    return cell['key'] in done


# --- checkpoint resolution -------------------------------------------------

def _needs_weights(cell) -> bool:
    return cell['script'] in ('pruning', 'bacp')


def attach_checkpoint(cell, *, root=None, same_seed=True, allow_missing=False):
    """Point a sparse rung at the dense checkpoint it starts from.

    This is what replaces hardcoded /dbfs paths. `same_seed=True` keeps the whole
    ladder chained to the dense run of the *same* seed, so a paired comparison is
    paired all the way down to initialisation rather than only from the pruning
    step onward.
    """
    if not _needs_weights(cell):
        return cell
    if cell['config'].get('trained_weights'):
        return cell

    from results import resolve_checkpoint
    seed = cell['seed'] if same_seed else None
    try:
        path = resolve_checkpoint(cell['model_name'], cell['dataset_name'],
                                  seed=seed, method='dense', root=root)
    except FileNotFoundError:
        try:
            if not same_seed:
                raise
            path = resolve_checkpoint(cell['model_name'], cell['dataset_name'],
                                      seed=None, method='dense', root=root)
            print(f'  ! no dense checkpoint for seed {seed}; falling back to {path}. '
                  f'The pairing is broken for this cell -- it is recorded, but treat '
                  f'the rung as unpaired.')
        except FileNotFoundError:
            if not allow_missing:
                raise
            # Dry-run only. Deliberately not a real-looking path: if this string
            # ever reaches a record, the run was mis-scheduled and should be
            # obvious at a glance rather than quietly loading nothing.
            path = '<PENDING dense checkpoint from rung A0>'
    cell = dict(cell)
    cell['config'] = dict(cell['config'], trained_weights=path)
    return cell


# --- execution -------------------------------------------------------------

def run_cell(cell, *, root=None, dry_run=False, python=None, echo=True,
             timeout=None, env=None):
    """Run one cell. Returns a dict describing what happened."""
    from results import record_failure, results_dir

    out_root = results_dir(root)
    (out_root / 'logs').mkdir(parents=True, exist_ok=True)
    log_path = out_root / 'logs' / f'{cell["key"]}.log'

    # Scheduling errors become rows, exactly like execution errors. Previously
    # a missing dense checkpoint raised FileNotFoundError out of run_cell, past
    # run_grid and past run_ladder.main -- aborting the whole sweep mid-loop with
    # no failure record, no summary, no table and not even a log file, since the
    # log is opened further down. And that is the DEFAULT first run: every sparse
    # cell needs a dense checkpoint that only rung A0 produces.
    try:
        cell = attach_checkpoint(cell, root=root, allow_missing=dry_run)
        argv = build_command(cell, python=python)
    except Exception as exc:                                       # noqa: BLE001
        reason = f'{type(exc).__name__}: {exc}'
        if echo:
            print(f'   NOT SCHEDULED: {reason}')
        if not dry_run:
            record_failure({**cell, 'cmd': [], 'method': cell['rung']},
                           returncode=-1, log_tail=reason,
                           experiment_group=cell['key'], root=root)
        return {'key': cell['key'], 'status': 'failed', 'returncode': -1,
                'elapsed_s': 0.0, 'error': reason, 'cmd': []}

    if dry_run:
        if echo:
            print(' '.join(argv))
        return {'key': cell['key'], 'status': 'dry_run', 'cmd': argv}

    run_env = dict(os.environ)
    # _finalize_run reads this and stamps it onto the record. It is the only
    # link between a manifest cell and its result, so it is set here and nowhere
    # else.
    run_env['BACP_EXPERIMENT_GROUP'] = cell['key']
    run_env.setdefault('PYTHONUNBUFFERED', '1')
    run_env.setdefault('MPLBACKEND', 'Agg')
    if env:
        run_env.update(env)

    t0 = time.perf_counter()
    header = (f'=== {cell["key"]}\n'
              f'=== rung {cell["rung"]} ({cell["label"]}): {cell["changes"]}\n'
              f'=== {" ".join(argv)}\n\n')
    if echo:
        print(f'-> {cell["key"]}  [{cell["rung"]}] {cell["label"]}')

    with log_path.open('w', encoding='utf-8', errors='replace') as log:
        log.write(header)
        log.flush()
        try:
            proc = subprocess.run(
                argv, cwd=str(_PROJECT / 'scripts'), env=run_env,
                stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            log.write(f'\n[RUNNER] timed out after {timeout}s\n')
            rc = -9

    elapsed = round(time.perf_counter() - t0, 1)

    if rc != 0:
        tail = ''
        try:
            tail = log_path.read_text(encoding='utf-8', errors='replace')[-4000:]
        except Exception:                                          # noqa: BLE001
            pass
        record_failure({**cell, 'cmd': argv, 'method': cell['rung']},
                       returncode=rc, log_tail=tail,
                       experiment_group=cell['key'], root=root)
        if echo:
            print(f'   FAILED rc={rc} after {elapsed}s -- see {log_path}')
        return {'key': cell['key'], 'status': 'failed', 'returncode': rc,
                'elapsed_s': elapsed, 'log': str(log_path), 'cmd': argv}

    # The script exited 0, but that is not the same as "a record exists". If
    # _finalize_run's own try/except swallowed something, the cell would be
    # silently re-run forever on the next resume. Say so once, here.
    wrote = cell['key'] in completed_keys(root)
    if echo:
        print(f'   ok in {elapsed}s' + ('' if wrote else '  [!] exited 0 but wrote no record'))
    return {'key': cell['key'], 'status': 'ok' if wrote else 'ok_no_record',
            'returncode': 0, 'elapsed_s': elapsed, 'log': str(log_path), 'cmd': argv}


def run_grid(tier=None, *, rungs=None, stages=None, priorities=None, seeds=None,
             root=None, dry_run=False, resume=True, python=None, timeout=None,
             stop_on_fail=False, echo=True):
    """Run every cell in a tier that is not already complete.

    Returns a summary dict. Ordering is by stage then rung then seed, which is
    also the dependency order: dense before sparse, and the gate rungs (C-null,
    C0) before anything that is measured against them.
    """
    grid = M.cells(tier, rungs=rungs)
    if stages:
        grid = [c for c in grid if c['stage'] in set(stages)]
    if priorities:
        grid = [c for c in grid if c['priority'] in set(priorities)]
    if seeds is not None:
        grid = [c for c in grid if c['seed'] in set(seeds)]

    order = {s: i for i, s in enumerate(['A', 'B', 'C', 'D', 'D4'])}
    grid.sort(key=lambda c: (order.get(c['stage'], 99), c['rung'], c['seed']))

    done = completed_keys(root) if resume else set()
    todo = [c for c in grid if c['key'] not in done]

    if echo:
        print(f'Tier {M.TIER if tier is None else tier}: {len(grid)} cells, '
              f'{len(grid) - len(todo)} already recorded, {len(todo)} to run.')

    results, failures = [], []
    for i, cell in enumerate(todo, 1):
        if echo:
            print(f'[{i}/{len(todo)}]', end=' ')
        res = run_cell(cell, root=root, dry_run=dry_run, python=python,
                       timeout=timeout, echo=echo)
        results.append(res)
        if res['status'] == 'failed':
            failures.append(res)
            if stop_on_fail:
                print('stopping: stop_on_fail is set')
                break

    summary = {
        'tier': M.TIER if tier is None else tier,
        'planned': len(grid),
        'skipped': len(grid) - len(todo),
        'attempted': len(results),
        'ok': sum(1 for r in results if r['status'] == 'ok'),
        'ok_no_record': sum(1 for r in results if r['status'] == 'ok_no_record'),
        'failed': len(failures),
        'failures': [f['key'] for f in failures],
        'results': results,
    }
    if echo:
        print(f"\n{summary['ok']} ok, {summary['failed']} failed, "
              f"{summary['skipped']} skipped.")
        for key in summary['failures']:
            print(f'  FAILED  {key}')
    return summary


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Run manifest cells.')
    ap.add_argument('--tier', type=int, default=None)
    ap.add_argument('--rungs', nargs='*', default=None)
    ap.add_argument('--stages', nargs='*', default=None)
    ap.add_argument('--seeds', type=int, nargs='*', default=None)
    ap.add_argument('--dry_run', action='store_true')
    ap.add_argument('--no_resume', action='store_true')
    ap.add_argument('--stop_on_fail', action='store_true')
    ap.add_argument('--timeout', type=int, default=None)
    a = ap.parse_args()
    run_grid(a.tier, rungs=a.rungs, stages=a.stages, seeds=a.seeds,
             dry_run=a.dry_run, resume=not a.no_resume,
             stop_on_fail=a.stop_on_fail, timeout=a.timeout)
