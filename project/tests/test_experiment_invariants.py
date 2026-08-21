"""Assertions against the results, not against the code.

Everything else in this suite proves the code does what it was written to do.
That is necessary and it is not sufficient: the entire suite would pass on a
method that has no effect whatsoever, and some of it deliberately pins known
defects so nobody "fixes" one silently and invalidates a comparison.

These are different. They run against `results/runs/*.json` and ask whether the
numbers a table would print are comparable to each other and to the literature:
did the run hit the sparsity it claims, is the sparsity denominator the same in
both arms, did the two arms share a protocol, is the seed recorded, was any of
the test set silently dropped. No unit test can substitute for them, because
none of these questions is about code.

Marked `results` and excluded from the default run, since they are vacuous until
a sweep has produced records.

    pytest -m results -s

The `-s` matters: the last test is a non-asserting ledger that prints run status,
so the command doubles as a dashboard.
"""

import json
import math
import sys
from pathlib import Path

import pytest

_PROJECT = Path(__file__).resolve().parents[1]
for _p in (str(_PROJECT),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytestmark = pytest.mark.results


# --- fixtures --------------------------------------------------------------

@pytest.fixture(scope='module')
def records():
    """Every record on disk. Skips the whole module when there are none."""
    from results import iter_records, results_dir
    root = results_dir()
    if not (root / 'runs').exists():
        pytest.skip(f'no records under {root}; run the ladder first')
    payloads = list(iter_records(root))
    if not payloads:
        pytest.skip('no records to check')
    return payloads


@pytest.fixture(scope='module')
def ladder_records(records):
    """Records produced by the manifest, i.e. carrying an experiment_group.

    Ad-hoc runs and the test suite's own fixtures also write records; those are
    not part of any comparison and must not be held to comparability invariants.
    """
    rows = [r for r in records
            if r.get('experiment_group') and r.get('status') == 'ok']
    if not rows:
        pytest.skip('no manifest-produced records yet')
    return rows


@pytest.fixture(scope='module')
def real_records(ladder_records):
    """Ladder records that are actual measurements, excluding tier-0 smoke."""
    rows = [r for r in ladder_records if not r.get('is_smoke')]
    if not rows:
        pytest.skip('only smoke records present; nothing here is a measurement')
    return rows


def _ids(rows):
    return [r.get('record_id', '?') for r in rows]


# --- schema and provenance -------------------------------------------------

def test_schema_version_matches(records):
    from results import SCHEMA_VERSION
    wrong = [r['record_id'] for r in records
             if r.get('schema_version') != SCHEMA_VERSION]
    assert not wrong, (
        f'{len(wrong)} record(s) predate schema v{SCHEMA_VERSION}. They may be '
        f'missing fields a table reads. Regenerate or exclude them: {wrong[:5]}')


def test_every_run_records_its_seed(ladder_records):
    """The seed used to be popped from args before the dataclass was built, so it
    reached nothing and no published number was reproducible after the fact."""
    bad = [r['record_id'] for r in ladder_records
           if r.get('seed') is None or r.get('seed_source') == 'unknown']
    assert not bad, f'seed missing or un-attributable: {bad[:5]}'


def test_provenance_is_recoverable(ladder_records):
    """A HEAD sha does not identify the code that ran when the tree is dirty --
    and it is routinely dirty during active work. A dirty run must carry the
    hash of its diff."""
    bad = [r['record_id'] for r in ladder_records
           if r.get('git_dirty') and not r.get('git_patch_sha1')]
    assert not bad, f'dirty runs with no patch hash: {bad[:5]}'

    missing = [r['record_id'] for r in ladder_records if not r.get('code_fingerprint')]
    assert not missing, f'no code fingerprint: {missing[:5]}'


def test_all_comparable_runs_share_one_code_fingerprint(real_records):
    """Two arms built from different source cannot be compared.

    A soft check by design -- it names the split rather than silently tolerating
    it, because mid-sweep code changes are the most common way a ladder quietly
    stops being a ladder.
    """
    prints = {}
    for r in real_records:
        prints.setdefault(r['code_fingerprint'], []).append(r['record_id'])
    assert len(prints) == 1, (
        'records span %d different code versions:\n%s' % (
            len(prints),
            '\n'.join(f'  {k[:12]}: {len(v)} run(s), e.g. {v[0]}'
                      for k, v in prints.items())))


# --- evaluation correctness ------------------------------------------------

def test_no_evaluation_samples_were_dropped(ladder_records):
    """drop_last=True on the eval loaders silently scored 9,728 of CIFAR-10's
    10,000 test images -- and *which* 272 depended on the batch size. Every
    headline number in the submitted paper carries that error."""
    bad = [(r['record_id'], r['eval_samples_dropped']) for r in ladder_records
           if r.get('eval_samples_dropped')]
    assert not bad, f'test samples never evaluated: {bad[:5]}'


@pytest.mark.parametrize('field,lo,hi', [
    ('test_acc_exact_pct', 0.0, 100.0),
    ('test_acc_pct', 0.0, 100.0),
    ('sparsity_mask', 0.0, 1.0),
    ('sparsity_value', 0.0, 1.0),
])
def test_metrics_are_in_range(ladder_records, field, lo, hi):
    bad = []
    for r in ladder_records:
        v = r.get(field)
        if v is None:
            continue
        if isinstance(v, float) and math.isnan(v):
            bad.append((r['record_id'], 'nan'))
        elif not (lo <= float(v) <= hi):
            bad.append((r['record_id'], v))
    assert not bad, f'{field} out of [{lo}, {hi}]: {bad[:5]}'


def test_exact_and_legacy_accuracy_agree(ladder_records):
    """They should differ only through the dropped-sample and batch-mean effects.

    A large gap means one of the two is measuring something else -- most likely
    that `evaluate_exact` and `evaluate` are looking at different loaders.
    """
    bad = []
    for r in ladder_records:
        a, b = r.get('test_acc_exact_pct'), r.get('test_acc_pct')
        if a is None or b is None:
            continue
        if abs(float(a) - float(b)) > 2.0:
            bad.append((r['record_id'], round(float(a), 2), round(float(b), 2)))
    assert not bad, f'exact vs legacy accuracy disagree by >2pp: {bad[:5]}'


# --- sparsity correctness --------------------------------------------------

_DYNAMIC = {'rigl', 'east'}


def test_sparsity_target_was_attained(ladder_records):
    """EAST is carved out on purpose: its cyclic schedule oscillates the density
    within a run by design, so the terminal mask legitimately sits off target.
    Every monotone pruner has no such excuse."""
    bad = []
    for r in ladder_records:
        # Compare against the target the pruner was actually given. Weight
        # sharing rewrites it against the UNIQUE parameter count, so a sharing
        # run legitimately lands on the adjusted figure and comparing it to the
        # requested one reports a violation that is not one.
        want = r.get('sparsity_target_adjusted') or r.get('sparsity_requested')
        got = r.get('sparsity_mask')
        if want is None or got is None:
            continue
        tol = 0.05 if r.get('pruner') == 'east' else 0.01
        if abs(float(got) - float(want)) > tol:
            bad.append((r['record_id'], r.get('pruner'),
                        round(float(want), 4), round(float(got), 4)))
    assert not bad, f'requested vs achieved sparsity (pruner, want, got): {bad[:5]}'


def test_value_sparsity_never_understates_mask_sparsity(ladder_records):
    """Under dynamic sparse training the two legitimately diverge: RigL and EAST
    zero-initialise regrown connections, so a structurally *active* weight reads
    as numerically zero and the value measure OVERSTATES sparsity -- measured at
    0.9277 against a true 0.8976. Overstating is the flattering direction, so the
    inequality must hold in exactly one direction."""
    bad = []
    for r in ladder_records:
        v, m = r.get('sparsity_value'), r.get('sparsity_mask')
        if v is None or m is None:
            continue
        if float(v) < float(m) - 1e-4:
            bad.append((r['record_id'], round(float(v), 4), round(float(m), 4)))
    assert not bad, f'value sparsity below mask sparsity: {bad[:5]}'


def test_mask_covers_the_prunable_set(ladder_records):
    """If the mask spans a different parameter set than `layer_check` calls
    prunable, the reported sparsity describes neither."""
    bad = [r['record_id'] for r in ladder_records
           if r.get('mask_total') is not None and r.get('mask_covers_prunable') is False]
    assert not bad, f'mask does not cover the prunable set: {bad[:5]}'


def test_active_parameter_count_matches_the_sparsity(ladder_records):
    bad = []
    for r in ladder_records:
        n, active, s = (r.get('prunable_params'), r.get('active_prunable_params'),
                        r.get('sparsity_mask'))
        if not n or active is None or s is None:
            continue
        implied = 1.0 - active / n
        if abs(implied - float(s)) > 0.02:
            bad.append((r['record_id'], round(implied, 4), round(float(s), 4)))
    assert not bad, f'active count implies a different sparsity: {bad[:5]}'


# --- comparability ---------------------------------------------------------

def _by_rung(rows):
    out = {}
    for r in rows:
        # key format: t{tier}.{rung}.{model}.{dataset}.s{sparsity}.seed{n}
        parts = (r.get('experiment_group') or '').split('.')
        rung = parts[1] if len(parts) > 1 else '?'
        out.setdefault(rung, []).append(r)
    return out


@pytest.mark.parametrize('field', ['scope_key', 'val_split_seed', 'model_name',
                                   'dataset_name'])
def test_field_is_single_valued_across_the_whole_ladder(real_records, field):
    """Three sparsity denominators give three numbers for one checkpoint
    (backbone / +head / +embeddings). If `scope_key` varies across rungs, the
    column headed "sparsity" means different things in different rows."""
    values = {}
    for r in real_records:
        values.setdefault(r.get(field), []).append(r['record_id'])
    assert len(values) == 1, (
        f'{field} varies across the ladder: ' +
        '; '.join(f'{k!r} in {len(v)} run(s)' for k, v in values.items()))


@pytest.mark.parametrize('field', ['epochs', 'batch_size', 'learning_rate',
                                   'optimizer_type', 'scheduler_type'])
def test_protocol_is_identical_across_the_ladder(real_records, field):
    """Turns "at matched protocol" from a sentence in the paper into a fact.

    If epochs or learning rate drift between rungs, every delta in the table is
    confounded with a training-budget difference.
    """
    values = {}
    for r in real_records:
        values.setdefault(r.get(field), []).append(r['record_id'])
    assert len(values) == 1, (
        f'{field} varies across the ladder: ' +
        '; '.join(f'{k!r}: {len(v)} run(s) e.g. {v[0]}' for k, v in values.items()))


def test_architecture_is_stable(real_records):
    """Same model name must mean the same parameter count.

    Weight sharing legitimately changes it, so those runs are excluded -- which
    is itself the point: `shared_params_unique` exists so that change is visible
    rather than showing up as an unexplained architecture drift.
    """
    counts = {}
    for r in real_records:
        if r.get('weight_sharing_en'):
            continue
        counts.setdefault(r.get('total_params'), []).append(r['record_id'])
    assert len(counts) <= 1, (
        'the same architecture reports different parameter counts: ' +
        '; '.join(f'{k}: {len(v)} run(s)' for k, v in counts.items()))


def test_smoke_and_real_runs_are_not_mixed(ladder_records):
    """A tier-0 row is not a measurement. If both kinds share a table, one of
    them is being read as something it is not."""
    smoke = [r['record_id'] for r in ladder_records if r.get('is_smoke')]
    real = [r['record_id'] for r in ladder_records if not r.get('is_smoke')]
    assert not (smoke and real), (
        f'{len(smoke)} smoke and {len(real)} real record(s) coexist. Move the '
        f'smoke records aside before generating any table: {smoke[:3]}')


def test_sparse_runs_are_chained_to_a_dense_checkpoint(real_records):
    """Every sparse rung starts from a dense baseline. If `trained_weights` is
    unset the run started from random or ImageNet weights instead, and it is not
    comparable to the rungs that did."""
    bad = [r['record_id'] for r in real_records
           if r.get('pruner') and not r.get('trained_weights')]
    assert not bad, f'sparse runs with no starting checkpoint: {bad[:5]}'


# --- reproducibility -------------------------------------------------------

def test_repeated_seeds_within_a_rung_are_not_duplicates(real_records):
    """Two records for one (rung, seed) mean either a re-run or a key collision.

    Either way the aggregate would average a number with itself and report a
    standard deviation that is too small.
    """
    seen = {}
    for r in real_records:
        key = r.get('experiment_group')
        seen.setdefault(key, []).append(r['record_id'])
    dupes = {k: v for k, v in seen.items() if len(v) > 1}
    assert not dupes, (
        'duplicate records for the same cell:\n' +
        '\n'.join(f'  {k}: {v}' for k, v in list(dupes.items())[:5]))


def test_each_rung_has_enough_seeds(real_records):
    """n=3 is not a sample size.

    At n=3 paired, t(.975, 2) = 4.303 and the minimum detectable effect is about
    3.1 sigma_d. A ladder reporting 1-point component effects at n=3 is reporting
    noise. This warns at fewer than 3 and is the reason the manifest defaults to
    5, and to 8 on the Stage C rungs that carry the paper's claim.
    """
    thin = {rung: sorted({r.get('seed') for r in rows})
            for rung, rows in _by_rung(real_records).items()
            if len({r.get('seed') for r in rows}) < 3}
    assert not thin, (
        'rungs with fewer than 3 seeds -- no error bar is defensible here:\n' +
        '\n'.join(f'  {k}: seeds {v}' for k, v in thin.items()))


# --- parity against the literature ----------------------------------------

def test_dense_baseline_is_within_reach_of_published(real_records):
    """The attack this answers: dense baselines in the submitted paper sit 1.7 to
    4.9 points below published figures for the same architecture and dataset, so
    every "matches or exceeds dense" claim reads as "exceeds our undertrained
    dense".

    Reference values live in results/reference/dense_reference.json rather than
    in code, so a reviewer can read them and a tolerance change shows up as a
    data diff. Skipped, not failed, when that file is absent.
    """
    from results import results_dir
    ref_path = results_dir() / 'reference' / 'dense_reference.json'
    if not ref_path.exists():
        pytest.skip(f'no reference file at {ref_path}')
    ref = json.loads(ref_path.read_text(encoding='utf-8'))

    bad = []
    for r in real_records:
        if r.get('pruner'):
            continue
        key = f"{r.get('model_name')}/{r.get('dataset_name')}"
        entry = ref.get(key)
        got = r.get('test_acc_exact_pct')
        if not entry or got is None:
            continue
        want, tol = float(entry['top1']), float(entry.get('tolerance', 1.0))
        if float(got) < want - tol:
            bad.append((r['record_id'], key, round(float(got), 2), want, tol,
                        entry.get('source')))
    assert not bad, (
        'dense baselines below published parity -- every downstream claim '
        'inherits this gap:\n' +
        '\n'.join(f'  {b[1]}: got {b[2]}, published {b[3]} +/-{b[4]} ({b[5]})'
                  for b in bad))


# --- ledger ----------------------------------------------------------------

def test_run_status_ledger(records, capsys):
    """Not an assertion -- a dashboard. `pytest -m results -s` prints it."""
    from collections import Counter

    ladder = [r for r in records if r.get('experiment_group')]
    with capsys.disabled():
        print()
        print(f'  {len(records)} record(s) total, {len(ladder)} from the manifest')
        status = Counter(r.get('status') for r in ladder)
        for name, n in status.most_common():
            print(f'    {name:<10} {n}')

        failures = [r for r in ladder if r.get('status') == 'failed']
        if failures:
            print(f'\n  {len(failures)} FAILED cell(s):')
            for r in failures[:10]:
                print(f'    {r.get("experiment_group")}: {r.get("error")}')

        smoke = sum(1 for r in ladder if r.get('is_smoke'))
        if smoke:
            print(f'\n  !! {smoke} record(s) are tier-0 smoke on truncated '
                  f'loaders. Not measurements.')

        ok = [r for r in ladder if r.get('status') == 'ok' and not r.get('is_smoke')]
        if ok:
            per_rung = Counter((r.get('experiment_group') or '.').split('.')[1]
                               for r in ok)
            print(f'\n  seeds per rung ({len(ok)} real run(s)):')
            for rung, n in sorted(per_rung.items()):
                flag = '' if n >= 3 else '   <- under-seeded'
                print(f'    {rung:<10} {n}{flag}')
    assert True

# --- checkpoint provenance -------------------------------------------------
#
# These three exist because of a real tier-0 failure: the dense rung succeeded
# and its checkpoint sat on disk, while all 21 sparse rungs failed to find it.
# save_path was recorded RELATIVE to the CWD, the runner launches each script
# with cwd=project/scripts, and resolve_checkpoint evaluated that path from the
# repo root. Nothing raised at record time; the symptom appeared one cell later
# as a missing baseline, which is a far more misleading thing to debug than a
# path bug. The unit suite could not have caught it -- the defect only exists
# in the relationship between two processes' working directories.


def test_checkpoint_paths_are_absolute(ladder_records):
    """A path in a record must not depend on who reads it, or from where.

    Any consumer of a record -- resolve_checkpoint, the tables, a human six
    months later -- runs from some other directory than the training script did.
    """
    bad = []
    for r in ladder_records:
        for field in ('save_path', 'trained_weights'):
            value = r.get(field)
            if value and not Path(value).is_absolute():
                bad.append(f"{r.get('record_id')}.{field} = {value!r}")
    assert not bad, (
        'relative checkpoint path(s) recorded; these resolve only from the '
        'directory the training script happened to run in:\n  '
        + '\n  '.join(bad))


def test_recorded_checkpoints_exist_on_disk(ladder_records):
    """The file a record names must actually be there.

    A record pointing at a checkpoint that has been moved or deleted is worse
    than no record: every downstream rung chains off it, so one stale path
    silently invalidates a whole stage.
    """
    missing = [f"{r.get('record_id')} -> {r.get('save_path')}"
               for r in ladder_records
               if r.get('save_path') and not Path(r['save_path']).is_file()]
    assert not missing, (
        'record(s) name a checkpoint that is not on disk:\n  '
        + '\n  '.join(missing))


def test_sparse_runs_chain_to_a_checkpoint_of_the_same_seed(real_records):
    """Pairing has to hold all the way down to initialisation.

    A sparse rung resolves its dense checkpoint at execution time. If it falls
    back to a different seed's checkpoint -- which the runner permits, with a
    warning, when the matching one is absent -- then the rung is no longer
    seed-paired with its comparator and every paired CI computed from it is
    overstated. The warning is easy to miss in a 65-cell sweep; this is not.
    """
    by_path = {r['save_path']: r for r in real_records
               if r.get('save_path') and r.get('method') == 'dense'}
    if not by_path:
        pytest.skip('no dense records with checkpoints to chain against')

    mismatched = []
    for r in real_records:
        weights = r.get('trained_weights')
        if not weights or r.get('method') == 'dense':
            continue
        parent = by_path.get(weights)
        if parent is None:
            continue          # covered by test_sparse_runs_are_chained_to_a_dense_checkpoint
        if parent.get('seed') != r.get('seed'):
            mismatched.append(
                f"{r.get('record_id')} (seed {r.get('seed')}) started from "
                f"{parent.get('record_id')} (seed {parent.get('seed')})")
    assert not mismatched, (
        'sparse run(s) chained to a dense checkpoint of a DIFFERENT seed, so '
        'they are not seed-paired with their comparator:\n  '
        + '\n  '.join(mismatched))
