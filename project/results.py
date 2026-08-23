"""Standardized experiment records.

Every training run writes exactly one immutable JSON record. `runs.csv` is a
*derived* flat index, regenerated from those records rather than appended to.

Why JSON-per-record is the source of truth and the CSV is not:

  1. Schema drift. Adding a field mid-project either invalidates an append-only
     CSV's header or silently drops the new column. Here `rebuild_index()` unions
     the keys across all records and leaves blanks for rows that predate a field.
  2. No atomic appends on DBFS. One file per record needs neither append semantics
     nor a lock, so two concurrent runs cannot interleave a row.
  3. Crash isolation. A six-hour run must not lose its result because an index
     write failed, so the record is written and fsynced first and the index is
     rebuilt inside a try/except.
  4. Nested data. Per-epoch histories, lambda trajectories and layer-wise sparsity
     are not columns.

The writer is stdlib-only (`json`, `csv`, `subprocess`). pandas is imported lazily,
inside the loader and aggregation helpers, so recording a result never depends on
it being installed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path

import torch

SCHEMA_VERSION = 1

# Ordered so `runs.csv` reads left-to-right as identity -> what was run -> result.
_CSV_PRIORITY = [
    'record_id', 'run_id', 'phase', 'status', 'experiment_group',
    'model_name', 'dataset_name', 'method', 'sparsity_requested', 'seed', 'is_smoke',
    'test_acc_exact_pct', 'test_acc_pct', 'test_loss_exact', 'test_ppl_exact',
    'sparsity_reported', 'sparsity_mask', 'sparsity_value',
]


# --- paths ----------------------------------------------------------------

def repo_root(start: Path | None = None) -> Path:
    """Walk up from this file to the directory containing .git."""
    here = (start or Path(__file__)).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / '.git').exists():
            return candidate
    return Path(__file__).resolve().parents[1]


def results_dir(path=None) -> Path:
    """Where records live.

    Resolved from the repo root, not the CWD: notebooks shell out to
    `../scripts/*.py` from a subdirectory, so a relative 'results/' would land
    somewhere different depending on who called. Override with BACP_RESULTS_DIR
    (set that to /dbfs/... on Databricks).
    """
    if path is not None:
        root = Path(path)
    elif os.environ.get('BACP_RESULTS_DIR'):
        root = Path(os.environ['BACP_RESULTS_DIR'])
    else:
        root = repo_root() / 'results'
    return root


def _ensure_dirs(root: Path) -> None:
    for sub in ('runs', 'logs', 'gates', 'tables', 'reference'):
        (root / sub).mkdir(parents=True, exist_ok=True)


# --- provenance -----------------------------------------------------------

def _git(*args, cwd=None):
    # encoding is explicit because git emits UTF-8 while Windows' locale codec
    # is cp1252: without it, a diff containing any non-cp1252 byte kills the
    # reader thread and the patch hash silently degrades to None -- a dirty
    # tree with no patch hash, which is exactly what capture_git exists to
    # prevent.
    try:
        out = subprocess.run(['git', *args], cwd=str(cwd or repo_root()),
                             capture_output=True, text=True, timeout=15,
                             encoding='utf-8', errors='replace')
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:                                             # noqa: BLE001
        return None


def capture_git(root: Path | None = None, save_patch_to: Path | None = None) -> dict:
    """Git provenance, including a hash of the uncommitted diff.

    A HEAD sha alone does not identify the code that ran when the tree is dirty --
    and it is routinely dirty during active work. So the diff is hashed and, if
    `save_patch_to` is given, written out, making the exact state recoverable.
    Never raises; every field degrades to None.
    """
    root = root or repo_root()
    sha = _git('rev-parse', 'HEAD', cwd=root)
    branch = _git('rev-parse', '--abbrev-ref', 'HEAD', cwd=root)
    porcelain = _git('status', '--porcelain', cwd=root) or ''
    lines = [l for l in porcelain.splitlines() if l.strip()]
    untracked = [l[3:] for l in lines if l.startswith('??')]

    patch = _git('diff', 'HEAD', cwd=root)
    patch_sha1 = None
    if patch:
        patch_sha1 = hashlib.sha1(patch.encode('utf-8', 'replace')).hexdigest()
        if save_patch_to is not None:
            # Content-addressed, not one copy per record. A sweep runs dozens of
            # cells against the same working tree, so a per-record copy stores
            # the identical ~160 KB diff dozens of times -- 12 MB for 75 records
            # in practice, growing linearly with every sweep. The record keeps
            # git_patch_sha1, which is the actual identifier; this is just the
            # bytes it names.
            try:
                shared = save_patch_to.parent / 'patches' / f'{patch_sha1}.patch'
                shared.parent.mkdir(parents=True, exist_ok=True)
                if not shared.exists():
                    shared.write_text(patch, encoding='utf-8')
            except Exception:                                     # noqa: BLE001
                pass

    return {
        'git_sha': sha,
        'git_branch': branch,
        'git_dirty': bool(lines),
        'git_n_dirty': len(lines),
        'git_untracked': len(untracked),
        'git_untracked_paths': untracked,
        'git_patch_sha1': patch_sha1,
    }


def code_fingerprint(root: Path | None = None) -> str:
    """sha256 over the project's source, excluding tests and notebooks.

    Gates use this to detect staleness: a result produced before the code changed
    must not silently satisfy a downstream gate. Excluding tests and notebooks
    means editing a notebook or adding a test does not invalidate a 40-hour run.
    """
    root = root or repo_root()
    project = root / 'project'
    digest = hashlib.sha256()
    if not project.exists():
        return 'unknown'
    for path in sorted(project.rglob('*.py')):
        rel = path.relative_to(project).as_posix()
        if rel.startswith(('tests/', 'notebooks/', 'test_notebooks/')):
            continue
        digest.update(rel.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


# Distribution name for packages whose import name differs from it.
_DIST_NAMES = {'sklearn': 'scikit-learn', 'cv2': 'opencv-python'}


def _version(module_name):
    """Version string, WITHOUT importing the package.

    `importlib.import_module('transformers')` just to read `__version__` cost
    4.74 s per record on this machine -- transformers pulls in most of its model
    registry on import. Package metadata answers the same question in about a
    millisecond, and every run writes at least one record.

    Falls back to the import only if metadata is unavailable (an editable
    install without a dist-info, say), so the field never silently becomes None
    for a package that is genuinely present.
    """
    try:
        from importlib.metadata import version
        return version(_DIST_NAMES.get(module_name, module_name))
    except Exception:                                             # noqa: BLE001
        pass
    try:
        import importlib
        return getattr(importlib.import_module(module_name), '__version__', None)
    except Exception:                                             # noqa: BLE001
        return None


def capture_environment() -> dict:
    """Everything needed to say where a number came from.

    None of this was recorded before -- not the torch version, not the GPU, not
    even whether CUDA was available.
    """
    cuda_ok = torch.cuda.is_available()
    env = {
        'hostname': socket.gethostname(),
        'platform': platform.platform(),
        'python_version': sys.version.split()[0],
        'torch_version': torch.__version__,
        'torchvision_version': _version('torchvision'),
        'transformers_version': _version('transformers'),
        'numpy_version': _version('numpy'),
        'cuda_version': torch.version.cuda if cuda_ok else None,
        'cudnn_version': torch.backends.cudnn.version() if cuda_ok else None,
        'gpu_name': torch.cuda.get_device_name(0) if cuda_ok else None,
        'gpu_count': torch.cuda.device_count() if cuda_ok else 0,
        'gpu_mem_total_gb': None,
        'cudnn_deterministic': bool(torch.backends.cudnn.deterministic),
        'cudnn_benchmark': bool(torch.backends.cudnn.benchmark),
        'deterministic_algorithms': bool(torch.are_deterministic_algorithms_enabled()),
    }
    if cuda_ok:
        try:
            props = torch.cuda.get_device_properties(0)
            env['gpu_mem_total_gb'] = round(props.total_memory / 1024 ** 3, 2)
        except Exception:                                         # noqa: BLE001
            pass
    return env


# --- measurement ----------------------------------------------------------

def count_parameters(model, pruner=None) -> dict:
    """Absolute parameter counts, with the prunable denominator made explicit.

    Only the sparsity *fraction* was ever recorded, which is uninterpretable on its
    own: the prunable set is 63.3% of DistilBERT with embeddings excluded and 99.7%
    with them included, so the same fraction describes wildly different things.
    """
    from pruning_factory import layer_check

    total = trainable = prunable = prunable_zero = nonzero_all = 0
    layer_sparsity = {}

    for name, param in model.named_parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
        zeros = int(torch.sum(param == 0).item())
        nonzero_all += n - zeros
        if layer_check(name, param):
            prunable += n
            prunable_zero += zeros
            layer_sparsity[name] = {'numel': n, 'zeros': zeros}

    mask_total = mask_active = None
    if pruner is not None and getattr(pruner, 'masks', None):
        mask_total = sum(m.numel() for m in pruner.masks.values())
        mask_active = int(sum(m.sum().item() for m in pruner.masks.values()))
        for name, mask in pruner.masks.items():
            if name in layer_sparsity:
                layer_sparsity[name]['mask_active'] = int(mask.sum().item())

    return {
        'total_params': total,
        'trainable_params': trainable,
        'prunable_params': prunable,
        'nonprunable_params': total - prunable,
        'nonzero_all_params': nonzero_all,
        'prunable_zero_params': prunable_zero,
        'active_prunable_params': mask_active if mask_active is not None
                                  else prunable - prunable_zero,
        'mask_total': mask_total,
        'mask_covers_prunable': (mask_total == prunable) if mask_total is not None else None,
        'layer_sparsity': layer_sparsity,
    }


def measure_sparsity(model, pruner=None) -> dict:
    """Recompute sparsity rather than trusting whatever the metrics dict carried.

    Both measures are kept because they legitimately differ under dynamic sparse
    training: RigL and EAST zero-initialise regrown connections, so a structurally
    active weight reads as zero. The value-based figure therefore *overstates*
    sparsity -- the flattering direction -- by ~3 points in practice.
    """
    from pruning_factory import check_mask_sparsity, check_model_sparsity, report_sparsity

    value = check_model_sparsity(model)
    mask = check_mask_sparsity(pruner)
    total = sum(p.numel() for p in model.parameters())
    zeros = int(sum(torch.sum(p == 0).item() for p in model.parameters()))
    return {
        'sparsity_reported': report_sparsity(model, pruner),
        'sparsity_mask': mask,
        'sparsity_value': value,
        'sparsity_gap': (value - mask) if mask is not None else None,
        'sparsity_all_params': (zeros / total) if total else 0.0,
    }


@torch.no_grad()
def evaluate_exact(trainer, loader=None, split='test') -> dict:
    """Sample/token-weighted metrics, and a count of anything the loader dropped.

    The reported metrics are a batch-mean of batch-means. For classification with
    equal batches that equals the sample mean, but a partial final batch or a
    varying token count (WikiText-2) breaks the equivalence. Perplexity is also
    reported as mean(exp(batch_loss)), which by Jensen's inequality exceeds
    exp(mean_loss) -- biased upward. This computes both correctly:
    acc = correct/total and ppl = exp(sum_loss/sum_count).
    """
    import torch.nn.functional as F

    if loader is None:
        loader = (getattr(trainer, 'ft_testloader', None)
                  or getattr(trainer, 'testloader', None)
                  or getattr(trainer, 'valloader', None))
    if loader is None:
        return {}

    from training_utils import _handle_data_to_device

    model = trainer.model
    was_training = model.training
    model.eval()

    correct = count = 0
    loss_sum = 0.0
    t0 = time.perf_counter()

    for batch in loader:
        data, labels = _handle_data_to_device(trainer, batch)
        if isinstance(data, (list, tuple)):
            data = data[0]
        out = model(data)
        logits = out.logits if hasattr(out, 'logits') else out

        if logits.dim() == 3:                      # masked language modelling
            mask = labels != -100
            n = int(mask.sum().item())
            if n == 0:
                continue
            flat_logits = logits[mask]
            flat_labels = labels[mask]
        else:
            n = labels.numel()
            flat_logits, flat_labels = logits, labels

        loss_sum += F.cross_entropy(flat_logits, flat_labels, reduction='sum').item()
        correct += int((flat_logits.argmax(dim=-1) == flat_labels).sum().item())
        count += n

    if was_training:
        model.train()

    dataset_len = None
    try:
        dataset_len = len(loader.dataset)
    except Exception:                                             # noqa: BLE001
        pass

    seen_samples = count
    dropped = None
    if dataset_len is not None and logits_is_2d(loader):
        # What SHOULD have been scored. Normally the whole split; under tier-0
        # smoke the loaders are deliberately truncated, so the expectation is the
        # truncation budget rather than the dataset length.
        #
        # Without this, a smoke run reports ~9,900 "dropped" samples and the
        # invariant that guards against drop_last silently losing test images
        # fires on every single tier-0 record. An invariant that always fails is
        # an invariant that gets muted -- and then the real regression it exists
        # to catch walks straight through it.
        expected = dataset_len
        limit = int(getattr(trainer, 'limit_eval_batches', 0) or 0)
        if limit > 0:
            per_batch = getattr(loader, 'batch_size', None) or 0
            if per_batch:
                expected = min(dataset_len, limit * per_batch)
        dropped = max(expected - seen_samples, 0)

    mean_loss = loss_sum / count if count else None
    return {
        'test_acc_exact_pct': (100.0 * correct / count) if count else None,
        'test_loss_exact': mean_loss,
        'test_ppl_exact': safe_exp(mean_loss),
        'eval_samples_seen': seen_samples,
        'eval_samples_dropped': dropped,
        'eval_split': split,
        'wall_eval_s': round(time.perf_counter() - t0, 3),
    }


def safe_exp(x):
    """exp() that saturates instead of raising.

    math.exp overflows above ~709, and an untrained or badly-diverged model easily
    produces a loss past that. Perplexity is a reporting convenience, so it must
    never be the thing that destroys a completed run's record -- return inf and let
    the table show it.
    """
    import math
    if x is None:
        return None
    try:
        return math.exp(x)
    except OverflowError:
        return float('inf')


def logits_is_2d(loader) -> bool:
    """True when one dataset item corresponds to one scored sample.

    For classification the sample count and the dataset length are comparable, so a
    dropped-sample count is meaningful. For MLM the unit is a masked *token*, so
    comparing against the number of text blocks would be nonsense.
    """
    try:
        first = loader.dataset[0]
    except Exception:                                             # noqa: BLE001
        return True
    if isinstance(first, dict):
        labels = first.get('labels')
        return not (hasattr(labels, 'dim') and labels.dim() >= 1)
    return True


def measure_compute(model, input_shape, batch_sizes=(1,), device=None,
                    pruner=None, num_runs=50) -> dict:
    """Latency and FLOPs.

    Two FLOP figures, deliberately named apart. `gflops_dense` comes from fvcore,
    which reads shapes and is therefore **sparsity-blind** -- a 99%-sparse model
    reports the same count as dense. `gflops_sparse_est` scales that by the measured
    mask density and is analytic, not measured: standard dense kernels realise none
    of it. Keeping both under distinct names is what stops an unstructured-sparsity
    FLOP reduction being reported as a speedup.
    """
    out = {'flops_method': None, 'gflops_dense': None, 'gflops_sparse_est': None,
           'latency_ms_bs1': None, 'latency_ms_batch': None,
           'throughput_samples_s': None, 'inference_dtype': None}

    device = device or next(model.parameters()).device
    out['inference_dtype'] = str(next(model.parameters()).dtype)

    try:
        from compute_metrics import calculate_inference_time
    except Exception:                                             # noqa: BLE001
        calculate_inference_time = None

    if calculate_inference_time is not None:
        for bs in batch_sizes:
            x = torch.randn(bs, *input_shape, device=device)
            try:
                ms = calculate_inference_time(model, x, num_runs=num_runs)
            except Exception:                                     # noqa: BLE001
                continue
            if bs == 1:
                out['latency_ms_bs1'] = round(ms, 4)
            else:
                out['latency_ms_batch'] = round(ms, 4)
                out['throughput_samples_s'] = round(bs / (ms / 1000.0), 2)

    try:
        from compute_metrics import calculate_gflops
        x = torch.randn(1, *input_shape, device=device)
        dense = calculate_gflops(model, x)
        out['gflops_dense'] = round(dense, 4)
        out['flops_method'] = 'fvcore_dense'
        from pruning_factory import check_mask_sparsity
        s = check_mask_sparsity(pruner)
        if s is not None:
            out['gflops_sparse_est'] = round(dense * (1.0 - s), 4)
            out['flops_method'] = 'fvcore_dense+mask_density'
    except Exception:                                             # noqa: BLE001
        pass

    return out


# --- the record -----------------------------------------------------------

@dataclass
class RunRecord:
    schema_version: int = SCHEMA_VERSION
    record_id: str = ''
    run_id: str = ''
    phase: str = ''
    experiment_group: str | None = None
    experiment_type: str = ''
    status: str = 'ok'
    error: str | None = None
    notes: str = ''

    started_at: str | None = None
    ended_at: str | None = None
    duration_s: float | None = None
    cmdline: str = ''

    # provenance + environment (flattened from capture_git / capture_environment)
    git_sha: str | None = None
    git_branch: str | None = None
    git_dirty: bool | None = None
    git_n_dirty: int | None = None
    git_untracked: int | None = None
    git_patch_sha1: str | None = None
    code_fingerprint: str | None = None
    hostname: str | None = None
    platform: str | None = None
    python_version: str | None = None
    torch_version: str | None = None
    torchvision_version: str | None = None
    transformers_version: str | None = None
    numpy_version: str | None = None
    cuda_version: str | None = None
    cudnn_version: int | None = None
    gpu_name: str | None = None
    gpu_count: int | None = None
    gpu_mem_total_gb: float | None = None
    cudnn_deterministic: bool | None = None
    cudnn_benchmark: bool | None = None
    deterministic_algorithms: bool | None = None

    seed: int | None = None
    seed_source: str = 'unknown'
    val_split_seed: int | None = None
    # The validation FRACTION, not just its seed. It became a knob when
    # runs had to be reproducible across the change that made the split
    # apply to contrastive recipes, so a record must carry it or two runs
    # trained on different amounts of data look identical on paper.
    val_split: float | None = None

    # model + data
    model_name: str | None = None
    model_family: str | None = None
    model_type: str | None = None
    model_task: str | None = None
    pretrained: bool | None = None
    # 'imagenet_checkpoint' | 'random' | 'hf_hub'. Distinct from `pretrained`,
    # which only says what was asked for. load_weights fails soft, so a run can
    # request ImageNet init and get random init with nothing but a warning --
    # and on a fresh machine it always will, since every spec.weight path points
    # into /dbfs.
    init_source: str | None = None
    dataset_name: str | None = None
    num_classes: int | None = None
    image_size: int | None = None
    n_views: int | None = None
    batch_size: int | None = None
    num_workers: int | None = None
    amp_enabled: bool | None = None
    train_batches: int | None = None
    eval_batches: int | None = None
    eval_split: str | None = None
    eval_samples_seen: int | None = None
    eval_samples_dropped: int | None = None
    embedded_dim: int | None = None
    num_out_features: int | None = None

    total_params: int | None = None
    trainable_params: int | None = None
    prunable_params: int | None = None
    nonprunable_params: int | None = None
    nonzero_all_params: int | None = None
    active_prunable_params: int | None = None
    mask_total: int | None = None
    mask_covers_prunable: bool | None = None
    shared_params_total: int | None = None
    shared_params_unique: int | None = None

    # method + sparsity scope
    method: str | None = None
    pruner: str | None = None
    is_bacp: bool | None = None
    contrastive_mode: str | None = None
    proj_mode: str | None = None
    lambdas_json: str | None = None
    learnable_lambdas: bool | None = None
    tau: float | None = None
    num_snapshots: int | None = None
    n_snapshots_used: int | None = None
    distill_mode: str | None = None
    lambda_kd: float | None = None
    dyrelu_en: bool | None = None
    dyrelu_phasing_en: bool | None = None
    weight_sharing_en: bool | None = None
    sparsity_requested: float | None = None
    sparsity_target_adjusted: float | None = None
    sparsity_scheduler: str | None = None
    delta_T: int | None = None
    east_s_min: float | None = None
    east_prune_rate: float | None = None
    east_Tc_ratio: float | None = None
    rigl_alpha: float | None = None
    layerwise_alloc: str | None = None
    wanda_group: str | None = None
    prune_task_head: bool | None = None
    prune_embeddings: bool | None = None
    scope_key: str | None = None
    initial_sparsity: float | None = None

    # protocol
    optimizer_type: str | None = None
    learning_rate: float | None = None
    weight_decay: float | None = None
    scheduler_type: str | None = None
    epochs: int | None = None
    recovery_epochs: int | None = None
    patience: int | None = None
    epochs_ft: int | None = None
    optimizer_type_ft: str | None = None
    learning_rate_ft: float | None = None
    total_steps: int | None = None
    warmup_steps: int | None = None
    epochs_completed: int | None = None
    early_stopped: bool | None = None
    # Tier-0 smoke truncation. Non-zero means the run saw only part of the data;
    # test_experiment_invariants refuses to treat such a row as a measurement.
    limit_train_batches: int | None = None
    limit_eval_batches: int | None = None
    is_smoke: bool | None = None
    trained_weights: str | None = None
    save_path: str | None = None
    checkpoint_sha256: str | None = None
    checkpoint_bytes: int | None = None
    trained_weights_sha256: str | None = None

    # metrics
    test_acc_pct: float | None = None
    test_loss: float | None = None
    test_ppl: float | None = None
    test_acc_exact_pct: float | None = None
    test_loss_exact: float | None = None
    test_ppl_exact: float | None = None
    metrics_are_batch_means: bool = True
    final_train_loss: float | None = None
    sparsity_reported: float | None = None
    sparsity_mask: float | None = None
    sparsity_value: float | None = None
    sparsity_gap: float | None = None
    sparsity_all_params: float | None = None

    # compute
    wall_total_s: float | None = None
    wall_eval_s: float | None = None
    s_per_epoch_mean: float | None = None
    s_per_epoch_std: float | None = None
    peak_gpu_mem_gb: float | None = None
    latency_ms_bs1: float | None = None
    latency_ms_batch: float | None = None
    throughput_samples_s: float | None = None
    gflops_dense: float | None = None
    gflops_sparse_est: float | None = None
    flops_method: str | None = None
    inference_dtype: str | None = None

    # JSON-only payload
    detail: dict = field(default_factory=dict)

    def to_flat_dict(self) -> dict:
        d = asdict(self)
        d.pop('detail', None)
        return d

    def to_json_dict(self) -> dict:
        return asdict(self)


# --- config access --------------------------------------------------------

def _cfg(trainer, name, default=None):
    """Read a config field, preferring the snapshot taken at construction.

    Necessary because BaCPTrainer.finetune() overwrites `learning_rate` and
    `optimizer_type` with their `_ft` counterparts and sets `prune = False`. Reading
    those live after a full run would record the fine-tuning values as though they
    had been the contrastive ones.
    """
    snapshot = getattr(trainer, 'logger_params', None) or {}
    if name in snapshot:
        return snapshot[name]
    return getattr(trainer, name, default)


def _is_bacp(trainer) -> bool:
    """Duck-type rather than read `is_bacp`, which finetune() flips to False."""
    return hasattr(trainer, 'lambda_history')


def canonical_method(trainer) -> str:
    """The groupby key used by every table."""
    pruner = _cfg(trainer, 'pruning_type')
    sparsity = _cfg(trainer, 'target_sparsity')
    if not (pruner and sparsity):
        return 'dense'
    base = f'bacp+{pruner}' if _is_bacp(trainer) else pruner
    if _is_bacp(trainer):
        lambdas = _cfg(trainer, 'lambdas')
        # A CE-only run is a different method, not a variant of the same one.
        if lambdas and len(lambdas) == 4 and all(float(x) == 0.0 for x in lambdas[:3]):
            base = f'ce_only+{pruner}'
        mode = _cfg(trainer, 'distill_mode')
        if mode and mode != 'none':
            base = f'kd_{mode}+{pruner}'
    return base


def make_run_id(trainer, seed=None, datestamp=None) -> str:
    parts = [
        str(_cfg(trainer, 'model_name', 'model')),
        str(_cfg(trainer, 'dataset_name', 'data')),
        canonical_method(trainer),
    ]
    sparsity = _cfg(trainer, 'target_sparsity')
    if sparsity:
        parts.append(f's{sparsity}')
    parts.append(f'seed{seed if seed is not None else _cfg(trainer, "seed", "NA")}')
    parts.append(str(datestamp or _cfg(trainer, 'datestamp', '')))
    return '_'.join(p for p in parts if p)


_ALIASES = {
    'val_loss': 'test_loss',
    'accuracy': 'test_acc_pct',
    'perplexity': 'test_ppl',
    'Finetuning Val Loss': 'test_loss',
    'Finetuning Accuracy': 'test_acc_pct',
    'Finetuning Perplexity': 'test_ppl',
}


def normalise_metrics(metrics: dict) -> dict:
    """Reconcile the two trainers' incompatible metric keys.

    Trainer.evaluate returns {val_loss, accuracy, perplexity}; BaCPTrainer.evaluate
    returns the same quantities as {Finetuning Val Loss, Finetuning Accuracy,
    Finetuning Perplexity}. Unknown keys are preserved under a metric_* prefix
    rather than dropped, so a future addition to evaluate() is not lost silently.
    """
    out = {}
    for key, value in (metrics or {}).items():
        if key == 'sparsity':
            out['_sparsity_from_metrics'] = value
        elif key in _ALIASES:
            out[_ALIASES[key]] = value
        else:
            slug = ''.join(c if c.isalnum() else '_' for c in str(key)).strip('_').lower()
            out[f'metric_{slug}'] = value
    return out


def _abspath(path):
    """Absolute form of a checkpoint path, resolved against the current CWD.

    Called inside the training process, where the CWD is the one save_path was
    built against. Returns the input unchanged if it is empty or already
    absolute, so it is safe to apply twice.
    """
    if not path:
        return path
    try:
        return str(Path(path).resolve())
    except Exception:                                             # noqa: BLE001
        return path


def _sha256_file(path, limit_bytes=None):
    try:
        p = Path(path)
        if not p.is_file():
            return None, None
        size = p.stat().st_size
        digest = hashlib.sha256()
        with p.open('rb') as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b''):
                digest.update(chunk)
        return digest.hexdigest(), size
    except Exception:                                             # noqa: BLE001
        return None, None


# --- writing --------------------------------------------------------------

def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=1, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def record_run(trainer, metrics, *, phase=None, experiment_group=None,
               status='ok', error=None, notes='', extra=None,
               hash_checkpoints=True, histories=True, exact_metrics=True,
               root=None, strict=False):
    """Write one record. Returns the RunRecord, or None if recording failed.

    strict=False by default and deliberately: a bug in this function must never
    destroy the result of a six-hour training run.
    """
    try:
        return _record_run_inner(
            trainer, metrics, phase=phase, experiment_group=experiment_group,
            status=status, error=error, notes=notes, extra=extra,
            hash_checkpoints=hash_checkpoints, histories=histories,
            exact_metrics=exact_metrics, root=root,
        )
    except Exception:                                             # noqa: BLE001
        if strict:
            raise
        import traceback
        print('[RESULTS] failed to record this run; the run itself is unaffected:')
        traceback.print_exc()
        return None


def _record_run_inner(trainer, metrics, *, phase, experiment_group, status, error,
                      notes, extra, hash_checkpoints, histories, exact_metrics, root):
    from pruning_factory import excluded_keywords, set_prunable_scope

    out_root = results_dir(root)
    _ensure_dirs(out_root)

    # The prunable scope is module-level state in pruning_factory and leaks between
    # configs inside one kernel. Every sparsity number depends on it, so re-assert
    # it from this run's config before measuring anything.
    prune_head = bool(_cfg(trainer, 'prune_task_head', False))
    prune_emb = bool(_cfg(trainer, 'prune_embeddings', False))
    set_prunable_scope(prune_head, prune_emb, getattr(trainer, 'model', None))

    scope_key = 'backbone'
    if prune_head:
        scope_key += '+head'
    if prune_emb:
        scope_key += '+embeddings'

    seed = _cfg(trainer, 'seed')
    seed_source = 'args' if seed is not None else 'unknown'
    if seed is None:
        import utils
        seed = getattr(utils, '_ACTIVE_SEED', None)
        seed_source = 'global' if seed is not None else 'unknown'

    run_id = make_run_id(trainer, seed=seed)
    phase = phase or 'unknown'
    record_id = f'{run_id}__{phase}'
    n = 1
    while (out_root / 'runs' / f'{record_id}.json').exists():
        n += 1
        record_id = f'{run_id}__{phase}__{n}'

    model = getattr(trainer, 'model', None)
    pruner = getattr(trainer, 'pruner', None)

    rec = RunRecord(
        record_id=record_id, run_id=run_id, phase=phase,
        experiment_group=experiment_group,
        experiment_type=str(_cfg(trainer, 'experiment_type', '')),
        status=status, error=error, notes=notes,
        started_at=getattr(trainer, 'run_started_at', None),
        ended_at=datetime.now(timezone.utc).isoformat(),
        cmdline=' '.join(sys.argv),
        code_fingerprint=code_fingerprint(),
        seed=seed, seed_source=seed_source,
        val_split_seed=_cfg(trainer, 'val_split_seed'),
        val_split=_cfg(trainer, 'val_split'),
        method=canonical_method(trainer),
        pruner=_cfg(trainer, 'pruning_type'),
        is_bacp=_is_bacp(trainer),
        scope_key=scope_key,
        prune_task_head=prune_head, prune_embeddings=prune_emb,
    )

    patch_path = out_root / 'runs' / f'{record_id}.patch'
    for key, value in capture_git(save_patch_to=patch_path).items():
        if key == 'git_untracked_paths':
            continue
        setattr(rec, key, value)
    for key, value in capture_environment().items():
        if hasattr(rec, key):
            setattr(rec, key, value)

    # --- config fields, all via _cfg
    for name in ('model_name', 'dataset_name', 'num_classes', 'image_size', 'n_views',
                 'batch_size', 'num_workers', 'contrastive_mode', 'proj_mode',
                 'learnable_lambdas', 'tau', 'num_snapshots', 'distill_mode',
                 'lambda_kd', 'dyrelu_en', 'dyrelu_phasing_en', 'weight_sharing_en',
                 'sparsity_scheduler', 'delta_T', 'initial_sparsity', 'optimizer_type',
                 'learning_rate', 'scheduler_type', 'epochs', 'recovery_epochs',
                 'patience', 'epochs_ft', 'optimizer_type_ft', 'learning_rate_ft',
                 'weight_decay',
                 'total_steps', 'warmup_steps', 'trained_weights', 'embedded_dim',
                 'num_out_features', 'model_type', 'layerwise_alloc', 'wanda_group',
                 'limit_train_batches', 'limit_eval_batches'):
        setattr(rec, name, _cfg(trainer, name))

    rec.amp_enabled = bool(_cfg(trainer, 'enable_mixed_precision', False))
    rec.is_smoke = bool(_cfg(trainer, 'limit_train_batches', 0)
                        or _cfg(trainer, 'limit_eval_batches', 0))
    rec.pretrained = True     # _create_base_model hardcodes pretrained=True
    # ...but asking is not getting. This is what actually happened.
    rec.init_source = getattr(model, 'init_source', None) if model is not None else None
    # ABSOLUTE, resolved here rather than stored as written. save_path is built
    # relative to the CWD ('./research/bacp/...'), and the runner launches every
    # script with cwd=project/scripts -- so a relative path in the record is only
    # meaningful from that directory. resolve_checkpoint runs from the repo root
    # and silently found nothing, which failed all 21 sparse rungs of a tier-0
    # sweep while the dense checkpoint sat on disk the whole time.
    rec.save_path = _abspath(getattr(trainer, 'save_path', None))
    rec.trained_weights = _abspath(rec.trained_weights)
    rec.sparsity_requested = _cfg(trainer, 'requested_sparsity',
                                  _cfg(trainer, 'target_sparsity'))
    rec.sparsity_target_adjusted = _cfg(trainer, 'adjusted_sparsity',
                                        _cfg(trainer, 'target_sparsity'))
    lambdas = _cfg(trainer, 'lambdas')
    if lambdas:
        rec.lambdas_json = json.dumps([float(x) for x in lambdas])
    if model is not None:
        rec.model_family = getattr(model, 'model_family', None)
        rec.model_task = getattr(model, 'model_task', None)
        inner = getattr(model, 'model', None)
        rec.shared_params_total = getattr(inner, 'total_params', None)
        rec.shared_params_unique = getattr(inner, 'unique_params', None)
    if pruner is not None:
        rec.east_s_min = getattr(pruner, 's_min', None)
        rec.east_prune_rate = getattr(pruner, 'prune_rate', None)
        rec.east_Tc_ratio = getattr(pruner, 'Tc_ratio', None)
        rec.rigl_alpha = getattr(pruner, 'alpha', None)

    # --- metrics
    for key, value in normalise_metrics(metrics).items():
        if key == '_sparsity_from_metrics':
            continue
        if hasattr(rec, key):
            setattr(rec, key, value)

    if model is not None:
        for key, value in measure_sparsity(model, pruner).items():
            setattr(rec, key, value)
        counts = count_parameters(model, pruner)
        layer_sparsity = counts.pop('layer_sparsity')
        for key, value in counts.items():
            if hasattr(rec, key):
                setattr(rec, key, value)
    else:
        layer_sparsity = {}

    reported = (metrics or {}).get('sparsity')
    if reported is not None and rec.sparsity_reported is not None:
        if abs(float(reported) - float(rec.sparsity_reported)) > 1e-4:
            rec.notes = (rec.notes + ' sparsity_mismatch').strip()

    if exact_metrics and model is not None:
        for key, value in (evaluate_exact(trainer) or {}).items():
            if hasattr(rec, key):
                setattr(rec, key, value)

    # --- timing
    from training_utils import epoch_times, main_loop_epochs
    times = epoch_times(trainer)
    if times:
        import statistics
        rec.s_per_epoch_mean = round(statistics.fmean(times), 3)
        rec.s_per_epoch_std = round(statistics.stdev(times), 3) if len(times) > 1 else 0.0
    t0 = getattr(trainer, '_t0', None)
    if t0 is not None:
        rec.wall_total_s = round(time.perf_counter() - t0, 3)
        rec.duration_s = rec.wall_total_s

    # Main-loop epochs only. Recovery and fine-tuning epochs also pass through
    # _log_metrics, so counting every mark would make early_stopped meaningless.
    completed = main_loop_epochs(trainer)
    if completed:
        rec.epochs_completed = completed
        if rec.epochs is not None:
            rec.early_stopped = completed < rec.epochs
    if torch.cuda.is_available():
        try:
            rec.peak_gpu_mem_gb = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)
        except Exception:                                         # noqa: BLE001
            pass

    for name, attr in (('trainloader', 'train_batches'), ('testloader', 'eval_batches')):
        loader = getattr(trainer, name, None)
        if loader is not None:
            try:
                setattr(rec, attr, len(loader))
            except Exception:                                     # noqa: BLE001
                pass

    if hash_checkpoints:
        rec.checkpoint_sha256, rec.checkpoint_bytes = _sha256_file(rec.save_path)
        rec.trained_weights_sha256, _ = _sha256_file(rec.trained_weights)

    losses = getattr(trainer, 'train_losses', None)
    if losses:
        rec.final_train_loss = float(losses[-1])

    if extra:
        for key, value in extra.items():
            if hasattr(rec, key):
                setattr(rec, key, value)
            else:
                rec.detail[key] = value

    # --- JSON-only payload
    rec.detail.update({
        'logger_params': {k: v for k, v in (getattr(trainer, 'logger_params', {}) or {}).items()
                          if isinstance(v, (int, float, str, bool, list, dict))},
        'epoch_times_s': times,
        'excluded_keywords': list(excluded_keywords()),
        'layer_sparsity': layer_sparsity if histories else {},
    })
    if histories:
        for name in ('train_losses', 'accuracies', 'total_losses', 'prc_losses',
                     'fic_losses', 'snc_losses', 'ce_losses', 'training_acc',
                     'finetuning_loss', 'finetuning_acc', 'lambda_history'):
            value = getattr(trainer, name, None)
            if value:
                rec.detail[name] = _jsonable(value)
    if _is_bacp(trainer):
        rec.n_snapshots_used = len(getattr(trainer, 'snapshots', []) or [])

    _atomic_write_json(out_root / 'runs' / f'{record_id}.json', rec.to_json_dict())

    try:
        with (out_root / 'runs.jsonl').open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(rec.to_flat_dict(), default=str) + '\n')
    except Exception:                                             # noqa: BLE001
        pass
    try:
        rebuild_index(root=out_root)
    except Exception:                                             # noqa: BLE001
        print('[RESULTS] record written but the index rebuild failed; '
              'run results.rebuild_index() to repair')

    print(f'[RESULTS] {record_id}')
    return rec


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, 'item'):
        try:
            return value.item()
        except Exception:                                         # noqa: BLE001
            return str(value)
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def record_failure(cell: dict, *, returncode: int, log_tail: str = '',
                   experiment_group=None, root=None):
    """Record a run that died, so a failure is a row rather than a silence.

    Without this, a crashed cell is indistinguishable from one that was never
    scheduled -- and a table would show a blank either way.
    """
    out_root = results_dir(root)
    _ensure_dirs(out_root)

    run_id = cell.get('key') or 'unknown'
    record_id = f'{run_id}__failed'
    n = 1
    while (out_root / 'runs' / f'{record_id}.json').exists():
        n += 1
        record_id = f'{run_id}__failed__{n}'

    rec = RunRecord(
        record_id=record_id, run_id=run_id, phase='failed', status='failed',
        error=f'returncode={returncode}',
        experiment_group=experiment_group,
        ended_at=datetime.now(timezone.utc).isoformat(),
        code_fingerprint=code_fingerprint(),
        cmdline=' '.join(cell.get('cmd', [])),
    )
    for key in ('model_name', 'dataset_name', 'method', 'seed'):
        if key in cell:
            setattr(rec, key, cell[key])
    rec.sparsity_requested = cell.get('sparsity')
    for key, value in capture_git().items():
        if key != 'git_untracked_paths' and hasattr(rec, key):
            setattr(rec, key, value)
    rec.detail['log_tail'] = log_tail[-4000:] if log_tail else ''

    _atomic_write_json(out_root / 'runs' / f'{record_id}.json', rec.to_json_dict())
    try:
        rebuild_index(root=out_root)
    except Exception:                                             # noqa: BLE001
        pass
    print(f'[RESULTS] recorded FAILURE {record_id} (rc={returncode})')
    return rec


# --- reading --------------------------------------------------------------

def iter_records(root=None):
    out_root = results_dir(root)
    for path in sorted((out_root / 'runs').glob('*.json')):
        try:
            yield json.loads(path.read_text(encoding='utf-8'))
        except Exception:                                         # noqa: BLE001
            print(f'[RESULTS] skipping unreadable record {path.name}')


def rebuild_index(root=None) -> int:
    """Regenerate runs.csv from the JSON records. Returns the row count."""
    out_root = results_dir(root)
    _ensure_dirs(out_root)

    rows = []
    for payload in iter_records(out_root):
        payload.pop('detail', None)
        rows.append(payload)

    known = [f.name for f in fields(RunRecord) if f.name != 'detail']
    extra = sorted({k for r in rows for k in r} - set(known))
    ordered = _CSV_PRIORITY + [c for c in known if c not in _CSV_PRIORITY] + extra

    path = out_root / 'runs.csv'
    with path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=ordered, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in ordered})
    return len(rows)


def load_runs(root=None, *, dedupe=True, rebuild=False, status=('ok',),
              phase_priority=('finetune', 'prune', 'contrastive', 'dense', 'eval_only')):
    """Load runs.csv as a DataFrame. Requires pandas; the writer path does not."""
    import pandas as pd

    out_root = results_dir(root)
    if rebuild or not (out_root / 'runs.csv').exists():
        rebuild_index(out_root)

    df = pd.read_csv(out_root / 'runs.csv')
    if df.empty:
        return df
    if status:
        df = df[df['status'].isin(status)]
    if dedupe and not df.empty:
        rank = {p: i for i, p in enumerate(phase_priority)}
        df = df.assign(_rank=df['phase'].map(lambda p: rank.get(p, len(rank))))
        df = (df.sort_values(['run_id', '_rank', 'ended_at'])
                .drop_duplicates(subset=['run_id', 'phase'], keep='last')
                .drop(columns='_rank'))
    return df.reset_index(drop=True)


def load_record(record_id: str, root=None) -> dict:
    path = results_dir(root) / 'runs' / f'{record_id}.json'
    return json.loads(path.read_text(encoding='utf-8'))


def is_smoke_record(payload) -> bool:
    """Whether a record describes a tier-0 smoke run.

    Two signals because they are set by different layers and either can be the
    only one present: `is_smoke` is derived from the loader limits at record
    time, the `.smoke` key suffix is appended by the notebook cell builder.
    """
    if payload.get('is_smoke'):
        return True
    return (payload.get('experiment_group') or '').endswith('.smoke')


def resolve_checkpoint(model_name, dataset_name, *, seed=None, method='dense',
                       root=None, allow_smoke=False) -> str:
    """Find a completed checkpoint by querying the records.

    This is what replaces hardcoded /dbfs paths in the notebooks: a pruning run asks
    for its dense baseline by name and gets the one that actually finished and whose
    file still exists.

    `allow_smoke` is the one exception, and it is directional. A smoke cell
    chaining from a smoke checkpoint is the whole point of 00_smoke_test_all --
    it checks the prune and BaCP pipelines end to end on 2 batches. A REAL cell
    resolving a smoke checkpoint is the silent-wrong-number hazard this guard
    exists to stop. So callers pass allow_smoke=True only when the cell asking
    is itself a smoke cell; the default refuses.
    """
    out_root = results_dir(root)
    candidates = []
    smoke_skipped = 0
    for payload in iter_records(out_root):
        if payload.get('status') != 'ok':
            continue
        if payload.get('model_name') != model_name:
            continue
        if payload.get('dataset_name') != dataset_name:
            continue
        if payload.get('method') != method:
            continue
        if seed is not None and payload.get('seed') != seed:
            continue
        # A smoke checkpoint is 2 epochs over 2 batches. Seeding a real sparse run
        # from one does not fail -- it trains, converges to something, and writes a
        # record indistinguishable from a good one. A plausible wrong number is
        # worse than an error, and the notebook README tells people to run
        # 00_smoke_test_all before "any cell, in any order", which is this path.
        if is_smoke_record(payload) and not allow_smoke:
            smoke_skipped += 1
            continue
        candidates.append(payload)

    # Candidate roots for a RELATIVE save_path. Records written before paths were
    # absolutised say './research/bacp/...', which is only meaningful from the
    # directory the training script ran in -- project/scripts for a runner-driven
    # sweep, the repo root for a hand-run script, project/ for a notebook.
    roots = [Path.cwd(), repo_root(), repo_root() / 'project',
             repo_root() / 'project' / 'scripts']

    for payload in sorted(candidates, key=lambda p: p.get('ended_at') or '', reverse=True):
        path = payload.get('save_path')
        if not path:
            continue
        candidate = Path(path)
        if candidate.is_file():
            return str(candidate.resolve())
        if not candidate.is_absolute():
            for root in roots:
                rehydrated = (root / candidate).resolve()
                if rehydrated.is_file():
                    print(f'[RESULTS] rehydrated a relative checkpoint path '
                          f'against {root}')
                    return str(rehydrated)

    raise FileNotFoundError(
        f'No completed {method} checkpoint for {model_name}/{dataset_name}'
        + (f' seed={seed}' if seed is not None else '')
        + f'. {len(candidates)} matching record(s), none with a file on disk'
        + (f' ({smoke_skipped} smoke record(s) ignored)' if smoke_skipped else '')
        + '. Run the model\'s own notebook first: '
        f'project/test_notebooks/<family>/{model_name}/{model_name}.ipynb'
    )


# --- gates ----------------------------------------------------------------

def register_gate(name: str, checks, root=None) -> dict:
    """Record whether a notebook's preconditions held.

    `checks` is a list of (label, passed, detail). The code fingerprint is stored so
    a downstream gate can detect that the code changed after this one passed.
    """
    out_root = results_dir(root)
    _ensure_dirs(out_root)

    normalised = [{'label': str(l), 'ok': bool(ok), 'detail': str(d)}
                  for l, ok, d in checks]
    payload = {
        'name': name,
        'ok': all(c['ok'] for c in normalised),
        'checks': normalised,
        'git_sha': capture_git().get('git_sha'),
        'code_fingerprint': code_fingerprint(),
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(out_root / 'gates' / f'{name}.json', payload)

    status = 'PASS' if payload['ok'] else 'FAIL'
    print(f'[GATE {status}] {name}')
    for c in normalised:
        print(f"   {'ok ' if c['ok'] else 'FAIL'}  {c['label']}: {c['detail']}")
    return payload


def require_gate(name: str, *, allow_stale=False, root=None) -> dict:
    """Raise unless the named gate passed against the current code.

    The staleness check is the point: it stops a 40-hour result silently satisfying
    a gate after the code underneath it changed.
    """
    path = results_dir(root) / 'gates' / f'{name}.json'
    if not path.exists():
        raise RuntimeError(f"Gate '{name}' has not run. Run that notebook first.")

    payload = json.loads(path.read_text(encoding='utf-8'))
    if not payload.get('ok'):
        failed = [c['label'] for c in payload.get('checks', []) if not c['ok']]
        raise RuntimeError(f"Gate '{name}' failed: {failed}")

    current = code_fingerprint()
    if not allow_stale and payload.get('code_fingerprint') != current:
        raise RuntimeError(
            f"Gate '{name}' is stale: project code changed since it passed "
            f"({payload.get('code_fingerprint', '?')[:12]} -> {current[:12]}). "
            f"Re-run that notebook, or pass allow_stale=True if you are certain "
            f"the change cannot affect it."
        )
    print(f"[GATE ok] {name}")
    return payload
