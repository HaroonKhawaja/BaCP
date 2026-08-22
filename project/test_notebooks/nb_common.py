"""Shared helpers for the per-family test notebooks.

The notebooks in vision_models/, vision_transformers/ and llms/ all do the same
five things, so they live here once:

    setup()                  find the repo, set paths, print the hardware
    fetch_imagenet_weights() put real ImageNet weights where the registry looks
    preflight()              REFUSE to train if the init is not what was asked
    make_cell()              one test = one config dict, from the paper protocol
    run() / run_group()      run it, streaming every epoch into the cell
    results_table()          our numbers next to the paper's published ones

Protocols come from the original paper (Paper/BaCP__Appendix.pdf, Tables 1-3)
with ONE declared deviation: the pruning regime is modernized to the standard
gradual schedule (total epochs + prune every delta_T steps, cubic ramp to 80%,
final 20% recovery) -- see the TRAINING REGIME note below. Static pruning at
0.95 / 0.97 / 0.99 from a PRETRAINED model. Nothing here touches the
experiment code.

This file lives under test_notebooks/, which results.code_fingerprint excludes,
so editing it never invalidates a run record.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# --- repo bootstrap ---------------------------------------------------------

def find_repo():
    """Walk up from this file to the repo root.

    Prefer the directory holding .git. Fall back to the one holding
    project/model_factory.py, because Databricks Git folders do NOT expose
    .git through the workspace filesystem -- requiring it would crash this
    module at import time on exactly the machine it is meant to run on.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / '.git').exists():
            return parent
    for parent in here.parents:
        if (parent / 'project' / 'model_factory.py').exists():
            return parent
    raise RuntimeError('could not find the repo root above ' + str(here))


REPO = find_repo()

for _p in (REPO / 'project', REPO / 'project' / 'scripts'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# --- parsing training output ------------------------------------------------

# Matches "Epoch [3/50]" and the prefixed variants "Recovery Epoch [1/10]" /
# "Fine-tuning Epoch [2/50]" that trainer.py and bacp.py print each epoch.
_EPOCH = re.compile(r'Epoch \[(\d+)/(\d+)\]')


def _field(name, line):
    """Pull 'name: <number>' out of a log line, or None.

    \\b keeps 'loss:' from matching inside 'val_loss:' (underscore is a word
    char, so there is no boundary there), and IGNORECASE lets the same name
    hit both trainer formats: 'loss:'/'accuracy:' from Trainer and
    'Total Loss:'/'Training Accuracy:' from BaCPTrainer.
    """
    m = re.search(rf'\b{name}: (nan|-?[0-9.]+(?:[eE][-+]?\d+)?)', line,
                  re.IGNORECASE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return float('nan')


def on_databricks():
    """True when running on a Databricks cluster."""
    return bool(os.environ.get('DATABRICKS_RUNTIME_VERSION'))


def setup(quiet=False):
    """Set the environment up and print what machine we are on.

    Results go to BACP_RESULTS_DIR. On Databricks that defaults to
    /dbfs/research/bacp_results so records survive the cluster; anywhere else
    it is <repo>/results.
    """
    if 'BACP_RESULTS_DIR' not in os.environ:
        if on_databricks():
            os.environ['BACP_RESULTS_DIR'] = '/dbfs/research/bacp_results'
        else:
            os.environ['BACP_RESULTS_DIR'] = str(REPO / 'results')
    os.environ['PYTHONUNBUFFERED'] = '1'

    info = {'repo': str(REPO),
            'results': os.environ['BACP_RESULTS_DIR'],
            'databricks': on_databricks()}
    try:
        import torch
        info['torch'] = torch.__version__
        info['cuda'] = torch.cuda.is_available()
        info['gpus'] = torch.cuda.device_count() if info['cuda'] else 0
        if info['gpus']:
            info['gpu_name'] = torch.cuda.get_device_name(0)
    except Exception as exc:                                       # noqa: BLE001
        info['torch_error'] = str(exc)

    if not quiet:
        print(f"repo       {info['repo']}")
        print(f"results    {info['results']}")
        print(f"databricks {info['databricks']}")
        if 'torch' in info:
            print(f"torch      {info['torch']}  cuda={info['cuda']}  "
                  f"gpus={info['gpus']} {info.get('gpu_name', '')}")
        else:
            print(f"!! torch not importable: {info.get('torch_error')}")
    return info


# --- the protocols, one dict per model --------------------------------------
#
# Source: Paper/BaCP__Appendix.pdf Tables 1-3, plus the main paper's Training
# Protocol paragraph (Section 4.1) and the original sweep scripts
# (project/scripts/sweep_scripts/), which encode the argv actually used.
#
#   base   keys shared by every phase of that model
#   dense  the pretrained baseline, finetuned on the task   (baseline_script)
#   prune  I.P.: iterative prune + recovery                  (pruning_script)
#   bacp   contrastive pruning + AdamW finetune              (bacp_script)
#
# TRAINING REGIME (deliberately modernized -- a declared deviation from the
# appendix's "5 epochs + 10 recovery", whose code interleaved a 10-epoch
# recovery block after EVERY pruning epoch, a shape no current literature
# uses). The standard gradual-pruning regime (Zhu & Gupta 2017,
# arXiv 1710.01878; Gale, Elsen & Hooker 2019, arXiv 1902.09574) is one
# continuous run:
#
#   epochs           total epoch budget for the pruning phase
#   delta_T          prune every delta_T optimizer steps (~1 epoch here)
#   cubic ramp       sparsity rises to the target over the FIRST 80% of
#                    training (BasePruner: end_idx = 0.8 * total_steps)
#   final 20%        recovery: mask fixed at the target, training continues
#   recovery_epochs  0 -- the interleaved per-epoch recovery is disabled;
#                    recovery is the schedule's built-in tail instead
#
# delta_T is set to ~one epoch of optimizer steps per family:
#   resnet/vgg  45,000/512 ~ 88    vit  45,000/256 ~ 176    llm  67,349/64 ~ 1052

# num_workers=24: the Standard_NC24ads_A100_v4 node has 24 vCPUs, and one
# training job owns the box (runs are serialised), so the loaders get all of
# them. enable_tqdm=True streams a per-batch progress bar; both nb_common.run
# and runner.run_cell merge stderr into stdout, so the bar reaches the cell
# and the job log.
_CV_BASE = dict(model_type='cv', dataset_name='cifar10', num_classes=10,
                image_size=32, batch_size=512, num_workers=24,
                enable_tqdm=True)

_CNN_DENSE = dict(optimizer_type='sgd', learning_rate=0.01,
                  scheduler_type='linear_with_warmup', epochs=100, patience=20)
# prune_task_head=True for the vision families: the published vision-line
# comparators prune their output layers (Han et al. prune AlexNet's fc8; SNIP
# thresholds globally over all weights; RigL/GraNet mask the classifier by
# default -- see docs/citations.md section 4). The encoder/projection head
# remains excluded unconditionally (frozen, discarded, never in the
# denominator). LLM families keep heads dense -- the PLM convention of CAP
# and WANDA-as-published.
_CNN_PRUNE = dict(optimizer_type='sgd', learning_rate=0.01,
                  epochs=60, recovery_epochs=0,
                  sparsity_scheduler='cubic', delta_T=88,
                  prune_task_head=True,
                  # wanda_group='global': the project's original adaptation --
                  # Wanda's metric under the same global-threshold allocation
                  # as magnitude and SNIP, so the columns differ only in
                  # score. Ignored by the other pruners.
                  wanda_group='global')
_CNN_BACP = dict(optimizer_type='sgd', learning_rate=0.1,
                 epochs=60, recovery_epochs=0,
                 sparsity_scheduler='cubic', delta_T=88,
                 prune_task_head=True, wanda_group='global',
                 # tau=0.15. Measured under the corrected rectangular CAP
                 # loss: 0.15 beats 0.07 at every sparsity, in both the
                 # head-dense and head-pruned scopes (ResNet-50/CIFAR-10
                 # magnitude trios, 2026-08-21). The old operative 0.07 was
                 # calibrated for the buggy square loss, which halved the
                 # effective contrastive scale.
                 tau=0.15, lambdas=(0.25, 0.25, 0.25, 0.25),
                 enable_finetune=True, optimizer_type_ft='adamw',
                 learning_rate_ft=1e-4, epochs_ft=50)

_VIT_BASE = dict(model_type='cv', dataset_name='cifar10', num_classes=10,
                 image_size=224, batch_size=256, num_workers=24,
                 enable_tqdm=True)

_LLM_BASE = dict(model_type='llm', dataset_name='sst2', num_classes=2,
                 batch_size=64, num_workers=24, enable_tqdm=True)

FAMILIES = {
    'resnet34': dict(base=_CV_BASE, dense=_CNN_DENSE, prune=_CNN_PRUNE,
                     bacp=_CNN_BACP),
    'resnet50': dict(base=_CV_BASE, dense=_CNN_DENSE, prune=_CNN_PRUNE,
                     bacp=_CNN_BACP),
    'resnet101': dict(base=_CV_BASE, dense=_CNN_DENSE, prune=_CNN_PRUNE,
                      bacp=_CNN_BACP),
    'vgg11': dict(base=_CV_BASE, dense=_CNN_DENSE, prune=_CNN_PRUNE,
                  bacp=_CNN_BACP),
    # VGG-19's BaCP LR is 0.05, not 0.1: "reduced to prevent gradient
    # explosion due to its depth" (appendix Table 1).
    'vgg19': dict(base=_CV_BASE, dense=_CNN_DENSE, prune=_CNN_PRUNE,
                  bacp=dict(_CNN_BACP, learning_rate=0.05)),
    # ViTs: hub weights, 224px inputs, batch 256. I.P. uses AdamW 1e-3 and
    # BaCP's contrastive phase SGD 0.01 (appendix B.5 note).
    'vit-tiny': dict(base=_VIT_BASE, dense=_CNN_DENSE,
                     prune=dict(_CNN_PRUNE, optimizer_type='adamw',
                                learning_rate=1e-3, delta_T=176),
                     bacp=dict(_CNN_BACP, learning_rate=0.01, delta_T=176)),
    'vit-small': dict(base=_VIT_BASE, dense=_CNN_DENSE,
                      prune=dict(_CNN_PRUNE, optimizer_type='adamw',
                                 learning_rate=1e-3, delta_T=176),
                      bacp=dict(_CNN_BACP, learning_rate=0.01, delta_T=176)),
    # LLMs on SST-2. Dense finetune AdamW 2e-5; I.P. AdamW 5e-5; BaCP's
    # contrastive LR from appendix Table 3 (DistilBERT 5e-5, RoBERTa 1e-5),
    # finetune AdamW 1e-4. GLUE's test split is unlabelled, so the reported
    # "test" accuracy is the validation set -- state that wherever it appears.
    'distilbert-base-uncased': dict(
        base=_LLM_BASE,
        dense=dict(optimizer_type='adamw', learning_rate=2e-5,
                   scheduler_type='linear_with_warmup', epochs=5),
        prune=dict(optimizer_type='adamw', learning_rate=5e-5,
                   epochs=15, recovery_epochs=0,
                   sparsity_scheduler='cubic', delta_T=1052),
        bacp=dict(optimizer_type='adamw', learning_rate=5e-5,
                  epochs=15, recovery_epochs=0,
                  sparsity_scheduler='cubic', delta_T=1052,
                  tau=0.15, lambdas=(0.25, 0.25, 0.25, 0.25),
                  enable_finetune=True, optimizer_type_ft='adamw',
                  learning_rate_ft=1e-4, epochs_ft=10)),
    'roberta-base': dict(
        base=_LLM_BASE,
        dense=dict(optimizer_type='adamw', learning_rate=2e-5,
                   scheduler_type='linear_with_warmup', epochs=5),
        prune=dict(optimizer_type='adamw', learning_rate=5e-5,
                   epochs=15, recovery_epochs=0,
                   sparsity_scheduler='cubic', delta_T=1052),
        bacp=dict(optimizer_type='adamw', learning_rate=1e-5,
                  epochs=15, recovery_epochs=0,
                  sparsity_scheduler='cubic', delta_T=1052,
                  tau=0.15, lambdas=(0.25, 0.25, 0.25, 0.25),
                  enable_finetune=True, optimizer_type_ft='adamw',
                  learning_rate_ft=1e-4, epochs_ft=10)),
}

SPARSITIES = (0.95, 0.97, 0.99)
PRUNERS = ('magnitude', 'snip', 'wanda')     # names in PRUNER_REGISTRY


# --- published numbers (main paper, Table 1; CIFAR-10 / SST-2 rows) ---------
#
# {(model, dataset): {'dense': acc, (pruner, sparsity): (I.P., BaCP)}}
# resnet34 has no published row; the notebooks say so instead of borrowing one.

PAPER = {
    ('resnet50', 'cifar10'): {
        'dense': 93.02,
        ('magnitude', 0.95): (91.97, 93.58), ('snip', 0.95): (90.77, 93.12), ('wanda', 0.95): (91.78, 93.15),
        ('magnitude', 0.97): (91.39, 93.27), ('snip', 0.97): (90.58, 92.71), ('wanda', 0.97): (90.81, 93.02),
        ('magnitude', 0.99): (89.87, 92.32), ('snip', 0.99): (88.18, 91.90), ('wanda', 0.99): (86.59, 90.87),
    },
    ('resnet101', 'cifar10'): {
        'dense': 91.80,
        ('magnitude', 0.95): (90.43, 93.23), ('snip', 0.95): (90.61, 93.75), ('wanda', 0.95): (90.09, 93.09),
        ('magnitude', 0.97): (90.22, 92.90), ('snip', 0.97): (89.86, 92.64), ('wanda', 0.97): (89.67, 92.75),
        ('magnitude', 0.99): (89.51, 92.70), ('snip', 0.99): (87.83, 92.20), ('wanda', 0.99): (86.46, 91.92),
    },
    ('vgg11', 'cifar10'): {
        'dense': 89.94,
        ('magnitude', 0.95): (90.10, 90.44), ('snip', 0.95): (89.60, 90.37), ('wanda', 0.95): (89.49, 90.30),
        ('magnitude', 0.97): (89.61, 90.65), ('snip', 0.97): (89.64, 90.46), ('wanda', 0.97): (89.04, 90.23),
        ('magnitude', 0.99): (88.97, 90.44), ('snip', 0.99): (88.24, 90.11), ('wanda', 0.99): (88.28, 89.06),
    },
    ('vgg19', 'cifar10'): {
        'dense': 91.78,
        ('magnitude', 0.95): (91.53, 92.57), ('snip', 0.95): (91.52, 92.47), ('wanda', 0.95): (91.46, 92.18),
        ('magnitude', 0.97): (91.66, 92.60), ('snip', 0.97): (91.34, 91.40), ('wanda', 0.97): (90.90, 91.80),
        ('magnitude', 0.99): (90.56, 92.12), ('snip', 0.99): (89.74, 84.76), ('wanda', 0.99): (90.50, 91.79),
    },
    ('vit-tiny', 'cifar10'): {
        'dense': 97.46,
        ('magnitude', 0.95): (85.88, 91.57), ('snip', 0.95): (84.46, 90.43), ('wanda', 0.95): (72.48, 85.59),
        ('magnitude', 0.97): (84.12, 89.78), ('snip', 0.97): (82.64, 88.40), ('wanda', 0.97): (72.04, 83.19),
        ('magnitude', 0.99): (74.21, 87.01), ('snip', 0.99): (77.60, 85.00), ('wanda', 0.99): (69.79, 81.25),
    },
    ('vit-small', 'cifar10'): {
        'dense': 98.40,
        ('magnitude', 0.95): (91.19, 92.22), ('snip', 0.95): (91.93, 88.84), ('wanda', 0.95): (88.14, 87.78),
        ('magnitude', 0.97): (90.58, 92.39), ('snip', 0.97): (90.72, 89.28), ('wanda', 0.97): (84.86, 90.02),
        ('magnitude', 0.99): (86.28, 90.67), ('snip', 0.99): (88.11, 89.07), ('wanda', 0.99): (80.32, 81.76),
    },
    ('distilbert-base-uncased', 'sst2'): {
        'dense': 91.11,
        ('magnitude', 0.95): (82.57, 85.46), ('snip', 0.95): (82.09, 84.50), ('wanda', 0.95): (81.97, 84.86),
        ('magnitude', 0.97): (81.25, 83.29), ('snip', 0.97): (80.05, 84.25), ('wanda', 0.97): (79.93, 82.21),
        ('magnitude', 0.99): (80.53, 82.33), ('snip', 0.99): (79.93, 81.25), ('wanda', 0.99): (80.41, 81.61),
    },
    ('roberta-base', 'sst2'): {
        'dense': 94.83,
        ('magnitude', 0.95): (81.85, 83.41), ('snip', 0.95): (87.38, 87.86), ('wanda', 0.95): (82.45, 84.86),
        ('magnitude', 0.97): (80.77, 83.17), ('snip', 0.97): (85.82, 86.06), ('wanda', 0.97): (81.49, 83.05),
        ('magnitude', 0.99): (78.12, 82.57), ('snip', 0.99): (81.41, 84.38), ('wanda', 0.99): (80.29, 82.09),
    },
}


# --- weights ---------------------------------------------------------------

# torchvision constructors matching the local architectures key-for-key
# (verified: identical state_dict keys and shapes).
_TORCHVISION = {
    'resnet34': ('resnet34', 'ResNet34_Weights'),
    'resnet50': ('resnet50', 'ResNet50_Weights'),
    'resnet101': ('resnet101', 'ResNet101_Weights'),
    'vgg11': ('vgg11', 'VGG11_Weights'),
    'vgg19': ('vgg19', 'VGG19_Weights'),
}


def fetch_imagenet_weights(model_name):
    """Put real ImageNet weights at the exact path the registry expects.

    The registry's weight paths point into /dbfs, and load_weights fails SOFT:
    a missing file warns and the run silently continues from random init. This
    function downloads the torchvision ImageNet state_dict and saves it where
    the registry looks, so 'pretrained' actually means pretrained. HF models
    (ViT, BERT) skip this -- their weights come from the hub automatically.
    """
    from model_factory import MODELS
    spec = MODELS[model_name]

    if spec.weight == '':
        print(f'{model_name}: weights come from the HF hub, nothing to fetch')
        return None
    if os.path.exists(spec.weight):
        mb = os.path.getsize(spec.weight) / 1e6
        print(f'{model_name}: weights already at {spec.weight} ({mb:.0f} MB)')
        return spec.weight

    import torch
    import torchvision.models as tvm
    ctor_name, weights_enum = _TORCHVISION[model_name]
    print(f'{model_name}: downloading torchvision ImageNet weights...')
    weights = getattr(tvm, weights_enum).IMAGENET1K_V1
    state = getattr(tvm, ctor_name)(weights=weights).state_dict()

    path = Path(spec.weight)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f'{model_name}: saved {len(state)} tensors -> {path}')
    return str(path)


def preflight(model_name, num_classes):
    """Build the model and REFUSE to continue on the wrong init.

    This is the check whose absence cost a day of training: load_weights fails
    soft, so a run that asked for pretrained weights can silently train from
    random init. Run this cell before spending any GPU time.
    """
    from model_factory import initialize_model_components, MODELS
    comps = initialize_model_components(model_name, pretrained=True,
                                        dyrelu_en=False,
                                        dyrelu_phasing_en=False,
                                        num_classes=num_classes)
    expected = 'hf_hub' if MODELS[model_name].weight == '' else 'imagenet_checkpoint'
    got = comps['init_source']
    n_params = sum(p.numel() for p in comps['model'].parameters())
    print(f'{model_name}: {n_params/1e6:.1f}M params, init_source={got}')

    if got != expected:
        raise RuntimeError(
            f'\n{"!" * 70}\n'
            f'{model_name} initialized from {got!r}, expected {expected!r}.\n'
            f'Training now would produce a "pretrained" baseline that was\n'
            f'never pretrained. Run fetch_imagenet_weights({model_name!r})\n'
            f'first, then re-run this cell.\n{"!" * 70}')
    print('preflight OK: the model is actually pretrained')
    return {'init_source': got, 'params': n_params}


# --- building one test ------------------------------------------------------

def make_cell(model_name, phase, seed=1, pruner=None, sparsity=None,
              smoke=False, variant=None, **overrides):
    """One test as a runner-compatible cell dict.

    phase: 'dense' (baseline_script), 'prune' (pruning_script, the I.P.
    baseline) or 'bacp' (bacp_script). Sparse phases need `pruner` and
    `sparsity`. `smoke=True` shrinks everything to a 2-batch pipeline check.
    Any extra keyword overrides the config (e.g. epochs=3).

    variant: optional key suffix (e.g. 'legacy') so an ablation arm's records
    coexist with the main table's instead of silently satisfying its keys --
    the same mechanism as the .smoke suffix.
    """
    if model_name not in FAMILIES:
        raise KeyError(f'{model_name!r} not in FAMILIES: {sorted(FAMILIES)}')
    if phase not in ('dense', 'prune', 'bacp'):
        raise ValueError(f"phase must be dense|prune|bacp, got {phase!r}")

    family = FAMILIES[model_name]
    config = dict(family['base'])
    config.update(family[phase])
    config['model_name'] = model_name
    config['experiment_type'] = f'static-{phase}'
    if on_databricks():
        config['databricks_env'] = True      # checkpoints under /dbfs

    if phase == 'dense':
        tag = 'dense'
    else:
        if pruner is None or sparsity is None:
            raise ValueError(f'phase {phase!r} needs pruner= and sparsity=')
        config['pruning_type'] = pruner
        config['target_sparsity'] = sparsity
        tag = f's{sparsity}.{pruner}'

    if smoke:
        config.update(epochs=2, limit_train_batches=2, limit_eval_batches=2,
                      num_workers=0)
        if phase == 'bacp':
            config.update(epochs_ft=2, recovery_epochs=1)
        elif phase == 'prune':
            config['recovery_epochs'] = 1

    config.update(overrides)

    dataset = config['dataset_name']
    key = f'static.{phase}.{model_name}.{dataset}.{tag}.seed{seed}'
    if variant:
        key += f'.{variant}'
    if smoke:
        key += '.smoke'

    return {
        'key': key,
        'script': 'baseline' if phase == 'dense' else
                  ('pruning' if phase == 'prune' else 'bacp'),
        'rung': f'static-{phase}',
        'label': f'{model_name} {phase} {tag}' + (f' [{variant}]' if variant else ''),
        'changes': '',
        'stage': 'static',
        'seed': seed,
        'model_name': model_name,
        'dataset_name': dataset,
        'sparsity': sparsity,
        'config': config,
    }


# --- sanity check -----------------------------------------------------------

def sanity_check(cells, control=None):
    """Verify a batch of cells BEFORE any GPU time is spent. Returns True if
    everything passes.

    Prints three sections, in this order:

      1. TRAINING PROTOCOL  epochs, delta_T, schedule, recovery, optimiser/LR
                            per phase, tau, lambdas, prunable scope.
      2. THE RUN            model, phase, pruner, sparsity, seed, record key,
                            whether that record already EXISTS (would skip),
                            and -- when `control` is given -- the exact set of
                            config keys that differ from it.
      3. THE DATASET        name, splits, batch size, image size, classes, and
                            which split the reported accuracy comes from.

    A FAIL is a stop: fix it before submitting.
    """
    import runner as R

    cells = list(cells)
    if not cells:
        print('SANITY: no cells given'); return False
    done = R.completed_keys()
    ok = True

    # -- 1. protocol ---------------------------------------------------------
    print('=' * 78)
    print('1. TRAINING PROTOCOL')
    print('=' * 78)
    c0 = cells[0]['config']
    phase = cells[0]['rung'].replace('static-', '')
    keys = ['epochs', 'delta_T', 'sparsity_scheduler', 'recovery_epochs',
            'optimizer_type', 'learning_rate', 'patience',
            'tau', 'lambdas', 'contrastive_mode', 'proj_mode', 'num_snapshots',
            'enable_finetune', 'optimizer_type_ft', 'learning_rate_ft',
            'epochs_ft', 'prune_task_head', 'prune_embeddings', 'wanda_group',
            'num_workers', 'enable_tqdm']
    for k in keys:
        if k in c0:
            print(f'   {k:<20} {c0[k]}')
    if phase in ('prune', 'bacp'):
        ep, dt = c0.get('epochs'), c0.get('delta_T')
        print(f'   -> ramp: cubic to target over first 80% of {ep} epochs, '
              f'final 20% recovery at fixed mask; mask update every {dt} steps')
        if c0.get('recovery_epochs'):
            print('   FAIL  recovery_epochs != 0 re-enables the legacy '
                  'interleaved recovery block'); ok = False

    # -- 2. the runs ---------------------------------------------------------
    print()
    print('=' * 78)
    print('2. THE RUN(S)')
    print('=' * 78)
    for c in cells:
        exists = c['key'] in done
        mark = 'EXISTS -> WILL SKIP' if exists else 'will run'
        print(f'   {c["key"]}')
        print(f'      model={c["model_name"]} phase={c["rung"]} '
              f'pruner={c["config"].get("pruning_type", "-")} '
              f'sparsity={c["sparsity"]} seed={c["seed"]}  [{mark}]')
        if exists:
            print('      NOTE  delete its record to force a re-run')
    if control is not None:
        a, b = cells[0]['config'], control['config']
        diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        print()
        print(f'   ablation vs control {control["key"]}')
        print(f'      keys that differ: {diff}')
        for k in diff:
            print(f'         {k}: control={b.get(k)!r} -> arm={a.get(k)!r}')
        if not diff:
            print('      FAIL  nothing differs; this is not an ablation'); ok = False

    # -- 3. the dataset ------------------------------------------------------
    print()
    print('=' * 78)
    print('3. THE DATASET')
    print('=' * 78)
    ds = c0.get('dataset_name')
    print(f'   dataset       {ds}')
    print(f'   batch_size    {c0.get("batch_size")}')
    print(f'   num_classes   {c0.get("num_classes")}')
    if 'image_size' in c0:
        print(f'   image_size    {c0["image_size"]}')
    if ds == 'cifar10':
        print('   splits        50,000 train -> 40,000 train / 10,000 val '
              '(0.2 split); 10,000 test')
        print('   reported acc  the 10,000-image TEST split, all of it '
              '(drop_last=False)')
    elif ds == 'sst2':
        print('   splits        67,349 train / 872 validation / 1,821 test')
        print('   reported acc  the VALIDATION split -- GLUE test labels are '
              'not public. State this wherever the number appears.')
    else:
        print('   splits        (not described here; verify before reporting)')

    print()
    print('=' * 78)
    print('SANITY: PASS' if ok else 'SANITY: FAIL -- fix before submitting')
    print('=' * 78)
    return ok


# --- running ---------------------------------------------------------------

def _log_path(key):
    from results import results_dir
    d = results_dir() / 'logs'
    d.mkdir(parents=True, exist_ok=True)
    return d / f'{key}.log'


def run(cell, gpu=0):
    """Run one cell, streaming every epoch into the notebook output.

    Skips cells that already have an ok run record -- delete the record under
    results/runs/ to re-run. Output is also written to results/logs/<key>.log.
    Ctrl-C / interrupt kills the training subprocess cleanly.
    """
    import runner as R

    if cell['key'] in R.completed_keys():
        print(f"SKIP {cell['key']} -- already recorded. "
              f"Delete its record in results/runs/ to re-run.")
        return None

    # Sparse cells start from the dense checkpoint of the SAME seed.
    if cell['script'] != 'baseline':
        cell = R.attach_checkpoint(cell)
        print(f"checkpoint  {cell['config'].get('trained_weights')}")

    argv = R.build_command(cell, python=sys.executable)
    env = {**os.environ,
           'BACP_EXPERIMENT_GROUP': cell['key'],
           'CUDA_VISIBLE_DEVICES': str(gpu),
           'PYTHONUNBUFFERED': '1',
           'MPLBACKEND': 'Agg'}

    print(f"RUN  {cell['key']}   (gpu {gpu})")
    hist = {'epoch': [], 'acc': [], 'loss': [], 'sparsity': []}
    first_nan = None
    t_prev = time.perf_counter()

    log_file = _log_path(cell['key']).open('w', encoding='utf-8',
                                           errors='replace')
    log_file.write(f"=== {cell['key']}\n=== {' '.join(argv)}\n\n")

    proc = subprocess.Popen(argv, cwd=str(REPO / 'project' / 'scripts'),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, env=env,
                            encoding='utf-8', errors='replace')
    try:
        for raw in proc.stdout:
            line = raw.rstrip()
            log_file.write(raw)
            m = _EPOCH.search(line)
            if not m:
                if line:
                    print(line)
                continue

            ep, tot = int(m.group(1)), int(m.group(2))
            acc = _field('accuracy', line)
            loss = _field('loss', line)
            sp = _field('sparsity', line)
            now = time.perf_counter()
            per, t_prev = now - t_prev, now

            hist['epoch'].append(ep)
            hist['acc'].append(acc)
            hist['loss'].append(loss)
            hist['sparsity'].append(sp)

            phase_tag = line.split('Epoch')[0].strip() or 'train'
            acc_s = f'{acc:6.2f}' if acc is not None else '     -'
            loss_s = f'{loss:9.4f}' if loss is not None else '        -'
            sp_s = f'{sp:.4f}' if sp is not None else '-'
            print(f'{phase_tag:<12} {ep:4d}/{tot}  acc {acc_s}  loss {loss_s}'
                  f'  sparsity {sp_s}  [{per:5.1f}s/ep'
                  f'  eta {per * (tot - ep) / 60:5.1f}m]')

            if first_nan is None and loss is not None and loss != loss:
                first_nan = ep
                print(f'*** LOSS WENT NaN AT EPOCH {ep}/{tot} ***')
    except KeyboardInterrupt:
        proc.terminate()
        print('interrupted -- subprocess terminated')
    finally:
        proc.wait()
        log_file.close()

    ok = cell['key'] in R.completed_keys()
    if proc.returncode == 0 and ok:
        print(f"DONE {cell['key']}")
    elif proc.returncode == 0:
        print(f"!! exited 0 but wrote no record -- check "
              f"results/logs/{cell['key']}.log")
    else:
        # A failure is a row, on this path too: without the record, a crashed
        # run is indistinguishable from one that was never scheduled.
        try:
            from results import record_failure
            tail = _log_path(cell['key']).read_text(encoding='utf-8',
                                                    errors='replace')[-4000:]
            record_failure(cell, returncode=proc.returncode, log_tail=tail,
                           experiment_group=cell['key'])
        except Exception as exc:                                   # noqa: BLE001
            print(f'(could not write the failure record: {exc})')
        print(f"FAILED rc={proc.returncode} -- see "
              f"results/logs/{cell['key']}.log")
    return hist, first_nan


def run_group(cells, gpu=0):
    """Run several cells back to back, then print a one-line-each summary."""
    outcomes = []
    for cell in cells:
        out = run(cell, gpu=gpu)
        if out is None:
            outcomes.append((cell['key'], 'skipped (already recorded)'))
            continue
        hist, first_nan = out
        accs = [a for a in hist['acc'] if a is not None]
        best = f'best acc {max(accs):.2f}' if accs else 'no acc parsed'
        nan = f', NaN at epoch {first_nan}' if first_nan else ''
        outcomes.append((cell['key'], best + nan))
    print('\n--- group summary ---')
    for key, verdict in outcomes:
        print(f'  {key:<60} {verdict}')
    return outcomes


# --- results ----------------------------------------------------------------

# Which record phase carries the final number, best first.
_PHASE_PRIORITY = ('finetune', 'prune', 'dense', 'contrastive')


def _our_numbers(model_name, dataset):
    """{(phase, tag): {seed: acc}} from the ok run records of this model."""
    from results import iter_records
    prefix = 'static.'
    out = {}
    for rec in iter_records():
        group = rec.get('experiment_group') or ''
        if rec.get('status') != 'ok' or not group.startswith(prefix):
            continue
        parts = group.split('.')
        # static.<phase>.<model>.<dataset>.<tag...>.seed<n>[.smoke]
        if 'smoke' in parts or model_name not in group or dataset not in group:
            continue
        acc = rec.get('test_acc_pct')
        if acc is None:
            continue
        phase = rec.get('phase')
        prio = (_PHASE_PRIORITY.index(phase)
                if phase in _PHASE_PRIORITY else len(_PHASE_PRIORITY))
        best = out.setdefault(group, (prio, acc))
        if prio < best[0]:
            out[group] = (prio, acc)
    return {k: v[1] for k, v in out.items()}


def results_table(model_name, dataset=None):
    """Print our accuracy per pruner x sparsity next to the paper's numbers."""
    dataset = dataset or FAMILIES[model_name]['base']['dataset_name']
    ours = _our_numbers(model_name, dataset)
    paper = PAPER.get((model_name, dataset), {})

    def mean(keys):
        vals = [ours[k] for k in keys if k in ours]
        return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)

    def find(phase, tag):
        pre = f'static.{phase}.{model_name}.{dataset}.{tag}.'
        return [k for k in ours if k.startswith(pre)]

    def fmt(x):
        return f'{x:6.2f}' if x is not None else '     -'

    print(f'{model_name} / {dataset} -- ours vs paper Table 1 '
          f'(- means not run / not published)')
    print(f'{"test":<24} {"ours":>8} {"n":>2} {"paper I.P.":>10} '
          f'{"paper BaCP":>10}')

    acc, n = mean(find('dense', 'dense'))
    print(f'{"dense":<24} {fmt(acc):>8} {n:>2} '
          f'{fmt(paper.get("dense")):>10} {"":>10}')

    for pruner in PRUNERS:
        for sp in SPARSITIES:
            pub = paper.get((pruner, sp), (None, None))
            tag = f's{sp}.{pruner}'
            acc, n = mean(find('prune', tag))
            print(f'{"I.P. " + pruner + " @ " + str(sp):<24} {fmt(acc):>8} '
                  f'{n:>2} {fmt(pub[0]):>10} {"":>10}')
            acc, n = mean(find('bacp', tag))
            print(f'{"BaCP " + pruner + " @ " + str(sp):<24} {fmt(acc):>8} '
                  f'{n:>2} {"":>10} {fmt(pub[1]):>10}')
    if (model_name, dataset) not in PAPER:
        print('note: no published row for this model/dataset in the paper '
              '-- the nearest comparator is stated in the notebook header.')
