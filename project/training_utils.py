import os
import logging
from typing import Optional, Dict, Any
from copy import deepcopy
import time
from datetime import datetime, timezone

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR

from logger import Logger
from utils import get_device, load_weights, set_seed
from dataset_factory import get_dataloaders
from model_factory import ClassificationAndEncoderNetwork
from dyrelu_adapter import set_t_for_dyrelu_adapter
from weight_sharing import apply_weight_sharing_resnet
from tqdm import tqdm

from pruning_factory import (
    get_pruner,
    check_model_sparsity,
    report_sparsity,
    set_prunable_scope,
)

def _create_base_model(args, is_main_model=False):
    """Factory function for creating a single model instance."""
    adapt = args.image_size <= 64
    dyrelu_en = args.dyrelu_en if is_main_model else False
    dyrelu_phasing_en = args.dyrelu_phasing_en if is_main_model else False

    model = ClassificationAndEncoderNetwork(
        model_name=args.model_name,
        num_classes=args.num_classes,
        num_out_features=args.num_out_features,
        device=args.device,
        adapt=adapt,
        pretrained=True,
        freeze=False,
        dyrelu_en=dyrelu_en,
        dyrelu_phasing_en=dyrelu_phasing_en,
    )
    return model


def _apply_proj_mode(args, models):
    """Configure how the contrastive projection head behaves across models.

      'tied_frozen' (default) -- every model shares ONE encoder_head module, and it
          is frozen. The head is retained (dimensionality reduction, separate
          contrastive space) but the student can no longer lower PrC/FiC/SnC by
          moving it, so the only remaining route is the backbone. Sharing also
          fixes a second problem by construction: the teachers' heads were never
          trained or loaded (encoder_head is created in
          ClassificationAndEncoderNetwork.__init__, which runs *after* the
          load_weights call in BaseModelWrapper.__init__, and the partial-load path
          explicitly filters encoder_head keys), so the student had been aligning
          to a random projection of the teacher's features. With one shared W, the
          same map applies on both sides and its distortion cancels.

      'none' -- no head on the contrastive path; contrast pooled features directly,
          as CAP does with the [CLS] state.

      'current' -- the original behaviour: each model keeps its own head, the
          student's trainable. Retained so existing results stay reproducible.
    """
    # Non-contrastive runs have no projection head and no use for one: the head
    # exists solely to map features into the space the contrastive loss operates
    # in. TrainingArguments does not even declare proj_mode, so the getattr below
    # used to hand a baseline or pruning run the 'tied_frozen' default and then
    # raise because no encoder_head exists -- which meant baseline_script.py and
    # pruning_script.py both crashed during __post_init__ for every CLI
    # invocation (num_out_features has no CLI flag, so it is always None there).
    # The unit tests missed it because their fixtures set num_out_features.
    if not getattr(args, 'is_bacp', False):
        for model in [m for m in models if m is not None]:
            model.proj_mode = 'none'
        return

    mode = getattr(args, 'proj_mode', 'tied_frozen')
    models = [m for m in models if m is not None]

    for model in models:
        model.proj_mode = mode

    if mode == 'none':
        print('[PROJ] proj_mode=none: contrasting pooled backbone features directly')
        return

    if mode == 'current':
        print('[PROJ] proj_mode=current: per-model trainable heads (legacy behaviour)')
        return

    if mode != 'tied_frozen':
        raise ValueError(f"Unknown proj_mode {mode!r}; expected one of "
                         "'tied_frozen', 'none', 'current'")

    student = models[0]
    if not hasattr(student, 'encoder_head'):
        raise RuntimeError("proj_mode='tied_frozen' needs num_out_features set so "
                           "that encoder_head exists")

    for param in student.encoder_head.parameters():
        param.requires_grad = False
    for other in models[1:]:
        other.encoder_head = student.encoder_head

    print('[PROJ] proj_mode=tied_frozen: one frozen projection head shared across '
          f'{len(models)} model(s)')


def _initialize_models(args):
    """Initializes Model(s), applies Weight Sharing, and loads weights."""
    if getattr(args, 'is_bacp', False):
        # BaCP Mode: Create PT, FT, and Current models
        model_pt = _create_base_model(args, is_main_model=False)
        curr_model = _create_base_model(args, is_main_model=True)

        model_ft = deepcopy(model_pt).to(args.device)
        load_weights(model_ft, args.trained_weights)

        # Weight sharing applies to the model being pruned only -- never to the
        # frozen teachers, whose whole purpose is to supply an unmodified reference
        # representation. It was previously never applied on the BaCP path at all,
        # so --weight_sharing_en was a silent no-op here while still tripping
        # _calculate_adjusted_sparsity into an AttributeError.
        if getattr(args, 'weight_sharing_en', False):
            apply_weight_sharing_resnet(curr_model)

        args.model = curr_model
        args.model_pt = model_pt
        args.model_ft = model_ft
        args.embedded_dim = args.model.embedded_dim
        _apply_proj_mode(args, [curr_model, model_pt, model_ft])
        print('[MODEL] Initialized BaCP models')

    else:
        # Standard Mode
        print('[MODEL] Initializing Model')
        args.model = _create_base_model(args, is_main_model=True)

        # Load BEFORE sharing. state_dict() does not de-duplicate aliased
        # Parameters even though named_parameters() does, so an unshared checkpoint
        # loaded into an already-shared model writes every alias into the same
        # storage and the last write wins -- leaving the shared weight holding
        # whichever block happened to come last rather than the master's value.
        if args.trained_weights:
            load_weights(args.model, args.trained_weights)

        if args.weight_sharing_en:
            apply_weight_sharing_resnet(args.model)

        args.embedded_dim = args.model.embedded_dim
        _apply_proj_mode(args, [args.model])

    print('[MODEL] Model Configured:')
    print(f'  > Model: {args.model_name}')
    print(f'  > Device: {args.device}')


def _initialize_dyrelu_phasing(args):
    """Configures the DyReLU adapter schedule."""
    if args.dyrelu_phasing_en:
        print(f"[DyReLU] Initializing Phasing Schedule...")
        t_start = 10
        
        if not getattr(args, 'is_bacp', False):
            total_epochs = args.epochs + (args.epochs * args.recovery_epochs)
            t_start = int(total_epochs * 0.2)
            t_end = int(total_epochs * 0.8)
        else:
            duration = int(args.epochs + args.recovery_epochs + (0.25 * args.epochs_ft))
            max_dur = int(args.epochs + args.recovery_epochs + args.epochs_ft)
            t_end = min(t_start + duration, max_dur)

        set_t_for_dyrelu_adapter(args.model, t_start, t_end)
        print(f"  > Start: Epoch {t_start} | End: Epoch {t_end}")


def _initialize_all(args):
    """Run the eight initialisers in dependency order.

    The single source of truth for this sequence. It used to be duplicated
    between TrainingArguments.__post_init__ and BaCPTrainingArguments.__post_init__,
    and the two copies drifted -- which is exactly how _initialize_pruner came to
    read args.weight_sharing_en and args.delta_T, fields that only one of the two
    dataclasses declared. One sequence plus one contract test parametrized over
    both dataclasses closes that class of bug for good.

    The order is load-bearing:
        models     -> sets args.model, args.embedded_dim (and the BaCP teachers)
        data       -> sets args.trainloader / valloader / testloader
        optimizer  -> needs args.model
        scheduler  -> needs args.optimizer and len(args.trainloader)
        pruner     -> needs args.model and len(args.trainloader)
        paths      -> sets args.datestamp, args.save_path, args.logger
        log params -> snapshots every scalar attribute set above
        dyrelu     -> needs args.model
    """
    _initialize_models(args)
    _initialize_data_loaders(args)
    _initialize_optimizer(args)
    _initialize_scheduler(args)
    _initialize_pruner(args)
    _initialize_paths_and_logger(args)
    _initialize_log_parameters(args)
    _initialize_dyrelu_phasing(args)


class _LimitedLoader:
    """A DataLoader truncated to its first `n` batches.

    Exists so tier-0 smoke runs can exercise every code path on CPU in seconds.
    It is deliberately a wrapper with a correct `__len__` rather than an early
    `break` in the training loop, because `len(trainloader)` is load-bearing in
    three places -- the scheduler's total_steps, the pruner's total_steps, and
    `train_batches` in the run record. An early break would leave all three
    describing a run that did not happen.

    Truncation is always recorded (`limit_train_batches` / `limit_eval_batches` on
    the record), so a smoke run can never be mistaken for a real one.
    """

    def __init__(self, loader, n):
        self.loader = loader
        self.n = int(n)

    def __iter__(self):
        import itertools
        return iter(itertools.islice(self.loader, self.n))

    def __len__(self):
        return min(self.n, len(self.loader))

    def __getattr__(self, name):
        # dataset, batch_size, collate_fn, ... all still resolve.
        return getattr(self.loader, name)


def _maybe_limit(loader, n):
    if loader is None or not n or n <= 0:
        return loader
    return _LimitedLoader(loader, n)


def _initialize_data_loaders(args):
    print("[DATA] Initializing Loaders")
    args.cache_dir = '/dbfs/cache' if args.databricks_env else './cache'
    args.n_views = 1 if not getattr(args, 'is_bacp', False) else args.n_views

    data = get_dataloaders(args)
    args.trainloader = data.get("trainloader")
    args.valloader = data.get("valloader")
    args.testloader = data.get("testloader")

    n_train = getattr(args, 'limit_train_batches', 0)
    n_eval = getattr(args, 'limit_eval_batches', 0)
    args.trainloader = _maybe_limit(args.trainloader, n_train)
    args.valloader = _maybe_limit(args.valloader, n_eval)
    args.testloader = _maybe_limit(args.testloader, n_eval)
    if n_train or n_eval:
        print(f'  > [SMOKE] loaders truncated: train={n_train or "all"} '
              f'eval={n_eval or "all"} -- results are NOT valid measurements')

    print(f'  > Train Loader Initialized' if args.trainloader else '  > Train Loader NOT initialized')
    print(f'  > Validation Loader Initialized' if args.valloader else '  > Validation Loader NOT initialized')
    print(f'  > Test Loader Initialized' if args.testloader else '  > Test Loader NOT initialized')
    print(f"  > Batch Size: {args.batch_size}")
    print(f'  > Cache Directory: {args.cache_dir}')


def _no_decay_param_groups(model, weight_decay):
    """Split parameters so weight decay skips the weight-sharing scale factors.

    apply_weight_sharing_resnet gives every shared block a scalar `scaler`
    parameter. Under the default weight_decay=5e-4 those scalars decay toward
    zero over training, progressively silencing the very blocks that sharing
    introduced -- a slow, silent degradation that looks like the method simply not
    working. Scale and gain parameters are conventionally exempt from decay.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith('scaler') or param.ndim == 0:
            no_decay.append(param)
        else:
            decay.append(param)

    groups = [{'params': decay, 'weight_decay': weight_decay}]
    if no_decay:
        groups.append({'params': no_decay, 'weight_decay': 0.0})
        print(f'  > {len(no_decay)} scale parameter(s) exempted from weight decay')
    return groups


def _initialize_optimizer(args):
    print("[OPTIMIZER] Initializing Optimizer")
    lr = getattr(args, 'learning_rate', 0.0)
    opt_type = getattr(args, 'optimizer_type', 'None')

    # Weight decay was hardcoded at 5e-4 -- GraNet's value. EAST, which this
    # project's anchor cell reproduces, uses 1e-4. A hardcoded constant cannot be
    # recorded either, so two runs under different literature protocols would
    # have looked identical in the record.
    wd = getattr(args, 'weight_decay', 5e-4)
    if opt_type == 'adamw':
        args.optimizer = optim.AdamW(args.model.parameters(), lr=lr,
                                     weight_decay=wd)
    elif opt_type == 'sgd':
        args.optimizer = optim.SGD(_no_decay_param_groups(args.model, wd),
                                   lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")

    print(f'  > Optimizer type: {opt_type}')
    print(f'  > Optimizer learning rate: {lr}')
    print(f'  > Weight decay: {wd}')
    

def _initialize_scheduler(args):
    print("[SCHEDULER] Initializing LR Scheduler")
    if not getattr(args, "trainloader", None): return

    scheduler_type = getattr(args, "scheduler_type", None)
    if scheduler_type == "linear_with_warmup":
        args.total_steps = int(args.epochs * len(args.trainloader))
        args.warmup_steps = int(args.total_steps * 0.1)
        
        args.scheduler = get_linear_schedule_with_warmup(
            optimizer=args.optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.total_steps,
        )
        print(f"  > Linear Scheduler Configured:")
        print(f"  > Warmup steps: {args.warmup_steps}")
        print(f"  > Total steps: {args.total_steps}")

    elif scheduler_type == 'cosine':
        args.total_steps = int(args.epochs * len(args.trainloader))
        args.scheduler = CosineAnnealingLR(
            optimizer=args.optimizer,
            T_max=args.total_steps,
            eta_min=0.00001
        )
        print("  > Cosine Annealing Scheduler Configured.")
        print(f"  > Total steps: {args.total_steps}")
    else:
        args.scheduler = None
        print("  > No scheduler initialized")


def _calculate_adjusted_sparsity(args):
    """Re-express the target sparsity against the *unique* parameter count.

    Under weight sharing the model has fewer unique parameters than nominal, so
    pruning the unique set to the nominal target would leave far fewer live
    weights than the unshared baseline. Rescaling keeps the absolute number of
    surviving weights the same, which is what makes a shared model comparable to
    an unshared one at the same reported sparsity.
    """
    model = args.model.model if hasattr(args.model, 'model') else args.model

    if not hasattr(model, 'total_params') or not hasattr(model, 'unique_params'):
        raise RuntimeError(
            "_calculate_adjusted_sparsity needs model.total_params and "
            "model.unique_params, which are set by apply_weight_sharing_resnet(). "
            "Either weight_sharing_en was set without weight sharing having been "
            "applied, or sharing ran on a different model object."
        )

    total_w = model.total_params
    unique_w = model.unique_params

    keep_w = total_w * (1.0 - args.target_sparsity)
    unique_w_ratio = keep_w / unique_w
    adjusted_target = 1.0 - unique_w_ratio

    # The rescaling can leave the interval entirely: with total=100, unique=50 and
    # a target of 0.3, keep_w=70 exceeds the 50 unique weights available, giving
    # an adjusted target of -0.4. Silently handing a negative sparsity to the
    # pruner produces nonsense, so say what went wrong.
    if adjusted_target < 0.0:
        raise ValueError(
            f"Target sparsity {args.target_sparsity} is unreachable under weight "
            f"sharing: keeping {keep_w:,.0f} of {total_w:,} nominal weights needs "
            f"more than the {unique_w:,} unique weights that exist. Raise "
            f"target_sparsity above {1.0 - unique_w / total_w:.4f}."
        )

    return round(min(adjusted_target, 1.0), 4)


def _initialize_pruner(args):
    print('[SPARSITY SCHEDULER] Initializing Pruner')

    # Declare the prunable scope BEFORE anything measures sparsity. The pruner and
    # every reporting function must agree on which parameters are in scope, or the
    # reported figure describes a different set of weights than the one being pruned.
    set_prunable_scope(
        prune_task_head=getattr(args, 'prune_task_head', False),
        prune_embeddings=getattr(args, 'prune_embeddings', False),
        model=args.model,
    )

    # No pruner exists yet, so the value-based measure is the only one available.
    args.initial_sparsity = check_model_sparsity(args.model)
    args.prune = (args.pruning_type and args.target_sparsity)

    if not args.prune:
        args.pruner = None
        return
    
    if args.pruning_module:
        args.pruner = args.pruning_module
        return
    
    # Keep both numbers. _calculate_adjusted_sparsity rewrites target_sparsity in
    # place, so without this the requested value survives only in stdout and every
    # record would report the adjusted figure as though it had been asked for.
    # This runs before _initialize_log_parameters (step 7 of _initialize_all), so
    # both land in logger_params for free.
    args.requested_sparsity = args.target_sparsity

    if args.weight_sharing_en:
        old_target = args.target_sparsity
        args.target_sparsity = _calculate_adjusted_sparsity(args)
        print(f"  > Weight Sharing Enabled: Adjusted Target {old_target} -> {args.target_sparsity:.4f}")

    args.adjusted_sparsity = args.target_sparsity

    # total_epochs = args.epochs + args.recovery_epochs
    total_epochs = args.epochs
    is_east = (args.pruning_type == 'east')
    args.pruner = get_pruner(
        method_name=args.pruning_type,
        model=args.model,
        epochs=total_epochs,
        sparsity=args.target_sparsity,

        # kwargs
        scheduler_type=args.sparsity_scheduler,
        delta_T=args.delta_T,
        total_steps=len(args.trainloader) * args.epochs,
        alpha=0.3 if args.pruning_type == 'rigl' else None,

        # How the global budget is split across layers. Only LocalMagnitudePrune
        # reads it; every other pruner either fixes the split (RigL/EAST use ERK)
        # or derives it implicitly from a global threshold. Kept a config field
        # rather than a pruner default so the choice lands in logger_params and
        # therefore in the record -- 'erk' vs 'uniform' changes what the number
        # means, so it cannot be an undeclared default.
        layerwise_alloc=getattr(args, 'layerwise_alloc', 'erk'),

        # EAST-only. Tc_ratio was previously never passed at all, so
        # EASTPruner.__init__ evaluated int(self.end_idx * None) and raised a
        # TypeError -- meaning the EAST pruner had never once constructed.
        # WANDA-only. Without a calibration loader the activation norms are all
        # 1.0 and WANDA degenerates to magnitude pruning -- so the loader is
        # wired here rather than left to the caller, and WandaPruner prints a
        # warning if it ever ends up without one.
        calibration_loader=args.trainloader if args.pruning_type == 'wanda' else None,
        calibration_batches=getattr(args, 'calibration_batches', 8),
        # Wanda's comparison group. Recorded because Wanda's own Appendix A finds
        # per-output does NOT beat layer-wise on image classifiers, so the choice
        # is ours to defend and must appear in the record rather than in a default.
        wanda_group=getattr(args, 'wanda_group', 'output'),

        s_min=getattr(args, 's_min', 0.05) if is_east else None,
        prune_rate=getattr(args, 'prune_rate', 0.1) if is_east else None,
        Tc_ratio=getattr(args, 'Tc_ratio', 0.5) if is_east else None,
    )
    print(f"  > Pruner '{args.pruning_type}' initialized.")
    print(f'  > Sparsity Scheduler: {args.sparsity_scheduler}')
    print(f'  > Initial Sparsity: {args.initial_sparsity}')
    print(f'  > Target Sparsity: {args.target_sparsity}')


def _handle_wanda_calibration(args):
    if args.prune and getattr(args.pruner, 'is_wanda', False):
        if not args.recover:
            print("[Wanda] Starting calibration")
            args.pruner.calibrate(args.trainloader, num_batches=100)
        else:
            print("[Wanda] Skipping calibration (Recovery Mode)")


# ----------------------------------------------------------------------
# Training Loop Steps
# ----------------------------------------------------------------------
def _optimizer_step(args, loss, global_step):
    """
    Performs standard ZeroGrad -> Backward -> Prune -> Step -> Mask sequence.
    """
    scaler = getattr(args, "scaler", None)
    
    args.optimizer.zero_grad()
    
    if scaler:
        scaler.scale(loss).backward()
        scaler.unscale_(args.optimizer)
    else:
        loss.backward()

    # === Pruning Step ===
    if args.pruner:
        _step_pruning_step(args, global_step)

    if scaler:
        scaler.step(args.optimizer)
        scaler.update()
    else:
        args.optimizer.step()

    if args.pruner:
        args.pruner.apply_mask()

    # Step per-iteration scheduler (e.g. Linear w/ Warmup)
    if args.scheduler:
        args.scheduler.step()


# _epoch_pruning_step was removed. It called args.pruner.step(epoch) -- an
# *epoch* index -- while the pruner's end_idx and Tc are derived from total_steps,
# i.e. a global *step* index. With both call sites live the same pruner was driven
# on two incompatible clocks, and the epoch-indexed one also passed no optimizer,
# so reset_optimizer_states was skipped on those updates. Pruning is now
# step-driven only, via _optimizer_step -> _step_pruning_step. Its sole reference
# in trainer.py was already commented out.


def _step_pruning_step(args, step):
    """
    Called at the START of every step.
    Handles mask updates.
    """
    if args.pruner and not args.recover:
        if step % args.delta_T == 0 and step <= args.pruner.end_idx:
            args.pruner.step(step, args.optimizer)

            # Refresh the cached sparsity after a mask update. This used to be done
            # by _epoch_pruning_step, which was removed for driving the pruner on an
            # epoch clock while its end_idx is step-indexed. Without it,
            # args.current_sparsity kept its initial value for the whole run, so
            # _get_sparsity_key -- which keys the accuracy history and labels the
            # final metrics -- reported the sparsity the model started at rather
            # than the one it reached. Only recomputed on an actual mask update, so
            # the cost is once per delta_T steps.
            args.current_sparsity = report_sparsity(args.model, args.pruner)


# def _apply_pruning(args, epoch, step):
#     """
#     Applies the pruning mask or scheduling update based on the pruner type 
#     and current training state (epoch/step).
#     """
#     if not args.prune or args.recover:
#         # Skip pruning if it's disabled or if we are in a recovery phase
#         return
    
#     if isinstance(args.pruner, RigLPruner):
#         args.global_step += 1
#         # Pruning stops after args.Tc epochs
#         if epoch < args.Tc:
#             # Prune every delta_t steps
#             if args.global_step % args.delta_t == 0:
#                 s = args.initial_sparsity
#                 args.pruner.ratio_step(epoch, args.Tc, s, s)
#                 args.pruner.prune(args.model)
#     else:
#         # Prune once per epoch (at step 1)
#         if step == 1:
#             args.pruner.ratio_step(epoch, args.epochs, args.initial_sparsity, args.target_sparsity)
#             args.pruner.prune(args.model)


# def _handle_optimizer_and_pruning(args, loss, epoch, step):
#     """Handle backpropagation, pruning, and weight update in a single step."""
#     scaler = getattr(args, "scaler", None)

#     args.optimizer.zero_grad()

#     # Back-propagation (w/ scaling if enabled)
#     if scaler is None:
#         loss.backward()
#     else:
#         scaler.scale(loss).backward()

#     # Pruning the model
#     if args.pruner is not None:
#         _apply_pruning(args, epoch, step)
    
#     # Gradient Clipping
#     # torch.nn.utils.clip_grad_norm_(args.model.parameters(), max_norm=1.0)

#     # Optimizer step (w/ scaling if enabled)
#     if scaler is None:
#         args.optimizer.step()
#     else:
#         scaler.step(args.optimizer)
#         scaler.update()
    
#     # Scheduler step
#     if args.scheduler:
#         args.scheduler.step()

#     # Applying mask and updating sparsity
#     if args.pruner is not None:
#         args.pruner.apply_mask(args.model)

#         # Update model sparsity once per epoch if pruning is active
#         if args.prune and step == 1:
#             args.current_sparsity = report_sparsity(args.model, args.pruner)


def _initialize_paths_and_logger(args):
    print(f'[PATHS] Initializing Logger')
    now = datetime.now()
    args.datestamp = now.strftime('%Y%m%d_%H%M%S')
    date_dir = now.strftime('%Y%m%d')
    base_dir = '/dbfs' if args.databricks_env else '.'
    
    args.base_path = os.path.join(
        base_dir, 'research', 'bacp', 
        args.model_name, args.dataset_name, date_dir
    )

    prune_str = f"{args.pruning_type}_{args.target_sparsity}" if args.prune else "dense"
    fname = f"{args.model_name}_{args.dataset_name}_{prune_str}_{args.experiment_type}_{args.datestamp}.pt"
    
    args.save_path = os.path.join(args.base_path, fname)
    
    if args.log_epochs:
        log_name = f"{args.dataset_name}/{args.experiment_type}/{prune_str}/{args.datestamp}"
        args.logger = Logger(args.model_name, log_name)
    else:
        args.logger = None

    print(f"  > Save Path: {args.save_path}")


# ----------------------------------------------------------------------
# Data and Logging Helpers
# ----------------------------------------------------------------------

def _initialize_log_parameters(args):
    """Initialize parameters for logging purposes."""
    allowed_types = (int, float, str, bool, torch.Tensor, np.ndarray, list, dict, type(None))
    args.logger_params = {
        k: v for k, v in vars(args).items()
        if isinstance(v, allowed_types) and v is not None
    }


def _handle_data_to_device(args, batch_data):
    """Move one collated batch onto the device and split it into (data, labels).

    Handles the two shapes default_collate produces here:
      n_views=1 -> [Tensor[B,C,H,W], Tensor[B]]
      n_views=2 -> [[Tensor[B,C,H,W], Tensor[B,C,H,W]], Tensor[B]]
    plus HuggingFace-style dict batches.
    """
    if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
        data, labels = batch_data
        if isinstance(data, (list, tuple)):
            data = [d.to(args.device) for d in data]
        else:
            data = data.to(args.device)
        labels = labels.to(args.device)
        return data, labels

    if isinstance(batch_data, dict):
        data = {k: v.to(args.device) for k, v in batch_data.items()}
        return data, data.get('labels', None)

    # Neither branch matched: `data` and `labels` used to be left unbound, so the
    # caller saw an UnboundLocalError from inside this function rather than a
    # description of the batch that caused it.
    length = len(batch_data) if hasattr(batch_data, '__len__') else 'n/a'
    raise ValueError(
        f"Unsupported batch structure: {type(batch_data).__name__} of length {length}. "
        "Expected a (data, labels) pair -- where data is a tensor or a list of view "
        "tensors -- or a dict batch."
    )


def _handle_tqdm_logs(args, batchloader, metrics):
    if args.enable_tqdm:
        disp = {k: f"{v:.4f}" for k,v in metrics.items() if isinstance(v, (int, float))}
        batchloader.set_postfix(**disp)


def _log_metrics(args, info, metrics, run=None):
    # Epoch timing is collected here because this is the single point every training
    # loop passes through once per epoch -- Trainer.train, Trainer._retrain,
    # BaCPTrainer.train/_retrain/finetune. Nothing measured wall-clock before, and
    # with no run-end timestamp the duration could not even be derived after the fact.
    marks = args.__dict__.setdefault('_epoch_marks', [])
    marks.append((info.split(' - ')[0], time.perf_counter()))

    # Mask-based when a pruner owns the masks: counting exact zeros overstates
    # sparsity under dynamic sparse training, where regrown weights are active but
    # numerically zero until their first update.
    metrics['sparsity'] = round(report_sparsity(args.model, getattr(args, 'pruner', None)), 4)
    # Print to console
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            info += f" - {k}: {v:.4f}"
    print(info)

    # Log to WandB / File
    if run: run.log(metrics)
    if args.logger: args.logger.log_epochs(info)


def _get_sparsity_key(args):
    """Get current model sparsity as a rounded key."""
    return round(args.current_sparsity, 4)


def epoch_times(args, include_final=False):
    """Per-epoch wall-clock seconds, derived from the marks _log_metrics leaves.

    The final evaluation also passes through _log_metrics (labelled 'Final'), so it
    is excluded by default -- otherwise a 2-epoch run reports 3 epochs completed.
    """
    marks = getattr(args, '_epoch_marks', None) or []
    t0 = getattr(args, '_t0', None)
    if not marks or t0 is None:
        return []
    if not include_final:
        marks = [(label, t) for label, t in marks if not label.startswith('Final')]
    if not marks:
        return []
    stamps = [t0] + [t for _, t in marks]
    return [round(b - a, 3) for a, b in zip(stamps, stamps[1:])]


def main_loop_epochs(args):
    """Count of main-loop epochs completed, excluding recovery and fine-tuning.

    Compared against `epochs` to decide whether early stopping fired, so it must not
    count 'Recovery Epoch' or 'Fine-tuning Epoch' marks.
    """
    marks = getattr(args, '_epoch_marks', None) or []
    return sum(1 for label, _ in marks if label.startswith('Epoch ['))


def _initialize_logs(args):
    # Run-start markers for the results record. Set here rather than in
    # __post_init__ so construction time (model download, dataset build) is not
    # charged to training.
    args.run_started_at = datetime.now(timezone.utc).isoformat()
    args._t0 = time.perf_counter()
    args._epoch_marks = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    if args.logger is not None:
        args.logger.create_log()
        args.logger.log_hyperparameters(args.logger_params)
    else:
        pass


def _finalize_run(args, metrics, phase=None, experiment_group=None):
    """Write one standardized result record. Never allowed to break a finished run.

    Kept in training_utils rather than in the trainers so `results` stays out of
    their import graph, and so both trainers share one call site's worth of logic.
    """
    try:
        from results import record_run
    except Exception as e:                                    # noqa: BLE001
        print(f"[RESULTS] not recorded ({type(e).__name__}: {e})")
        return None

    if phase is None:
        # finetune() sets is_bacp=False before its own evaluate(), so the two BaCP
        # evaluations self-label without needing an explicit argument.
        if not hasattr(args, 'lambda_history'):
            phase = 'prune' if getattr(args, 'pruning_type', None) else 'dense'
        else:
            phase = 'contrastive' if getattr(args, 'is_bacp', False) else 'finetune'

    # One record per phase per run. evaluate() finalises, and on the BaCP path it
    # is called twice -- once at the end of finetune() (bacp.py) and again by the
    # script afterwards -- so every BaCP cell wrote two identical 'finetune'
    # records and paid for a second full evaluation. At tier 1 that is 65
    # redundant test-set passes. The duplicate was invisible in the table because
    # load_ladder keeps the newest record per cell.
    seen = getattr(args, '_finalized_phases', None)
    if seen is None:
        seen = set()
        setattr(args, '_finalized_phases', seen)
    if phase in seen:
        print(f'[RESULTS] {phase} already recorded for this run; skipping duplicate')
        return None
    seen.add(phase)

    return record_run(
        args, metrics, phase=phase,
        experiment_group=experiment_group or os.environ.get('BACP_EXPERIMENT_GROUP'),
        strict=False,
    )





















