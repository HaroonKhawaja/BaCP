import os
import logging
from typing import Optional, Dict, Any
from copy import deepcopy
from datetime import datetime

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
    check_model_sparsity
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


def _initialize_models(args):
    """Initializes Model(s), applies Weight Sharing, and loads weights."""
    if getattr(args, 'is_bacp', False):
        # BaCP Mode: Create PT, FT, and Current models
        model_pt = _create_base_model(args, is_main_model=False)
        curr_model = _create_base_model(args, is_main_model=True)

        model_ft = deepcopy(model_pt).to(args.device)
        load_weights(model_ft, args.trained_weights)

        args.model = curr_model
        args.model_pt = model_pt
        args.model_ft = model_ft
        args.embedded_dim = args.model.embedded_dim
        print('[MODEL] Initialized BaCP models')

    else:
        # Standard Mode
        print('[MODEL] Initializing Model')
        args.model = _create_base_model(args, is_main_model=True)

        # Apply Weight Sharing BEFORE loading weights
        if args.weight_sharing_en:
            apply_weight_sharing_resnet(args.model)

        if args.trained_weights:
            load_weights(args.model, args.trained_weights)
        
        args.embedded_dim = args.model.embedded_dim

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


def _initialize_data_loaders(args):
    print("[DATA] Initializing Loaders")
    args.cache_dir = '/dbfs/cache' if args.databricks_env else './cache'
    args.n_views = 1 if not getattr(args, 'is_bacp', False) else args.n_views

    data = get_dataloaders(args)
    args.trainloader = data.get("trainloader")
    args.valloader = data.get("valloader")
    args.testloader = data.get("testloader")

    print(f'  > Train Loader Initialized' if args.trainloader else '  > Train Loader NOT initialized')
    print(f'  > Validation Loader Initialized' if args.valloader else '  > Validation Loader NOT initialized')
    print(f'  > Test Loader Initialized' if args.testloader else '  > Test Loader NOT initialized')
    print(f"  > Batch Size: {args.batch_size}")
    print(f'  > Cache Directory: {args.cache_dir}')


def _initialize_optimizer(args):
    print("[OPTIMIZER] Initializing Optimizer")
    lr = getattr(args, 'learning_rate', 0.0)
    opt_type = getattr(args, 'optimizer_type', 'None')

    if opt_type == 'adamw':
        args.optimizer = optim.AdamW(args.model.parameters(), lr=lr)
    elif opt_type == 'sgd':
        args.optimizer = optim.SGD(args.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")

    print(f'  > Optimizer type: {opt_type}')
    print(f'  > Optimizer learning rate: {lr}')
    

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
    model = args.model.model if hasattr(args.model, 'model') else args.model
    total_w = model.total_params
    unique_w = model.unique_params

    keep_w = total_w * (1.0 - args.target_sparsity)
    unique_w_ratio = keep_w / unique_w
    adjusted_target = 1.0 - unique_w_ratio

    return round(adjusted_target, 4)


def _initialize_pruner(args):
    print('[SPARSITY SCHEDULER] Initializing Pruner')
    args.initial_sparsity = check_model_sparsity(args.model)
    args.prune = (args.pruning_type and args.target_sparsity)

    if not args.prune:
        args.pruner = None
        return
    
    if args.pruning_module:
        args.pruner = args.pruning_module
        return
    
    if args.weight_sharing_en:
        old_target = args.target_sparsity
        args.target_sparsity = _calculate_adjusted_sparsity(args)
        print(f"  > Weight Sharing Enabled: Adjusted Target {old_target} -> {args.target_sparsity:.4f}")

    # total_epochs = args.epochs + args.recovery_epochs
    total_epochs = args.epochs
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

        s_min=getattr(args, 's_min', 0.05) if args.pruning_type == 'east' else None,
        prune_rate=getattr(args, 'prune_rate', 0.1) if args.pruning_type == 'east' else None,

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


def _epoch_pruning_step(args, epoch):
    """
    Called at the END of every epoch. 
    Handles mask updates.
    """
    if args.pruner and not args.recover:
        args.pruner.step(epoch)
        args.current_sparsity = check_model_sparsity(args.model)


def _step_pruning_step(args, step):
    """
    Called at the START of every step. 
    Handles mask updates.
    """
    if args.pruner and not args.recover:
        if step % args.delta_T == 0 and step <= args.pruner.end_idx:
            args.pruner.step(step, args.optimizer)


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
#             args.current_sparsity = check_model_sparsity(args.model)


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
    if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
        data, labels = batch_data
        if isinstance(data, (list, tuple)) and len(data) == 2:
            data = [d.to(args.device) for d in data]
        else:
            data = data.to(args.device)
        labels = labels.to(args.device)
    elif isinstance(batch_data, dict):
        data = {k: v.to(args.device) for k, v in batch_data.items()}
        labels = data.get('labels', None)
    
    return data, labels


def _handle_tqdm_logs(args, batchloader, metrics):
    if args.enable_tqdm:
        disp = {k: f"{v:.4f}" for k,v in metrics.items() if isinstance(v, (int, float))}
        batchloader.set_postfix(**disp)


def _log_metrics(args, info, metrics, run=None):
    metrics['sparsity'] = round(check_model_sparsity(args.model), 4)
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


def _initialize_logs(args):
    if args.logger is not None:
        args.logger.create_log()
        args.logger.log_hyperparameters(args.logger_params)
    else:
        pass





















