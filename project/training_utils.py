import os
from copy import deepcopy
import logging
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from transformers import get_linear_schedule_with_warmup

from datasets import get_dataset_config_names
from logger import Logger
from utils import get_device, load_weights, set_seed
from dataset_utils_old import get_data
from datetime import datetime

from model_factory import ClassificationAndEncoderNetwork
from dataset_factory import get_dataloaders
from pruning_factory import (
    check_model_sparsity, PRUNING_REGISTRY, 
    MagnitudePrune, SNIPIterativePrune, WandaPrune, 
    RigLPruner
)
from dyrelu_adapter import set_t_for_dyrelu_adapter


# ----------------------------------------------------------------------
# Model Initialization Helpers
# ----------------------------------------------------------------------

def _create_base_model(args, is_main_model=False):
    """Factory function for creating a single model instance."""
    adapt = args.image_size <= 64

    # DyReLU arguments only apply to the main (current) model
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
    if getattr(args, 'is_bacp', False):
        model_pt = _create_base_model(args, is_main_model=False)
        curr_model = _create_base_model(args, is_main_model=True)

        # Fine-tuned model is a deepcopy of the PT model
        model_ft = deepcopy(model_pt).to(args.device)
        load_weights(model_ft, args.trained_weights)

        args.model = curr_model
        args.model_pt = model_pt
        args.model_ft = model_ft
        args.embedded_dim = args.model.embedded_dim
        print('[TRAINER] Initialized BaCP models')

    else:
        args.model = _create_base_model(args, is_main_model=True)
        if args.trained_weights:
            load_weights(args.model, args.trained_weights)
        else:
            print('[TRAINER] No custom weights provided')

        args.embedded_dim = args.model.embedded_dim
        print('[TRAINER] Initialized standard model')


def _initialize_dyrelu_phasing(args):
    """
    Configures the DyReLU adapter to phase out over a specific number of EPOCHS.
    """
    if args.dyrelu_phasing_en:
        # Starting phasing after a warmup
        t_start_epoch = 10

        if args.is_bacp == False:
            total_epochs = args.epochs +  (args.epochs * args.recovery_epochs)
            t_start_epoch = int(total_epochs * 0.1)
            t_end_epoch = int(total_epochs * 0.85)
            duration = t_end_epoch - t_start_epoch

        else:
            # Calculating total duration
            duration = int(args.epochs + args.recovery_epochs + (0.25 * args.epochs_ft))
            max_duration = int(args.epochs + args.recovery_epochs + args.epochs_ft)
            t_end_epoch = min(t_start_epoch + duration, max_duration)

        set_t_for_dyrelu_adapter(args.model, t_start_epoch, t_end_epoch)
        print(f"[DyReLU Phasing] Schedule Configured (Epoch-based):")
        print(f"  > Start Epoch: {t_start_epoch}")
        print(f"  > End Epoch:   {t_end_epoch}")
        print(f"  > Duration:    {int(duration)} epochs")


# ----------------------------------------------------------------------
# Data and Optimizer Initialization
# ----------------------------------------------------------------------

def _initialize_data_loaders(args):
    # Initializing cache directory for datasets
    args.cache_dir = '/dbfs/cache' if args.databricks_env else './cache'
    args.n_views = 1 if not args.is_bacp else args.n_views

    data = get_dataloaders(args)
    # data = get_data(args)
    args.trainloader = data.get("trainloader")
    args.valloader = data.get("valloader")
    args.testloader = data.get("testloader")

    print('[TRAINER] Train Loader Initialized' if args.trainloader else '[TRAINER] Train Loader not initialized')
    print('[TRAINER] Validation Loader Initialized' if args.valloader else '[TRAINER] Validation Loader not initialized')
    print('[TRAINER] Test Loader Initialized' if args.testloader else '[TRAINER] Test Loader not initialized')


def _initialize_optimizer(args):
    opt_type = getattr(args, 'optimizer_type', None)
    lr = getattr(args, 'learning_rate', None)

    if opt_type == 'adamw':
        args.optimizer = optim.AdamW(args.model.parameters(), lr=lr)
    elif opt_type == 'sgd':
        args.optimizer = optim.SGD(args.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    else:
        raise ValueError(f"Invalid optimizer type: {opt_type}")

    print(f'[TRAINER] Optimizer type w/ learning rate: ({opt_type}, {lr})')
    

def _initialize_scheduler(args):
    if not getattr(args, "trainloader", None):
        raise RuntimeError("trainloader required for scheduler initialization.")

    scheduler_type = getattr(args, "scheduler_type", None)
    if scheduler_type:
        args.total_steps = int(getattr(args, "epochs", 0) * len(args.trainloader))
        args.warmup_steps = int(args.total_steps * 0.1)

        if scheduler_type == "linear_with_warmup":
            args.scheduler = get_linear_schedule_with_warmup(
                optimizer=args.optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=args.total_steps,
            )
            print(f"[TRAINER] Linear scheduler initialized with warmup steps: {args.warmup_steps} and total steps: {args.total_steps}")
        else:
            raise ValueError(f"Invalid scheduler type: {scheduler_type}")
    else:
        args.scheduler = None
        print("[TRAINER] No scheduler initialized")


# ----------------------------------------------------------------------
# Path, Logger, and Pruning Initialization
# ----------------------------------------------------------------------

def _initialize_pruner(args):
    args.initial_sparsity = check_model_sparsity(args.model)
    args.prune = (args.pruning_type and args.target_sparsity and args.pruning_module is None)
    
    if args.prune:
        total_scheduled_epochs = args.epochs + args.recovery_epochs

        args.pruner = PRUNING_REGISTRY[args.pruning_type](
            model=args.model,
            epochs=total_scheduled_epochs,
            s_end=args.target_sparsity, 
            sparsity_scheduler=args.sparsity_scheduler
        )
        print('[TRAINER] Pruning module initialized')
        print(f'[TRAINER] {args.pruning_type=}, {args.target_sparsity=}, {args.sparsity_scheduler=}, {args.initial_sparsity=}')
    else:
        args.pruner = args.pruning_module

    if isinstance(args.pruner, RigLPruner):
        # RigL-specific parameters (Tc is the epoch where pruning stops)
        args.Tc = int(args.epochs) 
        args.delta_t = 100
        args.global_step = 0


def _initialize_paths_and_logger(args):
    now = datetime.now()
    args.datestamp = now.strftime('%Y%m%d_%H%M%S') # e.g., '20251205_142200'
    date_dir = now.strftime('%Y%m%d')

    base_dir = '/dbfs' if args.databricks_env else '.'
    
    # Base path now includes the unique datestamp directory
    # Structure: .../bacp/model_name/dataset_name/DATETIME/
    args.base_path = os.path.join(
        base_dir, 'research', 'bacp', 
        args.model_name, args.dataset_name, 
        date_dir
    )

    is_pruning = args.prune or args.pruning_module is not None
    prune_type = args.pruning_type or ""
    sparsity = str(args.target_sparsity) if is_pruning and args.target_sparsity else ""
    
    # Build weights path (filename)
    weights_parts = [args.model_name, args.dataset_name]
    if is_pruning:
        weights_parts.extend([prune_type, sparsity])
    weights_parts.append(args.experiment_type)
    weights_parts.append(args.datestamp)
    weights_path = '_'.join(filter(None, weights_parts)) + '.pt'

    # Build logger path: /dataset_name/experiment_type/[pruning_type/sparsity]/datestamp
    logger_parts = [args.dataset_name, args.experiment_type]
    if is_pruning:
        logger_parts.extend([prune_type, sparsity])
    
    logger_parts.append(args.datestamp) 
        
    logger_path = os.path.join(*filter(None, logger_parts))

    args.save_path = os.path.join(args.base_path, weights_path)
    args.logger = Logger(args.model_name, logger_path) if args.log_epochs else None
    print(f'[TRAINER] Saving model to: {args.save_path}')


# ----------------------------------------------------------------------
# Optimization and Pruning Step Handlers
# ----------------------------------------------------------------------

def _apply_pruning(args, epoch, step):
    """
    Applies the pruning mask or scheduling update based on the pruner type 
    and current training state (epoch/step).
    """
    if not args.prune or args.recover:
        # Skip pruning if it's disabled or if we are in a recovery phase
        return
    
    if isinstance(args.pruner, (MagnitudePrune, SNIPIterativePrune, WandaPrune)):
        # Prune once per epoch (at step 1)
        if step == 1:
            args.pruner.ratio_step(epoch, args.epochs, args.initial_sparsity, args.target_sparsity)
            args.pruner.prune(args.model)

    elif isinstance(args.pruner, RigLPruner):
        args.global_step += 1
        # Pruning stops after args.Tc epochs
        if epoch < args.Tc:
            # Prune every delta_t steps
            if args.global_step % args.delta_t == 0:
                s = args.initial_sparsity
                args.pruner.ratio_step(epoch, args.Tc, s, s)
                args.pruner.prune(args.model)


def _handle_optimizer_and_pruning(args, loss, epoch, step):
    """Handle backpropagation, pruning, and weight update in a single step."""
    scaler = getattr(args, "scaler", None)

    args.optimizer.zero_grad()

    # Back-propagation (w/ scaling if enabled)
    if scaler is None:
        loss.backward()
    else:
        scaler.scale(loss).backward()

    # Pruning the model
    if args.pruner is not None:
        _apply_pruning(args, epoch, step)
    
    # Optimizer step (w/ scaling if enabled)
    if scaler is None:
        args.optimizer.step()
    else:
        scaler.step(args.optimizer)
        scaler.update()
    
    # Scheduler step
    if args.scheduler:
        args.scheduler.step()

    # Applying mask and updating sparsity
    if args.pruner is not None:
        args.pruner.apply_mask(args.model)

        # Update model sparsity once per epoch if pruning is active
        if args.prune and step == 1:
            args.current_sparsity = check_model_sparsity(args.model)


def _handle_wanda_hooks(args):
    if args.prune and hasattr(args.pruner, 'is_wanda') and args.pruner.is_wanda:
        if not args.recover:    # We don't want to register hooks during recovery
            args.pruner.calibrate(args.model, args.trainloader)
        else:
            pass


def _initialize_log_parameters(args):
    """Initialize parameters for logging purposes."""
    allowed_types = (int, float, str, bool, torch.Tensor, np.ndarray, list, dict, type(None))
    args.logger_params = {
        k: v for k, v in vars(args).items()
        if isinstance(v, allowed_types) and v is not None
    }


# ----------------------------------------------------------------------
# Data and Logging Helpers
# ----------------------------------------------------------------------

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
        display_metrics = {}
        for k, v in metrics.items():
            if v is None or not isinstance(v, (int, float)):
                continue
            display_metrics[k] = f"{v:.4f}"
        batchloader.set_postfix(**display_metrics)
    else:
        return


def _log_metrics(args, info, metrics, run=None):
    metrics['sparsity'] = _get_sparsity_key(args)
    # Creating information string
    for k, v in metrics.items():
        if v is None or not isinstance(v, (int, float)):
            continue
        info += f" - {k}: {v:.4f}"
    print(info)

    # Logging to wandb or logger
    if run: 
        run.log(metrics)

    if args.logger is not None:
        args.logger.log_epochs(info)


def _get_sparsity_key(args):
    """Get current model sparsity as a rounded key."""
    return round(args.current_sparsity, 4)


def _initialize_logs(args):
    if args.logger is not None:
        args.logger.create_log()
        args.logger.log_hyperparameters(args.logger_params)
    else:
        pass





















