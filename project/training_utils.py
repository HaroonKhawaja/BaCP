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

from model_factory import ClassificationAndEncoderNetwork
from dataset_factory import get_dataloaders
from pruning_factory import check_model_sparsity, PRUNER_DICT
from dyrelu_adapter import set_t_for_dyrelu_adapter
from pruning_factory import apply_erk_initialization


def _initialize_models(args):
    if getattr(args, 'is_bacp', False):
        models = create_models_for_bacp(args)

        args.model = models["curr_model"]
        args.model_pt = models["model_pt"]
        args.model_ft = models["model_ft"]
        args.embedded_dim = args.model.embedded_dim
        print('[TRAINER] Initialized BaCP models')

    else:
        adapt = True if args.image_size and args.image_size <= 64 else False
        args.model = ClassificationAndEncoderNetwork(
            model_name=args.model_name,
            num_classes=args.num_classes,
            num_out_features=args.num_out_features,
            device=args.device,
            adapt=adapt,
            pretrained=True,
            freeze=False,
            dyrelu_en=args.dyrelu_en,
            dyrelu_phasing_en=args.dyrelu_phasing_en,
        )
        args.embedded_dim = args.model.embedded_dim

        if getattr(args, 'use_erk_init', False):
            if getattr(args, 'erk_init_sparsity', None) is None:
                raise ValueError("`erk_init_sparsity` must be set when `use_erk_init` is True.")
            
            erk_masks = apply_erk_initialization(
                model=args.model,
                target_sparsity=args.erk_init_sparsity,
                erk_power_scale=getattr(args, 'erk_power_scale', 1.0)
            )
            print("[TRAINER] ERK initialization complete.")

        if args.trained_weights:
            loaded = load_weights(args.model, args.trained_weights)
            if loaded:
                print("[TRAINER] Weights loaded")
            else: 
                raise ValueError("[TRAINER] Failed to load weights")

    if args.dyrelu_phasing_en:
        if args.recovery_epochs and args.ft_epochs:
            # t_end = args.epochs + (args.recovery_epochs * args.epochs) + (0.25 * args.ft_epochs)
            t_end = int(0.75 * args.ft_epochs)
        else:
            t_end = args.epochs

        set_t_for_dyrelu_adapter(args.model, 0, t_end)


def create_models_for_bacp(args):
    adapt = True if args.image_size and args.image_size <= 64 else False
    model_pt = ClassificationAndEncoderNetwork(
        model_name=args.model_name,
        num_classes=args.num_classes,
        num_out_features=args.num_out_features,
        device=args.device,
        adapt=adapt,
        pretrained=True,
        freeze=False,
    )
    
    # Current projection model
    model = ClassificationAndEncoderNetwork(
        model_name=args.model_name,
        num_classes=args.num_classes,
        num_out_features=args.num_out_features,
        device=args.device,
        adapt=adapt,
        pretrained=True,
        freeze=False,
        dyrelu_en=args.dyrelu_en,
        dyrelu_phasing_en=args.dyrelu_phasing_en,
    )
    if args.encoder_trained_weights:
        load = load_weights(model, args.encoder_trained_weights)
        if load:
            print("[TRAINER] Trained encoder weights loaded successfully")
        else:
            raise ValueError("[TRAINER] Failed to load enocoder weights")

    # Fine-tuned projection model
    model_ft = deepcopy(model_pt).to(args.device)
    if args.trained_weights:
        load = load_weights(model_ft, args.trained_weights)
        if load:
            print("[TRAINER] Weights loaded successfully")
        else:
            raise ValueError("[TRAINER] Failed to load weights")

    return {
        "model_pt": model_pt, 
        "curr_model": model, 
        "model_ft": model_ft
        }


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


def _initialize_pruner(args):
    args.initial_sparsity = check_model_sparsity(args.model)
    args.prune = True if (args.pruning_type and args.target_sparsity and args.pruning_module is None) else False

    if args.prune:
        args.pruner = PRUNER_DICT[args.pruning_type](
            model=args.model,
            epochs=args.epochs,
            target_sparsity=args.target_sparsity, 
            sparsity_scheduler=args.sparsity_scheduler
        )
        print('[TRAINER] Pruning initialized')
        print('[TRAINER] Pruning type:', args.pruning_type)
        print('[TRAINER] Target sparsity:', args.target_sparsity)
        print('[TRAINER] Sparsity scheduler:', args.sparsity_scheduler)
        print(f'[TRAINER] Current sparsity: {args.initial_sparsity:.4f}')
    else:
        args.pruner = args.pruning_module

        # args.pruner = None
        # if args.apply_mask_only:
        #     print('[TRAINER] Finetuning initialized')
        #     print('[TRAINER] Pruning type:', args.pruning_type)
        #     print(f'[TRAINER] Current sparsity: {args.initial_sparsity:.4f}')
        # else:
        #     print('[TRAINER] Pruning not initialized')


def _initialize_paths_and_logger(args):
    base_dir = '/dbfs' if args.databricks_env else '.'
    args.base_path = os.path.join(base_dir, 'research/bacp', args.model_name, args.dataset_name)

    if args.prune or args.pruning_module is not None:
        weights_path = f'{args.model_name}_{args.dataset_name}_{args.pruning_type}_{args.target_sparsity}_{args.experiment_type}.pt'
        logger_path = os.path.join(args.dataset_name, args.experiment_type, args.pruning_type or "", str(args.target_sparsity))
    else:
        weights_path = f"{args.model_name}_{args.dataset_name}_{args.experiment_type}.pt"
        logger_path = os.path.join(args.dataset_name, args.experiment_type)

    args.save_path = os.path.join(args.base_path, weights_path)
    args.logger = Logger(args.model_name, logger_path) if args.log_epochs else None
    print(f'[TRAINER] Saving model to: {args.save_path}')


def _apply_pruning(args, epoch, step):
    if not args.prune or args.recover:
        return
    if step == 1:
        args.pruner.ratio_step(epoch, args.epochs, args.initial_sparsity, args.target_sparsity)
        args.pruner.prune(args.model)


def _handle_optimizer_and_pruning(args, loss, epoch, step):
    """Handle backpropagation, pruning, and weight update in a single step."""
    args.optimizer.zero_grad()

    scaler = getattr(args, "scaler", None)
    if scaler is None:
        loss.backward()
    else:
        scaler.scale(loss).backward()

    _apply_pruning(args, epoch, step)

    if scaler is None:
        args.optimizer.step()
    else:
        scaler.step(args.optimizer)
        scaler.update()
    
    if args.scheduler:
        args.scheduler.step()

    if args.pruner is not None:
        args.pruner.apply_mask(args.model)
        
        if step == 1 and args.prune and not args.recover:
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





















