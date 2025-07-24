import os
from copy import deepcopy
import matplotlib.pyplot as plt
import math 
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss

import evaluate
from datasets import get_dataset_config_names
from datasets.utils.logging import disable_progress_bar
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from constants import *
from dataset_utils import get_glue_data, get_wikitext2_data, get_cv_data, CV_DATASETS
from logger import Logger
from loss_fn import *
from models import ClassificationNetwork, EncoderProjectionNetwork
from unstructured_pruning import MovementPrune, check_model_sparsity, PRUNER_DICT
from utils import get_device, load_weights

def _detect_model_type(args):
    if args.model_task in [get_dataset_config_names("glue"), 'squad', 'wikitext2', 'sst2']:
        args.model_type = 'llm'
        args.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    elif args.model_task in CV_DATASETS:
        args.model_type = 'cv'
    else:
        raise ValueError(f'Cannot auto-detect model type for task: {args.model_task}')

def _detect_num_classes(args):
    task_num_classes = {
        'cifar10': 10,
        'cifar100': 100,
        'svhn': 10,
        'mnist': 10,
        'fmnist': 10,
        'emnist': 47,
        'food101': 101,
        'flowers102': 102,
        'caltech101': 101,
        'sst2': 2,
        'qqp': 2,
        'wikitext2': args.tokenizer.vocab_size if hasattr(args, 'tokenizer') else None,
    }
    if args.model_task in task_num_classes:
        args.num_classes = task_num_classes[args.model_task]
    else:
        raise ValueError(f'Invalid model task: {args.model_task}. Choose from {list(task_num_classes.keys())}')

def _detect_criterion(args):
    if args.criterion_type == 'supervised':
        args.criterion = CrossEntropyLoss()
    elif args.criterion_type == 'contrastive':
        args.criterion = SupConLoss(2, 0.15, 0.15, args.batch_size)
    else:
        args.criterion = None

def _detect_cv_image_size(args):
    if args.model_type == 'cv':
        if args.model_name.startswith('vit'):        
            args.image_size = 224
        else:
            args.image_size = 32
    else:
        args.image_size = None
    print(f'[TRAINER] Image size: {args.image_size}')

def _initialize_models(args):
    if args.is_bacp:
        models = create_models_for_bacp(args.model_name, args.model_task, args.image_size, args.finetuned_weights, args.device, current_finetuned_weights=args.current_finetuned_weights)
        args.model = models["curr_model"]
        args.pre_trained_model = models["pt_model"]
        args.finetuned_model = models["ft_model"]
        args.embedded_dim = args.model.embedding_dim
        print('[TRAINER] Initialized BaCP models')

    else:
        if args.criterion_type == 'supervised':
            args.model = ClassificationNetwork(
                model_name=args.model_name, 
                num_classes=args.num_classes, 
                adapt=(True if args.image_size and args.image_size <= 64 else False), 
                model_task=args.model_task
                )
        elif args.criterion_type == 'contrastive':
            args.model = EncoderProjectionNetwork(
                model_name=args.model_name, 
                adapt=(True if args.image_size and args.image_size <= 64 else False),
                model_task=args.model_task,
                )
        else:
            raise ValueError(f"Invalid criterion type: {args.criterion_type}")

        args.embedded_dim = args.model.embedding_dim
        print('[TRAINER] Initialized models')

        if args.finetuned_weights:
            print(f'[TRAINER] Loading weights: {args.finetuned_weights}')
            loaded = load_weights(args.model, args.finetuned_weights)
            print("[TRAINER] Weights loaded" if loaded else "[TRAINER] Failed to load weights")

def create_models_for_bacp(model_name, model_task, image_size, finetuned_weights, device, output_dimensions=128, current_finetuned_weights=None):
    pre_trained_model = EncoderProjectionNetwork(
        model_name=model_name, 
        output_dims=output_dimensions, 
        adapt=(True if image_size and image_size <= 64 else False), 
        model_task=model_task)
    pre_trained_model.to(device)

    # Current projection model
    model = deepcopy(pre_trained_model).to(device)
    if current_finetuned_weights:
        load = load_weights(model, current_finetuned_weights)
        if load:
            print("[TRAINER] Current weights loaded successfully")
        else:
            raise ValueError("[TRAINER] Failed to load current weights")

    # Fine-tuned projection model
    finetuned_model = deepcopy(pre_trained_model).to(device)
    load = load_weights(finetuned_model, finetuned_weights)
    if load:
        print("[TRAINER] Weights loaded successfully")
    else:
        raise ValueError("[TRAINER] Failed to load weights")

    return {
        "pt_model": pre_trained_model, 
        "curr_model": model, 
        "ft_model": finetuned_model
        }

def _initialize_optimizer(args):
    if args.optimizer_type == 'adamw':
        args.optimizer = optim.AdamW(args.model.parameters(), lr=args.learning_rate)
    elif args.optimizer_type == 'sgd':
        args.optimizer = optim.SGD(args.model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    else:
        raise ValueError(f"Invalid optimizer type: {args.optimizer_type}")

    print(f'[TRAINER] Optimizer type w/ learning rate: ({args.optimizer_type}, {args.learning_rate})')
    
def _initialize_scheduler(args):
    if args.scheduler_type:
        args.total_steps = args.epochs * len(args.trainloader)
        args.warmup_steps = int(args.total_steps * 0.1)

        if args.scheduler_type == "linear_with_warmup":
            args.scheduler = get_linear_schedule_with_warmup(
                optimizer=args.optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=args.total_steps,
            )
            print(f"[TRAINER] Linear scheduler initialized with warmup steps: {args.warmup_steps} and total steps: {args.total_steps}")
        else:
            raise ValueError(f"Invalid scheduler type for LLM: {args.scheduler_type}")
    else:
        args.scheduler = None
        print("[TRAINER] No scheduler initialized")

def _initialize_data_loaders(args):
    if args.model_type == 'llm':
        if args.model_task in get_dataset_config_names("glue"):
            data = get_glue_data(args.tokenizer, args.model_task, args.batch_size)
        elif args.model_task == 'wikitext2':
            data = get_wikitext2_data(args.tokenizer, args.batch_size, args.num_workers)

    elif args.model_type == 'cv':
        if hasattr(args, 'is_bacp') and args.is_bacp:
            data = get_cv_data(args.model_task, args.batch_size, learning_type='contrastive', size=args.image_size, num_workers=args.num_workers)
        else:
            data = get_cv_data(args.model_task, args.batch_size, learning_type=args.criterion_type, size=args.image_size, num_workers=args.num_workers)

    args.data = data
    args.trainloader = data["trainloader"]
    args.valloader = data["valloader"]
    args.testloader = data["testloader"] if args.model_task != 'sst2' else None

    print('[TRAINER] Data Initialized for model task:', args.model_task)
    print('[TRAINER] Batch size:', args.batch_size)
    print('[TRAINER] Number of dataloders:', len(data))

def _initialize_pruner(args):
    args.initial_sparsity = check_model_sparsity(args.model)
    args.prune = args.pruning_type is not None and args.target_sparsity > 0.0 and not args.finetune

    if args.prune:
        args.pruner = PRUNER_DICT[args.pruning_type](
            args.pruning_epochs, 
            args.target_sparsity, 
            args.model,
            args.sparsity_scheduler
        )
    else:
        args.pruner = args.pruner


    if args.prune:
        print('[TRAINER] Pruning initialized')
        print('[TRAINER] Pruning type:', args.pruning_type)
        print('[TRAINER] Target sparsity:', args.target_sparsity)
        print('[TRAINER] Sparsity scheduler:', args.sparsity_scheduler)
        print('[TRAINER] Pruning epochs:', args.pruning_epochs)
        print(f'[TRAINER] Current sparsity: {args.initial_sparsity:.4f}')
    elif args.finetune:
        print('[TRAINER] Finetuning initialized')
        print('[TRAINER] Pruning type:', args.pruning_type)
        print(f'[TRAINER] Current sparsity: {args.initial_sparsity:.4f}')
    else:
        print('[TRAINER] Pruning not initialized')

def _initialize_contrastive_losses(args, tau=0.07):
    args.n_views = 2
    args.temperature = tau
    args.base_temperature = tau
    args.supervised_loss = SupConLoss(args.n_views, args.temperature, args.base_temperature, args.batch_size)
    args.unsupervised_loss = NTXentLoss(args.n_views, args.temperature)

def _initialize_paths_and_logger(args):
    db = '/dbfs' if args.db else '.'
    args.base_path = os.path.join(db, 'research', args.model_name, args.model_task)

    if args.prune or args.finetune:
        weights_path = f'{args.model_name}_{args.model_task}_{args.pruning_type}_{args.target_sparsity}_{args.learning_type}.pt'
        logger_path = os.path.join(args.model_task, args.learning_type, args.pruning_type or "", str(args.target_sparsity))
    else:
        weights_path = f"{args.model_name}_{args.model_task}_{args.learning_type}.pt"
        logger_path = os.path.join(args.model_task, args.learning_type)

    args.save_path = os.path.join(args.base_path, weights_path)
    args.logger = Logger(args.model_name, logger_path) if args.log_epochs else None
    print(f'[TRAINER] Saving model to: {args.save_path}')

def _apply_pruning(args, epoch, step):
    if isinstance(args.pruner, MovementPrune):
        args.pruner.update_movement_scores(args.model, lr=args.optimizer.param_groups[0]['lr'])

    if not args.prune or args.pruner is None or args.recover or args.finetune:
        return
    
    if isinstance(args.pruner, MovementPrune):
        if step + 1 == len(args.trainloader):
            args.pruner.ratio_step(epoch, args.epochs, args.initial_sparsity, args.target_sparsity)
            args.pruner.prune(args.model)
    elif step == 0:
        args.pruner.ratio_step(epoch, args.epochs, args.initial_sparsity, args.target_sparsity)
        args.pruner.prune(args.model)
    

def _handle_optimizer_and_pruning(args, loss, epoch, step):
    """Handle backpropagation, pruning, and weight update in a single step."""
    args.optimizer.zero_grad()

    if args.enable_mixed_precision:
        args.scaler.scale(loss).backward()
        _apply_pruning(args, epoch, step)

        if not getattr(args, "skip_optimizer_step", False):
            args.scaler.step(args.optimizer)
            args.scaler.update()
    else:
        loss.backward()
        _apply_pruning(args, epoch, step)

        if not getattr(args, "skip_optimizer_step", False):
            args.optimizer.step()

    if args.scheduler and not getattr(args, "skip_optimizer_step", False):
        args.scheduler.step()

    if args.pruner is not None:
        args.pruner.apply_mask(args.model)













