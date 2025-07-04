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
from dataset_utils import get_glue_data, get_squad_data, get_wikitext2_data, get_cv_data, CV_DATASETS
from logger import Logger
from loss_fn import *
from models import ClassificationNetwork, EncoderProjectionNetwork
from unstructured_pruning import check_model_sparsity, PRUNER_DICT
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
    elif args.model_task in ['squad']:
        args.num_classes = None
    else:
        raise ValueError(f'Invalid model task: {args.model_task}. Choose from {list(task_num_classes.keys())}')

def _detect_criterion(args):
    args.criterion = None if args.model_type == 'llm' else CrossEntropyLoss()

def _detect_cv_image_size(args):
    if args.model_type == 'cv':
        # if args.model_name.startswith('vit') or args.model_task in ['food101', 'caltech101']:        
        if args.model_task in ['food101', 'caltech101']:
            args.image_size = 224
        elif args.model_task == 'flowers102':
            args.image_size = 256
        else:
            args.image_size = 32
    else:
        args.image_size = None
    print(f'[TRAINER] Image size: {args.image_size}')

def _initialize_models(args):
    if args.is_bacp:
        models = create_models_for_bacp(args.model_name, args.model_task, args.image_size, args.finetuned_weights, args.device)
        args.model = models["curr_model"]
        args.pre_trained_model = models["pt_model"]
        args.finetuned_model = models["ft_model"]
        args.embedded_dim = args.model.embedding_dim
        print('[TRAINER] Initialized BaCP models')

    else:
        args.model = ClassificationNetwork(
            model_name=args.model_name, 
            num_classes=args.num_classes, 
            adapt=(True if args.image_size and args.image_size <= 64 else False), 
            model_task=args.model_task
            )
        if args.model_type == 'llm' and args.model_task == 'squad':
            args.squad_metric = evaluate.load(args.model_task)
        args.embedded_dim = args.model.embedding_dim
        print('[TRAINER] Initialized models')

        if args.finetuned_weights:
            print(f'[TRAINER] Loading weights: {args.finetuned_weights}')
            loaded = load_weights(args.model, args.finetuned_weights)
            print("[TRAINER] Weights loaded" if loaded else "[TRAINER] Failed to load weights")

def create_models_for_bacp(model_name, model_task, image_size, finetuned_weights, device, output_dimensions=128):
    pre_trained_model = EncoderProjectionNetwork(
        model_name=model_name, 
        output_dims=output_dimensions, 
        adapt=(True if image_size and image_size <= 64 else False), 
        model_task=model_task)
    pre_trained_model.to(device)

    # Current projection model
    model = deepcopy(pre_trained_model).to(device)

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
        elif args.model_task == 'squad':
            data = get_squad_data(args.tokenizer, args.batch_size, 1.0, args.num_workers)
        elif args.model_task == 'wikitext2':
            data = get_wikitext2_data(args.tokenizer, args.batch_size, args.num_workers)

    elif args.model_type == 'cv':
        if hasattr(args, 'is_bacp') and args.is_bacp:
            data = get_cv_data(args.model_task, args.batch_size, learning_type='contrastive', size=args.image_size, num_workers=args.num_workers)
        else:
            data = get_cv_data(args.model_task, args.batch_size, args.image_size, args.num_workers)

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
        if "wanda" in args.pruning_type:
            args.pruner = PRUNER_DICT[args.pruning_type](
                args.pruning_epochs, 
                args.target_sparsity,
                args.model, 
                args.sparsity_scheduler
            )
        else:
            args.pruner = PRUNER_DICT[args.pruning_type](
                args.pruning_epochs, 
                args.target_sparsity, 
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
        print('[TRAINER] Current sparsity:', args.initial_sparsity)
    elif args.finetune:
        print('[TRAINER] Finetuning initialized')
        print('[TRAINER] Pruning type:', args.pruning_type)
        print('[TRAINER] Current sparsity:', args.initial_sparsity)
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
        logger_path = os.path.join(args.model_task, args.learning_type, args.pruning_type, str(args.target_sparsity))
    else:
        weights_path = f"{args.model_name}_{args.model_task}_{args.learning_type}.pt"
        logger_path = os.path.join(args.model_task, args.learning_type)

    args.save_path = os.path.join(args.base_path, weights_path)
    args.logger = Logger(args.model_name, logger_path) if args.log_epochs else None
    print(f'[TRAINER] Saving model to: {args.save_path}')

def _apply_pruning(args, epoch, step):
    if (not args.prune and args.pruner is None) or args.recover or args.finetune:
        return
    if step == 0:
        args.pruner.ratio_step(epoch, args.epochs, args.initial_sparsity, args.target_sparsity)
        args.pruner.prune(args.model)

def _handle_optimizer_and_pruning(args, loss, epoch, step):
    """Handle backpropagation, pruning, and weight update in a single step."""
    args.optimizer.zero_grad()

    if args.enable_mixed_precision:
        args.scaler.scale(loss).backward()

        if hasattr(args, 'pruning_type') and args.pruning_type == 'movement_pruning':
            args.scaler.unscale_(args.optimizer)
            if 'vgg' in args.model_name.lower():
                torch.nn.utils.clip_grad_norm_(args.model.parameters(), max_norm=1.0)

        _apply_pruning(args, epoch, step)

        if not getattr(args, "skip_optimizer_step", False):
            args.scaler.step(args.optimizer)
            args.scaler.update()
    else:
        loss.backward()

        if hasattr(args, 'pruning_type') and args.pruning_type == 'movement_pruning':
            if 'vgg' in args.model_name.lower():
                torch.nn.utils.clip_grad_norm_(args.model.parameters(), max_norm=1.0)
    
        _apply_pruning(args, epoch, step)

        if not getattr(args, "skip_optimizer_step", False):
            args.optimizer.step()

    if args.scheduler and not getattr(args, "skip_optimizer_step", False):
        args.scheduler.step()

    if args.finetune or (args.prune and args.pruner is not None):
        args.pruner.apply_mask(args.model)

class LRFinder:
    def __init__(self, trainer, min_lr=1e-6, max_lr=0.5, lr_steps=101):
        self.trainer = trainer
        self.max_steps = int(len(trainer.trainloader) * 0.1)
        self.lrs = torch.linspace(min_lr, max_lr, steps=lr_steps)
        self.losses = torch.zeros(lr_steps)
        
    @classmethod
    def find_lr(cls, trainer, min_lr=1e-6, max_lr=0.5):
        self = cls(trainer, min_lr, max_lr)
        self._run()
    
    def _run(self):

        original_amp = self.trainer.enable_mixed_precision
        self.trainer.enable_mixed_precision = False
        self.trainer.enable_tqdm = False
        self.trainer.skip_optimizer_step = True
        self.trainer.trainloader = self.trainer.data['unaugmentedloader']

        i = 0
        lr_batch = tqdm(self.lrs)
        for lr in lr_batch:

            if self.trainer.optimizer_type == 'adamw':
                self.trainer.optimizer = optim.AdamW(self.trainer.model.parameters(), lr=lr)
            elif self.trainer.optimizer_type == 'sgd':
                self.trainer.optimizer = optim.SGD(self.trainer.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
            else:
                raise ValueError(f"Invalid optimizer type: {self.trainer.optimizer_type}")

            self.trainer.model.to(self.trainer.device)
            running_loss = self.trainer._run_train_epoch(0, "", 1)
            self.losses[i] = running_loss
            i += 1

            desc = f'loss for lr({lr:.2e}) = {running_loss:.5f}'
            lr_batch.set_description(desc)

        
        self.trainer.enable_mixed_precision = original_amp

        plt.plot(self.lrs.tolist(), self.losses.tolist())
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate Finder')
        plt.grid(True)
        plt.show()

    

















