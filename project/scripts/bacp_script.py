import sys
import os
import argparse
from pathlib import Path

sys.path.append(os.path.abspath('..'))

import torch
import wandb
from datasets.utils.logging import disable_progress_bar

from trainer import Trainer, TrainingArguments
from bacp import BaCPTrainer, BaCPTrainingArguments
from utils import set_seed, get_device

# -------------------
# Environment setup
# -------------------
disable_progress_bar()
os.environ["HF_DATASETS_CACHE"] = "/dbfs/hf_datasets"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def wandb_login():
    """Log in to W&B using environment variable key."""
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key is None:
        raise ValueError("WANDB_API_KEY not set in environment.")
    wandb.login(key=wandb_api_key)

def sweep_train(args):
    # Display device info
    device = get_device()
    experiment_type = 'bacp_pruning'

    # Setup training arguments
    args_dict = vars(args)
    log_to_wandb = args_dict.pop('log_to_wandb')
    seed = args_dict.pop('seed')
    args_dict['experiment_type'] = experiment_type
    set_seed(seed)

    print(f"\nUsing device: {device}")
    print(f"Training {args.model_name} on {args.dataset_name}, Experiment: {experiment_type}")
    print(f"Final args for this run:\n{args_dict}")

    # Start a W&B run
    group = f'{args.model_name}-{experiment_type}'
    name = f'{args.model_name}-{args.dataset_name}-{args.pruning_type}-{args.target_sparsity}'
    with wandb.init(
        project='Backbone-Contrastive-Pruning',
        group=group,
        name=name,
        tags=[args.model_name, args.dataset_name, experiment_type, args.pruning_type, str(args.target_sparsity), 'sweep'],
    ) as run:
        # Merge sweep config into CLI/default args
        config = run.config
        args_dict.update(config)

        learning_rate_bacp = args_dict.pop('learning_rate_bacp')
        optimizer_type_bacp = args_dict.pop('optimizer_type_bacp')
        tau = args_dict.pop('tau')

        learning_rate_ip = args_dict.pop('learning_rate_ip')
        optimizer_type_ip = args_dict.pop('optimizer_type_ip')

        # BaCP trainer
        bacp_dict = args_dict.copy()
        bacp_dict.update({
            'learning_rate': learning_rate_bacp, 
            'optimizer_type': optimizer_type_bacp, 
            'tau': tau, 
            'epochs': 5
            })
        bacp_args = BaCPTrainingArguments(**bacp_dict)
        bacp_trainer = BaCPTrainer(bacp_args)
        bacp_trainer.train(run)

        # Baseline trainer
        pruner = bacp_trainer.get_pruner()
        trainer_dict = args_dict.copy()
        trainer_dict.update({
            'learning_rate': learning_rate_ip, 
            'optimizer_type': optimizer_type_ip, 
            'epochs': 50, 
            'pruning_module': pruner,
            'trained_weights': bacp_trainer.save_path,
            'experiment_type': 'bacp_finetuning'
            })
        training_args = TrainingArguments(**trainer_dict)
        trainer = Trainer(training_args)
        trainer.train(run)

        # Evaluate and log metrics
        metrics = trainer.evaluate(run)
        wandb.log(metrics)

        print("\nRun finished. Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="BaCP training script"
    )

    # Required args
    parser.add_argument('--model_name', type=str, choices=['resnet50', 'resnet101', 'vgg11', 'vgg19'])
    parser.add_argument('--model_type', type=str, choices=['cv', 'llm'])
    parser.add_argument('--dataset_name', type=str, choices=['cifar10', 'cifar100'])
    parser.add_argument('--num_classes', type=int)
    parser.add_argument('--pruning_type', type=str)
    parser.add_argument('--target_sparsity', type=float)
    parser.add_argument('--trained_weights', type=str)

    # Defaults
    parser.add_argument('--num_out_features', type=int, default=128)
    parser.add_argument('--sparsity_scheduler', type=str, default='cubic')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--scheduler_type', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--recovery_epochs', type=int, default=10)
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--databricks_env', type=bool, default=True)
    parser.add_argument('--log_to_wandb', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--enable_tqdm', type=bool, default=False)

    return parser.parse_args()

def main():
    """Main training pipeline for command-line usage."""
    args = parse_args()
    trainer, metrics = run_training(args)

    for key, value in metrics.items():
        print(f"{key}: {value}")

if __name__ == '__main__':
    main()












