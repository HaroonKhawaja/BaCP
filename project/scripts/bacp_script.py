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

disable_progress_bar()
os.environ["HF_DATASETS_CACHE"] = "/dbfs/hf_datasets"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def wandb_login():
    """Log in to W&B using environment variable key."""
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key is None:
        raise ValueError("WANDB_API_KEY not set in environment.")
    wandb.login(key=wandb_api_key)

def run_training(args):
    # Display device info
    device = get_device()

    # Setup training arguments
    args_dict = vars(args)
    log_to_wandb    = args_dict.pop('log_to_wandb')
    experiment_type = args_dict.pop('experiment_type')
    seed            = args_dict.pop('seed')
    set_seed(seed)

    print(f"\nUsing device: {device}")
    print(f"Training {args.model_name} on {args.dataset_name}, Experiment: {experiment_type}")
    print(f"Final args for this run:\n{args_dict}")

    # Start a W&B run
    group = f'{args.model_name}-{experiment_type}'
    name = f'{args.model_name}-{args.dataset_name}-{args.pruning_type}-{args.target_sparsity}'

    if log_to_wandb:
        wandb_login()
        with wandb.init(
            project='Backbone-Contrastive-Pruning',
            group=group,
            name=name,
            tags=[args.model_name, args.dataset_name, experiment_type, args.pruning_type, str(args.target_sparsity)],
        ) as run:
            # BaCP training parameters
            learning_rate_bacp = args_dict.pop('learning_rate_bacp')
            optimizer_type_bacp = args_dict.pop('optimizer_type_bacp')
            tau = args_dict.pop('tau')

            # Finetuning training parameters
            learning_rate_ip = args_dict.pop('learning_rate_ip')
            optimizer_type_ip = args_dict.pop('optimizer_type_ip')

            # BaCP trainer
            bacp_dict = args_dict.copy()
            bacp_dict.update({
                'learning_rate': learning_rate_bacp, 
                'optimizer_type': optimizer_type_bacp, 
                'tau': tau, 
                })
            bacp_args = BaCPTrainingArguments(**bacp_dict)
            bacp_trainer = BaCPTrainer(bacp_args)

            bacp_trainer.train(run)
            run.log_model(path=bacp_trainer.save_path, name=name)

            # Baseline trainer
            pruner = bacp_trainer.get_pruner()
            trainer_dict = args_dict.copy()
            trainer_dict.update({
                'learning_rate': learning_rate_ip, 
                'optimizer_type': optimizer_type_ip, 
                'epochs': 50, 
                'trained_weights': bacp_trainer.save_path,
                'experiment_type': 'bacp_finetuning',
                'pruning_module': pruner,
                })
            training_args = TrainingArguments(**trainer_dict)
            trainer = Trainer(training_args)
            trainer.train(run)

            # Evaluate and log metrics
            metrics = trainer.evaluate(run)
            wandb.log(metrics)

            run.log_model(path=trainer.save_path, name=name)
            run.finish()
    else:
        raise NotImplementedError

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

    # Default optimizers, learning rate, and tau (we can edit these later)
    parser.add_argument('--learning_rate_bacp', type=float, default=0.1)
    parser.add_argument('--optimizer_type_bacp', type=str, default='sgd')
    parser.add_argument('--tau', type=float, default=0.15)

    parser.add_argument('--learning_rate_ip', type=float, default=0.005)
    parser.add_argument('--optimizer_type_ip', type=str, default='sgd')
    
    # Default epochs of 5, batch size of 512, and no training scheduler
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--scheduler_type', type=str, default=None)

    # Default cubic sparsity scheduler and 10 recovery epochs
    parser.add_argument('--sparsity_scheduler', type=str, default='cubic')
    parser.add_argument('--recovery_epochs', type=int, default=10)

    # Default small image size of 32, embedding dimensions of 128
    parser.add_argument('--num_out_features', type=int, default=128)
    parser.add_argument('--image_size', type=int, default=32)

    parser.add_argument('--databricks_env', type=bool, default=True)
    parser.add_argument('--log_to_wandb', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--enable_tqdm', type=bool, default=False)
    parser.add_argument('--experiment_type', type=str, default='bacp_pruning')

    return parser.parse_args()

def main():
    """Main training pipeline for command-line usage."""
    args = parse_args()
    run_training(args)

if __name__ == '__main__':
    main()












