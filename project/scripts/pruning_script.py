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

# Environment setup
disable_progress_bar()
os.environ["HF_DATASETS_CACHE"] = "/dbfs/hf_datasets"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

TARGET_SPARSITY_LOW  = 0.95
TARGET_SPARSITY_MID  = 0.97
TARGET_SPARSITY_HIGH = 0.99

def wandb_login():
    wandb_api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=wandb_api_key)

def run_training(args):
    # Display device info
    device = get_device()
    experiment_type = 'pruning'    

    # Setup training arguments
    args_dict = vars(args)
    log_to_wandb = args_dict.pop('log_to_wandb')
    seed = args_dict.pop('seed')
    args_dict['experiment_type'] = experiment_type
    set_seed(seed)

    print(f"\nUsing device: {device}")
    print(f"Training {args.model_name} on {args.dataset_name}")
    
    training_args = TrainingArguments(**args_dict)
    trainer = Trainer(training_args)
    
    if log_to_wandb:
        wandb_login()
        group = f'{args.model_name}-{experiment_type}'
        name = f'{args.model_name}-{args.dataset_name}-{args.pruning_type}-{str(args.target_sparsity)}'
        with wandb.init(
            project='Backbone-Contrastive-Pruning',
            group=group,
            name=name,
            tags=[args.model_name, args.dataset_name, experiment_type, args.pruning_type, str(args.target_sparsity)],
            config=trainer.logger_params,
        ) as run:
            trainer.train(run)
            metrics = trainer.evaluate(run)

            run.log_model(
                path=trainer.save_path, 
                name=name, 
                )
            run.finish()
    else:
        trainer.train() 
        metrics = trainer.evaluate()
    return trainer, metrics

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train models with optional pruning and contrastive learning"
    )

    # Arguments
    parser.add_argument('--model_name', type=str, choices=['resnet50', 'resnet101', 'vgg11', 'vgg19'])
    parser.add_argument('--model_type', type=str, choices=['cv', 'llm'])
    parser.add_argument('--dataset_name', type=str, choices=['cifar10', 'cifar100'])
    parser.add_argument('--num_out_features', type=int)

    # Pruning arguments
    parser.add_argument('--pruning_type', type=str, choices=['magnitude_pruning', 'snip_pruning', 'wanda_pruning'])
    parser.add_argument('--target_sparsity', type=float, choices=[TARGET_SPARSITY_LOW, TARGET_SPARSITY_MID, TARGET_SPARSITY_HIGH])
    parser.add_argument('--trained_weights', type=str)

    # Default pruning arguments
    parser.add_argument('--sparsity_scheduler', type=str, default='cubic')
    parser.add_argument('--recovery_epochs', type=int, default=10)
    parser.add_argument('--retrain', type=bool, default=True)

    # Default arguments
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--optimizer_type', type=str, default='sgd')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--scheduler_type', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=20)
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





































