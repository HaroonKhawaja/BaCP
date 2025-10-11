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

def wandb_login():
    wandb_api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=wandb_api_key)

def run_training(args):
    # Display device info
    device = get_device()

    # Setup training arguments
    args_dict = vars(args)
    log_to_wandb = args_dict.pop('log_to_wandb')
    experiment_type = args_dict.pop('experiment_type') 
    seed = args_dict.pop('seed')
    set_seed(seed)

    print(f"\nUsing device: {device}")
    print(f"Training {args.model_name} on {args.dataset_name}")
    
    training_args = TrainingArguments(**args_dict)
    trainer = Trainer(training_args)

    if log_to_wandb:
        wandb_login()
        group = f'{args.model_name}-{experiment_type}'
        name = f'{args.model_name}-{args.dataset_name}'
        with wandb.init(
            project='Backbone-Contrastive-Pruning',
            group=group,
            name=name,
            tags=[args.model_name, args.dataset_name, experiment_type],
            config=trainer.logger_params
        ) as run:
            
            trainer.train(run)

            # Evaluate and log metrics
            metrics = trainer.evaluate(run)
            wandb.log(metrics)

            run.log_model(path=trainer.save_path, name=name)
            run.finish()
    else:
        trainer.train() 
        metrics = trainer.evaluate()
    
    print("\nRun finished. Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train models with optional pruning and contrastive learning"
    )

    # Arguments
    parser.add_argument('--model_name', type=str, choices=['resnet50', 'resnet101', 'vgg11', 'vgg19'])
    parser.add_argument('--model_type', type=str, choices=['cv', 'llm'])
    parser.add_argument('--dataset_name', type=str, choices=['cifar10', 'cifar100'])
    parser.add_argument('--num_classes', type=int)

    # Default arguments
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--optimizer_type', type=str, default='sgd')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--scheduler_type', type=str, default='linear_with_warmup')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--databricks_env', type=bool, default=True)
    parser.add_argument('--log_to_wandb', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--enable_tqdm', type=bool, default=False)
    parser.add_argument('--experiment_type', type=str, default='baseline')

    return parser.parse_args()

def main():
    """Main training pipeline for command-line usage."""
    args = parse_args()
    run_training(args)

if __name__ == '__main__':
    main()





































