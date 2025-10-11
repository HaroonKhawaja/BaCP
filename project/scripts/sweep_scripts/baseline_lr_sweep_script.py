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

def run_training(args, sweep_run=None):
    """Run training with optional sweep parameters."""
    set_seed(args.seed)
    
    # Display device info
    device = get_device()
    print(f"Using device: {device}")
    print(f"Training {args.model_name} on {args.dataset_name}")
    
    # Override learning rate if running in sweep mode
    lr = args.lr
    if sweep_run is not None:
        lr = wandb.config.learning_rate
        print(f"Sweep learning rate: {lr}")
    
    # Setup training arguments
    training_args = TrainingArguments(
        model_name=args.model_name,
        model_type=args.model_type,
        dataset_name=args.dataset_name,
        num_out_features=args.num_out_features,
        batch_size=args.batch_size,
        optimizer_type_and_lr=(args.optimizer_type, lr),
        scheduler_type=args.scheduler_type,
        epochs=args.epochs,
        image_size=args.image_size,
        patience=args.patience,
        experiment_type=args.experiment_type,
        db=args.databricks_env,
        enable_tqdm=args.tqdm,
    )
    
    trainer = Trainer(training_args)
    
    if sweep_run is not None:
        # We're in a sweep, run is already initialized
        trainer.train(sweep_run)
        metrics = trainer.evaluate(sweep_run)
        sweep_run.log_model(
            path=trainer.save_path, 
            name=f'{args.model_name}-{args.experiment_type}-lr{lr}',
        )
    elif args.log_to_wandb:
        # Regular run with wandb
        wandb_login()
        group = f'{args.model_name}'
        name = f'{args.model_name}-{args.experiment_type}'
        with wandb.init(
            project='Backbone-Contrastive-Pruning',
            group=group,
            name=name,
            tags=[args.model_name, args.dataset_name, args.experiment_type],
            config=trainer.logger_params,
        ) as run:
            trainer.train(run)
            metrics = trainer.evaluate(run)
            run.log_model(
                path=trainer.save_path, 
                name=name,
            )
    else:
        # No wandb logging
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

    # Default arguments
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--optimizer_type', type=str, default='sgd')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--scheduler_type', type=str, choices=['linear_with_warmup'])
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--experiment_type', type=str, default='baseline')
    parser.add_argument('--databricks_env', type=bool, default=True)
    parser.add_argument('--log_to_wandb', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tqdm', type=bool, default=False)
    parser.add_argument('--sweep', action='store_true', help='Run in sweep mode')

    return parser.parse_args()

def sweep_train():
    """Training function for wandb sweep."""
    # Initialize wandb run for this sweep iteration
    wandb_login()
    
    # Parse base arguments
    args = parse_args()
    
    # Get sweep config values
    group = f'{args.model_name}-sweep'
    name = f'{args.model_name}-{args.experiment_type}-lr{wandb.config.learning_rate}'
    
    with wandb.init(
        group=group,
        name=name,
        tags=[args.model_name, args.dataset_name, args.experiment_type, 'sweep', str(wandb.config.learning_rate)],
    ) as run:
        # Log all configuration parameters
        wandb.config.update({
            'model_name': args.model_name,
            'model_type': args.model_type,
            'dataset_name': args.dataset_name,
            'num_out_features': args.num_out_features,
            'batch_size': args.batch_size,
            'optimizer_type': args.optimizer_type,
            'scheduler_type': args.scheduler_type,
            'epochs': args.epochs,
            'image_size': args.image_size,
            'patience': args.patience,
            'experiment_type': args.experiment_type,
            'seed': args.seed,
        }, allow_val_change=True)
        
        # Run training with sweep
        trainer, metrics = run_training(args, sweep_run=run)
        
        # Log final metrics
        for key, value in metrics.items():
            print(f"{key}: {value}")

def main():
    args = parse_args()
    if args.sweep:
        # Run in sweep mode
        sweep_train()
    else:
        # Run regular training
        trainer, metrics = run_training(args)
        for key, value in metrics.items():
            print(f"{key}: {value}")

sweep_config = {
    'method': 'grid',
    'metric': {'name': 'avg_acc', 'goal': 'maximize'},
    'parameters': {
        'learning_rate': {'values': [0.1, 0.05, 0.01, 0.005, 0.001]},
    }
}

if __name__ == '__main__':
    args = parse_args()
    
    if args.sweep:
        wandb_login()
        sweep_id = wandb.sweep(
            sweep_config, 
            project='Backbone-Contrastive-Pruning'
        )
        print(f"Starting sweep with ID: {sweep_id}")
        wandb.agent(sweep_id, function=sweep_train, count=5)
    else:
        # Run single training
        main()


