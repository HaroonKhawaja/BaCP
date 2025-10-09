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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train models with optional pruning and contrastive learning"
    )

    # Required args
    parser.add_argument('--model_name', type=str, choices=['resnet50', 'resnet101', 'vgg11', 'vgg19'])
    parser.add_argument('--model_type', type=str, choices=['cv', 'llm'])
    parser.add_argument('--dataset_name', type=str, choices=['cifar10', 'cifar100'])
    parser.add_argument('--num_classes', type=int)
    parser.add_argument('--num_out_features', type=int)
    parser.add_argument('--pruning_type', type=str)
    parser.add_argument('--target_sparsity', type=float)
    parser.add_argument('--trained_weights', type=str)

    # Defaults
    parser.add_argument('--sparsity_scheduler', type=str, default='cubic')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--scheduler_type', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--recovery_epochs', type=int, default=10)
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--databricks_env', type=bool, default=True)
    parser.add_argument('--enable_tqdm', type=bool, default=False)

    return parser.parse_args()

def wandb_login():
    """Log in to W&B using environment variable key."""
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key is None:
        raise ValueError("WANDB_API_KEY not set in environment.")
    wandb.login(key=wandb_api_key)

sweep_count = 0
def sweep_train(args):
    global sweep_count

    """Main training loop for a single sweep run."""
    device = get_device()
    args_dict = vars(args)

    set_seed(42)
    experiment_type = 'bacp_pruning'
    args_dict['experiment_type'] = experiment_type

    # Start a W&B run
    sweep_count += 1
    group = f'{args.model_name}-{experiment_type}-sweep'
    name = f'{args.model_name}-{args.dataset_name}-{args.pruning_type}-{args.target_sparsity}-sweep-{str(sweep_count)}'
    with wandb.init(
        project='Backbone-Contrastive-Pruning',
        group=group,
        name=name,
        tags=[args.model_name, args.dataset_name, experiment_type, args.pruning_type, str(args.target_sparsity), 'sweep'],
    ) as run:
        # Merge sweep config into CLI/default args
        config = run.config
        args_dict.update(config)

        print(f"\nUsing device: {device}")
        print(f"Final args for this run:\n{args_dict}")

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

sweep_config = {
    'method': 'grid',
    'metric': {'name': 'accuracy', 'goal': 'maximize'},
    'parameters': {
        'optimizer_type_bacp': {'values': ['adamw']},
        'optimizer_type_ip': {'values': ['sgd', 'adamw']},
        'tau': {'values': [  0.5, 0.7, 0.9]},
        'learning_rate_bacp': {'values': [0.05, 0.01, 0.005, 0.001]},
        'learning_rate_ip': {'values': [0.001, 0.0005, 0.0001]},
    }
}

if __name__ == '__main__':
    args = parse_args()
    wandb_login()

    sweep_id = wandb.sweep(sweep_config, project='Backbone-Contrastive-Pruning')
    wandb.agent(sweep_id, function=lambda: sweep_train(args), count=2*2*7*5*5)
