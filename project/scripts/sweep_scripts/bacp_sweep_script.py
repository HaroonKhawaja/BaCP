import sys
import os
import argparse
from pathlib import Path

# -------------------------
# Path and imports
# -------------------------
sys.path.append(os.path.abspath('..'))

import torch
import wandb
from datasets.utils.logging import disable_progress_bar

from trainer import Trainer, TrainingArguments
from bacp import BaCPTrainer, BaCPTrainingArguments
from utils import set_seed, get_device


# -------------------------
# Environment setup
# -------------------------
disable_progress_bar()
os.environ["HF_DATASETS_CACHE"] = "/dbfs/hf_datasets"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# -------------------------
# CLI Argument Parsing
# -------------------------
def parse_args():
    """Parse command line arguments for BaCP and baseline training."""
    parser = argparse.ArgumentParser(
        description="Train models with optional pruning and contrastive learning."
    )

    # Required arguments
    parser.add_argument('--model_name', type=str, required=True,
                        choices=['resnet50', 'resnet101', 'vgg11', 'vgg19'])
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['cv', 'llm'])
    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=['cifar10', 'cifar100'])
    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument('--pruning_type', type=str, required=True)
    parser.add_argument('--target_sparsity', type=float, required=True)
    parser.add_argument('--trained_weights', type=str, required=True)

    # Optional defaults
    parser.add_argument('--num_out_features', type=int, default=128)
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
    """Authenticate with Weights & Biases."""
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if not wandb_api_key:
        raise EnvironmentError(
            "WANDB_API_KEY is not set in your environment variables."
        )
    wandb.login(key=wandb_api_key)

def filter_kwargs(params_dict, target_class):
    """Extract only the parameters that target_class accepts."""
    import inspect
    sig = inspect.signature(target_class.__init__)
    valid_params = set(sig.parameters.keys()) - {'self'}
    return {k: v for k, v in params_dict.items() if k in valid_params}

def run_bacp_stage(args_dict, run):
    """Train the BaCP model (pretraining and pruning)."""
    bacp_params = args_dict.copy()
    bacp_params.update({
        'learning_rate': 0.1,
        'optimizer_type': 'sgd',
        'epochs': 5,
        'experiment_type': 'bacp_pretraining'
    })
    # Filter to only BaCPTrainingArguments parameters
    bacp_params = filter_kwargs(bacp_params, BaCPTrainingArguments)
    bacp_args = BaCPTrainingArguments(**bacp_params)
    bacp_trainer = BaCPTrainer(bacp_args)
    bacp_trainer.train(run)
    return bacp_trainer

def run_finetune_stage(args_dict, pruner, pretrained_path, run):
    """Fine-tune the pruned model (baseline)."""
    trainer_params = args_dict.copy()
    trainer_params.update({
        'learning_rate': trainer_params.pop('learning_rate_ip'),
        'optimizer_type': 'sgd',
        'epochs': 50,
        'pruning_module': pruner,
        'trained_weights': pretrained_path,
        'experiment_type': 'bacp_finetuning'
    })
    # Filter to only TrainingArguments parameters
    trainer_params = filter_kwargs(trainer_params, TrainingArguments)
    train_args = TrainingArguments(**trainer_params)
    trainer = Trainer(train_args)
    trainer.train(run)
    return trainer

def sweep_train(args):
    """Main training loop for a single sweep iteration."""
    set_seed(42)
    device = get_device()
    args_dict = vars(args).copy()

    sweep_name = f"{args.model_name}-{args.dataset_name}-{args.pruning_type}-{args.target_sparsity}"
    group_name = f"{args.model_name}-bacp-sweep"

    with wandb.init(
        project="Backbone-Contrastive-Pruning",
        group=group_name,
        name=f"{sweep_name}-run",
        tags=[
            args.model_name, args.dataset_name, 'bacp_pruning',
            args.pruning_type, str(args.target_sparsity), 'sweep'
        ],
    ) as run:
        config = run.config
        args_dict.update(config)

        print(f"\n[Device] {device}")
        print(f"[Config] Final parameters:\n{args_dict}")

        # BaCP pretraining + pruning
        lr = args_dict.pop('learning_rate_ip')
        bacp_trainer = run_bacp_stage(args_dict, run)

        # Fine-tuning
        args_dict.update({'learning_rate_ip': lr})
        pruner = bacp_trainer.get_pruner()
        baseline_trainer = run_finetune_stage(args_dict, pruner, bacp_trainer.save_path, run)

        # Evaluation
        metrics = baseline_trainer.evaluate(run)
        wandb.log(metrics)

        for key, val in metrics.items():
            print(f"  {key}: {val:.4f}")

def get_sweep_config():
    """Return W&B sweep configuration."""
    return {
        'method': 'grid',
        'metric': {'name': 'accuracy', 'goal': 'maximize'},
        'parameters': {
            'tau': {'values': [0.01, 0.07, 0.15, 0.3, 0.5, 1.0]},
            'learning_rate_ip': {'values': [0.005, 0.001, 0.0005, 0.0001]},
        }
    }


def main():
    args = parse_args()
    wandb_login()

    sweep_config = get_sweep_config()
    sweep_id = wandb.sweep(sweep_config, project="Backbone-Contrastive-Pruning")

    # Number of runs = len(tau_values) × len(lr_values)
    # total_runs = 5 * 4
    total_runs = 1

    wandb.agent(
        sweep_id,
        function=lambda: sweep_train(args),
        count=total_runs
    )


if __name__ == "__main__":
    main()