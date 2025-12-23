import os
import argparse
import wandb
import datetime

def wandb_login():
    api_key = os.getenv("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)
    else:
        print("[WARNING] WANDB_API_KEY not found, skipping login.")


def get_timestamp():
    timestamp = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    return timestamp

models      = ['resnet34', 'resnet50', 'resnet101', 'vgg11', 'vgg19']
model_types = ['cv', 'llm']
datasets    = ['cifar10', 'cifar100']


def baseline_parse_args():
    parser = argparse.ArgumentParser(
        description="Train a baseline dense model."
    )

    # Required arguments
    parser.add_argument('--model_name',     type=str, choices=models,       required=True)
    parser.add_argument('--model_type',     type=str, choices=model_types,  required=True)
    parser.add_argument('--dataset_name',   type=str, choices=datasets,     required=True)
    parser.add_argument('--num_classes',    type=int,                       required=True)

    # Optional arguments with defaults
    parser.add_argument('--batch_size',      type=int,      default=512)
    parser.add_argument('--optimizer_type',  type=str,      default='sgd')
    parser.add_argument('--learning_rate',   type=float,    default=0.1)
    parser.add_argument('--image_size',      type=int,      default=32)
    parser.add_argument('--epochs',          type=int,      default=100)
    parser.add_argument('--scheduler_type',  type=str,      default='linear_with_warmup')
    parser.add_argument('--patience',        type=int,      default=20)
    parser.add_argument('--experiment_type', type=str,      default='baseline')
    parser.add_argument('--log_epochs',          action='store_true', help="Enable detailed logging per epoch to the local logger.")
    parser.add_argument('--enable_tqdm',         action='store_true', help="Enable TQDM progress bars during training.")
    parser.add_argument('--databricks_env',      action='store_true', help="Set paths for Databricks environment (e.g., /dbfs).")

    parser.add_argument('--num_workers',     type=int,      default=os.cpu_count())

    # DyReLU phasing
    parser.add_argument('--dyrelu_en',         action='store_true')
    parser.add_argument('--dyrelu_phasing_en', action='store_true')

    # Non-trainer args
    parser.add_argument('--log_to_wandb',        action='store_true', help="Log results to Weights & Biases (WANDB).")
    parser.add_argument('--seed',                type=int,       default=42)

    return parser.parse_args()


def pruning_parse_args():
    parser = argparse.ArgumentParser(
        description="Prune a model using a specific pruning strategy"
    )

    # Required arguments
    parser.add_argument('--model_name',         type=str, choices=models,       required=True)
    parser.add_argument('--model_type',         type=str, choices=model_types,  required=True)
    parser.add_argument('--dataset_name',       type=str, choices=datasets,     required=True)
    parser.add_argument('--num_classes',        type=int,                       required=True)
    parser.add_argument('--trained_weights',    type=str,                       required=True)
    parser.add_argument('--pruning_type',       type=str,                       required=True)
    parser.add_argument('--target_sparsity',    type=float,                     required=True)
    parser.add_argument('--sparsity_scheduler', type=str,                       required=True)
    

    # Optional arguments with defaults
    parser.add_argument('--batch_size',      type=int,      default=512)
    parser.add_argument('--optimizer_type',  type=str,      default='sgd')
    parser.add_argument('--learning_rate',   type=float,    default=0.1)
    parser.add_argument('--image_size',      type=int,      default=32)

    parser.add_argument('--epochs',          type=int,      default=5)
    parser.add_argument('--recovery_epochs', type=int,      default=10)
    parser.add_argument('--scheduler_type',  type=str,      default=None)
    parser.add_argument('--patience',        type=int,      default=None)
    parser.add_argument('--experiment_type', type=str,      default='pruning')
    parser.add_argument('--log_epochs',      action='store_true', help="Enable detailed logging per epoch to the local logger.")
    parser.add_argument('--enable_tqdm',     action='store_true', help="Enable TQDM progress bars during training.")
    parser.add_argument('--databricks_env',  action='store_true', help="Set paths for Databricks environment (e.g., /dbfs).")
    parser.add_argument('--num_workers',     type=int,      default=os.cpu_count())

    # DyReLU phasing
    parser.add_argument('--dyrelu_en',         action='store_true')
    parser.add_argument('--dyrelu_phasing_en', action='store_true')

    # Non-trainer args
    parser.add_argument('--log_to_wandb',        action='store_true', help="Log results to Weights & Biases (WANDB).")
    parser.add_argument('--seed',                type=int,       default=42)

    return parser.parse_args()


def bacp_parse_args():
    parser = argparse.ArgumentParser(
        description="BaCP method for pruning a model using a specific pruning strategy"
    )

    # Required arguments
    parser.add_argument('--model_name',          type=str, choices=models,       required=True)
    parser.add_argument('--model_type',          type=str, choices=model_types,  required=True)
    parser.add_argument('--dataset_name',        type=str, choices=datasets,     required=True)
    parser.add_argument('--num_classes',         type=int,                       required=True)
    parser.add_argument('--trained_weights',     type=str,                       required=True)
    parser.add_argument('--pruning_type',        type=str,                       required=True)
    parser.add_argument('--target_sparsity',     type=float,                     required=True)
    parser.add_argument('--sparsity_scheduler',  type=str,                       required=True)
    parser.add_argument('--recovery_epochs',     type=int,                       required=True)
    
    # Optional arguments with defaults
    parser.add_argument('--batch_size',          type=int,       default=512)
    parser.add_argument('--tau',                 type=float,     default=0.07, help="Temperature parameter for InfoNCE loss (Contrastive Learning).")

    # Optimizer and lr for pretraining and fine tuning
    parser.add_argument('--optimizer_type',      type=str,       default='sgd', help="Optimizer for pre-training phase (Contrastive Loss).")
    parser.add_argument('--learning_rate',       type=float,     default=0.1, help="Learning rate for pre-training phase.")
    parser.add_argument('--epochs',              type=int,       default=50, help="Number of epochs for pre-training phase.")

    # Finetuning arguments
    parser.add_argument('--enable_finetune',     action='store_true', help="Enable supervised fine-tuning phase after pre-training.")
    parser.add_argument('--optimizer_type_ft',   type=str,       default='sgd', help="Optimizer for fine-tuning phase (Classification Loss).")
    parser.add_argument('--learning_rate_ft',    type=float,     default=0.005, help="Learning rate for fine-tuning phase.")
    parser.add_argument('--epochs_ft',           type=int,       default=100, help="Number of epochs for fine-tuning phase.")

    parser.add_argument('--image_size',          type=int,       default=32)
    parser.add_argument('--scheduler_type',      type=str,       default=None)
    parser.add_argument('--patience',            type=int,       default=None)
    parser.add_argument('--experiment_type',     type=str,       default='bacp')
    
    # Boolean flags corrected to use action='store_true' (default=False)
    parser.add_argument('--log_epochs',          action='store_true', help="Enable detailed logging per epoch to the local logger.")
    parser.add_argument('--enable_tqdm',         action='store_true', help="Enable TQDM progress bars during training.")
    parser.add_argument('--databricks_env',      action='store_true', help="Set paths for Databricks environment (e.g., /dbfs).")
    parser.add_argument('--num_workers',         type=int,       default=os.cpu_count())

    # DyReLU phasing
    parser.add_argument('--dyrelu_en',           action='store_true', help="Enable DyReLU activation function.")
    parser.add_argument('--dyrelu_phasing_en',   action='store_true', help="Enable phasing mechanism for DyReLU.")
    
    # Non-trainer args
    parser.add_argument('--log_to_wandb',        action='store_true', help="Log results to Weights & Biases (WANDB).")
    parser.add_argument('--seed',                type=int,       default=42)

    return parser.parse_args()






