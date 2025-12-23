import sys
import os
sys.path.append(os.path.abspath('..'))

import torch
import wandb
from pathlib import Path

from bacp import BaCPTrainer, BaCPTrainingArguments
from utils import set_seed, get_device
from scripting_utils import (
    wandb_login,
    bacp_parse_args,
    # get_timestamp removed from imports
)

def run_training(args):
    # Display device info
    device = get_device()

    # Setup training arguments
    args_dict = vars(args)

    # Pop non-trainer arguments before creating BaCPTrainingArguments
    log_to_wandb = args_dict.pop('log_to_wandb')
    seed = args_dict.pop('seed')
    set_seed(seed)
    
    # Initialize arguments and trainer
    bacp_training_args = BaCPTrainingArguments(**args_dict)
    bacp_trainer = BaCPTrainer(bacp_training_args)
    
    # Access the final, initialized arguments from the trainer
    final_args = bacp_trainer    

    if log_to_wandb:
        wandb_login()
        
        datestamp = final_args.datestamp
        group = f'{final_args.model_name}-{final_args.experiment_type}'
        name = f'{final_args.model_name}-{final_args.experiment_type}-{final_args.dataset_name}-{datestamp}'
        
        tags = [
            final_args.model_name, 
            final_args.experiment_type, 
            final_args.dataset_name, 
            final_args.pruning_type, 
            str(final_args.target_sparsity)
        ]
        
        with wandb.init(
            project='Backbone-Contrastive-Pruning',
            group=group,
            name=name,
            tags=[tag for tag in tags if tag], 
            config=bacp_trainer.logger_params
        ) as run:
            bacp_trainer.train(run)

            # Evaluate and log metrics
            metrics = bacp_trainer.evaluate(run)
            wandb.log(metrics)

            # Use the save path already set in the trainer
            run.log_model(path=bacp_trainer.save_path, name=name)

    else:
        bacp_trainer.train() 
        metrics = bacp_trainer.evaluate()

    print("\nRun finished. Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")


def main():
    args = bacp_parse_args()
    run_training(args)

if __name__ == '__main__':
    main()