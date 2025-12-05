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
    get_timestamp
)

def run_training(args):
    # Display device info
    device = get_device()

    # Setup training arguments
    args_dict = vars(args)

    log_to_wandb = args_dict.pop('log_to_wandb')
    seed = args_dict.pop('seed')
    set_seed(seed)
    
    bacp_training_args = BaCPTrainingArguments(**args_dict)
    bacp_trainer = BaCPTrainer(bacp_training_args)
    if log_to_wandb:
        wandb_login()
        group = f'{args.model_name}-{args.experiment_type}'
        name = f'{args.model_name}-{args.experiment_type}-{args.dataset_name}-{get_timestamp()}'
        with wandb.init(
            project='Backbone-Contrastive-Pruning',
            group=group,
            name=name,
            tags=[args.model_name, args.experiment_type, args.dataset_name, args.pruning_type, str(args.target_sparsity)],
            config=bacp_trainer.logger_params
        ) as run:
            # bacp_trainer.train(run)
            bacp_trainer.finetune(run)

            # Evaluate and log metrics
            metrics = bacp_trainer.evaluate(run)
            wandb.log(metrics)

            run.log_model(path=bacp_trainer.save_path, name=name)
            run.finish()
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





































