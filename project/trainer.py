import re
import os
import string
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from datasets.utils.logging import disable_progress_bar
from unstructured_pruning import check_model_sparsity
from logger import Logger
import contextlib
from constants import *
from utils import *
from training_utils import (
    _initialize_models,
    _initialize_data_loaders,
    _initialize_optimizer,
    _initialize_scheduler,
    _initialize_pruner,
    _initialize_paths_and_logger,
    _handle_optimizer_and_pruning,
)
from dataclasses import dataclass, field
from typing import Any, Dict

disable_progress_bar()
os.environ["HF_DATASETS_CACHE"] = "/dbfs/hf_datasets"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class TrainingArguments:
    model_name:             str
    model_type:             str
    dataset_name:           str
    num_classes:            int
    batch_size:             int
    optimizer_type:         str
    learning_rate:          float
    criterion:              nn.Module = nn.CrossEntropyLoss()
    num_out_features:       int = None
    image_size:             int = 32        # Image size for resizing
    epochs:                 int = 5         # Number of epochs to train
    scheduler_type:         str = None      # Scheduler type, e.g., "linear_with_warmup"
    patience:               int = 20        # Number of epochs to wait before early stopping
    trained_weights:        str = None      # Path to pretrained weights
    experiment_type:        str = ""        # Type of experiment, e.g., "baseline" or "pruning"
    log_epochs:             bool = False    # Whether to log epochs in directory
    enable_tqdm:            bool = False    # Whether to enable tqdm progress bar
    enable_mixed_precision: bool = True     # Whether to enable mixed precision training
    databricks_env:         bool = True     # Whether to save model weights to DBFS (only when using databricks)
    num_workers:            int = os.cpu_count()

    # Pruning arguments
    pruning_type:           str = None      # Pruning type, e.g., "magnitude_pruning"
    target_sparsity:        float = None    # Target sparsity for pruning
    sparsity_scheduler:     str = None      # Sparsity scheduler, e.g., "cubic"
    recovery_epochs:        int = None      # Number of epochs to recover after pruning
    retrain:                bool = False    # Whether to retrain after pruning
    pruning_module:         object = None   # Pruning module

    def __post_init__(self):
        self.scaler = GradScaler() if self.enable_mixed_precision else None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_bacp = False

        _initialize_models(self)
        _initialize_data_loaders(self)
        _initialize_optimizer(self)
        _initialize_scheduler(self)
        _initialize_pruner(self)
        _initialize_paths_and_logger(self)

class Trainer:
    """Unified trainer class for training, fine-tuning, and pruning both CV and LLM models."""
    def __init__(self, training_args):
        for key, value in vars(training_args).items():
            setattr(self, key, value)

        # Initialize training state
        self.recover = False
        self.unchanged = 0
        self.train_losses = []
        self.val_accuracies = []
        self.accuracies = [] if self.target_sparsity is None else {}
        self.current_sparsity = check_model_sparsity(self.model)
        self.context = autocast(device_type=self.device) if self.enable_mixed_precision else contextlib.nullcontext()
        self._initialize_log_parameters()

    def _initialize_log_parameters(self):
        """Initialize parameters for logging purposes."""
        allowed_types = (int, float, str, bool, torch.Tensor, np.ndarray, list, dict, type(None))
        self.logger_params = {
            k: v for k, v in vars(self).items()
            if isinstance(v, allowed_types) and v is not None
        }

    def train(self, run=None):
        """Main training loop that handles training, pruning, retraining, and logging."""
        self._initialize_logs()

        for epoch in range(self.epochs):
            curr_epoch_str = f"Epoch [{epoch+1}/{self.epochs}]"

            # Training
            desc = f"Training {curr_epoch_str}"
            loss = self._run_train_epoch(epoch, desc)

            # Validation
            desc = f"Validation {curr_epoch_str}"
            metrics = self._run_validation_epoch(desc)

            # Appends training loss and validation accuracy to lists
            self._update_metric_lists(loss, metrics.get('accuracy'))              
            
            # Logs these metrics to the logger or to W&B
            metrics['loss'] = loss
            self._log_metrics(curr_epoch_str, metrics, run) 

            # Saving model (early stopping based on validation accuracy)
            if not self._handle_save(epoch):
                break
            
            # Pruning and recovery schedule
            if self.retrain:
                self._retrain(run)

    def evaluate(self, load=True, run=None):
        """Evaluate model performnace on the testing dataset"""
        if load:
            if self.save_path and load_weights(self.model, self.save_path):
                print("[TRAINER] Weights loaded successfully")
            else:
                print("[TRAINER] Failed to load weights")

        self.model.eval()
        self.model.to(self.device)

        desc = "Evaluating"
        metrics = self._run_validation_epoch(desc, 'eval')

        sparsity = self._get_sparsity_key()
        final_metrics = {}
        for key, value in metrics.items():
            if value is None:
                continue
            final_metrics[key] = value
        final_metrics['sparsity'] = sparsity

        self._log_metrics('Final', final_metrics, run)

        return final_metrics
    
    def _retrain(self, run=None):
        """Recover model performance by running additional training epochs."""
        self.recover = True

        for epoch in range(self.recovery_epochs):
            curr_rec_epoch_str = f"Recovery Epoch [{epoch+1}/{self.recovery_epochs}]"

            # Recovery training
            desc = f"Training {curr_rec_epoch_str}"
            loss = self._run_train_epoch(epoch, desc)

            # Recovery validation
            desc = f"Validation {curr_rec_epoch_str}"
            metrics = self._run_validation_epoch(desc)

            # Appends training loss and validation accuracy to lists
            self._update_metric_lists(loss, metrics.get('accuracy'))   

            # Logs these metrics to the logger or to W&B
            metrics['loss'] = loss
            self._log_metrics(curr_rec_epoch_str, metrics, run) 
            
            # Saving model
            self._handle_save(epoch)
        self.recover = False

    def _run_train_epoch(self, epoch, desc=""):
        """Run a training epoch."""
        self._handle_wanda_hooks()

        self.model.train()
        total_loss = 0
        batchloader = tqdm(self.trainloader, desc=desc, leave=False) if self.enable_tqdm else self.trainloader

        for step, batch in enumerate(batchloader):
            # Unpacking batch and moving to device
            data, labels = self._handle_data_to_device(batch)

            with self.context:
                outputs = self.model(data)
                if hasattr(outputs, 'loss') and outputs.loss:
                    loss = outputs.loss
                elif hasattr(outputs, 'logits'):
                    loss = self.criterion(outputs.logits, labels)
                else:
                    loss = self.criterion(outputs, labels)
            total_loss += loss.item()

            # Handling backprop, optimization, and pruning
            _handle_optimizer_and_pruning(self, loss, epoch, step)

            running_loss = total_loss / (step + 1)
            self._handle_tqdm_logs(batchloader, {'loss': running_loss})

        avg_loss = total_loss / len(self.trainloader)    
        return avg_loss

    def _run_validation_epoch(self, desc="", mode="val"):
        """Run a validation epoch."""
        self.model.eval()
        val_loss, val_acc, val_perp = 0, 0, 0

        dataloader = self.testloader if (mode=='eval' and self.testloader) else self.valloader
        batchloader = tqdm(dataloader, desc=desc, leave=False) if self.enable_tqdm else dataloader

        with torch.no_grad():
            for step, batch in enumerate(batchloader):
                # Unpacking batch and moving to device
                data, labels = self._handle_data_to_device(batch)

                with self.context:
                    outputs = self.model(data)

                metrics = self._handle_metrics(outputs, labels)
                val_loss += metrics.get('batch_val_loss', 0)
                val_acc += metrics.get('batch_accuracy', 0)
                val_perp += metrics.get('batch_perplexity', 0)

                self._handle_tqdm_logs(batchloader, metrics)

        avg_loss = val_loss / len(dataloader)
        avg_accuracy = val_acc / len(dataloader)
        avg_perplexity = val_perp / len(dataloader)
        return {
            'val_loss': avg_loss if avg_loss > 0.0 else None,
            'accuracy': avg_accuracy if avg_accuracy > 0.0 else None,
            'perplexity': avg_perplexity if avg_perplexity > 1.0 else None
        }

    def _get_sparsity_key(self):
        """Get current model sparsity as a rounded key."""
        return round(self.current_sparsity, 4)
    
    def _handle_tqdm_logs(self, batchloader, metrics):
        if self.enable_tqdm:
            display_metrics = {}
            for key, value in metrics.items():
                if value is None:
                    continue
                display_metrics[key] = f"{value:.4f}"
            batchloader.set_postfix(**display_metrics)
        else:
            return

    def _handle_wanda_hooks(self):
        if self.prune and hasattr(self.pruner, 'is_wanda') and self.pruner.is_wanda:
            if not self.recover:    # We don't want to register hooks during recovery
                self.pruner.register_hooks(self.model)
            else:
                pass
    
    def _handle_data_to_device(self, data):
        if isinstance(data, list):
            data = [x.to(self.device) for x in data]
            data, labels = data
        elif isinstance(data, dict):
            data = {k: v.to(self.device) for k, v in data.items()}
            labels = data.get('labels', None)
        else:
            raise ValueError(f"Data type {type(data)} not supported")
        return data, labels

    def _handle_metrics(self, outputs, labels):
        current_correct = 0
        current_samples = 0
        current_loss = 0

        if self.model_type == 'llm':
            if self.dataset_name == 'wikitext2':
                mask = (labels != -100)

                logits = outputs.logits[mask]                        
                masked_labels = labels[mask]
                preds = torch.argmax(logits, dim=-1)

                current_correct = (preds == masked_labels).sum().item()
                current_samples = mask.sum().item()
                current_loss = outputs.loss.item()
            else:
                preds = torch.argmax(outputs.logits, dim=1)
                current_correct = (preds == labels).sum().item()
                current_samples = labels.size(0)
                
        else:
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            preds = logits.max(1)[1]
            current_correct = (preds == labels).sum().item()
            current_samples = labels.size(0)

        avg_loss = current_loss / current_samples
        avg_accuracy = 100 * current_correct / current_samples
        avg_perplexity = torch.exp(torch.tensor(avg_loss)).item()
        return {
            'batch_val_loss': avg_loss,
            'batch_accuracy': avg_accuracy,
            'batch_perplexity': avg_perplexity,
        }

    def _initialize_logs(self):
        if self.logger is not None:
            self.logger.create_log()
            self.logger.log_hyperparameters(self.logger_params)
        else:
            pass
    
    def _log_metrics(self, info, metrics, run=None):
        metrics['sparsity'] = self._get_sparsity_key()
        # Creating information string
        for k, v in metrics.items():
            if v is None:
                continue
            info += f" - {k}: {v:.4f}"
        print(info)

        # Logging to wandb or logger
        if run: 
            run.log(metrics)

        if self.logger is not None:
            self.logger.log_epochs(info)

    def _update_metric_lists(self, loss, accuracy):
        self.train_losses.append(loss)
        if self.target_sparsity is not None:
            sparsity_key = self._get_sparsity_key()
            self.accuracies.setdefault(sparsity_key, []).append(accuracy)
        else:
            self.accuracies.append(accuracy)

    def _handle_save(self, epoch):
        """Saves the model or stops training if no improvements are seen."""
        if self._save_model(epoch):
                print(f"[TRAINER] weights saved!")
                self.unchanged = 0
        else:
            self.unchanged += 1
            if self.unchanged >= self.patience:
                print(f"[TRAINER] Training stopped. No improvements for {self.unchanged} epochs.")
                return False
        return True

    def _save_model(self, epoch):
        """Save the model based on the validation accuracy."""
        if self.save_path:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        if self.prune and self.pruner:
            sparsity_key = self._get_sparsity_key()
            accuracy_list = self.accuracies[sparsity_key]

            if len(accuracy_list) <= 1:
                torch.save(self.model.state_dict(), self.save_path)
                return True
            elif len(accuracy_list) > 1:
                if accuracy_list[-1] > max(accuracy_list[:-1]):
                    torch.save(self.model.state_dict(), self.save_path)
                    return True
            return False
        else:
            if len(self.accuracies) <= 1:
                torch.save(self.model.state_dict(), self.save_path)
                return True
            elif len(self.accuracies) > 1:
                if self.accuracies[-1] > max(self.accuracies[:-1]):
                    torch.save(self.model.state_dict(), self.save_path)
                    return True
            return False
        

















        