import os
import torch
import torch.nn as nn
import contextlib
from tqdm import tqdm
from torch.amp import GradScaler, autocast
from dataclasses import dataclass
from training_utils import (
    _initialize_models,
    _initialize_data_loaders,
    _initialize_optimizer,
    _initialize_scheduler,
    _initialize_pruner,
    _initialize_paths_and_logger,
    _initialize_log_parameters,
    _initialize_dyrelu_phasing,
    _initialize_logs,
    
    _optimizer_step,
    _epoch_pruning_step,
    _step_pruning_step,
    _handle_wanda_calibration,
    
    _handle_data_to_device,
    _handle_tqdm_logs,
    _log_metrics,
    _get_sparsity_key,
)
from dyrelu_adapter import step_dyrelu_adapter
from pruning_factory import check_model_sparsity
from utils import load_weights

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
    image_size:             int = 32
    epochs:                 int = 5
    scheduler_type:         str = None
    patience:               int = None
    trained_weights:        str = None
    experiment_type:        str = ""
    log_epochs:             bool = False
    enable_tqdm:            bool = False
    enable_mixed_precision: bool = True
    databricks_env:         bool = True
    num_workers:            int = os.cpu_count()

    # Pruning arguments
    pruning_type:           str = None
    target_sparsity:        float = None
    sparsity_scheduler:     str = None
    recovery_epochs:        int = 0
    pruning_module:         object = None
    delta_T:                int = 100

    # DyReLU / EAST
    dyrelu_en:              bool = False
    dyrelu_phasing_en:      bool = False
    weight_sharing_en:      bool = False

    def __post_init__(self):
        self.scaler = GradScaler() if self.enable_mixed_precision else None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.retrain = (self.recovery_epochs > 0)
        self.is_bacp = False

        _initialize_models(self)
        _initialize_data_loaders(self)
        _initialize_optimizer(self)
        _initialize_scheduler(self)

        _initialize_pruner(self)

        _initialize_paths_and_logger(self)
        _initialize_log_parameters(self)
        _initialize_dyrelu_phasing(self)

class Trainer:
    """
    Unified trainer for CV/LLM models supporting:
    - Mixed Precision (AMP)
    - Sparse Training (RigL, Wanda, Magnitude)
    - Weight Sharing (EAST)
    - DyReLU Phasing
    """
    def __init__(self, training_args):
        for key, value in vars(training_args).items():
            setattr(self, key, value)

        # State tracking
        self.recover = False
        self.unchanged = 0
        self.train_losses = []
        self.val_accuracies = []
        self.accuracies = {} if self.target_sparsity is not None else []
        self.current_sparsity = check_model_sparsity(self.model)
        self.context = autocast(device_type=self.device) if self.enable_mixed_precision else contextlib.nullcontext()

    def train(self, run=None):
        """Main training workflow."""
        _initialize_logs(self)

        for epoch in range(self.epochs):
            curr_epoch_str = f"Epoch [{epoch+1}/{self.epochs}]"

            # Training
            loss = self._run_train_epoch(epoch, f"Training {curr_epoch_str}")

            # Pruning Step
            # _epoch_pruning_step(self, epoch)

            # Validation
            metrics = self._run_validation_epoch(f"Validation {curr_epoch_str}")

            # Logging & Metrics
            self._update_metric_lists(loss, metrics.get('accuracy'))
            metrics['loss'] = loss
            _log_metrics(self, curr_epoch_str, metrics, run)

            # Checkpoint
            if not self._handle_save(epoch):
                break
            
            # Recovery Phase
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

        sparsity = _get_sparsity_key(self)
        final_metrics = {}
        for key, value in metrics.items():
            if value is None:
                continue
            final_metrics[key] = value
        final_metrics['sparsity'] = sparsity

        _log_metrics(self, 'Final', final_metrics, run) 

        return final_metrics
    
    def _retrain(self, run=None):
        """Recovery phase: Trained without changing the masks"""
        self.recover = True
        print(f"[TRAINER] Starting Recovery for {self.recovery_epochs} epochs...")

        for epoch in range(self.recovery_epochs):
            curr_str = f"Recovery Epoch [{epoch+1}/{self.recovery_epochs}]"

            loss = self._run_train_epoch(epoch, f"Training {curr_str}")
            metrics = self._run_validation_epoch(f"Validation {curr_str}")

            self._update_metric_lists(loss, metrics.get('accuracy'))
            metrics['loss'] = loss
            _log_metrics(self, curr_str, metrics, run)
            
            self._handle_save(epoch)

        self.recover = False


    def _run_train_epoch(self, epoch, desc=""):
        """Run a training epoch."""
        # _handle_wanda_hooks(self)
        # _handle_wanda_calibration(self)

        self.model.train()
        total_loss = 0
    
        steps_per_epoch = len(self.trainloader)
        batchloader = tqdm(self.trainloader, desc=desc, leave=False) if self.enable_tqdm else self.trainloader

        for step, batch in enumerate(batchloader):
            # Unpacking batch and moving to device
            data, labels = _handle_data_to_device(self, batch)

            with self.context:
                outputs = self.model(data)
                if hasattr(outputs, 'loss') and outputs.loss:
                    loss = outputs.loss
                elif hasattr(outputs, 'logits'):
                    loss = self.criterion(outputs.logits, labels)
                else:
                    loss = self.criterion(outputs, labels)

            # Optimizer + pruning step
            global_step = epoch * steps_per_epoch + step
            _optimizer_step(self, loss, global_step)

            total_loss += loss.item()
            running_loss = total_loss / (step + 1)
            _handle_tqdm_logs(self, batchloader, {'loss': running_loss})

        # DyReLU Phasing Step (End of Epoch)
        if self.dyrelu_phasing_en:
            step_dyrelu_adapter(self.model)
        return total_loss / len(self.trainloader)


    def _run_validation_epoch(self, desc="", mode="val"):
        """Run a validation epoch."""
        self.model.eval()
        val_loss, val_acc, val_perp = 0, 0, 0

        dataloader = self.testloader if (mode == 'eval' and self.testloader) else self.valloader
        if not dataloader: return {}

        batchloader = tqdm(dataloader, desc=desc, leave=False) if self.enable_tqdm else dataloader

        with torch.no_grad():
            for step, batch in enumerate(batchloader):
                # Unpacking batch and moving to device
                data, labels = _handle_data_to_device(self, batch)

                with self.context:
                    outputs = self.model(data)

                metrics = self._handle_metrics(outputs, labels)
                val_loss += metrics.get('batch_val_loss', 0)
                val_acc += metrics.get('batch_accuracy', 0)
                val_perp += metrics.get('batch_perplexity', 0)

                _handle_tqdm_logs(self, batchloader, metrics)

        avg_loss = val_loss / len(dataloader)
        avg_accuracy = val_acc / len(dataloader)
        avg_perplexity = val_perp / len(dataloader)
        return {
            'val_loss': avg_loss if avg_loss > 0.0 else None,
            'accuracy': avg_accuracy if avg_accuracy > 0.0 else None,
            'perplexity': avg_perplexity if avg_perplexity > 1.0 else None
        }
    
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

    def _update_metric_lists(self, loss, accuracy):
        self.train_losses.append(loss)
        if self.target_sparsity is not None:
            sparsity_key = _get_sparsity_key(self)
            self.accuracies.setdefault(sparsity_key, []).append(accuracy)
        else:
            self.accuracies.append(accuracy)

    def _handle_save(self, epoch):
        """Determines if model should be saved based on accuracy improvement."""
        if self.save_path:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        # Get relevant accuracy history
        if isinstance(self.accuracies, dict):
            key = _get_sparsity_key(self)
            hist = self.accuracies.get(key, [])
        else:
            hist = self.accuracies

        # Logic: Save if first epoch OR if improved over previous best
        improved = False
        if len(hist) <= 1:
            improved = True
        elif len(hist) > 1 and hist[-1] > max(hist[:-1]):
            improved = True

        if improved:
            torch.save(self.model.state_dict(), self.save_path)
            print("[TRAINER] Checkpoint saved.")
            self.unchanged = 0
            return True
        else:
            self.unchanged += 1
            if self.patience and self.unchanged >= self.patience:
                print(f"[TRAINER] Early stopping triggered (Patience: {self.patience})")
                return False
            return True
        
