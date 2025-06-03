from logger import Logger
from utils import get_device, load_weights
from constants import *
from transformers import AutoTokenizer

import os
import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from datasets.utils.logging import disable_progress_bar
from unstructured_pruning import check_model_sparsity, PRUNER_DICT
from models import ClassificationNetwork
from dataset_utils import get_glue_data, get_squad_data, get_cv_data, CV_DATASETS
from datasets import get_dataset_config_names
from torch.nn import CrossEntropyLoss

disable_progress_bar()
os.environ["HF_DATASETS_CACHE"] = "/dbfs/hf_datasets"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class CVTrainingArguments:
    def __init__(self, 
                 model_name, 
                 model_task,
                 batch_size, 
                 image_size=32,
                 pruning_type=None,
                 target_sparsity=0,
                 sparsity_scheduler="linear",
                 finetuned_weights=None,
                 num_classes=10,
                 criterion=CrossEntropyLoss(),
                 learning_rate=2e-5, 
                 optimizer_type='adamw',
                 learning_type="",
                 epochs=5, 
                 pruning_epochs=None,
                 recovery_epochs=10,
                 log_epochs=True,
                 enable_tqdm=True,
                 enable_mixed_precision=True,
                 num_workers=24,
                 pruner=None,
                 finetune=False,
                 delta_t=500,
                 db=True):
        
        # Base model configuration
        self.model_type = 'cv'
        self.model_name = model_name
        self.model_task = model_task
        self.finetuned_weights = finetuned_weights
        self.num_classes = num_classes
        self.criterion = criterion

        # Training parameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.epochs = epochs
        self.pruning_epochs = epochs if pruning_epochs is None else pruning_epochs
        self.recovery_epochs = recovery_epochs
        self.recover = False
        self.finetune = finetune

        # Technical settings
        self.enable_tqdm = enable_tqdm
        self.enable_mixed_precision = enable_mixed_precision
        self.device = get_device()
        self.scaler = GradScaler() if self.enable_mixed_precision else None

        # Initializing model, tokenizer, and other components
        self._initialize_model(model_name, finetuned_weights, num_classes) 

        # Initializing dataloaders
        self._initialize_data_loaders(model_task, batch_size, image_size, num_workers)

        # Initializing pruner
        self._initialize_pruner(pruning_type, target_sparsity, sparsity_scheduler, delta_t, pruner)   

        # Initializing paths and logger
        self._initialize_paths_and_logger(db, learning_type, log_epochs)

    def _initialize_model(self, model_name, finetuned_weights, num_classes):
        """Initialize the models required for training."""
        print("[TRAINER] Initializing model")
        self.model = ClassificationNetwork(model_name, num_classes=num_classes)
        self.embedded_dim = self.model.embedding_dim
        if finetuned_weights:
            loaded = load_weights(self.model, finetuned_weights)
            print("[TRAINER] Weights loaded" if loaded else "[TRAINER] Failed to load weights")

        # Initializing optimizer
        if self.optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)
        else:
            raise ValueError(f"Invalid optimizer type: {self.optimizer_type}")

    def _initialize_data_loaders(self, model_task, batch_size, image_size, num_workers):
        """Initialize data loaders for the specified task."""
        print(f"[TRAINER] Initializing data loaders for {model_task}")
        if model_task in CV_DATASETS:
            data = get_cv_data(model_task, batch_size, image_size, num_workers)
            if len(data) >= 2:
                self.trainloader = data["trainloader"]
                self.valloader = data["valloader"]
                self.testloader = data.get("testloader", None)
            else:
                raise ValueError(f"Expected at least trainloader and valloader for {model_task}, got {len(data)} loaders")
        else:
            raise ValueError(f"{model_task} dot not exist in cv models. Existing datasets are: {CV_DATASETS}")
    
    def _initialize_pruner(self, pruning_type, target_sparsity, sparsity_scheduler, delta_t, pruner):
        """Initialize the pruner based on the pruning type."""
        self.pruning_type = pruning_type
        self.target_sparsity = target_sparsity
        self.sparsity_scheduler = sparsity_scheduler
        self.delta_t = int(min(0.5 * len(self.trainloader), delta_t))
        self.prune = pruning_type is not None and target_sparsity > 0 and not self.finetune

        if self.prune:
            if "wanda" in pruning_type:
                self.pruner = PRUNER_DICT[pruning_type](
                    self.pruning_epochs, 
                    target_sparsity, 
                    self.model, 
                    self.sparsity_scheduler
                )
            else:
                self.pruner = PRUNER_DICT[pruning_type](
                    self.pruning_epochs, 
                    target_sparsity, 
                    self.sparsity_scheduler
                )
            print("[TRAINER] Pruning enabled")
            print(f"[TRAINER] Initializing pruner for {pruning_type}")
            print(f"[TRAINER] Pruning scheduler: {self.sparsity_scheduler}")
            print(f"[TRAINER] Pruning target sparsity: {target_sparsity}")
        else:
            self.pruner = pruner
            self.pruning_epochs = 0
            self.recovery_epochs = 0
            print("[TRAINER] Pruning disabled")
        
        self.current_sparsity = check_model_sparsity(self.model)
        print(f"[TRAINER] Current model sparsity: {self.current_sparsity}")

    def _initialize_paths_and_logger(self, db, learning_type, log_epochs):
        """Initialize weight paths and logger."""
        self.learning_type = learning_type
        base_dir = "/dbfs" if db else "."

        if self.prune:
            self.save_path = f"{base_dir}/research/{self.model_name}/{self.model_task}/{self.model_name}_{self.pruning_type}_{self.target_sparsity}.pt"
        else:
            if self.finetune and self.pruning_type is not None:
                weights_path = f"{self.model_name}_{learning_type}_{self.pruning_type}_{self.current_sparsity:.2f}"
            else:
                weights_path = f"{self.model_name}_{learning_type}"
                
            self.save_path = f"{base_dir}/research/{self.model_name}/{self.model_task}/{weights_path}.pt"

        self.logger = Logger(self.model_name, learning_type) if log_epochs else None
        print(f"[TRAINER] Saving model checkpoints to {self.save_path}")

class Trainer:
    """
    Trainer class for training, fine-tuning, and pruning an CV classification models.
    """
    def __init__(self, training_args):
        for key, value in vars(training_args).items():
            setattr(self, key, value)

        self.recover = False
        self.train_losses = []
        self.val_accuracies = []

        self._initialize_log_parameters()
        
    def _initialize_log_parameters(self):
        """Initialize parameters for logging purposes."""
        logger_params = {
            # Model information
            'model_type': getattr(self, 'model_type', None),
            'model_name': getattr(self, 'model_name', None),
            'model_task': getattr(self, 'model_task', None),
            'num_classes': getattr(self, 'num_classes', None),
            'criterion': getattr(self, 'criterion', None),
            'embedding_dim': getattr(self, 'embedded_dim', None),
            
            # Training parameters
            'epochs': getattr(self, 'epochs', None),
            'pruning_epochs': getattr(self, 'pruning_epochs', None),
            'recovery_epochs': getattr(self, 'recovery_epochs', None),
            'batch_size': getattr(self, 'batch_size', None),
            'learning_rate': getattr(self, 'learning_rate', None),
            'learning_type': getattr(self, 'learning_type', None),
            
            # Pruning parameters
            'prune': getattr(self, 'prune', False),
            'pruning_type': getattr(self, 'pruning_type', None),
            'target_sparsity': getattr(self, 'target_sparsity', None),
            'sparsity_scheduler': getattr(self, 'sparsity_scheduler', None),
            'delta_t': getattr(self, 'delta_t', None),
            
            # Technical settings
            'enable_mixed_precision': getattr(self, 'enable_mixed_precision', None),
            'device': str(getattr(self, 'device', None)) if getattr(self, 'device', None) is not None else None,
            
            # Paths
            'save_path': getattr(self, 'save_path', None),
            'finetuned_weights': getattr(self, 'finetuned_weights', None),
            
            # Current model state
            'current_sparsity': getattr(self, 'current_sparsity', None),
        }
        self.logger_params = {k: v for k, v in logger_params.items() if v is not None}

    def train(self):
        """
        Main training loop that handles training, pruning, retraining, and logging.
        """
        self.model.to(self.device)

        # Initializing Logger
        if self.logger is not None:
            self.logger.create_log()
            self.logger.log_hyperparameters(self.logger_params)
                
        if self.enable_mixed_precision:
            print("[TRAINER] Training with mixed precision enabled")
        print(f"[TRAINER] Initial model sparsity: {check_model_sparsity(self.model):.2f}")

        for epoch in range(self.epochs):
            if self.prune and self.pruner:
                self.val_accuracies = []

            # Training
            self.model.train()
            self.total_loss = 0.0
            desc = f"Training Epoch [{epoch+1}/{self.epochs}]"
            self.run_train_epoch(desc)

            # Validation
            self.model.eval()
            self.total_correct, self.total_samples = 0, 0
            desc = f"Validation Epoch [{epoch+1}/{self.epochs}]"
            self.run_validation_epoch(desc)

            avg_loss = self.total_loss / len(self.trainloader)
            avg_acc = (self.total_correct / self.total_samples) * 100

            self.train_losses.append(avg_loss)
            self.val_accuracies.append(avg_acc)

            info = (
                f"Epoch [{epoch+1}/{self.epochs}]: Avg Loss: {avg_loss:.4f} | "
                f"Avg Accuracy: {avg_acc:.2f} | Model Sparsity: {check_model_sparsity(self.model):.2f}\n"
            )
            print(info)
            
            if self.logger is not None:
                self.logger.log_epochs(info)
            
            if self.handle_save(epoch):
                print(f"[TRAINER] weights saved!")

            # Pruning and recovery schedule
            if self.prune and self.pruner and epoch < self.pruning_epochs:
                self.fine_tune()
                self.pruner.ratio_step()

    def fine_tune(self):
        """
        Recovery finetuning after pruning
        """
        self.recover = True
        for epoch in range(self.recovery_epochs):
            self.model.train()
            self.total_loss = 0.0
            desc = f"Recovery Epoch [{epoch+1}/{self.recovery_epochs}]"
            self.run_train_epoch(desc)

            self.model.eval()
            self.total_correct, self.total_samples = 0, 0
            desc = f"Validation Epoch [{epoch+1}/{self.recovery_epochs}]"
            self.run_validation_epoch(desc)

            avg_loss = self.total_loss / len(self.trainloader)
            avg_acc = (self.total_correct / self.total_samples) * 100

            self.train_losses.append(avg_loss)
            self.val_accuracies.append(avg_acc)

            info = (
                f"Recovery epoch [{epoch+1}/{self.recovery_epochs}]: Avg Loss: {avg_loss:.4f} | "
                f"Avg Accuracy: {avg_acc:.2f} | Model Sparsity: {check_model_sparsity(self.model):.2f}\n"
            )
            print(info)

            if self.logger is not None:
                self.logger.log_epochs(info)

            if self.handle_save(epoch):
                print(f"[TRAINER] weights saved!")

        self.recover = False

    def evaluate(self):
        """
        Load weights and evaluate the model on validation or test set.
        Returns accuracy in percentage.
        """
        print(f"[TRAINER] Loading weights: {self.save_path}")
        if self.save_path and load_weights(self.model, self.save_path):
            print("[TRAINER] Weights loaded successfully")
        else:
            print("[TRAINER] Failed to load weights")

        print(f"[TRAINER] Model Sparsity: {check_model_sparsity(self.model):.2f}")

        self.model.eval()
        self.model.to(self.device)

        total_correct, total_samples = 0, 0
        batchloader = self.testloader or self.valloader
        if self.enable_tqdm:
            batchloader = tqdm(batchloader, desc="Evaluating")

        with torch.no_grad():
            for batch in batchloader:
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    labels = batch["label"]

                elif isinstance(batch, list):
                    batch = [x.to(self.device) for x in batch]
                    batch, labels = batch

                if self.enable_mixed_precision:
                    with autocast(device_type=self.device):
                        outputs = self.model(batch)
                else:
                    outputs = self.model(batch)

                preds = torch.argmax(outputs.logits, dim=1) if self.model_type == 'llm' else outputs.max(1)[1]
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

                if self.enable_tqdm:
                    running_acc = (total_correct / total_samples) * 100
                    batchloader.set_postfix({"Acc": f"{running_acc:.2f}%"})

        accuracy = round((total_correct / total_samples) * 100, 2)
        return accuracy

    def run_train_epoch(self, desc):
        """
        Run one full training epoch.
        """
        if (self.prune and self.pruner) and (hasattr(self.pruner, 'is_wanda') and self.pruner.is_wanda) and (not self.recover):
            self.pruner.register_hooks(self.model)

        batchloader = tqdm(self.trainloader, desc=desc) if self.enable_tqdm else self.trainloader
        for step, batch in enumerate(batchloader):
            if isinstance(batch, dict):
                batch = {k: v.to(self.device) for k, v in batch.items()}
            elif isinstance(batch, list):
                batch = [x.to(self.device) for x in batch]
                batch, labels = batch

            self.optimizer.zero_grad()

            if self.enable_mixed_precision:
                with autocast(device_type=self.device):
                    outputs = self.model(batch)
                    loss = outputs.loss if self.model_type == 'llm' else self.criterion(outputs, labels)
            else:
                outputs = self.model(batch)
                loss = outputs.loss if self.model_type == 'llm' else self.criterion(outputs, labels)

            self.total_loss += loss.item()

            # Handling backprop, optimization, pruning, and logic
            self.handle_optimizer_and_pruning(loss, step)

            running_loss = self.total_loss / (step + 1)
            if self.enable_tqdm:
                batchloader.set_postfix(Loss=f"{running_loss:.4f}", Sparsity=f"{check_model_sparsity(self.model):.2f}")
            
    def run_validation_epoch(self, desc):
        """
        Run one full validation epoch.
        """
        batchloader = tqdm(self.valloader, desc=desc) if self.enable_tqdm else self.valloader
        with torch.no_grad():
            for step, batch in enumerate(batchloader):
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    labels = batch["label"]

                elif isinstance(batch, list):
                    batch = [x.to(self.device) for x in batch]
                    batch, labels = batch
                
                if self.enable_mixed_precision:
                    with autocast(device_type=self.device):
                        outputs = self.model(batch)
                else:
                    outputs = self.model(batch)

                preds = torch.argmax(outputs.logits, dim=1) if self.model_type == 'llm' else outputs.max(1)[1]
                correct = (preds == labels).sum()
                self.total_correct += correct.item()
                self.total_samples += labels.size(0)

                if self.enable_tqdm:
                    running_acc = (self.total_correct / self.total_samples) * 100
                    sparsity = check_model_sparsity(self.model)
                    batchloader.set_postfix(Accuracy=f"{running_acc:.2f}", Sparsity=f"{sparsity:.2f}")

    def handle_optimizer_and_pruning(self, loss, step):
        """Handle backpropagation, pruning, and weight update in a single step."""
        if self.enable_mixed_precision:
            self.scaler.scale(loss).backward()
            self.apply_pruning(step)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.apply_pruning(step)
            self.optimizer.step()

        if self.finetune or (self.prune and self.pruner is not None):
            self.pruner.apply_mask(self.model)
            
    def apply_pruning(self, step):
        """Apply pruning based on the pruning configuration."""
        if (not self.prune and self.pruner is None) or self.recover or self.finetune:
            return
        
        if self.pruner.sparsity_scheduler == "linear":
            if step == self.delta_t:
                self.pruner.prune(self.model)
            
        elif self.pruner.sparsity_scheduler == "cubic":
            if step >= 0 and step % self.delta_t == 0:
                self.pruner.cubic_scheduler(step, 0, self.delta_t, check_model_sparsity(self.model))
                self.pruner.prune(self.model)

    def handle_save(self, epoch):
        """Save the model based on the validation accuracy"""
        if self.save_path:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        if epoch == 0:
            torch.save(self.model.state_dict(), self.save_path)
            return True
        else:
            if len(self.val_accuracies) > 1 and self.val_accuracies[-1] > max(self.val_accuracies[:-1]):
                torch.save(self.model.state_dict(), self.save_path)
                return True
            else:
                return False
            