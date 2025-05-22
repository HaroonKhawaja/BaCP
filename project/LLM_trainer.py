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
from dataset_utils import get_glue_data
from datasets import get_dataset_config_names

disable_progress_bar()
os.environ["HF_DATASETS_CACHE"] = "/dbfs/hf_datasets"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class LLMTrainingArguments:
    """Configuration class for training and pruning setting for LLM classifier.
        Args:
        model_name (str): Pretrained model name.
        model_task (str): Task/dataset name (e.g., GLUE task - sst2, qqp, etc).
        batch_size (int): Batch size for training.
        pruning_type (str): Type of pruning method to use.
        target_sparsity (float): Desired sparsity level for pruning.
        pruning_scheduler (str): Pruning schedule type.
        finetuned_weights (str or None): Path to pretrained weights to load.
        num_classes (int): Number of output classes.
        learning_rate (float): Learning rate.
        learning_type (str): Identifier for the learning method.
        epochs (int): Total number of training epochs.
        pruning_epochs (int or None): Number of pruning epochs, defaults to total epochs.
        recovery_epochs (int): Number of finetuning epochs after pruning.
        log_epochs (bool): Whether to log training info per epoch.
        enable_tqdm (bool): Whether to use tqdm progress bars.
        enable_mixed_precision (bool): Use mixed precision training.
        num_workers (int): DataLoader workers count.
        prune (bool): Enable pruning.
        pruner (object or None): External pruner instance to use.
        finetune (bool): Flag for finetuning mode.
        use_testloader (bool): Evaluate on test loader instead of validation loader.
        delta_t (int): Steps interval for pruning schedule.
        db (bool): Use DBFS directory for saving weights if True.
    """
    def __init__(self, 
                 model_name, 
                 model_task,
                 batch_size, 
                 pruning_type=None,
                 target_sparsity=0,
                 pruning_scheduler="linear",
                 finetuned_weights=None,
                 num_classes=2,
                 learning_rate=2e-5, 
                 learning_type="",
                 epochs=10, 
                 pruning_epochs=None,
                 recovery_epochs=5,
                 log_epochs=True,
                 enable_tqdm=True,
                 enable_mixed_precision=True,
                 num_workers=8,
                 prune=True,
                 pruner=None,
                 finetune=False,
                 use_testloader=False,
                 delta_t=500,
                 db=True):
        self.model_name = model_name
        self.model_task = model_task
        self.num_classes = num_classes
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_type = learning_type
        self.epochs = epochs
        self.pruning_epochs = pruning_epochs or epochs
        self.recovery_epochs = recovery_epochs
        self.log_epochs = log_epochs
        self.enable_tqdm = enable_tqdm
        self.enable_mixed_precision = enable_mixed_precision
        self.num_workers = num_workers
        
        self.prune = prune
        self.finetune = finetune
        self.pruning_type = pruning_type
        self.pruning_scheduler = pruning_scheduler
        self.target_sparsity = target_sparsity
        self.delta_t = delta_t
        
        self.use_testloader = use_testloader
        self.db = db

        self.device = get_device()

        # Initializing model
        self.model = ClassificationNetwork(model_name, num_classes=num_classes)
        self.embedded_dim = self.model.embedding_dim
        
        # Loading pretrained weights if provided
        if finetuned_weights:
            loaded = load_weights(self.model, finetuned_weights)
            print("[LLM TRAINER] Weights loaded" if loaded else "[LLM TRAINER] Failed to load weights")
        
        # Initialize pruner if pruning enabled
        if prune:
            print(self.pruning_type)
            if "wanda" in pruning_type:
                self.pruner = PRUNER_DICT[pruning_type](self.pruning_epochs, target_sparsity, self.model, self.pruning_scheduler)
            elif self.pruning_type == "movement_pruning":
                self.pruner = PRUNER_DICT[pruning_type](
                self.model,            
                self.pruning_epochs,    
                target_sparsity,        
                self.pruning_scheduler  
        ) 
            else:
                self.pruner = PRUNER_DICT[pruning_type](self.pruning_epochs, target_sparsity, self.pruning_scheduler)
            print("[LLM TRAINER] Pruning enabled")
            print(f"[LLM TRAINER] Pruning scheduler: {self.pruning_scheduler}")

        else:
            self.pruner = pruner
            print("[LLM TRAINER] Pruning disabled")


        print(f"[LLM TRAINER] Pruning type: {self.pruning_type}")
        print(f"[LLM TRAINER] Model Sparsity: {check_model_sparsity(self.model):.2f}")
            
        # Load tokenizer and datasets
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model_task in get_dataset_config_names("glue"):
            data = get_glue_data(model_name, self.tokenizer, model_task, batch_size, num_workers)
            if len(data) >= 2:
                self.trainloader = data["trainloader"]
                self.valloader = data["valloader"]
                self.testloader = data.get("testloader", None)
            else:
                raise ValueError(f"Expected at least trainloader and valloader for {model_task}, got {len(data)} loaders")

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        base_dir = "/dbfs" if db else "."
        if prune:
            self.save_path = f"{base_dir}/research/{model_name}/{model_task}/{model_name}_{pruning_type}_{target_sparsity}.pt"
        else:
            self.save_path = f"{base_dir}/research/{model_name}/{model_task}/{model_name}_{learning_type}.pt"
        print(f"[LLM TRAINER] Saving model at {self.save_path}")

        self.logger = Logger(model_name, learning_type) if self.log_epochs else None
        self.scaler = GradScaler() if self.enable_mixed_precision else None
    
class LLMTrainer:
    """
    Trainer class for training, fine-tuning, and pruning an LLM classification models.
    """
    def __init__(self, training_args):
        for key, value in vars(training_args).items():
            setattr(self, key, value)

        self.recover = False
        self.train_losses = []
        self.val_accuracies = []

    def train(self):
        """
        Main training loop that handles training, pruning, retraining, and logging.
        """
        self.model.to(self.device)

        # Initializing Logger
        if self.log_epochs:
            self.logger.create_log()
            self.logger.log_hyperparameters(self.get_hyperparameters())
                
        if self.enable_mixed_precision:
            print("[LLM TRAINER] Training with mixed precision enabled")
        print(f"[LLM TRAINER] Initial model sparsity: {check_model_sparsity(self.model):.2f}")

        for epoch in range(self.epochs):
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
                print(f"[LLM TRAINER] weights saved!")

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

        self.recover = False

    def evaluate(self):
        """
        Load weights and evaluate the model on validation or test set.
        Returns accuracy in percentage.
        """
        print(f"[LLM TRAINER] Loading weights: {self.save_path}")
        if self.save_path and load_weights(self.model, self.save_path):
            print("[LLM TRAINER] Weights loaded successfully")
        else:
            print("[LLM TRAINER] Failed to load weights")

        print(f"[LLM TRAINER] Model Sparsity: {check_model_sparsity(self.model):.2f}")

        self.model.eval()
        self.model.to(self.device)

        total_correct, total_samples = 0, 0
        batchloader = self.testloader if self.use_testloader and self.testloader else self.valloader
        if self.enable_tqdm:
            batchloader = tqdm(batchloader, desc="Evaluating")

        with torch.no_grad():
            for batch in batchloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch["label"]

                if self.enable_mixed_precision:
                    with autocast(device_type=self.device):
                        outputs = self.model(batch)
                else:
                    outputs = self.model(batch)

                preds = torch.argmax(outputs.logits, dim=1)
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
        if self.prune and self.pruner and self.pruner.is_wanda and not self.recover:
            self.pruner.register_hooks(self.model)

        batchloader = tqdm(self.trainloader, desc=desc) if self.enable_tqdm else self.trainloader
        for step, batch in enumerate(batchloader):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            self.optimizer.zero_grad()

            if self.enable_mixed_precision:
                with autocast(device_type=self.device):
                    outputs = self.model(batch)
                    loss = outputs.loss
            else:
                outputs = self.model(batch)
                loss = outputs.loss

            self.total_loss += loss.item()

            # Handling backprop, optimization, pruning, and logic
            self.handle_optimizer_and_pruning(loss, step)

            running_loss = self.total_loss / (step + 1)
            if self.enable_tqdm:
                batchloader.set_postfix(Loss=f"{running_loss:.4f}", Sparsity=f"{check_model_sparsity(self.model):.2f}")

    def run_validation_epoch(self, desc):
        """
        Run one full validationg epoch.
        """
        batchloader = tqdm(self.valloader, desc=desc) if self.enable_tqdm else self.valloader
        with torch.no_grad():
            for batch in batchloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch["label"]
                
                if self.enable_mixed_precision:
                    with autocast(device_type=self.device):
                        outputs = self.model(batch)
                else:
                    outputs = self.model(batch)

                preds = torch.argmax(outputs.logits, dim=1)
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
        if not self.prune and self.pruner is None or self.recover:
            return
        
        if self.pruner.sparsity_scheduler == "linear":
            if step == self.delta_t:
                self.pruner.prune(self.model)
            
        elif self.pruner.sparsity_scheduler == "cubic":
            if step >= 0 and step % self.delta_t == 0:
                current_sparsity = check_model_sparsity(self.model)
                self.pruner.cubic_scheduler(step, 0, self.delta_t, current_sparsity)
                self.pruner.prune(self.model)

    def handle_save(self, epoch):
        """Save the model based on the validation accuracy"""
        if self.save_path:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        if epoch == 0 or self.prune:
            torch.save(self.model.state_dict(), self.save_path)
            return True
        else:
            if self.val_accuracies[-1] > max(self.val_accuracies[:-1]):
                torch.save(self.model.state_dict(), self.save_path)
                return True
            else:
                return False
            
    def get_hyperparameters(self):
        """
        Get all hyperparameters for logging.
        Returns a dictionary with all important hyperparameters.
        """
        return {
            'model_name': self.model_name,
            'model_task': self.model_task,
            'num_classes': self.num_classes,
            'embedding_dim': self.embedded_dim,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'learning_type': self.learning_type,
            'epochs': self.epochs,
            'optimizer': str(self.optimizer),
            'enable_mixed_precision': self.enable_mixed_precision,
            'prune': self.prune,
            'pruning_type': self.pruning_type if self.prune else "None",
            'pruning_scheduler': self.pruning_scheduler,
            'target_sparsity': self.target_sparsity,
            'pruning_epochs': self.pruning_epochs,
            'recovery_epochs': self.recovery_epochs,
            'delta_t': self.delta_t,
            'finetune': self.finetune,
            'save_path': self.save_path,
            'initial_sparsity': check_model_sparsity(self.model),
            'device': str(self.device)
        }
