import re
import os
import string
import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import evaluate
from datasets.utils.logging import disable_progress_bar
from unstructured_pruning import check_model_sparsity
from logger import Logger
import contextlib
from constants import *
from utils import *
from training_utils import (
    _detect_model_type, _detect_num_classes, _detect_cv_image_size,
    _initialize_models, _initialize_optimizer, _initialize_scheduler,
    _initialize_data_loaders, _initialize_pruner, _detect_criterion,
    _initialize_paths_and_logger, _handle_optimizer_and_pruning,
)

disable_progress_bar()
os.environ["HF_DATASETS_CACHE"] = "/dbfs/hf_datasets"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class TrainingArguments:
    def __init__(self, 
                model_name, 
                model_task,
                batch_size, 
                optimizer_type,
                learning_rate, 

                # Pruning parameters
                pruner=None,
                pruning_type=None,
                target_sparsity=0.0,
                sparsity_scheduler="linear",
                pruning_epochs=None,
    
                # Training parameters
                criterion_type='supervised',
                epochs=5, 
                recovery_epochs=10,
                scheduler_type=None,
                patience=20,
                finetuned_weights=None,
                finetune=False,
                learning_type="",

                # Extra parameters
                log_epochs=True,
                enable_tqdm=True,
                enable_mixed_precision=True,
                db=True,
                num_workers=24):

        self.model_name = model_name
        self.model_task = model_task
        self.batch_size = batch_size
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.is_bacp = False

        # Pruning parameters
        self.pruner = pruner
        self.pruning_type = pruning_type
        self.target_sparsity = target_sparsity
        self.sparsity_scheduler = sparsity_scheduler
        self.pruning_epochs = epochs or pruning_epochs

        # Training parameters
        self.criterion_type = criterion_type
        self.epochs = epochs
        self.recovery_epochs = recovery_epochs
        self.scheduler_type = scheduler_type
        self.patience = patience
        self.finetuned_weights = finetuned_weights
        self.finetune = finetune
        self.learning_type = learning_type

        # Extra parameters
        self.log_epochs = log_epochs
        self.enable_tqdm = enable_tqdm
        self.enable_mixed_precision = enable_mixed_precision
        self.db = db
        self.num_workers = num_workers
        self.scaler = GradScaler() if self.enable_mixed_precision else None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        _detect_model_type(self)
        _detect_num_classes(self)
        _detect_criterion(self)
        _detect_cv_image_size(self)
        _initialize_models(self)
        _initialize_optimizer(self)
        _initialize_data_loaders(self)
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
        self.accuracy_per_sparsity = {}
        self.accuracies = []

        self._initialize_log_parameters()
    
    def _get_sparsity_key(self):
        """Get current model sparsity as a rounded key."""
        return round(check_model_sparsity(self.model), 4)
    
    def _initialize_log_parameters(self):
        """Initialize parameters for logging purposes."""
        logger_params = {
            'model_type': getattr(self, 'model_type', None),
            'model_name': getattr(self, 'model_name', None),
            'model_task': getattr(self, 'model_task', None),
            'num_classes': getattr(self, 'num_classes', None),
            'criterion': getattr(self, 'criterion', None),
            'embedding_dim': getattr(self, 'embedded_dim', None),
            'epochs': getattr(self, 'epochs', None),
            'pruning_epochs': getattr(self, 'pruning_epochs', None),
            'recovery_epochs': getattr(self, 'recovery_epochs', None),
            'batch_size': getattr(self, 'batch_size', None),
            'learning_rate': getattr(self, 'learning_rate', None),
            'learning_type': getattr(self, 'learning_type', None),
            'scheduler_type': getattr(self, 'scheduler_type', None),
            'optimizer_type': getattr(self, 'optimizer_type', None),
            'total_steps': getattr(self, 'total_steps', None),
            'warmup_steps': getattr(self, 'warmup_steps', None),
            'prune': getattr(self, 'prune', False),
            'pruning_type': getattr(self, 'pruning_type', None),
            'target_sparsity': getattr(self, 'target_sparsity', None),
            'sparsity_scheduler': getattr(self, 'sparsity_scheduler', None),
            'delta_t': getattr(self, 'delta_t', None),
            'enable_mixed_precision': getattr(self, 'enable_mixed_precision', None),
            'device': str(getattr(self, 'device', None)) if getattr(self, 'device', None) is not None else None,
            'save_path': getattr(self, 'save_path', None),
            'finetuned_weights': getattr(self, 'finetuned_weights', None),
            'current_sparsity': getattr(self, 'current_sparsity', None),
        }
        self.logger_params = {k: v for k, v in logger_params.items() if v is not None}

    def train(self):
        """Main training loop that handles training, pruning, retraining, and logging."""
        self.model.to(self.device)

        # Initializing Logger
        if self.logger is not None:
            self.logger.create_log()
            self.logger.log_hyperparameters(self.logger_params)
                
        if self.enable_mixed_precision:
            print("[TRAINER] Training with mixed precision enabled")
        print(f"[TRAINER] Initial model sparsity: {self._get_sparsity_key()}")

        for epoch in range(self.epochs):
            # Training
            desc = f"Training Epoch [{epoch+1}/{self.epochs}]"
            avg_loss = self._run_train_epoch(epoch, desc)

            # Validation
            if self.criterion_type == 'supervised':
                desc = f"Validation Epoch [{epoch+1}/{self.epochs}]"
                avg_acc, avg_perplexity = self._run_validation_epoch(desc)
            else:
                avg_acc, avg_perplexity = 0, 0

            self.train_losses.append(avg_loss)
            if self.prune and self.pruner:
                sparsity_key = self._get_sparsity_key()
                if sparsity_key not in self.accuracy_per_sparsity:
                    self.accuracy_per_sparsity[sparsity_key] = []
                self.accuracy_per_sparsity[sparsity_key].append(avg_acc)
            else:
                self.accuracies.append(avg_acc)
                
            info = (f"Epoch [{epoch+1}/{self.epochs}]: Avg Loss: {avg_loss:.4f} | "
                    f"Avg Accuracy: {avg_acc} | Avg Perplexity: {avg_perplexity} |"
                    f"Model Sparsity: {self._get_sparsity_key()}\n")
            print(info)
            
            if self.logger is not None:
                self.logger.log_epochs(info)
            
            if self._handle_save(epoch):
                print(f"[TRAINER] weights saved!")
                self.unchanged = 0
            else:
                self.unchanged += 1
                if self.unchanged >= self.patience:
                    print(f"[TRAINER] Training stopped. No improvements for {self.unchanged} epochs.")
                    break

            # Pruning and recovery schedule
            if self.prune and self.pruner and epoch < self.pruning_epochs:
                self.fine_tune()

    def fine_tune(self):
        """Recovery finetuning after pruning."""
        self.recover = True
        for epoch in range(self.recovery_epochs):
            desc = f"Recovery Epoch [{epoch+1}/{self.recovery_epochs}]"
            avg_loss = self._run_train_epoch(epoch, desc)

            if self.criterion_type == 'supervised':
                desc = f"Validation Epoch [{epoch+1}/{self.recovery_epochs}]"
                avg_acc, avg_perplexity = self._run_validation_epoch(desc)

                self.train_losses.append(avg_loss)
                sparsity_key = self._get_sparsity_key()
                if sparsity_key not in self.accuracy_per_sparsity:
                    self.accuracy_per_sparsity[sparsity_key] = []
                self.accuracy_per_sparsity[sparsity_key].append(avg_acc)
            else:
                avg_acc = 0

            # Format info message based on model type and task
            if self.model_type == 'llm' and self.model_task == 'wikitext2':
                info = (f"Recovery epoch [{epoch+1}/{self.recovery_epochs}]: Avg Loss: {avg_loss:.4f} | "
                       f"Avg Accuracy: {avg_acc} | Avg Perplexity: {avg_perplexity} | "
                       f"Model Sparsity: {self._get_sparsity_key()}\n")
            else:
                info = (f"Recovery epoch [{epoch+1}/{self.recovery_epochs}]: Avg Loss: {avg_loss:.4f} | "
                       f"Avg Accuracy: {avg_acc} | "
                       f"Model Sparsity: {self._get_sparsity_key()}\n")
            print(info)

            if self.logger is not None:
                self.logger.log_epochs(info)
            
            if self._handle_save(epoch):
                print(f"[TRAINER] weights saved!")
        self.recover = False

    def _run_train_epoch(self, epoch, desc="", max_steps=None):
        """Run one full training epoch."""
        if (self.prune and self.pruner) and (hasattr(self.pruner, 'is_wanda') and self.pruner.is_wanda) and (not self.recover):
            self.pruner.register_hooks(self.model)

        self.model.train()
        total_loss = 0
        batchloader = tqdm(self.trainloader, desc=desc, leave=False) if self.enable_tqdm else self.trainloader

        for step, batch in enumerate(batchloader):
            # Handling different batch formats
            if self.criterion_type == 'contrastive':
                images, labels = batch
                images1, images2 = images 
                batch = {
                    'data1': images1.to(self.device),
                    'data2': images2.to(self.device),
                }
                labels = labels.to(self.device)
            elif isinstance(batch, list):
                batch = [x.to(self.device) for x in batch]
                batch, labels = batch
            elif isinstance(batch, dict):
                batch = {k: v.to(self.device) for k, v in batch.items()}
            else:
                batch.to(self.device)

            # Forward pass with mixed precision
            with autocast(device_type=self.device) if self.enable_mixed_precision else contextlib.nullcontext():
                if self.criterion_type == 'supervised':
                    outputs = self.model(batch)
                    if hasattr(outputs, 'loss') and outputs.loss:
                        loss = outputs.loss
                    elif hasattr(outputs, 'logits'):
                        loss = self.criterion(outputs.logits, labels)
                    else:
                        loss = self.criterion(outputs, labels)
                elif self.criterion_type == 'contrastive':
                    embeddings_1 = self.model(batch['data1'])
                    embeddings_2 = self.model(batch['data2'])
                    feature_embeddings = torch.cat((embeddings_1, embeddings_2))
                    loss = self.criterion(feature_embeddings, labels)

            total_loss += loss.item()

            # Handling backprop, optimization, and pruning
            _handle_optimizer_and_pruning(self, loss, epoch, step)

            running_loss = total_loss / (step + 1)
            if self.enable_tqdm:
                batchloader.set_postfix(Loss=f"{running_loss:.4f}", Sparsity=f"{self._get_sparsity_key()}")
            
            if max_steps and step >= max_steps:
                return running_loss
            
        avg_loss = total_loss / len(self.trainloader)    
        return avg_loss

    def _run_validation_epoch(self, desc="", mode="val"):
        """Run one full validation epoch."""
        self.model.eval()
        total_correct, total_samples = 0, 0
        total_loss, avg_perplexity = 0, 0

        if mode == 'eval' and self.testloader:
            batchloader = tqdm(self.testloader, desc=desc) if self.enable_tqdm else self.testloader
        else:
            batchloader = tqdm(self.valloader, desc=desc) if self.enable_tqdm  else self.valloader
            
        with torch.no_grad():
            for step, batch in enumerate(batchloader):
                # Batch handling
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                elif isinstance(batch, list):
                    batch = [x.to(self.device) for x in batch]
                    batch, labels = batch
                else:
                    batch.to(self.device)
                
                # Loss Handling
                if self.enable_mixed_precision:
                    with autocast(device_type=self.device):
                        outputs = self.model(batch)
                else:
                    outputs = self.model(batch)
                
                # Prediction Handling
                if self.model_type == 'llm':
                    if self.model_task == 'wikitext2':
                        labels = batch["labels"]
                        
                        # Generating mask and masking logits/labels
                        mask = (labels != -100)
                        logits = outputs.logits[mask]                        
                        labels = labels[mask]

                        # Accuracy calculation
                        preds = torch.argmax(logits, dim=-1)
                        correct = (preds == labels).sum()
                        total_correct += correct.item()
                        total_samples += mask.sum().item()
                        avg_acc = (total_correct / total_samples) * 100

                        # Perplexity calculation
                        total_loss += outputs.loss
                        avg_perplexity = torch.exp(total_loss / (step + 1)).item()

                    else:
                        labels = batch["labels"]
                        preds = torch.argmax(outputs.logits, dim=1)
                        correct = (preds == labels).sum()
                        total_correct += correct.item()
                        total_samples += batch['input_ids'].size(0)
                        avg_acc = (total_correct / total_samples) * 100
                else:
                    if hasattr(outputs, 'logits'):
                        outputs = outputs.logits
                    preds = outputs.max(1)[1]
                    correct = (preds == labels).sum()
                    total_correct += correct.item()
                    total_samples += labels.size(0)
                    avg_acc = (total_correct / total_samples) * 100

                metrics = {
                    'accuracy': f"{avg_acc:.2f}",
                    'perplexity': f"{avg_perplexity:.3f}",
                    'sparsity': f'{self._get_sparsity_key()}'
                }
                if self.enable_tqdm:
                    batchloader.set_postfix(**metrics)

        return metrics['accuracy'], metrics['perplexity']

    def evaluate(self, load=True):
        """Load weights and evaluate the model on validation or test set. Returns accuracy in percentage."""
        if load:
            print(f"[TRAINER] Loading weights: {self.save_path}")
            if self.save_path and load_weights(self.model, self.save_path):
                print("[TRAINER] Weights loaded successfully")
            else:
                print("[TRAINER] Failed to load weights")

        print(f"[TRAINER] Model Sparsity: {self._get_sparsity_key()}")

        self.model.eval()
        self.model.to(self.device)

        desc = "Evaluating"
        avg_acc, avg_perplexity = self._run_validation_epoch(desc, 'eval')

        return {
            "average_accuracy": avg_acc, 
            "average_perplexity": avg_perplexity if avg_perplexity else None,             
        }

    def _extract_normalized_answer(self, input_ids, start_idx, end_idx):
        answer_ids = input_ids[start_idx: end_idx + 1]
        answer = self.tokenizer.decode(answer_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        def remove_articles(text):
            regex = re.compile(r'\b(a|an|the)\b', re.IGNORECASE)
            return re.sub(regex, ' ', text)
        
        def white_space_fix(text):
            return ' '.join(text.split())
        
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(char for char in text if char not in exclude)

        return white_space_fix(remove_articles(remove_punc(answer))).strip()
        

    def _handle_save(self, epoch):
        """Save the model based on the validation accuracy."""
        if self.save_path:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        if self.prune and self.pruner:
            sparsity_key = self._get_sparsity_key()
            accuracy_list = self.accuracy_per_sparsity[sparsity_key]

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
        

















        