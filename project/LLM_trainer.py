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
from dataset_utils import get_glue_data, get_squad_data
from datasets import get_dataset_config_names
import evaluate

disable_progress_bar()
os.environ["HF_DATASETS_CACHE"] = "/dbfs/hf_datasets"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class LLMTrainingArguments:
    """Configuration class for training and pruning settings for LLM classifier.
    Args:
        model_name (str): Pretrained model name.
        model_task (str): Task/dataset name (e.g., GLUE task - sst2, qqp, etc).
        batch_size (int): Batch size for training.
        pruning_type (str, optional): Type of pruning method to use. Defaults to None.
        target_sparsity (float): Desired sparsity level for pruning. Defaults to 0.
        sparsity_scheduler (str): Pruning schedule type. Defaults to "linear".
        finetuned_weights (str, optional): Path to pretrained weights to load. Defaults to None.
        num_classes (int): Number of output classes. Defaults to 2.
        learning_rate (float): Learning rate. Defaults to 2e-5.
        learning_type (str): Identifier for the learning method. Defaults to "".
        epochs (int): Total number of training epochs. Defaults to 10.
        pruning_epochs (int, optional): Number of pruning epochs, defaults to total epochs if None.
        recovery_epochs (int): Number of finetuning epochs after pruning. Defaults to 5.
        log_epochs (bool): Whether to log training info per epoch. Defaults to True.
        enable_tqdm (bool): Whether to use tqdm progress bars. Defaults to True.
        enable_mixed_precision (bool): Use mixed precision training. Defaults to True.
        num_workers (int): DataLoader workers count. Defaults to 24.
        pruner (object, optional): External pruner instance to use. Defaults to None.
        finetune (bool): Flag for finetuning mode. Defaults to False.
        delta_t (int): Steps interval for pruning schedule. Defaults to 500.
        db (bool): Use DBFS directory for saving weights if True. Defaults to True.
    """
    def __init__(self, 
                 model_name, 
                 model_task,
                 batch_size, 
                 learning_rate, 
                 pruning_type=None,
                 target_sparsity=0,
                 sparsity_scheduler="linear",
                 finetuned_weights=None,
                 num_classes=2,
                 optimizer_type='adamw',
                 scheduler_type=None,
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
        self.model_type = 'llm'
        self.model_name = model_name
        self.model_task = model_task
        self.finetuned_weights = finetuned_weights
        self.num_classes = num_classes

        # Training parameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
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
        self._initialize_model(model_name, finetuned_weights, model_task)

        # Initializing dataloaders
        self._initialize_data_loaders(model_task, batch_size, num_workers)

        # Initializing pruner
        self._initialize_pruner(pruning_type, target_sparsity, sparsity_scheduler, delta_t, pruner)    

        # Initializing paths and logger
        self._initialize_paths_and_logger(db, learning_type, log_epochs)

    def _initialize_model(self, model_name, finetuned_weights, model_task):
        """Initialize the models required for training."""
        print("[LLM TRAINER] Initializing model")
        self.model = ClassificationNetwork(model_name, model_task=model_task, num_classes=self.num_classes)

        self.embedded_dim = self.model.embedding_dim
        if finetuned_weights:
            loaded = load_weights(self.model, finetuned_weights)
            print("[LLM TRAINER] Weights loaded" if loaded else "[LLM TRAINER] Failed to load weights")

        # Initializing tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Initializing optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)

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
            print("[LLM TRAINER] Pruning enabled")
            print(f"[LLM TRAINER] Initializing pruner for {pruning_type}")
            print(f"[LLM TRAINER] Pruning scheduler: {self.sparsity_scheduler}")
            print(f"[LLM TRAINER] Pruning target sparsity: {target_sparsity}")
        else:
            self.pruner = pruner
            print("[LLM TRAINER] Pruning disabled")
        
        self.current_sparsity = check_model_sparsity(self.model)
        print(f"[LLM TRAINER] Current model sparsity: {self.current_sparsity:.4f}")

    def _initialize_data_loaders(self, model_task, batch_size, num_workers):
        """Initialize data loaders for the specified task."""
        print(f"[LLM TRAINER] Initializing data loaders for {model_task}")
        if model_task in get_dataset_config_names("glue"):
            data = get_glue_data(self.tokenizer, model_task, batch_size, num_workers)
            if len(data) >= 2:
                self.trainloader = data["trainloader"]
                self.valloader = data["valloader"]
                self.testloader = data.get("testloader", None)
            else:
                raise ValueError(f"Expected at least trainloader and valloader for {model_task}, got {len(data)} loaders")
        
        elif model_task in 'squad':
            data = get_squad_data(self.tokenizer, batch_size, 1.0, num_workers)
            if len(data) >= 2:
                self.trainloader = data["trainloader"]
                self.valloader = data["valloader"]
                self.testloader = data.get("testloader", None)
            else:
                raise ValueError(f"Expected at least trainloader and valloader for {model_task}, got {len(data)} loaders")
            
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
                weights_path = f"{self.model_name}_{self.model_task}_{learning_type}"
                
            self.save_path = f"{base_dir}/research/{self.model_name}/{self.model_task}/{weights_path}.pt"
        
        self.logger = Logger(self.model_name, learning_type) if log_epochs else None
        print(f"[LLM TRAINER] Saving model checkpoints to {self.save_path}")

class LLMTrainer:
    """
    Trainer class for training, fine-tuning, and pruning an LLM classification models.
    """
    def __init__(self, training_args):
        for key, value in vars(training_args).items():
            setattr(self, key, value)

        if self.model_task == 'squad':
            self.squad_metric = evaluate.load(self.model_task)

        self.recover = False
        self.train_losses = []
        self.accuracy_per_sparsity = {}

        self._initialize_log_parameters()
    
    def _get_sparsity_key(self):
        return round(check_model_sparsity(self.model), 4)
    
    def _initialize_log_parameters(self):
        """Initialize parameters for logging purposes."""
        logger_params = {
            # Model information
            'model_type': getattr(self, 'model_type', None),
            'model_name': getattr(self, 'model_name', None),
            'model_task': getattr(self, 'model_task', None),
            'num_classes': getattr(self, 'num_classes', None),
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
            print("[LLM TRAINER] Training with mixed precision enabled")
        print(f"[LLM TRAINER] Initial model sparsity: {self._get_sparsity_key()}")

        for epoch in range(self.epochs):
            # Training
            desc = f"Training Epoch [{epoch+1}/{self.epochs}]"
            avg_loss = self._run_train_epoch(desc)

            # Validation
            desc = f"Validation Epoch [{epoch+1}/{self.epochs}]"
            avg_acc, avg_f1 = self._run_validation_epoch(desc)

            self.train_losses.append(avg_loss)
            sparsity_key = self._get_sparsity_key()
            if sparsity_key not in self.accuracy_per_sparsity:
                self.accuracy_per_sparsity[sparsity_key] = []
            self.accuracy_per_sparsity[sparsity_key].append(avg_acc)

            info = (
                f"Epoch [{epoch+1}/{self.epochs}]: Avg Loss: {avg_loss:.4f} | "
                f"Avg Accuracy: {avg_acc:.2f} | Avg F1: {avg_f1:.2f} | "
                f"Model Sparsity: {self._get_sparsity_key()}\n"
            )
            print(info)
            
            if self.logger is not None:
                self.logger.log_epochs(info)
            
            if self._handle_save(epoch):
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
            desc = f"Recovery Epoch [{epoch+1}/{self.recovery_epochs}]"
            avg_loss = self._run_train_epoch(desc)

            desc = f"Validation Epoch [{epoch+1}/{self.recovery_epochs}]"
            avg_acc, avg_f1 = self._run_validation_epoch(desc)

            self.train_losses.append(avg_loss)
            sparsity_key = self._get_sparsity_key()
            if sparsity_key not in self.accuracy_per_sparsity:
                self.accuracy_per_sparsity[sparsity_key] = []
            self.accuracy_per_sparsity[sparsity_key].append(avg_acc)

            info = (
                f"Recovery epoch [{epoch+1}/{self.recovery_epochs}]: Avg Loss: {avg_loss:.4f} | "
                f"Avg Accuracy: {avg_acc:.2f} | Avg F1: {avg_f1:.2f} | "
                f"Model Sparsity: {self._get_sparsity_key()}\n"
            )
            print(info)

            if self.logger is not None:
                self.logger.log_epochs(info)
            
            if self._handle_save(epoch):
                print(f"[LLM TRAINER] weights saved!")

        self.recover = False

    def _run_train_epoch(self, desc):
        """
        Run one full training epoch.
        """
        if (self.prune and self.pruner) and (hasattr(self.pruner, 'is_wanda') and self.pruner.is_wanda) and (not self.recover):
            self.pruner.register_hooks(self.model)

        self.model.train()
        total_loss = 0
        batchloader = tqdm(self.trainloader, desc=desc) if self.enable_tqdm else self.trainloader
        for step, batch in enumerate(batchloader):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            if self.enable_mixed_precision:
                with autocast(device_type=self.device):
                    outputs = self.model(batch)
                    loss = outputs.loss
            else:
                outputs = self.model(batch)
                loss = outputs.loss

            total_loss += loss.item()

            # Handling backprop, optimization, pruning, and logic
            self._handle_optimizer_and_pruning(loss, step)

            running_loss = total_loss / (step + 1)
            if self.enable_tqdm:
                batchloader.set_postfix(Loss=f"{running_loss:.4f}", Sparsity=f"{self._get_sparsity_key()}")

        avg_loss = total_loss / len(self.trainloader)    
        return avg_loss

    def _extract_normalized_answer(self, input_ids, start_idx, end_idx):
        import string
        import re

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
        
        return white_space_fix(remove_articles(remove_punc(answer.lower())))

    def _run_validation_epoch(self, desc):
        """Run one full validation epoch."""
        self.model.eval()
        total_correct, total_f1, total_samples, avg_f1 = 0, 0, 0, 0
        batchloader = tqdm(self.valloader, desc=desc) if self.enable_tqdm else self.valloader
        with torch.no_grad():
            for batch in batchloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                if self.enable_mixed_precision:
                    with autocast(device_type=self.device):
                        outputs = self.model(batch)
                else:
                    outputs = self.model(batch)

                if self.model_task == 'squad':
                    predictions, references = [], [] 
                    for i in range(batch["input_ids"].size(0)):
                        input_ids = batch["input_ids"][i]
                        start_logit = outputs.start_logits[i]
                        end_logit = outputs.end_logits[i]

                        start_idx = torch.argmax(start_logit).item()
                        end_idx = torch.argmax(end_logit).item()
                        if start_idx > end_idx:
                            end_idx = start_idx 

                        pred_text = self._extract_normalized_answer(input_ids, start_idx, end_idx)
                        true_text = self._extract_normalized_answer(input_ids, batch["start_positions"][i].item(), batch["end_positions"][i].item())

                        predictions.append({"id": str(i), "prediction_text": pred_text})
                        references.append({"id": str(i), "answers": {"text": [true_text], "answer_start": [0]}})
                    
                    metrics = self.squad_metric.compute(predictions=predictions, references=references)
                    total_correct += metrics['exact_match']
                    total_f1 += metrics['f1'] 
                    total_samples += 1

                    avg_acc = (total_correct / total_samples)
                    avg_f1 = (total_f1 / total_samples)
                    if self.enable_tqdm:
                        batchloader.set_postfix(Accuracy=f"{avg_acc:.2f}", F1=f"{avg_f1:.2f}", Sparsity=f"{self._get_sparsity_key()}")

                else:
                    labels = batch["label"]
                    preds = torch.argmax(outputs.logits, dim=1)
                    correct = (preds == labels).sum()
                    total_correct += correct.item()
                    total_samples += batch['input_ids'].size(0)
                    avg_acc = (total_correct / total_samples) * 100
                    
                    if self.enable_tqdm:
                        batchloader.set_postfix(Accuracy=f"{avg_acc:.2f}", Sparsity=f"{self._get_sparsity_key()}")
        
        return avg_acc, avg_f1
    
    def evaluate(self, load=True):
        """
        Load weights and evaluate the model on validation or test set.
        Returns accuracy in percentage.
        """
        if load:
            print(f"[LLM TRAINER] Loading weights: {self.save_path}")
            if self.save_path and load_weights(self.model, self.save_path):
                print("[LLM TRAINER] Weights loaded successfully")
            else:
                print("[LLM TRAINER] Failed to load weights")

        print(f"[LLM TRAINER] Model Sparsity: {self._get_sparsity_key()}")

        self.model.eval()
        self.model.to(self.device)

        desc = "Evaluating"
        avg_acc, avg_f1 = self._run_validation_epoch(desc)
        if self.model_task == 'squad':
            return {
                "average_accuracy": avg_acc, 
                "average_f1_score": avg_f1                
            }
        else:
            return {
                "average_accuracy": avg_acc, 
            }
    
    def _handle_optimizer_and_pruning(self, loss, step):
        """Handle backpropagation, pruning, and weight update in a single step."""
        self.optimizer.zero_grad()

        if self.enable_mixed_precision:
            self.scaler.scale(loss).backward()
            self._apply_pruning(step)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self._apply_pruning(step)
            self.optimizer.step()

        if self.finetune or (self.prune and self.pruner is not None):
            self.pruner.apply_mask(self.model)
            
    def _apply_pruning(self, step):
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
        
    def _handle_save(self, epoch):
        """Save the model based on the validation accuracy"""
        if self.save_path:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

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




