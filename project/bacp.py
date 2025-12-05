import os
import torch
import torch.nn as nn
import contextlib
from tqdm import tqdm
from copy import deepcopy
from torch.amp import autocast, GradScaler
from dataclasses import dataclass
from types import SimpleNamespace
from training_utils import (
    _initialize_models,
    _initialize_data_loaders,
    _initialize_optimizer,
    _initialize_scheduler,
    _initialize_pruner,
    _initialize_paths_and_logger,
    _handle_optimizer_and_pruning,
    _handle_wanda_hooks,
    _initialize_log_parameters,
    _handle_data_to_device,
    _handle_tqdm_logs,
    _log_metrics,
    _get_sparsity_key,
    _initialize_logs
)
from dataset_factory import load_cv_dataloaders
from dyrelu_adapter import step_dyrelu_adapter, set_t_for_dyrelu_adapter
from loss_functions import SupConLoss, NTXentLoss
from pruning_factory import *
from utils import *

@dataclass
class BaCPTrainingArguments:
    model_name:             str
    model_type:             str
    dataset_name:           str
    num_classes:            int
    batch_size:             int
    optimizer_type:         str
    learning_rate:          float
    tau:                    float = 0.07
    num_out_features:       int = 128       # Model embedded vector size
    image_size:             int = 32        # Image size for resizing
    epochs:                 int = 5         # Number of epochs to train
    scheduler_type:         str = None      # Scheduler type, e.g., "linear_with_warmup"
    patience:               int = None      # Number of epochs to wait before early stopping
    trained_weights:        str = None      # Path to pretrained weights
    experiment_type:        str = ""        # Type of experiment, e.g., "bacp_baseline" or "bacp"
    log_epochs:             bool = False    # Whether to log epochs in directory
    enable_tqdm:            bool = False    # Whether to enable tqdm progress bar
    enable_mixed_precision: bool = True     # Whether to enable mixed precision training
    databricks_env:         bool = True     # Whether to save model weights to DBFS (only when using databricks)
    num_workers:            int = os.cpu_count()

    # Pruning arguments
    pruning_type:           str = None      # Pruning type, e.g., "magnitude_pruning"
    target_sparsity:        float = None    # Target sparsity for pruning
    sparsity_scheduler:     str = None      # Sparsity scheduler, e.g., "cubic"
    recovery_epochs:        int = 0         # Number of epochs to recover after pruning
    pruning_module:         object = None   # Pruning module

    # Fine-tuning arguments
    enable_finetune:        bool = True
    epochs_ft:              int = 100
    optimizer_type_ft:      str = 'sgd'      # Optimizer type for fine-tuning
    learning_rate_ft:       float = 0.005    # Learning rate for fine-tuning

    # BaCP arguments
    lambdas:                list = None     # List of lambdas for BaCP
    learnable_lambdas:      bool = False    # Whether to learn lambdas or keep them static
    n_views:                int = 2         # Number of views for contrast  

    # DyReLU Phasing
    dyrelu_en:              bool = False
    dyrelu_phasing_en:      bool = False
    
    def __post_init__(self):
        self.scaler = GradScaler() if self.enable_mixed_precision else None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.supervised_criterion = SupConLoss(self.tau, self.tau, self.device).to(self.device)
        self.unsupervised_criterion = NTXentLoss(self.tau, self.device).to(self.device)
        self.retrain = True if self.recovery_epochs > 0 else False
        self.is_bacp = True

        _initialize_models(self)
        _initialize_data_loaders(self)
        _initialize_optimizer(self)
        _initialize_scheduler(self)
        _initialize_pruner(self)
        _initialize_paths_and_logger(self)
        _initialize_log_parameters(self)

class BaCPTrainer:
    def __init__(self, bacp_training_args):
        for key, value in vars(bacp_training_args).items():
            setattr(self, key, value)

        # Initialize training state
        self.recover = False
        self.snapshots = []
        self.current_sparsity = check_model_sparsity(self.model)
        self.context = autocast(device_type=self.device) if self.enable_mixed_precision else contextlib.nullcontext()
        self._initialize_lambdas()
        self._initialize_metric_lists()

        # Initializing list for GPU-resident snapshots
        self.ss_models_on_device = []

        # Initializing DyReLU Phasing Schedule
        self._initialize_dyrelu_phasing()

        self.model.train()
        self.model_pt.eval()
        self.model_ft.eval()


    def _initialize_dyrelu_phasing(self):
        """
        Calculates the total number of steps for the DyReLU phase-out 
        and sets the schedule on the model.
        The DyReLU is phased out linearly from t_start (100% DyReLU) 
        to t_end (100% standard ReLU).
        """
        if self.dyrelu_phasing_en:
            t_start = 0

            # Phasing runs through pre-training (self.epochs), recovery (self.recovery_epochs), 
            # and a fraction of the fine-tuning phase (0.25 * self.epochs_ft).
            t_end = self.epochs = self.recovery_epochs + (0.25 * self.epochs_ft)
            t_end = int(t_end)

            set_t_for_dyrelu(self.model, t_start, t_end)
            print(f"[DyReLU Phasing] Schedule set: t_start={t_start}, t_end={t_end}")


    def _initialize_lambdas(self):
        # Default static value of 0.25 (equal weightage of each)
        if self.lambdas is None:
            self.lambda1, self.lambda2, self.lambda3, self.lambda4 = 0.25, 0.25, 0.25, 0.25
        else:
            if self.learnable_lambdas:
                self.lambdas = [nn.Parameter(torch.tensor(l, requires_grad=True)).to(self.device) for l in self.lambdas]
                self.lambda1, self.lambda2, self.lambda3, self.lambda4 = self.lambdas
            else:
                self.lambda1, self.lambda2, self.lambda3, self.lambda4 = self.lambdas

        if isinstance(self.lambda1, nn.Parameter):
            self.optimizer.add_param_group({'params': [self.lambda1, self.lambda2, self.lambda3, self.lambda4]})
            print("- Lambdas are learnable.")

        self.lambda_history = {
            'lambda1': [], 'lambda2': [], 'lambda3': [], 'lambda4': [],
        }


    def _initialize_metric_lists(self):
        self.total_losses = {}
        self.prc_losses = {}
        self.snc_losses = {}
        self.fic_losses = {}
        self.ce_losses = {}
        self.training_acc = {}
        self.finetuning_loss = {}
        self.finetuning_acc = {}


    def train(self, run=None):
        _initialize_logs(self)

        for epoch in range(self.epochs):
            # Training phase
            curr_epoch_str = f"Epoch [{epoch+1}/{self.epochs}]"

            # Training
            desc = f"Training {curr_epoch_str}"
            metrics = self._run_train_epoch(epoch, desc)

            # Appends training loss and validation accuracy to lists
            self._update_metric_lists(metrics)    

            # Logs these metrics to the logger or to W&B
            _log_metrics(self, curr_epoch_str, metrics, run)      

            # Saving model 
            self._handle_save(epoch)
            
            # Pruning and recovery schedule
            if self.retrain:
                self._retrain(run) 

            self._handle_snapshot_creation()
            torch.save(self.model.state_dict(), self.save_path)
        
        if self.enable_finetune:
            # self.finetune(run)
            pass

    
    def finetune(self, run=None):
        if self.enable_finetune == False:
            return

        torch.cuda.empty_cache()

        # Disabling pruning. Only the sparse masked is applied
        self.is_bacp = False
        self.criterion = nn.CrossEntropyLoss()
        self.prune = False
        self.recover = True
        self.pruner = self.get_pruner()
        self.current_sparsity = check_model_sparsity(self.model)

        self.trained_weights = self.save_path
        self.model.train()
        load_weights(self.model, self.trained_weights)

        self.save_path = self.save_path.replace('.pt', '_finetune.pt')
        print(f"[TRAINER] Saving model to {self.save_path}")

        model_sparsity = check_model_sparsity(self.model)
        if model_sparsity > self.target_sparsity:
            print(f"[TRAINER] Model sparsity is {model_sparsity}. Target sparsity is {self.target_sparsity}. Skipping finetuning.")
            return

        # Creating new dataloaders
        data = load_cv_dataloaders(
            dataset_name=self.dataset_name, 
            t_type='supervised',
            batch_size=self.batch_size, 
            size=self.image_size,
            n_views=1,
            num_workers=self.num_workers,
            cache_dir=self.cache_dir,
            )
        
        self.ft_trainloader = data.get("trainloader")
        self.ft_valloader = data.get("valloader")
        self.ft_testloader = data.get("testloader")

        optimizer_args = {
            "optimizer_type":   self.optimizer_type_ft,
            "learning_rate":    self.learning_rate_ft,
            "model":            self.model
        }
        _initialize_optimizer(SimpleNamespace(**optimizer_args))

        for epoch in range(self.epochs_ft):
            # Training phase
            curr_ft_epoch_str = f"Fine-tuning Epoch [{epoch+1}/{self.epochs_ft}]"

            # Recovery training
            desc = f"Training {curr_ft_epoch_str}"
            ft_loss = self._run_finetune_epoch(epoch, desc)

            # Recovery validation
            desc = f"Validation {curr_ft_epoch_str}"
            ft_metrics = self._run_finetune_validation_epoch(desc)

            # Appends training loss and validation accuracy to lists
            ft_metrics['Finetuning Loss'] = ft_loss
            self._update_metric_lists(ft_metrics=ft_metrics)   

            # Logs these metrics to the logger or to W&B
            _log_metrics(self, curr_ft_epoch_str, ft_metrics, run) 
            
            # Saving model
            self._handle_save(epoch)

        final_metrics = self.evaluate(self.save_path, run)
        for k, v in final_metrics.items():
            if v is None:
                continue
            print(f"{k}: {v:.5f}")


    def evaluate(self, weights=None, run=None):
        """Evaluate model performnace on the testing dataset"""
        if weights is not None:
            if self.save_path and load_weights(self.model, self.save_path):
                print("[TRAINER] Weights loaded successfully")
            else:
                print("[TRAINER] Failed to load weights")
        else:
            print("[TRAINER] No weights provided. Will evaluate current model")

        self.model.eval()
        self.model.to(self.device)

        desc = "Evaluating"
        metrics = self._run_finetune_validation_epoch(desc, 'eval')

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
        """Recover model performance by running additional training epochs."""
        self.recover = True
        
        for epoch in range(self.recovery_epochs):
            curr_rec_epoch_str = f"Recovery Epoch [{epoch+1}/{self.recovery_epochs}]"

            # Recovery training
            desc = f"Training {curr_rec_epoch_str}"
            metrics = self._run_train_epoch(epoch, desc)

            # Appends training loss and validation accuracy to lists
            self._update_metric_lists(metrics)      

            # Logs these metrics to the logger or to W&B
            _log_metrics(self, curr_rec_epoch_str, metrics, run)     

            # Saving model 
            self._handle_save(epoch) 

        self.recover = False


    def _run_train_epoch(self, epoch, desc=""):
        """Run a training epoch."""
        _handle_wanda_hooks(self)

        if len(self.snapshots) > len(self.ss_models_on_device):
            for i in range(len(self.ss_models_on_device), len(self.snapshots)):
                self.ss_models_on_device.append(self.snapshots[i].to(self.device))

        total_loss, total_prc, total_snc, total_fic, total_ce, total_acc = 0, 0, 0, 0, 0, 0
        batchloader = tqdm(self.trainloader, desc=desc, leave=False) if self.enable_tqdm else self.trainloader

        for step, batch in enumerate(batchloader):
            # Unpacking batch and moving to device
            data, labels = _handle_data_to_device(self, batch)
            data1, data2 = data

            with self.context:
                if self.model_type == 'cv':
                    curr_emb, curr_logits = self._get_embeddings_and_logits(data1)
                else:
                    raise ValueError("Model type not supported")

                with torch.no_grad():
                    if self.model_type == 'cv':
                        pt_emb = self.model_pt(data2, return_emb=True)
                        pt_emb = pt_emb.logits if hasattr(pt_emb, 'logits') else pt_emb

                        ft_emb = self.model_ft(data2, return_emb=True)
                        ft_emb = ft_emb.logits if hasattr(ft_emb, 'logits') else ft_emb
                    else:
                        raise ValueError("Model type not supported")

                # PrC Module
                L_prc_raw = self._calculate_contrastive_loss(curr_emb, pt_emb, labels)
                
                # FiC Module
                L_fic_raw = self._calculate_contrastive_loss(curr_emb, ft_emb, labels)

                # SnC Module
                L_snc_raw = torch.tensor(0.0, device=self.device)
                if len(self.ss_models_on_device) > 0:
                    for ss_model in self.ss_models_on_device:
                        with torch.no_grad():
                            ss_emb = ss_model(data2, return_emb=True)
                            ss_emb = ss_emb.logits if hasattr(ss_emb, 'logits') else ss_emb

                        L_snc_raw += self._calculate_contrastive_loss(curr_emb, ss_emb, labels)
                    L_snc_raw /= len(self.ss_models_on_device)

                # CE Module
                L_ce_raw = nn.CrossEntropyLoss()(curr_logits, labels)

                # Applying lambdas and combining losses
                loss, prc_loss, snc_loss, fic_loss, ce_loss = self._combine_losses(
                    L_prc_raw, L_snc_raw, L_fic_raw, L_ce_raw
                )

                # Accuracy
                preds = curr_logits.max(1)[1]
                accuracy = ((preds == labels).sum() / labels.size(0)) * 100

            _handle_optimizer_and_pruning(self, loss, epoch, step)

            total_loss += loss.item()
            total_prc += prc_loss.item()
            total_fic += fic_loss.item()
            total_snc += snc_loss.item()
            total_ce  += ce_loss.item()
            total_acc += accuracy.item()

            # Calculating running mean of loss components
            metrics = {
                "Total Loss": _to_float(total_loss / (step + 1)),
                "PrC Loss": _to_float(total_prc / (step + 1)),
                "SnC Loss": _to_float(total_snc / (step + 1)),
                "FiC Loss": _to_float(total_fic / (step + 1)),
                "CE Loss": _to_float(total_ce / (step + 1)),
                "Training Accuracy": _to_float(total_acc / (step + 1)),
                "lambdas":[
                    _to_float(self.lambda1),
                    _to_float(self.lambda2),
                    _to_float(self.lambda3),
                    _to_float(self.lambda4),
                    ],
            }
            _handle_tqdm_logs(self, batchloader, metrics)

        if self.dyrelu_phasing_en:
            step_dyrelu_adapter(self.model)

        return metrics
    

    def _run_finetune_epoch(self, epoch, desc=""):
        self.model.train()
        ft_total_loss = 0
        batchloader = tqdm(self.ft_trainloader, desc=desc, leave=False) if self.enable_tqdm else self.ft_trainloader

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
            ft_total_loss += loss.item()

            # Handling backprop, optimization, and pruning
            _handle_optimizer_and_pruning(self, loss, epoch, step)

            running_loss = ft_total_loss / (step + 1)
            _handle_tqdm_logs(self, batchloader, {'Finetuning Loss': running_loss})

        if self.dyrelu_phasing_en:
            step_dyrelu_adapter(self.model)

        avg_loss = ft_total_loss / len(batchloader)    
        return avg_loss
    

    def _run_finetune_validation_epoch(self, desc="", mode="val"):
        """Run a validation epoch."""
        self.model.eval()
        val_loss, val_acc, val_perp = 0, 0, 0

        dataloader = self.ft_testloader if (mode=='eval' and self.ft_testloader) else self.ft_valloader
        batchloader = tqdm(dataloader, desc=desc, leave=False) if self.enable_tqdm else dataloader

        with torch.no_grad():
            for step, batch in enumerate(batchloader):
                # Unpacking batch and moving to device
                data, labels = _handle_data_to_device(self, batch)

                with self.context:
                    outputs = self.model(data)

                metrics = self._handle_metrics(outputs, labels)
                val_loss += metrics.get('FT Batch Val Loss', 0)
                val_acc += metrics.get('FT Batch Accuracy', 0)
                val_perp += metrics.get('FT Batch Perplexity', 0)

                _handle_tqdm_logs(self, batchloader, metrics)

        avg_loss = val_loss / len(dataloader)
        avg_accuracy = val_acc / len(dataloader)
        avg_perplexity = val_perp / len(dataloader)
        return {
            'Finetuning Val Loss': avg_loss if avg_loss > 0.0 else None,
            'Finetuning Accuracy': avg_accuracy if avg_accuracy > 0.0 else None,
            'Finetuning Perplexity': avg_perplexity if avg_perplexity > 1.0 else None
        }


    def _get_embeddings_and_logits(self, data):
        """
        Perform a single forward pass on the main model to extract 
        features, logits (classification), and embeddings (contrastive).
        """
        if self.model_type == 'cv':
            features = self.model(data, return_feat=True)
            features = features.logits if hasattr(features, 'logits') else features
            logits = self.model.cls_head(features)   
            embeddings = self.model.get_embeddings(features)
        else:
            raise ValueError("Model type not supported")
        return embeddings, logits


    def _calculate_contrastive_loss(self, current_emb, target_emb, labels):
        """
        Calculates the combined Supervised and Unsupervised Contrastive Loss 
        (L_sup + L_unsup). Used for PrC, FiC, and SnC modules.
        """
        L_sup = self.supervised_criterion(current_emb, target_emb, labels)
        L_unsup = self.unsupervised_criterion(current_emb, target_emb)
        return L_sup + L_unsup
    
    
    def _combine_losses(self, prc_raw, snc_raw, fic_raw, ce_raw):
        """
        Applies the lambda weights to the four loss components and sums them.
        """
        prc_loss = prc_raw * self.lambda1
        fic_loss = fic_raw * self.lambda2
        snc_loss = snc_raw * self.lambda3
        ce_loss = ce_raw * self.lambda4
        
        total_loss = prc_loss + snc_loss + fic_loss + ce_loss
        return total_loss, prc_loss, snc_loss, fic_loss, ce_loss


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
            'FT Batch Val Loss': avg_loss,
            'FT Batch Accuracy': avg_accuracy,
            'FT Batch Perplexity': avg_perplexity,
        }
                
    
    def _handle_snapshot_creation(self):
        """
        Creates a snapshot of the current model and stores it on the CPU (to save GPU memory).
        The model is deepcopied to CPU here.
        """
        if len(self.snapshots) < self.epochs - 1:
            # Keep on CPU to save GPU memory. The transfer will happen once per epoch.
            ss_model = deepcopy(self.model).to('cpu').eval() 
            self.snapshots.append(ss_model)


    def get_pruner(self):
        # Loading sparse weights
        load_weights(self.model, self.save_path)

        # Creating pruning module
        pruning_module = PRUNING_REGISTRY[self.pruning_type](
            self.model, self.epochs, self.target_sparsity, self.sparsity_scheduler
        )

        # Setting sparse mask
        zero_masks = {name: (param != 0).float() for name, param in self.model.named_parameters()}

        pruning_module.masks = zero_masks
        return pruning_module
        

    def _update_metric_lists(self, metrics=None, ft_metrics=None):
        sparsity_key = _get_sparsity_key(self)

        if metrics is not None:
            self.total_losses.setdefault(sparsity_key, []).append(metrics.get('Total Loss'))
            self.prc_losses.setdefault(sparsity_key, []).append(metrics.get('PrC Loss'))
            self.snc_losses.setdefault(sparsity_key, []).append(metrics.get('SnC Loss'))
            self.fic_losses.setdefault(sparsity_key, []).append(metrics.get('FiC Loss'))
            self.ce_losses.setdefault(sparsity_key, []).append(metrics.get('CE Loss'))
            self.training_acc.setdefault(sparsity_key, []).append(metrics.get('Training Accuracy'))

            lambda1, lambda2, lambda3, lambda4 = metrics.get('lambdas')
            self.lambda_history['lambda1'].append(lambda1)
            self.lambda_history['lambda2'].append(lambda2)
            self.lambda_history['lambda3'].append(lambda3)
            self.lambda_history['lambda4'].append(lambda4)
        
        if ft_metrics is not None:
            self.finetuning_loss.setdefault(sparsity_key, []).append(ft_metrics.get('Finetuning Loss'))
            self.finetuning_acc.setdefault(sparsity_key, []).append(ft_metrics.get('Finetuning Accuracy'))


    def _handle_save(self, epoch):
        """Saves the model or stops training if no improvements are seen."""
        if self._save_model(epoch):
            print(f"[TRAINER] weights saved!")
    

    def _save_model(self, epoch):
        """Save the model based on the total loss."""
        if self.save_path:
            dir_path = os.path.dirname(self.save_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)

        sparsity_key = _get_sparsity_key(self)
        if self.is_bacp:
            loss_list = self.total_losses[sparsity_key]

            if len(loss_list) <= 1:
                torch.save(self.model.state_dict(), self.save_path)
                return True
            elif len(loss_list) > 1:
                if loss_list[-1] < min(loss_list[:-1]):
                    torch.save(self.model.state_dict(), self.save_path)
                    return True
            return False
        else:
            acc_list = self.finetuning_acc[sparsity_key]
            if len(acc_list) <= 1:
                torch.save(self.model.state_dict(), self.save_path)
                return True
            elif len(acc_list) > 1:
                if acc_list[-1] > max(acc_list[:-1]):
                    torch.save(self.model.state_dict(), self.save_path)
                    return True
            return False
        
        
def _to_float(x):
    return float(x.detach().cpu()) if isinstance(x, torch.nn.Parameter) or torch.is_tensor(x) else float(x)















