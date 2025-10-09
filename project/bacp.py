import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm
from copy import deepcopy
from torch.amp import autocast, GradScaler
from unstructured_pruning import *
from utils import *
import contextlib
from training_utils import (
    _initialize_models, _initialize_optimizer, _initialize_scheduler,
    _initialize_data_loaders, _initialize_pruner,
    _initialize_paths_and_logger, _handle_optimizer_and_pruning
)
from loss_functions import SupConLoss, NTXentLoss
from dataclasses import dataclass
import numpy as np

@dataclass
class BaCPTrainingArguments:
    model_name:             str
    model_type:             str
    dataset_name:           str
    num_classes:            int
    batch_size:             int
    optimizer_type:         str
    learning_rate:          float
    tau:                    float
    image_size:             int = 32        # Image size for resizing
    epochs:                 int = 5         # Number of epochs to train
    scheduler_type:         str = None      # Scheduler type, e.g., "linear_with_warmup"
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
    recovery_epochs:        int = 10        # Number of epochs to recover after pruning
    retrain:                bool = True     # Whether to retrain after pruning

    # BaCP arguments
    lambdas:                list = None     # List of lambdas for BaCP
    learnable_lambdas:      bool = False    # Whether to learn lambdas or keep them static
    n_views:                int = 2         # Number of views for contrast  

    
    def __post_init__(self):
        self.scaler = GradScaler() if self.enable_mixed_precision else None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.supervised_criterion = SupConLoss(self.tau, self.tau, self.device).to(self.device)
        self.unsupervised_criterion = NTXentLoss(self.tau, self.device).to(self.device)
        self.is_bacp = True

        _initialize_models(self)
        _initialize_data_loaders(self)
        _initialize_optimizer(self)
        _initialize_scheduler(self)
        _initialize_pruner(self)
        _initialize_paths_and_logger(self)

class BaCPTrainer:
    def __init__(self, bacp_training_args):
        for key, value in vars(bacp_training_args).items():
            setattr(self, key, value)

        # Initialize training state
        self.recover = False
        self.snapshots = []
        self.current_sparsity = check_model_sparsity(self.model)
        self.context = autocast(device_type=self.device) if self.enable_mixed_precision else contextlib.nullcontext()
        self._initialize_heads()
        self._initialize_lambdas()
        self._initialize_metric_lists()
        self._initialize_log_parameters()

        self.model.train()
        self.pre_trained_model.eval()
        self.finetuned_model.eval()

    def _initialize_heads(self):
        if self.num_classes is not None:
            self.classification_head = nn.Linear(self.embedded_dim, self.num_classes).to(self.device)
        else:
            self.classification_head = None

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
            'lambda1': [],
            'lambda2': [],
            'lambda3': [],
            'lambda4': [],
        }

    def _initialize_metric_lists(self):
        self.total_losses = {}
        self.prc_losses = {}
        self.snc_losses = {}
        self.fic_losses = {}
        self.ce_losses = {}

    def _initialize_log_parameters(self):
        """Initialize parameters for logging purposes."""
        allowed_types = (int, float, str, bool, torch.Tensor, np.ndarray, list, dict, type(None))
        self.logger_params = {
            k: v for k, v in vars(self).items()
            if isinstance(v, allowed_types) and v is not None
        }
    
    def train(self, run):
        self._initialize_logs()

        for epoch in range(self.epochs):
            # Training phase
            curr_epoch_str = f"Epoch [{epoch+1}/{self.epochs}]"

            # Training
            desc = f"Training {curr_epoch_str}"
            loss_metrics = self._run_train_epoch(epoch, desc)

            # Appends training loss and validation accuracy to lists
            self._update_metric_lists(loss_metrics)    

            # Logs these metrics to the logger or to W&B
            self._log_metrics(curr_epoch_str, loss_metrics, run)      

            # Saving model 
            self._handle_save(epoch)
            
            # Pruning and recovery schedule
            if self.retrain:
                self._retrain(run) 

            self._handle_snapshot_creation()
    
    def _retrain(self, run=None):
        """Recover model performance by running additional training epochs."""
        self.recover = True
        
        for epoch in range(self.recovery_epochs):
            curr_rec_epoch_str = f"Recovery Epoch [{epoch+1}/{self.recovery_epochs}]"

            # Recovery training
            desc = f"Training {curr_rec_epoch_str}"
            loss_metrics = self._run_train_epoch(epoch, desc)

            # Appends training loss and validation accuracy to lists
            self._update_metric_lists(loss_metrics)      

            # Logs these metrics to the logger or to W&B
            self._log_metrics(curr_rec_epoch_str, loss_metrics, run)          

            # Saving model 
            self._handle_save(epoch)
        self.recover = False

    def _run_train_epoch(self, epoch, desc=""):
        """Run a training epoch."""
        self._handle_wanda_hooks()

        avg_total_loss, avg_prc_loss, avg_snc_loss, avg_fic_loss, avg_ce_loss = 0, 0, 0, 0, 0
        batchloader = tqdm(self.trainloader, desc=desc, leave=False) if self.enable_tqdm else self.trainloader

        for step, batch in enumerate(batchloader):
            # Unpacking batch and moving to device
            data, labels = self._handle_data_to_device(batch)
            data1, data2 = data

            with self.context:
                curr_emb, curr_logits = self._get_embeddings_and_logits(
                    data1 if self.model_type == 'cv' else raise ValueError("Model type not supported")
                )

                with torch.no_grad():
                    if self.model_type == 'cv':
                        pt_emb = self.pre_trained_model(data2)
                        pt_emb = pt_emb.logits if hasattr(pt_emb, 'logits') else pt_emb

                        ft_emb = self.finetuned_model(data2)
                        ft_emb = ft_emb.logits if hasattr(ft_emb, 'logits') else ft_emb
                    else:
                        raise ValueError("Model type not supported")

                # PrC Module
                L_prc_sup = self.supervised_criterion(curr_emb, pt_emb, labels)
                L_prc_unsup = self.unsupervised_criterion(curr_emb, pt_emb)
                prc_loss = (L_prc_sup + L_prc_unsup) * self.lambda1

                # FiC Module
                L_fic_sup = self.supervised_criterion(curr_emb, ft_emb, labels)
                L_fic_unsup = self.unsupervised_criterion(curr_emb, ft_emb)
                fic_loss = (L_fic_sup + L_fic_unsup) * self.lambda2

                # SnC Module
                L_snc_sup = torch.tensor(0.0, device=self.device)
                L_snc_unsup = torch.tensor(0.0, device=self.device)
                for ss_model in self.snapshots:
                    ss_model.to(self.device)
                    with torch.no_grad():
                        if self.model_type == 'cv':
                            ss_emb = ss_model(data2)
                            ss_emb = ss_emb.logits if hasattr(ss_emb, 'logits') else ss_emb
                        else:
                            raise ValueError("Model type not supported")
                    ss_model.to('cpu')

                    L_snc_sup += self.supervised_criterion(curr_emb, ss_emb, labels)
                    L_snc_unsup += self.unsupervised_criterion(curr_emb, ss_emb)

                snc_loss = ((L_snc_sup + L_snc_unsup) / len(self.snapshots)) * self.lambda3

                # CE Module
                ce_loss = nn.CrossEntropyLoss()(curr_logits, labels) * self.lambda4

            total_loss = prc_loss + snc_loss + fic_loss + ce_loss
            _handle_optimizer_and_pruning(self, total_loss, epoch, step)

            # Calculating running mean of loss components
            avg_prc_loss += prc_loss / (step + 1)
            avg_snc_loss += snc_loss / (step + 1)
            avg_fic_loss += fic_loss / (step + 1)
            avg_ce_loss += ce_loss / (step + 1)
            avg_total_loss += total_loss / (step + 1)

            loss_metrics = {
                "total_loss": ,
                "prc_loss": avg_prc_loss.item(),
                "snc_loss": avg_snc_loss.item(),
                "fic_loss": avg_fic_loss.item(),
                "ce_loss": avg_ce_loss.item(),
                "lambda1": _to_float(self.lambda1),
                "lambda2": _to_float(self.lambda2),
                "lambda3": _to_float(self.lambda3),
                "lambda4": _to_float(self.lambda4),
            }
            self._handle_tqdm_logs(batchloader, loss_metrics)
        return loss_metrics


    def _get_sparsity_key(self):
        """Get current model sparsity as a rounded key."""
        return round(self.current_sparsity, 4)
    
    def _initialize_logs(self):
        if self.logger is not None:
            self.logger.create_log()
            self.logger.log_hyperparameters(self.logger_params)
        else:
            return

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

    def _handle_snapshot_creation(self):
        if len(self.snapshots) < self.epochs - 1:
            ss_model = deepcopy(self.model).to('cpu').eval()
            self.snapshots.append(ss_model)

    def _handle_wanda_hooks(self):
        if self.prune and hasattr(self.pruner, 'is_wanda') and self.pruner.is_wanda:
            if not self.recover:    # We don't want to register hooks during recovery
                self.pruner.register_hooks(self.model)
            else:
                pass
    
    def _handle_data_to_device(self, data):
        if self.is_bacp:
            data
        if isinstance(data, list):
            data = [x.to(self.device) for x in data]
            data, labels = data
        elif isinstance(data, dict):
            data = {k: v.to(self.device) for k, v in data.items()}
            labels = data.get('labels', None)
        else:
            raise ValueError(f"Data type {type(data)} not supported")
        return data, labels
    
    def _update_metric_lists(self, loss, accuracy=None):
        sparsity_key = self._get_sparsity_key()
        self.total_losses.setdefault(sparsity_key, []).append(loss.get('total_loss'))
        self.prc_losses.setdefault(sparsity_key, []).append(loss.get('prc_loss'))
        self.snc_losses.setdefault(sparsity_key, []).append(loss.get('snc_loss'))
        self.fic_losses.setdefault(sparsity_key, []).append(loss.get('fic_loss'))
        self.ce_losses.setdefault(sparsity_key, []).append(loss.get('ce_loss'))
        self.lambda_history['lambda1'].append(loss.get('lambda1'))
        self.lambda_history['lambda2'].append(loss.get('lambda2'))
        self.lambda_history['lambda3'].append(loss.get('lambda3'))
        self.lambda_history['lambda4'].append(loss.get('lambda4'))

    def _get_embeddings_and_logits(self, data):
        features = self._extract_features(data)
        logits = self._extract_logits(features)       
        embeddings = self._extract_embeddings(features) 
        return embeddings, logits
    
    def _extract_features(self, data):
        if self.model_type == 'cv':
            features = self.model(data, return_feat=True)
            features = features.logits if hasattr(features, 'logits') else features
            return features
        else:
            raise ValueError("Model type not supported")

    def _extract_logits(self, features):
        if self.model_type == 'cv':
            return self.model.cls_head(features)
        else:
            raise ValueError("Model type not supported")
 
    def _extract_embeddings(self, features):
        if self.model_type == 'cv':
            return self.model.get_embeddings(features)
        else:
            raise ValueError("Model type not supported")

    def _handle_save(self, epoch):
        """Saves the model or stops training if no improvements are seen."""
        if self._save_model(epoch):
            print(f"[TRAINER] weights saved!")
    
    def _save_model(self, epoch):
        """Save the model based on the total loss."""
        if self.save_path:
            dir_path = 
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        sparsity_key = self._get_sparsity_key()
        loss_list = self.total_losses[sparsity_key]

        if len(loss_list) <= 1:
            torch.save(self.model.state_dict(), self.save_path)
            return True
        elif len(loss_list) > 1:
            if loss_list[-1] < min(loss_list[:-1]):
                torch.save(self.model.state_dict(), self.save_path)
                return True
        return False

    def create_pruning_module(self):
        # Loading sparse weights
        load_weights(self.model, self.save_path)

        # Creating pruning module
        pruning_module = PRUNER_DICT[self.pruning_type](
            self.model, self.epochs, self.target_sparsity, self.sparsity_scheduler
        )

        # Setting sparse mask
        zero_masks = {name: (param != 0).float() for name, param in self.model.named_parameters()}

        pruning_module.masks = zero_masks
        return pruning_module
        
        
def _to_float(x):
    return float(x.detach().cpu()) if isinstance(x, torch.nn.Parameter) or torch.is_tensor(x) else float(x)















