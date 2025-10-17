import os
import torch
import torch.nn as nn
import contextlib
from tqdm import tqdm
from copy import deepcopy
from torch.amp import autocast, GradScaler
from dataclasses import dataclass
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
from loss_functions import SupConLoss, NTXentLoss
from unstructured_pruning import *
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
    tau:                    float
    num_out_features:       int = 128       # Model embedded vector size
    image_size:             int = 32        # Image size for resizing
    epochs:                 int = 5         # Number of epochs to train
    scheduler_type:         str = None      # Scheduler type, e.g., "linear_with_warmup"
    trained_weights:        str = None      # Path to pretrained weights
    encoder_trained_weights:str = None      # Path to pretrained encoder weights (pt_model)
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
    pruning_module:         object = None   # Pruning module

    # Fine-tuning arguments
    ft_epochs:              int = 50        # Number of epochs to fine-tune pruned model

    # BaCP arguments
    lambdas:                list = None     # List of lambdas for BaCP
    learnable_lambdas:      bool = False    # Whether to learn lambdas or keep them static
    n_views:                int = 2         # Number of views for contrast  

    # DyReLU Phasing
    dyrelu_enabled:   bool = False
    
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

        self.model.train()
        self.model_pt.eval()
        self.model_ft.eval()

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
        self.training_acc = {}

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

    def finetune(self, run=None):
        for epoch in range(self.ft_epochs):
            # Training phase
            curr_epoch_str = f"Fine-tuning Epoch [{epoch+1}/{self.ft_epochs}]"

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



    def _run_train_epoch(self, epoch, desc=""):
        """Run a training epoch."""
        _handle_wanda_hooks(self)

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
                L_prc_sup = self.supervised_criterion(curr_emb, pt_emb, labels)
                L_prc_unsup = self.unsupervised_criterion(curr_emb, pt_emb)
                prc_loss = (L_prc_sup + L_prc_unsup) * self.lambda1

                # FiC Module
                L_fic_sup = self.supervised_criterion(curr_emb, ft_emb, labels)
                L_fic_unsup = self.unsupervised_criterion(curr_emb, ft_emb)
                fic_loss = (L_fic_sup + L_fic_unsup) * self.lambda2

                # SnC Module
                if len(self.snapshots) > 0:
                    L_snc_sup = torch.tensor(0.0, device=self.device)
                    L_snc_unsup = torch.tensor(0.0, device=self.device)
                    for ss_model in self.snapshots:
                        ss_model = ss_model.to(self.device)
                        with torch.no_grad():
                            ss_emb = ss_model(data2, return_emb=True)
                            ss_emb = ss_emb.logits if hasattr(ss_emb, 'logits') else ss_emb

                        L_snc_sup += self.supervised_criterion(curr_emb, ss_emb, labels)
                        L_snc_unsup += self.unsupervised_criterion(curr_emb, ss_emb)

                    snc_loss = ((L_snc_sup + L_snc_unsup) / len(self.snapshots)) * self.lambda3
                else:
                    snc_loss = torch.tensor(0.0, device=self.device)

                # CE Module
                ce_loss = nn.CrossEntropyLoss()(curr_logits, labels) * self.lambda4
                preds = curr_logits.max(1)[1]
                accuracy = ((preds == labels).sum() / labels.size(0)) * 100

            loss = prc_loss + snc_loss + fic_loss + ce_loss
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

        return metrics

    def _run_finetune_epochs(self, epoch, desc=""):
        ft_total_loss, ft_acc = 0, 0
        train_batchloader = tqdm(self.trainloader, desc=desc, leave=False) if self.enable_tqdm else self.trainloader
        val_batchloader = tqdm(self.valloader, desc=desc, leave=False) if self.enable_tqdm else self.valloader

        



    def _handle_snapshot_creation(self):
        if len(self.snapshots) < self.epochs - 1:
            ss_model = deepcopy(self.model).to('cpu').eval()
            self.snapshots.append(ss_model)
    
    def _update_metric_lists(self, metrics, accuracy=None):
        sparsity_key = _get_sparsity_key(self)
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
        loss_list = self.total_losses[sparsity_key]

        if len(loss_list) <= 1:
            torch.save(self.model.state_dict(), self.save_path)
            return True
        elif len(loss_list) > 1:
            if loss_list[-1] < min(loss_list[:-1]):
                torch.save(self.model.state_dict(), self.save_path)
                return True
        return False

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

    def get_pruner(self):
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















