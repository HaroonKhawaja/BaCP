import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy
from torch.amp import autocast, GradScaler
from datasets import get_dataset_config_names

from models import ClassificationNetwork, EncoderProjectionNetwork
from loss_fn import SupConLoss, NTXentLoss
from unstructured_pruning import PRUNER_DICT
from transformers import AutoTokenizer
from dataset_utils import get_glue_data, get_squad_data, get_cv_data, CV_DATASETS
from logger import Logger
from utils import *
from constants import *
from unstructured_pruning import *
import contextlib

def create_models_for_bacp(model_name, finetuned_weights, model_task, output_dimensions=128):
    pre_trained_model = EncoderProjectionNetwork(model_name, output_dimensions, model_task)
    pre_trained_model.to(get_device())

    # Current projection model
    current_model = deepcopy(pre_trained_model).to(get_device())

    # Fine-tuned projection model
    finetuned_model = deepcopy(pre_trained_model).to(get_device())
    load_weights(finetuned_model, finetuned_weights)

    return {
        "pt_model": pre_trained_model, 
        "curr_model": current_model, 
        "ft_model": finetuned_model
        }
    
class BaCPTrainingArgumentsCNN:
    def __init__(self, 
                 # Model configuration
                 model_name,
                 model_task,
                 finetuned_weights, # Need to have this for BaCP
                 num_classes=10,
                 
                 # Pruning parameters
                 pruning_type=None,
                 target_sparsity=0.0,
                 sparsity_scheduler="linear",
                 delta_t=500,
                 
                 # Training parameters
                 batch_size=32,
                 image_size=32,
                 learning_rate=0.01,
                 optimizer_type='sgd',
                 epochs=5,
                 pruning_epochs=None,
                 recovery_epochs=10,
                 
                 # Technical settings
                 log_epochs=True,
                 enable_tqdm=True,
                 enable_mixed_precision=True,
                 num_workers=24,
                 db=True):
        
        # Base model configuration
        self.model_type = 'cnn'
        self.model_name = model_name
        self.model_task = model_task
        self.finetuned_weights = finetuned_weights
        self.num_classes = num_classes
        
        # Training parameters
        self.batch_size = batch_size
        self.image_size = image_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.epochs = epochs
        self.pruning_epochs = epochs if pruning_epochs is None else pruning_epochs
        self.recovery_epochs = recovery_epochs
        self.recover = False
        
        # Technical settings
        self.enable_tqdm = enable_tqdm
        self.enable_mixed_precision = enable_mixed_precision
        self.device = get_device()
        self.scaler = GradScaler() if self.enable_mixed_precision else None
        
        # Initializing models, tokenizers, and other components
        self._initialize_models(model_name, finetuned_weights)

        # Initializing data loaders
        self._initialize_data_loaders(model_task, batch_size, image_size, num_workers)
                
        # Initializing pruner
        self._initialize_pruner(pruning_type, target_sparsity, sparsity_scheduler, delta_t)
        
        # Initializing contrastive learning components
        self._initialize_contrastive_learning()
        
        # Initializing paths and logger
        self._initialize_paths_and_logger(db, model_name, model_task, pruning_type, target_sparsity, log_epochs)
        
    def _initialize_models(self, model_name, finetuned_weights):
        """Initialize the models required for BaCP."""
        print("[BaCP TRAINER] Initializing models")
        models = create_models_for_bacp(model_name, finetuned_weights)
        self.current_model = models["curr_model"]
        self.pre_trained_model = models["pt_model"]
        self.finetuned_model = models["ft_model"]
        self.embedded_dim = self.current_model.embedding_dim
        
        # Initializing optimizer
        if self.optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(self.current_model.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == 'sgd':
            self.optimizer = optim.SGD(self.current_model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)
        else:
            raise ValueError(f"Invalid optimizer type: {self.optimizer_type}")
    
    def _initialize_pruner(self, pruning_type, target_sparsity, sparsity_scheduler, delta_t):
        """Initialize the pruner based on the pruning type."""
        self.pruning_type = pruning_type
        self.target_sparsity = target_sparsity
        self.sparsity_scheduler = sparsity_scheduler
        self.delta_t = int(min(0.5 * len(self.trainloader), delta_t))
        self.prune = pruning_type is not None and target_sparsity > 0

        if "wanda" in pruning_type:
            self.pruner = PRUNER_DICT[pruning_type](
                self.pruning_epochs, 
                target_sparsity,
                self.current_model, 
                self.sparsity_scheduler
            )
        else:
            self.pruner = PRUNER_DICT[pruning_type](
                self.pruning_epochs, 
                target_sparsity, 
                self.sparsity_scheduler
            )

        print(f"[BaCP TRAINER] Initializing pruner for {pruning_type}")
        print(f"[BaCP TRAINER] Pruning scheduler: {self.sparsity_scheduler}")
        if self.prune:
            print(f"[BaCP TRAINER] Pruning target sparsity: {target_sparsity}")
    
    def _initialize_data_loaders(self, model_task, batch_size, image_size, num_workers):
        """Initialize data loaders for the specified task."""
        print(f"[BaCP TRAINER] Initializing data loaders for {model_task}")
        if model_task in CV_DATASETS:
            data = get_cv_data(model_task, batch_size, learning_type='contrastive', size=image_size, num_workers=num_workers)
            if len(data) >= 2:
                self.trainloader = data["trainloader"]
                self.valloader = data["valloader"]
                self.testloader = data.get("testloader", None)
            else:
                raise ValueError(f"Expected at least trainloader and valloader for {model_task}, got {len(data)} loaders")
        else:
            raise ValueError(f"{model_task} dot not exist in cv models. Existing datasets are: {CV_DATASETS}")

    def _initialize_contrastive_learning(self):
        """Initialize contrastive loss functions."""
        self.n_views = 2
        self.temperature = TEMP
        self.base_temperature = BASE_TEMP
        
        self.supervised_loss = SupConLoss(self.n_views, self.temperature, self.base_temperature, self.batch_size)
        self.unsupervised_loss = NTXentLoss(self.n_views, self.temperature)
    
    def _initialize_paths_and_logger(self, db, model_name, model_task, pruning_type, target_sparsity, log_epochs):
        """Initialize weight paths and logger."""
        dir_name = "/dbfs" if db else "."
        base_path = f"{dir_name}/research/{model_name}/{model_task}/{model_name}_{pruning_type}_{target_sparsity}_bacp"
        
        self.cm_save_path = f"{base_path}_cm.pt"
        self.pm_save_path = f"{base_path}_pm.pt"
        self.fm_save_path = f"{base_path}_fm.pt"
        
        self.logger = Logger(model_name, f"bacp_pruning/sparsity_{self.target_sparsity}") if log_epochs else None
        print(f"[BaCP TRAINER] Saving model checkpoints to {base_path}_cm/pm/fm.pt")

class BaCPTrainingArgumentsLLM:
    def __init__(self, 
                 # Model configuration
                 model_name,
                 model_task,
                 finetuned_weights, # Need to have this for BaCP
                 num_classes=2,
                 
                 # Pruning parameters
                 pruning_type=None,
                 target_sparsity=0.0,
                 sparsity_scheduler="linear",
                 delta_t=500,
                 
                 # Training parameters
                 batch_size=32,
                 learning_rate=2e-5,
                 optimizer_type='adamw',
                 epochs=5,
                 pruning_epochs=None,
                 recovery_epochs=10,
                 
                 # Technical settings
                 log_epochs=True,
                 enable_tqdm=True,
                 enable_mixed_precision=True,
                 num_workers=24,
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
        self.optimizer_type = optimizer_type
        self.epochs = epochs
        self.pruning_epochs = epochs if pruning_epochs is None else pruning_epochs
        self.recovery_epochs = recovery_epochs
        self.recover = False
        
        # Technical settings
        self.enable_tqdm = enable_tqdm
        self.enable_mixed_precision = enable_mixed_precision
        self.device = get_device()
        self.scaler = GradScaler() if self.enable_mixed_precision else None
        
        # Initializing models, tokenizers, and other components
        self._initialize_models(model_name, finetuned_weights, model_task)
        
        # Initializing data loaders
        self._initialize_data_loaders(model_task, batch_size, num_workers)

        # Initializing pruner
        self._initialize_pruner(pruning_type, target_sparsity, sparsity_scheduler, delta_t)
        
        # Initializing contrastive learning components
        self._initialize_contrastive_learning()
        
        # Initializing paths and logger
        self._initialize_paths_and_logger(db, model_name, model_task, pruning_type, target_sparsity, log_epochs)
        
    def _initialize_models(self, model_name, finetuned_weights, model_task):
        """Initialize the models required for BaCP."""
        print("[BaCP TRAINER] Initializing models")
        models = create_models_for_bacp(model_name, finetuned_weights, model_task)
        self.current_model = models["curr_model"]
        self.pre_trained_model = models["pt_model"]
        self.finetuned_model = models["ft_model"]
        self.embedded_dim = self.current_model.embedding_dim
        
        # Initializing tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initializing optimizer
        if self.optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(self.current_model.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == 'sgd':
            self.optimizer = optim.SGD(self.current_model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)
        else:
            raise ValueError(f"Invalid optimizer type: {self.optimizer_type}")
    
    def _initialize_pruner(self, pruning_type, target_sparsity, sparsity_scheduler, delta_t):
        """Initialize the pruner based on the pruning type."""
        self.pruning_type = pruning_type
        self.target_sparsity = target_sparsity
        self.sparsity_scheduler = sparsity_scheduler
        self.delta_t = int(min(0.5 * len(self.trainloader), delta_t))
        self.prune = pruning_type is not None and target_sparsity > 0

        if "wanda" in pruning_type:
            self.pruner = PRUNER_DICT[pruning_type](
                self.pruning_epochs, 
                target_sparsity,
                self.current_model, 
                self.sparsity_scheduler
            )
        else:
            self.pruner = PRUNER_DICT[pruning_type](
                self.pruning_epochs, 
                target_sparsity, 
                self.sparsity_scheduler
            )

        print(f"[BaCP TRAINER] Initializing pruner for {pruning_type}")
        print(f"[BaCP TRAINER] Pruning scheduler: {self.sparsity_scheduler}")
        if self.prune:
            print(f"[BaCP TRAINER] Pruning target sparsity: {target_sparsity}")
    
    def _initialize_data_loaders(self, model_task, batch_size, num_workers):
        """Initialize data loaders for the specified task."""
        print(f"[BaCP TRAINER] Initializing data loaders for {model_task}")
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
    
    def _initialize_contrastive_learning(self):
        """Initialize contrastive loss functions."""
        self.n_views = 2
        self.temperature = TEMP
        self.base_temperature = BASE_TEMP
        
        self.supervised_loss = SupConLoss(self.n_views, self.temperature, self.base_temperature, self.batch_size)
        self.unsupervised_loss = NTXentLoss(self.n_views, self.temperature)
    
    def _initialize_paths_and_logger(self, db, model_name, model_task, pruning_type, target_sparsity, log_epochs):
        """Initialize weight paths and logger."""
        dir_name = "/dbfs" if db else "."
        base_path = f"{dir_name}/research/{model_name}/{model_task}/{model_name}_{pruning_type}_{target_sparsity}_bacp"
        
        self.cm_save_path = f"{base_path}_cm.pt"
        self.pm_save_path = f"{base_path}_pm.pt"
        self.fm_save_path = f"{base_path}_fm.pt"
        
        self.logger = Logger(model_name, f"bacp_pruning/sparsity_{self.target_sparsity}") if log_epochs else None
        print(f"[BaCP TRAINER] Saving model checkpoints to {base_path}_cm/pm/fm.pt")

class BaCPTrainer:
    def __init__(self, bacp_training_args, lambdas=[0.25, 0.25, 0.25, 0.25]):
        for key, value in vars(bacp_training_args).items():
            setattr(self, key, value)

        self.classification_head = nn.Linear(128, self.num_classes).to(self.device)
        # self.classification_head = nn.Linear(self.embedded_dim, self.num_classes).to(self.device)
        self._initialize_lambdas(lambdas)
        self._initialize_metric_lists()
        self._initialize_log_parameters()
        self.snapshots = []

    def _initialize_lambdas(self, lambdas=[0.25, 0.25, 0.25, 0.25]):
        if isinstance(lambdas, list) and len(lambdas) == 4:
            # static parameters
            self.lambda1, self.lambda2, self.lambda3, self.lambda4 = lambdas
        elif isinstance(lambdas, float):
            # learnable parameters
            self.lambda1 = nn.Parameter(torch.tensor(lambdas, requires_grad=True)).to(self.device)
            self.lambda2 = nn.Parameter(torch.tensor(lambdas, requires_grad=True)).to(self.device)
            self.lambda3 = nn.Parameter(torch.tensor(lambdas, requires_grad=True)).to(self.device)
            self.lambda4 = nn.Parameter(torch.tensor(lambdas, requires_grad=True)).to(self.device)
        else:
            # Learnable parameters - Uniform distribution
            self.lambda1 = nn.Parameter(torch.rand(1, requires_grad=True)).to(self.device)
            self.lambda2 = nn.Parameter(torch.rand(1, requires_grad=True)).to(self.device)
            self.lambda3 = nn.Parameter(torch.rand(1, requires_grad=True)).to(self.device)
            self.lambda4 = nn.Parameter(torch.rand(1, requires_grad=True)).to(self.device)

        if isinstance(self.lambda1, nn.Parameter):
            self.optimizer.add_param_group({'params': [self.lambda1, self.lambda2, self.lambda3, self.lambda4]})

    def _initialize_metric_lists(self):
        self.avg_losses = {}
        self.avg_PrC = {}
        self.avg_SnC = {}
        self.avg_FiC = {}
        self.avg_CE = {}
    
    def _get_sparsity_key(self):
        return round(check_model_sparsity(self.current_model), 4)

    def _initialize_log_parameters(self):
        logger_params = {
            # Model information
            'model_type': self.model_type,
            'model_name': self.model_name,
            'model_task': self.model_task if hasattr(self, 'model_task') else None,
            
            # Training parameters
            'epochs': self.epochs,
            'pruning_epochs': self.pruning_epochs,
            'recovery_epochs': self.recovery_epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            
            # Pruning parameters
            'pruning_type': self.pruning_type if hasattr(self, 'pruning_type') else None,
            'target_sparsity': self.target_sparsity,
            
            # Loss parameters
            'n_views': self.n_views,
            'temperature': self.temperature,
            'base_temperature': self.base_temperature,
            'num_classes': self.num_classes,
            
            # Lambda values
            'lambda1': self.lambda1.item() if isinstance(self.lambda1, nn.Parameter) else self.lambda1,
            'lambda2': self.lambda2.item() if isinstance(self.lambda2, nn.Parameter) else self.lambda2,
            'lambda3': self.lambda3.item() if isinstance(self.lambda3, nn.Parameter) else self.lambda3,
            'lambda4': self.lambda4.item() if isinstance(self.lambda4, nn.Parameter) else self.lambda4,
            
            # Model architecture
            'embedding_dim': self.embedded_dim if hasattr(self, 'embedded_dim') else None,
            
            # Paths
            'save_path': {
                'current_model': self.cm_save_path,
                'pretrained_model': self.pm_save_path,
                'finetuned_model': self.fm_save_path
            }
        }
        self.logger_params = {k: v for k, v in logger_params.items() if v is not None}
    
    def train(self):
        if hasattr(self, 'logger') and self.logger is not None:
            self.logger.create_log()
            self.logger.log_hyperparameters(self.logger_params)
        
        self.current_model.train()
        self.pre_trained_model.eval()
        self.finetuned_model.eval()

        for epoch in range(self.epochs):
            # Training phase
            desc = f"Training Epoch ({self.model_type}) [{epoch+1}/{self.epochs}]"
            self.train_epoch(desc)
            
            if len(self.snapshots) < self.pruning_epochs:
                state = deepcopy(self.current_model.state_dict())
                snapshot_model = self._create_model_from_snapshot(state)
                self.snapshots.append(snapshot_model)

            # Printing statistics
            sparsity = self._get_sparsity_key()
            info = (
                f"Epoch [{epoch+1}/{self.epochs}]: "
                f"Avg Total Loss: {self.avg_losses[sparsity][-1]:.4f} | "
                f"Avg PrC Loss: {self.avg_PrC[sparsity][-1]:.4f} | "
                f"Avg SnC Loss: {self.avg_SnC[sparsity][-1]:.4f} | "
                f"Avg FiC Loss: {self.avg_FiC[sparsity][-1]:.4f} | "
                f"Avg CE Loss: {self.avg_CE[sparsity][-1]:.4f} | "
                f"Model Sparsity: {sparsity}\n"
            )           
            print(info)

            # Logging information if logger provided
            if self.logger is not None:
                self.logger.log_epochs(info)

            # Saving model
            if self._handle_save(epoch):
                print(f"[BaCP] weights saved!")

            if self.pruner is not None:
                self.retrain()                    
                self.pruner.ratio_step()
            
    def retrain(self):
        self.recover = True
        for epoch in range(self.recovery_epochs):
            # Training phase
            desc = f"Retraining epoch [{epoch+1}/{self.recovery_epochs}]"
            self.train_epoch(desc)

            # Printing statistics
            sparsity = self._get_sparsity_key()
            info = (
                f"Retraining Epoch [{epoch+1}/{self.recovery_epochs}]: "
                f"Avg Total Loss: {self.avg_losses[sparsity][-1]:.4f} | "
                f"Avg PrC Loss: {self.avg_PrC[sparsity][-1]:.4f} | "
                f"Avg SnC Loss: {self.avg_SnC[sparsity][-1]:.4f} | "
                f"Avg FiC Loss: {self.avg_FiC[sparsity][-1]:.4f} | "
                f"Avg CE Loss: {self.avg_CE[sparsity][-1]:.4f} | "
                f"Model Sparsity: {sparsity}\n"
            )           
            print(info)

            # Logging information if logger provided
            if self.logger is not None:
                self.logger.log_epochs(info)
            
            # Saving model
            if self._handle_save(epoch):
                print(f"[BaCP] weights saved!")

        self.recover = False

    def train_epoch(self, desc):
        if (self.prune and self.pruner) and (hasattr(self.pruner, 'is_wanda') and self.pruner.is_wanda) and (not self.recover):
            self.pruner.register_hooks(self.current_model)

        losses, prc_losses, snc_losses, fic_losses, ce_losses, total = 0, 0, 0, 0, 0, 0

        batchloader = tqdm(self.trainloader, desc=desc) if self.enable_tqdm else self.trainloader
        for batch_idx, batch_data in enumerate(batchloader):
            if self.model_type == 'cnn':
                images, labels = batch_data
                images1, images2 = images 
                batch = {
                    'data1': images1.to(self.device),
                    'data2': images2.to(self.device),
                    'label': labels.to(self.device)
                }
            else:  # LLM
                batch = {k: v.to(self.device) for k, v in batch_data.items()}

            labels = batch["label"]

            self.optimizer.zero_grad()

            with autocast(device_type=self.device) if self.enable_mixed_precision else contextlib.nullcontext():
                current_embeddings, current_logits = self._get_embeddings_and_logits(
                    batch if self.model_type == 'llm' else batch['data1']
                )

                with torch.no_grad():
                    if self.model_type == 'llm':
                        pretrained_embeddings = self.pre_trained_model(batch).logits
                        finetuned_embeddings = self.finetuned_model(batch).logits
                    elif self.model_type == 'cnn':
                        pretrained_embeddings = self.pre_trained_model(batch['data2'])
                        finetuned_embeddings = self.finetuned_model(batch['data2'])
                
                # PrC Module
                sup_features_prc = torch.cat((current_embeddings, pretrained_embeddings))
                L_prc_sup = self._supervised_criterion(sup_features_prc, labels)
                L_prc_unsup = self._unsupervised_criterion(current_embeddings, pretrained_embeddings)
                L_prc_total = L_prc_sup + L_prc_unsup

                # CE Loss
                CE_loss = nn.CrossEntropyLoss()(current_logits, labels)

                # FiC Module
                sup_features_fic = torch.cat((current_embeddings, finetuned_embeddings))
                L_fic_sup = self._supervised_criterion(sup_features_fic, labels)
                L_fic_unsup = self._unsupervised_criterion(current_embeddings, finetuned_embeddings)
                L_fic_total = L_fic_sup + L_fic_unsup

                # SnC Module
                L_snc_sup = torch.tensor(0.0, device=get_device())
                L_snc_unsup = torch.tensor(0.0, device=get_device())
                for snapshot_model in self.snapshots:
                    with torch.no_grad():
                        snapshot_embeddings = snapshot_model(batch).logits if self.model_type == 'llm' else snapshot_model(batch['data2'])
                    sup_features_snc = torch.cat((current_embeddings, snapshot_embeddings))
                    L_snc_sup += self._supervised_criterion(sup_features_snc, labels)
                    L_snc_unsup += self._unsupervised_criterion(current_embeddings, snapshot_embeddings)
                L_snc_total = L_snc_sup + L_snc_unsup

            # Total loss calculation
            total_loss = (
                self.lambda1 * CE_loss +
                self.lambda2 * L_prc_total +
                self.lambda3 * L_snc_total +
                self.lambda4 * L_fic_total
                )

            self._handle_optimizer_and_pruning(total_loss, batch_idx)

            losses += total_loss.item()
            prc_losses += L_prc_total.item()
            snc_losses += L_snc_total.item()
            fic_losses += L_fic_total.item()
            ce_losses += CE_loss.item()
            total += 1

            if self.enable_tqdm:
                batchloader.set_postfix({
                    'Loss': total_loss.item(), 
                    'PrC Loss': L_prc_total.item(), 
                    'SnC Loss': L_snc_total.item(), 
                    'FiC Loss': L_fic_total.item(), 
                    'CE Loss': CE_loss.item()
                })
        
        self._update_losses(losses, prc_losses, snc_losses, fic_losses, ce_losses, total)

    def _update_losses(self, losses, prc_losses, snc_losses, fic_losses, ce_losses, total):
        sparsity_key = self._get_sparsity_key()
        if sparsity_key not in self.avg_losses:
            self.avg_losses[sparsity_key] = []
            self.avg_PrC[sparsity_key] = []
            self.avg_SnC[sparsity_key] = []
            self.avg_FiC[sparsity_key] = []
            self.avg_CE[sparsity_key] = []
        
        self.avg_losses[sparsity_key].append(losses / total)
        self.avg_PrC[sparsity_key].append(prc_losses / total)
        self.avg_SnC[sparsity_key].append(snc_losses / total)
        self.avg_FiC[sparsity_key].append(fic_losses / total)
        self.avg_CE[sparsity_key].append(ce_losses / total)

    # def _get_embeddings_and_logits(self, data_batch):
    #     if self.model_type == 'llm':
    #         outputs = self.current_model(data_batch)
    #         intermediate_embeddings = F.normalize(outputs.hidden_states[-1][:, 0, :], dim=1)
    #         logits = self.classification_head(intermediate_embeddings)
    #         embeddings = outputs.logits
    #         return embeddings, logits
    #     else:
    #         if self.model_name in ['vgg11', 'vgg19']:
    #             x = self.current_model.model.features(data_batch)
    #             x = self.current_model.model.avgpool(x)
    #             x = x.reshape(x.shape[0], -1)

    #             for i in range(6):
    #                 x = self.current_model.model.classifier[i](x)
            
    #         elif self.model_name in ['vitb16', 'vitl16']:
    #             x = self.current_model.model.conv_proj(data_batch)
    #             class_token = self.current_model.model.class_token.expand(self.batch_size, -1, -1)
    #             x = torch.cat((class_token, x.flatten(2).transpose(1, 2)), axis=1)
    #             x = self.current_model.model.encoder(x)
    #             x = x[:, 0, :]

    #         else:
    #             x = self.current_model.model.conv1(data_batch)
    #             x = self.current_model.model.bn1(x)
    #             x = self.current_model.model.relu(x)
    #             x = self.current_model.model.maxpool(x)
    #             x = self.current_model.model.layer1(x)
    #             x = self.current_model.model.layer2(x)
    #             x = self.current_model.model.layer3(x)
    #             x = self.current_model.model.layer4(x)
    #             x = self.current_model.model.avgpool(x)
    #             x = x.reshape(x.shape[0], -1)

    #         logits = self.classification_head(x)
    #         if self.model_name in ['vgg11', 'vgg19']:
    #             embeddings = self.current_model.model.classifier[-1](x)
    #             embeddings = F.normalize(embeddings, dim=1)
    #         elif self.model_name in ['vitb16', 'vitl16']:
    #             embeddings = self.current_model.model.heads(x)  
    #             embeddings = F.normalize(embeddings, dim=1)
    #         else:
    #             embeddings = self.current_model.model.fc(x)
    #             embeddings = F.normalize(embeddings, dim=1)
    #         return embeddings, logits
    
    def _get_embeddings_and_logits(self, data_batch):
        if self.model_type == 'llm':
            outputs = self.current_model(data_batch)
            embeddings = outputs.logits
            logits = self.classification_head(embeddings)
        else:
            embeddings = self.current_model(data_batch)
            logits = self.classification_head(embeddings)

        return embeddings, logits

    def _handle_optimizer_and_pruning(self, total_loss, batch_idx):
        """Handle backpropagation, pruning, and weight update in a single step."""
        if self.enable_mixed_precision:
            self.scaler.scale(total_loss).backward()
            self._apply_pruning(batch_idx)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            self._apply_pruning(batch_idx)
            self.optimizer.step()

        if self.prune and self.pruner is not None:
            self.pruner.apply_mask(self.current_model)

    def _apply_pruning(self, step):
        """Apply pruning based on the pruning configuration."""
        if (not self.prune and self.pruner is None) or self.recover:
            return
        
        if self.pruner.sparsity_scheduler == "linear":
            if step == self.delta_t:
                self.pruner.prune(self.current_model)
            
        elif self.pruner.sparsity_scheduler == "cubic":
            if step >= 0 and step % self.delta_t == 0:
                current_sparsity = check_model_sparsity(self.current_model)
                self.pruner.cubic_scheduler(step, 0, self.delta_t, current_sparsity)
                self.pruner.prune(self.current_model)

    def _handle_save(self, epoch):
        if self.cm_save_path:
            os.makedirs(os.path.dirname(self.cm_save_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.pm_save_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.fm_save_path), exist_ok=True)

        sparsity_key = self._get_sparsity_key()
        loss_list = self.avg_losses[sparsity_key]

        save = False
        if len(loss_list) == 1:
            save = True
        elif len(loss_list) > 2 and loss_list[-1] < min(loss_list[1:-1]):
            save = True
        
        if save:
            torch.save(self.current_model.state_dict(), self.cm_save_path)
            torch.save(self.pre_trained_model.state_dict(), self.pm_save_path)
            torch.save(self.finetuned_model.state_dict(), self.fm_save_path)
        return save
    
    def _create_model_from_snapshot(self, snapshot_state):
        model = deepcopy(self.current_model)
        model.load_state_dict(snapshot_state)
        model.to(self.device)
        model.eval()
        return model
    
    def _supervised_criterion(self, features, labels):
        loss = self.supervised_loss(features, labels)
        return loss

    def _unsupervised_criterion(self, features1, features2):
        loss = self.unsupervised_loss(features1, features2)
        loss += self.unsupervised_loss(features2, features1)
        return (loss)/2
    
    def generate_mask_from_model(self):
        load_weights(self.current_model, self.cm_save_path)
        zero_masks = {}
        for name, param in self.current_model.named_parameters():
            mask = (param != 0).float()
            zero_masks[name] = mask
        self.set_mask(zero_masks)
        print("[BaCP TRAINER] Mask generated from current model.")
    
    def set_mask(self, masks):
        if self.pruner is not None:
            self.pruner.masks = masks

    def get_pruner(self):
        return self.pruner

    def get_metrics(self):
        return self.avg_losses, self.avg_PrC, self.avg_SnC, self.avg_FiC, self.avg_CE
