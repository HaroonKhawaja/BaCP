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
from dataset_utils import get_glue_data
from logger import Logger
from utils import *
from constants import *
from unstructured_pruning import *
import contextlib

def create_models_for_bacp(model_name, finetuned_weights, output_dimensions=128):
    pre_trained_model = EncoderProjectionNetwork(model_name, output_dimensions)
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
                 epochs=10,
                 pruning_epochs=None,
                 recovery_epochs=5,
                 
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
        
        # Initializing pruner
        self._initialize_pruner(pruning_type, target_sparsity, sparsity_scheduler, delta_t)
        
        # Initializing data loaders
        self._initialize_data_loaders(model_task, batch_size, num_workers)
        
        # Initializing contrastive learning components
        self._initialize_contrastive_learning()
        
        # Initializing paths and logger
        self._initialize_paths_and_logger(db, model_name, model_task, pruning_type, target_sparsity, log_epochs)
        
    
    def _initialize_models(self, model_name, finetuned_weights):
        """Initialize the models required for BaCP."""
        print("[BaCP TRAINER] Initializing models...")
        models = create_models_for_bacp(model_name, finetuned_weights)
        self.current_model = models["curr_model"]
        self.pre_trained_model = models["pt_model"]
        self.finetuned_model = models["ft_model"]
        self.embedded_dim = self.current_model.embedding_dim
        
        # Initializing tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initializing classification head and optimizer
        self.classification_head = nn.Linear(self.embedded_dim, self.num_classes).to(self.device)
        self.optimizer = optim.AdamW(self.current_model.parameters(), lr=self.learning_rate)
    
    def _initialize_pruner(self, pruning_type, target_sparsity, sparsity_scheduler, delta_t):
        """Initialize the pruner based on the pruning type."""
        self.pruning_type = pruning_type
        self.target_sparsity = target_sparsity
        self.sparsity_scheduler = sparsity_scheduler
        self.delta_t = delta_t
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
            data = get_glue_data(self.model_name, self.tokenizer, model_task, batch_size, num_workers)
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
    def __init__(self, bacp_training_args):
        for key, value in vars(bacp_training_args).items():
            setattr(self, key, value)

        self._initialize_lambdas()
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
        self.avg_losses = [[0 for _ in range(self.recovery_epochs)] for _ in range(self.epochs)]
        self.avg_PrC = [[0 for _ in range(self.recovery_epochs)] for _ in range(self.epochs)]
        self.avg_SnC = [[0 for _ in range(self.recovery_epochs)] for _ in range(self.epochs)]
        self.avg_FiC = [[0 for _ in range(self.recovery_epochs)] for _ in range(self.epochs)]
        self.avg_CE = [[0 for _ in range(self.recovery_epochs)] for _ in range(self.epochs)]

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
        self.pre_trained_model.train()
        self.finetuned_model.train()

        for epoch in range(self.epochs):
            self.current_epoch = epoch

            # Training phase
            if self.model_type == 'llm':
                desc = f"Training Epoch ({self.model_type}) [{epoch+1}/{self.epochs}]"
                self.train_epoch_llm(desc)
            else:
                self.train_epoch_cnn()
            
            if len(self.snapshots) < self.pruning_epochs:
                state = deepcopy(self.current_model.state_dict())
                snapshot_model = self.create_model_from_snapshot(state)
                self.snapshots.append(snapshot_model)

            # Printing statistics
            sparsity = check_model_sparsity(self.current_model) 
            info = (
                f"Epoch [{epoch+1}/{self.epochs}]: "
                f"Avg Total Loss: {self.avg_losses[self.current_epoch][-1]:.4f} | "
                f"Avg PrC Loss: {self.avg_PrC[self.current_epoch][-1]:.4f} | "
                f"Avg SnC Loss: {self.avg_SnC[self.current_epoch][-1]:.4f} | "
                f"Avg FiC Loss: {self.avg_FiC[self.current_epoch][-1]:.4f} | "
                f"Avg CE Loss: {self.avg_CE[self.current_epoch][-1]:.4f} | "
                f"Model Sparsity: {sparsity:.3f}\n"
            )           
            print(info)

            # Logging information if logger provided
            if self.logger is not None:
                self.logger.log_epochs(info)

            # Saving model
            if self.handle_save(epoch):
                print(f"[BaCP] weights saved!")
            else:
                print("[BaCP] weights not saved!")

            if self.pruner is not None:
                self.retrain()                    
                self.pruner.ratio_step()
            
    def retrain(self):
        self.recover = True
        for epoch in range(self.recovery_epochs):
            # Training phase
            if self.model_type == 'llm':
                desc = f"Retraining epoch [{epoch+1}/{self.recovery_epochs}]"
                self.train_epoch_llm(desc)
            else:
                self.train_epoch_cnn()

            # Printing statistics
            sparsity = check_model_sparsity(self.current_model) 
            info = (
                f"Retraining Epoch [{epoch+1}/{self.recovery_epochs}]: "
                f"Avg Total Loss: {self.avg_losses[self.current_epoch][-1]:.4f} | "
                f"Avg PrC Loss: {self.avg_PrC[self.current_epoch][-1]:.4f} | "
                f"Avg SnC Loss: {self.avg_SnC[self.current_epoch][-1]:.4f} | "
                f"Avg FiC Loss: {self.avg_FiC[self.current_epoch][-1]:.4f} | "
                f"Avg CE Loss: {self.avg_CE[self.current_epoch][-1]:.4f} | "
                f"Model Sparsity: {sparsity:.3f}"
            )           
            print(info)

            # Logging information if logger provided
            if self.logger is not None:
                self.logger.log_epochs(info)

        self.recover = False

    def train_epoch_llm(self, desc):
        if (self.prune and self.pruner) and (hasattr(self.pruner, 'is_wanda') and self.pruner.is_wanda) and (not self.recover):
            self.pruner.register_hooks(self.current_model)
        
        self.total_loss = 0.0
        losses, prc_losses, snc_losses, fic_losses, ce_losses = [], [], [], [], []
        
        batchloader = tqdm(self.trainloader, desc=desc) if self.enable_tqdm else self.trainloader
        for batch_idx, batch in enumerate(batchloader):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            labels = batch["label"]

            self.optimizer.zero_grad()

            with autocast(device_type=self.device) if self.enable_mixed_precision else contextlib.nullcontext():
                current_embeddings, current_logits = self.get_embeddings_and_logits(batch)
                with torch.no_grad():
                    pretrained_embeddings = self.pre_trained_model(batch).logits
                    finetuned_embeddings = self.finetuned_model(batch).logits

                # PrC Module
                sup_features_prc = torch.cat((current_embeddings, pretrained_embeddings))
                L_prc_sup = self.supervised_criterion(sup_features_prc, labels)
                L_prc_unsup = self.unsupervised_criterion(current_embeddings, pretrained_embeddings)
                L_prc_total = L_prc_sup + L_prc_unsup

                # CE Loss
                CE_loss = nn.CrossEntropyLoss()(current_logits, labels)

                # FiC Module
                sup_features_fic = torch.cat((current_embeddings, finetuned_embeddings))
                L_fic_sup = self.supervised_criterion(sup_features_fic, labels)
                L_fic_unsup = self.unsupervised_criterion(current_embeddings, finetuned_embeddings)
                L_fic_total = L_fic_sup + L_fic_unsup

                # SnC Module
                L_snc_sup = torch.tensor(0.0, device=get_device())
                L_snc_unsup = torch.tensor(0.0, device=get_device())
                for snapshot_model in self.snapshots:
                    with torch.no_grad():
                        snapshot_embeddings = snapshot_model(batch).logits

                    sup_features_snc = torch.cat((current_embeddings, snapshot_embeddings))
                    L_snc_sup += self.supervised_criterion(sup_features_snc, labels)
                    L_snc_unsup += self.unsupervised_criterion(current_embeddings, snapshot_embeddings)
                L_snc_total = L_snc_sup + L_snc_unsup

            # Total loss calculation
            self.total_loss = (
                self.lambda1 * CE_loss +
                self.lambda2 * L_prc_total +
                self.lambda3 * L_snc_total +
                self.lambda4 * L_fic_total
                )

            self.handle_optimizer_and_pruning(batch_idx)

            losses.append(self.total_loss.item())
            prc_losses.append(L_prc_total.item())
            snc_losses.append(L_snc_total.item())
            fic_losses.append(L_fic_total.item())
            ce_losses.append(CE_loss.item())

            if self.enable_tqdm:
                batchloader.set_postfix({'PrC Loss': L_prc_total.item(), 'SnC Loss': L_snc_total.item(), 'FiC Loss': L_fic_total.item(), 'CE Loss': CE_loss.item(), 'Loss': self.total_loss.item()})

        self.avg_losses[self.current_epoch].append(sum(losses) / len(losses))          
        self.avg_PrC[self.current_epoch].append(sum(prc_losses) / len(prc_losses))
        self.avg_SnC[self.current_epoch].append(sum(snc_losses) / len(snc_losses))
        self.avg_FiC[self.current_epoch].append(sum(fic_losses) / len(fic_losses))
        self.avg_CE[self.current_epoch].append(sum(ce_losses) / len(ce_losses))

    def get_embeddings_and_logits(self, data_batch):
        if self.model_type == 'llm':
            outputs = self.current_model(data_batch)
            intermediate_embeddings = F.normalize(outputs.hidden_states[-1][:, 0, :], dim=1)
            logits = self.classification_head(intermediate_embeddings)
            embeddings = outputs.logits
            return embeddings, logits
        else:
            if self.model_name in ['vgg11', 'vgg19']:
                x = self.current_model.backbone.features(data_batch)
                x = self.current_model.backbone.avgpool(x)
                x = x.reshape(x.shape[0], -1)

                for i in range(6):
                    x = self.current_model.backbone.classifier[i](x)
            
            elif self.model_name in ['vitb16', 'vitl16']:
                x = self.current_model.backbone.conv_proj(data_batch)
                class_token = self.current_model.backbone.class_token.expand(self.batch_size, -1, -1)
                x = torch.cat((class_token, x.flatten(2).transpose(1, 2)), axis=1)
                x = self.current_model.backbone.encoder(x)
                x = x[:, 0, :]

            else:
                x = self.current_model.backbone.conv1(data_batch)
                x = self.current_model.backbone.bn1(x)
                x = self.current_model.backbone.relu(x)
                x = self.current_model.backbone.maxpool(x)
                x = self.current_model.backbone.layer1(x)
                x = self.current_model.backbone.layer2(x)
                x = self.current_model.backbone.layer3(x)
                x = self.current_model.backbone.layer4(x)
                x = self.current_model.backbone.avgpool(x)
                x = x.reshape(x.shape[0], -1)

            logits = self.classification_head(x)
            if self.model_name in ['vgg11', 'vgg19']:
                embeddings = self.current_model.backbone.classifier[-1](x)
                embeddings = F.normalize(embeddings, dim=1)
            elif self.model_name in ['vitb16', 'vitl16']:
                embeddings = self.current_model.backbone.heads(x)  
                embeddings = F.normalize(embeddings, dim=1)
            else:
                embeddings = self.current_model.backbone.fc(x)
                embeddings = F.normalize(embeddings, dim=1)
            return embeddings, logits

    def handle_optimizer_and_pruning(self, batch_idx):
        """Handle backpropagation, pruning, and weight update in a single step."""
        if self.enable_mixed_precision:
            self.scaler.scale(self.total_loss).backward()
            self.apply_pruning(batch_idx)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.total_loss.backward()
            self.apply_pruning(batch_idx)
            self.optimizer.step()

        if self.prune and self.pruner is not None:
            self.pruner.apply_mask(self.current_model)

    def apply_pruning(self, step):
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

    def handle_save(self, epoch):
        if epoch == 0:
            os.makedirs(os.path.dirname(self.cm_save_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.pm_save_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.fm_save_path), exist_ok=True)
            
            torch.save(self.current_model.state_dict(), self.cm_save_path)
            torch.save(self.pre_trained_model.state_dict(), self.pm_save_path)
            torch.save(self.finetuned_model.state_dict(), self.fm_save_path)
            return True
        else:
            current_epoch_losses = self.avg_losses[self.current_epoch]
            if len(current_epoch_losses) == 0:
                torch.save(self.current_model.state_dict(), self.cm_save_path)
                torch.save(self.pre_trained_model.state_dict(), self.pm_save_path)
                torch.save(self.finetuned_model.state_dict(), self.fm_save_path)
                return True
            elif current_epoch_losses[-1] < min(current_epoch_losses[:-1]):
                torch.save(self.current_model.state_dict(), self.cm_save_path)
                torch.save(self.pre_trained_model.state_dict(), self.pm_save_path)
                torch.save(self.finetuned_model.state_dict(), self.fm_save_path)
                return True
            else:
                return False
    
    def create_model_from_snapshot(self, snapshot_state):
        model = deepcopy(self.current_model)
        model.load_state_dict(snapshot_state)
        model.to(self.device)
        model.eval()
        return model
    
    def generate_mask_from_model(self):
        load_weights(self.current_model, self.cm_save_path)
        zero_masks = {}
        for name, param in self.current_model.named_parameters():
            mask = (param != 0).float()
            zero_masks[name] = mask
        self.set_mask(zero_masks)
        print("[BaCP TRAINER] Mask generated from current model.")

    def supervised_criterion(self, features, labels):
        loss = self.supervised_loss(features, labels)
        return loss

    def unsupervised_criterion(self, features1, features2):
        loss = self.unsupervised_loss(features1, features2)
        loss += self.unsupervised_loss(features2, features1)
        return (loss)/2
    
    def set_mask(self, masks):
        if self.pruner is not None:
            self.pruner.masks = masks

    def get_pruner(self):
        return self.pruner

    def get_metrics(self):
        return self.avg_losses, self.avg_PrC, self.avg_SnC, self.avg_FiC, self.avg_CE

class BaCPLearner(object):
    def __init__(self, current_model, pre_trained_model, finetuned_model, config):
        self.device = get_device()
        
        # Models
        self.current_model = current_model.to(self.device)
        self.pre_trained_model = pre_trained_model.to(self.device)
        self.finetuned_model = finetuned_model.to(self.device)
        
        # Hyper-parameters
        # Required configuration parameters
        self.model_name = config['model_name']
        self.optimizer = config['optimizer']
        self.batch_size = config['batch_size']  
        self.logger = config['logger']                    
        self.target_sparsity = config['target_sparsity']   
        self.criterion = config['criterion']             

        # Optional configuration parameters with defaults
        self.scheduler = config.get('scheduler', None)    
        self.n_views = config.get('n_views', 2)
        self.epochs = config.get('epochs', BACP_EPOCHS)
        self.recovery_epochs = config.get('recovery_epochs', 10)
        self.pruning_epochs = config.get('pruning_epochs', self.epochs)
        self.temperature = config.get('temperature', TEMP)  
        self.base_temperature = config.get('base_temperature', BASE_TEMP)  
        self.num_classes = config.get('num_classes', CIFAR10_CLASSES)  
        self.pruner = config.get('pruner', None)    
        self.save_path = config.get('save_path', None)    

        # Knowledge distillation (need to tweak)
        self.use_kd = config.get('use_kd', False)
        self.kd_temp = config.get('kd_temp', 5)
        self.alpha = config.get('alpha', 0.5)
        self.beta = config.get('beta', 0.5)

        # Initializing lambdas
        self.initialize_lambdas(config)
        
        # Current, Pre-trained, and Fine-tuned model paths for saving weights
        self.cm_save_path = f"{self.save_path}_cm.pt"
        self.pm_save_path = f"{self.save_path}_pm.pt"
        self.fm_save_path = f"{self.save_path}_fm.pt"
        
        self.supervised_loss = SupConLoss(self.n_views, self.temperature, self.base_temperature, self.batch_size)
        self.unsupervised_loss = NTXentLoss(self.n_views, self.temperature)
        
        embeddings_dict = {
            "resnet50": 2048,
            "resnet101":2048,
            "vgg11":    4096,
            "vgg19":    4096,
            "vitb16":   768,
            "vitl16":   1024,
        }
        embedding = embeddings_dict[self.model_name]
        self.classification_head = nn.Linear(embedding, self.num_classes).to(get_device())

        self.avg_losses = []
        self.avg_PrC = []
        self.avg_SnC = []
        self.avg_FiC = []
        self.avg_CE = []

    def initialize_lambdas(self, config):
        """
        Initializes the lambdas for the current model.
        """
        lambdas = config.get('lambdas', [0.25, 0.25, 0.25, 0.25])
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

    def create_classification_net(self, hasFrozenBackbone=True):
        """
        Creates a classification network from the current model. To be used for fine-tuning after training has been completed.
        """
        cls_model = ClassificationNetwork(self.model_name, self.num_classes, hasFrozenBackbone).to(get_device())
        load_weights(cls_model, self.cm_save_path) 
        print(f"Sparsity ratio of model: {get_model_sparsity(cls_model)}")
        return cls_model.to(self.device)
        
    def generate_mask_from_model(self):
        """
        Generates a mask from the current models weights. 
        """
        load_weights(self.current_model, self.cm_save_path)
        zero_masks = {}
        for name, param in self.current_model.named_parameters():
            mask = (param != 0).float()
            zero_masks[name] = mask
        self.set_mask(zero_masks)

    def set_mask(self, masks):
        if self.pruner is not None:
            self.pruner.masks = masks

    def get_pruner(self):
        return self.pruner

    def get_losses(self):
        return self.avg_losses, self.avg_PrC, self.avg_SnC, self.avg_FiC, self.avg_CE 

    def supervised_criterion(self, features, labels):
        """
        Computes the supervised loss between the features of two images.
        """
        loss = self.supervised_loss(features, labels)
        return loss

    def unsupervised_criterion(self, features1, features2):
        """
        Computes the unsupervised loss between the features of two images. Loss is computed in both directions to ensure symmetry.
        """
        loss = self.unsupervised_loss(features1, features2)
        loss += self.unsupervised_loss(features2, features1)
        return (loss)/2

    def get_embeddings_and_logits(self, images):
        """ 
        Returns embeddings and logits from the current model based on the type of the model.
        """
        if self.model_name in ['vgg11', 'vgg19']:
            x = self.current_model.backbone.features(images)
            x = self.current_model.backbone.avgpool(x)
            x = x.reshape(x.shape[0], -1)

            for i in range(6):
                x = self.current_model.backbone.classifier[i](x)
        
        elif self.model_name in ['vitb16', 'vitl16']:
            x = self.current_model.backbone.conv_proj(images)
            class_token = self.current_model.backbone.class_token.expand(self.batch_size, -1, -1)
            x = torch.cat((class_token, x.flatten(2).transpose(1, 2)), axis=1)
            x = self.current_model.backbone.encoder(x)
            x = x[:, 0, :]

        else:
            x = self.current_model.backbone.conv1(images)
            x = self.current_model.backbone.bn1(x)
            x = self.current_model.backbone.relu(x)
            x = self.current_model.backbone.maxpool(x)
            x = self.current_model.backbone.layer1(x)
            x = self.current_model.backbone.layer2(x)
            x = self.current_model.backbone.layer3(x)
            x = self.current_model.backbone.layer4(x)
            x = self.current_model.backbone.avgpool(x)
            x = x.reshape(x.shape[0], -1)

        logits = self.classification_head(x)
        if self.model_name in ['vgg11', 'vgg19']:
            embeddings = self.current_model.backbone.classifier[-1](x)
            embeddings = F.normalize(embeddings, dim=1)
        elif self.model_name in ['vitb16', 'vitl16']:
            embeddings = self.current_model.backbone.heads(x)  
            embeddings = F.normalize(embeddings, dim=1)
        else:
            embeddings = self.current_model.backbone.fc(x)
            embeddings = F.normalize(embeddings, dim=1)

        return embeddings, logits
        
    def train(self, train_loader):
        # Initializing logger
        if self.logger is not None:
            self.logger.create_log()
            self.logger.log_hyperparameters({
                'epochs': self.epochs, 
                'n_views': self.n_views, 
                'optimizer': self.optimizer,
                'scheduler': self.scheduler, 
                'batch_size': self.batch_size, 
                'temperature': self.temperature, 
                'base_temperature': self.base_temperature
                })
            
        self.snapshots = []
        for epoch in range(self.epochs):
            # Prune the model
            model_sparsity = get_model_sparsity(self.current_model)
            print(f"Model sparsity: {model_sparsity:.3f}")

            losses = []
            prc_losses = []
            snc_losses = []
            fic_losses = []
            ce_losses = []
            
            # N randomly sampled sample/lable pairs
            batch = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}")

            # No need for multi-view batch: Dense to sparse paper used batch size of N. SimClr uses batch size of 2N.
            for idx, (images, labels) in enumerate(batch):
                images1, images2 = images
                images1, images2, labels = images1.to(self.device), images2.to(self.device), labels.to(self.device)
                
                if images1.size(0) != images2.size(0) or images1.size(0) < self.batch_size:
                    continue            
                
                # Splitting embeddings into n views
                current_embeddings, current_logits = self.get_embeddings_and_logits(images1)
                with torch.no_grad():
                    pretrain_embeddings = self.pre_trained_model(images2)
                    finetune_embeddings = self.finetuned_model(images2)

                # PrC Module
                sup_features_prc = torch.cat((current_embeddings, pretrain_embeddings))
                L_prc_sup = self.supervised_criterion(sup_features_prc, labels)
                L_prc_unsup = self.unsupervised_criterion(current_embeddings, pretrain_embeddings)
                L_prc_total = L_prc_sup + L_prc_unsup

                # CE Loss
                CE_loss = nn.CrossEntropyLoss()(current_logits, labels)

                # FiC Module - Choice of either using knowledge distillation or contrastive learning
                if self.use_kd:
                    KD_loss = F.kl_div(F.log_softmax(current_embeddings/self.kd_temp, dim=1), F.softmax(finetune_embeddings/self.kd_temp, dim=1), reduction='batchmean')
                    KD_loss = (KD_loss * self.kd_temp **2)
                    L_fic_total = self.alpha * KD_loss + self.beta * CE_loss
                else:
                    sup_features_fic = torch.cat((current_embeddings, finetune_embeddings))
                    L_fic_sup = self.supervised_criterion(sup_features_fic, labels)
                    L_fic_unsup = self.unsupervised_criterion(current_embeddings, finetune_embeddings)
                    L_fic_total = L_fic_sup + L_fic_unsup

                # SnC Module
                L_snc_sup = torch.tensor(0.0, device=get_device())
                L_snc_unsup = torch.tensor(0.0, device=get_device())
                for snapshot in self.snapshots:
                    snapshot_embeddings = snapshot(images1)
                    sup_features_snc = torch.cat((current_embeddings, snapshot_embeddings))
                    L_snc_sup += self.supervised_criterion(sup_features_snc, labels)
                    L_snc_unsup += self.unsupervised_criterion(current_embeddings, snapshot_embeddings)
                L_snc_total = L_snc_sup + L_snc_unsup
                
                # Total loss calculation
                total_loss = (
                    self.lambda1 * CE_loss +
                    self.lambda2 * L_prc_total +
                    self.lambda3 * L_snc_total +
                    self.lambda4 * L_fic_total
                    )
                
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Updating pruner masks and applying them to the model
                if idx == 0:
                    self.pruner.prune(self.current_model)
                self.optimizer.step()
                self.pruner.apply_mask(self.current_model)

                losses.append(total_loss.item())
                prc_losses.append(L_prc_total.item())
                snc_losses.append(L_snc_total.item())
                fic_losses.append(L_fic_total.item())
                ce_losses.append(CE_loss.item())
                batch.set_postfix({'PrC Loss': L_prc_total.item(), 'SnC Loss': L_snc_total.item(), 'FiC Loss': L_fic_total.item(), 'CE Loss': CE_loss.item(), 'Loss': total_loss.item()})

            # Averaging the losses
            self.avg_PrC.append(sum(prc_losses) / len(prc_losses))
            self.avg_SnC.append(sum(snc_losses) / len(snc_losses))
            self.avg_FiC.append(sum(fic_losses) / len(fic_losses))
            self.avg_CE.append(sum(ce_losses) / len(ce_losses))
            self.avg_losses.append(sum(losses) / len(losses))

            # Saving pruned model as a snapshot
            if len(self.snapshots) < self.pruning_epochs:
                snapshot = deepcopy(self.current_model)
                self.snapshots.append(snapshot)
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            info = f"Epoch {epoch + 1}/{self.epochs}: Total Loss: {self.avg_losses[-1]:.3f} - PrC Loss: {self.avg_PrC[-1]:.3f} - SnC Loss: {self.avg_SnC[-1]:.3f} - FiC Loss: {self.avg_FiC[-1]:.3f} - CE Loss: {self.avg_CE[-1]:.3f}\n" 
            print(info)
            if self.logger is not None:
                self.logger.log_epochs(info)       

            # Increasing pruning ratio for next iteration
            self.retrain(train_loader)
            self.pruner.ratio_step()

            model_sparsity = get_model_sparsity(self.current_model)
            print(f"Model sparsity: {model_sparsity:.3f}")
            
            if self.save_path is not None:
                torch.save(self.current_model.state_dict(), self.cm_save_path)
                torch.save(self.pre_trained_model.state_dict(), self.pm_save_path)
                torch.save(self.finetuned_model.state_dict(), self.fm_save_path)


    def retrain(self, train_loader):
        self.current_model.train()
        for epoch in range(self.recovery_epochs):
            losses, prc_losses, snc_losses, fic_losses, ce_losses = [], [], [], [], []
            
            batch = tqdm(train_loader, desc=f"Recovery epoch {epoch + 1}/{self.recovery_epochs}")
            for idx, (images, labels) in enumerate(batch):
                images1, images2 = images
                images1, images2, labels = images1.to(self.device), images2.to(self.device), labels.to(self.device)
                
                if images1.size(0) != images2.size(0) or images1.size(0) < self.batch_size:
                    continue            
                
                current_embeddings, current_logits = self.get_embeddings_and_logits(images1)
                with torch.no_grad():
                    pretrain_embeddings = self.pre_trained_model(images2)
                    finetune_embeddings = self.finetuned_model(images2)

                # PrC Module
                sup_features_prc = torch.cat((current_embeddings, pretrain_embeddings))
                L_prc_sup = self.supervised_criterion(sup_features_prc, labels)
                L_prc_unsup = self.unsupervised_criterion(current_embeddings, pretrain_embeddings)
                L_prc_total = L_prc_sup + L_prc_unsup

                # CE Loss
                CE_loss = nn.CrossEntropyLoss()(current_logits, labels)

                # FiC Module - Choice of either using knowledge distillation or contrastive learning
                if self.use_kd:
                    KD_loss = F.kl_div(F.log_softmax(current_embeddings/self.kd_temp, dim=1), F.softmax(finetune_embeddings/self.kd_temp, dim=1), reduction='batchmean')
                    KD_loss = (KD_loss * self.kd_temp **2)
                    L_fic_total = self.alpha * KD_loss + self.beta * CE_loss
                else:
                    sup_features_fic = torch.cat((current_embeddings, finetune_embeddings))
                    L_fic_sup = self.supervised_criterion(sup_features_fic, labels)
                    L_fic_unsup = self.unsupervised_criterion(current_embeddings, finetune_embeddings)
                    L_fic_total = L_fic_sup + L_fic_unsup

                # SnC Module
                L_snc_sup = torch.tensor(0.0, device=get_device())
                L_snc_unsup = torch.tensor(0.0, device=get_device())
                for snapshot in self.snapshots:
                    snapshot_embeddings = snapshot(images1)
                    sup_features_snc = torch.cat((current_embeddings, snapshot_embeddings))
                    L_snc_sup += self.supervised_criterion(sup_features_snc, labels)
                    L_snc_unsup += self.unsupervised_criterion(current_embeddings, snapshot_embeddings)
                L_snc_total = L_snc_sup + L_snc_unsup
                
                total_loss = (
                    self.lambda1 * CE_loss +
                    self.lambda2 * L_prc_total +
                    self.lambda3 * L_snc_total +
                    self.lambda4 * L_fic_total
                    )
                
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                # Only applying mask
                self.pruner.apply_mask(self.current_model)
                
                losses.append(total_loss.item())
                prc_losses.append(L_prc_total.item())
                snc_losses.append(L_snc_total.item())
                fic_losses.append(L_fic_total.item())
                ce_losses.append(CE_loss.item())
                
                batch.set_postfix({'PrC Loss': L_prc_total.item(),
                                   'SnC Loss': L_snc_total.item(),
                                   'FiC Loss': L_fic_total.item(),
                                   'CE Loss': CE_loss.item(),
                                   'Loss': total_loss.item()})
           
            self.avg_PrC.append(sum(prc_losses) / len(prc_losses))
            self.avg_SnC.append(sum(snc_losses) / len(snc_losses))
            self.avg_FiC.append(sum(fic_losses) / len(fic_losses))
            self.avg_CE.append(sum(ce_losses) / len(ce_losses))
            self.avg_losses.append(sum(losses) / len(losses))            

            info = f"Recovery Epoch {epoch + 1}/{self.recovery_epochs}: Total Loss: {self.avg_losses[-1]:.3f} - PrC Loss: {self.avg_PrC[-1]:.3f} - SnC Loss: {self.avg_SnC[-1]:.3f} - FiC Loss: {self.avg_FiC[-1]:.3f} - CE Loss: {self.avg_CE[-1]:.3f}\n" 
            print(info)
            if self.logger is not None:
                self.logger.log_epochs(info)       
            
  