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
    _detect_model_type, _detect_num_classes, _detect_cv_image_size,
    _initialize_models, _initialize_optimizer, _initialize_scheduler,
    _initialize_data_loaders, _initialize_pruner, _initialize_contrastive_losses,
    _initialize_paths_and_logger, _handle_optimizer_and_pruning
)

class BaCPTrainingArguments:
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
                 tau=0.15,
                 epochs=5,
                 recovery_epochs=10,
                 scheduler_type=None,
                 patience=20,
                 finetuned_weights=None,
                 current_finetuned_weights=None,
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
        self.is_bacp = True

        # Pruning parameters
        self.pruner = None
        self.pruning_type = pruning_type
        self.target_sparsity = target_sparsity
        self.sparsity_scheduler = sparsity_scheduler
        self.pruning_epochs = epochs or pruning_epochs

        # Training parameters
        self.tau = tau
        self.epochs = epochs
        self.recovery_epochs = recovery_epochs
        self.scheduler_type = scheduler_type
        self.patience = patience
        self.finetuned_weights = finetuned_weights
        self.current_finetuned_weights = current_finetuned_weights
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
        _detect_cv_image_size(self)
        _initialize_models(self)
        _initialize_optimizer(self)
        _initialize_scheduler(self)
        _initialize_data_loaders(self)
        _initialize_pruner(self)
        _initialize_contrastive_losses(self, self.tau)
        _initialize_paths_and_logger(self)

class BaCPTrainer:
    def __init__(self, bacp_training_args, lambdas=[0.25, 0.25, 0.25, 0.25]):
        for key, value in vars(bacp_training_args).items():
            setattr(self, key, value)

        self._initialize_heads()
        self._initialize_lambdas(lambdas)
        self._initialize_metric_lists()
        self._initialize_log_parameters()
        self.snapshots = []
        self.recover = False

    def _initialize_heads(self):
        if self.num_classes is not None:
            self.classification_head = nn.Linear(self.embedded_dim, self.num_classes).to(self.device)
        else:
            self.classification_head = None

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
        return round(check_model_sparsity(self.model), 4)

    def _initialize_log_parameters(self):
        logger_params = {
            'model_name': self.model_name,
            'model_task': self.model_task,
            'model_type': getattr(self, 'model_type', None),
            'num_classes': getattr(self, 'num_classes', None),
            
            # Training Config
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'optimizer_type': self.optimizer_type,
            'epochs': self.epochs,
            'recovery_epochs': self.recovery_epochs,
            'patience': self.patience,
            
            # Pruning Config
            'pruning_type': self.pruning_type,
            'target_sparsity': self.target_sparsity,
            'sparsity_scheduler': self.sparsity_scheduler,
            'pruning_epochs': self.pruning_epochs,
            
            # Contrastive Learning
            'n_views': getattr(self, 'n_views', None),
            'temperature': getattr(self, 'temperature', None),
            'base_temperature': getattr(self, 'base_temperature', None),
            
            # Technical Settings
            'device': str(self.device),
            'enable_mixed_precision': self.enable_mixed_precision,
            'num_workers': self.num_workers,
            
            # Data Info
            'train_batches': len(self.trainloader) if hasattr(self, 'trainloader') else None,
            'val_batches': len(self.valloader) if hasattr(self, 'valloader') else None,
            
            # Paths
            'current_model_path': getattr(self, 'save_path', None),
        }
        self.logger_params = {k: v for k, v in logger_params.items() if v is not None}
        
    def train(self):
        if hasattr(self, 'logger') and self.logger is not None:
            self.logger.create_log()
            self.logger.log_hyperparameters(self.logger_params)
        
        self.model.train()
        self.pre_trained_model.eval()
        self.finetuned_model.eval()

        for epoch in range(self.epochs):
            # Training phase
            desc = f"Training Epoch [{epoch+1}/{self.epochs}]"
            self.train_epoch(epoch, desc)
            
            if len(self.snapshots) < self.pruning_epochs:
                state = deepcopy(self.model.state_dict())
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
            
    def retrain(self):
        self.recover = True
        for epoch in range(self.recovery_epochs):
            # Training phase
            desc = f"Retraining epoch [{epoch+1}/{self.recovery_epochs}]"
            self.train_epoch(epoch, desc)

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

    def train_epoch(self, epoch, desc):
        if (self.prune and self.pruner) and (hasattr(self.pruner, 'is_wanda') and self.pruner.is_wanda) and (not self.recover):
            self.pruner.register_hooks(self.model)

        losses, prc_losses, snc_losses, fic_losses, ce_losses, total = 0, 0, 0, 0, 0, 0

        batchloader = tqdm(self.trainloader, desc=desc, leave=False) if self.enable_tqdm else self.trainloader
        for step, batch_data in enumerate(batchloader):
            if self.model_type == 'cv':
                images, labels = batch_data
                images1, images2 = images 
                batch = {
                    'data1': images1.to(self.device),
                    'data2': images2.to(self.device),
                    'labels': labels.to(self.device)
                }
            else:  # LLM
                batch = {k: v.to(self.device) for k, v in batch_data.items()}
                input_data = batch

            labels = batch["labels"]
            if self.model_task == 'wikitext2':
                mask = (labels != -100) # Creating a mask to remove unnessary tokens
                batch = {k: v for k, v in batch.items() if k != 'labels'}
            else:
                mask = None

            if hasattr(self, 'disable') and self.disable == 'use_different_data_view':
                input_data = batch['data2'].to(self.device)
            elif hasattr(self, 'disable') and self.disable == 'use_same_data_view':
                input_data = batch['data1'].to(self.device)
            else:
                input_data = batch['data2'].to(self.device)

            with autocast(device_type=self.device) if self.enable_mixed_precision else contextlib.nullcontext():

                current_embeddings, current_logits = self._get_embeddings_and_logits(
                    batch if self.model_type == 'llm' else batch['data1'], mask
                )
                with torch.no_grad():
                    pretrained_embeddings = self.pre_trained_model(input_data)
                    finetuned_embeddings = self.finetuned_model(input_data)
                    if hasattr(pretrained_embeddings, 'logits') and hasattr(finetuned_embeddings, 'logits'):
                        pretrained_embeddings = pretrained_embeddings.logits
                        finetuned_embeddings = finetuned_embeddings.logits

                if mask is not None:
                    labels = labels[mask]
                    current_logits = current_logits[mask]
                    current_embeddings = current_embeddings[mask]
                    pretrained_embeddings = pretrained_embeddings[mask]
                    finetuned_embeddings = finetuned_embeddings[mask]

                CE_loss = nn.CrossEntropyLoss()(current_logits, labels)

                if  hasattr(self, 'disable') and self.disable == 'disable_unsupervised_loss':
                    # PrC Module
                    sup_features_prc = torch.cat((current_embeddings, pretrained_embeddings))
                    L_prc_sup = self._supervised_criterion(sup_features_prc, labels)
                    L_prc_total = L_prc_sup

                    # FiC Module
                    sup_features_fic = torch.cat((current_embeddings, finetuned_embeddings))
                    L_fic_sup = self._supervised_criterion(sup_features_fic, labels)
                    L_fic_total = L_fic_sup

                    # SnC Module
                    L_snc_sup = torch.tensor(0.0, device=self.device)
                    for snapshot_model in self.snapshots:
                        with torch.no_grad():
                            snapshot_embeddings = snapshot_model(batch).logits if self.model_type == 'llm' else snapshot_model(input_data)
                            if mask is not None:
                                snapshot_embeddings = snapshot_embeddings[mask]
                        sup_features_snc = torch.cat((current_embeddings, snapshot_embeddings))
                        L_snc_sup += self._supervised_criterion(sup_features_snc, labels)
                    L_snc_total = L_snc_sup

                elif  hasattr(self, 'disable') and self.disable == 'disable_supervised_loss':
                    # PrC Module
                    L_prc_unsup = self._unsupervised_criterion(current_embeddings, pretrained_embeddings)
                    L_prc_total = L_prc_unsup

                    # FiC Module
                    L_fic_unsup = self._unsupervised_criterion(current_embeddings, finetuned_embeddings)
                    L_fic_total = L_fic_unsup

                    # SnC Module
                    L_snc_unsup = torch.tensor(0.0, device=self.device)
                    for snapshot_model in self.snapshots:
                        with torch.no_grad():
                            snapshot_embeddings = snapshot_model(batch).logits if self.model_type == 'llm' else snapshot_model(input_data)
                            if mask is not None:
                                snapshot_embeddings = snapshot_embeddings[mask]
                        L_snc_unsup += self._unsupervised_criterion(current_embeddings, snapshot_embeddings)
                    L_snc_total = L_snc_unsup
                
                elif  hasattr(self, 'disable') and self.disable == 'disable_all_loss':
                    L_prc_total = torch.tensor(0.0, device=self.device)
                    L_snc_total = torch.tensor(0.0, device=self.device)
                    L_fic_total = torch.tensor(0.0, device=self.device)

                else:
                    # PrC Module
                    sup_features_prc = torch.cat((current_embeddings, pretrained_embeddings))
                    L_prc_sup = self._supervised_criterion(sup_features_prc, labels)
                    L_prc_unsup = self._unsupervised_criterion(current_embeddings, pretrained_embeddings)
                    L_prc_total = L_prc_sup + L_prc_unsup

                    # FiC Module
                    sup_features_fic = torch.cat((current_embeddings, finetuned_embeddings))
                    L_fic_sup = self._supervised_criterion(sup_features_fic, labels)
                    L_fic_unsup = self._unsupervised_criterion(current_embeddings, finetuned_embeddings)
                    L_fic_total = L_fic_sup + L_fic_unsup

                    # SnC Module
                    L_snc_sup = torch.tensor(0.0, device=self.device)
                    L_snc_unsup = torch.tensor(0.0, device=self.device)
                    for snapshot_model in self.snapshots:
                        with torch.no_grad():
                            snapshot_embeddings = snapshot_model(input_data)
                            if hasattr(snapshot_embeddings, 'logits'):
                                snapshot_embeddings = snapshot_embeddings.logits
                            if mask is not None:
                                snapshot_embeddings = snapshot_embeddings[mask]
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

            _handle_optimizer_and_pruning(self, total_loss, epoch, step)

            losses += total_loss.item()
            ce_losses += (CE_loss.item() * self.lambda1)
            prc_losses += (L_prc_total.item() * self.lambda2)
            snc_losses += (L_snc_total.item() * self.lambda3)
            fic_losses += (L_fic_total.item() * self.lambda4)
            total += 1

            if self.enable_tqdm:
                batchloader.set_postfix({
                    'Loss': total_loss.item(), 
                    'PrC Loss': (L_prc_total.item() * self.lambda2), 
                    'SnC Loss': (L_snc_total.item() * self.lambda3), 
                    'FiC Loss': (L_fic_total.item() * self.lambda4), 
                    'CE Loss': (CE_loss.item() * self.lambda1),
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

    def _get_embeddings_and_logits(self, data_batch, mask=None):
        raw_features = self._extract_raw_features(data_batch)
        logits = self.classification_head(raw_features)       
        embeddings = self._extract_contrastive_embeddings(raw_features) 
        return embeddings, logits
    
    def _extract_raw_features(self, data_batch):
        if self.model_type == 'llm':
            outputs = self.model(data_batch)
            if self.model_task in ['squad', 'wikitext2']:
                return outputs.hidden_states[-1]
            else:
                return outputs.hidden_states[-1][:, 0, :] # CLS token
        else:
            raw_features = self.model(data_batch, extract_raw=True)
            if hasattr(raw_features, 'logits'):
                return raw_features.logits
            return raw_features
        
            # if model_family == 'vgg':
            #     x = self.model.model.features(data_batch)
            #     x = self.model.model.avgpool(x)
            #     x = x.reshape(x.shape[0], -1)
            #     x = self.model.model.classifier[:-1](x)
            #     return x
            # elif model_family == 'vit':
            #     batch_size = data_batch.shape[0]
            #     x = self.model.model.conv_proj(data_batch)
            #     class_token = self.model.model.class_token.expand(batch_size, -1, -1)
            #     x = torch.cat((class_token, x.flatten(2).transpose(1, 2)), axis=1)
            #     x = self.model.model.encoder(x)
            #     return x[:, 0, :]
            # elif model_family == 'resnet':
            #     x = self.model.model.conv1(data_batch)
            #     x = self.model.model.bn1(x)
            #     x = self.model.model.relu(x)
            #     x = self.model.model.maxpool(x)
            #     x = self.model.model.layer1(x)
            #     x = self.model.model.layer2(x)
            #     x = self.model.model.layer3(x)
            #     x = self.model.model.layer4(x)
            #     x = self.model.model.avgpool(x)
            #     x = x.reshape(x.shape[0], -1)
            #     return x
            # else:
            #     raise ValueError(f"Model family {model_family} not supported.")

    def _extract_contrastive_embeddings(self, raw_features):
        if self.model_type == 'llm':
            if hasattr(self.model.model, 'vocab_projector'):
                embeddings = self.model.model.vocab_projector(raw_features)
            elif hasattr(self.model.model, 'lm_head'):
                embeddings = self.model.model.lm_head(raw_features)
            else:
                raise ValueError(f"Model {self.model.model} does not have a projector layer")
            return F.normalize(embeddings, dim=1)
        else:
            model_family = self.model.model_family
            embeddings = self.model.projection_head(raw_features)
            # if model_family == 'vgg':
            #     embeddings = self.model.model.classifier[-1](raw_features)
            # elif model_family == 'vit':
            #     embeddings = self.model.model.heads(raw_features)  
            # elif model_family == 'resnet':
            #     embeddings = self.model.model.fc(raw_features)
            # else:
            #     raise ValueError(f"Model family {model_family} not supported.")
            return F.normalize(embeddings, dim=1)

    def _handle_save(self, epoch):
        if self.save_path:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        sparsity_key = self._get_sparsity_key()
        loss_list = self.avg_losses[sparsity_key]

        if len(loss_list) <= 1:
            torch.save(self.model.state_dict(), self.save_path)
            return True
        elif len(loss_list) > 1:
            if loss_list[-1] > min(loss_list[:-1]):
                torch.save(self.model.state_dict(), self.save_path)
                return True
        return False
    
    def _create_model_from_snapshot(self, snapshot_state):
        model = deepcopy(self.model)
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
        load_weights(self.model, self.save_path)
        zero_masks = {}
        for name, param in self.model.named_parameters():
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
