import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from abc import abstractmethod, ABC
from typing import Dict

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def layer_check(name, param):
    if param is None:
        return False
    
    if param.dim() <= 1 or not param.requires_grad:
        return False
    
    exclusion_keywords = [
        'hyperfucntion', 'relu',
        'encoder_head'



        # 'cls_head',
        # 'encoder_head',
        # 'hyperfunction',
        # 'fc',   # ResNet
        # 'classifier.6', # VGG
        # 'embeddings', 'conv_proj', 'pos_embedding', 'heads', 'classifier.weight', # ViT        
        # 'projection_head', # Encoder heads
        # 'vocab_projector', 'vocab_transform', # DistilBERT
        # 'classifier.dense', 'classifier.out_proj', 'lm_head', # RoBERTA

        # 'heads', 'conv_proj', 'fc', 'classifier', 'embeddings', 
        # 'class_token', 'pos_embedding', 'vocab_transform', 'lm_head',
        # 'projection_head'
        ]
    
    if any(keyword in name.lower() for keyword in exclusion_keywords):
        return False
    return True

def check_model_sparsity(model):
    sum_zero, total_count = 0, 0
    for name, param in model.named_parameters():
        if layer_check(name, param):
            sum_zero += torch.sum(param == 0).item()
            total_count += param.numel()

    return sum_zero / total_count

def check_sparsity_distribution(model, verbose=True):
    names = []
    sparsities = []

    total_weights = 0
    total_zero_weights = 0
    total_backbone_weights = 0
    total_backbone_zero_weights = 0

    print("\nSPARSITY DISTRIBUTION PER LAYER")
    print("-" * 80)

    for name, param in model.named_parameters():
        if param.dim() > 1 and param.requires_grad:                
            num_zero_weights = torch.sum(param.data == 0).item()
            num_layer_weights = param.data.numel()
            layer_sparsity = num_zero_weights / num_layer_weights

            total_weights += num_layer_weights
            total_zero_weights += num_zero_weights

            if layer_check(name, param):
                total_backbone_weights += num_layer_weights
                total_backbone_zero_weights += num_zero_weights

                names.append(name)
                sparsities.append(layer_sparsity)

            if verbose:
                print(f"{name}:\t{layer_sparsity * 100:.4f}%\t|\tsparsity: ({num_zero_weights}/{num_layer_weights})")

    overall_sparsity = total_zero_weights / total_weights if total_weights > 0 else 0
    backbone_sparsity = total_backbone_zero_weights / total_backbone_weights if total_backbone_weights > 0 else 0

    if verbose:
        print("\nMODEL SUMMARY")
        print("-" * 80)
        print(f"Model sparsity:\t\t\t{overall_sparsity:.4f}")
        print(f"Backbone sparsity:\t\t{backbone_sparsity:.4f}")
        print(f"Backbone Sparse Weights to Total:{total_backbone_zero_weights:.4f} - {total_backbone_weights:.4f}")
        print(f"Total parameters analyzed:\t{total_weights}")
        print(f"Number of non-zero parameters:\t{total_weights - total_zero_weights}")
        print(f"Number of zero parameters:\t{total_zero_weights}\n")

    plt.figure(figsize=(10, 0.4 * len(names)))
    bars = plt.barh(range(len(names)), sparsities, color='skyblue')
    plt.yticks(range(len(names)), names)
    plt.xlabel("Sparsity (Fraction of Zero Weights)")
    plt.title("Layer-wise Sparsity Distribution")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.xlim(0, 1.0)
    plt.tight_layout()
    plt.show()


def erk_initialization(model: torch.nn.Module,target_sparsity: float,erk_power_scale: float = 1.0,) -> Dict[str, torch.Tensor]:
    # target_sparsity is fraction of weights to zero out (0.0 - 1.0)
    if not (0.0 <= target_sparsity <= 1.0):
        raise ValueError("target_sparsity must be between 0 and 1")

    prunable_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.dim() >= 2 and "weight" in name:
            prunable_params[name] = param
    erk_scores = {}
    for name, param in prunable_params.items():
        if param.dim() == 2:  # Linear: (out, in)
            n_out, n_in = param.shape
        elif param.dim() == 4:  # Conv2d: (out, in, k_h, k_w)
            n_out, n_in, k_h, k_w = param.shape
            n_in = n_in * k_h * k_w
        else:
            # treat remaining dims by flattening input side
            shape = param.shape
            n_out = shape[0]
            n_in = int(torch.prod(torch.tensor(shape[1:])).item())
        # ERK score proportional to (n_in + n_out) / (n_in * n_out)
        erk_scores[name] = (n_in + n_out) / (n_in * n_out)
    if not erk_scores:
        return {}

    # normalize
    total_score = sum(erk_scores.values())
    normalized = {k: (v / total_score) ** erk_power_scale for k, v in erk_scores.items()}

    total_params = float(sum(p.numel() for p in prunable_params.values()))
    target_remaining = total_params * (1.0 - target_sparsity)
    expected_remaining_per_layer = {k: normalized[k] * p.numel() for k, p in prunable_params.items() if k in normalized}
    sum_expected = sum(expected_remaining_per_layer.values())
    if sum_expected == 0:
        raise RuntimeError("Sum of expected remaining parameters is zero; check model shapes and erk scores.")
    scale = float(target_remaining) / float(sum_expected)
    masks: Dict[str, torch.Tensor] = {}
    for name, param in prunable_params.items():
        if name in normalized:
            layer_density = normalized[name] * scale  # fraction kept
        else:
            layer_density = 1.0 - target_sparsity
        layer_density = float(max(0.0, min(1.0, layer_density)))  # clamp to [0,1]
        keep_prob = layer_density
        mask = (torch.rand_like(param) < keep_prob).to(dtype=param.dtype)
        masks[name] = mask
    return masks


def apply_erk_initialization(model: torch.nn.Module,target_sparsity: float,erk_power_scale: float = 1.0):
    masks = erk_initialization(model, target_sparsity, erk_power_scale)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in masks:
                param.mul_(masks[name])
    return masks


class Pruner(ABC):
    def __init__(self, model, epochs, target_sparsity, sparsity_scheduler="cubic"):
        self.epochs = epochs
        self.target_ratio = target_sparsity
        self.ratio = 0.0
        self.masks = {}

        assert sparsity_scheduler in ["linear", "cubic"], "Invalid sparsity scheduler"
        self.sparsity_scheduler = sparsity_scheduler

        for name, param in model.named_parameters():
            if layer_check(name, param):
                self.masks[name] = torch.ones_like(param)
        
    @abstractmethod
    def prune(self, model):
        pass
            
    @torch.no_grad()  
    def apply_mask(self, model):
        for name, param in model.named_parameters():
            if name not in self.masks:
                continue
            
            # Zero-ing out the weights
            param.data.mul_(self.masks[name])

    def ratio_step(self, epoch, epochs, initial_sparsity, target_sparsity):
        if self.sparsity_scheduler == 'linear':
            self.linear_scheduler(epoch, epochs, initial_sparsity, target_sparsity)
        elif self.sparsity_scheduler == 'cubic':
            self.cubic_scheduler(epoch, epochs, initial_sparsity, target_sparsity)
            
    def linear_scheduler(self, epoch, total_epochs, initial_sparsity, final_sparsity):
        t = epoch + 1
        sparsity = (final_sparsity / total_epochs) * t
        self.ratio = min(sparsity, self.target_ratio)
        print(f"\n[Pruner] Linear sparsity ratio increased to {self.ratio:.3f}.\n")

    def cubic_scheduler(self, epoch, total_epochs, initial_sparsity, final_sparsity):
        t = epoch + 1
        sparsity = final_sparsity + (initial_sparsity - final_sparsity) * ((1 - t / total_epochs) ** 3)
        self.ratio = min(sparsity, self.target_ratio)
        print(f"\n[Pruner] Cubic Sparsity ratio increased to {self.ratio:.3f}.\n")

    def reset(self):
        self.ratio = self.target_ratio / self.epochs
        print(f"[Pruner] Sparsity ratio reset to initial value: {self.ratio:.3f}.\n")
    
    def set_mask(self, masks):
        self.masks = masks

class MagnitudePrune(Pruner):
    def prune(self, model):
        all_importances = []
        importance_cache = {}
        for name, param in model.named_parameters():
            if layer_check(name, param):
                # Calculating importance
                importance = torch.abs(self.masks[name] * param)
                importance_cache[name] = importance
                all_importances.append(importance.view(-1))

        # Calculating global importance and threshold
        global_importances = torch.cat(all_importances)
        total_weights = global_importances.numel()
        num_to_zero = max(1, min(total_weights, round(total_weights * self.ratio)))
        threshold = torch.kthvalue(global_importances, num_to_zero).values.item()
        
        # Updating masks
        for name, importance in importance_cache.items():
            self.masks[name] = torch.gt(importance, threshold).float()

class SNIPIterativePrune(Pruner):
    def prune(self, model):
        all_importances = []
        importance_cache = {}
        for name, param in model.named_parameters():
            if layer_check(name, param):
                # Calculating importance
                importance = torch.abs(self.masks[name] * param * param.grad.detach())
                importance_cache[name] = importance
                all_importances.append(importance.view(-1))

        # Calculating global importance and threshold
        global_importances = torch.cat(all_importances)
        total_weights = global_importances.numel()
        num_to_zero = max(1, min(total_weights, round(total_weights * self.ratio)))
        threshold = torch.kthvalue(global_importances, num_to_zero).values.item()
        
        # Updating masks
        for name, importance in importance_cache.items():
            self.masks[name] = torch.gt(importance, threshold).float()

class WandaPrune(Pruner):
    def __init__(self, model, target_sparsity, epochs, sparsity_scheduler):
        super().__init__(model, target_sparsity, epochs, sparsity_scheduler)
        self.main_layers = {}
        self.wrapped_layers = {}
        self.current_epoch = 0
        self.hooks = []
        self.is_wanda = True
        self.device = next(model.parameters()).device

        if hasattr(model.model, 'distilbert') and hasattr(model.model.distilbert, 'transformer') and hasattr(model.model.distilbert.transformer, 'layer'):
            self.prefix = "distilbert.transformer.layer"
        elif hasattr(model.model, 'roberta') and hasattr(model.model.roberta, 'encoder') and hasattr(model.model.roberta.encoder, 'layer'):
            self.prefix = "roberta.encoder.layer"
        elif hasattr(model.model, 'encoder') and hasattr(model.model.encoder, 'layers'):
            self.prefix = "encoder.layers"
        elif hasattr(model.model, 'vit') and hasattr(model.model.vit, 'encoder') and hasattr(model.model.vit.encoder, 'layer'):
            self.prefix = "vit.encoder.layer"

        self.init_layers(model)
    
    def init_layers(self, model):
        for module_name, module in model.named_modules():
            name = f'{module_name}.weight'
            if hasattr(module, 'weight') and layer_check(name, module.weight):
                self.main_layers[name] = module.to(self.device)
                self.wrapped_layers[name] = self.WrappedLayer(module.to(self.device), name)

        # for name, module in model.named_modules():
        #     for child_name, child in module.named_children():
        #       if isinstance(child, torch.nn.Linear) and self.prefix in name:                
        #             full_name = f"{name}.{child_name}"
        #             print(full_name)
        #             self.main_layers[full_name] = child.to(self.device)
        #             self.wrapped_layers[full_name] = self.WrappedLayer(child.to(self.device), full_name)
            
    def register_hooks(self, model):
        print("[Pruner] Adding hooks")
        for name in self.wrapped_layers:
            self.hooks.append(self.main_layers[name].register_forward_hook(self.activation_hook(name, model))) 

    def calibrate(self, model, trainloader, num_calibration_batches=100):
        from training_utils import _handle_data_to_device
        self.register_hooks(model)

        print(f"[Pruner] Calibrating with {num_calibration_batches} batches...")
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                if i > num_calibration_batches:
                    break
                # Handling different batch formats
                data, labels = _handle_data_to_device(self, batch)

                # Forward pass
                if len(data) == 2:
                    for d in data:
                        _ = model(d, return_emb=True)
                else:
                    _ = model(data)

                # Print progress every 10 batches
                if (i + 1) % 10 == 0:
                    print(f"  Calibration progress: {i+1}/{num_calibration_batches} batches")
        
        model.train()
        print("[Pruner] Calibration complete!")

        self.remove_hooks()

    def remove_hooks(self):
        print("[Pruner] Removing hooks")
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def activation_hook(self, name, model):
        def hook(_, inputs, outputs):
            if model.training:
                input_tensor = inputs[0] if isinstance(inputs, tuple) else inputs
                if name in self.wrapped_layers:
                    self.wrapped_layers[name].add_batch(input_tensor.data.to(self.device))
        return hook
    
    def prune(self, model):
        all_importances = []
        importance_cache = {}

        for name in self.wrapped_layers:
            if self.wrapped_layers[name].scaler_row is None:
                continue

            W = self.main_layers[name].weight.data
            scaler_row = self.wrapped_layers[name].scaler_row.to(self.device).float()
            scaler_row = torch.clamp(torch.nan_to_num(scaler_row, nan=0.0, posinf=1e6, neginf=0.0), min=0.0)

            if W.dim() == 4:
                scaler_row = scaler_row.reshape(1, -1, 1, 1)
            else: 
                scaler_row = scaler_row.reshape(1, -1)

            # Calculating local importance and threshold
            importance = (torch.abs(W) * torch.sqrt(scaler_row + 1e-10))
            importance_cache[name] = importance
            all_importances.append(importance.view(-1))

        # Calculating global importance and threshold
        global_importances = torch.cat(all_importances)
        total_weights = global_importances.numel()
        num_to_zero = max(1, min(total_weights, round(total_weights * self.ratio)))
        threshold = torch.kthvalue(global_importances, num_to_zero).values.item()

        # Updating masks
        for name, importance in importance_cache.items():
            self.masks[name] = torch.gt(importance, threshold).float()

    class WrappedLayer:
        def __init__(self, layer, name):
            self.layer = layer
            self.device, self.out_channels, self.in_channels, self.scaler_row = None, 0, 0, None
            if hasattr(self.layer, 'weight') and self.layer.weight is not None:
                self.out_channels, self.in_channels = self.layer.weight.shape[0], self.layer.weight.shape[1]
                self.device = self.layer.weight.device
                self.scaler_row = torch.zeros((self.in_channels), device=self.device)
            else:
                self.device = layer.device
                
            self.layer_name = name
            self.nsamples = 0

        def add_batch(self, activation_tensor):
            if self.scaler_row is None:
                return

            self.nsamples += 1

            # Handling batch size of 1
            if len(activation_tensor.shape) == 2:
                activation_tensor = activation_tensor.unsqueeze(0)

            if len(activation_tensor.shape) == 3:
                activation_tensor = activation_tensor.reshape(-1, activation_tensor.shape[-1])
            elif len(activation_tensor.shape) == 4:
                activation_tensor = activation_tensor.permute(0, 2, 3, 1)           # Shape: [B, H, W, C]
                activation_tensor = activation_tensor.reshape(-1, self.in_channels) # Shape: [B*H*W, C]
            
            total_activations = activation_tensor.shape[0]  # B*H*C

            activation_norm = torch.norm(activation_tensor, dim=0, p=2) ** 2    # Shape: [C]
            self.scaler_row += activation_norm / total_activations

PRUNER_DICT = {
    "magnitude_pruning": MagnitudePrune,
    "snip_pruning": SNIPIterativePrune,
    "wanda_pruning": WandaPrune
}




