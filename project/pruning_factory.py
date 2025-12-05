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
import math

def layer_check(name, param):
    if param is None:
        return False
    if param.dim() <= 1 or not param.requires_grad:
        return False
    
    exclusion_keywords = [
        'hyperfunction', 'relu',
        'encoder_head'
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
    return (sum_zero / total_count) if total_count > 0 else 0.0


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


class Pruner(ABC):
    def __init__(self, model, epochs, s_end, sparsity_scheduler):
        self.epochs = epochs
        self.s_end = s_end
        self.s_curr = check_model_sparsity(model)
        self.sparsity_scheduler = sparsity_scheduler
        self.s_target = 0

        self.masks = {}
        for name, param in model.named_parameters():
            if layer_check(name, param):
                self.masks[name] = torch.ones_like(param)
        
    @abstractmethod
    def prune(self, model):
        pass
            
    @torch.no_grad()  
    def apply_mask(self, model):
        for name, param in model.named_parameters():
            if name in self.masks:
                param.data.mul_(self.masks[name])

    def ratio_step(self, t, T, s_init=None, s_end=None):
        if self.sparsity_scheduler == 'linear':
            self.linear_scheduler(t, T, s_init, s_end)

        elif self.sparsity_scheduler == 'cubic':
            self.cubic_scheduler(t, T, s_init, s_end)

        elif self.sparsity_scheduler == 'f_decay':
            alpha = 0.3
            self.f_decay_scheduler(t, T, alpha)   

        elif self.sparsity_scheduler == 'cyclic':
            if s_init is None or s_end is None:
                raise ValueError("cyclic needs s_init (s_min) and s_end (s_max)")
            self.cyclic_scheduler(t, T, s_init, s_end)  

        else:
            raise ValueError(f"Unknown sparsity scheduler: {self.sparsity_scheduler}")
    
    # Monotonic schedulers
    def linear_scheduler(self, t, T, s_init, s_end):
        t += 1
        if s_init is None or s_end is None:
            raise ValueError("linear scheduler needs s_init and s_end")
        sparsity = (s_end / T) * t
        self.s_curr = min(sparsity, self.s_end)
        print(f"[Pruner] Model sparity increased to {self.s_curr:.4f}.")

    def cubic_scheduler(self, t, T, s_init, s_end):
        t += 1
        if s_init is None or s_end is None:
            raise ValueError("cubic scheduler needs s_init and s_end")
        sparsity = s_end + (s_init - s_end) * ((1 - t / T) ** 3)
        self.s_curr = min(sparsity, self.s_end)
        print(f"[Pruner] Model sparity increased to {self.s_curr:.4f}.")
    
    # Non-monotonic schedulers
    def f_decay_scheduler(self, t, T, alpha):
        t = torch.tensor(t, dtype=torch.float)
        self.s_target = 0.5 * alpha * (1 + torch.cos(math.pi * t / T)).item()
        print(f"[Pruner] Fraction of weights to prune/regrow is {self.s_target:.8f}.")

    def cyclic_scheduler(self, t, Tc, s_min, s_max):
        cosine = torch.cos(2*torch.tensor(np.pi) * t  / Tc)
        self.s_target = s_min + ((s_max - s_min) / 2) * (1 - cosine)
        print(f"[Pruner] Fraction of weights to prune/regrow is {self.s_target:.8f}.")

    def set_mask(self, masks):
        self.masks = masks


class MagnitudePrune(Pruner):
    def prune(self, model):
        all_importances = []
        importance_cache = {}
        for name, param in model.named_parameters():
            if name in self.masks:
                mask = self.masks[name]
                importance = torch.abs(param) * mask
                importance_cache[name] = importance
                all_importances.append(importance.view(-1))

        # Calculating global importance and threshold
        global_importances = torch.cat(all_importances)
        total_weights = global_importances.numel()

        k = max(0, min(total_weights, int(total_weights * self.s_curr)))
        threshold = torch.kthvalue(global_importances, k).values.item()

        # Updating masks
        for name, importance in importance_cache.items():
            self.masks[name] = torch.gt(importance, threshold).float()


class SNIPIterativePrune(Pruner):
    def prune(self, model):
        all_scores = []
        scores_cache = {}
        for name, param in model.named_parameters():
            if layer_check(name, param):
                mask = self.masks[name]
                score = torch.abs(param * param.grad.detach()) * mask
                scores_cache[name] = score
                all_scores.append(score.view(-1))

        # Calculating global scores and threshold
        global_scores = torch.cat(all_scores)
        norm_factor = torch.sum(global_scores)
        global_scores = global_scores / norm_factor

        k = int(len(global_scores) * (1 - self.s_curr))
        keep_values, _ = torch.topk(global_scores, k, sorted=True)
        threshold = keep_values[-1]
        
        # Updating masks
        for name, score in scores_cache.items():
            self.masks[name] = torch.ge((score / norm_factor), threshold).float()


class WandaPrune(Pruner):
    def __init__(self, model, epochs, s_end, sparsity_scheduler):
        super().__init__(model, epochs, s_end, sparsity_scheduler)
        self.main_layers = {}
        self.wrapped_layers = {}
        self.current_epoch = 0
        self.hooks = []
        self.is_wanda = True
        self.device = next(model.parameters()).device

        self.init_layers(model)
    
    def init_layers(self, model):
        for module_name, module in model.named_modules():
            name = f'{module_name}.weight'
            if hasattr(module, 'weight') and layer_check(name, module.weight):
                self.main_layers[name] = module.to(self.device)
                self.wrapped_layers[name] = self.WrappedLayer(module.to(self.device), name)

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

        k = max(0, min(total_weights, int(total_weights * self.s_curr)))
        threshold = torch.kthvalue(global_importances, k).values.item()

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


class RigLPruner(Pruner):
    def __init__(
        self, 
        model, 
        epochs, 
        s_end, 
        sparsity_scheduler
        ):
        super().__init__(model, epochs, s_end, sparsity_scheduler)
        self.model = model
        self.device = next(model.parameters()).device
        self.prunable_params = {}
        self.init_layers(model)
        self.apply_erk_to_model(model)


    def init_layers(self, model):
        for name, param in model.named_parameters():
            if layer_check(name, param):
                self.prunable_params[name] = param


    def apply_erk_to_model(self, model, erk_power_scale=1.0):
        masks = self._erk_init(model, erk_power_scale)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in masks:
                    param.mul_(masks[name])
        self.masks = masks
        self.s_curr = self.s_end


    def _erk_init(self, model, erk_power_scale):
        erk_scores = {}
        for name, param in self.prunable_params.items():
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

        # normalizing and applying power scale
        total_score = sum(erk_scores.values())
        normalized = {k: (v / total_score) ** erk_power_scale for k, v in erk_scores.items()}

        total_params = float(sum(p.numel() for p in self.prunable_params.values()))
        target_remaining = total_params * (1.0 - self.s_end)

        # Expected remaining per layer
        expected = {k: normalized[k] * self.prunable_params[k].numel() for k in normalized}
        sum_expected = sum(expected.values())
        if sum_expected == 0:
            raise RuntimeError("ERK failed: sum expected is zero")

        scale = float(target_remaining) / float(sum_expected)

        masks = {}
        rng = torch.Generator(device=self.device)
        for name, param in self.prunable_params.items():
            if name in normalized:
                layer_density = normalized[name] * scale  # fraction kept
            else:
                layer_density = 1.0 - self.s_end
            layer_density = float(max(0.0, min(1.0, layer_density)))  # clamp to [0,1]
            keep_prob = layer_density

            # mask = (torch.rand_like(param, device=self.device, generator=rng) < keep_prob).to(dtype=param.dtype)
            mask = (torch.rand_like(param, device=self.device) < keep_prob).to(dtype=param.dtype)
            masks[name] = mask
        return masks


    def prune(self, model):
        prune_imps, prune_cache = [], {}
        regrow_imps, regrow_cache = [], {}

        total_weights = 0
        for name, param in model.named_parameters():
            if name in self.masks:
                mask = self.masks[name]

                # Active weight importance
                prune_imp = torch.abs(param) * mask
                prune_cache[name] = prune_imp
                prune_imps.append(prune_imp.view(-1))

                # Inactive gradient importance
                grad = param.grad
                if grad is None:
                    regrow_imp = torch.zeros_like(prune_imp)
                else:
                    regrow_imp = torch.abs(grad.data) * (1.0 - mask)
                regrow_cache[name] = regrow_imp
                regrow_imps.append(regrow_imp.view(-1))

                total_weights += mask.numel()


        global_prune_imps = torch.cat(prune_imps)
        global_regrow_imps = torch.cat(regrow_imps)

        total_prune_weights = (global_prune_imps > 0).sum().item()
        total_regrow_weights = (global_regrow_imps > 0).sum().item()
        total_weights = global_prune_imps.numel()

        k = int(total_weights * self.s_target)
        k = min(k, total_prune_weights, total_regrow_weights)
        if k == 0:
            return

        # Removing pruned weights to focus only on active/inactive weights
        active_prune_imps = global_prune_imps[global_prune_imps > 0]
        regrow_prune_imps = global_regrow_imps[global_regrow_imps > 0]

        prune_values, prune_indices = torch.topk(active_prune_imps, k, largest=False)
        regrow_values, regrow_indices = torch.topk(regrow_prune_imps, k, largest=True)

        prune_thresh = prune_values.max().item()
        regrow_thresh = regrow_values.min().item()

        for name, param in model.named_parameters():
            if name in self.masks:
                old_mask = self.masks[name]
                numel = old_mask.numel()

                prune_imp = prune_cache[name]
                prune_mask = torch.gt(prune_imp, prune_thresh).float()

                regrow_imp = regrow_cache[name]
                regrow_mask = torch.ge(regrow_imp, regrow_thresh).float()
                param.data *= (1 - regrow_mask)

                new_mask = prune_mask + regrow_mask
                new_mask = torch.clamp(new_mask, 0, 1)

                self.masks[name] = new_mask


PRUNING_REGISTRY = {
    "magnitude_pruning": MagnitudePrune,
    "snip_pruning":      SNIPIterativePrune,
    "wanda_pruning":     WandaPrune,
    "rigl_pruning":      RigLPruner,
}




