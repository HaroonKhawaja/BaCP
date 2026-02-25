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
        'encoder_head',
    ]

    if any(keyword in name.lower() for keyword in exclusion_keywords):
        return False
    return True



class BasePruner(ABC):
    def __init__(self, model, total_epochs, target_sparsity, **kwargs):
        self.model = model
        self.device = next(model.parameters()).device
        self.total_epochs = total_epochs
        self.target_sparsity = target_sparsity

        self.scheduler_type = kwargs.get('scheduler_type', None)
        self.delta_T = kwargs.get('delta_T', None)
        self.total_steps = kwargs.get('total_steps', None)
        if self.scheduler_type is None or self.delta_T is None or self.total_steps is None:
            raise ValueError("scheduler_type, delta_T, and total_steps must be provided.")

        self.final_idx = self.total_steps if self.total_steps else self.total_epochs
        self.end_idx = round(self.final_idx * 0.8)
        print(f"  > Recovery : {(self.final_idx - self.end_idx):,.0f} {'steps' if self.delta_T else 'epochs'} / {self.final_idx}")

        self.current_sparsity = 0.0
        self.prunable_params = self._init_prunable_params()
        self.masks = self._init_dense_masks()

    def _init_prunable_params(self):
        prunable_params = {}
        for name, param in self.model.named_parameters():
            if layer_check(name, param):
                prunable_params[name] = param
        return prunable_params


    def _init_dense_masks(self):
        masks = {}
        for name, param in self.prunable_params.items():
            masks[name] = torch.ones_like(param)        
        return masks


    def calculate_erk_densities(self, model):
        layer_names = []
        layer_params = [] 
        layer_mass = []
        
        for name, param in model.named_parameters():
            if layer_check(name, param):
                n_out = param.shape[0]
                n_in = np.prod(param.shape[1:])
                
                layer_names.append(name)
                layer_params.append(param.numel())
                layer_mass.append(n_in + n_out)

        layer_params = np.array(layer_params)
        layer_mass = np.array(layer_mass)
        
        # The total number of parameters we are allowed to keep
        total_params = np.sum(layer_params)

        global_density = 1 - self.target_sparsity
        target_params = int(global_density * total_params)
        
        # 2. Iterative Solver
        # We maintain a mask of layers that have NOT yet been fixed to 1.0
        is_unconstrained = np.ones(len(layer_names), dtype=bool)
        
        densities = np.zeros(len(layer_names))
        
        while True:
            # Calculate the scaling factor for the unconstrained layers
            # Scale = (Remaining Budget) / (Sum of ER Mass of Unconstrained Layers)
            
            # Budget already used by layers fixed at 1.0
            used_budget = np.sum(layer_params[~is_unconstrained])
            remaining_budget = target_params - used_budget
            
            # Mass of layers we can still adjust
            remaining_mass = np.sum(layer_mass[is_unconstrained])
            
            # Standard ERK formula: density = scale * (er_mass / num_params)
            # But we solve for scale first:
            # scale * remaining_mass = remaining_budget
            scale = remaining_budget / remaining_mass
            
            # Calculate proposed densities for unconstrained layers
            # density[i] = scale * (mass[i] / params[i])
            # We only update indices where is_unconstrained is True
            current_densities = scale * (layer_mass / layer_params)
            
            # Check constraints
            # Identify layers that exceed density 1.0
            # We only care about violations in the currently unconstrained set
            violations = (current_densities > 1.0) & is_unconstrained
            
            if not np.any(violations):
                # No layers violated the 1.0 limit. We are done.
                densities[is_unconstrained] = current_densities[is_unconstrained]
                break
            
            # Fix violating layers to 1.0 and remove them from the solver loop
            densities[violations] = 1.0
            is_unconstrained[violations] = False
            
            # Proceed to next iteration to redistribute the budget...

        # 3. Format Output
        density_dict = {name: d for name, d in zip(layer_names, densities)}
        return density_dict


    def step(self, current_idx, optimizer=None):
        t = current_idx + 1
        T = self.end_idx
        s_final = self.target_sparsity

        if self.scheduler_type == 'linear':
            self.current_sparsity = (s_final / T) * t

        elif self.scheduler_type == 'cubic':
            self.current_sparsity = s_final * (1 - (1 - t/T)**3)
            
        elif self.scheduler_type in ('one_shot', 'f_decay', 'cyclic'):
            self.current_sparsity = s_final
        
        self.current_sparsity = min(self.current_sparsity, s_final)

        self.optimizer = optimizer

        self.update_masks(t)
        self.apply_mask()

    
    @torch.no_grad()
    def apply_mask(self):
        """Enforces the mask on the model weights."""
        for name, param in self.model.named_parameters():
            if name in self.masks:
                param.data.mul_(self.masks[name])
    
    @torch.no_grad()
    def reset_optimizer_states(self, name, param):
        if param in self.optimizer.state:
            state = self.optimizer.state[param]
            mask = self.masks[name]

            # SGD Optimizer
            if 'momentum_buffer' in state:
                state['momentum_buffer'].mul_(mask)

            # Adam/AdamW Optimizer    
            if 'exp_avg' in state:
                state['exp_avg'].mul_(mask)
            if 'exp_avg_sq' in state:
                state['exp_avg_sq'].mul_(mask)
            if 'max_exp_avg_sq' in state:
                state['max_exp_avg_sq'].mul_(mask)

    @abstractmethod
    def update_masks(self, epoch):
        pass

    # Non-monotonic schedulers
    def f_decay_scheduler(self, t, T, alpha):
        t = torch.tensor(t, dtype=torch.float)
        self.s_target = 0.5 * alpha * (1 + torch.cos(math.pi * t / T)).item()

    def cyclic_scheduler(self, t, Tc, s_min, s_max):
        cosine = torch.cos(2*torch.tensor(np.pi) * t  / Tc)
        self.s_target = s_min + ((s_max - s_min) / 2) * (1 - cosine)


class GlobalMagnitudePruner(BasePruner):
    def update_masks(self, epoch):
        if self.current_sparsity == 0: return

        # Calculating scores
        all_scores = {}
        for name, param in self.prunable_params.items():
            all_scores[name] = torch.abs(param)
        
        # Calculating global threshold
        global_scores = torch.cat([
            score.view(-1) for score in all_scores.values()
        ])
        k = int(global_scores.numel() * self.current_sparsity)
        if k == 0: return
        threshold = torch.kthvalue(global_scores, k).values.item()

        # Updating masks
        for name, param in self.prunable_params.items():
            score = all_scores[name]
            self.masks[name] = (score > threshold).float()


class LocalMagnitudePrune(BasePruner):
    def __init__(self, model, total_epochs, target_sparsity, scheduler_type):
        super().__init__(model, total_epochs, target_sparsity, scheduler_type)
        self.erk_densities = self.calculate_erk_densities()

    def update_masks(self, epoch):
        if self.current_sparsity == 0: return

        progress = self.current_sparsity / self.target_sparsity
        for name, param in self.prunable_params.items():
            final_density = self.erk_densities.get(name, 1.0 - self.target_sparsity)
            current_sparsity = (progress * (1.0 - final_density))

            k = int(param.numel() * current_sparsity)
            if k < 1: continue

            score = torch.abs(param)
            threshold = torch.kthvalue(score.view(-1), k).values.item()
            self.masks[name] = (score > threshold).float()


class SNIPPruner(BasePruner):
    """Prunes based on Connection Sensitivity (Weight * Gradient)."""

    def update_masks(self, epoch):
        if self.current_sparsity == 0: return

        # Calculating scores
        all_scores = {}
        for name, param in self.prunable_params.items():
            if param.grad is None: continue
            all_scores[name] = torch.abs(param * param.grad)
            
        if not all_scores: return

        # Calculating global threshold
        global_scores = torch.cat([
            score.view(-1) for score in all_scores.values()
        ])
        k = int(global_scores.numel() * self.current_sparsity)
        if k == 0: return
        threshold = torch.kthvalue(global_scores, k).values.item()

        # Updating masks
        for name, param in self.prunable_params.items():
            score = all_scores[name]
            self.masks[name] = (score > threshold).float()


# class WandaPrune(Pruner):
#     """Weight and Activation pruning."""

#     class WrappedLayer:
#         def __init__(self, layer, name):
#             self.layer = layer
#             self.scaler_row = None
#             self.nsamples = 0

#             if hasattr(layer, 'weight'):
#                 self.rows = layer.weight.shape[0]
#                 self.cols = layer.weight.shape[1]
#                 self.device = layer.weight.device
#                 self.scaler_row = torch.zeros(self.cols, device=self.device)

#         def add_batch(self, inp):
#             if self.scaler_row is None: return

#             if len(inp.shape) == 2: inp = inp.unsqueeze(0)
#             tmp = inp.shape[0]

#             if len(inp.shape) == 3: inp = inp.reshape((-1, inp.shape[-1]))
#             elif len(inp.shape) == 4: inp = inp.permute(0, 2, 3, 1).reshape(-1, inp.shape[1])

#             scaler = torch.norm(inp, p=2, dim=1) ** 2 / tmp
#             self.scaler_row += scaler
#             self.nsamples += 1


#     def __init__(self, model, total_epochs, target_sparsity, scheduler_type):
#         super().__init__(model, total_epochs, target_sparsity, scheduler_type)
#         self.wrapped_layers = {}
#         self._wrap_layers()
#         self.hooks = self.register_hooks()


#     def _wrap_layers(self):
#         check = False
#         for name, module in self.model.named_modules():
#             if name in self.prunable_params:
#                 check = True
#                 self.wrapped_layers[name] = self.WrappedLayer(module)
#         if check: print("[Wanda] Layers are wrapped.")
    

#     def activation_hook(self, name):
#         def hook(_, inputs, outputs):
#             if self.model.training:
#                 input_tensor = inputs[0] if isinstance(inputs, tuple) else inputs
#                 if name, wrapper in self.wrapped_layers.items():
#                     wrapper.add_batch(input_tensor.data.to(self.device))
#         return hook


#     def register_hooks(self):
#         print("[Wanda] Adding hooks")
#         for name, wrapper in self.wrapped_layers.items():
#             hook = wrapper.layer.register_forward_hook(
#                 self.activation_hook(name)
#             )
#             self.hooks.append(hook) 


#     def remove_hooks(self):
#         for hook in self.hooks: hook.remove()
#         self.hooks = []


#     def calibrate(self, loader, num_batches=100):
#         from training_utils import _handle_data_to_device

#         print(f"[Wanda] Calibrating with {num_batches} batches.")
#         model.eval()

#         with torch.no_grad():
#             for i, batch in enumerate(trainloader):
#                 if i > num_calibration_batches: break
#                 data, labels = _handle_data_to_device(self, batch)

#                 # Forward pass
#                 if len(data) == 2:
#                     for d in data:
#                         _ = model(d, return_emb=True)
#                 else:
#                     _ = model(data)

#                 # Print progress every 10 batches
#                 if (i + 1) % 10 == 0:
#                     print(f"  > Calibration progress: {i+1}/{num_calibration_batches} batches")
        
#         self.remove_hooks()
#         self.model.train()
#         print("[Wanda] Calibration done.")

    
#     def update_masks(self, epoch):
#         raise NotImplementedError()

#         # if self.current_sparsity == 0: return

#         # all_importances = []
#         # importance_cache = {}

#         # for name in self.wrapped_layers:
#         #     if self.wrapped_layers[name].scaler_row is None:
#         #         continue

#         #     W = self.main_layers[name].weight.data
#         #     scaler_row = self.wrapped_layers[name].scaler_row.to(self.device).float()
#         #     scaler_row = torch.clamp(torch.nan_to_num(scaler_row, nan=0.0, posinf=1e6, neginf=0.0), min=0.0)

#         #     if W.dim() == 4:
#         #         scaler_row = scaler_row.reshape(1, -1, 1, 1)
#         #     else: 
#         #         scaler_row = scaler_row.reshape(1, -1)

#         #     # Calculating local importance and threshold
#         #     importance = (torch.abs(W) * torch.sqrt(scaler_row + 1e-10))
#         #     importance_cache[name] = importance
#         #     all_importances.append(importance.view(-1))

#         # # Calculating global importance and threshold
#         # global_importances = torch.cat(all_importances)
#         # total_weights = global_importances.numel()

#         # k = max(0, min(total_weights, int(total_weights * self.s_curr)))
#         # threshold = torch.kthvalue(global_importances, k).values.item()

#         # # Updating masks
#         # for name, importance in importance_cache.items():
#         #     self.masks[name] = torch.gt(importance, threshold).float()


class RigLPruner(BasePruner):
    def __init__(self, model, total_epochs, target_sparsity, alpha=0.3, **kwargs):
        super(RigLPruner, self).__init__(model, total_epochs, target_sparsity, **kwargs)
        self.alpha = alpha
        self.erk_densities = self.calculate_erk_densities(model)
        self._init_sparse_masks()
        self.apply_mask()

    def _init_sparse_masks(self):
        print("[RigL] Initializing Sparse Masks")
        for n, p in self.prunable_params.items():
            if n in self.erk_densities:
                self.masks[n] = torch.bernoulli(torch.full_like(p, self.erk_densities[n]))

    def update_masks(self, current_idx):
        self.f_decay_scheduler(current_idx, self.end_idx, self.alpha)

        for name, param in self.prunable_params.items():
            if param.grad is None: continue
            mask = self.masks[name]
            total_active = mask.sum().item()
            k = int(total_active * self.s_target)
            if k < 1: continue

            param_view = torch.abs(param.data).view(-1)
            grad_view = torch.abs(param.grad.data).view(-1)
            mask_view = mask.view(-1)

            mag_active = param_view.clone()
            mag_active[mask_view == 0] = float('inf')

            drop_indices = torch.topk(mag_active, k, largest=False).indices

            survivor_mask = mask_view.clone()
            survivor_mask[drop_indices] = 0.0

            grad_candidates = grad_view.clone()
            grad_candidates[survivor_mask == 1] = float('-inf')

            grow_indices = torch.topk(grad_candidates, k, largest=True).indices

            new_mask = survivor_mask 
            new_mask[grow_indices] = 1.0
            self.masks[name] = new_mask.view_as(mask)
            param.data.view(-1)[grow_indices] = 0.0

            if hasattr(self, 'optimizer') and self.optimizer is not None:
                self.reset_optimizer_states(name, param)
                
           
class EASTPruner(BasePruner):
    def __init__(self, model, total_epochs, target_sparsity, **kwargs):
        super(EASTPruner, self).__init__(model, total_epochs, target_sparsity, **kwargs)
        self.s_max = target_sparsity
        self.delta_T = kwargs.get('delta_T', None)
        self.prune_rate = kwargs.get('prune_rate', None)
        self.Tc_ratio = kwargs.get('Tc_ratio', None)
        
        # Use end_idx to determine Cycle length regardless of Epoch vs Step mode
        self.Tc = int(self.end_idx * self.Tc_ratio) 
        
        self.erk_densities = self.calculate_erk_densities(model)
        self._init_sparse_masks()
        self.apply_mask()

    def _init_sparse_masks(self):
        print(f"[EAST] Initializing Sparse Masks at s_max={self.s_max}")
        for n, p in self.prunable_params.items():
            if n in self.erk_densities:
                self.masks[n] = torch.bernoulli(torch.full_like(p, self.erk_densities[n]))

    def update_masks(self, current_idx):
        if current_idx % self.delta_T != 0: return
        is_cyclic = current_idx <= self.Tc
        
        if is_cyclic:
            # 1. Calculate the global target sparsity (s_target) based on the cosine schedule
            norm_t = current_idx / self.Tc
            s_min = getattr(self, 's_min', 0.05) # Define s_min, assuming it was passed in kwargs or has a default
            s_target = s_min + (self.s_max - s_min) * 0.5 * (1 - math.cos(2 * math.pi * norm_t))
            
            # 2. Calculate current global sparsity (s_curr)
            total_active_global = sum(m.sum().item() for m in self.masks.values())
            total_params_global = sum(m.numel() for m in self.masks.values())
            s_curr = 1.0 - (total_active_global / total_params_global)
            
            # 3. Calculate global difference (positive means we need to prune more)
            diff = s_target - s_curr
        else:
            diff = 0
            s_target = self.s_max
            s_curr = self.s_max

        # --- Vectorized Update (Preserving ERK Distribution) ---
        for name, param in self.prunable_params.items():
            if param.grad is None: continue
            
            mask = self.masks[name]
            numel = param.numel()
            active_params = mask.sum().item()

            # Determine k for this specific layer
            if is_cyclic:
                # To preserve ERK, we scale the currently active parameters of THIS layer
                # by the ratio required to reach the global target sparsity.
                if s_curr < 1.0: # Prevent division by zero if network is completely dense
                    target_active = active_params * ((1.0 - s_target) / (1.0 - s_curr))
                else:
                    target_active = active_params
                
                if diff > 0: # Pruning phase (s_target > s_curr, so we need fewer active params)
                    k_prune = max(0, int(active_params - target_active))
                    k_grow = 0
                elif diff < 0: # Growing phase (s_target < s_curr, so we need more active params)
                    k_grow = max(0, int(target_active - active_params))
                    k_prune = 0
                else:
                    k_prune = k_grow = 0
            else:
                # Non-cyclic phase: drop and grow a flat percentage of currently active connections
                k_prune = k_grow = int(active_params * self.prune_rate)

            if k_prune < 1 and k_grow < 1: continue

            new_mask = mask.clone().view(-1)

            # 1. Prune (Weakest Active)
            if k_prune > 0:
                # Add a large penalty to inactive weights so they are not selected for pruning
                mag_active = torch.abs(param) + (1.0 - mask) * 1e9
                prune_idx = torch.topk(mag_active.view(-1), k_prune, largest=False).indices
                new_mask[prune_idx] = 0.0

            # 2. Grow (Strongest Gradient Inactive)
            if k_grow > 0:
                # Only consider gradients of currently inactive weights
                grad_inactive = torch.abs(param.grad) * (1.0 - mask)
                grow_idx = torch.topk(grad_inactive.view(-1), k_grow, largest=True).indices
                new_mask[grow_idx] = 1.0

            new_mask = new_mask.view_as(mask)
            
            # 3. Zero-Init New Weights
            if k_grow > 0:
                param.data[(new_mask == 1) & (mask == 0)] = 0.0
            
            self.masks[name] = new_mask
            
            # 4. Reset Optimizer States
            if hasattr(self, 'optimizer') and self.optimizer is not None:
                self.reset_optimizer_states(name, param)


            
PRUNER_REGISTRY = {
    "magnitude": GlobalMagnitudePruner,
    "local_magnitude": LocalMagnitudePrune,
    "snip": SNIPPruner,
    "rigl": RigLPruner,
    "east": EASTPruner,
    # "wanda": WandaPruner
}

def get_pruner(method_name, model, epochs, sparsity, **kwargs):
    if method_name not in PRUNER_REGISTRY:
        raise ValueError(f"Pruner {method_name} not found. Available: {list(PRUNER_REGISTRY.keys())}")
    return PRUNER_REGISTRY[method_name](model, epochs, sparsity, **kwargs)


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

    if verbose:
        print("\n" + "="*100)
        print(f"{'LAYER NAME':<50} | {'SPARSITY':<10} | {'ZERO / TOTAL PARAMS':<25}")
        print("-" * 100)

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
                print(f"{name:<50} | {layer_sparsity * 100:>7.6f}%   | {num_zero_weights:>9,} / {num_layer_weights:<9,}")

    # Calculate Summaries
    overall_sparsity = total_zero_weights / total_weights if total_weights > 0 else 0
    backbone_sparsity = total_backbone_zero_weights / total_backbone_weights if total_backbone_weights > 0 else 0
    non_zero_params = total_weights - total_zero_weights
    active_backbone_params = total_backbone_weights - total_backbone_zero_weights

    if verbose:
        print("-" * 100)
        print(f"{'MODEL SUMMARY':<50} | {'VALUE':<10} | {'DETAILS':<25}")
        print("-" * 100)
        print(f"{'Overall Model Sparsity':<50} | {overall_sparsity * 100:>7.4f}%   | -")
        print(f"{'Backbone Sparsity':<50} | {backbone_sparsity * 100:>7.4f}%   | {total_backbone_zero_weights:,} / {total_backbone_weights:,} (Zero/Total)")
        print(f"{'Active Backbone Parameters':<50} | {active_backbone_params:>10,} | -")
        print(f"{'Total Parameters':<50} | {total_weights:>10,} | -")
        print(f"{'Non-Zero Parameters (Active)':<50} | {non_zero_params:>10,} | -")
        print(f"{'Zero Parameters (Pruned)':<50} | {total_zero_weights:>10,} | -")
        print("=" * 100 + "\n")

    # Plotting
    if names: # Only plot if we found backbone layers
        plt.figure(figsize=(10, 0.2 * len(names)))
        bars = plt.barh(range(len(names)), sparsities, color='skyblue')
        plt.yticks(range(len(names)), names)
        plt.xlabel("Sparsity (Fraction of Zero Weights)")
        plt.title("Layer-wise Sparsity Distribution")
        plt.grid(axis="x", linestyle="--", alpha=0.5)
        plt.xlim(0, 1.0)
        
        # Add a vertical line for target sparsity if high
        plt.axvline(x=backbone_sparsity, color='r', linestyle=':', label=f'Avg: {backbone_sparsity:.2f}')
        plt.legend()

        plt.tight_layout()
        plt.show()
    else:
        print("No backbone layers found to plot.")



