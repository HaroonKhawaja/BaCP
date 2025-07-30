import os
import torch
import torch.nn as nn
from abc import abstractmethod, ABC
from utils import *

def layer_check(name, param):
    if param.dim() <= 1 or not param.requires_grad:
        return False
    
    exclusion_keywords = [
        'fc',   # ResNet
        'classifier.6', # VGG
        'embeddings', 'conv_proj', 'pos_embedding', 'heads', 'classifier.weight', # ViT        
        'projection_head', # Encoder heads
        'vocab_projector', 'vocab_transform', # DistilBERT

        'classifier.dense', 'classifier.out_proj', # RoBERTA

        # 'heads', 'conv_proj', 'fc', 'classifier', 'embeddings', 
        # 'class_token', 'pos_embedding', 'vocab_transform', 'lm_head',
        # 'projection_head'
        ]
    if any(keyword in name for keyword in exclusion_keywords):
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
    def __init__(self, epochs, target_ratio, model, sparsity_scheduler="cubic"):
        self.epochs = epochs
        self.target_ratio = target_ratio
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
            self.linear_scheduler()
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

class MovementPrune(Pruner):
    def __init__(self, epochs, target_ratio, model, pruning_scheduler):
        super().__init__(epochs, target_ratio, model, pruning_scheduler)
        self.movement_scores = {
            name: torch.zeros_like(param)
            for name, param in model.named_parameters()
            if layer_check(name, param)
        }

    @torch.no_grad()
    def update_movement_scores(self, model, lr):
        for name, param in model.named_parameters():
            if name not in self.movement_scores:
                continue
            self.movement_scores[name] += -lr * param.grad

    def prune(self, model):
        all_importances = []
        importance_cache = {}

        for name, param in model.named_parameters():
            if layer_check(name, param):
                importance = self.movement_scores[name] * param.data
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
    def __init__(self, epochs, target_ratio, model, pruning_scheduler):
        super().__init__(epochs, target_ratio, model, pruning_scheduler)
        self.main_layers = {}
        self.wrapped_layers = {}
        self.current_epoch = 0
        self.hooks = []
        self.is_wanda = True
        self.device = model.model.device if hasattr(model.model, 'device') else get_device()

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

    def remove_hooks(self):
        print("\n[Pruner] Removing hooks")
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

            # Calculating local importance and local threshold
            importance = (torch.abs(W) * torch.sqrt(scaler_row + 1e-10))
            importance_cache[name] = importance
            
            layer_importances  = importance.view(-1)
            total_weights = layer_importances.numel()
            num_to_zero = max(1, min(total_weights, round(total_weights * self.ratio)))
            local_threshold = torch.kthvalue(layer_importances, num_to_zero).values.item()

            self.masks[name] = torch.gt(importance, local_threshold).float()

    class WrappedLayer:
        def __init__(self, layer, name):
            self.layer = layer
            self.device, self.rows, self.cols, self.scaler_row = None, 0, 0, None
            if hasattr(self.layer, 'weight') and self.layer.weight is not None:
                self.rows, self.cols = self.layer.weight.shape[0], self.layer.weight.shape[1]
                self.device = self.layer.weight.device
                self.scaler_row = torch.zeros((self.cols), device=self.device)
            else:
                self.device = layer.device
                
            self.layer_name = name
            self.nsamples = 0

        def add_batch(self, input_tensor):
            if self.scaler_row is None:
                return

            # Handling batch size of 1
            if len(input_tensor.shape) == 2:
                input_tensor = input_tensor.unsqueeze(0)
            batch_size = input_tensor.shape[0]

            if len(input_tensor.shape) == 3:
                input_tensor = input_tensor.reshape(-1, input_tensor.shape[-1])
            elif len(input_tensor.shape) == 4:
                input_tensor = input_tensor.permute(0, 2, 3, 1)
                input_tensor = input_tensor.reshape(-1, self.cols)

            input_tensor = input_tensor.to(self.device).type(torch.float32)

            self.scaler_row = self.scaler_row.float()
            self.scaler_row *= self.nsamples / (self.nsamples + batch_size)
            self.nsamples += batch_size

            batch_norm = torch.norm(input_tensor, dim=0, p=2) ** 2
            self.scaler_row += batch_norm / self.nsamples

PRUNER_DICT = {
    "magnitude_pruning": MagnitudePrune,
    "snip_pruning": SNIPIterativePrune,
    "wanda_pruning": WandaPrune
}




