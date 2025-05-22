import torch
from abc import abstractmethod, ABC
from utils import *
import os
from functools import partial

def layer_check(name, param):
    if param.dim() > 1 and param.requires_grad and not any(keyword in name for keyword in ['fc', 'classifier', 'embeddings', 'conv1']):
        return True
    return False

def check_model_sparsity(model):
    sum_zero, total_count = 0, 0
    for name, param in model.named_parameters():
        if layer_check(name, param):
            sum_zero += torch.sum(param == 0).item()
            total_count += param.numel()

    return sum_zero / total_count

class Pruner(ABC):
    def __init__(self, epochs, target_ratio, sparsity_scheduler="linear", target_layer='no_target_layer'):
        # assert target_layer in ['no_target_layer', 'self_attention'], 'Invalid layer name'
        self.epochs = epochs
        self.target_ratio = target_ratio
        self.ratio = target_ratio / epochs
        self.amount = self.ratio
        self.masks = {}
        self.target_layer = target_layer
        self.is_wanda = False

        assert sparsity_scheduler in ["linear", "cubic"], "Invalid sparsity scheduler"
        self.sparsity_scheduler = sparsity_scheduler
        
    @abstractmethod
    def prune(self, model):
        pass
            
    @torch.no_grad()  
    def apply_mask(self, model):
        for name, param in model.named_parameters():
            if layer_check(name, param):
                if name in self.masks:
                    param.data *= self.masks[name]
                    if param.grad is not None:
                        param.grad.data *= self.masks[name]

    def ratio_step(self):
        if self.sparsity_scheduler == "linear":
            self.linear_ratio_step()
        else:
            return
            
    def linear_scheduler(self):
        if self.sparsity_scheduler == "linear":
            self.ratio += self.amount
            self.ratio = min(self.ratio, self.target_ratio)
            print(f"\n[Pruner] Linear sparsity ratio increased to {self.ratio:.3f} for the next pruning iteration.\n")

    def cubic_scheduler(self, step, t0, delta_t, s_initial):
        if self.sparsity_scheduler == "cubic":
            progress = min(1.0, (step - t0) / (self.epochs * delta_t))
            self.ratio = self.target_ratio + (s_initial - self.target_ratio) * ((1 - progress)**3)
            self.ratio = min(self.ratio, self.target_ratio)
            print(f"\n[Pruner] Cubic Sparsity ratio increased to {self.ratio:.3f} for the next pruning iteration.\n")

    def reset(self):
        self.ratio = self.target_ratio / self.epochs
        print(f"[Pruner] Sparsity ratio reset to initial value: {self.ratio:.3f}.\n")
    
    def set_mask(self, masks):
        self.masks = masks

class MagnitudePrune(Pruner):
    def prune(self, model):
        all_importances = []
        for name, param in model.named_parameters():
            if layer_check(name, param):
                if name not in self.masks:
                    self.masks[name] = torch.ones_like(param)
                
                # Calculating importance
                importance = torch.abs(self.masks[name] * param)
                all_importances.append(importance.view(-1))

        # Calculating global importance and threshold
        global_importances = torch.cat(all_importances)
        total_weights = global_importances.numel()
        num_to_zero = max(1, min(total_weights, round(total_weights * self.ratio)))
        threshold = torch.kthvalue(global_importances, num_to_zero).values.item()
        
        # Updating masks
        for name, param in model.named_parameters():
            if name in self.masks:
                importance = torch.abs(self.masks[name] * param)
                new_mask = torch.gt(importance, threshold).float()
                self.masks[name] = new_mask

class LocalMagnitudePrune(Pruner):
    def prune(self, model):
        for name, param in model.named_parameters():
            if param.dim() > 1 and param.requires_grad and self.target_layer not in name:
                if name not in self.masks:
                    self.masks[name] = torch.ones_like(param)
                
                # Calculating local importance and threshold
                importance = torch.abs(self.masks[name] * param).view(-1)
                total_weights = importance.numel()
                num_to_zero = max(1, min(total_weights, round(total_weights * self.ratio)))
                threshold = torch.kthvalue(importance, num_to_zero).values.item()

                # Updating masks
                new_mask = torch.gt(importance.view(param.shape), threshold).float()
                self.masks[name] = new_mask

class MovementPruner(Pruner):
    def __init__(self, model, pruning_epochs, target_sparsity, scheduler="linear", momentum=0.9):
        super().__init__(pruning_epochs, target_sparsity, scheduler)
        self.momentum = momentum
        self.scores   = {}
        self.masks    = {}
        for name, param in model.named_parameters():
            if layer_check(name, param):
                self.scores[name] = torch.zeros_like(param.data)
                self.masks[name]  = torch.ones_like(param.data)
                param.register_hook(partial(self._accumulate, name, param))

    def _accumulate(self, name, param, grad):
        inst_move = param.data * grad
        self.scores[name].mul_(self.momentum).add_(inst_move, alpha=1 - self.momentum)
        return grad

    def prune(self, model):
        all_scores = torch.cat([
            torch.abs(self.scores[name] * self.masks[name]).view(-1)
            for name in self.scores
        ])
        k = max(1, int(round(all_scores.numel() * self.ratio)))
        thresh = torch.kthvalue(all_scores, k)[0].item()
        for name, param in model.named_parameters():
            if name in self.masks:
                score = torch.abs(self.scores[name] * self.masks[name])
                mask  = (score > thresh).to(param.dtype)
                self.masks[name] = mask
                if param.grad is not None:
                    param.grad.data.mul_(mask)
        for name, param in model.named_parameters():
            if name in self.masks:
                param.data.mul_(self.masks[name])

class LocalMovementPrune(Pruner):
    def prune(self, model):
        for name, param in model.named_parameters():
            if param.dim() > 1 and param.requires_grad and self.target_layer not in name:
                if name not in self.masks:
                    self.masks[name] = torch.ones_like(param)

                # Calculating local importance and threshold
                importance = torch.abs(self.masks[name] * param * param.grad).view(-1)
                total_weights = importance.numel()
                num_to_zero = max(1, min(total_weights, round(total_weights * self.ratio)))
                threshold = torch.kthvalue(importance, num_to_zero).values.item()

                # Updating masks
                new_mask = torch.gt(importance.view(param.shape), threshold).float()
                self.masks[name] = new_mask
    
class WandaPrune(Pruner):
    def __init__(self, epochs, target_ratio, model, pruning_scheduler):
        super().__init__(epochs, target_ratio, pruning_scheduler)
        self.main_layers = {}
        self.wrapped_layers = {}
        self.current_epoch = 0
        self.hooks = []
        self.is_wanda = True
        self.device = model.model.device

        if hasattr(model.model, 'distilbert') and hasattr(model.model.distilbert, 'transformer') and hasattr(model.model.distilbert.transformer, 'layer'):
            self.transformer_layers = model.model.distilbert.transformer.layer
            self.prefix = "distilbert.transformer.layer"
        else:
            raise ValueError("Model does not contain layers.")

        self.init_layers(model)
    
    def init_layers(self, model):
        for layer_idx, layer in enumerate(self.transformer_layers):
            for name, module in model.named_modules():
                for child_name, child in module.named_children():
                    if isinstance(child, torch.nn.Linear) and self.prefix in name:
                        full_name = f"{name}.{child_name}"
                        self.main_layers[full_name] = child.to(self.device)
                        self.wrapped_layers[full_name] = self.WrappedLayer(child.to(self.device), full_name)

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
                output_tensor = outputs[0] if isinstance(outputs, tuple) else outputs
                if name in self.wrapped_layers:
                    self.wrapped_layers[name].add_batch(input_tensor.data.to(self.device), output_tensor.data.to(self.device))
        return hook

    def prune(self, model):
        for name in self.wrapped_layers:
            if self.wrapped_layers[name].scaler_row is None:
                continue

            W = self.main_layers[name].weight.data
            scaler_row = self.wrapped_layers[name].scaler_row.to(self.device).float()
            scaler_row = torch.clamp(torch.nan_to_num(scaler_row, nan=0.0, posinf=1e6, neginf=0.0), min=0.0)

            name = f'{name}.weight'
            if name not in self.masks:
                self.masks[name] = torch.ones_like(W)

            # Calculating local importance and threshold
            importance = (torch.abs(W) * torch.sqrt(scaler_row.reshape(1, -1) + 1e-10)).view(-1)
            total_weights = importance.numel()
            num_to_zero = max(1, min(total_weights, round(total_weights * self.ratio)))
            threshold = torch.kthvalue(importance, num_to_zero).values.item()

            # Updating masks
            new_mask = torch.gt(importance.view(W.shape), threshold).float()
            self.masks[name] = new_mask

        self.remove_hooks()

    class WrappedLayer:
        def __init__(self, layer, name):
            self.layer = layer
            self.device, self.rows, self.cols, self.scaler_row = None, 0, 0, None
            if hasattr(self.layer, 'weight') and self.layer.weight is not None:
                self.rows, self.cols = self.layer.weight.shape
                self.device = self.layer.weight.device
                self.scaler_row = torch.zeros((self.cols), device=self.device)
            self.layer_name = name
            self.nsamples = 0

        def add_batch(self, input_tensor, output_tensor):
            if self.scaler_row is None:
                return
            
            if len(input_tensor.shape) == 2:
                input_tensor = input_tensor.unsqueeze(0)
            tmp = input_tensor.shape[0]

            if len(input_tensor.shape) == 3:
                input_tensor = input_tensor.reshape(-1, input_tensor.shape[-1])

            input_tensor = input_tensor.to(self.device).type(torch.float32)

            self.scaler_row = self.scaler_row.float()
            self.scaler_row *= self.nsamples / (self.nsamples + tmp)
            self.nsamples += tmp

            batch_norm = torch.norm(input_tensor, dim=0, p=2) ** 2
            self.scaler_row += batch_norm / self.nsamples

PRUNER_DICT = {
    "magnitude_pruning": MagnitudePrune,
    "local_magnitude_pruning": LocalMagnitudePrune,
    "movement_pruning": MovementPrune,
    "local_movement_pruning": LocalMovementPrune,
    "wanda_pruning": WandaPrune
}




