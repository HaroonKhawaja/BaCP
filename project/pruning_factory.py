import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from abc import abstractmethod, ABC
from typing import Dict

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

# Parameters the pruner never owns. Grouped by why, because the choice changes what
# a reported sparsity figure means -- and at extreme sparsity these are not rounding
# errors. A dense Linear(2048, 100) head is 204,800 weights; at 99% sparsity on
# ResNet-50 the whole surviving budget is roughly 250,000.

# EAST's DyReLU hyperfunctions. Correct to exclude: phasing removes them from the
# forward pass by the end of training, so they are not part of the deployed model.
_EXCLUDE_ACTIVATION = ('hyperfunction', 'relu')

# The contrastive projection head. Not part of the task model at all -- it is
# discarded after BaCP training.
_EXCLUDE_PROJECTION = ('encoder_head',)

# Token and position embeddings. Excluding these is not optional for language
# models: DistilBERT and RoBERTa tie word_embeddings to the output projection, so
# pruning the embedding matrix also prunes the LM head. On DistilBERT that single
# tensor is 23.4M of 66M parameters, and including it made the prunable set 99.7%
# of the model -- meaning a "99% sparsity" figure would be dominated by the
# embedding table rather than describing the transformer blocks.
_EXCLUDE_EMBEDDINGS = ('embedding',)

# Task heads. Matches the protocol stated in the paper's appendix D.4 -- "the fc
# layer was kept dense", "the final classifier.6 layer was kept dense",
# "embeddings and classification heads were kept dense".
#
# 'classifier' is in the LLM set only. For HF models the module named
# `classifier` IS the task head (DistilBERT's pre_classifier/classifier,
# RoBERTa's classifier.dense/out_proj). But for VGG, `model.classifier.0` and
# `.3` are the 102.8M + 16.8M BACKBONE Linears -- remove_last_layer strips only
# the final layer and the replacement head is `cls_head` -- so excluding the
# substring there exempted 92.8% of VGG-11 from the sparsity denominator
# (defect M2). Which set applies is decided by the model handed to
# set_prunable_scope; with no model we keep the conservative full set.
_EXCLUDE_TASK_HEAD_CV = (
    'cls_head',                          # this project's replacement classifier
)
_EXCLUDE_TASK_HEAD_LLM = (
    'cls_head',                          # this project's replacement classifier
    'vocab_projector', 'vocab_transform',  # DistilBERT MLM head
    'lm_head',                           # RoBERTa MLM head
    'classifier',                        # HF sequence-classification heads
)

# --- Prunable scope -------------------------------------------------------
#
# Whether the task head and the embeddings are prunable is a *choice*, and it has
# to be the same choice for the pruner, check_model_sparsity and
# check_sparsity_distribution -- otherwise the reported sparsity describes a
# different set of weights than the one being pruned. Hence one module-level
# setting rather than a per-call argument: a single source of truth.
#
# Set it once via set_prunable_scope(), which _initialize_pruner does before any
# sparsity is measured.

_PRUNE_TASK_HEAD = False
_PRUNE_EMBEDDINGS = False
_MODEL_TYPE = None      # 'cv' | 'llm' | None (unknown -> conservative LLM set)


def set_prunable_scope(prune_task_head=False, prune_embeddings=False, model=None):
    """Declare which parameter groups the pruner owns.

    prune_task_head:
        Include the final classifier. The paper's appendix D.4 says heads were kept
        dense ("the fc layer was kept dense"), but pruning everything is the
        convention in dynamic sparse training, so both are legitimate -- just not
        interchangeable. At 99% sparsity on ResNet-50/CIFAR-100 a dense
        Linear(2048, 100) is 204,800 weights against a surviving budget of roughly
        250,000, so this flag moves the headline number substantially. State which
        convention you report.

    prune_embeddings:
        Include token and position embeddings. Off by default, and dangerous for
        transformers: DistilBERT and RoBERTa tie word_embeddings to the output
        projection, so pruning the embedding matrix also prunes the LM head. On
        DistilBERT that single tensor is 23.4M of 66M parameters, which makes the
        prunable set 99.7% of the model -- a "99% sparsity" figure would then be
        describing the embedding table, not the transformer blocks.
    """
    global _PRUNE_TASK_HEAD, _PRUNE_EMBEDDINGS, _MODEL_TYPE
    _PRUNE_TASK_HEAD = bool(prune_task_head)
    _PRUNE_EMBEDDINGS = bool(prune_embeddings)
    # Which task-head keyword set applies (see _EXCLUDE_TASK_HEAD_CV/_LLM).
    # The wrapper carries model_type; with no model we leave it None and
    # excluded_keywords() falls back to the conservative LLM set.
    _MODEL_TYPE = getattr(model, 'model_type', None) if model is not None else None

    scope = ['backbone']
    if _PRUNE_TASK_HEAD:
        scope.append('task head')
    if _PRUNE_EMBEDDINGS:
        scope.append('embeddings')
    print(f"[SPARSITY SCOPE] Prunable: {', '.join(scope)}")

    if _PRUNE_EMBEDDINGS and model is not None and _has_tied_embeddings(model):
        print("  ! WARNING: this model ties its input embeddings to its output "
              "projection. Pruning embeddings therefore also prunes the LM head, "
              "and the embedding table will dominate the sparsity denominator.")


def _has_tied_embeddings(model):
    """True when input and output embeddings share storage."""
    inner = getattr(model, 'model', model)
    get_in = getattr(inner, 'get_input_embeddings', None)
    get_out = getattr(inner, 'get_output_embeddings', None)
    if not callable(get_in) or not callable(get_out):
        return False
    try:
        emb_in, emb_out = get_in(), get_out()
    except Exception:
        return False
    if emb_in is None or emb_out is None:
        return False
    w_in = getattr(emb_in, 'weight', None)
    w_out = getattr(emb_out, 'weight', None)
    return w_in is not None and w_out is not None and w_in.data_ptr() == w_out.data_ptr()


def excluded_keywords():
    """The keyword set layer_check currently excludes, given the active scope."""
    excluded = _EXCLUDE_ACTIVATION + _EXCLUDE_PROJECTION
    if not _PRUNE_EMBEDDINGS:
        excluded += _EXCLUDE_EMBEDDINGS
    if not _PRUNE_TASK_HEAD:
        excluded += (_EXCLUDE_TASK_HEAD_CV if _MODEL_TYPE == 'cv'
                     else _EXCLUDE_TASK_HEAD_LLM)
    return excluded


def layer_check(name, param):
    """True if the pruner owns this parameter.

    Always excludes 1-D tensors (norms and biases), frozen tensors, the DyReLU
    hyperfunctions and the contrastive projection head. The task head and the
    embeddings are governed by set_prunable_scope().
    """
    if param is None:
        return False
    if param.dim() <= 1 or not param.requires_grad:
        return False

    lowered = name.lower()
    if any(keyword in lowered for keyword in excluded_keywords()):
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


    def calculate_erk_densities(self, model=None):
        """Erdos-Renyi-Kernel per-layer densities for the current target sparsity.

        `model` is optional and defaults to self.model. It has to accept both
        forms: LocalMagnitudePrune calls it with no argument (which was a
        TypeError, so that pruner never constructed) while EASTPruner passes the
        model explicitly.
        """
        model = model if model is not None else self.model

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
    """Per-layer magnitude pruning under a declared layer-wise budget.

    `layerwise_alloc` chooses *how the global budget is split across layers*, which
    is a separate mechanism from *which weights within a layer are cut*:

      'erk'     -- Erdos-Renyi-Kernel (Evci et al. 2020). Wide layers get lower
                   density than narrow ones. The default, and what RigL and EAST
                   both use, so it is the like-for-like setting.
      'uniform' -- every layer at the same density, 1 - target_sparsity. This is
                   the honest floor, and it is required to disentangle "ERK helps"
                   from "ERK protects the small classifier layer": the two are
                   confounded in a single rung, because under ERK the classifier's
                   ER mass is large relative to its parameter count and it ends up
                   near-dense whether or not it is nominally in the prunable scope.

    See project/experiments/manifest.py rungs A1-A4, which run these as a 2x2
    against prune_task_head for exactly that reason.
    """

    # BasePruner takes its settings as **kwargs. The old signature accepted
    # scheduler_type positionally and forwarded it as a 4th positional argument,
    # which was a TypeError -- so this pruner could never be constructed, and
    # get_pruner would also have handed it a delta_T it had nowhere to put.
    def __init__(self, model, total_epochs, target_sparsity, **kwargs):
        super().__init__(model, total_epochs, target_sparsity, **kwargs)
        self.layerwise_alloc = kwargs.get('layerwise_alloc') or 'erk'
        if self.layerwise_alloc not in ('erk', 'uniform'):
            raise ValueError(
                f"layerwise_alloc must be 'erk' or 'uniform', got "
                f"{self.layerwise_alloc!r}"
            )
        if self.layerwise_alloc == 'uniform':
            flat = 1.0 - self.target_sparsity
            self.erk_densities = {n: flat for n in self.prunable_params}
        else:
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


class WandaPruner(BasePruner):
    """Weight-and-activation pruning (Sun et al. 2023, arXiv 2306.11695).

    Importance of weight W[i,j] is

        S[i,j] = |W[i,j]| * ||X[:,j]||_2

    where ||X[:,j]||_2 is the L2 norm of the j-th input feature aggregated over a
    calibration set. The insight is that magnitude alone mis-ranks weights whose
    input channel is systematically small or large: a large weight on a dead
    channel contributes nothing, and a small weight on a high-variance channel can
    matter a great deal.

    **The comparison group is a CHOICE here, and the citation does not settle it.**
    Wanda ranks within each output row and reports that grouping as crucial -- but
    only for LLMs. Its own Appendix A runs the same comparison on image
    classifiers and finds the opposite: "we do not observe similar trend in image
    classification models, suggesting that our observations regarding pruning per
    output might be unique to LLMs", with layer-wise slightly better for both
    ConvNeXt-B and DeiT-B. That text is in the ICLR 2024 camera-ready, not a
    preprint artefact.

    This project prunes ResNet/VGG on CIFAR. So there is no Wanda evidence for
    per-output ranking in our setting in either direction, and the only vision
    evidence that exists points the other way. `group` therefore defaults to
    'output' (Wanda's headline configuration, so the column means what its name
    says) but is a first-class, recorded knob, and the paper must present the
    choice as ours rather than lean on the citation. Cite Wanda for the metric
    |W| * ||X||_2, which does transfer; do not cite it for the grouping.

      'output' -- rank within each row of W. Every output neuron keeps the same
                  fraction of its inputs.
      'layer'  -- one threshold for the whole layer, as this file's magnitude and
                  SNIP pruners use. Wanda's vision experiments favour this.

    **The activation norm is accumulated, not sampled.** scaler_row holds a running
    mean of ||X[:,j]||^2 over calibration batches in incremental form, so the
    statistic does not depend on how many batches happened to be available.

    **The Conv2d convention is ours, not Wanda's.** Wanda's reference
    implementation (lib/layerwrapper.py) handles nn.Linear only -- its reshape
    branch is gated on isinstance(layer, nn.Linear) -- so there is no published
    convention to follow for 4-D weights. We fold the kernel into the input
    dimension: a weight of shape (out, in, kh, kw) is scored against F.unfold'd
    input columns, giving the same [rows x columns] matrix Wanda scores for a
    Linear layer. Scoring per (in, kh, kw) position against a per-channel norm
    instead would ignore that different kernel positions see different input
    statistics. Stated here because the paper has to state it.

    Calibration is driven by set_calibration_loader(). Without it every norm falls
    back to 1, which is *exactly magnitude pruning*. That fallback prints a warning
    rather than passing silently, because an unnoticed fallback would put a "WANDA"
    column in a table that actually contains magnitude numbers -- the most
    misleading failure this class could have, and the reason the previous
    commented-out draft was worse than no implementation at all.
    """

    def __init__(self, model, total_epochs, target_sparsity, **kwargs):
        super().__init__(model, total_epochs, target_sparsity, **kwargs)
        self.calibration_loader = kwargs.get('calibration_loader')
        self.calibration_batches = int(kwargs.get('calibration_batches') or 8)
        self.group = kwargs.get('wanda_group') or 'output'
        if self.group not in ('output', 'layer'):
            raise ValueError(
                f"wanda_group must be 'output' or 'layer', got {self.group!r}")
        self.scaler_row = {}          # param name -> Tensor[in_features]
        self._nsamples = {}
        self._calibrated = False
        self._calibration_step = None
        self._fallback_layers = set()
        self._modules = self._find_scored_modules()

    def _find_scored_modules(self):
        """param name -> (module name, module), for prunable Linear/Conv2d weights."""
        out = {}
        for mod_name, module in self.model.named_modules():
            if not isinstance(module, (nn.Linear, nn.Conv2d)):
                continue
            param_name = f'{mod_name}.weight' if mod_name else 'weight'
            if param_name in self.prunable_params:
                out[param_name] = (mod_name, module)
        return out

    def set_calibration_loader(self, loader, batches=None):
        self.calibration_loader = loader
        if batches is not None:
            self.calibration_batches = int(batches)
        self._calibrated = False

    @torch.no_grad()
    def calibrate(self, loader=None, device=None):
        """Accumulate per-input-feature activation norms over a few batches.

        Hooks only the scored modules, so this costs a handful of forward passes
        and no backward pass.
        """
        loader = loader or self.calibration_loader
        if loader is None:
            print('  ! [WANDA] no calibration loader: activation norms default to '
                  '1.0, which makes this EXACTLY magnitude pruning. Any table '
                  'column produced from this run is mislabelled.')
            self._calibrated = True
            return False

        device = device or self.device
        self.scaler_row = {}
        self._nsamples = {}

        def make_hook(param_name):
            def hook(module, inputs, _output):
                x = inputs[0].detach()
                if isinstance(module, nn.Conv2d):
                    # Fold the kernel into the input dimension so the scored
                    # matrix matches the Linear case exactly.
                    x = F.unfold(x, kernel_size=module.kernel_size,
                                 dilation=module.dilation, padding=module.padding,
                                 stride=module.stride)          # [B, C*kh*kw, L]
                    x = x.transpose(1, 2).reshape(-1, x.shape[1])
                else:
                    x = x.reshape(-1, x.shape[-1])
                x = x.float()

                n_new = x.shape[0]
                n_old = self._nsamples.get(param_name, 0)
                prev = self.scaler_row.get(param_name)
                if prev is None:
                    prev = torch.zeros(x.shape[1], device=x.device)

                # Incremental mean of squared column norms, so the statistic does
                # not depend on the number of calibration batches available.
                prev *= n_old / (n_old + n_new)
                prev += x.pow(2).sum(dim=0) / (n_old + n_new)
                self.scaler_row[param_name] = prev
                self._nsamples[param_name] = n_old + n_new
            return hook

        handles = [module.register_forward_hook(make_hook(pname))
                   for pname, (_, module) in self._modules.items()]

        was_training = self.model.training
        self.model.eval()
        seen = 0
        try:
            for i, batch in enumerate(loader):
                if i >= self.calibration_batches:
                    break
                seen = i + 1
                data = batch[0] if isinstance(batch, (list, tuple)) else batch
                if isinstance(data, (list, tuple)):
                    data = data[0]
                if isinstance(data, dict):
                    data = {k: v.to(device) for k, v in data.items()
                            if hasattr(v, 'to')}
                    # Positionally, NOT splatted. ClassificationAndEncoderNetwork
                    # takes the whole batch as one argument and does the splatting
                    # itself in _llm_forward. `self.model(**data)` raised
                    # "unexpected keyword argument 'input_ids'" on every GLUE run
                    # with --pruning_type wanda, at the first mask update.
                    self.model(data)
                else:
                    self.model(data.to(device))
        finally:
            for h in handles:
                h.remove()
            if was_training:
                self.model.train()

        self._calibrated = True
        print(f'  > [WANDA] calibrated on {seen} batch(es) over '
              f'{len(self.scaler_row)} of {len(self.prunable_params)} prunable '
              f'layer(s)')
        return True

    def update_masks(self, epoch):
        """Recompute the masks.

        Calibration is ONE-SHOT: it runs on the first mask update and is never
        repeated, so under a gradual schedule the weights are re-read live while
        the activation norms stay frozen at the first update. That is a protocol
        decision, not an oversight -- re-calibrating every update would cost a
        forward pass over the calibration set every delta_T steps -- but it is a
        decision the paper has to state, so `calibration_step` is recorded and
        pinned by a test rather than left implicit.
        """
        if self.current_sparsity == 0:
            return
        if not self._calibrated:
            self._calibration_step = epoch
            self.calibrate()
            if self._fallback_layers:
                pass  # populated below, after the first scoring pass

        for name, param in self.prunable_params.items():
            W = param.detach()
            flat = W.reshape(W.shape[0], -1)            # [out, in*kh*kw]

            norms = self.scaler_row.get(name)
            if norms is not None and norms.numel() == flat.shape[1]:
                scale = norms.sqrt().to(flat.device).unsqueeze(0)
            else:
                # No statistic for this layer -> magnitude ranking FOR THAT LAYER,
                # announced rather than silent. Grouped and depth-wise Conv2d hit
                # this deterministically, because F.unfold is group-unaware and
                # produces C_in*kh*kw columns where the weight has only
                # (C_in/groups)*kh*kw -- so a ResNeXt-style backbone would have
                # been magnitude-pruned throughout while reporting itself as WANDA.
                self._fallback_layers.add(name)
                scale = torch.ones(1, flat.shape[1], device=flat.device)

            scores = flat.abs() * scale

            if self.group == 'output':
                # Rank within each output row: every neuron keeps the same
                # fraction of its inputs. Wanda's headline configuration --
                # established for LLMs, explicitly NOT established for vision.
                k = int(flat.shape[1] * self.current_sparsity)
                if k < 1:
                    continue
                k = min(k, flat.shape[1] - 1)
                idx = torch.argsort(scores, dim=1)[:, :k]
                mask = torch.ones_like(flat)
                mask.scatter_(1, idx, 0.0)
            else:
                # One threshold for the whole layer. Rows may then lose very
                # different fractions, up to and including a whole neuron -- which
                # is the risk the per-output grouping exists to avoid, and the
                # reason both are offered rather than one being assumed.
                k = int(scores.numel() * self.current_sparsity)
                if k < 1:
                    continue
                k = min(k, scores.numel() - 1)
                threshold = torch.kthvalue(scores.reshape(-1), k).values
                mask = (scores > threshold).to(flat.dtype)
            self.masks[name] = mask.reshape(W.shape)

        if self._fallback_layers and not getattr(self, '_warned_fallback', False):
            self._warned_fallback = True
            print(f'  ! [WANDA] {len(self._fallback_layers)} layer(s) had no '
                  f'usable activation statistic and were MAGNITUDE-pruned: '
                  f'{sorted(self._fallback_layers)[:5]}'
                  f'{" ..." if len(self._fallback_layers) > 5 else ""}. '
                  f'Any WANDA column covering these layers is mislabelled.')


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

        # `or default` rather than kwargs.get(k, default): _initialize_pruner
        # passes these as an explicit None for every non-EAST pruner, so a plain
        # .get() would return None even though the key is present.
        self.prune_rate = kwargs.get('prune_rate') or 0.1
        self.Tc_ratio = kwargs.get('Tc_ratio') or 0.5

        # s_min must be assigned to self. update_masks reads it with
        # getattr(self, 's_min', 0.05), which silently returned the 0.05 default
        # on every run because nothing ever set the attribute -- so the s_min
        # that callers passed had no effect at all.
        self.s_min = kwargs.get('s_min') or 0.05

        # Use end_idx to determine Cycle length regardless of Epoch vs Step mode
        self.Tc = max(1, int(self.end_idx * self.Tc_ratio))
        
        self.erk_densities = self.calculate_erk_densities(model)
        self._init_sparse_masks()
        self.apply_mask()

    def _init_sparse_masks(self):
        print(f"[EAST] Initializing Sparse Masks at s_max={self.s_max}")
        for n, p in self.prunable_params.items():
            if n in self.erk_densities:
                self.masks[n] = torch.bernoulli(torch.full_like(p, self.erk_densities[n]))

    def update_masks(self, current_idx):
        # NOTE: no `current_idx % self.delta_T` gate here. There used to be one,
        # and it made this method a permanent no-op: _step_pruning_step already
        # gates on `step % delta_T == 0`, then BasePruner.step does
        # `t = current_idx + 1` before calling update_masks(t). So this saw
        # step + 1, and (step + 1) % delta_T is never 0 for any delta_T > 1.
        # EAST's masks therefore never updated once, staying frozen at their
        # Bernoulli ERK initialisation for the whole run.
        is_cyclic = current_idx <= self.Tc

        if is_cyclic:
            # 1. Global target sparsity from the cyclic cosine schedule.
            #
            # Phase matters. cos(0) = 1, so the naive
            #   s_min + (s_max - s_min) * 0.5 * (1 - cos(2*pi*t/Tc))
            # starts the cycle at s_min -- while _init_sparse_masks has just
            # initialised the masks at s_max. At t=0 that reads as "grow from 99%
            # sparse to 5% sparse", densifying the network to 95% density and
            # abandoning the sparsity budget entirely.
            #
            # Anchored at s_max instead: the cycle starts and ends at s_max and
            # dips to s_min at the half-cycle, which is the exploration behaviour
            # EAST describes (Li et al. 2024, arXiv 2411.13545).
            norm_t = current_idx / self.Tc
            s_min = self.s_min
            s_target = self.s_max - (self.s_max - s_min) * 0.5 * (1 - math.cos(2 * math.pi * norm_t))


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
            active_params = mask.sum().item()
            inactive_params = mask.numel() - active_params

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

            # Clamp to what this layer can actually supply. k is derived from the
            # *global* sparsity ratio but spent per-layer, so a layer whose ERK
            # density differs from the global average can be asked to prune more
            # weights than it has active, or grow more than it has inactive --
            # torch.topk then raises "selected index k out of range". Previously
            # unreachable because update_masks always early-returned.
            k_prune = int(min(k_prune, active_params))
            k_grow = int(min(k_grow, inactive_params))

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
    "wanda": WandaPruner,
}

def get_pruner(method_name, model, epochs, sparsity, **kwargs):
    if method_name not in PRUNER_REGISTRY:
        raise ValueError(f"Pruner {method_name} not found. Available: {list(PRUNER_REGISTRY.keys())}")
    return PRUNER_REGISTRY[method_name](model, epochs, sparsity, **kwargs)


def check_model_sparsity(model):
    """Fraction of prunable weights that are exactly zero.

    Note this measures *values*, not the mask. Under dynamic sparse training the two
    diverge: RigL and EAST zero-initialise regrown connections, so a freshly grown
    weight is structurally active but numerically zero and is counted here as pruned.
    Measured on a tiny ResNet, RigL's mask sparsity held at 0.8976 while this
    function reported 0.9277 -- i.e. it *overstates* sparsity, the flattering
    direction.

    For monotone pruning (magnitude, SNIP) nothing regrows, so the two agree and
    this is the right measure. When a pruner is present, prefer check_mask_sparsity.
    """
    sum_zero, total_count = 0, 0
    for name, param in model.named_parameters():
        if layer_check(name, param):
            sum_zero += torch.sum(param == 0).item()
            total_count += param.numel()
    return (sum_zero / total_count) if total_count > 0 else 0.0


def check_mask_sparsity(pruner):
    """Fraction of prunable weights the mask has switched off.

    The structural measure, and the correct one to report for dynamic sparse
    training: it counts connections, so it is unaffected by a regrown weight
    happening to sit at zero before its first update.
    """
    if pruner is None or not getattr(pruner, 'masks', None):
        return None
    active = sum(m.sum().item() for m in pruner.masks.values())
    total = sum(m.numel() for m in pruner.masks.values())
    return (1.0 - active / total) if total > 0 else 0.0


def report_sparsity(model, pruner=None):
    """The sparsity figure to report: mask-based when a pruner owns the masks."""
    mask_sparsity = check_mask_sparsity(pruner)
    return mask_sparsity if mask_sparsity is not None else check_model_sparsity(model)


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
        plt.barh(range(len(names)), sparsities, color='skyblue')
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



