"""BaCP: Backbone Contrastive Pruning -- the contrastive pruning trainer.

The module decomposition (PrC, SnC, FiC) and the four-term objective

    L = l1*L_PrC + l2*L_FiC + l3*L_SnC + l4*L_CE

(sum(l_i) = 1 by convention: the 0.25 defaults satisfy it, and user-supplied
lambdas are used as given, not renormalized)

follow CAP: Xu, Luo, Wang, Chang, Huang, Huang & Huang, "From Dense to Sparse:
Contrastive Pruning for Better Pre-trained Language Model Compression",
AAAI 2022 (vol. 36, pp. 11547-11555), arXiv 2112.07198. CAP defines the three
teachers -- PrC contrasts against the frozen pretrained model, SnC against
saved sparse snapshots, FiC against the fine-tuned model -- and BaCP applies
that decomposition to vision backbones. The decomposition itself is CAP's
contribution, not this project's.

The contrastive functional is cap_contrastive_loss (loss_functions.py), CAP
Eq. 1. Pruning is delegated to pruning_factory; each criterion carries its own
citation there.
"""

import os
import torch
import torch.nn as nn
import contextlib
from tqdm import tqdm
from copy import deepcopy
from torch.amp import autocast, GradScaler
from dataclasses import dataclass
from types import SimpleNamespace
# Explicit imports, not `from x import *`. This is load bearing: in the presence
# of a star import pyflakes downgrades UndefinedName to the far weaker
# ImportStarUsage, so no linter could flag a call to a function that does not
# exist. That is precisely how three broken call sites survived here
# (_handle_wanda_hooks, _handle_optimizer_and_pruning, PRUNING_REGISTRY).
from training_utils import (
    amp_dtype_and_scaler,
    _get_sparsity_key,
    _handle_data_to_device,
    _handle_tqdm_logs,
    _finalize_run,
    _initialize_all,
    _initialize_logs,
    _initialize_optimizer,
    _log_metrics,
    _maybe_limit,
    _optimizer_step,
    reset_grad_norm_log,
    summarize_grad_norm_log,
)
from dataset_factory import get_dataloaders
from dyrelu_adapter import step_dyrelu_adapter, set_t_for_dyrelu_adapter
from loss_functions import (CAPContrastiveLoss, SupConLoss, NTXentLoss,
                            kd_kl_loss, feature_distill_loss)
# Aliased: BaCPTrainer has its own get_pruner() method, and an unaliased import
# would make the two names visually indistinguishable at the call site.
from pruning_factory import check_model_sparsity, layer_check, get_pruner as build_pruner
from utils import load_weights, set_seed

@dataclass
class BaCPTrainingArguments:
    model_name:             str
    model_type:             str
    dataset_name:           str
    num_classes:            int
    batch_size:             int
    optimizer_type:         str
    learning_rate:          float
    tau:                    float = 0.07
    num_out_features:       int = 128       # Model embedded vector size
    image_size:             int = 32        # Image size for resizing
    epochs:                 int = 5         # Number of epochs to train
    scheduler_type:         str = None      # Scheduler type, e.g., "linear_with_warmup"
    patience:               int = None      # Number of epochs to wait before early stopping
    trained_weights:        str = None      # Path to pretrained weights
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
    recovery_epochs:        int = 0         # Number of epochs to recover after pruning
    pruning_module:         object = None   # Pruning module

    # Fine-tuning arguments
    enable_finetune:        bool = True
    epochs_ft:              int = 100
    optimizer_type_ft:      str = 'sgd'      # Optimizer type for fine-tuning
    learning_rate_ft:       float = 0.005    # Learning rate for fine-tuning

    # BaCP arguments
    lambdas:                list = None     # List of lambdas for BaCP
    learnable_lambdas:      bool = False    # Whether to learn lambdas or keep them static
    n_views:                int = 2         # Number of views for contrast

    # 'cap'    -- one CAP Eq.1 functional evaluated twice (instance-level and
    #             label-level positives) over a shared denominator, with
    #             rectangular student-anchor/teacher-candidate similarities.
    # 'legacy' -- the original SupConLoss + NTXentLoss sum over a square 2B x 2B
    #             matrix. Kept so published numbers stay reproducible.
    contrastive_mode:       str = 'cap'

    # 'tied_frozen' -- one frozen projection head shared by student and teachers.
    # 'none'        -- no head on the contrastive path (CAP-faithful).
    # 'current'     -- per-model trainable heads (original behaviour).
    proj_mode:              str = 'tied_frozen'

    # Cap on retained snapshots for SnC. Each one costs an extra forward pass per
    # batch and sits on the GPU, so this bounds both time and memory. The code
    # previously allowed up to epochs - 1 while the paper reported 2.
    num_snapshots:          int = 2
    # 'first' fills the cap in the opening epochs (what every run so far did);
    # 'even' spreads the snapshots across the pruning phase.
    snapshot_schedule:      str = 'first'

    # Distillation controls. These are NOT part of BaCP; they are the arms that
    # make BaCP's claim falsifiable. 'kl' is Hinton response distillation on the
    # fine-tuned dense teacher's logits, 'feature' is FitNets-style matching on
    # its embeddings, 'both' runs the two together. They exist because
    # without them the evidence supports "a dense teacher helps", which is a
    # weaker and different claim than the paper makes.
    #
    # The teacher is model_ft and not model_pt in both cases, deliberately: the
    # pretrained model's head is not aligned to this task, so its logits carry no
    # usable class posterior and its features are the wrong ones to regress onto.
    distill_mode:           str = 'none'    # 'none' | 'kl' | 'feature' | 'both'
    lambda_kd:              float = 0.0
    kd_temperature:         float = 4.0
    feature_distill_metric: str = 'cosine'  # 'cosine' | 'mse'

    # DyReLU Phasing
    dyrelu_en:              bool = False
    dyrelu_phasing_en:      bool = False

    # EAST weight sharing. Declared here because _initialize_pruner reads
    # args.weight_sharing_en unconditionally -- it was previously present only on
    # TrainingArguments, so every BaCP run raised AttributeError inside
    # __post_init__ before training could start.
    weight_sharing_en:      bool = False

    # Mask-update interval in optimizer steps. Same story: _initialize_pruner and
    # _step_pruning_step both read args.delta_T.
    delta_T:                int = 100

    # How the global sparsity budget is split across layers. Only read by
    # LocalMagnitudePrune ('erk' | 'uniform'); RigL and EAST always use ERK.
    # A config field rather than a pruner default so it reaches logger_params and
    # the run record -- the two settings report different things.
    layerwise_alloc:        str = 'erk'
    # Wanda's comparison group: 'output' (per output neuron, Wanda's headline
    # setting for LLMs) or 'layer' (one threshold per layer). Wanda's Appendix A
    # finds per-output does NOT transfer to image classifiers, so this is our
    # design choice and is recorded as one.
    wanda_group:            str = 'output'
    # L2 / weight decay. 1e-4 is EAST's value (TMLR 2025, Table B.1); GraNet uses
    # 5e-4. Which one is correct depends entirely on which paper's numbers you
    # intend to sit beside, so it is a declared field rather than a constant.
    weight_decay:           float = 5e-4
    # Tier-0 smoke controls: truncate the loaders to N batches. 0 = no limit.
    # Always recorded, so a truncated run cannot be mistaken for a real one.
    limit_train_batches:    int = 0
    limit_eval_batches:     int = 0
    # Prunable scope. The paper's appendix D.4 keeps task heads dense; dynamic
    # sparse training conventionally prunes everything. Both are defensible, but
    # they report different quantities -- see pruning_factory.set_prunable_scope.
    prune_task_head:        bool = False
    # Off by default and dangerous for transformers: DistilBERT and RoBERTa tie
    # word_embeddings to the output projection, so pruning embeddings also prunes
    # the LM head, and the embedding table then dominates the sparsity denominator.
    prune_embeddings:       bool = False

    # Run seed. Previously popped from args_dict in the scripts before the dataclass
    # was constructed, so it never reached args and was never recorded anywhere --
    # making every run un-reproducible after the fact.
    seed:                   int = 42
    # Train/val split seed, deliberately separate from `seed`. See
    # dataset_factory.load_cv_datasets for why the split must not track the run seed.
    val_split_seed:         int = 1234
    val_split:              float = 0.10
    # torch.use_deterministic_algorithms. Off by default: it hard-errors on ops with
    # no deterministic kernel and costs throughput. Recorded either way.
    deterministic:          bool = False

    # See TrainingArguments.auto_init.
    auto_init:              bool = True

    def __post_init__(self):
        # A distillation arm that is silently off is worse than one that crashes:
        # results.canonical_method names the run 'kd_<mode>+<pruner>' from
        # distill_mode alone, so the record would claim a KD arm while the
        # objective contained no KD term at all, and the results table would score
        # that null as a real teacher effect.
        _mode = getattr(self, 'distill_mode', 'none')
        if _mode not in (None, 'none') and not getattr(self, 'lambda_kd', 0.0):
            raise ValueError(
                f"distill_mode={_mode!r} with lambda_kd=0 is a no-op, but the run "
                f"would still be recorded as a KD arm. Set --lambda_kd > 0, or "
                f"--distill_mode none."
            )
        if _mode in (None, 'none') and getattr(self, 'lambda_kd', 0.0):
            raise ValueError(
                f"lambda_kd={self.lambda_kd} has no effect with "
                f"distill_mode='none'. Set --distill_mode, or --lambda_kd 0."
            )

        self.amp_dtype, self.scaler = (
            amp_dtype_and_scaler(
                'cuda' if torch.cuda.is_available() else 'cpu')
            if self.enable_mixed_precision else (None, None))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Legacy pair, kept for contrastive_mode='legacy'.
        self.supervised_criterion = SupConLoss(self.tau, self.tau, self.device).to(self.device)
        self.unsupervised_criterion = NTXentLoss(self.tau, self.device).to(self.device)
        # CAP Eq.1, used when contrastive_mode='cap' (the default).
        self.cap_criterion = CAPContrastiveLoss(self.tau).to(self.device)
        self.retrain = True if self.recovery_epochs > 0 else False
        self.is_bacp = True

        if not self.auto_init:
            return

        # Seed here, not only in the scripts: this guarantees the recorded seed is
        # the one actually active when models were built and the val split drawn.
        set_seed(self.seed)
        if self.deterministic:
            os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
            torch.use_deterministic_algorithms(True)

        _initialize_all(self)

class BaCPTrainer:
    def __init__(self, bacp_training_args):
        for key, value in vars(bacp_training_args).items():
            setattr(self, key, value)

        # Initialize training state
        self.recover = False
        self.snapshots = []

        # Fine-tuning loaders are only built by finetune(). Default them to None so
        # evaluate() can fall back to the main test loader: --enable_finetune is a
        # store_true flag whose False default overrides the dataclass default of
        # True, so the out-of-the-box invocation skips finetune() entirely and then
        # raised AttributeError on the unassigned ft_testloader.
        self.ft_trainloader = None
        self.ft_valloader = None
        self.ft_testloader = None

        # _handle_metrics needs a criterion to report a real loss. finetune() sets
        # this, but evaluate() can be reached without finetune() having run.
        self.criterion = nn.CrossEntropyLoss()
        self.current_sparsity = check_model_sparsity(self.model)
        self.context = (autocast(device_type=self.device, dtype=self.amp_dtype)
                        if self.enable_mixed_precision else contextlib.nullcontext())
        self._initialize_lambdas()
        self._initialize_metric_lists()

        # Initializing list for GPU-resident snapshots
        self.ss_models_on_device = []

        # NOTE: the DyReLU phasing schedule is already configured by
        # _initialize_all() during BaCPTrainingArguments.__post_init__. A second
        # call used to live here as `self._initialize_dyrelu_phasing()`, which is
        # not a method -- _initialize_dyrelu_phasing is a module-level function in
        # training_utils, so this raised AttributeError before training began.

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
            'lambda1': [], 'lambda2': [], 'lambda3': [], 'lambda4': [],
        }


    def _initialize_metric_lists(self):
        self.total_losses = {}
        self.prc_losses = {}
        self.snc_losses = {}
        self.fic_losses = {}
        self.ce_losses = {}
        self.training_acc = {}
        self.finetuning_loss = {}
        self.finetuning_acc = {}


    def train(self, run=None):
        _initialize_logs(self)
        reset_grad_norm_log()

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

            self._handle_snapshot_creation(epoch)
            torch.save(self.model.state_dict(), self.save_path)
        
        summarize_grad_norm_log(f'{self.model_name} bacp contrastive')

        if self.enable_finetune:
            self.finetune(run)

    
    def finetune(self, run=None):
        if self.enable_finetune == False:
            return

        torch.cuda.empty_cache()

        # Disabling pruning. Only the sparse masked is applied
        self.is_bacp = False
        self.criterion = nn.CrossEntropyLoss()
        self.prune = False
        self.recover = True
        self.pruner = self.get_pruner()
        self.current_sparsity = check_model_sparsity(self.model)

        self.trained_weights = self.save_path
        self.model.train()
        load_weights(self.model, self.trained_weights)

        # Decide BEFORE renaming. This used to rewrite save_path and then
        # return without writing anything, leaving the run record pointing at a
        # checkpoint that never existed -- and since every downstream rung
        # resolves its starting weights from a record, one skipped fine-tune
        # silently broke the rungs beneath it.
        model_sparsity = check_model_sparsity(self.model)
        print(f"[CHECK] SPARSITY : {model_sparsity}")

        # Tolerance, because this fired on a 7e-7 margin: a model at 0.9990007
        # against a 0.999 target is AT its target, not past it. Value-based
        # sparsity also drifts above the mask under DST (regrown weights are
        # zero-initialised), so an exact comparison here compares the wrong two
        # numbers as well as the wrong way.
        if model_sparsity > self.target_sparsity + 1e-4:
            print(f"[TRAINER] Model sparsity is {model_sparsity}. Target sparsity "
                  f"is {self.target_sparsity}. Skipping finetuning.")
            return

        self.save_path = self.save_path.replace('.pt', '_finetune.pt')
        print(f"[TRAINER] Saving model to {self.save_path}")

        # Rebuild single-view supervised loaders for the fine-tuning phase.
        # Routed through get_dataloaders rather than load_cv_dataloaders directly:
        # the direct call raised "Unsupported dataset: sst2" for every language model,
        # so BaCP + DistilBERT/RoBERTa + --enable_finetune had no working path.
        # supervised_override forces n_views=1 even though self.is_bacp is still True
        # at this point (finetune() clears it a few lines above, but not before this).
        data = get_dataloaders(self, supervised_override=True)


        # Honour limit_*_batches here too. These loaders are built by finetune()
        # rather than by _initialize_data_loaders, so they used to escape
        # truncation entirely -- a tier-0 BaCP cell ran its contrastive phase on
        # 2 batches and then fine-tuned on all 45,000 images, turning a 30-second
        # smoke run into a 2.2-hour one. A limit is a statement about the run, so
        # every loader the run builds has to respect it.
        n_train = getattr(self, 'limit_train_batches', 0)
        n_eval = getattr(self, 'limit_eval_batches', 0)
        self.ft_trainloader = _maybe_limit(data.get("trainloader"), n_train)
        self.ft_valloader = _maybe_limit(data.get("valloader"), n_eval)
        self.ft_testloader = _maybe_limit(data.get("testloader"), n_eval)
        if n_train or n_eval:
            print(f'  > [SMOKE] fine-tuning loaders truncated: '
                  f'train={n_train or "all"} eval={n_eval or "all"}')

        # Swap in the fine-tuning optimizer and learning rate. This used to build
        # the optimizer on a throwaway SimpleNamespace, setting `.optimizer` on that
        # temporary object and leaving self.optimizer untouched -- so every
        # fine-tuning epoch ever run used the *contrastive* optimizer and learning
        # rate, and optimizer_type_ft / learning_rate_ft had no effect at all.
        self.optimizer_type = self.optimizer_type_ft
        self.learning_rate = self.learning_rate_ft
        _initialize_optimizer(self)

        reset_grad_norm_log()
        for epoch in range(self.epochs_ft):
            # Training phase
            curr_ft_epoch_str = f"Fine-tuning Epoch [{epoch+1}/{self.epochs_ft}]"

            # Recovery training
            desc = f"Training {curr_ft_epoch_str}"
            ft_loss = self._run_finetune_epoch(epoch, desc)

            model_sparsity = check_model_sparsity(self.model)
            print(f"[CHECK] SPARSITY : {model_sparsity}")

            # Recovery validation
            desc = f"Validation {curr_ft_epoch_str}"
            ft_metrics = self._run_finetune_validation_epoch(desc)

            # Appends training loss and validation accuracy to lists
            ft_metrics['Finetuning Loss'] = ft_loss
            self._update_metric_lists(ft_metrics=ft_metrics)   

            # Logs these metrics to the logger or to W&B
            _log_metrics(self, curr_ft_epoch_str, ft_metrics, run) 
            
            # Saving model
            self._handle_save(epoch)

        summarize_grad_norm_log(f'{self.model_name} bacp finetune')

        final_metrics = self.evaluate(self.save_path, run)
        for k, v in final_metrics.items():
            if v is None:
                continue
            print(f"{k}: {v:.5f}")


    def evaluate(self, weights=None, run=None):
        """Evaluate model performnace on the testing dataset"""
        if weights is not None:
            if self.save_path and load_weights(self.model, self.save_path):
                print("[TRAINER] Weights loaded successfully")
            else:
                print("[TRAINER] Failed to load weights")
        else:
            print("[TRAINER] No weights provided. Will evaluate current model")

        self.model.eval()
        self.model.to(self.device)

        desc = "Evaluating"
        metrics = self._run_finetune_validation_epoch(desc, 'eval')

        sparsity = _get_sparsity_key(self)
        final_metrics = {}
        for key, value in metrics.items():
            if value is None:
                continue
            final_metrics[key] = value
        final_metrics['sparsity'] = sparsity

        _log_metrics(self, 'Final', final_metrics, run)
        _finalize_run(self, final_metrics)

        return final_metrics


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


    def _run_train_epoch(self, epoch, desc=""):
        """Run a training epoch."""
        # A call to _handle_wanda_hooks(self) used to sit here. That function has
        # never existed anywhere in the codebase -- the WandaPruner it belonged to
        # is entirely commented out in pruning_factory.py, and only the unrelated
        # _handle_wanda_calibration was ever written. Removed rather than stubbed.

        if len(self.snapshots) > len(self.ss_models_on_device):
            for i in range(len(self.ss_models_on_device), len(self.snapshots)):
                self.ss_models_on_device.append(self.snapshots[i].to(self.device))

        total_loss, total_prc, total_snc, total_fic, total_ce, total_acc = 0, 0, 0, 0, 0, 0
        total_kd = 0
        batchloader = tqdm(self.trainloader, desc=desc, leave=False) if self.enable_tqdm else self.trainloader

        for step, batch in enumerate(batchloader):
            # Unpacking batch and moving to device
            data, labels = _handle_data_to_device(self, batch)
            data1, data2 = self._split_views(data)

            with self.context:
                curr_emb, curr_logits = self._get_embeddings_and_logits(data1)

                with torch.no_grad():
                    pt_emb = self.model_pt(data2, return_emb=True)
                    ft_emb = self.model_ft(data2, return_emb=True)

                # PrC Module
                L_prc_raw = self._calculate_contrastive_loss(curr_emb, pt_emb, labels)
                
                # FiC Module
                L_fic_raw = self._calculate_contrastive_loss(curr_emb, ft_emb, labels)

                # SnC Module
                L_snc_raw = torch.tensor(0.0, device=self.device)
                if len(self.ss_models_on_device) > 0:
                    for ss_model in self.ss_models_on_device:
                        with torch.no_grad():
                            ss_emb = ss_model(data2, return_emb=True)

                        L_snc_raw += self._calculate_contrastive_loss(curr_emb, ss_emb, labels)
                    L_snc_raw /= len(self.ss_models_on_device)

                # CE Module
                L_ce_raw = self._task_loss(curr_logits, labels)

                # Distillation control (rungs C1/C2). Off unless --distill_mode is
                # set, so it costs nothing on the BaCP path.
                L_kd_raw = self._distillation_loss(curr_emb, curr_logits, data1, labels)

                # Applying lambdas and combining losses
                loss, prc_loss, snc_loss, fic_loss, ce_loss, kd_loss = self._combine_losses(
                    L_prc_raw, L_snc_raw, L_fic_raw, L_ce_raw, L_kd_raw
                )

                accuracy = self._task_accuracy(curr_logits, labels)

            # zero_grad -> backward -> prune -> optimizer.step -> re-apply mask.
            # Replaces a call to _handle_optimizer_and_pruning, which does not
            # exist (an older commented-out version in training_utils calls
            # pruner.ratio_step()/pruner.prune(), an API BasePruner no longer has,
            # so restoring it would only trade a NameError for an AttributeError).
            #
            # The index must be step-indexed, not epoch-indexed: _step_pruning_step
            # gates on `step % delta_T` against a pruner end_idx derived from
            # total_steps, so passing an epoch number would make the comparison
            # meaningless.
            _optimizer_step(self, loss, epoch * len(self.trainloader) + step)

            total_loss += loss.item()
            total_prc += prc_loss.item()
            total_fic += fic_loss.item()
            total_snc += snc_loss.item()
            total_ce  += ce_loss.item()
            total_kd  += kd_loss.item()
            total_acc += accuracy.item()

            # Calculating running mean of loss components
            metrics = {
                "Total Loss": _to_float(total_loss / (step + 1)),
                "PrC Loss": _to_float(total_prc / (step + 1)),
                "SnC Loss": _to_float(total_snc / (step + 1)),
                "FiC Loss": _to_float(total_fic / (step + 1)),
                "CE Loss": _to_float(total_ce / (step + 1)),
                "KD Loss": _to_float(total_kd / (step + 1)),
                "Training Accuracy": _to_float(total_acc / (step + 1)),
                "lambdas":[
                    _to_float(self.lambda1),
                    _to_float(self.lambda2),
                    _to_float(self.lambda3),
                    _to_float(self.lambda4),
                    ],
            }
            _handle_tqdm_logs(self, batchloader, metrics)

        if self.dyrelu_phasing_en:
            step_dyrelu_adapter(self.model)

        return metrics
    

    def _run_finetune_epoch(self, epoch, desc=""):
        self.model.train()
        ft_total_loss = 0
        batchloader = tqdm(self.ft_trainloader, desc=desc, leave=False) if self.enable_tqdm else self.ft_trainloader

        for step, batch in enumerate(batchloader):
            # Unpacking batch and moving to device
            data, labels = _handle_data_to_device(self, batch)

            with self.context:
                outputs = self.model(data)
                if hasattr(outputs, 'loss') and outputs.loss:
                    loss = outputs.loss
                elif hasattr(outputs, 'logits'):
                    loss = self.criterion(outputs.logits, labels)
                else:
                    loss = self.criterion(outputs, labels)
            ft_total_loss += loss.item()

            # Handling backprop, optimization, and pruning. self.recover is True
            # throughout fine-tuning, so _step_pruning_step short-circuits and the
            # masks are held fixed -- only apply_mask() runs, maintaining sparsity
            # without pruning further.
            _optimizer_step(self, loss, epoch * len(self.ft_trainloader) + step)

            running_loss = ft_total_loss / (step + 1)
            _handle_tqdm_logs(self, batchloader, {'Finetuning Loss': running_loss})

        if self.dyrelu_phasing_en:
            step_dyrelu_adapter(self.model)

        avg_loss = ft_total_loss / len(batchloader)    
        return avg_loss
    

    def _run_finetune_validation_epoch(self, desc="", mode="val"):
        """Run a validation epoch."""
        self.model.eval()
        val_loss, val_acc, val_perp = 0, 0, 0

        # Fall back to the main loaders when the fine-tuning ones were never built,
        # which is the case whenever enable_finetune is False -- the default for the
        # CLI, since --enable_finetune is a store_true flag.
        if mode == 'eval':
            dataloader = self.ft_testloader or self.testloader
        else:
            dataloader = self.ft_valloader or self.valloader or self.ft_testloader or self.testloader

        if dataloader is None:
            print("[TRAINER] No validation/test loader available; skipping evaluation")
            return {'Finetuning Val Loss': None, 'Finetuning Accuracy': None,
                    'Finetuning Perplexity': None}

        batchloader = tqdm(dataloader, desc=desc, leave=False) if self.enable_tqdm else dataloader

        with torch.no_grad():
            for step, batch in enumerate(batchloader):
                # Unpacking batch and moving to device
                data, labels = _handle_data_to_device(self, batch)

                with self.context:
                    outputs = self.model(data)

                metrics = self._handle_metrics(outputs, labels)
                val_loss += metrics.get('FT Batch Val Loss', 0)
                val_acc += metrics.get('FT Batch Accuracy', 0)
                val_perp += metrics.get('FT Batch Perplexity', 0)

                _handle_tqdm_logs(self, batchloader, metrics)

        avg_loss = val_loss / len(dataloader)
        avg_accuracy = val_acc / len(dataloader)
        avg_perplexity = val_perp / len(dataloader)
        return {
            'Finetuning Val Loss': avg_loss if avg_loss > 0.0 else None,
            'Finetuning Accuracy': avg_accuracy if avg_accuracy > 0.0 else None,
            'Finetuning Perplexity': avg_perplexity if avg_perplexity > 1.0 else None
        }


    def _task_loss(self, logits, labels):
        """Cross-entropy over whichever shape the task produces.

        Classification gives [B, C] against [B]. Masked language modelling gives
        [B, T, V] against [B, T] with -100 on unmasked positions, so the tensors
        have to be flattened first -- nn.CrossEntropyLoss ignores index -100 by
        default, which is exactly the convention DataCollatorForLanguageModeling
        writes.
        """
        if logits.dim() == 3:
            # With every position ignored, CrossEntropyLoss averages over zero
            # elements and returns NaN -- which would propagate through backward and
            # destroy the weights rather than merely skipping a batch. Unlikely at
            # the default 15% masking, but unrecoverable when it happens.
            if not bool((labels != -100).any()):
                return logits.sum() * 0.0
            return nn.CrossEntropyLoss()(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
            )
        return nn.CrossEntropyLoss()(logits, labels)

    def _task_accuracy(self, logits, labels):
        """Top-1 accuracy; for MLM, over the masked positions only."""
        if logits.dim() == 3:
            mask = labels != -100
            if not bool(mask.any()):
                return torch.tensor(0.0, device=logits.device)
            preds = logits.argmax(dim=-1)[mask]
            return (preds == labels[mask]).float().mean() * 100

        preds = logits.argmax(dim=1)
        return (preds == labels).float().mean() * 100

    def _split_views(self, data):
        """Return the two views fed to the student and to the frozen teachers.

        Images arrive as a list of two independently augmented tensors. Text arrives
        as a single dict: no text augmentation is applied, so there is no second
        view to unpack.

        That is not a shortcut. In CAP the two "views" of a text example are the
        same tokens seen through two *different models* -- the pruned student and a
        frozen teacher -- which is precisely what the cross-model contrastive term
        compares. Duplicating the batch is the faithful formulation, not a fallback.
        """
        if isinstance(data, (list, tuple)):
            if len(data) != 2:
                raise ValueError(
                    f"Expected 2 augmented views, got {len(data)}. BaCP's contrastive "
                    "step pairs one student view against one teacher view."
                )
            return data[0], data[1]
        return data, data

    def _get_embeddings_and_logits(self, data):
        """One forward pass on the trainable model, yielding embeddings and logits.

        Delegates to the wrapper's forward_all so vision and language share a path:
        for CV the features come from the stripped backbone and logits from cls_head;
        for LLMs the features are the [CLS] last-hidden-state and the "logits" come
        from the model's own retained head (a sequence classifier, or the vocabulary
        projection for MLM).
        """
        features, output = self.model.forward_all(data)
        logits = output.logits if hasattr(output, 'logits') else output
        embeddings = self.model.get_embeddings(features)
        return embeddings, logits


    def _calculate_contrastive_loss(self, current_emb, target_emb, labels):
        """Cross-model contrastive loss for one teacher. Used by PrC, FiC and SnC.

        contrastive_mode='cap' (default) evaluates a single CAP Eq.1 functional
        twice -- instance-level then label-level positives -- over a shared
        denominator, with the student as anchor and the teacher as candidates.

        contrastive_mode='legacy' reproduces the original SupConLoss + NTXentLoss
        sum. See loss_functions.cap_contrastive_loss for why the two are not
        equivalent and why the legacy composition worked against itself.
        """
        if getattr(self, 'contrastive_mode', 'cap') == 'legacy':
            L_sup = self.supervised_criterion(current_emb, target_emb, labels)
            L_unsup = self.unsupervised_criterion(current_emb, target_emb)
            return L_sup + L_unsup

        return self.cap_criterion(current_emb, target_emb, labels)
    
    
    def _distillation_loss(self, curr_emb, curr_logits, data1, labels):
        """The dense-teacher control term. Zero unless --distill_mode is set.

        The teacher forward runs on `data1` -- the *same* view the student saw --
        because KD is a per-example agreement objective, not a cross-view one.
        Feeding the teacher the second view would turn C1 into a noisier hybrid of
        distillation and augmentation-invariance, and its comparison against C0b
        (which isolates view count) would no longer be clean.

        The extra forward pass is charged only when distillation is on. Stage C's
        FLOP matching is maintained by _run_train_epoch running the PrC/FiC
        teacher forwards unconditionally regardless of lambda, so the
        rung-to-rung difference is one teacher pass either way.
        """
        if self.distill_mode in (None, 'none') or not self.lambda_kd:
            return torch.tensor(0.0, device=self.device)

        with torch.no_grad():
            t_features, t_output = self.model_ft.forward_all(data1)
            t_logits = t_output.logits if hasattr(t_output, 'logits') else t_output
            t_emb = self.model_ft.get_embeddings(t_features)

        loss = torch.tensor(0.0, device=self.device)
        if self.distill_mode in ('kl', 'both'):
            loss = loss + kd_kl_loss(curr_logits, t_logits, T=self.kd_temperature)
        if self.distill_mode in ('feature', 'both'):
            loss = loss + feature_distill_loss(curr_emb, t_emb,
                                               mode=self.feature_distill_metric)
        return loss


    def _combine_losses(self, prc_raw, snc_raw, fic_raw, ce_raw, kd_raw=None):
        """
        Applies the lambda weights to the loss components and sums them.

        The distillation term carries its own weight rather than sharing the four
        lambdas. That is a design choice with a reason: the submitted paper's
        ablation zeroed one of four 0.25 weights and left the rest alone, so every
        "no X" arm was simultaneously a removal *and* a 25% downweighting of the
        whole objective -- two interventions wearing one label. Keeping
        lambda_kd outside the simplex means turning distillation on does not
        silently rescale the contrastive terms, so C0b -> C1 is one change.
        """
        prc_loss = prc_raw * self.lambda1
        fic_loss = fic_raw * self.lambda2
        snc_loss = snc_raw * self.lambda3
        ce_loss = ce_raw * self.lambda4

        if kd_raw is None:
            kd_raw = torch.tensor(0.0, device=self.device)
        kd_loss = kd_raw * self.lambda_kd

        total_loss = prc_loss + snc_loss + fic_loss + ce_loss + kd_loss
        return total_loss, prc_loss, snc_loss, fic_loss, ce_loss, kd_loss


    def _handle_metrics(self, outputs, labels):
        current_correct = 0
        current_samples = 0
        current_loss = 0

        # See Trainer._handle_metrics for why current_loss must be a sum here.
        if self.model_type == 'llm':
            if self.dataset_name == 'wikitext2':
                mask = (labels != -100)

                logits = outputs.logits[mask]
                masked_labels = labels[mask]
                preds = torch.argmax(logits, dim=-1)

                current_correct = (preds == masked_labels).sum().item()
                current_samples = mask.sum().item()
                current_loss = outputs.loss.item() * current_samples
            else:
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                current_correct = (preds == labels).sum().item()
                current_samples = labels.size(0)
                current_loss = self.criterion(logits, labels).item() * current_samples

        else:
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            preds = logits.max(1)[1]
            current_correct = (preds == labels).sum().item()
            current_samples = labels.size(0)
            current_loss = self.criterion(logits, labels).item() * current_samples

        if current_samples == 0:
            return {'FT Batch Val Loss': 0.0, 'FT Batch Accuracy': 0.0, 'FT Batch Perplexity': 1.0}

        avg_loss = current_loss / current_samples
        avg_accuracy = 100 * current_correct / current_samples
        avg_perplexity = torch.exp(torch.tensor(avg_loss)).item()
        return {
            'FT Batch Val Loss': avg_loss,
            'FT Batch Accuracy': avg_accuracy,
            'FT Batch Perplexity': avg_perplexity,
        }
                
    
    def _handle_snapshot_creation(self, epoch=None):
        """Snapshot the current model onto the CPU for use as an SnC teacher.

        Bounded by num_snapshots (default 2, matching the paper) rather than by
        epochs - 1. Every retained snapshot costs an additional forward pass per
        batch and is moved onto the GPU in _run_train_epoch, so an unbounded cap
        made both step time and GPU memory scale with the epoch count.

        snapshot_schedule decides WHEN the cap is spent:

          'first' (default, and what every run so far used) fills greedily, so a
              50-epoch run takes both snapshots at the ends of epochs 1 and 2.
              The cubic schedule has applied 0% and 8% of the target sparsity by
              then, so SnC spends the remaining 48 epochs anchored to two
              essentially dense models.
          'even' spreads them across the pruning phase, so the anchors track the
              sparsification instead of predating it.

        The default is unchanged, so existing keys keep their meaning.
        """
        cap = min(getattr(self, 'num_snapshots', 2), max(self.epochs - 1, 0))
        if cap <= 0 or len(self.snapshots) >= cap:
            return

        if getattr(self, 'snapshot_schedule', 'first') == 'even':
            if epoch is None:
                return
            # cap points spread over the run: cap=5, epochs=50 -> 8, 17, 25, 33, 42
            targets = {round(self.epochs * (i + 1) / (cap + 1)) for i in range(cap)}
            if (epoch + 1) not in targets:
                return

        # Keep on CPU to save GPU memory. The transfer happens once per epoch.
        ss_model = deepcopy(self.model).to('cpu').eval()
        self.snapshots.append(ss_model)


    def get_pruner(self):
        """Rebuild a pruner whose masks are recovered from the saved sparse weights.

        Three things were wrong here and none of them could ever have run:
          * the registry is named PRUNER_REGISTRY, not PRUNING_REGISTRY;
          * BasePruner takes its scheduler settings as **kwargs, so passing
            sparsity_scheduler as a 4th positional argument was a TypeError; and
          * scheduler_type/delta_T/total_steps are all required.
        """
        pruning_module = build_pruner(
            method_name=self.pruning_type,
            model=self.model,
            epochs=self.epochs,
            sparsity=self.target_sparsity,
            scheduler_type=self.sparsity_scheduler,
            delta_T=self.delta_T,
            total_steps=len(self.trainloader) * self.epochs,
        )

        # Loading sparse weights
        load_weights(self.model, self.save_path)

        # Recover the mask from which weights are exactly zero. Restricted to the
        # parameters the pruner actually owns: spanning all of named_parameters()
        # would also freeze any weight that merely happens to equal 0.0 -- a
        # BatchNorm bias, a dead unit -- for the remainder of training.
        zero_masks = {
            name: (param != 0).float()
            for name, param in self.model.named_parameters()
            if layer_check(name, param)
        }

        pruning_module.masks = zero_masks
        return pruning_module
        

    def _update_metric_lists(self, metrics=None, ft_metrics=None):
        sparsity_key = _get_sparsity_key(self)

        if metrics is not None:
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
        
        if ft_metrics is not None:
            self.finetuning_loss.setdefault(sparsity_key, []).append(ft_metrics.get('Finetuning Loss'))
            self.finetuning_acc.setdefault(sparsity_key, []).append(ft_metrics.get('Finetuning Accuracy'))


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
        if self.is_bacp:
            loss_list = self.total_losses[sparsity_key]

            if len(loss_list) <= 1:
                torch.save(self.model.state_dict(), self.save_path)
                return True
            elif len(loss_list) > 1:
                if loss_list[-1] < min(loss_list[:-1]):
                    torch.save(self.model.state_dict(), self.save_path)
                    return True
            return False
        else:
            acc_list = self.finetuning_acc[sparsity_key]
            if len(acc_list) <= 1:
                torch.save(self.model.state_dict(), self.save_path)
                return True
            elif len(acc_list) > 1:
                if acc_list[-1] > max(acc_list[:-1]):
                    torch.save(self.model.state_dict(), self.save_path)
                    return True
            return False
        
        
def _to_float(x):
    return float(x.detach().cpu()) if isinstance(x, torch.nn.Parameter) or torch.is_tensor(x) else float(x)















