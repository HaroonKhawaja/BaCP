import os
import torch
import torch.nn as nn
import contextlib
from tqdm import tqdm
from torch.amp import GradScaler, autocast
from dataclasses import dataclass
from training_utils import (
    amp_dtype_and_scaler,
    _finalize_run,
    _initialize_all,
    _initialize_logs,

    _initialize_optimizer,
    _optimizer_step,
    _step_pruning_step,

    _handle_data_to_device,
    _handle_tqdm_logs,
    _log_metrics,
    _get_sparsity_key,
)
from dyrelu_adapter import step_dyrelu_adapter
from pruning_factory import check_model_sparsity
from utils import load_weights, set_seed

@dataclass
class TrainingArguments:
    model_name:             str
    model_type:             str
    dataset_name:           str
    num_classes:            int
    batch_size:             int
    optimizer_type:         str
    learning_rate:          float
    criterion:              nn.Module = nn.CrossEntropyLoss()
    num_out_features:       int = None
    image_size:             int = 32
    epochs:                 int = 5
    scheduler_type:         str = None
    patience:               int = None
    trained_weights:        str = None
    experiment_type:        str = ""
    log_epochs:             bool = False
    enable_tqdm:            bool = False
    enable_mixed_precision: bool = True
    databricks_env:         bool = True
    num_workers:            int = os.cpu_count()

    # Pruning arguments
    pruning_type:           str = None
    target_sparsity:        float = None
    sparsity_scheduler:     str = None
    recovery_epochs:        int = 0

    # Post-pruning supervised fine-tune, mask frozen. Exists so the I.P.
    # baseline can be run on the SAME budget as BaCP (which fine-tunes after
    # its contrastive phase); otherwise every reported delta compares 110
    # epochs of BaCP against 60 of I.P.
    enable_finetune:        bool = False
    optimizer_type_ft:      str = 'adamw'
    learning_rate_ft:       float = 1e-4
    epochs_ft:              int = 0
    # Skip the pruning epochs and fine-tune a checkpoint that is already sparse.
    # --trained_weights then points at a finished pruning run's checkpoint rather
    # than the dense one, and the mask is read back off the zeros. Appending a
    # fine-tune to an existing I.P. run this way costs 25 epochs instead of 75,
    # and is equivalent because finetune() builds a fresh optimizer regardless.
    finetune_only:          bool = False
    pruning_module:         object = None
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

    # DyReLU / EAST
    dyrelu_en:              bool = False
    dyrelu_phasing_en:      bool = False
    weight_sharing_en:      bool = False

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

    # Set False to construct the arguments object without building models,
    # dataloaders, optimizer or pruner. Only the cheap prefix in __post_init__
    # runs. Used by the test suite so the dataclass is constructible on CPU with
    # no /dbfs checkpoints; not exposed on the CLI.
    auto_init:              bool = True

    def __post_init__(self):
        self.amp_dtype, self.scaler = (
            amp_dtype_and_scaler(
                'cuda' if torch.cuda.is_available() else 'cpu')
            if self.enable_mixed_precision else (None, None))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.retrain = (self.recovery_epochs > 0)
        self.is_bacp = False

        if not self.auto_init:
            return

        # Seed here, not only in the scripts: this guarantees the recorded seed is
        # the one actually active when models were built and the val split drawn.
        set_seed(self.seed)
        if self.deterministic:
            os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
            torch.use_deterministic_algorithms(True)

        _initialize_all(self)

class Trainer:
    """
    Unified trainer for CV/LLM models supporting:
    - Mixed Precision (AMP)
    - Sparse Training (RigL, Wanda, Magnitude)
    - Weight Sharing (EAST)
    - DyReLU Phasing
    """
    def __init__(self, training_args):
        for key, value in vars(training_args).items():
            setattr(self, key, value)

        # State tracking
        self.recover = False
        self.unchanged = 0
        self.train_losses = []
        self.val_accuracies = []
        self.accuracies = {} if self.target_sparsity is not None else []
        self.current_sparsity = check_model_sparsity(self.model)
        self.context = (autocast(device_type=self.device, dtype=self.amp_dtype)
                        if self.enable_mixed_precision else contextlib.nullcontext())

    def train(self, run=None):
        """Main training workflow."""
        _initialize_logs(self)

        if self.finetune_only:
            self._recover_masks_from_weights()
            self.finetune(run)
            return

        for epoch in range(self.epochs):
            curr_epoch_str = f"Epoch [{epoch+1}/{self.epochs}]"

            # Training
            loss = self._run_train_epoch(epoch, f"Training {curr_epoch_str}")

            # Pruning Step
            # _epoch_pruning_step(self, epoch)

            # Validation
            metrics = self._run_validation_epoch(f"Validation {curr_epoch_str}")

            # Logging & Metrics
            self._update_metric_lists(loss, metrics.get('accuracy'))
            metrics['loss'] = loss
            _log_metrics(self, curr_epoch_str, metrics, run)

            # Checkpoint
            if not self._handle_save(epoch):
                break
            
            # Recovery Phase
            if self.retrain:
                self._retrain(run)

        if self.enable_finetune and self.epochs_ft:
            self.finetune(run)


    def _recover_masks_from_weights(self):
        """Take the mask from the sparse checkpoint already loaded into the model.

        Mirrors BaCPTrainer.get_pruner. Restricted to the parameters the pruner
        owns: spanning all of named_parameters() would also freeze any weight
        that merely happens to be 0.0 -- a BatchNorm bias, a dead unit -- for the
        whole fine-tune.

        The sparsity that comes back is asserted against target_sparsity rather
        than trusted, because pointing --trained_weights at a dense checkpoint by
        mistake would otherwise fine-tune a dense model and silently report it as
        a sparse result.
        """
        from pruning_factory import layer_check

        if self.pruner is None:
            raise ValueError(
                'finetune_only needs a pruner: pass --pruning_type and --target_sparsity')

        self.pruner.masks = {
            name: (param != 0).float()
            for name, param in self.model.named_parameters()
            if layer_check(name, param)
        }
        self.pruner.apply_mask()

        got = check_model_sparsity(self.model)
        self.current_sparsity = got
        if abs(got - self.target_sparsity) > 0.01:
            raise ValueError(
                f'finetune_only: checkpoint is {got:.6f} sparse but --target_sparsity '
                f'is {self.target_sparsity}. Point --trained_weights at the sparse '
                f'checkpoint of a finished pruning run, not the dense one.')
        print(f'[TRAINER] finetune_only: mask recovered from checkpoint, '
              f'sparsity {got:.6f}, {len(self.pruner.masks)} layers')

    def finetune(self, run=None):
        """Supervised fine-tune with the mask frozen.

        Mirrors BaCPTrainer.finetune so the two arms share a budget and a
        fine-tuning optimizer, leaving the contrastive objective as the only
        difference between them.

        `recover = True` is what freezes the mask: _step_pruning_step skips
        every mask update while it is set, and _optimizer_step still calls
        pruner.apply_mask() afterwards, so sparsity is enforced but the
        surviving set never changes. The loaders need no rebuilding here --
        unlike BaCP, this trainer is already on the single-view supervised
        recipe.
        """
        print(f"[TRAINER] Fine-tuning for {self.epochs_ft} epochs "
              f"({self.optimizer_type_ft} @ {self.learning_rate_ft}), mask frozen")
        self.recover = True
        self.optimizer_type = self.optimizer_type_ft
        self.learning_rate = self.learning_rate_ft
        _initialize_optimizer(self)
        self.scheduler = None

        for epoch in range(self.epochs_ft):
            curr = f"Fine-tuning Epoch [{epoch+1}/{self.epochs_ft}]"
            loss = self._run_train_epoch(epoch, f"Training {curr}")
            metrics = self._run_validation_epoch(f"Validation {curr}")
            self._update_metric_lists(loss, metrics.get('accuracy'))
            metrics['loss'] = loss
            _log_metrics(self, curr, metrics, run)
            self._handle_save(epoch)

        self.recover = False


    def evaluate(self, load=True, run=None):
        """Evaluate model performnace on the testing dataset"""
        if load:
            if self.save_path and load_weights(self.model, self.save_path):
                print("[TRAINER] Weights loaded successfully")
            else:
                print("[TRAINER] Failed to load weights")

        self.model.eval()
        self.model.to(self.device)
        
        desc = "Evaluating"
        metrics = self._run_validation_epoch(desc, 'eval')

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
        """Recovery phase: Trained without changing the masks"""
        self.recover = True
        print(f"[TRAINER] Starting Recovery for {self.recovery_epochs} epochs...")

        for epoch in range(self.recovery_epochs):
            curr_str = f"Recovery Epoch [{epoch+1}/{self.recovery_epochs}]"

            loss = self._run_train_epoch(epoch, f"Training {curr_str}")
            metrics = self._run_validation_epoch(f"Validation {curr_str}")

            self._update_metric_lists(loss, metrics.get('accuracy'))
            metrics['loss'] = loss
            _log_metrics(self, curr_str, metrics, run)
            
            self._handle_save(epoch)

        self.recover = False


    def _run_train_epoch(self, epoch, desc=""):
        """Run a training epoch."""
        # _handle_wanda_hooks(self)
        # _handle_wanda_calibration(self)

        self.model.train()
        total_loss = 0
    
        steps_per_epoch = len(self.trainloader)
        batchloader = tqdm(self.trainloader, desc=desc, leave=False) if self.enable_tqdm else self.trainloader

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

            # Optimizer + pruning step
            global_step = epoch * steps_per_epoch + step
            _optimizer_step(self, loss, global_step)

            total_loss += loss.item()
            running_loss = total_loss / (step + 1)
            _handle_tqdm_logs(self, batchloader, {'loss': running_loss})

        # DyReLU Phasing Step (End of Epoch)
        if self.dyrelu_phasing_en:
            step_dyrelu_adapter(self.model)
        return total_loss / len(self.trainloader)


    def _run_validation_epoch(self, desc="", mode="val"):
        """Run a validation epoch."""
        self.model.eval()
        val_loss, val_acc, val_perp = 0, 0, 0

        dataloader = self.testloader if (mode == 'eval' and self.testloader) else self.valloader
        if not dataloader: return {}

        batchloader = tqdm(dataloader, desc=desc, leave=False) if self.enable_tqdm else dataloader

        with torch.no_grad():
            for step, batch in enumerate(batchloader):
                # Unpacking batch and moving to device
                data, labels = _handle_data_to_device(self, batch)

                with self.context:
                    outputs = self.model(data)

                metrics = self._handle_metrics(outputs, labels)
                val_loss += metrics.get('batch_val_loss', 0)
                val_acc += metrics.get('batch_accuracy', 0)
                val_perp += metrics.get('batch_perplexity', 0)

                _handle_tqdm_logs(self, batchloader, metrics)

        avg_loss = val_loss / len(dataloader)
        avg_accuracy = val_acc / len(dataloader)
        avg_perplexity = val_perp / len(dataloader)
        return {
            'val_loss': avg_loss if avg_loss > 0.0 else None,
            'accuracy': avg_accuracy if avg_accuracy > 0.0 else None,
            'perplexity': avg_perplexity if avg_perplexity > 1.0 else None
        }
    
    def _handle_metrics(self, outputs, labels):
        current_correct = 0
        current_samples = 0
        current_loss = 0

        # current_loss accumulates a SUM over samples so the division below
        # recovers a mean. Both branches previously got this wrong: the CV branch
        # never assigned it at all (so every vision run reported val_loss None and
        # perplexity None, leaving accuracy as the only metric and forcing early
        # stopping to key off accuracy), and the wikitext2 branch assigned
        # outputs.loss -- already a mean over masked tokens -- which was then
        # divided by the token count a second time, pinning avg_loss near 0 and
        # perplexity at ~1.0 for every batch.
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
            return {'batch_val_loss': 0.0, 'batch_accuracy': 0.0, 'batch_perplexity': 1.0}

        avg_loss = current_loss / current_samples
        avg_accuracy = 100 * current_correct / current_samples
        avg_perplexity = torch.exp(torch.tensor(avg_loss)).item()
        return {
            'batch_val_loss': avg_loss,
            'batch_accuracy': avg_accuracy,
            'batch_perplexity': avg_perplexity,
        }

    def _update_metric_lists(self, loss, accuracy):
        self.train_losses.append(loss)
        if self.target_sparsity is not None:
            sparsity_key = _get_sparsity_key(self)
            self.accuracies.setdefault(sparsity_key, []).append(accuracy)
        else:
            self.accuracies.append(accuracy)

    def _handle_save(self, epoch):
        """Determines if model should be saved based on accuracy improvement."""
        if self.save_path:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        # Get relevant accuracy history
        if isinstance(self.accuracies, dict):
            key = _get_sparsity_key(self)
            hist = self.accuracies.get(key, [])
        else:
            hist = self.accuracies

        # With val_split=0 there is no validation loader, so
        # _run_validation_epoch returns {} and every entry here is None.
        # Comparing None > None raises, which used to kill the run at epoch 2.
        # Fall back to the training loss in that case -- the same criterion
        # BaCPTrainer._save_model uses when it selects on total_losses -- so
        # the two arms share a selection rule instead of one silently
        # crashing.
        scored = [v for v in hist if v is not None]
        if not scored:
            hist = [-l for l in self.train_losses]   # lower loss = better

        # Logic: Save if first epoch OR if improved over previous best
        improved = False
        if len(hist) <= 1:
            improved = True
        elif len(hist) > 1 and hist[-1] is not None \
                and hist[-1] > max(v for v in hist[:-1] if v is not None):
            improved = True

        if improved:
            torch.save(self.model.state_dict(), self.save_path)
            print("[TRAINER] Checkpoint saved.")
            self.unchanged = 0
            return True
        else:
            self.unchanged += 1
            if self.patience and self.unchanged >= self.patience:
                print(f"[TRAINER] Early stopping triggered (Patience: {self.patience})")
                return False
            return True
        
