import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import pickle

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[SEEDING] Global seed set to {seed}")

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def get_num_workers():
    return os.cpu_count()

def preview_dataloader(dataloader):
    images, _ = next(iter(dataloader))
    if isinstance(images, (list, tuple)) and len(images) == 2:
        images1, images2 = images
        images = torch.cat([img.unsqueeze(0) for pair in zip(images1, images2) for img in pair])[:24]
    else:
        images = images[:24]

    img_grid = make_grid(images, nrow=12, normalize=True, pad_value=0.9).permute(1, 2, 0)

    plt.figure(figsize=(10,5))
    plt.title('Augmented image examples of the CIFAR-10 dataset')
    plt.imshow(img_grid)
    plt.axis('off')
    plt.show()

def freeze_weights(model):
    for param in model.parameters():
        param.requires_grad = False
        
def load_weights(model, path):
    """Load a checkpoint into `model`. Returns True only if weights were loaded.

    Callers must check the return value. A False here means the model still holds
    whatever it was initialised with -- which, for a run that asked for pretrained
    weights, means random init. Silently continuing from that point produces a
    "pretrained" baseline that was never pretrained.
    """
    print(f"[LOADING] Loading weights: {path}")

    if path is None:
        print("> Could not load weights: no path given")
        return False

    if not os.path.exists(path):
        print(f"> Could not load weights. Path does not exist: {path}")
        return False

    try:
        state_dict = torch.load(path, map_location=get_device(), weights_only=True)
        model.load_state_dict(state_dict)
        # Printed only *after* the load actually succeeds. This message used to be
        # emitted before torch.load was even called, so the logs claimed success
        # on runs that then fell through to the partial-load path -- or failed.
        print(f"> Full load successful!: {path}")
        return True
    except Exception as e:
        # Was a bare `except:`, which also swallowed KeyboardInterrupt and
        # SystemExit and hid the actual reason for the failure.
        print(f"> Could not load weights: {type(e).__name__}: {e}")
        print("> Attempting partial load")

    state_dict = torch.load(path, map_location=get_device(), weights_only=True)
    filtered_state_dict = {
        k: v for k, v in state_dict.items()
        if not any(head_key in k for head_key in ['encoder_head', 'fc', 'classifier', 'head', 'vocab_projector'])
        }
    try:
        missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
        # `strict=False` tolerates ANY number of missing keys, so an unguarded
        # `return True` here let a checkpoint that matched zero tensors report
        # init_source='imagenet_checkpoint' on an effectively random model --
        # the same provenance lie the full-load path was fixed to prevent.
        # Require that at least half the model's tensors were actually loaded.
        n_model = len(model.state_dict())
        n_loaded = n_model - len(missing)
        ok = n_loaded >= 0.5 * n_model
        print(f"> Partial load: {n_loaded}/{n_model} tensors loaded "
              f"(missing: {len(missing)}, unexpected: {len(unexpected)}) -> "
              f"{'accepted' if ok else 'REJECTED (below half); treating as not loaded'}")
        return ok
    except Exception as e:
        raise RuntimeError(f"Error loading weights from {path}: {e}") from e
              
def graph_losses_n_accs(losses, train_accs, test_accs):
    _, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(losses, label="Loss", color="blue", linewidth=2)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(train_accs, label="Training Accuracy", color="green", linewidth=2)
    axes[1].plot(test_accs, label="Testing Accuracy", color="red", linewidth=2)
    axes[1].set_title("Training and Testing Accuracy")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def save_object(object, filepath):
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(object, f)
        print(f"Object saved successfully to {filepath}")
        return True
    except Exception as e:
        print(f"Failed to save object: {e}")
        return False

def load_object(filepath):
    try:
        with open(filepath, 'rb') as f:
            trainer = pickle.load(f)
        print(f"Object loaded successfully from {filepath}")
        return trainer
    except Exception as e:
        print(f"Failed to load object: {e}")
        return None


def print_dynamic_lambdas_statistics(trainer_instance):
    if not hasattr(trainer_instance, 'lambda_history'):
        print("Trainer instance does not have a 'lambda_history'.")
        return

    # Order matters and was previously wrong. BaCPTrainer._combine_losses maps
    # lambda1 -> PrC, lambda2 -> FiC, lambda3 -> SnC, lambda4 -> CE, and
    # lambda_history is keyed lambda1..lambda4 in that order. The old labels read
    # ['CE', 'PrC', 'SnC', 'FiC'], so every dynamic-lambda figure produced by this
    # function had three of its four series mislabelled.
    # Pinned by test_losses.py::test_lambda_to_loss_mapping.
    lambda_types = ['PrC lambda', 'FiC lambda', 'SnC lambda', 'CE lambda']
    lambda_history = trainer_instance.lambda_history

    plt.figure(figsize=(10, 6))
    for i, key in enumerate(lambda_history):
        values = torch.tensor(lambda_history[key]).detach().cpu().numpy()
        plt.plot(values, label=lambda_types[i])
    plt.title(f"Dynamic Lambda Value Evolution - ({trainer_instance.pruning_type} - {trainer_instance.target_sparsity})")
    plt.xlabel("Epoch")
    plt.ylabel("Lambda Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("=== Lambda Statistics ===")
    for i, key in enumerate(lambda_history):
        values = torch.tensor(lambda_history[key])
        initial = values[0].item()
        final = values[-1].item()
        change = final - initial
        percent_change = (change / initial * 100) if initial != 0 else float('inf')
        mean_val = values.mean().item()
        std_val = values.std().item()

        print(f"{lambda_types[i]}:")
        print(f"  Initial: {initial:.4f}")
        print(f"  Final:   {final:.4f}")
        print(f"  Change:  {change:.4f} ({percent_change:+.2f}%)")
        print(f"  Mean:    {mean_val:.4f}")
        print(f"  StdDev:  {std_val:.4f}")
        print()

def print_statistics(metrics, trainer_instance):
    print("\n" + "="*60)
    print("TRAINING STATISTICS SUMMARY")
    print("="*60)
    
    print("\nPerformance Metrics:")
    print("-" * 30)
    if 'accuracy' in metrics:
        print(f"  Accuracy:     {metrics['accuracy']:.2f}%")
    if 'perplexity' in metrics:
        print(f"  Perplexity:   {metrics['perplexity']:.3f}")
    if 'loss' in metrics:
        print(f"  Loss:         {metrics['loss']:.4f}")
    
    # Model Information
    if trainer_instance is not None:
        print("\nModel Information:")
        print("-" * 30)
        
        # Parameter count
        total_params = sum(p.numel() for p in trainer_instance.model.parameters())
        trainable_params = sum(p.numel() for p in trainer_instance.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"  Total Parameters:     {total_params:,}")
        print(f"  Trainable Parameters: {trainable_params:,}")
        if frozen_params > 0:
            print(f"  Frozen Parameters:    {frozen_params:,}")
        
        try:
            if 'sparsity' in metrics:
                sparsity = metrics['sparsity']
            else:
                pass
            print(f"  Model Sparsity:       {sparsity:.4f} ({sparsity*100:.2f}%)")
        except:
            pass
        
        # Training Configuration
        print("\nTraining Configuration:")
        print("-" * 30)
        print(f"  Model:                {trainer_instance.model_name}")
        print(f"  Task:                 {trainer_instance.model_task}")
        print(f"  Learning Type:        {trainer_instance.learning_type}")
        print(f"  Batch Size:           {trainer_instance.batch_size}")
        print(f"  Learning Rate:        {trainer_instance.learning_rate}")
        print(f"  Optimizer:            {trainer_instance.optimizer_type}")
        
        if hasattr(trainer_instance, 'epochs'):
            print(f"  Epochs:               {trainer_instance.epochs}")
        
        if hasattr(trainer_instance, 'prune') and trainer_instance.prune:
            print("\nPruning Configuration:")
            print("-" * 30)
            print(f"  Pruning Type:         {trainer_instance.pruning_type}")
            print(f"  Target Sparsity:      {trainer_instance.target_sparsity}")
            print(f"  Sparsity Scheduler:   {trainer_instance.sparsity_scheduler}")
            if hasattr(trainer_instance, 'recovery_epochs'):
                print(f"  Recovery Epochs:      {trainer_instance.recovery_epochs}")

        
        # System Info
        print("\nSystem Information:")
        print("-" * 30)
        print(f"  Device:               {trainer_instance.device}")
        print(f"  Mixed Precision:      {trainer_instance.enable_mixed_precision}")
        print(f"  Workers:              {trainer_instance.num_workers}")
        
        print("\n" + "="*60)