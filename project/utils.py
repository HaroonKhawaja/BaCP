import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from constants import *

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_mask_from_model(model):
    return {name: (param != 0).float() for name, param in model.named_parameters()}
    
def set_optimizer(opt_type, config):
    params = config['model'].parameters()
    lr = config['lr']
    weight_decay = config['weight_decay']

    if opt_type == 'sgd':
        return optim.SGD(params, lr=lr, momentum=config['momentum'], weight_decay=weight_decay)
    elif opt_type == 'adam':
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif opt_type == 'adamw':
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Invalid optimizer type: {opt_type}")
    
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
    if not os.path.exists(path):
        print(f"[ERROR] Could not load weights. Path does not exist: {path}")
        raise Exception("Error loading weights: {path}")
    try:
        state_dict = torch.load(path, map_location=get_device())
        model.load_state_dict(state_dict)
        return True
    except:
        print(f"[ERROR] Could not load weights: {path}")
        print(f"[ERROR] Attempting partial load")
    
        state_dict = torch.load(path, map_location=get_device())
        filtered_state_dict = {
            k: v for k, v in state_dict.items() 
            if not any(head_key in k for head_key in ['fc', 'classifier', 'head', 'vocab_projector'])
            }
        try:
            model.load_state_dict(filtered_state_dict, strict=False)
            return True
        except:
            raise Exception("Error loading weights: {path}")
    
    return False
              
def load_projection_model_weights(model, path):
    if not os.path.exists(path):
        print("Weights not loaded: path does not exist.")
        return
    weights = torch.load(path)
    filtered_weights = {k: v for k, v in weights.items() if 'fc' not in k}
    model.load_state_dict(filtered_weights, strict=False)
    print(f'Projection model weights loaded: {path}')

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