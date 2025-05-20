import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.amp import autocast 
import tqdm
from copy import deepcopy
    
def plot_weight_distribution(model, ngraphs=8, nonzeros_only=False, bins=256):
    """Plots the weight tensors per layer

    Args:
        model (nn.Module): The pytorch module from which the tensor weight distribution will be plotted
        nonzeros_only (bool, optional): Plots only the non-zero value weight tensors. Defaults to False.
        bins (int, optional): The distribution of weights across a set number of bins for a histogram plot. Defaults to 256.
    """
    
    nrows = int((ngraphs / 4) + 1)
    ncols = int((ngraphs / nrows) + 1)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(50, 24))
    axes = axes.flatten()
    plot_index = 0
    
    for name, param in model.named_parameters():
        # getting all the weight parameters
        if param.dim() > 1:
            ax = axes[plot_index]
            if nonzeros_only:
                param_cpu = param.detach().reshape(-1).cpu()
                param_cpu = param_cpu[param_cpu != 0.0].view(-1)
                ax.hist(param_cpu, bins=bins, density=True,
                        color = 'blue', alpha = 0.5)
            else:   
                param_cpu = param.detach().reshape(-1).cpu()
                ax.hist(param_cpu, bins=bins, density=True,
                        color = 'blue', alpha = 0.5)
            
            ax.set_xlabel(name)
            ax.set_ylabel('density')
            plot_index += 1
            
    for j in range(plot_index, len(axes)):
        fig.delaxes(axes[j])
        
    fig.suptitle('Histogram of Weights')
    fig.tight_layout()
    plt.show()

def plot_sensitivity_scan(model, dense_acc, weight_dict, sparsity_ratios, is_structured=False):
    """Plots the models accuracy across layers with different sparsity ratios at every layer. 
    The dense weight accuracy is plotted as a dashed red line while the sparse weight accuracies are plotted as 
    a blue line.

    Args:
        model (nn.Module): The pytorch model whose feature and classifier names will be used in these plots.
        dense_acc (float): The accuracy of the model using dense weights.
        weight_dict (dict): A dictionary containing all the accuracies for each layer across varying sparsity ratios.
        sparsity_ratios (list): A list of sparsity raios.
        is_structured (bool): To boolean value used to display the amount of plots based on being structured and unstructured.
    """
    conv_name_weights = [(name, weights) for (name, weights) in model.named_parameters() if weights.dim() > 1]
    
    rows = 3
    cols = int(len(conv_name_weights)/rows + 1)
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 7))
    axes = axes.flatten()
    
    fig.suptitle('Sensitivity Analysis: Model Accuracy vs. Pruning Ratio')
    for i, (name, weights) in enumerate(conv_name_weights):
        if is_structured and i == (len(conv_name_weights)-1):
            i -= 1
            break
        
        weight_accuracy = weight_dict[name]
        axes[i].axhline(y=dense_acc, color='r', linestyle='--', label="Dense Model Accuracy")
        
        axes[i].plot(sparsity_ratios, weight_accuracy, color='b', label="Accuracy after Pruning")
        axes[i].set_title(name)
        axes[i].set_xlabel("Pruning Ratio")
        axes[i].set_ylabel("Accuracy")
        
        axes[i].set_xticks(sparsity_ratios)
        
        axes[i].set_ylim(0, 100)
        # axes[i].yaxis.set_major_locator(ticker.MultipleLocator(20))  # Show y-ticks every 10 units

        axes[i].set_xlim(sparsity_ratios[0], sparsity_ratios[-1])
        
        axes[i].grid(axis='x')
        axes[i].grid(axis='y')
        axes[i].legend(loc='best')
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.show()

def get_total_parameters(model, count_nonzero_only=False):
    num_elements = 0
    for param in model.parameters():
        if count_nonzero_only:
            num_elements += param.count_nonzero()
        else:
            num_elements += param.numel()
            
    return num_elements

def get_inference_time(model, weights, dataloader, runs=100, device="cuda"):
    import time
    model.eval()
    
    times = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            if i == runs:
                break
            
            start = time.time()
            _ = model(images.to(device))
            end = time.time()
            
            times.append(end - start)
            
    return np.mean(np.array(times))

def plot_num_parameters_distribution(model):
    num_parameters = dict()
    for name, param in model.named_parameters():
        if param.dim() > 1:
            num_parameters[name] = param.numel()
    fig = plt.figure(figsize=(8, 6))
    plt.grid(axis='y')
    plt.bar(list(num_parameters.keys()), list(num_parameters.values()))
    plt.title('#Parameter Distribution')
    plt.ylabel('Number of Parameters')
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.show()
