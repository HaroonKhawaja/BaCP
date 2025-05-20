import torch
from tqdm import tqdm
from utils import *

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def handle_optimizer_and_pruning(optimizer, pruner, model, fromBaCP, isFinetune, batch_idx):
    """
    Apply pruning to the model based on certain conditions.
    Update the weights of the model and mask if pruning is enabled.
    
    Args:
        fromBaCP: Flag indicating that the model is already pruned from the BaCP framework
        isFinetune: Flag indicating that the model is being fine-tuned
        batch_idx: Current batch index
    """
    # Only calculating new mask once at the start of each epoch (batch_idx=0)
    # We skip pruning if using a pre-pruned model or in the finetuning phase
    if pruner is not None and not fromBaCP and not isFinetune and batch_idx == 0:
        pruner.prune(model)
    
    # Updating the weights of the model
    optimizer.step()

    # Applying the mask to zero out the weights (and gradients)
    if pruner is not None:
        pruner.apply_mask(model)

def handle_save(model, save_path, epoch, test_accuracies):
    """
    Save model weights based on improvements.
    """
    if epoch == 0:
        print(f"weights saved!")
        torch.save(model.state_dict(), save_path)
        return True
    else:
        if test_accuracies[-1] > max(test_accuracies[:-1]):
            print(f"weights saved!")
            torch.save(model.state_dict(), save_path)
            return True
        else:
            return False

def train(model, config, pruner=None, fromBaCP=False):
    """
    Main training loop for supervised learning.
    """
    device = get_device()

    # Required configuration parameters
    trainloader = config['trainloader']
    testloader = config['testloader']
    optimizer = config['optimizer']
    criterion = config['criterion']
    batch_size = config['batch_size']
    epochs = config['epochs']

    # Optional configuration parameters with defaults
    scheduler = config.get('scheduler', None)
    logger = config.get('logger', None)
    save_path = config.get('save_path', None)
    lambda_reg = config.get('lambda_reg', 0)
    pruning_type = config.get('pruning_type', 'None')
    recover_epochs = config.get('recover_epochs', 0)
    stop_epochs = config.get('stop_epochs', 10)

    # Initializing tracking variables
    no_change_for_n_epochs = 0
    losses, train_accuracies, test_accuracies = [], [], []
    
    # Initializing Logger
    if logger is not None:
        logger.create_log()
        logger.log_hyperparameters({
            'save_path': save_path, 
            'epochs': epochs, 
            'optimizer': optimizer, 
            'criterion': criterion, 
            'batch_size': batch_size, 
            'lambda_reg': lambda_reg, 
            'recover_epochs': recover_epochs, 
            'pruning_type': pruning_type
            })
    
    # Main training loop
    for epoch in range(epochs):    
        desc=f"Training Model [{epoch + 1}/{epochs}]"

        avg_loss, train_accuracy = run_epoch(
            model, trainloader, optimizer, criterion, 
            lambda_reg, device, desc, pruner, fromBaCP
            )
        losses.append(avg_loss)
        train_accuracies.append(train_accuracy)

        # Evaluating model after each epoch on test set
        test_accuracy = test(model, testloader)
        test_accuracies.append(test_accuracy)

        # Printing statistics
        info = f"Epoch {epoch + 1}/{epochs}: Average loss: {avg_loss:.5f} - Training Accuracy: {train_accuracy:.2f}% - Testing Accuracy: {test_accuracy:.2f}%\n" 
        print(info)

        # Logging information if logger provided
        if logger is not None:
            logger.log_epochs(info) 

        # Updating scheduler if provided
        if scheduler is not None:
            scheduler.step(epoch)
        
        # If pruning is enabled, update pruning ratio and fine-tune
        if pruner and not fromBaCP:
            fine_tune(
                model, trainloader, testloader, optimizer, criterion, 
                pruner, lambda_reg, fromBaCP, recover_epochs, logger
                )
            print(f"Sparsity of pruned model: {get_model_sparsity(model):.3f}")
            pruner.ratio_step()

        # Saving weights after each epoch
        if save_path:
            if handle_save(model, save_path, epoch, test_accuracies):
                no_change_for_n_epochs = 0
            else:
                no_change_for_n_epochs += 1

        # Early stopping if no improvements for set amount of epochs
        if no_change_for_n_epochs >= stop_epochs:
            print(f"No improvement in accuracy for {stop_epochs} epochs. Training stopped!")
            break

    return losses, train_accuracies, test_accuracies

def fine_tune(model, trainloader, testloader, optimizer, criterion, pruner, lambda_reg, fromBaCP, num_epochs=0, logger=None):
    """
    Fine-tune the model after pruning to recover accuracy.
    """
    device = get_device()
    losses, train_accuracies, test_accuracies = [], [], []

    # Running fine-tuning loop
    for epoch in range(num_epochs):
        desc = f"Fine-tuning Epoch [{epoch + 1}/{num_epochs}]"

        avg_loss, train_accuracy = run_epoch(
            model, trainloader, optimizer, criterion, lambda_reg, 
            device, desc, pruner, fromBaCP, isFinetune=True
            )
        losses.append(avg_loss)
        train_accuracies.append(train_accuracy)        

        # Evaluating model after each epoch on test set
        test_accuracy = test(model, testloader)
        test_accuracies.append(test_accuracy)

        # Printing statistics
        info = f"Recovery epoch {epoch + 1}/{num_epochs}: Average loss: {avg_loss:.5f} - Training Accuracy: {train_accuracy:.2f}% - Testing Accuracy: {test_accuracy:.2f}%\n" 
        print(info)

        # Logging information if logger provided
        if logger is not None:
            logger.log_epochs(info) 

def run_epoch(model, trainloader, optimizer, criterion, lambda_reg, device, desc="", pruner=None, fromBaCP=False, isFinetune=False):
    """
    Run a single training epoch.
    """
    model.train()
    epoch_loss, correct, total = 0, 0, 0

    batch = tqdm(trainloader, desc=desc)
    for batch_idx, (images, labels) in enumerate(batch):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        # Adding L2 regularization
        l2_reg = sum(param.pow(2).sum() for param in model.parameters())
        total_loss = loss + (lambda_reg * l2_reg)

        # Backward pass: gradients are recalculated after each batch
        total_loss.backward()   
        
        # Updating optimizer and mask if pruning is enabled
        handle_optimizer_and_pruning(optimizer, pruner, model, fromBaCP, isFinetune, batch_idx)

        epoch_loss += total_loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        batch_acc = 100 * correct / total
        batch.set_postfix(
            total_loss=f"{total_loss.item():.3f}", 
            batch_acc=f"{batch_acc:.2f}%", 
            sparsity=get_model_sparsity(model).item()
            )
    
    # Calculating average loss and training accuracy
    avg_loss = epoch_loss / len(trainloader)
    train_acc = 100 * correct / total
    return avg_loss, train_acc

def test(model, dataloader):
    """
    Evaluating model accuracy on test set.
    """
    model.eval()  
    correct, total = 0, 0
    device = get_device()
    
    with torch.no_grad(): 
        batch = tqdm(dataloader, desc="Evaluating Model")
        for images, labels in batch:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)  
            _, predicted = outputs.max(1)  
            total += labels.size(0) 
            correct += predicted.eq(labels).sum().item() 
            
            batch.set_postfix(acc=f"{100*(correct/total):.2f}%")
    # Calculating final accuracy
    accuracy = 100 * (correct / total) 
    return accuracy




            