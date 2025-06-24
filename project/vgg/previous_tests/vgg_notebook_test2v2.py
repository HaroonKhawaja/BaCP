# Databricks notebook source
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from bacp import BaCPLearner, create_models_for_cap
from contrastive_learning import ContrastiveLearner
from supervised_learning import train, test
from models import EncoderProjectionNetwork, ClassificationNetwork
from datasets_class import CreateDatasets
from logger import Logger
from unstructured_pruning import MovementPrune, MagnitudePrune, RigLScheduler
from pruning_utils import *
from utils import *
from constants import *

# COMMAND ----------

# DBTITLE 1,CIFAR-10 Dataloaders
datasets = CreateDatasets('/dbfs/datasets')

# Data for supervised learning
trainset_c10_cls_fn, testset_c10_cls_fn = datasets.get_dataset_fn('supervised', DATASET)
trainset_c10_cls, testset_c10_cls = trainset_c10_cls_fn(), testset_c10_cls_fn()

trainloader_c10_cls = DataLoader(trainset_c10_cls, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
testloader_c10_cls = DataLoader(testset_c10_cls, BATCH_SIZE, shuffle=False)

# 2-view augmented data for supervised contrastive learning+
trainset_c10_cl_fn, testset_c10_cl_fn = datasets.get_dataset_fn('supcon', DATASET)
trainset_c10_cl, testset_c10_cl = trainset_c10_cl_fn(), testset_c10_cl_fn()

trainloader_c10_cl = DataLoader(trainset_c10_cl, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
testloader_c10_cl = DataLoader(testset_c10_cl, BATCH_SIZE, shuffle=False)


# COMMAND ----------

# MAGIC %md
# MAGIC # Baseline Accuracies

# COMMAND ----------

# MAGIC %md
# MAGIC ## VGG 11

# COMMAND ----------

# Initializing projection model for supervised-contrastive learning
model_name = 'vgg11'    
projection_model = EncoderProjectionNetwork(model_name, 128)

# Initializing hyperparameters and classes
optimizer_type = 'sgd'
optimizer_cfg = {
    'model':        projection_model,
    'lr':           LR_VGG11,
    'momentum':     MOMENTUM,
    'weight_decay': WEIGHT_DECAY
}
optimizer = set_optimizer(optimizer_type, optimizer_cfg)
scheduler = None
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_supcon_c10.pt'
logger = Logger(model_name, learning_type='supcon')

config = {
    'n_views':          2,
    'optimizer':        optimizer,
    'epochs':           EPOCHS,
    'scheduler':        scheduler,
    'batch_size':       BATCH_SIZE,
    'temperature':      TEMP,
    'base_temperature': BASE_TEMP,
    'loss_type':        'supcon',
}

supcon_learner = ContrastiveLearner(projection_model, config)

# Set if trained
is_trained = False
if not is_trained:
    supcon_learner.train(trainloader_c10_cl, save_path, logger)


# COMMAND ----------

# Creating a classification net for downstream task
model_name = 'vgg11'    
cls_model = ClassificationNetwork(model_name, CIFAR10_CLASSES, False).to(get_device())

# Loading the projection models backbone weights into the new classification net
load_projection_model_weights(cls_model, f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_supcon_c10.pt')

# Initializing Hyperparameters
optimizer_type = 'adam'
optimizer_cfg = {
    'model':        cls_model,
    'lr':           LR_LINEAR,
    'momentum':     MOMENTUM,
    'weight_decay': WEIGHT_DECAY
}
optimizer = set_optimizer(optimizer_type, optimizer_cfg)
scheduler = None
criterion = nn.CrossEntropyLoss()
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_sup_c10.pt'
logger = Logger(model_name, learning_type='cls')

config = {
    'trainloader':      trainloader_c10_cls,
    'testloader':       testloader_c10_cls,
    'optimizer':        optimizer,
    'scheduler':        scheduler,
    'criterion':        nn.CrossEntropyLoss(),
    'epochs':           LINEAR_EPOCHS,
    "batch_size":       BATCH_SIZE,
    "save_path":        save_path,
    "logger":           logger,
    "lambda_reg":       0.0001,
    'recover_epochs':   0,
    'pruning_type':     "",
}

# Set True if trained
is_trained = False
if not is_trained:
    losses, train_accuracies, test_accuracies = train(cls_model,
                                                      config)
    graph_losses_n_accs(losses, train_accuracies, test_accuracies)
    
# Evaluating model
load_weights(cls_model, save_path)
acc = test(cls_model, testloader_c10_cls)
print(f"\nAccuracy of model is: {acc}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## VGG-19

# COMMAND ----------

# DBTITLE 1,VGG 19 SUPERVISED CONTRASTIVE LEARNING ON CIFAR 10
# Initializing projection model for supervised-contrastive learning
model_name = 'vgg19'    
projection_model = EncoderProjectionNetwork(model_name, 128)

# Initializing hyperparameters and classes
optimizer_type = 'sgd'
optimizer_cfg = {
    'model':        projection_model,
    'lr':           LR_VGG19,
    'momentum':     MOMENTUM,
    'weight_decay': WEIGHT_DECAY
}
optimizer = set_optimizer(optimizer_type, optimizer_cfg)
scheduler = None
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_supcon_c10.pt'
logger = Logger(model_name, learning_type='supcon')

config = {
    'n_views':          2,
    'optimizer':        optimizer,
    'epochs':           EPOCHS,
    'scheduler':        scheduler,
    'batch_size':       BATCH_SIZE,
    'temperature':      TEMP,
    'base_temperature': BASE_TEMP,
    'loss_type':        'supcon',
}

supcon_learner = ContrastiveLearner(projection_model, config)

# Set if trained
is_trained = False
if not is_trained:
    supcon_learner.train(trainloader_c10_cl, save_path, logger)


# COMMAND ----------

# Creating a classification net for downstream task
model_name = 'vgg19'    
cls_model = ClassificationNetwork(model_name, CIFAR10_CLASSES, False).to(get_device())

# Loading the projection models backbone weights into the new classification net
load_projection_model_weights(cls_model, f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_supcon_c10.pt')

# Initializing Hyperparameters
optimizer_type = 'adam'
optimizer_cfg = {
    'model':        cls_model,
    'lr':           LR_LINEAR,
    'momentum':     MOMENTUM,
    'weight_decay': WEIGHT_DECAY
}
optimizer = set_optimizer(optimizer_type, optimizer_cfg)
scheduler = None
criterion = nn.CrossEntropyLoss()
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_sup_c10.pt'
logger = Logger(model_name, learning_type='cls')

config = {
    'trainloader':      trainloader_c10_cls,
    'testloader':       testloader_c10_cls,
    'optimizer':        optimizer,
    'scheduler':        scheduler,
    'criterion':        nn.CrossEntropyLoss(),
    'epochs':           LINEAR_EPOCHS,
    "batch_size":       BATCH_SIZE,
    "save_path":        save_path,
    "logger":           logger,
    "lambda_reg":       0.0001,
    'recover_epochs':   0,
    'pruning_type':     "",
}

# Set True if trained
is_trained = False
if not is_trained:
    losses, train_accuracies, test_accuracies = train(cls_model,
                                                      config)
    graph_losses_n_accs(losses, train_accuracies, test_accuracies)
    
# Evaluating model
load_weights(cls_model, save_path)
acc = test(cls_model, testloader_c10_cls)
print(f"\nAccuracy of model is: {acc}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Pruning Accuracies

# COMMAND ----------

# MAGIC %md
# MAGIC ## VGG-11

# COMMAND ----------

# MAGIC %md
# MAGIC ### Magnitude Prune

# COMMAND ----------

# Creating a classification net for downstream task
model_name = 'vgg11'    
cls_model = ClassificationNetwork(model_name, CIFAR10_CLASSES, False).to(get_device())
load_weights(cls_model, f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_sup_c10.pt')

# Initializing Hyperparameters
optimizer_type = 'adam'
optimizer_cfg = {
    'model':        cls_model,
    'lr':           LR_LINEAR,
    'momentum':     MOMENTUM,
    'weight_decay': WEIGHT_DECAY
}
optimizer = set_optimizer(optimizer_type, optimizer_cfg)
scheduler = None
criterion = nn.CrossEntropyLoss()
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_sup_magp_c10.pt'
logger = Logger(model_name, learning_type='pruning')

# Initialing pruning method
pruner = MagnitudePrune(PRUNING_EPOCHS, TARGET_SPARSITY_MID)
pruning_type = 'magnitude_pruning'

config = {
    'trainloader':      trainloader_c10_cls,
    'testloader':       testloader_c10_cls,
    'optimizer':        optimizer,
    'scheduler':        scheduler,
    'criterion':        nn.CrossEntropyLoss(),
    'epochs':           PRUNING_EPOCHS,
    "batch_size":       BATCH_SIZE,
    "save_path":        save_path,
    "logger":           logger,
    "lambda_reg":       0,
    'recover_epochs':   RECOVER_EPOCHS,
    'pruning_type':     pruning_type,
    'stop_epochs':      10,
}

# Set False to train
is_trained = False
if not is_trained:
    losses, train_accuracies, test_accuracies = train(cls_model,
                                                      config,
                                                      pruner)
    torch.save(cls_model.state_dict(), save_path)
    graph_losses_n_accs(losses, 
                        train_accuracies, 
                        test_accuracies)

# Evaluating model
load_weights(cls_model, save_path)
print(f"\nSparsity of pruned model: {get_model_sparsity(cls_model):.3f}")
pruned_acc = test(cls_model, testloader_c10_cls)
print(f"\nAccuracy of pruned model is: {pruned_acc}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Movement Prune

# COMMAND ----------

# Creating a classification net for downstream task
model_name = 'vgg11'    
cls_model = ClassificationNetwork(model_name, CIFAR10_CLASSES, False).to(get_device())
load_weights(cls_model, f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_sup_c10.pt')

# Initializing Hyperparameters
optimizer_type = 'sgd'
optimizer_cfg = {
    'model':        cls_model,
    'lr':           LR_LINEAR,
    'momentum':     MOMENTUM,
    'weight_decay': WEIGHT_DECAY
}
optimizer = set_optimizer(optimizer_type, optimizer_cfg)
scheduler = None
criterion = nn.CrossEntropyLoss()
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_sup_mvmp_c10.pt'
logger = Logger(model_name, learning_type='pruning')

# Initialing pruning method
pruner = MovementPrune(PRUNING_EPOCHS, TARGET_SPARSITY_MID)
pruning_type = 'movement_pruning'

config = {
    'trainloader':      trainloader_c10_cls,
    'testloader':       testloader_c10_cls,
    'optimizer':        optimizer,
    'scheduler':        scheduler,
    'criterion':        nn.CrossEntropyLoss(),
    'epochs':           PRUNING_EPOCHS,
    "batch_size":       BATCH_SIZE,
    "save_path":        save_path,
    "logger":           logger,
    "lambda_reg":       0,
    'recover_epochs':   RECOVER_EPOCHS,
    'pruning_type':     pruning_type,
    'stop_epochs':      10,
}

# Set False to train
is_trained = False
if not is_trained:
    losses, train_accuracies, test_accuracies = train(cls_model,
                                                      config,
                                                      pruner)
    torch.save(cls_model.state_dict(), save_path)
    graph_losses_n_accs(losses, 
                        train_accuracies, 
                        test_accuracies)

# Evaluating model
load_weights(cls_model, save_path)
print(f"\nSparsity of pruned model: {get_model_sparsity(cls_model):.3f}")
pruned_acc = test(cls_model, testloader_c10_cls)
print(f"\nAccuracy of pruned model is: {pruned_acc}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### RigL Pruning

# COMMAND ----------

import os
def create_rigl_pruner(config, save_path):
    if save_path is not None and os.path.exists(save_path):
        checkpoint = torch.load(save_path)
        pruner = RigLScheduler(
            config['model'],
            config['optimizer'],
            state_dict=checkpoint)

    else:
        total_iterations = len(config['trainloader']) * config['epochs']
        T_end = int(0.75 * total_iterations)

        pruner = RigLScheduler(
            config['model'],
            config['optimizer'],
            config['dense_allocation'],
            sparsity_distribution='uniform',
            T_end=T_end,
            delta=50,
            grad_accumulation_n=4,
            static_topo=False,
            ignore_linear_layers=False,
            state_dict=None)
    return pruner

# COMMAND ----------

# Creating a classification net for downstream task
model_name = 'vgg11'    
cls_model = ClassificationNetwork(model_name, CIFAR10_CLASSES, False).to(get_device())
load_weights(cls_model, f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_sup_c10.pt')

# Initializing Hyperparameters
optimizer_type = 'adam'
optimizer_cfg = {
    'model':        cls_model,
    'lr':           0.00001,
    'momentum':     MOMENTUM,
    'weight_decay': WEIGHT_DECAY
}
optimizer = set_optimizer(optimizer_type, optimizer_cfg)
scheduler = None
criterion = nn.CrossEntropyLoss()
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_sup_riglp_c10.pt'
logger = Logger(model_name, learning_type='pruning')

# Initialing pruning method
epochs = 200
pruning_cfg = {
    'model':            cls_model,
    'optimizer':        optimizer,
    'dense_allocation': 1-TARGET_SPARSITY_MID,
    'trainloader':      trainloader_c10_cls,
    'epochs':           epochs,
}
pruning_type = 'rigl_pruning'
pruner = create_rigl_pruner(pruning_cfg, None)

config = {
    'trainloader':      trainloader_c10_cls,
    'testloader':       testloader_c10_cls,
    'optimizer':        optimizer,
    'scheduler':        scheduler,
    'criterion':        nn.CrossEntropyLoss(),
    'epochs':           epochs,
    "batch_size":       BATCH_SIZE,
    "save_path":        save_path,
    "logger":           logger,
    "lambda_reg":       0,
    'recover_epochs':   0,
    'pruning_type':     pruning_type,
    'stop_epochs':      50,
}

# Set False to train
is_trained = False
if not is_trained:
    losses, train_accuracies, test_accuracies = train(cls_model,
                                                      config,
                                                      pruner)
    torch.save(cls_model.state_dict(), save_path)
    graph_losses_n_accs(losses, 
                        train_accuracies, 
                        test_accuracies)

# Evaluating model
load_weights(cls_model, save_path)
print(f"\nSparsity of pruned model: {get_model_sparsity(cls_model):.3f}")
pruned_acc = test(cls_model, testloader_c10_cls)
print(f"\nAccuracy of pruned model is: {pruned_acc}")

# COMMAND ----------

checkpoint = torch.load(save_path)

# COMMAND ----------

# Saving the model, optimizwer, and pruning state dict
torch.save({
    'model_state_dict': cls_model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'rigl_state_dict': {str(k): v for k, v in pruner.__dict__.items()},
}, save_path)
torch.load()

# COMMAND ----------

# MAGIC %md
# MAGIC ## VGG-19

# COMMAND ----------

# MAGIC %md
# MAGIC ### Magnitude Prune

# COMMAND ----------

# Creating a classification net for downstream task
model_name = 'vgg19'    
cls_model = ClassificationNetwork(model_name, CIFAR10_CLASSES, False).to(get_device())
load_weights(cls_model, f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_sup_c10.pt')

# Initializing Hyperparameters
optimizer_type = 'sgd'
optimizer_cfg = {
    'model':        cls_model,
    'lr':           LEARNING_RATE,
    'momentum':     MOMENTUM,
    'weight_decay': WEIGHT_DECAY
}
optimizer = set_optimizer(optimizer_type, optimizer_cfg)
scheduler = None
criterion = nn.CrossEntropyLoss()
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_sup_magp_c10.pt'
logger = Logger(model_name, learning_type='pruning')

# Initialing pruning method
pruner = MagnitudePrune(PRUNING_EPOCHS, TARGET_SPARSITY)
pruning_type = 'magnitude_pruning'

config = {
    'trainloader':      trainloader_c10_cls,
    'testloader':       testloader_c10_cls,
    'optimizer':        optimizer,
    'scheduler':        scheduler,
    'criterion':        nn.CrossEntropyLoss(),
    'epochs':           PRUNING_EPOCHS,
    "batch_size":       BATCH_SIZE,
    "save_path":        save_path,
    "logger":           logger,
    "lambda_reg":       0,
    'recover_epochs':   RECOVER_EPOCHS,
    'pruning_type':     pruning_type,
}

# Set False to train
is_trained = False
if not is_trained:
    losses, train_accuracies, test_accuracies = train(cls_model,
                                                      config,
                                                      pruner)
    torch.save(cls_model.state_dict(), save_path)
    graph_losses_n_accs(losses, 
                        train_accuracies, 
                        test_accuracies)

# Evaluating model
load_weights(cls_model, save_path)
print(f"\nSparsity of pruned model: {get_model_sparsity(cls_model):.3f}")
pruned_acc = test(cls_model, testloader_c10_cls)
print(f"\nAccuracy of pruned model is: {pruned_acc}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Movement Prune

# COMMAND ----------

# Creating a classification net for downstream task
model_name = 'vgg19'    
cls_model = ClassificationNetwork(model_name, CIFAR10_CLASSES, False).to(get_device())
load_weights(cls_model, f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_sup_c10.pt')

# Initializing Hyperparameters
optimizer_type = 'sgd'
optimizer_cfg = {
    'model':        cls_model,
    'lr':           LEARNING_RATE,
    'momentum':     MOMENTUM,
    'weight_decay': WEIGHT_DECAY
}
optimizer = set_optimizer(optimizer_type, optimizer_cfg)
scheduler = None
criterion = nn.CrossEntropyLoss()
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_sup_mvmp_c10.pt'
logger = Logger(model_name, learning_type='pruning')

# Initialing pruning method
pruner = MovementPrune(PRUNING_EPOCHS, TARGET_SPARSITY)
pruning_type = 'movement_pruning'

config = {
    'trainloader':      trainloader_c10_cls,
    'testloader':       testloader_c10_cls,
    'optimizer':        optimizer,
    'scheduler':        scheduler,
    'criterion':        nn.CrossEntropyLoss(),
    'epochs':           PRUNING_EPOCHS,
    "batch_size":       BATCH_SIZE,
    "save_path":        save_path,
    "logger":           logger,
    "lambda_reg":       0,
    'recover_epochs':   RECOVER_EPOCHS,
    'pruning_type':     pruning_type,
}

# Set False to train
is_trained = False
if not is_trained:
    losses, train_accuracies, test_accuracies = train(cls_model,
                                                      config,
                                                      pruner)
    torch.save(cls_model.state_dict(), save_path)
    graph_losses_n_accs(losses, 
                        train_accuracies, 
                        test_accuracies)

# Evaluating model
load_weights(cls_model, save_path)
print(f"\nSparsity of pruned model: {get_model_sparsity(cls_model):.3f}")
pruned_acc = test(cls_model, testloader_c10_cls)
print(f"\nAccuracy of pruned model is: {pruned_acc}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### RigL Prune

# COMMAND ----------

# Creating a classification net for downstream task
model_name = 'vgg19'    
cls_model = ClassificationNetwork(model_name, CIFAR10_CLASSES, False).to(get_device())
load_weights(cls_model, f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_sup_c10.pt')

# Initializing Hyperparameters
optimizer_type = 'sgd'
optimizer_cfg = {
    'model':        cls_model,
    'lr':           LEARNING_RATE,
    'momentum':     MOMENTUM,
    'weight_decay': WEIGHT_DECAY
}
optimizer = set_optimizer(optimizer_type, optimizer_cfg)
scheduler = None
criterion = nn.CrossEntropyLoss()
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_sup_magp_c10.pt'
logger = Logger(model_name, learning_type='pruning')

# Initialing pruning method
total_iterations = len(trainloader_c10_cls) * RIGL_EPOCHS
T_end = int(0.75 * total_iterations)
pruning_type = 'rigl_pruner'
pruner = RigLScheduler(cls_model,
                       optimizer,
                       dense_allocation=(1-TARGET_SPARSITY),
                       sparsity_distribution='uniform',
                       T_end=T_end,
                       delta=100,
                       grad_accumulation_n=1,
                       static_topo=False,
                       ignore_linear_layers=False,
                       state_dict=None)

config = {
    'trainloader':      trainloader_c10_cls,
    'testloader':       testloader_c10_cls,
    'optimizer':        optimizer,
    'scheduler':        scheduler,
    'criterion':        nn.CrossEntropyLoss(),
    'epochs':           PRUNING_EPOCHS,
    "batch_size":       BATCH_SIZE,
    "save_path":        save_path,
    "logger":           logger,
    "lambda_reg":       0,
    'recover_epochs':   RECOVER_EPOCHS,
    'pruning_type':     pruning_type,
}

# Set False to train
is_trained = False
if not is_trained:
    losses, train_accuracies, test_accuracies = train(cls_model,
                                                      config,
                                                      pruner)
    torch.save(cls_model.state_dict(), save_path)
    graph_losses_n_accs(losses, 
                        train_accuracies, 
                        test_accuracies)

# Evaluating model
load_weights(cls_model, save_path)
print(f"\nSparsity of pruned model: {get_model_sparsity(cls_model):.3f}")
pruned_acc = test(cls_model, testloader_c10_cls)
print(f"\nAccuracy of pruned model is: {pruned_acc}")

# COMMAND ----------

# MAGIC %md
# MAGIC # BaCP Accuracies

# COMMAND ----------

# MAGIC %md
# MAGIC ## VGG-11

# COMMAND ----------

# MAGIC %md
# MAGIC ### Magnitude Prune

# COMMAND ----------

# Creating projection models for BaCP framework
model_name = 'vgg11'

# Projection networks
finetuned_weights = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_supcon_c10.pt'
pre_trained_model, current_model, finetuned_model = create_models_for_cap(model_name, finetuned_weights)

# Fine-tuned classification network
cls_model = ClassificationNetwork(model_name, CIFAR10_CLASSES, False).to(get_device())
load_weights(cls_model, f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_sup_c10.pt')

print(f"Current model sparsity: {get_model_sparsity(current_model)}")

# Initializing Hyperparameters
optimizer_type = 'sgd'
optimizer_cfg = {
    'model':        current_model,
    'lr':           LEARNING_RATE,
    'momentum':     MOMENTUM,
    'weight_decay': WEIGHT_DECAY
}
optimizer = set_optimizer(optimizer_type, optimizer_cfg)
scheduler = None
criterion = nn.CrossEntropyLoss()
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_bacp_magp_c10'
logger = Logger(model_name, learning_type='bacp')

# Initializing pruner
pruner = MagnitudePrune(BACP_EPOCHS, TARGET_SPARSITY_MID)

config = {
    'model_name':       model_name,
    'n_views':          2,
    'optimizer':        optimizer,
    'scheduler':        scheduler,               
    'criterion':        criterion,
    'temperature':      TEMP,
    'base_temperature': BASE_TEMP,
    'target_sparsity':  TARGET_SPARSITY_MID,   
    'logger':           logger,
    'epochs':           BACP_EPOCHS,         
    'batch_size':       BATCH_SIZE,     
    'num_classes':      CIFAR10_CLASSES,    # Change this based on dataloader
    'lambdas':          LAMBDAS,            # None lambas => lambdas become learnable parameters
    'save_path':        save_path,
    'pruner':           pruner,
}

cap_learner = BaCPLearner(current_model, pre_trained_model, finetuned_model, cls_model, config)

# Set False to train
is_trained = False
if not is_trained:
    cap_learner.cap_train(trainloader_c10_cl)

# COMMAND ----------

# Creating classification model with unfrozen parameters
model_name = 'vgg11'
cls_model = cap_learner.create_classification_net(False)
print(f"Current model sparsity: {get_model_sparsity(cls_model)}")

# Generate masks from model
cap_learner.generate_mask_from_model()

# Initializing Hyperparameters
optimizer_type = 'adam'
optimizer_cfg = {
    'model':        cls_model,
    'lr':           0.0001,
    'momentum':     MOMENTUM,
    'weight_decay': WEIGHT_DECAY
}
optimizer = set_optimizer(optimizer_type, optimizer_cfg)
scheduler = None
criterion = nn.CrossEntropyLoss()
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_bacp_magp_cls_c10.pt'
logger = Logger(model_name, learning_type='cls')

# Initializing pruner
pruner = cap_learner.get_pruner()
pruning_type = 'magnitude_pruning'

config = {
    'trainloader':      trainloader_c10_cls,
    'testloader':       testloader_c10_cls,
    'optimizer':        optimizer,
    'scheduler':        scheduler,
    'criterion':        nn.CrossEntropyLoss(),
    'epochs':           LINEAR_EPOCHS,
    "batch_size":       BATCH_SIZE,
    "save_path":        save_path,
    "logger":           logger,
    'lambda_reg':       0,
    'recover_epochs':   0,
    'pruning_type':     pruning_type,
}

# Set False to train
is_trained = False
if not is_trained:
    losses, train_accuracies, test_accuracies = train(cls_model,
                                                      config,
                                                      pruner,
                                                      True)
    graph_losses_n_accs(losses, 
                        train_accuracies, 
                        test_accuracies)

# Evaluating model
load_weights(cls_model, save_path)
acc = test(cls_model, testloader_c10_cls)
print(f"\nAccuracy of model is: {acc}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Movement Prune

# COMMAND ----------

# Creating projection models for BaCP framework
model_name = 'vgg11'
finetuned_weights = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_supcon_c10.pt'
pre_trained_model, current_model, finetuned_model = create_models_for_cap(model_name, finetuned_weights)
print(f"Current model sparsity: {get_model_sparsity(current_model)}")

# Initializing Hyperparameters
optimizer_type = 'sgd'
optimizer_cfg = {
    'model':        cls_model,
    'lr':           LEARNING_RATE,
    'momentum':     MOMENTUM,
    'weight_decay': WEIGHT_DECAY
}
optimizer = set_optimizer(optimizer_type, optimizer_cfg)
scheduler = None
criterion = nn.CrossEntropyLoss()
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_bacp_mvmp_c10'
logger = Logger(model_name, learning_type='bacp')

# Initializing pruner
pruner = MovementPrune(BACP_EPOCHS, TARGET_SPARSITY)

config = {
    'model_name':       model_name,
    'n_views':          2,
    'optimizer':        optimizer,
    'scheduler':        scheduler,               
    'criterion':        criterion,
    'temperature':      TEMPERATURE,
    'base_temperature': BASE_TEMPERATURE,
    'target_sparsity':  TARGET_SPARSITY,   
    'logger':           logger,
    'epochs':           BACP_EPOCHS,         
    'batch_size':       BATCH_SIZE,     
    'num_classes':      CIFAR10_CLASSES,    # Change this based on dataloader
    'lambdas':          LAMBDAS,               # None lambas => lambdas become learnable parameters
    'save_path':        save_path,
    'pruner':           pruner,
    'finetuning_epochs':0
}

cap_learner = BaCPLearner(current_model, pre_trained_model, finetuned_model, config)

# Set False to train
is_trained = False
if not is_trained:
    cap_learner.cap_train(trainloader_c10_cl)


# COMMAND ----------

# Creating classification model with unfrozen parameters
model_name = 'vgg11'
cls_model = cap_learner.create_classification_net(False)
print(f"Current model sparsity: {get_model_sparsity(cls_model)}")

# Generate masks from model
cap_learner.generate_mask_from_model()

# Initializing Hyperparameters
optimizer_type = 'sgd'
optimizer_cfg = {
    'model':        cls_model,
    'lr':           LEARNING_RATE,
    'momentum':     MOMENTUM,
    'weight_decay': WEIGHT_DECAY
}
optimizer = set_optimizer(optimizer_type, optimizer_cfg)
scheduler = None
criterion = nn.CrossEntropyLoss()
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_bacp_mvmp_cls_c10.pt'
logger = Logger(model_name, learning_type='cls')

# Initializing pruner
pruner = cap_learner.get_pruner()
pruning_type = 'movement_pruning'

config = {
    'trainloader':      trainloader_c10_cls,
    'testloader':       testloader_c10_cls,
    'optimizer':        optimizer,
    'scheduler':        scheduler,
    'criterion':        nn.CrossEntropyLoss(),
    'epochs':           FINETUNE_EPOCHS,
    "batch_size":       BATCH_SIZE,
    "save_path":        save_path,
    "logger":           logger,
    'lambda_reg':       0,
    'recover_epochs':   0,
    'pruning_type':     pruning_type,
}

# Set False to train
is_trained = False
if not is_trained:
    losses, train_accuracies, test_accuracies = train(cls_model,
                                                      config,
                                                      pruner,
                                                      True)
    graph_losses_n_accs(losses, 
                        train_accuracies, 
                        test_accuracies)

# Evaluating model
load_weights(cls_model, save_path)
acc = test(cls_model, testloader_c10_cls)
print(f"\nAccuracy of model is: {acc}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## VGG-19

# COMMAND ----------

# MAGIC %md
# MAGIC ### Magnitude **Prune**

# COMMAND ----------

# Creating projection models for BaCP framework
model_name = 'vgg19'
finetuned_weights = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_supcon_c10.pt'
pre_trained_model, current_model, finetuned_model = create_models_for_cap(model_name, finetuned_weights)
print(f"Current model sparsity: {get_model_sparsity(current_model)}")

# Initializing Hyperparameters
optimizer = optim.Adam(current_model.parameters(), lr=0.0005)
scheduler = None
criterion = nn.CrossEntropyLoss()
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_bacp_magp_c10'
logger = Logger(model_name, learning_type='bacp')

# Initializing pruner
pruner = MagnitudePrune(BACP_EPOCHS, TARGET_SPARSITY)

config = {
    'model_name':       model_name,
    'n_views':          2,
    'optimizer':        optimizer,
    'scheduler':        scheduler,               
    'criterion':        criterion,
    'temperature':      TEMPERATURE,
    'base_temperature': BASE_TEMPERATURE,
    'target_sparsity':  TARGET_SPARSITY,   
    'logger':           logger,
    'epochs':           BACP_EPOCHS,         
    'batch_size':       BATCH_SIZE,     
    'num_classes':      CIFAR10_CLASSES,    # Change this based on dataloader
    'lambdas':          LAMBDAS,               # None lambas => lambdas become learnable parameters
    'save_path':        save_path,
    'pruner':           pruner,
    'finetuning_epochs':0
}

cap_learner = BaCPLearner(current_model, pre_trained_model, finetuned_model, config)

# Set False to train
is_trained = False
if not is_trained:
    cap_learner.cap_train(trainloader_c10_cl)

# COMMAND ----------

# Creating classification model with unfrozen parameters
model_name = 'vgg19'
cls_model = cap_learner.create_classification_net(False)
print(f"Current model sparsity: {get_model_sparsity(cls_model)}")

# Generate masks from model
cap_learner.generate_mask_from_model()

# Initializing Hyperparameters
optimizer = optim.Adam(cls_model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
criterion = nn.CrossEntropyLoss()
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_bacp_magp_cls_c10.pt'
logger = Logger(model_name, learning_type='cls')

# Initializing pruner
pruner = cap_learner.get_pruner()
pruning_type = 'magnitude_pruning'

config = {
    'trainloader':      trainloader_c10_cls,
    'testloader':       testloader_c10_cls,
    'optimizer':        optimizer,
    'scheduler':        scheduler,
    'criterion':        nn.CrossEntropyLoss(),
    'epochs':           FINETUNE_EPOCHS,
    "batch_size":       BATCH_SIZE,
    "save_path":        save_path,
    "logger":           logger,
    'lambda_reg':       0,
    'recover_epochs':   0,
    'pruning_type':     pruning_type,
}

# Set False to train
is_trained = False
if not is_trained:
    losses, train_accuracies, test_accuracies = train(cls_model,
                                                      config,
                                                      pruner,
                                                      True)
    graph_losses_n_accs(losses, 
                        train_accuracies, 
                        test_accuracies)

# Evaluating model
load_weights(cls_model, save_path)
acc = test(cls_model, testloader_c10_cls)
print(f"\nAccuracy of model is: {acc}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Movement Prune

# COMMAND ----------

# Creating projection models for BaCP framework
model_name = 'vgg19'
finetuned_weights = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_supcon_c10.pt'
pre_trained_model, current_model, finetuned_model = create_models_for_cap(model_name, finetuned_weights)
print(f"Current model sparsity: {get_model_sparsity(current_model)}")

# Initializing Hyperparameters
optimizer = optim.Adam(current_model.parameters(), lr=0.0005)
scheduler = None
criterion = nn.CrossEntropyLoss()
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_bacp_mvmp_c10'
logger = Logger(model_name, learning_type='bacp')

# Initializing pruner
pruner = MovementPrune(BACP_EPOCHS, TARGET_SPARSITY)

config = {
    'model_name':       model_name,
    'n_views':          2,
    'optimizer':        optimizer,
    'scheduler':        scheduler,               
    'criterion':        criterion,
    'temperature':      TEMPERATURE,
    'base_temperature': BASE_TEMPERATURE,
    'target_sparsity':  TARGET_SPARSITY,   
    'logger':           logger,
    'epochs':           BACP_EPOCHS,         
    'batch_size':       BATCH_SIZE,     
    'num_classes':      CIFAR10_CLASSES,    # Change this based on dataloader
    'lambdas':          LAMBDAS,               # None lambas => lambdas become learnable parameters
    'save_path':        save_path,
    'pruner':           pruner,
    'finetuning_epochs':0
}

cap_learner = BaCPLearner(current_model, pre_trained_model, finetuned_model, config)

# Set False to train
is_trained = False
if not is_trained:
    cap_learner.cap_train(trainloader_c10_cl)

# Evaluating model
load_weights(cls_model, save_path)
acc = test(cls_model, testloader_c10_cls)
print(f"\nAccuracy of model is: {acc}")

# COMMAND ----------

# Creating classification model with unfrozen parameters
model_name = 'vgg19'
cls_model = cap_learner.create_classification_net(False)
print(f"Current model sparsity: {get_model_sparsity(cls_model)}")

# Generate masks from model
cap_learner.generate_mask_from_model()

# Initializing Hyperparameters
optimizer = optim.Adam(cls_model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
criterion = nn.CrossEntropyLoss()
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_bacp_mvmp_cls_c10.pt'
logger = Logger(model_name, learning_type='cls')

# Initializing pruner
pruner = cap_learner.get_pruner()
pruning_type = 'movement_pruning'

config = {
    'trainloader':      trainloader_c10_cls,
    'testloader':       testloader_c10_cls,
    'optimizer':        optimizer,
    'scheduler':        scheduler,
    'criterion':        nn.CrossEntropyLoss(),
    'epochs':           FINETUNE_EPOCHS,
    "batch_size":       BATCH_SIZE,
    "save_path":        save_path,
    "logger":           logger,
    'lambda_reg':       0,
    'recover_epochs':   0,
    'pruning_type':     pruning_type,
}

# Set False to train
is_trained = False
if not is_trained:
    losses, train_accuracies, test_accuracies = train(cls_model,
                                                      config,
                                                      pruner,
                                                      True)
    graph_losses_n_accs(losses, 
                        train_accuracies, 
                        test_accuracies)

# Evaluating model
load_weights(cls_model, save_path)
acc = test(cls_model, testloader_c10_cls)
print(f"\nAccuracy of model is: {acc}")
