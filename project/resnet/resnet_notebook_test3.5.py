# Databricks notebook source
# MAGIC %md
# MAGIC Without Knowledge-distillation, only standard contrastive learning.

# COMMAND ----------

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
from models import EncoderProjectionNetwork, ClassificationNetwork, make_resnet_for_cifar10
from datasets_class import CreateDatasets
from logger import Logger
from unstructured_pruning import MovementPrune, MagnitudePrune, RigLScheduler, create_rigl_pruner
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
# MAGIC ## ResNet-50

# COMMAND ----------

# Initializing projection model for supervised-contrastive learning
model_name = 'resnet50'    
projection_model = EncoderProjectionNetwork(model_name, 128)
make_resnet_for_cifar10(projection_model)

# Initializing hyperparameters and classes
optimizer_type = 'sgd'
optimizer_cfg = {
    'model':        projection_model,
    'lr':           0.001,
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
    'epochs':           EPOCHS_RESNET50,
    'scheduler':        scheduler,
    'batch_size':       BATCH_SIZE,
    'temperature':      TEMP,
    'base_temperature': BASE_TEMP,
    'loss_type':        'supcon',
}

supcon_learner = ContrastiveLearner(projection_model, config)

# Set if trained
is_trained = True
if not is_trained:
    supcon_learner.train(trainloader_c10_cl, save_path, logger)


# COMMAND ----------

# Creating a classification net for downstream task
model_name = 'resnet50'    
cls_model = ClassificationNetwork(model_name, CIFAR10_CLASSES, False).to(get_device())
make_resnet_for_cifar10(cls_model)
cls_model.to(get_device())

# Loading the projection models backbone weights into the new classification net
load_projection_model_weights(cls_model, f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_supcon_c10.pt')

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
    "lambda_reg":       5e-4,
    'recover_epochs':   0,
    'pruning_type':     "",
    'stop_epochs':     25,
}

# Set True if trained
is_trained = True
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
# MAGIC ## ResNet-101

# COMMAND ----------

# DBTITLE 1,VGG 19 SUPERVISED CONTRASTIVE LEARNING ON CIFAR 10
# Initializing projection model for supervised-contrastive learning
model_name = 'resnet101'    
projection_model = EncoderProjectionNetwork(model_name, 128)
make_resnet_for_cifar10(projection_model)
projection_model.to(get_device())

# Initializing hyperparameters and classes
optimizer_type = 'sgd'
optimizer_cfg = {
    'model':        projection_model,
    'lr':           0.0005,
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
    'epochs':           EPOCHS_RESNET101,
    'scheduler':        scheduler,
    'batch_size':       BATCH_SIZE,
    'temperature':      TEMP,
    'base_temperature': BASE_TEMP,
    'loss_type':        'supcon',
}

supcon_learner = ContrastiveLearner(projection_model, config)

# Set if trained
is_trained = True
if not is_trained:
    supcon_learner.train(trainloader_c10_cl, save_path, logger)


# COMMAND ----------

# Creating a classification net for downstream task
model_name = 'resnet101'    
cls_model = ClassificationNetwork(model_name, CIFAR10_CLASSES, False).to(get_device())
make_resnet_for_cifar10(cls_model)
cls_model.to(get_device())

# Loading the projection models backbone weights into the new classification net
load_projection_model_weights(cls_model, f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_supcon_c10.pt')

# Initializing Hyperparameters
optimizer_type = 'adam'
optimizer_cfg = {
    'model':        cls_model,
    'lr':           0.00005,
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
    "lambda_reg":       5e-4,
    'recover_epochs':   0,
    'pruning_type':     "",
    'stop_epochs':     25,
}

# Set True if trained
is_trained = True
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
# MAGIC ## ResNet-50

# COMMAND ----------

# MAGIC %md
# MAGIC ### Magnitude Prune

# COMMAND ----------

# Creating a classification net for downstream task
model_name = 'resnet50'    

cls_model = ClassificationNetwork(model_name, CIFAR10_CLASSES, False).to(get_device())
load_weights(cls_model, f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_sup_c10.pt')

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
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_sup_magp_c10.pt'
logger = Logger(model_name, learning_type='pruning')

# Initialing pruning method
pruner = MagnitudePrune(PRUNING_EPOCHS, 0.965)
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
    'recover_epochs':   10,
    'pruning_type':     pruning_type,
    'stop_epochs':      5,
}

# Set False to train
is_trained = True
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
model_name = 'resnet50'    

cls_model = ClassificationNetwork(model_name, CIFAR10_CLASSES, False).to(get_device())
load_weights(cls_model, f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_sup_c10.pt')

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
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_sup_mvmp_c10.pt'
logger = Logger(model_name, learning_type='pruning')

# Initialing pruning method
pruner = MagnitudePrune(PRUNING_EPOCHS, TARGET_SPARSITY_MID)
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
is_trained = True
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

# Creating a classification net for downstream task
model_name = 'resnet50'    
cls_model = ClassificationNetwork(model_name, CIFAR10_CLASSES, False).to(get_device())
load_weights(cls_model, f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_sup_c10.pt')

# Initializing Hyperparameters
optimizer_type = 'adam'
optimizer_cfg = {
    'model':        cls_model,
    'lr':           0.0005,
    'momentum':     MOMENTUM,
    'weight_decay': 5e-4,
}
optimizer = set_optimizer(optimizer_type, optimizer_cfg)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
criterion = nn.CrossEntropyLoss()
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_sup_riglp_c10.pt'
logger = Logger(model_name, learning_type='pruning')

# Initialing pruning method
epochs = 300
pruning_cfg = {
    'model':            cls_model,
    'optimizer':        optimizer,
    'dense_allocation': 0.2,
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
is_trained = True
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
# MAGIC ## ResNet-101
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Magnitude Prune

# COMMAND ----------

# Creating a classification net for downstream task
model_name = 'resnet101'    

cls_model = ClassificationNetwork(model_name, CIFAR10_CLASSES, False).to(get_device())
load_weights(cls_model, f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_sup_c10.pt')

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
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_sup_magp_c10.pt'
logger = Logger(model_name, learning_type='pruning')

# Initialing pruning method
pruner = MagnitudePrune(PRUNING_EPOCHS, 0.965)
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
    'recover_epochs':   10,
    'pruning_type':     pruning_type,
    'stop_epochs':      5,
}

# Set False to train
is_trained = True
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
model_name = 'resnet101'    

cls_model = ClassificationNetwork(model_name, CIFAR10_CLASSES, False).to(get_device())
load_weights(cls_model, f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_sup_c10.pt')

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
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_sup_mvmp_c10.pt'
logger = Logger(model_name, learning_type='pruning')

# Initialing pruning method
pruner = MagnitudePrune(PRUNING_EPOCHS, TARGET_SPARSITY_MID)
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
is_trained = True
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
model_name = 'resnet101'    
cls_model = ClassificationNetwork(model_name, CIFAR10_CLASSES, False).to(get_device())
load_weights(cls_model, f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_sup_c10.pt')

# Initializing Hyperparameters
optimizer_type = 'adam'
optimizer_cfg = {
    'model':        cls_model,
    'lr':           0.0005,
    'momentum':     MOMENTUM,
    'weight_decay': 5e-4,
}
optimizer = set_optimizer(optimizer_type, optimizer_cfg)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
criterion = nn.CrossEntropyLoss()
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_sup_riglp_c10.pt'
logger = Logger(model_name, learning_type='pruning')

# Initialing pruning method
epochs = 300
pruning_cfg = {
    'model':            cls_model,
    'optimizer':        optimizer,
    'dense_allocation': 0.2,
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
is_trained = True
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
# MAGIC ## ResNet-50
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Magnitude Prune

# COMMAND ----------

# Creating projection models for BaCP framework
model_name = 'resnet50'

# Projection networks
finetuned_weights = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_supcon_c10.pt'
pre_trained_model, current_model, finetuned_model = create_models_for_cap(model_name, finetuned_weights, True)
print(f"Current model sparsity: {get_model_sparsity(current_model)}")

# Initializing Hyperparameters
optimizer_type = 'adam'
optimizer_cfg = {
    'model':        current_model,
    'lr':           0.0005,
    'momentum':     MOMENTUM,
    'weight_decay': WEIGHT_DECAY
}
optimizer = set_optimizer(optimizer_type, optimizer_cfg)
scheduler = None
criterion = nn.CrossEntropyLoss()
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_bacp_magp_c10_v2'
logger = Logger(model_name, learning_type='bacp')

# Initializing pruner
pruner = MagnitudePrune(BACP_EPOCHS, TARGET_SPARSITY_MID)

config = {
    'model_name':       model_name,
    'optimizer':        optimizer,
    'scheduler':        scheduler,               
    'criterion':        criterion,
    'temperature':      0.15,
    'base_temperature': 1.0,
    'target_sparsity':  TARGET_SPARSITY_MID,   
    'logger':           logger,
    'epochs':           BACP_EPOCHS,         
    'batch_size':       BATCH_SIZE,     
    'num_classes':      CIFAR10_CLASSES, # Change this based on dataloader
    'lambdas':          [0.25, 0.25, 0.25, 0.25], # None lambas => lambdas become learnable parameters
    'save_path':        save_path,
    'pruner':           pruner,
    'recovery_epochs':  0,
}

cap_learner = BaCPLearner(current_model, pre_trained_model, finetuned_model, config)

# Set False to train
is_trained = False
if not is_trained:
    cap_learner.train(trainloader_c10_cl)

# COMMAND ----------

# Creating classification model with unfrozen parameters
model_name = 'resnet50'
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
    'stop_epochs':      25,
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
model_name = 'resnet50'

# Projection networks
finetuned_weights = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_supcon_c10.pt'
pre_trained_model, current_model, finetuned_model = create_models_for_cap(model_name, finetuned_weights, True)
print(f"Current model sparsity: {get_model_sparsity(current_model)}")

# Initializing Hyperparameters
optimizer_type = 'adam'
optimizer_cfg = {
    'model':        current_model,
    'lr':           0.0005,
    'momentum':     MOMENTUM,
    'weight_decay': WEIGHT_DECAY
}
optimizer = set_optimizer(optimizer_type, optimizer_cfg)
scheduler = None
criterion = nn.CrossEntropyLoss()
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_bacp_mvmp_c10'
logger = Logger(model_name, learning_type='bacp')

# Initializing pruner
pruner = MovementPrune(BACP_EPOCHS, TARGET_SPARSITY_MID)

config = {
    'model_name':       model_name,
    'n_views':          2,
    'optimizer':        optimizer,
    'scheduler':        scheduler,               
    'criterion':        criterion,
    'temperature':      0.1,
    'base_temperature': 1.0,
    'target_sparsity':  TARGET_SPARSITY_MID,   
    'logger':           logger,
    'epochs':           BACP_EPOCHS,         
    'batch_size':       BATCH_SIZE,     
    'num_classes':      CIFAR10_CLASSES, 
    'lambdas':          [0.25, 0.25, 0.25, 0.25],
    'save_path':        save_path,
    'pruner':           pruner,
}

cap_learner = BaCPLearner(current_model, pre_trained_model, finetuned_model, config)

# Set False to train
is_trained = True
if not is_trained:
    cap_learner.train(trainloader_c10_cl)

# COMMAND ----------

# Creating classification model with unfrozen parameters
model_name = 'resnet50'
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
    'epochs':           LINEAR_EPOCHS,
    "batch_size":       BATCH_SIZE,
    "save_path":        save_path,
    "logger":           logger,
    'lambda_reg':       0,
    'recover_epochs':   0,
    'pruning_type':     pruning_type,
    'stop_epochs':      25,
}

# Set False to train
is_trained = True
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
# MAGIC ## ResNet-101

# COMMAND ----------

# MAGIC %md
# MAGIC ### Magnitude **Prune**

# COMMAND ----------

# Creating projection models for BaCP framework
model_name = 'resnet101'

# Projection networks
finetuned_weights = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_supcon_c10.pt'
pre_trained_model, current_model, finetuned_model = create_models_for_cap(model_name, finetuned_weights, True)
print(f"Current model sparsity: {get_model_sparsity(current_model)}")

# Initializing Hyperparameters
optimizer_type = 'adam'
optimizer_cfg = {
    'model':        current_model,
    'lr':           0.0005,
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
    'temperature':      0.1,
    'base_temperature': 1.0,
    'target_sparsity':  TARGET_SPARSITY_MID,   
    'logger':           logger,
    'epochs':           BACP_EPOCHS,         
    'batch_size':       BATCH_SIZE,     
    'num_classes':      CIFAR10_CLASSES, # Change this based on dataloader
    'lambdas':          [0.25, 0.25, 0.25, 0.25], # None lambas => lambdas become learnable parameters
    'save_path':        save_path,
    'pruner':           pruner,
}

cap_learner = BaCPLearner(current_model, pre_trained_model, finetuned_model, config)

# Set False to train
is_trained = True
if not is_trained:
    cap_learner.train(trainloader_c10_cl)

# COMMAND ----------

# Creating classification model with unfrozen parameters
model_name = 'resnet101'
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
    'stop_epochs':      25,
}

# Set False to train
is_trained = True
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
model_name = 'resnet101'

# Projection networks
finetuned_weights = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_supcon_c10.pt'
pre_trained_model, current_model, finetuned_model = create_models_for_cap(model_name, finetuned_weights, True)
print(f"Current model sparsity: {get_model_sparsity(current_model)}")

# Initializing Hyperparameters
optimizer_type = 'adam'
optimizer_cfg = {
    'model':        current_model,
    'lr':           0.0005,
    'momentum':     MOMENTUM,
    'weight_decay': WEIGHT_DECAY
}
optimizer = set_optimizer(optimizer_type, optimizer_cfg)
scheduler = None
criterion = nn.CrossEntropyLoss()
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_bacp_mvmp_c10'
logger = Logger(model_name, learning_type='bacp')

# Initializing pruner
pruner = MovementPrune(BACP_EPOCHS, TARGET_SPARSITY_MID)

config = {
    'model_name':       model_name,
    'n_views':          2,
    'optimizer':        optimizer,
    'scheduler':        scheduler,               
    'criterion':        criterion,
    'temperature':      0.1,
    'base_temperature': 1.0,
    'target_sparsity':  TARGET_SPARSITY_MID,   
    'logger':           logger,
    'epochs':           BACP_EPOCHS,         
    'batch_size':       BATCH_SIZE,     
    'num_classes':      CIFAR10_CLASSES, # Change this based on dataloader
    'lambdas':          [0.25, 0.25, 0.25, 0.25], # None lambas => lambdas become learnable parameters
    'save_path':        save_path,
    'pruner':           pruner,
}

cap_learner = BaCPLearner(current_model, pre_trained_model, finetuned_model, config)

# Set False to train
is_trained = True
if not is_trained:
    cap_learner.train(trainloader_c10_cl)

# COMMAND ----------

# Creating classification model with unfrozen parameters
model_name = 'resnet101'
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
    'epochs':           LINEAR_EPOCHS,
    "batch_size":       BATCH_SIZE,
    "save_path":        save_path,
    "logger":           logger,
    'lambda_reg':       0,
    'recover_epochs':   0,
    'pruning_type':     pruning_type,
    'stop_epochs':      25,
}

# Set False to train
is_trained = True
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
