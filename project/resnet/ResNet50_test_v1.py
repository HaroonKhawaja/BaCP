# Databricks notebook source
# MAGIC %md
# MAGIC # ResNet-50 Testing Notebook

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import sys
import os
sys.path.append(os.path.abspath('..'))

import torch.nn as nn
import torch.optim as optim

# importing modules from main directory
from contrastive_learning import ContrastiveLearner
from models import EncoderProjectionNetwork, ClassificationNetwork
from datasets_class import CreateDatasets
from supervised_learning import train, test
from unstructured_pruning import MagnitudePrune, MovementPrune, LocalMagnitudePrune, LocalMovementPrune
from torch.utils.data import DataLoader
from bacp import BaCPLearner
from logger import Logger
from utils import *
from constants import *


# COMMAND ----------

# DBTITLE 0,CIFAR-10 Dataloaders
datasets = CreateDatasets('/dbfs/datasets')

# Data for supervised learning
trainset_c10_cls_fn, testset_c10_cls_fn = datasets.get_dataset_fn('supervised', DATASET)
trainset_c10_cls, testset_c10_cls = trainset_c10_cls_fn(), testset_c10_cls_fn()

trainloader_c10_cls = DataLoader(trainset_c10_cls, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
testloader_c10_cls = DataLoader(testset_c10_cls, BATCH_SIZE, shuffle=False)

# COMMAND ----------

# 2-view augmented data for supervised contrastive learning
trainset_c10_cl_fn, testset_c10_cl_fn = datasets.get_dataset_fn('supcon', DATASET)
trainset_c10_cl, testset_c10_cl = trainset_c10_cl_fn(), testset_c10_cl_fn()

trainloader_c10_cl = DataLoader(trainset_c10_cl, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
testloader_c10_cl = DataLoader(testset_c10_cl, BATCH_SIZE, shuffle=False)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Baseline Accuracies

# COMMAND ----------

# Initializing projection model for supervised-contrastive learning
model_name = 'resnet50'    
projection_model = EncoderProjectionNetwork(model_name)
make_resnet_for_cifar10(projection_model)

# Initializing hyperparameters and classes
optimizer_type = 'sgd'
optimizer_cfg = {
    'model':        projection_model,
    'lr':           0.005,
    'momentum':     MOMENTUM,
    'weight_decay': WEIGHT_DECAY
}
optimizer = set_optimizer(optimizer_type, optimizer_cfg)
save_path = f'/dbfs/{model_name}_weights/{model_name}_supcon_c10.pt'
logger = Logger(model_name, learning_type='baseline_accuracies')

config = {
    'optimizer':    optimizer,
    'epochs':       EPOCHS_RESNET50,
    'batch_size':   BATCH_SIZE,
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
cls_model.to(get_device())

# Loading the projection models backbone weights into the new classification net
load_weights(cls_model, f'/dbfs/{model_name}_weights/{model_name}_supcon_c10.pt')

# Initializing Hyperparameters
optimizer_type = 'adamw'
optimizer_cfg = {
    'model':        cls_model,
    'lr':           0.00001,
    'momentum':     MOMENTUM,
    'weight_decay': WEIGHT_DECAY_CLS,
}
save_path = f'/dbfs/{model_name}_weights/{model_name}_sup_c10.pt'
logger = Logger(model_name, learning_type='baseline_accuracies')

config = {
    'trainloader':      trainloader_c10_cls,
    'testloader':       testloader_c10_cls,
    'optimizer':        set_optimizer(optimizer_type, optimizer_cfg),
    'criterion':        nn.CrossEntropyLoss(),
    'batch_size':       BATCH_SIZE,
    'epochs':           LINEAR_EPOCHS,
    "save_path":        save_path,
    "logger":           logger,
}

# Set True if trained
is_trained = True
if not is_trained:
    losses, train_accuracies, test_accuracies = train(cls_model, config)
    graph_losses_n_accs(losses, train_accuracies, test_accuracies)
    
# Evaluating model
load_weights(cls_model, save_path)
acc = test(cls_model, testloader_c10_cls)
print(f"\nAccuracy of model is: {acc}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pruning Accuracies

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sparsity: 0.95

# COMMAND ----------

# MAGIC %md
# MAGIC #### Magnitude Prune - Movement Prune - Rigging the Lottery

# COMMAND ----------

#######################################
########## MAGNITUDE PRUNING ##########
#######################################

# Creating a classification net for downstream task
model_name = 'resnet50'    

cls_model = ClassificationNetwork(model_name, CIFAR10_CLASSES, False).to(get_device())
load_weights(cls_model, f'/dbfs/{model_name}_weights/{model_name}_sup_c10.pt')

# Initializing Hyperparameters
optimizer_type = 'adamw'
optimizer_cfg = {
    'model':        cls_model,
    'lr':           0.0001,
    'momentum':     MOMENTUM,
    'weight_decay': WEIGHT_DECAY_CLS
}
save_path = f'/dbfs/{model_name}_weights/{model_name}_magp_095.pt'
logger = Logger(model_name, learning_type='magnitude_pruning')

# Initialing pruning method
pruner = MagnitudePrune(PRUNING_EPOCHS, TARGET_SPARSITY_LOW)
pruning_type = 'magnitude_pruning'

config = {
    'trainloader':      trainloader_c10_cls,
    'testloader':       testloader_c10_cls,
    'optimizer':        set_optimizer(optimizer_type, optimizer_cfg),
    'criterion':        nn.CrossEntropyLoss(),
    'epochs':           PRUNING_EPOCHS,
    "batch_size":       BATCH_SIZE,
    "save_path":        save_path,
    "logger":           logger,
    'recover_epochs':   10,
    'pruning_type':     pruning_type,
}

# Set False to train
is_trained = False
if not is_trained:
    losses, train_accuracies, test_accuracies = train(cls_model, config, pruner)
    torch.save(cls_model.state_dict(), save_path)
    graph_losses_n_accs(losses, train_accuracies, test_accuracies)

# Evaluating model
load_weights(cls_model, save_path)
print(f"\nSparsity of pruned model: {get_model_sparsity(cls_model):.3f}")
pruned_acc = test(cls_model, testloader_c10_cls)
print(f"\nAccuracy of pruned model is: {pruned_acc}")

# COMMAND ----------

######################################
########## MOVEMENT PRUNING ##########
######################################

# Creating a classification net for downstream task
model_name = 'resnet50'    

cls_model = ClassificationNetwork(model_name, CIFAR10_CLASSES, False).to(get_device())
load_weights(cls_model, f'/dbfs/{model_name}_weights/{model_name}_sup_c10.pt')

# Initializing Hyperparameters
optimizer_type = 'adamw'
optimizer_cfg = {
    'model':        cls_model,
    'lr':           0.0001,
    'momentum':     MOMENTUM,
    'weight_decay': WEIGHT_DECAY_CLS
}
save_path = f'/dbfs/{model_name}_weights/{model_name}_movp_095.pt'
logger = Logger(model_name, learning_type='movement_pruning')

# Initialing pruning method
pruner = MovementPrune(PRUNING_EPOCHS, TARGET_SPARSITY_LOW)
pruning_type = 'movement_pruning'

config = {
    'trainloader':      trainloader_c10_cls,
    'testloader':       testloader_c10_cls,
    'optimizer':        set_optimizer(optimizer_type, optimizer_cfg),
    'criterion':        nn.CrossEntropyLoss(),
    'epochs':           PRUNING_EPOCHS,
    "batch_size":       BATCH_SIZE,
    "save_path":        save_path,
    "logger":           logger,
    'recover_epochs':   10,
    'pruning_type':     pruning_type,
}

# Set False to train
is_trained = False
if not is_trained:
    losses, train_accuracies, test_accuracies = train(cls_model, config, pruner)
    torch.save(cls_model.state_dict(), save_path)
    graph_losses_n_accs(losses, train_accuracies, test_accuracies)

# Evaluating model
load_weights(cls_model, save_path)
print(f"\nSparsity of pruned model: {get_model_sparsity(cls_model):.3f}")
pruned_acc = test(cls_model, testloader_c10_cls)
print(f"\nAccuracy of pruned model is: {pruned_acc}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sparsity: 0.97

# COMMAND ----------

# MAGIC %md
# MAGIC #### Magnitude Prune - Movement Prune - Rigging the Lottery

# COMMAND ----------

#######################################
########## MAGNITUDE PRUNING ##########
#######################################

# Creating a classification net for downstream task
model_name = 'resnet50'    

cls_model = ClassificationNetwork(model_name, CIFAR10_CLASSES, False).to(get_device())
load_weights(cls_model, f'/dbfs/{model_name}_weights/{model_name}_sup_c10.pt')

# Initializing Hyperparameters
optimizer_type = 'adamw'
optimizer_cfg = {
    'model':        cls_model,
    'lr':           0.0001,
    'momentum':     MOMENTUM,
    'weight_decay': WEIGHT_DECAY_CLS
}
save_path = f'/dbfs/{model_name}_weights/{model_name}_magp_097.pt'
logger = Logger(model_name, learning_type='magnitude_pruning')

# Initialing pruning method
pruner = MagnitudePrune(PRUNING_EPOCHS, TARGET_SPARSITY_MID)
pruning_type = 'magnitude_pruning'

config = {
    'trainloader':      trainloader_c10_cls,
    'testloader':       testloader_c10_cls,
    'optimizer':        set_optimizer(optimizer_type, optimizer_cfg),
    'criterion':        nn.CrossEntropyLoss(),
    'epochs':           PRUNING_EPOCHS,
    "batch_size":       BATCH_SIZE,
    "save_path":        save_path,
    "logger":           logger,
    'recover_epochs':   10,
    'pruning_type':     pruning_type,
}

# Set False to train
is_trained = False
if not is_trained:
    losses, train_accuracies, test_accuracies = train(cls_model, config, pruner)
    torch.save(cls_model.state_dict(), save_path)
    graph_losses_n_accs(losses, train_accuracies, test_accuracies)

# Evaluating model
load_weights(cls_model, save_path)
print(f"\nSparsity of pruned model: {get_model_sparsity(cls_model):.3f}")
pruned_acc = test(cls_model, testloader_c10_cls)
print(f"\nAccuracy of pruned model is: {pruned_acc}")

# COMMAND ----------

######################################
########## MOVEMENT PRUNING ##########
######################################

# Creating a classification net for downstream task
model_name = 'resnet50'    

cls_model = ClassificationNetwork(model_name, CIFAR10_CLASSES, False).to(get_device())
load_weights(cls_model, f'/dbfs/{model_name}_weights/{model_name}_sup_c10.pt')

# Initializing Hyperparameters
optimizer_type = 'adamw'
optimizer_cfg = {
    'model':        cls_model,
    'lr':           0.0001,
    'momentum':     MOMENTUM,
    'weight_decay': WEIGHT_DECAY_CLS
}
save_path = f'/dbfs/{model_name}_weights/{model_name}_movp_097.pt'
logger = Logger(model_name, learning_type='movement_pruning')

# Initialing pruning method
pruner = MovementPrune(PRUNING_EPOCHS, TARGET_SPARSITY_MID)
pruning_type = 'movement_pruning'

config = {
    'trainloader':      trainloader_c10_cls,
    'testloader':       testloader_c10_cls,
    'optimizer':        set_optimizer(optimizer_type, optimizer_cfg),
    'criterion':        nn.CrossEntropyLoss(),
    'epochs':           PRUNING_EPOCHS,
    "batch_size":       BATCH_SIZE,
    "save_path":        save_path,
    "logger":           logger,
    'recover_epochs':   10,
    'pruning_type':     pruning_type,
}

# Set False to train
is_trained = False
if not is_trained:
    losses, train_accuracies, test_accuracies = train(cls_model, config, pruner)
    torch.save(cls_model.state_dict(), save_path)
    graph_losses_n_accs(losses, train_accuracies, test_accuracies)

# Evaluating model
load_weights(cls_model, save_path)
print(f"\nSparsity of pruned model: {get_model_sparsity(cls_model):.3f}")
pruned_acc = test(cls_model, testloader_c10_cls)
print(f"\nAccuracy of pruned model is: {pruned_acc}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sparsity: 0.99
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #### Magnitude Prune - Movement Prune - Rigging the Lottery

# COMMAND ----------

#######################################
########## MAGNITUDE PRUNING ##########
#######################################

# Creating a classification net for downstream task
model_name = 'resnet50'    

cls_model = ClassificationNetwork(model_name, CIFAR10_CLASSES, False).to(get_device())
load_weights(cls_model, f'/dbfs/{model_name}_weights/{model_name}_sup_c10.pt')

# Initializing Hyperparameters
optimizer_type = 'adamw'
optimizer_cfg = {
    'model':        cls_model,
    'lr':           0.0001,
    'momentum':     MOMENTUM,
    'weight_decay': WEIGHT_DECAY_CLS
}
save_path = f'/dbfs/{model_name}_weights/{model_name}_magp_099.pt'
logger = Logger(model_name, learning_type='magnitude_pruning')

# Initialing pruning method
pruner = MagnitudePrune(PRUNING_EPOCHS, TARGET_SPARSITY_HIGH)
pruning_type = 'magnitude_pruning'

config = {
    'trainloader':      trainloader_c10_cls,
    'testloader':       testloader_c10_cls,
    'optimizer':        set_optimizer(optimizer_type, optimizer_cfg),
    'criterion':        nn.CrossEntropyLoss(),
    'epochs':           PRUNING_EPOCHS,
    "batch_size":       BATCH_SIZE,
    "save_path":        save_path,
    "logger":           logger,
    'recover_epochs':   10,
    'pruning_type':     pruning_type,
}

# Set False to train
is_trained = False
if not is_trained:
    losses, train_accuracies, test_accuracies = train(cls_model, config, pruner)
    torch.save(cls_model.state_dict(), save_path)
    graph_losses_n_accs(losses, train_accuracies, test_accuracies)

# Evaluating model
load_weights(cls_model, save_path)
print(f"\nSparsity of pruned model: {get_model_sparsity(cls_model):.3f}")
pruned_acc = test(cls_model, testloader_c10_cls)
print(f"\nAccuracy of pruned model is: {pruned_acc}")

# COMMAND ----------

######################################
########## MOVEMENT PRUNING ##########
######################################

# Creating a classification net for downstream task
model_name = 'resnet50'    

cls_model = ClassificationNetwork(model_name, CIFAR10_CLASSES, False).to(get_device())
load_weights(cls_model, f'/dbfs/{model_name}_weights/{model_name}_sup_c10.pt')

# Initializing Hyperparameters
optimizer_type = 'adamw'
optimizer_cfg = {
    'model':        cls_model,
    'lr':           0.0001,
    'momentum':     MOMENTUM,
    'weight_decay': WEIGHT_DECAY_CLS
}
save_path = f'/dbfs/{model_name}_weights/{model_name}_movp_099.pt'
logger = Logger(model_name, learning_type='movement_pruning')

# Initialing pruning method
pruner = MovementPrune(PRUNING_EPOCHS, TARGET_SPARSITY_HIGH)
pruning_type = 'movement_pruning'

config = {
    'trainloader':      trainloader_c10_cls,
    'testloader':       testloader_c10_cls,
    'optimizer':        set_optimizer(optimizer_type, optimizer_cfg),
    'criterion':        nn.CrossEntropyLoss(),
    'epochs':           PRUNING_EPOCHS,
    "batch_size":       BATCH_SIZE,
    "save_path":        save_path,
    "logger":           logger,
    'recover_epochs':   10,
    'pruning_type':     pruning_type,
}

# Set False to train
is_trained = False
if not is_trained:
    losses, train_accuracies, test_accuracies = train(cls_model, config, pruner)
    torch.save(cls_model.state_dict(), save_path)
    graph_losses_n_accs(losses, train_accuracies, test_accuracies)

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
# MAGIC ## Sparsity: 0.95

# COMMAND ----------

# MAGIC %md
# MAGIC ### ResNet-50
# MAGIC

# COMMAND ----------

#######################################
########## MAGNITUDE PRUNING ##########
#######################################

# Creating projection models for BaCP framework
model_name = 'resnet50'

# Projection networks
finetuned_weights = f'/dbfs/{model_name}_weights/{model_name}_supcon_c10.pt'
pre_trained_model, current_model, finetuned_model = create_models_for_bacp(model_name, finetuned_weights)
print(f"Current model sparsity: {get_model_sparsity(current_model)}")

# Initializing Hyperparameters
optimizer_type = 'adamw'
optimizer_cfg = {
    'model':        current_model,
    'lr':           0.0005,
    'momentum':     MOMENTUM,
    'weight_decay': WEIGHT_DECAY
}
save_path = f'/dbfs/{model_name}_weights/{model_name}_bacp_magp_095'
logger = Logger(model_name, learning_type='bacp_magnitude_pruning')

# Initializing pruner
pruner = MagnitudePrune(BACP_EPOCHS, TARGET_SPARSITY_LOW)

config = {
    'model_name':       model_name,
    'optimizer':        set_optimizer(optimizer_type, optimizer_cfg),
    'logger':           logger,
    'target_sparsity':  TARGET_SPARSITY_LOW,   
    'criterion':        nn.CrossEntropyLoss(),
    'batch_size':       BATCH_SIZE,     
    'pruner':           pruner,
    'save_path':        save_path,
}

bacp_learner = BaCPLearner(current_model, pre_trained_model, finetuned_model, config)

# Set False to train
is_trained = False
if not is_trained:
    bacp_learner.train(trainloader_c10_cl)

# COMMAND ----------

# Creating classification model with unfrozen parameters
model_name = 'resnet50'
cls_model = bacp_learner.create_classification_net(False)
print(f"Current model sparsity: {get_model_sparsity(cls_model)}")

# Generate masks from model
bacp_learner.generate_mask_from_model()

# Initializing Hyperparameters
optimizer_type = 'adamw'
optimizer_cfg = {
    'model':        cls_model,
    'lr':           0.0001,
    'momentum':     MOMENTUM,
    'weight_decay': WEIGHT_DECAY
}
criterion = nn.CrossEntropyLoss()
save_path = f'/dbfs/{model_name}_weights/{model_name}_bacp_magp_095_cls.pt'
logger = Logger(model_name, learning_type='bacp_magnitude_pruning')

# Initializing pruner
pruner = bacp_learner.get_pruner()
pruning_type = 'magnitude_pruning'

config = {
    'trainloader':      trainloader_c10_cls,
    'testloader':       testloader_c10_cls,
    'optimizer':        set_optimizer(optimizer_type, optimizer_cfg),
    'criterion':        nn.CrossEntropyLoss(),
    'epochs':           LINEAR_EPOCHS,
    "batch_size":       BATCH_SIZE,
    "save_path":        save_path,
    "logger":           logger,
    'pruning_type':     pruning_type,
}

# Set False to train
is_trained = False
if not is_trained:
    losses, train_accuracies, test_accuracies = train(cls_model, config, pruner, True)
    graph_losses_n_accs(losses, train_accuracies, test_accuracies)

# Evaluating model
load_weights(cls_model, save_path)
acc = test(cls_model, testloader_c10_cls)
print(f"\nAccuracy of model is: {acc}")

# COMMAND ----------

######################################
########## MOVEMENT PRUNING ##########
######################################

# Creating projection models for BaCP framework
model_name = 'resnet50'

# Projection networks
finetuned_weights = f'/dbfs/{model_name}_weights/{model_name}_supcon_c10.pt'
pre_trained_model, current_model, finetuned_model = create_models_for_bacp(model_name, finetuned_weights)
print(f"Current model sparsity: {get_model_sparsity(current_model)}")

# Initializing Hyperparameters
optimizer_type = 'adamw'
optimizer_cfg = {
    'model':        current_model,
    'lr':           0.0005,
    'momentum':     MOMENTUM,
    'weight_decay': WEIGHT_DECAY
}
save_path = f'/dbfs/{model_name}_weights/{model_name}_bacp_movp_095'
logger = Logger(model_name, learning_type='bacp_movement_pruning')

# Initializing pruner
pruner = MovementPrune(BACP_EPOCHS, TARGET_SPARSITY_LOW)

config = {
    'model_name':       model_name,
    'optimizer':        set_optimizer(optimizer_type, optimizer_cfg),
    'logger':           logger,
    'target_sparsity':  TARGET_SPARSITY_LOW,   
    'criterion':        nn.CrossEntropyLoss(),
    'batch_size':       BATCH_SIZE,     
    'pruner':           pruner,
    'save_path':        save_path,
}

bacp_learner = BaCPLearner(current_model, pre_trained_model, finetuned_model, config)

# Set False to train
is_trained = False
if not is_trained:
    bacp_learner.train(trainloader_c10_cl)

# COMMAND ----------

# Creating classification model with unfrozen parameters
model_name = 'resnet50'
cls_model = bacp_learner.create_classification_net(False)
print(f"Current model sparsity: {get_model_sparsity(cls_model)}")

# Generate masks from model
bacp_learner.generate_mask_from_model()

# Initializing Hyperparameters
optimizer_type = 'adamw'
optimizer_cfg = {
    'model':        cls_model,
    'lr':           0.0001,
    'momentum':     MOMENTUM,
    'weight_decay': WEIGHT_DECAY
}
criterion = nn.CrossEntropyLoss()
save_path = f'/dbfs/{model_name}_weights/{model_name}_bacp_movp_095_cls.pt'
logger = Logger(model_name, learning_type='bacp_movement_pruning')

# Initializing pruner
pruner = bacp_learner.get_pruner()
pruning_type = 'movement_pruning'

config = {
    'trainloader':      trainloader_c10_cls,
    'testloader':       testloader_c10_cls,
    'optimizer':        set_optimizer(optimizer_type, optimizer_cfg),
    'criterion':        nn.CrossEntropyLoss(),
    'epochs':           LINEAR_EPOCHS,
    "batch_size":       BATCH_SIZE,
    "save_path":        save_path,
    "logger":           logger,
    'pruning_type':     pruning_type,
}

# Set False to train
is_trained = False
if not is_trained:
    losses, train_accuracies, test_accuracies = train(cls_model, config, pruner, True)
    graph_losses_n_accs(losses, train_accuracies, test_accuracies)

# Evaluating model
load_weights(cls_model, save_path)
acc = test(cls_model, testloader_c10_cls)
print(f"\nAccuracy of model is: {acc}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sparsity: 0.97

# COMMAND ----------

# MAGIC %md
# MAGIC ### ResNet-50

# COMMAND ----------

#######################################
########## MAGNITUDE PRUNING ##########
#######################################

# Creating projection models for BaCP framework
model_name = 'resnet50'

# Projection networks
finetuned_weights = f'/dbfs/{model_name}_weights/{model_name}_supcon_c10.pt'
pre_trained_model, current_model, finetuned_model = create_models_for_bacp(model_name, finetuned_weights)
print(f"Current model sparsity: {get_model_sparsity(current_model)}")

# Initializing Hyperparameters
optimizer_type = 'adamw'
optimizer_cfg = {
    'model':        current_model,
    'lr':           0.0005,
    'momentum':     MOMENTUM,
    'weight_decay': WEIGHT_DECAY
}
save_path = f'/dbfs/{model_name}_weights/{model_name}_bacp_magp_097'
logger = Logger(model_name, learning_type='bacp_magnitude_pruning')

# Initializing pruner
pruner = MagnitudePrune(BACP_EPOCHS, TARGET_SPARSITY_MID)

config = {
    'model_name':       model_name,
    'optimizer':        set_optimizer(optimizer_type, optimizer_cfg),
    'logger':           logger,
    'target_sparsity':  TARGET_SPARSITY_MID,   
    'criterion':        nn.CrossEntropyLoss(),
    'batch_size':       BATCH_SIZE,     
    'pruner':           pruner,
    'save_path':        save_path,
}

bacp_learner = BaCPLearner(current_model, pre_trained_model, finetuned_model, config)

# Set False to train
is_trained = False
if not is_trained:
    bacp_learner.train(trainloader_c10_cl)

# COMMAND ----------

# Creating classification model with unfrozen parameters
model_name = 'resnet50'
cls_model = bacp_learner.create_classification_net(False)
print(f"Current model sparsity: {get_model_sparsity(cls_model)}")

# Generate masks from model
bacp_learner.generate_mask_from_model()

# Initializing Hyperparameters
optimizer_type = 'adamw'
optimizer_cfg = {
    'model':        cls_model,
    'lr':           0.0001,
    'momentum':     MOMENTUM,
    'weight_decay': WEIGHT_DECAY
}
criterion = nn.CrossEntropyLoss()
save_path = f'/dbfs/{model_name}_weights/{model_name}_bacp_magp_097_cls.pt'
logger = Logger(model_name, learning_type='bacp_magnitude_pruning')

# Initializing pruner
pruner = bacp_learner.get_pruner()
pruning_type = 'magnitude_pruning'

config = {
    'trainloader':      trainloader_c10_cls,
    'testloader':       testloader_c10_cls,
    'optimizer':        set_optimizer(optimizer_type, optimizer_cfg),
    'criterion':        nn.CrossEntropyLoss(),
    'epochs':           LINEAR_EPOCHS,
    "batch_size":       BATCH_SIZE,
    "save_path":        save_path,
    "logger":           logger,
    'pruning_type':     pruning_type,
}

# Set False to train
is_trained = False
if not is_trained:
    losses, train_accuracies, test_accuracies = train(cls_model, config, pruner, True)
    graph_losses_n_accs(losses, train_accuracies, test_accuracies)

# Evaluating model
load_weights(cls_model, save_path)
acc = test(cls_model, testloader_c10_cls)
print(f"\nAccuracy of model is: {acc}")

# COMMAND ----------

######################################
########## MOVEMENT PRUNING ##########
######################################

# Creating projection models for BaCP framework
model_name = 'resnet50'

# Projection networks
finetuned_weights = f'/dbfs/{model_name}_weights/{model_name}_supcon_c10.pt'
pre_trained_model, current_model, finetuned_model = create_models_for_bacp(model_name, finetuned_weights)
print(f"Current model sparsity: {get_model_sparsity(current_model)}")

# Initializing Hyperparameters
optimizer_type = 'adamw'
optimizer_cfg = {
    'model':        current_model,
    'lr':           0.0005,
    'momentum':     MOMENTUM,
    'weight_decay': WEIGHT_DECAY
}
save_path = f'/dbfs/{model_name}_weights/{model_name}_bacp_movp_097'
logger = Logger(model_name, learning_type='bacp_movement_pruning')

# Initializing pruner
pruner = MovementPrune(BACP_EPOCHS, TARGET_SPARSITY_MID)

config = {
    'model_name':       model_name,
    'optimizer':        set_optimizer(optimizer_type, optimizer_cfg),
    'logger':           logger,
    'target_sparsity':  TARGET_SPARSITY_MID,   
    'criterion':        nn.CrossEntropyLoss(),
    'batch_size':       BATCH_SIZE,     
    'pruner':           pruner,
    'save_path':        save_path,
}

bacp_learner = BaCPLearner(current_model, pre_trained_model, finetuned_model, config)

# Set False to train
is_trained = False
if not is_trained:
    bacp_learner.train(trainloader_c10_cl)

# COMMAND ----------

# Creating classification model with unfrozen parameters
model_name = 'resnet50'
cls_model = bacp_learner.create_classification_net(False)
print(f"Current model sparsity: {get_model_sparsity(cls_model)}")

# Generate masks from model
bacp_learner.generate_mask_from_model()

# Initializing Hyperparameters
optimizer_type = 'adamw'
optimizer_cfg = {
    'model':        cls_model,
    'lr':           0.0001,
    'momentum':     MOMENTUM,
    'weight_decay': WEIGHT_DECAY
}
criterion = nn.CrossEntropyLoss()
save_path = f'/dbfs/{model_name}_weights/{model_name}_bacp_movp_097_cls.pt'
logger = Logger(model_name, learning_type='bacp_movement_pruning')

# Initializing pruner
pruner = bacp_learner.get_pruner()
pruning_type = 'movement_pruning'

config = {
    'trainloader':      trainloader_c10_cls,
    'testloader':       testloader_c10_cls,
    'optimizer':        set_optimizer(optimizer_type, optimizer_cfg),
    'criterion':        nn.CrossEntropyLoss(),
    'epochs':           LINEAR_EPOCHS,
    "batch_size":       BATCH_SIZE,
    "save_path":        save_path,
    "logger":           logger,
    'pruning_type':     pruning_type,
}

# Set False to train
is_trained = False
if not is_trained:
    losses, train_accuracies, test_accuracies = train(cls_model, config, pruner, True)
    graph_losses_n_accs(losses, train_accuracies, test_accuracies)

# Evaluating model
load_weights(cls_model, save_path)
acc = test(cls_model, testloader_c10_cls)
print(f"\nAccuracy of model is: {acc}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sparsity: 0.99

# COMMAND ----------

# MAGIC %md
# MAGIC ### ResNet-50

# COMMAND ----------

#######################################
########## MAGNITUDE PRUNING ##########
#######################################

# Creating projection models for BaCP framework
model_name = 'resnet50'

# Projection networks
finetuned_weights = f'/dbfs/{model_name}_weights/{model_name}_supcon_c10.pt'
pre_trained_model, current_model, finetuned_model = create_models_for_bacp(model_name, finetuned_weights)
print(f"Current model sparsity: {get_model_sparsity(current_model)}")

# Initializing Hyperparameters
optimizer_type = 'adamw'
optimizer_cfg = {
    'model':        current_model,
    'lr':           0.0005,
    'momentum':     MOMENTUM,
    'weight_decay': WEIGHT_DECAY
}
save_path = f'/dbfs/{model_name}_weights/{model_name}_bacp_magp_099'
logger = Logger(model_name, learning_type='bacp_magnitude_pruning')

# Initializing pruner
pruner = MagnitudePrune(BACP_EPOCHS, TARGET_SPARSITY_HIGH)

config = {
    'model_name':       model_name,
    'optimizer':        set_optimizer(optimizer_type, optimizer_cfg),
    'logger':           logger,
    'target_sparsity':  TARGET_SPARSITY_HIGH,   
    'criterion':        nn.CrossEntropyLoss(),
    'batch_size':       BATCH_SIZE,     
    'pruner':           pruner,
    'save_path':        save_path,
}

bacp_learner = BaCPLearner(current_model, pre_trained_model, finetuned_model, config)

# Set False to train
is_trained = False
if not is_trained:
    bacp_learner.train(trainloader_c10_cl)

# COMMAND ----------

# Creating classification model with unfrozen parameters
model_name = 'resnet50'
cls_model = bacp_learner.create_classification_net(False)
print(f"Current model sparsity: {get_model_sparsity(cls_model)}")

# Generate masks from model
bacp_learner.generate_mask_from_model()

# Initializing Hyperparameters
optimizer_type = 'adamw'
optimizer_cfg = {
    'model':        cls_model,
    'lr':           0.0001,
    'momentum':     MOMENTUM,
    'weight_decay': WEIGHT_DECAY
}
criterion = nn.CrossEntropyLoss()
save_path = f'/dbfs/{model_name}_weights/{model_name}_bacp_magp_099_cls.pt'
logger = Logger(model_name, learning_type='bacp_magnitude_pruning')

# Initializing pruner
pruner = bacp_learner.get_pruner()
pruning_type = 'magnitude_pruning'

config = {
    'trainloader':      trainloader_c10_cls,
    'testloader':       testloader_c10_cls,
    'optimizer':        set_optimizer(optimizer_type, optimizer_cfg),
    'criterion':        nn.CrossEntropyLoss(),
    'epochs':           LINEAR_EPOCHS,
    "batch_size":       BATCH_SIZE,
    "save_path":        save_path,
    "logger":           logger,
    'pruning_type':     pruning_type,
}

# Set False to train
is_trained = False
if not is_trained:
    losses, train_accuracies, test_accuracies = train(cls_model, config, pruner, True)
    graph_losses_n_accs(losses, train_accuracies, test_accuracies)

# Evaluating model
load_weights(cls_model, save_path)
acc = test(cls_model, testloader_c10_cls)
print(f"\nAccuracy of model is: {acc}")

# COMMAND ----------

######################################
########## MOVEMENT PRUNING ##########
######################################

# Creating projection models for BaCP framework
model_name = 'resnet50'

# Projection networks
finetuned_weights = f'/dbfs/{model_name}_weights/{model_name}_supcon_c10.pt'
pre_trained_model, current_model, finetuned_model = create_models_for_bacp(model_name, finetuned_weights)
print(f"Current model sparsity: {get_model_sparsity(current_model)}")

# Initializing Hyperparameters
optimizer_type = 'adamw'
optimizer_cfg = {
    'model':        current_model,
    'lr':           0.0005,
    'momentum':     MOMENTUM,
    'weight_decay': WEIGHT_DECAY
}
save_path = f'/dbfs/{model_name}_weights/{model_name}_bacp_movp_099'
logger = Logger(model_name, learning_type='bacp_movement_pruning')

# Initializing pruner
pruner = MovementPrune(BACP_EPOCHS, TARGET_SPARSITY_HIGH)

config = {
    'model_name':       model_name,
    'optimizer':        set_optimizer(optimizer_type, optimizer_cfg),
    'logger':           logger,
    'target_sparsity':  TARGET_SPARSITY_HIGH,   
    'criterion':        nn.CrossEntropyLoss(),
    'batch_size':       BATCH_SIZE,     
    'pruner':           pruner,
    'save_path':        save_path,
}

bacp_learner = BaCPLearner(current_model, pre_trained_model, finetuned_model, config)

# Set False to train
is_trained = False
if not is_trained:
    bacp_learner.train(trainloader_c10_cl)

# COMMAND ----------

# Creating classification model with unfrozen parameters
model_name = 'resnet50'
cls_model = bacp_learner.create_classification_net(False)
print(f"Current model sparsity: {get_model_sparsity(cls_model)}")

# Generate masks from model
bacp_learner.generate_mask_from_model()

# Initializing Hyperparameters
optimizer_type = 'adamw'
optimizer_cfg = {
    'model':        cls_model,
    'lr':           0.0001,
    'momentum':     MOMENTUM,
    'weight_decay': WEIGHT_DECAY
}
criterion = nn.CrossEntropyLoss()
save_path = f'/dbfs/{model_name}_weights/{model_name}_bacp_movp_099_cls.pt'
logger = Logger(model_name, learning_type='bacp_movement_pruning')

# Initializing pruner
pruner = bacp_learner.get_pruner()
pruning_type = 'movement_pruning'

config = {
    'trainloader':      trainloader_c10_cls,
    'testloader':       testloader_c10_cls,
    'optimizer':        set_optimizer(optimizer_type, optimizer_cfg),
    'criterion':        nn.CrossEntropyLoss(),
    'epochs':           LINEAR_EPOCHS,
    "batch_size":       BATCH_SIZE,
    "save_path":        save_path,
    "logger":           logger,
    'pruning_type':     pruning_type,
}

# Set False to train
is_trained = False
if not is_trained:
    losses, train_accuracies, test_accuracies = train(cls_model, config, pruner, True)
    graph_losses_n_accs(losses, train_accuracies, test_accuracies)

# Evaluating model
load_weights(cls_model, save_path)
acc = test(cls_model, testloader_c10_cls)
print(f"\nAccuracy of model is: {acc}")
