# Databricks notebook source
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import torch
import numpy as np
import matplotlib.pyplot as plt
import os

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from cap import CAPLearner, create_models_for_cap, generate_mask_from_model
from contrastive_learning import ContrastiveLearner
from supervised_learning import train, test
from models import EncoderProjectionNet, CustomClassificationNet
from datasets_class import CreateDatasets
from logger import Logger
from unstructured_pruning import MovementPrune, MagnitudePrune, WandaPrune
from pruning_utils import *
from utils import *

# COMMAND ----------

# For dataloaders
BATCH_SIZE = 1024

# Epochs
EPOCHS = 100            # for supervised-contrastive learning
FINETUNE_EPOCHS = 50    # for fine-tuning on downstream tasks
PRUNING_EPOCHS = 5      #
RECOVER_EPOCHS = 5
BACP_EPOCHS = 5

CIFAR10_CLASSES = 10
TARGET_SPARSITY = 0.98
TEMPERATURE = 0.1

NUM_WORKERS = 24


# COMMAND ----------

datasets = CreateDatasets('/dbfs/datasets')

# COMMAND ----------

# DBTITLE 1,CIFAR-10/100 Sup Dataloaders
# Data for supervised learning
trainset_c10_cls_fn, testset_c10_cls_fn = datasets.get_dataset_fn('supervised', 'cifar10')
trainset_c10_cls, testset_c10_cls = trainset_c10_cls_fn(), testset_c10_cls_fn()

trainloader_c10_cls = DataLoader(trainset_c10_cls, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
testloader_c10_cls = DataLoader(testset_c10_cls, BATCH_SIZE, shuffle=False)

# 2-view augmented data for supervised contrastive learning+
trainset_c10_cl_fn, testset_c10_cl_fn = datasets.get_dataset_fn('supcon', 'cifar10')
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

# MAGIC %md
# MAGIC #### CIFAR-10

# COMMAND ----------

# Initializing projection model for supervised-contrastive learning
model_name = 'vgg11'    
projection_model = EncoderProjectionNet(model_name, 128)

# Initializing hyperparameters and classes
optimizer = optim.Adam(projection_model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

save_path = f'/dbfs/test2/{model_name}_weights/{model_name}_supcon_c10.pt'
config = {
    'n_views':          2,
    'optimizer':        optimizer,
    'epochs':           EPOCHS,
    'scheduler':        scheduler,
    'batch_size':       BATCH_SIZE,
    'temperature':      TEMPERATURE,
    'base_temperature': 1.0,
    'loss_type':        'supcon_l',
}

supcon_learner = ContrastiveLearner(projection_model, config)
logger = Logger(model_name, learning_type='supcon_l')

# Set if trained
is_trained = True
if not is_trained:
    supcon_learner.train(trainloader_c10_cl, save_path, logger)


# COMMAND ----------

# Creating a classification net for downstream task
model_name = 'vgg11'    
cls_model = CustomClassificationNet(model_name, CIFAR10_CLASSES, False).to(get_device())

# Loading the projection models backbone weights into the new classification net
load_projection_model_weights(cls_model, f'/dbfs/test2/{model_name}_weights/{model_name}_supcon_c10.pt')

# Initializing Hyperparameters
optimizer = optim.Adam(cls_model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

criterion = nn.CrossEntropyLoss()
logger = Logger(model_name, learning_type='sup_l')
save_path = f'/dbfs/test2/{model_name}_weights/{model_name}_sup_c10.pt'
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
    "lambda_reg":       0,
    'recover_epochs':   RECOVER_EPOCHS,

}

# Set True if trained
is_trained = True
if not is_trained:
    losses, train_accuracies, test_accuracies = train(cls_model,
                                                      config)
    
# Evaluating model
load_weights(cls_model, save_path)
acc = test(cls_model, testloader_c10_cls)
print(f"Accuracy of model is: {acc}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## VGG-19

# COMMAND ----------

# MAGIC %md
# MAGIC #### CIFAR-10

# COMMAND ----------

# DBTITLE 1,VGG 19 SUPERVISED CONTRASTIVE LEARNING ON CIFAR 10
# Initializing projection model for supervised-contrastive learning
model_name = 'vgg19'    
projection_model = EncoderProjectionNet(model_name, 128)

# Initializing hyperparameters and classes
optimizer = optim.Adam(projection_model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

save_path = f'/dbfs/test2/{model_name}_weights/{model_name}_supcon_c10.pt'
config = {
    'n_views':          2,
    'optimizer':        optimizer,
    'epochs':           EPOCHS,
    'scheduler':        scheduler,
    'batch_size':       BATCH_SIZE,
    'temperature':      TEMPERATURE,
    'base_temperature': 1.0,
    'loss_type':        'supcon_l',
}

supcon_learner = ContrastiveLearner(projection_model, config)
logger = Logger(model_name, learning_type='supcon_l')

# Set if trained
is_trained = True
if not is_trained:
    supcon_learner.train(trainloader_c10_cl, save_path, logger)


# COMMAND ----------

# Creating a classification net for downstream task
model_name = 'vgg19'    
cls_model = CustomClassificationNet(model_name, CIFAR10_CLASSES, False).to(get_device())

# Loading the projection models backbone weights into the new classification net
load_projection_model_weights(cls_model, f'/dbfs/test2/{model_name}_weights/{model_name}_supcon_c10.pt')

# Initializing Hyperparameters
optimizer = optim.Adam(cls_model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

criterion = nn.CrossEntropyLoss()
logger = Logger(model_name, learning_type='sup_l')
save_path = f'/dbfs/test2/{model_name}_weights/{model_name}_sup_c10.pt'

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
    "lambda_reg":       0,
    'recover_epochs':   RECOVER_EPOCHS,
}

# Set True if trained
is_trained = True
if not is_trained:
    losses, train_accuracies, test_accuracies = train(cls_model,
                                                      config)
    
# Evaluating model
load_weights(cls_model, save_path)
acc = test(cls_model, testloader_c10_cls)
print(f"Accuracy of model is: {acc}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Pruning Accuracies

# COMMAND ----------

# MAGIC %md
# MAGIC ## VGG-11

# COMMAND ----------

# MAGIC %md
# MAGIC ### CIFAR-10

# COMMAND ----------

# MAGIC %md
# MAGIC #### Magnitude Prune

# COMMAND ----------

# Creating classification model
model_name = 'vgg11'
cls_model = CustomClassificationNet(model_name, CIFAR10_CLASSES, False).to(get_device())
load_weights(cls_model, f'/dbfs/test2/{model_name}_weights/{model_name}_sup_c10.pt')

# Initializing Hyperparameters
optimizer = optim.Adam(cls_model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

criterion = nn.CrossEntropyLoss()
logger = Logger(model_name, learning_type='sup_l')
save_path = f'/dbfs/test2/{model_name}_weights/{model_name}_sup_magp_c10.pt'

# Initialing pruning method
pruner = MagnitudePrune(PRUNING_EPOCHS, TARGET_SPARSITY)

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
    'lambda_reg':       0,
    'recover_epochs':   RECOVER_EPOCHS,
}

# Set False to train
is_trained = False
if not is_trained:
    losses, train_accuracies, test_accuracies = train(cls_model,
                                                      config,
                                                      pruner)
    torch.save(cls_model.state_dict(), save_path)
    
    # Displaying loss-acc graph
    graph_losses_n_accs(losses, 
                        train_accuracies, 
                        test_accuracies)

# Evaluating model
load_weights(cls_model, save_path)
print(f"Sparsity of pruned model: {get_model_sparsity(cls_model):.3f}")
pruned_acc = test(cls_model, testloader_c10_cls)
print(f"Accuracy of pruned model is: {pruned_acc}")


# COMMAND ----------

# MAGIC %md
# MAGIC #### Movement Prune

# COMMAND ----------

# Creating classification model
model_name = 'vgg11'
cls_model = CustomClassificationNet(model_name, CIFAR10_CLASSES, False).to(get_device())
load_weights(cls_model, f'/dbfs/test2/{model_name}_weights/{model_name}_sup_c10.pt')

# Initializing Hyperparameters
optimizer = optim.Adam(cls_model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

criterion = nn.CrossEntropyLoss()
logger = Logger(model_name, learning_type='sup_l')
save_path = f'/dbfs/test2/{model_name}_weights/{model_name}_sup_mvmp_c10.pt'

# Initialing pruning method
pruner = MovementPrune(PRUNING_EPOCHS, TARGET_SPARSITY)

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
    'lambda_reg':       0,
    'recover_epochs':   RECOVER_EPOCHS,
}

# Set False to train
is_trained = False
if not is_trained:
    losses, train_accuracies, test_accuracies = train(cls_model,
                                                      config,
                                                      pruner)
    torch.save(cls_model.state_dict(), save_path)
    
    # Displaying loss-acc graph
    graph_losses_n_accs(losses, 
                        train_accuracies, 
                        test_accuracies)

# Evaluating model
load_weights(cls_model, save_path)
print(f"Sparsity of pruned model: {get_model_sparsity(cls_model):.3f}")
pruned_acc = test(cls_model, testloader_c10_cls)
print(f"Accuracy of pruned model is: {pruned_acc}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## VGG-19

# COMMAND ----------

# MAGIC %md
# MAGIC ### CIFAR-10

# COMMAND ----------

# MAGIC %md
# MAGIC #### Magnitude Prune

# COMMAND ----------

# Creating classification model
model_name = 'vgg19'
cls_model = CustomClassificationNet(model_name, CIFAR10_CLASSES, False).to(get_device())
load_weights(cls_model, f'/dbfs/test2/{model_name}_weights/{model_name}_sup_c10.pt')

# Initializing Hyperparameters
optimizer = optim.Adam(cls_model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

criterion = nn.CrossEntropyLoss()
logger = Logger(model_name, learning_type='sup_l')
save_path = f'/dbfs/test2/{model_name}_weights/{model_name}_sup_magp_c10.pt'

# Initialing pruning method
pruner = MagnitudePrune(PRUNING_EPOCHS, TARGET_SPARSITY)

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
    'lambda_reg':       0,
    'recover_epochs':   RECOVER_EPOCHS,
}

# Set False to train
is_trained = False
if not is_trained:
    losses, train_accuracies, test_accuracies = train(cls_model,
                                                      config,
                                                      pruner)
    torch.save(cls_model.state_dict(), save_path)
    
    # Displaying loss-acc graph
    graph_losses_n_accs(losses, 
                        train_accuracies, 
                        test_accuracies)

# Evaluating model
load_weights(cls_model, save_path)
print(f"Sparsity of pruned model: {get_model_sparsity(cls_model):.3f}")
pruned_acc = test(cls_model, testloader_c10_cls)
print(f"Accuracy of pruned model is: {pruned_acc}")


# COMMAND ----------

# MAGIC %md
# MAGIC #### Movement Prune

# COMMAND ----------

# Creating classification model
model_name = 'vgg19'
cls_model = CustomClassificationNet(model_name, CIFAR10_CLASSES, False).to(get_device())
load_weights(cls_model, f'/dbfs/test2/{model_name}_weights/{model_name}_sup_c10.pt')

# Initializing Hyperparameters
optimizer = optim.Adam(cls_model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

criterion = nn.CrossEntropyLoss()
logger = Logger(model_name, learning_type='sup_l')
save_path = f'/dbfs/test2/{model_name}_weights/{model_name}_sup_mvmp_c10.pt'

# Initialing pruning method
pruner = MovementPrune(PRUNING_EPOCHS, TARGET_SPARSITY)

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
    'lambda_reg':       0,
    'recover_epochs':   RECOVER_EPOCHS,
}

# Set False to train
is_trained = False
if not is_trained:
    losses, train_accuracies, test_accuracies = train(cls_model,
                                                      config,
                                                      pruner)
    torch.save(cls_model.state_dict(), save_path)
    
    # Displaying loss-acc graph
    graph_losses_n_accs(losses, 
                        train_accuracies, 
                        test_accuracies)

# Evaluating model
load_weights(cls_model, save_path)
print(f"Sparsity of pruned model: {get_model_sparsity(cls_model):.3f}")
pruned_acc = test(cls_model, testloader_c10_cls)
print(f"Accuracy of pruned model is: {pruned_acc}")


# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # BaCP Accuracies

# COMMAND ----------

# MAGIC %md
# MAGIC ## VGG-11

# COMMAND ----------

# MAGIC %md
# MAGIC ### CIFAR-10

# COMMAND ----------

# MAGIC %md
# MAGIC #### Magnitude Prune

# COMMAND ----------

# Creating projection models for BaCP framework
model_name = 'vgg11'
finetuned_weights = f'/dbfs/test2/{model_name}_weights/{model_name}_supcon_c10.pt'
pre_trained_model, current_model, finetuned_model = create_models_for_cap(model_name, finetuned_weights)
print(f"Current model sparsity: {get_model_sparsity(current_model)}")

# Hyper-parameters
optimizer = optim.Adam(current_model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

criterion = nn.CrossEntropyLoss()
base_temperature = 1.0
logger = Logger(model_name, 'cap_l')
save_path = f'/dbfs/test2/{model_name}_weights/{model_name}_bacp_magp_c10'

# Initializing pruner
pruner = MagnitudePrune(BACP_EPOCHS, TARGET_SPARSITY)

config = {
    'model_name':       model_name,
    'n_views':          2,
    'optimizer':        optimizer,
    'scheduler':        scheduler,               
    'criterion':        criterion,
    'temperature':      TEMPERATURE,
    'base_temperature': base_temperature,
    'target_sparsity':  TARGET_SPARSITY,   
    'logger':           logger,
    'epochs':           BACP_EPOCHS,         
    'batch_size':       BATCH_SIZE,     
    'num_classes':      CIFAR10_CLASSES,    # Change this based on dataloader
    'lambdas':          0.25,               # None lambas => lambdas become learnable parameters
    'save_path':        save_path,
    'pruner':           pruner,
    'finetuning_epochs':0
}

cap_learner = CAPLearner(current_model, pre_trained_model, finetuned_model, config)

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
optimizer = optim.Adam(cls_model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

criterion = nn.CrossEntropyLoss()
logger = Logger(model_name, learning_type='sup_l')
save_path = f'/dbfs/test2/{model_name}_weights/{model_name}_bacp_magp_cls_c10.pt'

# Initializing pruner
pruner = cap_learner.get_pruner()

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
    'recover_epochs':   RECOVER_EPOCHS,
}

# Set False to train
is_trained = False
if not is_trained:
    losses, train_accuracies, test_accuracies = train(cls_model,
                                                      config,
                                                      pruner,
                                                      True)
    # Displaying loss-acc graph
    graph_losses_n_accs(losses, 
                        train_accuracies, 
                        test_accuracies)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Movement Prune

# COMMAND ----------

# Creating projection models for BaCP framework
model_name = 'vgg11'
finetuned_weights = f'/dbfs/test2/{model_name}_weights/{model_name}_supcon_c10.pt'
pre_trained_model, current_model, finetuned_model = create_models_for_cap(model_name, finetuned_weights)
print(f"Current model sparsity: {get_model_sparsity(current_model)}")

# Hyper-parameters
optimizer = optim.Adam(current_model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

criterion = nn.CrossEntropyLoss()
base_temperature = 1.0
logger = Logger(model_name, 'cap_l')
save_path = f'/dbfs/test2/{model_name}_weights/{model_name}_bacp_mvmp_c10'

# Initializing pruner
pruner = MovementPrune(BACP_EPOCHS, TARGET_SPARSITY)

config = {
    'model_name':       model_name,
    'n_views':          2,
    'optimizer':        optimizer,
    'scheduler':        scheduler,               
    'criterion':        criterion,
    'temperature':      TEMPERATURE,
    'base_temperature': base_temperature,
    'target_sparsity':  TARGET_SPARSITY,   
    'logger':           logger,
    'epochs':           BACP_EPOCHS,         
    'batch_size':       BATCH_SIZE,     
    'num_classes':      CIFAR10_CLASSES,    # Change this based on dataloader
    'lambdas':          0.25,               # None lambas => lambdas become learnable parameters
    'save_path':        save_path,
    'pruner':           pruner,
    'finetuning_epochs':0
}

cap_learner = CAPLearner(current_model, pre_trained_model, finetuned_model, config)

# Set False to train
is_trained = False
if not is_trained:
    cap_learner.cap_train(trainloader_c10_cl)

# COMMAND ----------

# Creating classification model with unfrozen parameters
model_name = 'vgg11'
cls_model = cap_learner.create_classification_net(False)

# Generate masks from model
cap_learner.generate_mask_from_model()

# Initializing Hyperparameters
optimizer = optim.Adam(cls_model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

criterion = nn.CrossEntropyLoss()
logger = Logger(model_name, learning_type='sup_l')
save_path = f'/dbfs/test2/{model_name}_weights/{model_name}_bacp_mvmp_cls_c10.pt'

# Initializing pruner
pruner = cap_learner.get_pruner()

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
    'recover_epochs':   RECOVER_EPOCHS,
}

# Set False to train
is_trained = False
if not is_trained:
    losses, train_accuracies, test_accuracies = train(cls_model,
                                                      config,
                                                      pruner,
                                                      True)
    # Displaying loss-acc graph
    graph_losses_n_accs(losses, 
                        train_accuracies, 
                        test_accuracies)

# COMMAND ----------

# MAGIC %md
# MAGIC ## VGG-19

# COMMAND ----------

# MAGIC %md
# MAGIC ### CIFAR-10

# COMMAND ----------

# MAGIC %md
# MAGIC #### Magnitude **Prune**

# COMMAND ----------

# Creating projection models for BaCP framework
model_name = 'vgg19'
finetuned_weights = f'/dbfs/test2/{model_name}_weights/{model_name}_supcon_c10.pt'
pre_trained_model, current_model, finetuned_model = create_models_for_cap(model_name, finetuned_weights)
print(f"Current model sparsity: {get_model_sparsity(current_model)}")

# Hyper-parameters
optimizer = optim.Adam(current_model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

criterion = nn.CrossEntropyLoss()
base_temperature = 1.0
logger = Logger(model_name, 'cap_l')
save_path = f'/dbfs/test2/{model_name}_weights/{model_name}_bacp_magp_c10'

# Initializing pruner
pruner = MagnitudePrune(BACP_EPOCHS, TARGET_SPARSITY)

config = {
    'model_name':       model_name,
    'n_views':          2,
    'optimizer':        optimizer,
    'scheduler':        scheduler,               
    'criterion':        criterion,
    'temperature':      TEMPERATURE,
    'base_temperature': base_temperature,
    'target_sparsity':  TARGET_SPARSITY,   
    'logger':           logger,
    'epochs':           BACP_EPOCHS,         
    'batch_size':       BATCH_SIZE,     
    'num_classes':      CIFAR10_CLASSES,    # Change this based on dataloader
    'lambdas':          0.25,               # None lambas => lambdas become learnable parameters
    'save_path':        save_path,
    'pruner':           pruner,
    'finetuning_epochs':0
}

cap_learner = CAPLearner(current_model, pre_trained_model, finetuned_model, config)

# Set False to train
is_trained = False
if not is_trained:
    cap_learner.cap_train(trainloader_c10_cl)

# COMMAND ----------

# Creating classification model with unfrozen parameters
model_name = 'vgg19'
cls_model = cap_learner.create_classification_net(False)

# Generate masks from model
cap_learner.generate_mask_from_model()

# Initializing Hyperparameters
optimizer = optim.Adam(cls_model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

criterion = nn.CrossEntropyLoss()
logger = Logger(model_name, learning_type='sup_l')
save_path = f'/dbfs/test2/{model_name}_weights/{model_name}_bacp_magp_cls_c10.pt'

# Initializing pruner
pruner = cap_learner.get_pruner()

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
    'recover_epochs':   RECOVER_EPOCHS,
}

# Set False to train
is_trained = False
if not is_trained:
    losses, train_accuracies, test_accuracies = train(cls_model,
                                                      config,
                                                      pruner,
                                                      True)
    # Displaying loss-acc graph
    graph_losses_n_accs(losses, 
                        train_accuracies, 
                        test_accuracies)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Movement Prune

# COMMAND ----------

# Creating projection models for BaCP framework
model_name = 'vgg19'
finetuned_weights = f'/dbfs/test2/{model_name}_weights/{model_name}_supcon_c10.pt'
pre_trained_model, current_model, finetuned_model = create_models_for_cap(model_name, finetuned_weights)
print(f"Current model sparsity: {get_model_sparsity(current_model)}")

# Hyper-parameters
optimizer = optim.Adam(current_model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

criterion = nn.CrossEntropyLoss()
base_temperature = 1.0
logger = Logger(model_name, 'cap_l')
save_path = f'/dbfs/test2/{model_name}_weights/{model_name}_bacp_mvmp_c10'

# Initializing pruner
pruner = MovementPrune(BACP_EPOCHS, TARGET_SPARSITY)

config = {
    'model_name':       model_name,
    'n_views':          2,
    'optimizer':        optimizer,
    'scheduler':        scheduler,               
    'criterion':        criterion,
    'temperature':      TEMPERATURE,
    'base_temperature': base_temperature,
    'target_sparsity':  TARGET_SPARSITY,   
    'logger':           logger,
    'epochs':           BACP_EPOCHS,         
    'batch_size':       BATCH_SIZE,     
    'num_classes':      CIFAR10_CLASSES,    # Change this based on dataloader
    'lambdas':          0.25,               # None lambas => lambdas become learnable parameters
    'save_path':        save_path,
    'pruner':           pruner,
    'finetuning_epochs':0
}

cap_learner = CAPLearner(current_model, pre_trained_model, finetuned_model, config)

# Set False to train
is_trained = False
if not is_trained:
    cap_learner.cap_train(trainloader_c10_cl)

# COMMAND ----------

# Creating classification model with unfrozen parameters
model_name = 'vgg19'
cls_model = cap_learner.create_classification_net(False)

# Generate masks from model
cap_learner.generate_mask_from_model()

# Hyper-parameters
optimizer = optim.Adam(cls_model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

criterion = nn.CrossEntropyLoss()
logger = Logger(model_name, learning_type='sup_l')
save_path = f'/dbfs/test2/{model_name}_weights/{model_name}_bacp_mvmp_cls_c10.pt'

pruner = cap_learner.get_pruner()

config = {
    'trainloader':  trainloader_c10_cls,
    'testloader':   testloader_c10_cls,
    'optimizer':    optimizer,
    'scheduler':    scheduler,
    'criterion':    nn.CrossEntropyLoss(),
    'epochs':       FINETUNE_EPOCHS,
    "batch_size":   BATCH_SIZE,
    "save_path":    save_path,
    "logger":       logger,
    'lambda_reg':   0,
    'recover_epochs':   RECOVER_EPOCHS,
}

# Set False to train
is_trained = False
if not is_trained:
    losses, train_accuracies, test_accuracies = train(cls_model,
                                                      config,
                                                      pruner,
                                                      True)
    # Displaying loss-acc graph
    graph_losses_n_accs(losses, 
                        train_accuracies, 
                        test_accuracies)
