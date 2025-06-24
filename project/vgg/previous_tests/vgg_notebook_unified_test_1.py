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
model_name = 'vgg11'    
projection_model = EncoderProjectionNetwork(model_name, 128)

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
    'optimizer':        optimizer,
    'epochs':           EPOCHS_RESNET50,
    'scheduler':        scheduler,
    'batch_size':       BATCH_SIZE,
    'temperature':      TEMP,
    'base_temperature': BASE_TEMP,
}

supcon_learner = ContrastiveLearner(projection_model, config)

# Set if trained
is_trained = True
if not is_trained:
    supcon_learner.train(trainloader_c10_cl, save_path, logger)


# COMMAND ----------

# Creating a classification net for downstream task
model_name = 'vgg11'    
cls_model = ClassificationNetwork(model_name, CIFAR10_CLASSES, False).to(get_device())
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
model_name = 'vgg19'    
projection_model = EncoderProjectionNetwork(model_name, 128)
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
    'optimizer':        optimizer,
    'epochs':           EPOCHS_RESNET101,
    'scheduler':        scheduler,
    'batch_size':       BATCH_SIZE,
    'temperature':      TEMP,
    'base_temperature': BASE_TEMP,
}

supcon_learner = ContrastiveLearner(projection_model, config)

# Set if trained
is_trained = True
if not is_trained:
    supcon_learner.train(trainloader_c10_cl, save_path, logger)


# COMMAND ----------

# Creating a classification net for downstream task
model_name = 'vgg19'    
cls_model = ClassificationNetwork(model_name, CIFAR10_CLASSES, False).to(get_device())
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
# MAGIC ## Sparsity: 0.95

# COMMAND ----------

# MAGIC %md
# MAGIC ### VGG-11

# COMMAND ----------

# MAGIC %md
# MAGIC #### Magnitude Prune - Movement Prune - Rigging the Lottery

# COMMAND ----------

#######################################
########## MAGNITUDE PRUNING ##########
#######################################

# Creating a classification net for downstream task
model_name = 'vgg11'    

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
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_magp_095.pt'
logger = Logger(model_name, learning_type='pruning')

# Initialing pruning method
pruner = MagnitudePrune(PRUNING_EPOCHS, TARGET_SPARSITY_LOW)
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
    'recover_epochs':   10,
    'pruning_type':     pruning_type,
    'stop_epochs':      5,
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

######################################
########## MOVEMENT PRUNING ##########
######################################

# Creating a classification net for downstream task
model_name = 'vgg11'    

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
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_mvmp_095.pt'
logger = Logger(model_name, learning_type='pruning')

# Initialing pruning method
pruner = MovementPrune(PRUNING_EPOCHS, TARGET_SPARSITY_LOW)
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
    'recover_epochs':   10,
    'pruning_type':     pruning_type,
    'stop_epochs':      5,
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

# Creating a classification net for downstream task
model_name = 'vgg11'    
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
# MAGIC ### VGG-19
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #### Magnitude Pruning - Movement Pruning - Rigging the Lottery

# COMMAND ----------

#######################################
########## MAGNITUDE PRUNING ##########
#######################################

# Creating a classification net for downstream task
model_name = 'vgg19'    

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
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_magp_095.pt'
logger = Logger(model_name, learning_type='pruning')

# Initialing pruning method
pruner = MagnitudePrune(PRUNING_EPOCHS, TARGET_SPARSITY_LOW)
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
    'recover_epochs':   10,
    'pruning_type':     pruning_type,
    'stop_epochs':      5,
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

######################################
########## MOVEMENT PRUNING ##########
######################################

# Creating a classification net for downstream task
model_name = 'vgg19'    

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
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_mvmp_095.pt'
logger = Logger(model_name, learning_type='pruning')

# Initialing pruning method
pruner = MovementPrune(PRUNING_EPOCHS, TARGET_SPARSITY_LOW)
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
    'recover_epochs':   10,
    'pruning_type':     pruning_type,
    'stop_epochs':      5,
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

# Creating a classification net for downstream task
model_name = 'vgg19'    
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
# MAGIC ## Sparsity: 0.97

# COMMAND ----------

# MAGIC %md
# MAGIC ### VGG-11

# COMMAND ----------

# MAGIC %md
# MAGIC #### Magnitude Prune - Movement Prune - Rigging the Lottery

# COMMAND ----------

#######################################
########## MAGNITUDE PRUNING ##########
#######################################

# Creating a classification net for downstream task
model_name = 'vgg11'    

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
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_magp_097.pt'
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
    'recover_epochs':   10,
    'pruning_type':     pruning_type,
    'stop_epochs':      5,
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

######################################
########## MOVEMENT PRUNING ##########
######################################

# Creating a classification net for downstream task
model_name = 'vgg11'    

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
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_mvmp_097.pt'
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
    'recover_epochs':   10,
    'pruning_type':     pruning_type,
    'stop_epochs':      5,
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
# MAGIC ### VGG-19

# COMMAND ----------

# MAGIC %md
# MAGIC #### Magnitude Prune - Movement Prune - Rigging the Lottery

# COMMAND ----------

#######################################
########## MAGNITUDE PRUNING ##########
#######################################

# Creating a classification net for downstream task
model_name = 'vgg19'    

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
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_magp_097.pt'
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
    'recover_epochs':   10,
    'pruning_type':     pruning_type,
    'stop_epochs':      5,
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

######################################
########## MOVEMENT PRUNING ##########
######################################

# Creating a classification net for downstream task
model_name = 'vgg19'    

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
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_mvmp_097.pt'
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
    'recover_epochs':   10,
    'pruning_type':     pruning_type,
    'stop_epochs':      5,
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
# MAGIC ## Sparsity: 0.99
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### VGG-11

# COMMAND ----------

# MAGIC %md
# MAGIC #### Magnitude Prune - Movement Prune - Rigging the Lottery

# COMMAND ----------

#######################################
########## MAGNITUDE PRUNING ##########
#######################################

# Creating a classification net for downstream task
model_name = 'vgg11'    

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
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_magp_099.pt'
logger = Logger(model_name, learning_type='pruning')

# Initialing pruning method
pruner = MagnitudePrune(PRUNING_EPOCHS, TARGET_SPARSITY_HIGH)
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
    'recover_epochs':   10,
    'pruning_type':     pruning_type,
    'stop_epochs':      5,
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

######################################
########## MOVEMENT PRUNING ##########
######################################

# Creating a classification net for downstream task
model_name = 'vgg11'    

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
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_mvmp_099.pt'
logger = Logger(model_name, learning_type='pruning')

# Initialing pruning method
pruner = MovementPrune(PRUNING_EPOCHS, TARGET_SPARSITY_HIGH)
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
    'recover_epochs':   10,
    'pruning_type':     pruning_type,
    'stop_epochs':      5,
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
# MAGIC ### VGG-19

# COMMAND ----------

# MAGIC %md
# MAGIC #### Magnitude Prune - Movement Prune - Rigging the Lottery

# COMMAND ----------

#######################################
########## MAGNITUDE PRUNING ##########
#######################################

# Creating a classification net for downstream task
model_name = 'vgg19'    

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
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_magp_099.pt'
logger = Logger(model_name, learning_type='pruning')

# Initialing pruning method
pruner = MagnitudePrune(PRUNING_EPOCHS, TARGET_SPARSITY_HIGH)
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
    'recover_epochs':   10,
    'pruning_type':     pruning_type,
    'stop_epochs':      5,
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

######################################
########## MOVEMENT PRUNING ##########
######################################

# Creating a classification net for downstream task
model_name = 'vgg19'    

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
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_mvmp_099.pt'
logger = Logger(model_name, learning_type='pruning')

# Initialing pruning method
pruner = MovementPrune(PRUNING_EPOCHS, 0.985)
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
    'recover_epochs':   10,
    'pruning_type':     pruning_type,
    'stop_epochs':      5,
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
# MAGIC #

# COMMAND ----------

# MAGIC %md
# MAGIC # BaCP Accuracies

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sparsity: 0.95

# COMMAND ----------

# MAGIC %md
# MAGIC ### VGG-11
# MAGIC

# COMMAND ----------

#######################################
########## MAGNITUDE PRUNING ##########
#######################################

# Creating projection models for BaCP framework
model_name = 'vgg11'

# Projection networks
finetuned_weights = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_supcon_c10.pt'
pre_trained_model, current_model, finetuned_model = create_models_for_cap(model_name, finetuned_weights)
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
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_bacp_magp_095'
logger = Logger(model_name, learning_type='bacp')

# Initializing pruner
pruner = MagnitudePrune(BACP_EPOCHS, TARGET_SPARSITY_LOW)

config = {
    'model_name':       model_name,
    'optimizer':        optimizer,
    'scheduler':        scheduler,               
    'criterion':        criterion,
    'temperature':      0.1,
    'base_temperature': 1.0,
    'target_sparsity':  TARGET_SPARSITY_LOW,   
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
is_trained = False
if not is_trained:
    cap_learner.train(trainloader_c10_cl)

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
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_bacp_magp_095_ds.pt'
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

######################################
########## MOVEMENT PRUNING ##########
######################################

# Creating projection models for BaCP framework
model_name = 'vgg11'

# Projection networks
finetuned_weights = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_supcon_c10.pt'
pre_trained_model, current_model, finetuned_model = create_models_for_cap(model_name, finetuned_weights)
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
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_bacp_mvmp_095'
logger = Logger(model_name, learning_type='bacp')

# Initializing pruner
pruner = MovementPrune(BACP_EPOCHS, TARGET_SPARSITY_LOW)

config = {
    'model_name':       model_name,
    'optimizer':        optimizer,
    'scheduler':        scheduler,               
    'criterion':        criterion,
    'temperature':      0.1,
    'base_temperature': 1.0,
    'target_sparsity':  TARGET_SPARSITY_LOW,   
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
is_trained = False
if not is_trained:
    cap_learner.train(trainloader_c10_cl)

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
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_bacp_mvmp_095_ds.pt'
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
# MAGIC ### VGG-19

# COMMAND ----------

#######################################
########## MAGNITUDE PRUNING ##########
#######################################

# Creating projection models for BaCP framework
model_name = 'vgg19'

# Projection networks
finetuned_weights = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_supcon_c10.pt'
pre_trained_model, current_model, finetuned_model = create_models_for_cap(model_name, finetuned_weights)
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
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_bacp_magp_095'
logger = Logger(model_name, learning_type='bacp')

# Initializing pruner
pruner = MagnitudePrune(BACP_EPOCHS, TARGET_SPARSITY_LOW)

config = {
    'model_name':       model_name,
    'optimizer':        optimizer,
    'scheduler':        scheduler,               
    'criterion':        criterion,
    'temperature':      0.1,
    'base_temperature': 1.0,
    'target_sparsity':  TARGET_SPARSITY_LOW,   
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
is_trained = False
if not is_trained:
    cap_learner.train(trainloader_c10_cl)

# COMMAND ----------

# Creating classification model with unfrozen parameters
model_name = 'vgg19'
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
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_bacp_magp_095_ds.pt'
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

######################################
########## MOVEMENT PRUNING ##########
######################################

# Creating projection models for BaCP framework
model_name = 'vgg19'

# Projection networks
finetuned_weights = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_supcon_c10.pt'
pre_trained_model, current_model, finetuned_model = create_models_for_cap(model_name, finetuned_weights)
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
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_bacp_mvmp_095'
logger = Logger(model_name, learning_type='bacp')

# Initializing pruner
pruner = MovementPrune(BACP_EPOCHS, TARGET_SPARSITY_LOW)

config = {
    'model_name':       model_name,
    'optimizer':        optimizer,
    'scheduler':        scheduler,               
    'criterion':        criterion,
    'temperature':      0.1,
    'base_temperature': 1.0,
    'target_sparsity':  TARGET_SPARSITY_LOW,   
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
is_trained = False
if not is_trained:
    cap_learner.train(trainloader_c10_cl)

# COMMAND ----------

# Creating classification model with unfrozen parameters
model_name = 'vgg19'
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
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_bacp_mvmp_095_ds.pt'
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
# MAGIC ## Sparsity: 0.97

# COMMAND ----------

# MAGIC %md
# MAGIC ### VGG-11

# COMMAND ----------

#######################################
########## MAGNITUDE PRUNING ##########
#######################################

# Creating projection models for BaCP framework
model_name = 'vgg11'

# Projection networks
finetuned_weights = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_supcon_c10.pt'
pre_trained_model, current_model, finetuned_model = create_models_for_cap(model_name, finetuned_weights)
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
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_bacp_magp_097'
logger = Logger(model_name, learning_type='bacp')

# Initializing pruner
pruner = MagnitudePrune(BACP_EPOCHS, TARGET_SPARSITY_MID)

config = {
    'model_name':       model_name,
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
is_trained = False
if not is_trained:
    cap_learner.train(trainloader_c10_cl)

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
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_bacp_magp_097_ds.pt'
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

######################################
########## MOVEMENT PRUNING ##########
######################################

# Creating projection models for BaCP framework
model_name = 'vgg11'

# Projection networks
finetuned_weights = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_supcon_c10.pt'
pre_trained_model, current_model, finetuned_model = create_models_for_cap(model_name, finetuned_weights)
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
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_bacp_mvmp_097'
logger = Logger(model_name, learning_type='bacp')

# Initializing pruner
pruner = MovementPrune(BACP_EPOCHS, TARGET_SPARSITY_MID)

config = {
    'model_name':       model_name,
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
is_trained = False
if not is_trained:
    cap_learner.train(trainloader_c10_cl)

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
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_bacp_mvmp_097_ds.pt'
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
# MAGIC ### VGG-19

# COMMAND ----------

#######################################
########## MAGNITUDE PRUNING ##########
#######################################

# Creating projection models for BaCP framework
model_name = 'vgg19'

# Projection networks
finetuned_weights = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_supcon_c10.pt'
pre_trained_model, current_model, finetuned_model = create_models_for_cap(model_name, finetuned_weights)
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
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_bacp_magp_097'
logger = Logger(model_name, learning_type='bacp')

# Initializing pruner
pruner = MagnitudePrune(BACP_EPOCHS, TARGET_SPARSITY_MID)

config = {
    'model_name':       model_name,
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
is_trained = False
if not is_trained:
    cap_learner.train(trainloader_c10_cl)

# COMMAND ----------

# Creating classification model with unfrozen parameters
model_name = 'vgg19'
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
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_bacp_magp_097_ds.pt'
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

######################################
########## MOVEMENT PRUNING ##########
######################################

# Creating projection models for BaCP framework
model_name = 'vgg19'

# Projection networks
finetuned_weights = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_supcon_c10.pt'
pre_trained_model, current_model, finetuned_model = create_models_for_cap(model_name, finetuned_weights)
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
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_bacp_mvmp_097'
logger = Logger(model_name, learning_type='bacp')

# Initializing pruner
pruner = MovementPrune(BACP_EPOCHS, TARGET_SPARSITY_MID)

config = {
    'model_name':       model_name,
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
is_trained = False
if not is_trained:
    cap_learner.train(trainloader_c10_cl)

# COMMAND ----------

# Creating classification model with unfrozen parameters
model_name = 'vgg19'
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
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_bacp_mvmp_097_ds.pt'
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
# MAGIC ## Sparsity: 0.99

# COMMAND ----------

# MAGIC %md
# MAGIC ### VGG-11

# COMMAND ----------

#######################################
########## MAGNITUDE PRUNING ##########
#######################################

# Creating projection models for BaCP framework
model_name = 'vgg11'

# Projection networks
finetuned_weights = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_supcon_c10.pt'
pre_trained_model, current_model, finetuned_model = create_models_for_cap(model_name, finetuned_weights)
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
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_bacp_magp_099'
logger = Logger(model_name, learning_type='bacp')

# Initializing pruner
pruner = MagnitudePrune(BACP_EPOCHS, TARGET_SPARSITY_HIGH)

config = {
    'model_name':       model_name,
    'optimizer':        optimizer,
    'scheduler':        scheduler,               
    'criterion':        criterion,
    'temperature':      0.1,
    'base_temperature': 1.0,
    'target_sparsity':  TARGET_SPARSITY_HIGH,   
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
is_trained = False
if not is_trained:
    cap_learner.train(trainloader_c10_cl)

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
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_bacp_magp_099_ds.pt'
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

######################################
########## MOVEMENT PRUNING ##########
######################################

# Creating projection models for BaCP framework
model_name = 'vgg11'

# Projection networks
finetuned_weights = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_supcon_c10.pt'
pre_trained_model, current_model, finetuned_model = create_models_for_cap(model_name, finetuned_weights)
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
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_bacp_mvmp_099'
logger = Logger(model_name, learning_type='bacp')

# Initializing pruner
pruner = MovementPrune(BACP_EPOCHS, TARGET_SPARSITY_HIGH)

config = {
    'model_name':       model_name,
    'optimizer':        optimizer,
    'scheduler':        scheduler,               
    'criterion':        criterion,
    'temperature':      0.1,
    'base_temperature': 1.0,
    'target_sparsity':  TARGET_SPARSITY_HIGH,   
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
is_trained = False
if not is_trained:
    cap_learner.train(trainloader_c10_cl)

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
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_bacp_mvmp_099_ds.pt'
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
# MAGIC ### VGG-19

# COMMAND ----------

#######################################
########## MAGNITUDE PRUNING ##########
#######################################

# Creating projection models for BaCP framework
model_name = 'vgg19'

# Projection networks
finetuned_weights = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_supcon_c10.pt'
pre_trained_model, current_model, finetuned_model = create_models_for_cap(model_name, finetuned_weights)
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
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_bacp_magp_099'
logger = Logger(model_name, learning_type='bacp')

# Initializing pruner
pruner = MagnitudePrune(BACP_EPOCHS, TARGET_SPARSITY_HIGH)

config = {
    'model_name':       model_name,
    'optimizer':        optimizer,
    'scheduler':        scheduler,               
    'criterion':        criterion,
    'temperature':      0.1,
    'base_temperature': 1.0,
    'target_sparsity':  TARGET_SPARSITY_HIGH,   
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
is_trained = False
if not is_trained:
    cap_learner.train(trainloader_c10_cl)

# COMMAND ----------

# Creating classification model with unfrozen parameters
model_name = 'vgg19'
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
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_bacp_magp_099_ds.pt'
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

######################################
########## MOVEMENT PRUNING ##########
######################################

# Creating projection models for BaCP framework
model_name = 'vgg19'

# Projection networks
finetuned_weights = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_supcon_c10.pt'
pre_trained_model, current_model, finetuned_model = create_models_for_cap(model_name, finetuned_weights)
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
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_bacp_mvmp_099'
logger = Logger(model_name, learning_type='bacp')

# Initializing pruner
pruner = MovementPrune(BACP_EPOCHS, TARGET_SPARSITY_HIGH)

config = {
    'model_name':       model_name,
    'optimizer':        optimizer,
    'scheduler':        scheduler,               
    'criterion':        criterion,
    'temperature':      0.1,
    'base_temperature': 1.0,
    'target_sparsity':  TARGET_SPARSITY_HIGH,   
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
is_trained = False
if not is_trained:
    cap_learner.train(trainloader_c10_cl)

# COMMAND ----------

# Creating classification model with unfrozen parameters
model_name = 'vgg19'
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
save_path = f'/dbfs/{TEST_NO}/{model_name}_weights/{model_name}_bacp_magp_099_ds.pt'
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
