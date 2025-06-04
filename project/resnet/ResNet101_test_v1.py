# Databricks notebook source
# MAGIC %md
# MAGIC # ResNet-50 Testing Notebook

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC # Enables autoreload; learn more at https://docs.databricks.com/en/files/workspace-modules.html#autoreload-for-python-modules
# MAGIC # To disable autoreload; run %autoreload 0

# COMMAND ----------

# MAGIC %pip install torchinfo
# MAGIC %restart_python

# COMMAND ----------

import sys
import os
sys.path.append(os.path.abspath('..'))

from bacp import BaCPTrainer, BaCPTrainingArgumentsLLM, BaCPTrainingArgumentsCNN
from models import EncoderProjectionNetwork, ClassificationNetwork
from unstructured_pruning import MagnitudePrune, MovementPrune, LocalMagnitudePrune, LocalMovementPrune, WandaPrune, PRUNER_DICT, check_model_sparsity
from LLM_trainer import LLMTrainer, LLMTrainingArguments
from CV_trainer import Trainer, CVTrainingArguments
from dataset_utils import get_glue_data
from logger import Logger

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from tqdm import tqdm
from torchinfo import summary

from datasets.utils.logging import disable_progress_bar
disable_progress_bar()
import os
os.environ["HF_DATASETS_CACHE"] = "/dbfs/hf_datasets"
os.environ["TOKENIZERS_PARALLELISM"] = "false" 

from utils import *
from constants import *

device = get_device()
print(f"{device = }")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Baseline Accuracies

# COMMAND ----------

# Model initialization
model_name = "resnet101"
model_task = "cifar10"

training_args = CVTrainingArguments(
    model_name=model_name,
    model_task=model_task,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS_RESNET50,
    learning_type="baseline",
    learning_rate=0.001,
    optimizer_type='sgd',
)

trainer = Trainer(training_args=training_args)

if True:
    trainer.train()

acc = trainer.evaluate()
print(f"\nAccuracy = {acc}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pruning Accuracies

# COMMAND ----------

# MAGIC %md
# MAGIC ### Magnitude Pruning

# COMMAND ----------

# Model initialization
model_name = "resnet50"
model_task = "cifar10"

# Initializing finetuned weights path
finetuned_weights = f"/dbfs/research/{model_name}/{model_task}/{model_name}_baseline.pt"

# Initializing pruning args
pruning_type = "magnitude_pruning"
target_sparsity = TARGET_SPARSITY_LOW
learning_type = "pruning"

training_args = CVTrainingArguments(
    model_name=model_name,
    model_task=model_task,
    batch_size=BATCH_SIZE,
    finetuned_weights=finetuned_weights,
    pruning_type=pruning_type,
    target_sparsity=target_sparsity,
    learning_type=learning_type,
    learning_rate=0.01,
    optimizer_type='sgd',
    sparsity_scheduler='cubic'
)

trainer = Trainer(training_args=training_args)

if True:
    trainer.train()

acc = trainer.evaluate()
print(f"\nAccuracy = {acc}")

check_sparsity_distribution(trainer.model, verbose=False)

# COMMAND ----------

# Model initialization
model_name = "resnet50"
model_task = "cifar10"

# Initializing finetuned weights path
finetuned_weights = f"/dbfs/research/{model_name}/{model_task}/{model_name}_baseline.pt"

# Initializing pruning args
pruning_type = "magnitude_pruning"
target_sparsity = TARGET_SPARSITY_MID
learning_type = "pruning"

training_args = CVTrainingArguments(
    model_name=model_name,
    model_task=model_task,
    batch_size=BATCH_SIZE,
    finetuned_weights=finetuned_weights,
    pruning_type=pruning_type,
    target_sparsity=target_sparsity,
    learning_type=learning_type,
    learning_rate=0.01,
    optimizer_type='sgd',
    sparsity_scheduler='cubic'
)

trainer = Trainer(training_args=training_args)

if True:
    trainer.train()

acc = trainer.evaluate()
print(f"\nAccuracy = {acc}")

# COMMAND ----------

# Model initialization
model_name = "resnet50"
model_task = "cifar10"

# Initializing finetuned weights path
finetuned_weights = f"/dbfs/research/{model_name}/{model_task}/{model_name}_baseline.pt"

# Initializing pruning args
pruning_type = "magnitude_pruning"
target_sparsity = TARGET_SPARSITY_HIGH
learning_type = "pruning"

training_args = CVTrainingArguments(
    model_name=model_name,
    model_task=model_task,
    batch_size=BATCH_SIZE,
    finetuned_weights=finetuned_weights,
    pruning_type=pruning_type,
    target_sparsity=target_sparsity,
    learning_type=learning_type,
    learning_rate=0.01,
    optimizer_type='sgd',
    sparsity_scheduler='cubic'
)

trainer = Trainer(training_args=training_args)

if True:
    trainer.train()

acc = trainer.evaluate()
print(f"\nAccuracy = {acc}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Movement Pruning

# COMMAND ----------

# Model initialization
model_name = "resnet50"
model_task = "cifar10"

# Initializing finetuned weights path
finetuned_weights = f"/dbfs/research/{model_name}/{model_task}/{model_name}_baseline.pt"

# Initializing pruning args
pruning_type = 'movement_pruning'
target_sparsity = TARGET_SPARSITY_LOW
learning_type = "pruning"

training_args = CVTrainingArguments(
    model_name=model_name,
    model_task=model_task,
    batch_size=BATCH_SIZE,
    finetuned_weights=finetuned_weights,
    pruning_type=pruning_type,
    target_sparsity=target_sparsity,
    learning_type=learning_type,
    learning_rate=0.01,
    optimizer_type='sgd',
    sparsity_scheduler='cubic'
)

trainer = Trainer(training_args=training_args)

if True:
    trainer.train()

acc = trainer.evaluate()
print(f"\nAccuracy = {acc}")

# COMMAND ----------

# Model initialization
model_name = "resnet50"
model_task = "cifar10"

# Initializing finetuned weights path
finetuned_weights = f"/dbfs/research/{model_name}/{model_task}/{model_name}_baseline.pt"

# Initializing pruning args
pruning_type = 'movement_pruning'
target_sparsity = TARGET_SPARSITY_MID
learning_type = "pruning"

training_args = CVTrainingArguments(
    model_name=model_name,
    model_task=model_task,
    batch_size=BATCH_SIZE,
    finetuned_weights=finetuned_weights,
    pruning_type=pruning_type,
    target_sparsity=target_sparsity,
    learning_type=learning_type,

    learning_rate=0.01,
    optimizer_type='sgd',
    sparsity_scheduler='cubic'
)

trainer = Trainer(training_args=training_args)

if True:
    trainer.train()

acc = trainer.evaluate()
print(f"\nAccuracy = {acc}")

# COMMAND ----------

# Model initialization
model_name = "resnet50"
model_task = "cifar10"

# Initializing finetuned weights path
finetuned_weights = f"/dbfs/research/{model_name}/{model_task}/{model_name}_baseline.pt"

# Initializing pruning args
pruning_type = 'movement_pruning'
target_sparsity = TARGET_SPARSITY_HIGH
learning_type = "pruning"

training_args = CVTrainingArguments(
    model_name=model_name,
    model_task=model_task,
    batch_size=BATCH_SIZE,
    finetuned_weights=finetuned_weights,
    pruning_type=pruning_type,
    target_sparsity=target_sparsity,
    learning_type=learning_type,

    learning_rate=0.01,
    optimizer_type='sgd',
    sparsity_scheduler='cubic'
)

trainer = Trainer(training_args=training_args)

if True:
    trainer.train()

acc = trainer.evaluate()
print(f"\nAccuracy = {acc}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## BaCP Accuracies

# COMMAND ----------

# MAGIC %md
# MAGIC ### Magnitude Pruning

# COMMAND ----------

# Model initialization
model_name = "resnet50"
model_task = "cifar10"

# Initializing finetuned weights path
finetuned_weights = f"/dbfs/research/{model_name}/{model_task}/{model_name}_baseline.pt"

# Initializing pruning args
pruning_type = 'magnitude_pruning'
target_sparsity = TARGET_SPARSITY_LOW
learning_type = "pruning"

bacp_training_args = BaCPTrainingArgumentsCNN(
    model_name=model_name,
    model_task=model_task,
    batch_size=BATCH_SIZE,
    finetuned_weights=finetuned_weights,
    pruning_type=pruning_type,
    target_sparsity=target_sparsity,
    learning_rate=0.01,
    optimizer_type='sgd',
    sparsity_scheduler='cubic'
)
bacp_trainer = BaCPTrainer(bacp_training_args=bacp_training_args)
if True:
    bacp_trainer.train()

# Finetuning Phase
bacp_trainer.generate_mask_from_model()
pruner = bacp_trainer.get_pruner()

training_args = CVTrainingArguments(
    model_name=bacp_trainer.model_name,
    model_task=bacp_trainer.model_task,
    batch_size=bacp_trainer.batch_size,
    pruning_type=bacp_trainer.pruning_type,
    target_sparsity=target_sparsity,
    finetuned_weights=bacp_trainer.cm_save_path,
    epochs=10,
    pruner=pruner,
    finetune=True,
    learning_type="bacp_finetune",
)
trainer = Trainer(training_args)
if True:
    trainer.train()

acc = trainer.evaluate()
print(f"Accuracy = {acc}")


# COMMAND ----------

# Model initialization
model_name = "resnet50"
model_task = "cifar10"

# Initializing finetuned weights path
finetuned_weights = f"/dbfs/research/{model_name}/{model_task}/{model_name}_baseline.pt"

# Initializing pruning args
pruning_type = 'magnitude_pruning'
target_sparsity = TARGET_SPARSITY_MID
learning_type = "pruning"

bacp_training_args = BaCPTrainingArgumentsCNN(
    model_name=model_name,
    model_task=model_task,
    batch_size=BATCH_SIZE,
    finetuned_weights=finetuned_weights,
    pruning_type=pruning_type,
    target_sparsity=target_sparsity,
    learning_rate=0.01,
    optimizer_type='sgd',
    sparsity_scheduler='cubic'
)
bacp_trainer = BaCPTrainer(bacp_training_args=bacp_training_args)
if True:
    bacp_trainer.train()

# Finetuning Phase
bacp_trainer.generate_mask_from_model()
pruner = bacp_trainer.get_pruner()

training_args = CVTrainingArguments(
    model_name=bacp_trainer.model_name,
    model_task=bacp_trainer.model_task,
    batch_size=bacp_trainer.batch_size,
    pruning_type=bacp_trainer.pruning_type,
    target_sparsity=target_sparsity,
    finetuned_weights=bacp_trainer.cm_save_path,
    epochs=10,
    pruner=pruner,
    finetune=True,
    learning_type="bacp_finetune",
)
trainer = Trainer(training_args)
if True:
    trainer.train()

acc = trainer.evaluate()
print(f"Accuracy = {acc}")


# COMMAND ----------

# Model initialization
model_name = "resnet50"
model_task = "cifar10"

# Initializing finetuned weights path
finetuned_weights = f"/dbfs/research/{model_name}/{model_task}/{model_name}_baseline.pt"

# Initializing pruning args
pruning_type = 'magnitude_pruning'
target_sparsity = TARGET_SPARSITY_HIGH
learning_type = "pruning"

bacp_training_args = BaCPTrainingArgumentsCNN(
    model_name=model_name,
    model_task=model_task,
    batch_size=BATCH_SIZE,
    finetuned_weights=finetuned_weights,
    pruning_type=pruning_type,
    target_sparsity=target_sparsity,
    learning_rate=0.01,
    optimizer_type='sgd',
    sparsity_scheduler='cubic'
)
bacp_trainer = BaCPTrainer(bacp_training_args=bacp_training_args)
if True:
    bacp_trainer.train()

# Finetuning Phase
bacp_trainer.generate_mask_from_model()
pruner = bacp_trainer.get_pruner()

training_args = CVTrainingArguments(
    model_name=bacp_trainer.model_name,
    model_task=bacp_trainer.model_task,
    batch_size=bacp_trainer.batch_size,
    pruning_type=bacp_trainer.pruning_type,
    target_sparsity=target_sparsity,
    finetuned_weights=bacp_trainer.cm_save_path,
    epochs=10,
    pruner=pruner,
    finetune=True,
    learning_type="bacp_finetune",
)
trainer = Trainer(training_args)
if True:
    trainer.train()

acc = trainer.evaluate()
print(f"Accuracy = {acc}")


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
