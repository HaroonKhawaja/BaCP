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
from LLM_trainer import LLMTrainer, LLMTrainingArguments
from CV_trainer import Trainer, CVTrainingArguments

import torch
import torch.nn as nn
import torch.optim as optim

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
model_name = "resnet50"
model_task = "cifar10"

training_args = CVTrainingArguments(
    model_name=model_name,
    model_task=model_task,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS_RESNET50,
    learning_type="baseline",
    optimizer_type='sgd',
    learning_rate=0.01,
    scheduler_type='linear_with_warmup'
)
trainer = Trainer(training_args=training_args)
if False:
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
finetuned_weights = f"/dbfs/research/{model_name}/{model_task}/{model_name}_{model_task}_baseline.pt"

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

# COMMAND ----------

# Model initialization
model_name = "resnet50"
model_task = "cifar10"

# Initializing finetuned weights path
finetuned_weights = f"/dbfs/research/{model_name}/{model_task}/{model_name}_{model_task}_baseline.pt"

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
finetuned_weights = f"/dbfs/research/{model_name}/{model_task}/{model_name}_{model_task}_baseline.pt"

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
finetuned_weights = f"/dbfs/research/{model_name}/{model_task}/{model_name}_{model_task}_baseline.pt"

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
finetuned_weights = f"/dbfs/research/{model_name}/{model_task}/{model_name}_{model_task}_baseline.pt"

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
finetuned_weights = f"/dbfs/research/{model_name}/{model_task}/{model_name}_{model_task}_baseline.pt"

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
finetuned_weights = f"/dbfs/research/{model_name}/{model_task}/{model_name}_{model_task}_baseline.pt"

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
    optimizer_type='sgd',
    learning_rate=0.1,
    sparsity_scheduler='cubic'
)
bacp_trainer = BaCPTrainer(bacp_training_args=bacp_training_args)
if False:
    bacp_trainer.train()

# Finetuning Phase
bacp_trainer.generate_mask_from_model()
pruner = bacp_trainer.get_pruner()

training_args = CVTrainingArguments(
    model_name=bacp_trainer.model_name,
    model_task=bacp_trainer.model_task,
    batch_size=bacp_trainer.batch_size,
    pruning_type=bacp_trainer.pruning_type,
    target_sparsity=bacp_trainer.target_sparsity,
    finetuned_weights=bacp_trainer.cm_save_path,
    epochs=50,
    pruner=pruner,
    finetune=True,
    learning_type="bacp_finetune",
    optimizer_type='adamw',
    learning_rate=0.0001,
)
trainer = Trainer(training_args)
if True:
    trainer.train()

acc = trainer.evaluate()
print(f"Accuracy = {acc}")


# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# Model initialization
model_name = "resnet50"
model_task = "cifar10"

# Initializing finetuned weights path
finetuned_weights = f"/dbfs/research/{model_name}/{model_task}/{model_name}_{model_task}_baseline.pt"

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
    optimizer_type='sgd',
    learning_rate=0.1,
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
    target_sparsity=bacp_trainer.target_sparsity,
    finetuned_weights=bacp_trainer.cm_save_path,
    epochs=50,
    pruner=pruner,
    finetune=True,
    learning_type="bacp_finetune",
    optimizer_type='adamw',
    learning_rate=0.0001,
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
finetuned_weights = f"/dbfs/research/{model_name}/{model_task}/{model_name}_{model_task}_baseline.pt"

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
    optimizer_type='sgd',
    learning_rate=0.1,
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
    target_sparsity=bacp_trainer.target_sparsity,
    finetuned_weights=bacp_trainer.cm_save_path,
    epochs=50,
    pruner=pruner,
    finetune=True,
    learning_type="bacp_finetune",
    optimizer_type='adamw',
    learning_rate=0.0001,
)
trainer = Trainer(training_args)
if True:
    trainer.train()

acc = trainer.evaluate()
print(f"Accuracy = {acc}")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Movement Pruning

# COMMAND ----------

# Model initialization
model_name = "resnet50"
model_task = "cifar10"

# Initializing finetuned weights path
finetuned_weights = f"/dbfs/research/{model_name}/{model_task}/{model_name}_{model_task}_baseline.pt"

# Initializing pruning args
pruning_type = 'movement_pruning'
target_sparsity = TARGET_SPARSITY_LOW
learning_type = "pruning"

bacp_training_args = BaCPTrainingArgumentsCNN(
    model_name=model_name,
    model_task=model_task,
    batch_size=BATCH_SIZE,
    finetuned_weights=finetuned_weights,
    pruning_type=pruning_type,
    target_sparsity=target_sparsity,
    optimizer_type='sgd',
    learning_rate=0.1,
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
    target_sparsity=bacp_trainer.target_sparsity,
    finetuned_weights=bacp_trainer.cm_save_path,
    epochs=50,
    pruner=pruner,
    finetune=True,
    learning_type="bacp_finetune",
    optimizer_type='adamw',
    learning_rate=0.0001,
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
finetuned_weights = f"/dbfs/research/{model_name}/{model_task}/{model_name}_{model_task}_baseline.pt"

# Initializing pruning args
pruning_type = 'movement_pruning'
target_sparsity = TARGET_SPARSITY_MID
learning_type = "pruning"

bacp_training_args = BaCPTrainingArgumentsCNN(
    model_name=model_name,
    model_task=model_task,
    batch_size=BATCH_SIZE,
    finetuned_weights=finetuned_weights,
    pruning_type=pruning_type,
    target_sparsity=target_sparsity,
    optimizer_type='sgd',
    learning_rate=0.1,
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
    target_sparsity=bacp_trainer.target_sparsity,
    finetuned_weights=bacp_trainer.cm_save_path,
    epochs=50,
    pruner=pruner,
    finetune=True,
    learning_type="bacp_finetune",
    optimizer_type='adamw',
    learning_rate=0.0001,
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
finetuned_weights = f"/dbfs/research/{model_name}/{model_task}/{model_name}_{model_task}_baseline.pt"

# Initializing pruning args
pruning_type = 'movement_pruning'
target_sparsity = TARGET_SPARSITY_HIGH
learning_type = "pruning"

bacp_training_args = BaCPTrainingArgumentsCNN(
    model_name=model_name,
    model_task=model_task,
    batch_size=BATCH_SIZE,
    finetuned_weights=finetuned_weights,
    pruning_type=pruning_type,
    target_sparsity=target_sparsity,
    optimizer_type='sgd',
    learning_rate=0.1,
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
    target_sparsity=bacp_trainer.target_sparsity,
    finetuned_weights=bacp_trainer.cm_save_path,
    epochs=50,
    pruner=pruner,
    finetune=True,
    learning_type="bacp_finetune",
    optimizer_type='adamw',
    learning_rate=0.0001,
)
trainer = Trainer(training_args)
if True:
    trainer.train()

acc = trainer.evaluate()
print(f"Accuracy = {acc}")

