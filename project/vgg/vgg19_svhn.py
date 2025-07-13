# Databricks notebook source
# MAGIC %md
# MAGIC # VGG-19 Testing Notebook

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC # Enables autoreload; learn more at https://docs.databricks.com/en/files/workspace-modules.html#autoreload-for-python-modules
# MAGIC # To disable autoreload; run %autoreload 0

# COMMAND ----------

import sys
import os
sys.path.append(os.path.abspath('..'))

import torch
import torch.nn as nn
import torch.optim as optim
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()
os.environ["HF_DATASETS_CACHE"] = "/dbfs/hf_datasets"
os.environ["TOKENIZERS_PARALLELISM"] = "false" 

from trainer import Trainer, TrainingArguments
from bacp import BaCPTrainer, BaCPTrainingArguments
from unstructured_pruning import check_sparsity_distribution
from utils import *
from constants import *

device = get_device()
print(f"{device = }")


# COMMAND ----------

# Notebook specific variables
MODEL_NAME = 'vgg19'
MODEL_TASK = 'svhn'
TRAIN = False

# COMMAND ----------

# MAGIC %md
# MAGIC ## Baseline Accuracies

# COMMAND ----------

training_args = TrainingArguments(
    model_name=MODEL_NAME,
    model_task=MODEL_TASK,
    batch_size=BATCH_SIZE,
    optimizer_type='sgd',
    learning_rate=0.01,
    scheduler_type='linear_with_warmup',
    epochs=EPOCHS_VGG11,
    learning_type="baseline",
    patience=50,
)
trainer = Trainer(training_args=training_args)
if TRAIN:
    trainer.train()

metrics = trainer.evaluate()
print(f"\n{metrics}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pruning Accuracies

# COMMAND ----------

# MAGIC %md
# MAGIC ### Magnitude Pruning

# COMMAND ----------

# Initializing finetuned weights path
finetuned_weights = f"/dbfs/research/{MODEL_NAME}/{MODEL_TASK}/{MODEL_NAME}_{MODEL_TASK}_baseline.pt"
training_args = TrainingArguments(
    model_name=MODEL_NAME,
    model_task=MODEL_TASK,
    batch_size=BATCH_SIZE,
    optimizer_type='sgd',
    learning_rate=0.01,
    pruning_type="magnitude_pruning",
    target_sparsity=TARGET_SPARSITY_LOW,
    sparsity_scheduler='cubic',
    finetuned_weights=finetuned_weights,
    learning_type="pruning",
)
trainer = Trainer(training_args)
if TRAIN:
    trainer.train()

metrics = trainer.evaluate()
print(f"\n{metrics}")

# COMMAND ----------

# Initializing finetuned weights path
finetuned_weights = f"/dbfs/research/{MODEL_NAME}/{MODEL_TASK}/{MODEL_NAME}_{MODEL_TASK}_baseline.pt"
training_args = TrainingArguments(
    model_name=MODEL_NAME,
    model_task=MODEL_TASK,
    batch_size=BATCH_SIZE,
    optimizer_type='sgd',
    learning_rate=0.01,
    pruning_type="magnitude_pruning",
    target_sparsity=TARGET_SPARSITY_MID,
    sparsity_scheduler='cubic',
    finetuned_weights=finetuned_weights,
    learning_type="pruning",
)
trainer = Trainer(training_args)
if TRAIN:
    trainer.train()

metrics = trainer.evaluate()
print(f"\n{metrics}")

# COMMAND ----------

# Initializing finetuned weights path
finetuned_weights = f"/dbfs/research/{MODEL_NAME}/{MODEL_TASK}/{MODEL_NAME}_{MODEL_TASK}_baseline.pt"
training_args = TrainingArguments(
    model_name=MODEL_NAME,
    model_task=MODEL_TASK,
    batch_size=BATCH_SIZE,
    optimizer_type='sgd',
    learning_rate=0.01,
    pruning_type="magnitude_pruning",
    target_sparsity=TARGET_SPARSITY_HIGH,
    sparsity_scheduler='cubic',
    finetuned_weights=finetuned_weights,
    learning_type="pruning",
)
trainer = Trainer(training_args)
if TRAIN:
    trainer.train()

metrics = trainer.evaluate()
print(f"\n{metrics}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Movement Pruning

# COMMAND ----------

# Initializing finetuned weights path
finetuned_weights = f"/dbfs/research/{MODEL_NAME}/{MODEL_TASK}/{MODEL_NAME}_{MODEL_TASK}_baseline.pt"
training_args = TrainingArguments(
    model_name=MODEL_NAME,
    model_task=MODEL_TASK,
    batch_size=BATCH_SIZE,
    optimizer_type='sgd',
    learning_rate=0.01,
    pruning_type="movement_pruning",
    target_sparsity=TARGET_SPARSITY_LOW,
    sparsity_scheduler='cubic',
    finetuned_weights=finetuned_weights,
    learning_type="pruning",
)
trainer = Trainer(training_args)
if TRAIN:
    trainer.train()

metrics = trainer.evaluate()
print(f"\n{metrics}")

# COMMAND ----------

# Initializing finetuned weights path
finetuned_weights = f"/dbfs/research/{MODEL_NAME}/{MODEL_TASK}/{MODEL_NAME}_{MODEL_TASK}_baseline.pt"
training_args = TrainingArguments(
    model_name=MODEL_NAME,
    model_task=MODEL_TASK,
    batch_size=BATCH_SIZE,
    optimizer_type='sgd',
    learning_rate=0.01,
    pruning_type="movement_pruning",
    target_sparsity=TARGET_SPARSITY_MID,
    sparsity_scheduler='cubic',
    finetuned_weights=finetuned_weights,
    learning_type="pruning",
)
trainer = Trainer(training_args)
if TRAIN:
    trainer.train()

metrics = trainer.evaluate()
print(f"\n{metrics}")

# COMMAND ----------

# Initializing finetuned weights path
finetuned_weights = f"/dbfs/research/{MODEL_NAME}/{MODEL_TASK}/{MODEL_NAME}_{MODEL_TASK}_baseline.pt"
training_args = TrainingArguments(
    model_name=MODEL_NAME,
    model_task=MODEL_TASK,
    batch_size=BATCH_SIZE,
    optimizer_type='sgd',
    learning_rate=0.01,
    pruning_type="movement_pruning",
    target_sparsity=TARGET_SPARSITY_HIGH,
    sparsity_scheduler='cubic',
    finetuned_weights=finetuned_weights,
    learning_type="pruning",
)
trainer = Trainer(training_args)
if TRAIN:
    trainer.train()

metrics = trainer.evaluate()
print(f"\n{metrics}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## BaCP Accuracies

# COMMAND ----------

# MAGIC %md
# MAGIC ### Magnitude Pruning

# COMMAND ----------

# Initializing finetuned weights path
finetuned_weights = f"/dbfs/research/{MODEL_NAME}/{MODEL_TASK}/{MODEL_NAME}_{MODEL_TASK}_baseline.pt"

bacp_training_args = BaCPTrainingArguments(
    model_name=MODEL_NAME,
    model_task=MODEL_TASK,
    batch_size=BATCH_SIZE,
    optimizer_type='sgd',
    learning_rate=0.1,
    pruning_type='magnitude_pruning',
    target_sparsity=TARGET_SPARSITY_LOW,
    sparsity_scheduler='cubic',
    finetuned_weights=finetuned_weights,
    learning_type='bacp_pruning',
)
bacp_trainer = BaCPTrainer(bacp_training_args=bacp_training_args)
if TRAIN:
    bacp_trainer.train()

# Finetuning Phase
bacp_trainer.generate_mask_from_model()
training_args = TrainingArguments(
    model_name=bacp_trainer.model_name,
    model_task=bacp_trainer.model_task,
    batch_size=bacp_trainer.batch_size,
    optimizer_type='adamw',
    learning_rate=0.0001,
    pruner=bacp_trainer.get_pruner(),
    pruning_type=bacp_trainer.pruning_type,
    target_sparsity=bacp_trainer.target_sparsity,
    epochs=50,
    finetuned_weights=bacp_trainer.save_path,
    finetune=True,
    learning_type="bacp_finetune",
)
trainer = Trainer(training_args)
if TRAIN:
    trainer.train()

metrics = trainer.evaluate()
print(f"\n{metrics}")

# COMMAND ----------

# Initializing finetuned weights path
finetuned_weights = f"/dbfs/research/{MODEL_NAME}/{MODEL_TASK}/{MODEL_NAME}_{MODEL_TASK}_baseline.pt"

bacp_training_args = BaCPTrainingArguments(
    model_name=MODEL_NAME,
    model_task=MODEL_TASK,
    batch_size=BATCH_SIZE,
    optimizer_type='sgd',
    learning_rate=0.1,
    pruning_type='magnitude_pruning',
    target_sparsity=TARGET_SPARSITY_MID,
    sparsity_scheduler='cubic',
    finetuned_weights=finetuned_weights,
    learning_type='bacp_pruning',
)
bacp_trainer = BaCPTrainer(bacp_training_args=bacp_training_args)
if TRAIN:
    bacp_trainer.train()

# Finetuning Phase
bacp_trainer.generate_mask_from_model()
training_args = TrainingArguments(
    model_name=bacp_trainer.model_name,
    model_task=bacp_trainer.model_task,
    batch_size=bacp_trainer.batch_size,
    optimizer_type='adamw',
    learning_rate=0.0001,
    pruner=bacp_trainer.get_pruner(),
    pruning_type=bacp_trainer.pruning_type,
    target_sparsity=bacp_trainer.target_sparsity,
    epochs=50,
    finetuned_weights=bacp_trainer.save_path,
    finetune=True,
    learning_type="bacp_finetune",
)
trainer = Trainer(training_args)
if TRAIN:
    trainer.train()

metrics = trainer.evaluate()
print(f"\n{metrics}")

# COMMAND ----------

# Initializing finetuned weights path
finetuned_weights = f"/dbfs/research/{MODEL_NAME}/{MODEL_TASK}/{MODEL_NAME}_{MODEL_TASK}_baseline.pt"

bacp_training_args = BaCPTrainingArguments(
    model_name=MODEL_NAME,
    model_task=MODEL_TASK,
    batch_size=BATCH_SIZE,
    optimizer_type='sgd',
    learning_rate=0.1,
    pruning_type='magnitude_pruning',
    target_sparsity=TARGET_SPARSITY_HIGH,
    sparsity_scheduler='cubic',
    finetuned_weights=finetuned_weights,
    learning_type='bacp_pruning',
)
bacp_trainer = BaCPTrainer(bacp_training_args=bacp_training_args)
if TRAIN:
    bacp_trainer.train()

# Finetuning Phase
bacp_trainer.generate_mask_from_model()
training_args = TrainingArguments(
    model_name=bacp_trainer.model_name,
    model_task=bacp_trainer.model_task,
    batch_size=bacp_trainer.batch_size,
    optimizer_type='adamw',
    learning_rate=0.0001,
    pruner=bacp_trainer.get_pruner(),
    pruning_type=bacp_trainer.pruning_type,
    target_sparsity=bacp_trainer.target_sparsity,
    epochs=50,
    finetuned_weights=bacp_trainer.save_path,
    finetune=True,
    learning_type="bacp_finetune",
)
trainer = Trainer(training_args)
if TRAIN:
    trainer.train()

metrics = trainer.evaluate()
print(f"\n{metrics}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Movement Pruning

# COMMAND ----------

# Initializing finetuned weights path
finetuned_weights = f"/dbfs/research/{MODEL_NAME}/{MODEL_TASK}/{MODEL_NAME}_{MODEL_TASK}_baseline.pt"

bacp_training_args = BaCPTrainingArguments(
    model_name=MODEL_NAME,
    model_task=MODEL_TASK,
    batch_size=BATCH_SIZE,
    optimizer_type='sgd',
    learning_rate=0.1,
    pruning_type='movement_pruning',
    target_sparsity=TARGET_SPARSITY_LOW,
    sparsity_scheduler='cubic',
    finetuned_weights=finetuned_weights,
    learning_type='bacp_pruning',
)
bacp_trainer = BaCPTrainer(bacp_training_args=bacp_training_args)
if TRAIN:
    bacp_trainer.train()

# Finetuning Phase
bacp_trainer.generate_mask_from_model()
training_args = TrainingArguments(
    model_name=bacp_trainer.model_name,
    model_task=bacp_trainer.model_task,
    batch_size=bacp_trainer.batch_size,
    optimizer_type='adamw',
    learning_rate=0.0001,
    pruner=bacp_trainer.get_pruner(),
    pruning_type=bacp_trainer.pruning_type,
    target_sparsity=bacp_trainer.target_sparsity,
    epochs=50,
    finetuned_weights=bacp_trainer.save_path,
    finetune=True,
    learning_type="bacp_finetune",
)
trainer = Trainer(training_args)
if TRAIN:
    trainer.train()

metrics = trainer.evaluate()
print(f"\n{metrics}")

# COMMAND ----------

# Initializing finetuned weights path
finetuned_weights = f"/dbfs/research/{MODEL_NAME}/{MODEL_TASK}/{MODEL_NAME}_{MODEL_TASK}_baseline.pt"

bacp_training_args = BaCPTrainingArguments(
    model_name=MODEL_NAME,
    model_task=MODEL_TASK,
    batch_size=BATCH_SIZE,
    optimizer_type='sgd',
    learning_rate=0.1,
    pruning_type='movement_pruning',
    target_sparsity=TARGET_SPARSITY_MID,
    sparsity_scheduler='cubic',
    finetuned_weights=finetuned_weights,
    learning_type='bacp_pruning',
)
bacp_trainer = BaCPTrainer(bacp_training_args=bacp_training_args)
if TRAIN:
    bacp_trainer.train()

# Finetuning Phase
bacp_trainer.generate_mask_from_model()
training_args = TrainingArguments(
    model_name=bacp_trainer.model_name,
    model_task=bacp_trainer.model_task,
    batch_size=bacp_trainer.batch_size,
    optimizer_type='adamw',
    learning_rate=0.0001,
    pruner=bacp_trainer.get_pruner(),
    pruning_type=bacp_trainer.pruning_type,
    target_sparsity=bacp_trainer.target_sparsity,
    epochs=50,
    finetuned_weights=bacp_trainer.save_path,
    finetune=True,
    learning_type="bacp_finetune",
)
trainer = Trainer(training_args)
if TRAIN:
    trainer.train()

metrics = trainer.evaluate()
print(f"\n{metrics}")


# COMMAND ----------

# Initializing finetuned weights path
finetuned_weights = f"/dbfs/research/{MODEL_NAME}/{MODEL_TASK}/{MODEL_NAME}_{MODEL_TASK}_baseline.pt"

bacp_training_args = BaCPTrainingArguments(
    model_name=MODEL_NAME,
    model_task=MODEL_TASK,
    batch_size=BATCH_SIZE,
    optimizer_type='sgd',
    learning_rate=0.1,
    pruning_type='movement_pruning',
    target_sparsity=TARGET_SPARSITY_HIGH,
    sparsity_scheduler='cubic',
    finetuned_weights=finetuned_weights,
    learning_type='bacp_pruning',
)
bacp_trainer = BaCPTrainer(bacp_training_args=bacp_training_args)
if TRAIN:
    bacp_trainer.train()

# Finetuning Phase
bacp_trainer.generate_mask_from_model()
training_args = TrainingArguments(
    model_name=bacp_trainer.model_name,
    model_task=bacp_trainer.model_task,
    batch_size=bacp_trainer.batch_size,
    optimizer_type='adamw',
    learning_rate=0.0001,
    pruner=bacp_trainer.get_pruner(),
    pruning_type=bacp_trainer.pruning_type,
    target_sparsity=bacp_trainer.target_sparsity,
    epochs=50,
    finetuned_weights=bacp_trainer.save_path,
    finetune=True,
    learning_type="bacp_finetune",
)
trainer = Trainer(training_args)
if TRAIN:
    trainer.train()

metrics = trainer.evaluate()
print(f"\n{metrics}")
