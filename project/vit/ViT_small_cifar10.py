# Databricks notebook source
# MAGIC %md
# MAGIC # ViT-Small Testing Notebook

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC # Enables autoreload; learn more at https://docs.databricks.com/en/files/workspace-modules.html#autoreload-for-python-modules
# MAGIC # To disable autoreload; run %autoreload 0

# COMMAND ----------

import os
import sys
sys.path.append(os.path.abspath('..'))

from constants import (
    TARGET_SPARSITY_LOW, TARGET_SPARSITY_MID, TARGET_SPARSITY_HIGH,
    BATCH_SIZE_CNN, BATCH_SIZE_VIT, BATCH_SIZE_LLM,
    EPOCHS_SMALL_MODEL, EPOCHS_LARGE_MODEL, EPOCHS_VIT
)
from utils import get_device, get_num_workers, load_weights
from unstructured_pruning import check_model_sparsity, check_sparsity_distribution
from trainer import TrainingArguments, Trainer
from bacp import BaCPTrainingArguments, BaCPTrainer

from datasets.utils.logging import disable_progress_bar
disable_progress_bar()
os.environ["HF_DATASETS_CACHE"] = "/dbfs/hf_datasets"
os.environ["TOKENIZERS_PARALLELISM"] = "false" 


# COMMAND ----------

# Notebook specific variables
MODEL_NAME = 'vit_small'
MODEL_TASK = 'cifar10'
TRAIN = False

# COMMAND ----------

# MAGIC %md
# MAGIC ## Baseline Accuracies

# COMMAND ----------

training_args = TrainingArguments(
    model_name=MODEL_NAME,
    model_task=MODEL_TASK,
    batch_size=BATCH_SIZE_VIT,
    optimizer_type_and_lr=('sgd', 0.01),
    scheduler_type='linear_with_warmup',
    epochs=10,
    learning_type="baseline",
)
trainer = Trainer(training_args=training_args)
if TRAIN:
    trainer.train()

metrics = trainer.evaluate()
print(f"\n{metrics}")

# COMMAND ----------

FINETUNED_WEIGHTS = f"/dbfs/research/{MODEL_NAME}/{MODEL_TASK}/{MODEL_NAME}_{MODEL_TASK}_baseline.pt"


# COMMAND ----------

# MAGIC %md
# MAGIC ## Pruning Accuracies

# COMMAND ----------

# MAGIC %md
# MAGIC ### Magnitude Pruning

# COMMAND ----------

# Initializing finetuned weights path
training_args = TrainingArguments(
    model_name=MODEL_NAME,
    model_task=MODEL_TASK,
    batch_size=BATCH_SIZE_VIT,
    optimizer_type_and_lr=('sgd', 0.01),
    pruning_type="magnitude_pruning",
    target_sparsity=TARGET_SPARSITY_LOW,
    sparsity_scheduler='cubic',
    finetuned_weights=FINETUNED_WEIGHTS,
    learning_type="pruning",
)
trainer = Trainer(training_args)
if TRAIN:
    trainer.train()

metrics = trainer.evaluate()
print(f"\n{metrics}")


# COMMAND ----------

training_args = TrainingArguments(
    model_name=MODEL_NAME,
    model_task=MODEL_TASK,
    batch_size=BATCH_SIZE_VIT,
    optimizer_type_and_lr=('sgd', 0.01),
    pruning_type="magnitude_pruning",
    target_sparsity=TARGET_SPARSITY_MID,
    sparsity_scheduler='cubic',
    finetuned_weights=FINETUNED_WEIGHTS,
    learning_type="pruning",
)
trainer = Trainer(training_args)
if TRAIN:
    trainer.train()

metrics = trainer.evaluate()
print(f"\n{metrics}")

# COMMAND ----------

training_args = TrainingArguments(
    model_name=MODEL_NAME,
    model_task=MODEL_TASK,
    batch_size=BATCH_SIZE_VIT,
    optimizer_type_and_lr=('sgd', 0.01),
    pruning_type="magnitude_pruning",
    target_sparsity=TARGET_SPARSITY_HIGH,
    sparsity_scheduler='cubic',
    finetuned_weights=FINETUNED_WEIGHTS,
    learning_type="pruning",
)
trainer = Trainer(training_args)
if TRAIN:
    trainer.train()

metrics = trainer.evaluate()
print(f"\n{metrics}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### SNIP-it Pruning

# COMMAND ----------

training_args = TrainingArguments(
    model_name=MODEL_NAME,
    model_task=MODEL_TASK,
    batch_size=BATCH_SIZE_VIT,
    optimizer_type_and_lr=('sgd', 0.01),
    pruning_type="snip_pruning",
    target_sparsity=TARGET_SPARSITY_LOW,
    sparsity_scheduler='cubic',
    finetuned_weights=FINETUNED_WEIGHTS,
    learning_type="pruning",
)
trainer = Trainer(training_args)
if TRAIN:
    trainer.train()

metrics = trainer.evaluate()
print(f"\n{metrics}")


# COMMAND ----------

training_args = TrainingArguments(
    model_name=MODEL_NAME,
    model_task=MODEL_TASK,
    batch_size=BATCH_SIZE_VIT,
    optimizer_type_and_lr=('sgd', 0.01),
    pruning_type="snip_pruning",
    target_sparsity=TARGET_SPARSITY_MID,
    sparsity_scheduler='cubic',
    finetuned_weights=FINETUNED_WEIGHTS,
    learning_type="pruning",
)
trainer = Trainer(training_args)
if TRAIN:
    trainer.train()

metrics = trainer.evaluate()
print(f"\n{metrics}")


# COMMAND ----------

training_args = TrainingArguments(
    model_name=MODEL_NAME,
    model_task=MODEL_TASK,
    batch_size=BATCH_SIZE_VIT,
    optimizer_type_and_lr=('sgd', 0.01),
    pruning_type="snip_pruning",
    target_sparsity=TARGET_SPARSITY_HIGH,
    sparsity_scheduler='cubic',
    finetuned_weights=FINETUNED_WEIGHTS,
    learning_type="pruning",
)
trainer = Trainer(training_args)
if TRAIN:
    trainer.train()

metrics = trainer.evaluate()
print(f"\n{metrics}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Wanda Pruning

# COMMAND ----------

training_args = TrainingArguments(
    model_name=MODEL_NAME,
    model_task=MODEL_TASK,
    batch_size=BATCH_SIZE_VIT,
    optimizer_type_and_lr=('sgd', 0.01),
    pruning_type="wanda_pruning",
    target_sparsity=TARGET_SPARSITY_LOW,
    sparsity_scheduler='cubic',
    finetuned_weights=FINETUNED_WEIGHTS,
    learning_type="pruning",
)
trainer = Trainer(training_args)
if TRAIN:
    trainer.train()

metrics = trainer.evaluate()
print(f"\n{metrics}")


# COMMAND ----------

training_args = TrainingArguments(
    model_name=MODEL_NAME,
    model_task=MODEL_TASK,
    batch_size=BATCH_SIZE_VIT,
    optimizer_type_and_lr=('sgd', 0.01),
    pruning_type="wanda_pruning",
    target_sparsity=TARGET_SPARSITY_MID,
    sparsity_scheduler='cubic',
    finetuned_weights=FINETUNED_WEIGHTS,
    learning_type="pruning",
)
trainer = Trainer(training_args)
if TRAIN:
    trainer.train()

metrics = trainer.evaluate()
print(f"\n{metrics}")

# COMMAND ----------

training_args = TrainingArguments(
    model_name=MODEL_NAME,
    model_task=MODEL_TASK,
    batch_size=BATCH_SIZE_VIT,
    optimizer_type_and_lr=('sgd', 0.01),
    pruning_type="wanda_pruning",
    target_sparsity=TARGET_SPARSITY_HIGH,
    sparsity_scheduler='cubic',
    finetuned_weights=FINETUNED_WEIGHTS,
    learning_type="pruning",
)
trainer = Trainer(training_args)
if True:
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

bacp_training_args = BaCPTrainingArguments(
    model_name=MODEL_NAME,
    model_task=MODEL_TASK,
    batch_size=BATCH_SIZE_VIT,
    optimizer_type_and_lr=('sgd', 0.01),
    pruning_type='magnitude_pruning',
    target_sparsity=TARGET_SPARSITY_LOW,
    sparsity_scheduler='cubic',
    finetuned_weights=FINETUNED_WEIGHTS,
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
    optimizer_type_and_lr=('adamw', 0.0001),
    pruner=bacp_trainer.get_pruner(),
    pruning_type=bacp_trainer.pruning_type,
    target_sparsity=bacp_trainer.target_sparsity,
    finetuned_weights=bacp_trainer.save_path,
    finetune=True,
    learning_type="bacp_finetune",
    epochs=50,
)
trainer = Trainer(training_args)
if TRAIN:
    trainer.train()

metrics = trainer.evaluate()
print(f"\n{metrics}")


# COMMAND ----------

bacp_training_args = BaCPTrainingArguments(
    model_name=MODEL_NAME,
    model_task=MODEL_TASK,
    batch_size=BATCH_SIZE_VIT,
    optimizer_type_and_lr=('sgd', 0.01),
    pruning_type='magnitude_pruning',
    target_sparsity=TARGET_SPARSITY_MID,
    sparsity_scheduler='cubic',
    finetuned_weights=FINETUNED_WEIGHTS,
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
    optimizer_type_and_lr=('adamw', 0.0001),
    pruner=bacp_trainer.get_pruner(),
    pruning_type=bacp_trainer.pruning_type,
    target_sparsity=bacp_trainer.target_sparsity,
    finetuned_weights=bacp_trainer.save_path,
    finetune=True,
    learning_type="bacp_finetune",
    epochs=50,
)
trainer = Trainer(training_args)
if TRAIN:
    trainer.train()

metrics = trainer.evaluate()
print(f"\n{metrics}")


# COMMAND ----------

bacp_training_args = BaCPTrainingArguments(
    model_name=MODEL_NAME,
    model_task=MODEL_TASK,
    batch_size=BATCH_SIZE_VIT,
    optimizer_type_and_lr=('sgd', 0.01),
    pruning_type='magnitude_pruning',
    target_sparsity=TARGET_SPARSITY_HIGH,
    sparsity_scheduler='cubic',
    finetuned_weights=FINETUNED_WEIGHTS,
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
    optimizer_type_and_lr=('adamw', 0.0001),
    pruner=bacp_trainer.get_pruner(),
    pruning_type=bacp_trainer.pruning_type,
    target_sparsity=bacp_trainer.target_sparsity,
    finetuned_weights=bacp_trainer.save_path,
    finetune=True,
    learning_type="bacp_finetune",
    epochs=50,
)
trainer = Trainer(training_args)
if TRAIN:
    trainer.train()

metrics = trainer.evaluate()
print(f"\n{metrics}")


# COMMAND ----------

# MAGIC %md
# MAGIC ### SNIP-it Pruning

# COMMAND ----------

bacp_training_args = BaCPTrainingArguments(
    model_name=MODEL_NAME,
    model_task=MODEL_TASK,
    batch_size=BATCH_SIZE_VIT,
    optimizer_type_and_lr=('sgd', 0.01),
    pruning_type='snip_pruning',
    target_sparsity=TARGET_SPARSITY_LOW,
    sparsity_scheduler='cubic',
    finetuned_weights=FINETUNED_WEIGHTS,
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
    optimizer_type_and_lr=('adamw', 0.0001),
    pruner=bacp_trainer.get_pruner(),
    pruning_type=bacp_trainer.pruning_type,
    target_sparsity=bacp_trainer.target_sparsity,
    finetuned_weights=bacp_trainer.save_path,
    finetune=True,
    learning_type="bacp_finetune",
    epochs=50,
)
trainer = Trainer(training_args)
if TRAIN:
    trainer.train()

metrics = trainer.evaluate()
print(f"\n{metrics}")


# COMMAND ----------

bacp_training_args = BaCPTrainingArguments(
    model_name=MODEL_NAME,
    model_task=MODEL_TASK,
    batch_size=BATCH_SIZE_VIT,
    optimizer_type_and_lr=('sgd', 0.01),
    pruning_type='snip_pruning',
    target_sparsity=TARGET_SPARSITY_MID,
    sparsity_scheduler='cubic',
    finetuned_weights=FINETUNED_WEIGHTS,
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
    optimizer_type_and_lr=('adamw', 0.0001),
    pruner=bacp_trainer.get_pruner(),
    pruning_type=bacp_trainer.pruning_type,
    target_sparsity=bacp_trainer.target_sparsity,
    finetuned_weights=bacp_trainer.save_path,
    finetune=True,
    learning_type="bacp_finetune",
    epochs=50,
)
trainer = Trainer(training_args)
if TRAIN:
    trainer.train()

metrics = trainer.evaluate()
print(f"\n{metrics}")


# COMMAND ----------

bacp_training_args = BaCPTrainingArguments(
    model_name=MODEL_NAME,
    model_task=MODEL_TASK,
    batch_size=BATCH_SIZE_VIT,
    optimizer_type_and_lr=('sgd', 0.01),
    pruning_type='snip_pruning',
    target_sparsity=TARGET_SPARSITY_HIGH,
    sparsity_scheduler='cubic',
    finetuned_weights=FINETUNED_WEIGHTS,
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
    optimizer_type_and_lr=('adamw', 0.0001),
    pruner=bacp_trainer.get_pruner(),
    pruning_type=bacp_trainer.pruning_type,
    target_sparsity=bacp_trainer.target_sparsity,
    finetuned_weights=bacp_trainer.save_path,
    finetune=True,
    learning_type="bacp_finetune",
    epochs=50,
)
trainer = Trainer(training_args)
if TRAIN:
    trainer.train()

metrics = trainer.evaluate()
print(f"\n{metrics}")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Wanda Pruning

# COMMAND ----------

bacp_training_args = BaCPTrainingArguments(
    model_name=MODEL_NAME,
    model_task=MODEL_TASK,
    batch_size=BATCH_SIZE_VIT,
    optimizer_type_and_lr=('sgd', 0.01),
    pruning_type='wanda_pruning',
    target_sparsity=TARGET_SPARSITY_LOW,
    sparsity_scheduler='cubic',
    finetuned_weights=FINETUNED_WEIGHTS,
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
    optimizer_type_and_lr=('adamw', 0.0001),
    pruner=bacp_trainer.get_pruner(),
    pruning_type=bacp_trainer.pruning_type,
    target_sparsity=bacp_trainer.target_sparsity,
    finetuned_weights=bacp_trainer.save_path,
    finetune=True,
    learning_type="bacp_finetune",
    epochs=50,
)
trainer = Trainer(training_args)
if TRAIN:
    trainer.train()

metrics = trainer.evaluate()
print(f"\n{metrics}")

# COMMAND ----------


check_sparsity_distribution(trainer.model)

# COMMAND ----------

bacp_training_args = BaCPTrainingArguments(
    model_name=MODEL_NAME,
    model_task=MODEL_TASK,
    batch_size=BATCH_SIZE_VIT,
    optimizer_type_and_lr=('sgd', 0.01),
    pruning_type='wanda_pruning',
    target_sparsity=TARGET_SPARSITY_MID,
    sparsity_scheduler='cubic',
    finetuned_weights=FINETUNED_WEIGHTS,
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
    optimizer_type_and_lr=('adamw', 0.0001),
    pruner=bacp_trainer.get_pruner(),
    pruning_type=bacp_trainer.pruning_type,
    target_sparsity=bacp_trainer.target_sparsity,
    finetuned_weights=bacp_trainer.save_path,
    finetune=True,
    learning_type="bacp_finetune",
    epochs=50,
)
trainer = Trainer(training_args)
if TRAIN:
    trainer.train()

metrics = trainer.evaluate()
print(f"\n{metrics}")


# COMMAND ----------

bacp_training_args = BaCPTrainingArguments(
    model_name=MODEL_NAME,
    model_task=MODEL_TASK,
    batch_size=BATCH_SIZE_VIT,
    optimizer_type_and_lr=('sgd', 0.01),
    pruning_type='wanda_pruning',
    target_sparsity=TARGET_SPARSITY_HIGH,
    sparsity_scheduler='cubic',
    finetuned_weights=FINETUNED_WEIGHTS,
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
    optimizer_type_and_lr=('adamw', 0.0001),
    pruner=bacp_trainer.get_pruner(),
    pruning_type=bacp_trainer.pruning_type,
    target_sparsity=bacp_trainer.target_sparsity,
    finetuned_weights=bacp_trainer.save_path,
    finetune=True,
    learning_type="bacp_finetune",
    epochs=50,
)
trainer = Trainer(training_args)
if TRAIN:
    trainer.train()

metrics = trainer.evaluate()
print(f"\n{metrics}")

