import torch
import os
import matplotlib.pyplot as plt
from bacp import BaCPTrainer
from trainer import Trainer, TrainingArguments
from torch.amp import GradScaler
from training_utils import (
    _detect_model_type, _detect_num_classes, _detect_cv_image_size,
    _initialize_models, _initialize_optimizer, _initialize_scheduler,
    _initialize_data_loaders, _initialize_pruner, _initialize_contrastive_losses,
    _initialize_paths_and_logger, _handle_optimizer_and_pruning
)
import uuid

class TemperatureSweep():
    def __init__(self,
                 model_name,
                 model_task,
                 batch_size,
                 opt_type_and_lr,
                 finetune_opt_type_and_lr,
                 finetuned_weights,
                 
                 scheduler_type=None,
                 pruner=None,
                 pruning_type=None,
                 target_sparsity=0.0,
                 sparsity_scheduler='cubic',
                 pruning_epochs=None,

                 epochs = 5,
                 finetune_epochs=50,
                 recovery_epochs=10,
                 patience=None,
                 
                 learning_type='bacp_TS',

                 log_epochs=True,
                 enable_tqdm=True,
                 enable_mixed_precision=True,
                 db=True,
                 num_workers=24):
        self.tau_sweep = [0.03, 0.07, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]
        self.model_name = model_name
        self.model_task = model_task
        self.batch_size = batch_size
        self.optimizer_type, self.learning_rate = opt_type_and_lr
        self.optimizer_type_finetune, self.learning_rate_finetune = finetune_opt_type_and_lr
        self.finetuned_weights = finetuned_weights
        self.finetune_epochs = finetune_epochs
        self.is_bacp = True
        self.scheduler_type = None

        # Pruning parameters 
        self.pruner = None
        self.pruning_type = pruning_type
        self.target_sparsity = target_sparsity
        self.sparsity_scheduler = sparsity_scheduler
        self.pruning_epochs = epochs or pruning_epochs
        self.finetune=False
        
        # Training parameters
        self.epochs = epochs
        self.recovery_epochs = recovery_epochs
        self.learning_type = learning_type
        self.patience = patience or self.epochs

        # Extra parameters
        self.log_epochs = log_epochs
        self.enable_tqdm = enable_tqdm
        self.enable_mixed_precision = enable_mixed_precision
        self.db = db
        self.num_workers = num_workers
        self.scaler = GradScaler() if self.enable_mixed_precision else None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        _detect_model_type(self)
        _detect_num_classes(self)
        _detect_cv_image_size(self)
        _initialize_data_loaders(self)

    def sweep(self):
        self.history = {}

        print("------------------------------")
        print("STARTING TEMPERATURE SWEEP")
        print("------------------------------")
        for i, tau in enumerate(self.tau_sweep):
            print(f"TEST {i+1}/{len(self.tau_sweep)}")
            self.learning_type = f"bacp_TS_{tau}"
            print(f"TEMPERATURE: {tau}\n")
            _initialize_models(self)
            _initialize_optimizer(self)
            _initialize_scheduler(self)
            _initialize_contrastive_losses(self, tau)
            _initialize_pruner(self)
            _initialize_paths_and_logger(self)

            bacp_trainer = BaCPTrainer(self)
            bacp_trainer.train()

            bacp_trainer.generate_mask_from_model()
            training_args = TrainingArguments(
                model_name = bacp_trainer.model_name,
                model_task = bacp_trainer.model_task,
                batch_size = bacp_trainer.batch_size,               
                optimizer_type = self.optimizer_type_finetune,
                learning_rate = self.learning_rate_finetune,
                pruner = bacp_trainer.get_pruner(),
                pruning_type = bacp_trainer.pruning_type,
                target_sparsity = bacp_trainer.target_sparsity,
                epochs=self.finetune_epochs,
                finetuned_weights=bacp_trainer.save_path,
                finetune=True,
                learning_type=f'bacp_TS_finetune_{tau}',
            )
            trainer = Trainer(training_args)
            trainer.train()
            metrics = trainer.evaluate()
            avg_acc = metrics['average_accuracy']
            print(f"ACCURACY: {avg_acc}\n")

            self.history[tau] = avg_acc
        print("------------------------------")
        print("FINISHED TEMPERATURE SWEEP")
        print("------------------------------")


        print("PLOTTING RESULTS")
        plt.plot(self.tau_sweep, self.history.values())
        plt.xlabel("Temperature")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy vs Temperature (Sparsity {self.target_sparsity})")
        plt.grid()
        plt.show()

        random_id = uuid.uuid4()
        plt.savefig(f"accuracy_vs_temperature_{random_id}.png")

class LossFunctionSweep():
    def __init__(self,
                 model_name,
                 model_task,
                 batch_size,
                 opt_type_and_lr,
                 finetune_opt_type_and_lr,
                 finetuned_weights,
                 tau,
                 
                 scheduler_type=None,
                 pruner=None,
                 pruning_type=None,
                 target_sparsity=0.0,
                 sparsity_scheduler='cubic',
                 pruning_epochs=None,

                 epochs = 5,
                 finetune_epochs=50,
                 recovery_epochs=10,
                 patience=None,
                 
                 learning_type='bacp_TS',

                 log_epochs=True,
                 enable_tqdm=True,
                 enable_mixed_precision=True,
                 db=True,
                 num_workers=24):
        self.tau = tau
        self.model_name = model_name
        self.model_task = model_task
        self.batch_size = batch_size
        self.optimizer_type, self.learning_rate = opt_type_and_lr
        self.optimizer_type_finetune, self.learning_rate_finetune = finetune_opt_type_and_lr
        self.finetuned_weights = finetuned_weights
        self.finetune_epochs = finetune_epochs
        self.is_bacp = True
        self.scheduler_type = None

        # Pruning parameters 
        self.pruner = None
        self.pruning_type = pruning_type
        self.target_sparsity = target_sparsity
        self.sparsity_scheduler = sparsity_scheduler
        self.pruning_epochs = epochs or pruning_epochs
        self.finetune=False
        
        # Training parameters
        self.epochs = epochs
        self.recovery_epochs = recovery_epochs
        self.learning_type = learning_type
        self.patience = patience or self.epochs

        # Extra parameters
        self.log_epochs = log_epochs
        self.enable_tqdm = enable_tqdm
        self.enable_mixed_precision = enable_mixed_precision
        self.db = db
        self.num_workers = num_workers
        self.scaler = GradScaler() if self.enable_mixed_precision else None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        _detect_model_type(self)
        _detect_num_classes(self)
        _detect_cv_image_size(self)
        _initialize_data_loaders(self)

        self.configs = ['disable_supervised_loss', 'disable_unsupervised_loss', 'disable_all_loss', 'all_losses']        
        self.history = {}

    def sweep(self):
        print("------------------------------")
        print("STARTING LOSS FUNCTION SWEEP")
        print("------------------------------")

        for i, disable in enumerate(self.configs):
            print(f"TEST {i+1}/{len(self.configs)}: {disable}")

            self.learning_type = f"bacp_TS_{disable}"
            self.disable = disable

            _initialize_models(self)
            _initialize_optimizer(self)
            _initialize_scheduler(self)
            _initialize_contrastive_losses(self, self.tau)
            _initialize_pruner(self)
            _initialize_paths_and_logger(self)

            # Training BaCP
            bacp_trainer = BaCPTrainer(self)
            bacp_trainer.train()

            # Fine-tuning on downstream task
            bacp_trainer.generate_mask_from_model()
            training_args = TrainingArguments(
                model_name=bacp_trainer.model_name,
                model_task=bacp_trainer.model_task,
                batch_size=bacp_trainer.batch_size,
                optimizer_type=self.optimizer_type_finetune,
                learning_rate=self.learning_rate_finetune,
                pruner=bacp_trainer.get_pruner(),
                pruning_type=bacp_trainer.pruning_type,
                target_sparsity=bacp_trainer.target_sparsity,
                epochs=self.finetune_epochs,
                finetuned_weights=bacp_trainer.save_path,
                finetune=True,
                learning_type=f'finetune_{disable}',
            )
            trainer = Trainer(training_args)
            trainer.train()

            # Evaluating accuracy
            metrics = trainer.evaluate()
            acc = metrics['average_accuracy']
            self.history[disable] = acc
            print(f"ACCURACY: {acc:.3f}\n")

        print("------------------------------")
        print("FINISHED LOSS FUNCTION SWEEP")
        print("------------------------------")

        names = [
            "Unsupervised Only",
            "Supervised Only",
            "No Contrastive Loss",
            "All Losses (Current)",
        ]        
        values = list(self.history.values())

        # Summary
        print("RESULT SUMMARY:")
        for name, value in zip(names, values):
            print(f"{name}: {value:.3f}%")
        print("------------------------------")

        # Printing best outcome
        max_acc = max(values)
        max_idx = values.index(max_acc)
        max_name = names[max_idx]

        print(f"\nBEST RESULT: {max_acc:.3f}% ({max_name})")
        print(f"ACCURACY DIFFERENCE FROM THE BEST RESULT:")
        for name, value in zip(names, values):
            change = max_acc - value
            if change > 0:
                print(f"{name}: {change:+.3f}%")
        print("------------------------------")

        # Plotting accuracy graphs
        print("\nPLOTTING RESULTS")
        plt.figure(figsize=(10, 6))
        plt.plot(names, values, marker='o', linestyle='--', color='b', markerfacecolor='red')
        plt.xlabel("Loss Functions")
        plt.ylabel("Accuracy")
        plt.title(f"Ablation: Effect of Contrastive Loss Components on Accuracy (Sparsity {self.target_sparsity})")
        plt.grid()
        plt.tight_layout()
        plt.show()

        # Saving graph as picture
        random_id = uuid.uuid4()
        file_name = f"accuracy_vs_contrastive_components_{random_id}.png"
        plt.savefig(file_name)
        print(f"Saved as: {file_name}")

class BaCPLearningRateSweep():
    def __init__(self,
                 model_name,
                 model_task,
                 batch_size,
                 finetuned_weights,
                 
                 scheduler_type=None,
                 pruner=None,
                 pruning_type=None,
                 target_sparsity=0.0,
                 sparsity_scheduler='cubic',
                 pruning_epochs=None,

                 epochs=5,
                 finetune_epochs=10,
                 recovery_epochs=50,
                 patience=None,
                 learning_type='bacp_TS',

                 log_epochs=True,
                 enable_tqdm=True,
                 enable_mixed_precision=True,
                 db=True,
                 num_workers=24):
        self.optimizer_lr_sweep = {
            "sgd":    [0.01, 0.03, 0.05, 0.1, 0.3], 
            "adamw":  [0.0001, 0.0003, 0.001, 0.005, 0.01],
        }
        self.model_name = model_name
        self.model_task = model_task
        self.batch_size = batch_size
        self.finetuned_weights = finetuned_weights
        self.finetune_epochs = finetune_epochs
        self.scheduler_type = None
        self.is_bacp = True
        self.tau = 20

        # Pruning parameters 
        self.pruner = None
        self.pruning_type = pruning_type
        self.target_sparsity = target_sparsity
        self.sparsity_scheduler = sparsity_scheduler
        self.pruning_epochs = epochs or pruning_epochs
        self.finetune=False
        
        # Training parameters
        self.epochs = epochs
        self.recovery_epochs = recovery_epochs
        self.learning_type = learning_type
        self.patience = patience or self.epochs

        # Extra parameters
        self.log_epochs = log_epochs
        self.enable_tqdm = enable_tqdm
        self.enable_mixed_precision = enable_mixed_precision
        self.db = db
        self.num_workers = num_workers
        self.scaler = GradScaler() if self.enable_mixed_precision else None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        _detect_model_type(self)
        _detect_num_classes(self)
        _detect_cv_image_size(self)
        _initialize_data_loaders(self)

    def sweep(self):
        self.history = {}

        print("------------------------------")
        print("STARTING LEARNING RATE SWEEP")
        print("------------------------------")
        for optimizer_type in self.optimizer_lr_sweep:
            learning_rates = self.optimizer_lr_sweep[optimizer_type]
            for i, learning_rate in enumerate(learning_rates):
                print(f"TEST {i+1}/{len(learning_rates)} for {optimizer_type}")
                print(f"OPTIMIZER AND LR: {optimizer_type}({learning_rate})\n")

                self.learning_type = f"bacp_LRS_{optimizer_type}_{learning_rate}"
                self.optimizer_type = optimizer_type
                self.learning_rate = learning_rate

                _initialize_models(self)
                _initialize_optimizer(self)
                _initialize_scheduler(self)
                _initialize_contrastive_losses(self, self.tau)
                _initialize_pruner(self)
                _initialize_paths_and_logger(self)

                bacp_trainer = BaCPTrainer(self)
                bacp_trainer.train()

                bacp_trainer.generate_mask_from_model()
                training_args = TrainingArguments(
                    model_name = bacp_trainer.model_name,
                    model_task = bacp_trainer.model_task,
                    batch_size = bacp_trainer.batch_size,               
                    optimizer_type = 'adamw',
                    learning_rate = 0.0001,
                    pruner = bacp_trainer.get_pruner(),
                    pruning_type = bacp_trainer.pruning_type,
                    target_sparsity = bacp_trainer.target_sparsity,
                    epochs=self.finetune_epochs,
                    finetuned_weights=bacp_trainer.save_path,
                    finetune=True,
                    learning_type=f'{bacp_trainer.learning_type}_finetune',
                )
                trainer = Trainer(training_args)
                trainer.train()
                metrics = trainer.evaluate()
                avg_acc = metrics['average_accuracy']
                print(f"ACCURACY: {avg_acc}\n")

                if optimizer_type not in self.history:
                    self.history[optimizer_type] = {}
                self.history[optimizer_type][learning_rate] = avg_acc

        print("------------------------------")
        print("FINISHED LEARNING RATE SWEEP")
        print("------------------------------")

        print("PLOTTING RESULTS")
        # Make a subplot for both types of optimizers
        fig, axs = plt.subplots(1, 2, figsize=(10, 10))
        for i, optimizer_type in enumerate(self.optimizer_lr_sweep):
            axs[i].plot(self.optimizer_lr_sweep[optimizer_type], self.history[optimizer_type].values())
            axs[i].set_xlabel("Learning Rate")
            axs[i].set_ylabel("Accuracy")
            axs[i].set_title(f"Accuracy vs Learning Rate ({optimizer_type})")
            axs[i].grid()
        plt.tight_layout()
        plt.show()

        random_id = uuid.uuid4()
        plt.savefig(f"accuracy_vs_opt_n_lr_{random_id}.png")

class LearningRateSweep():
    def __init__(self,
                 model_name,
                 model_task,
                 batch_size,
                 finetuned_weights,
                 
                 scheduler_type=None,
                 pruner=None,
                 pruning_type=None,
                 target_sparsity=0.0,
                 sparsity_scheduler='cubic',
                 pruning_epochs=None,

                 epochs=5,
                 finetune_epochs=50,
                 recovery_epochs=10,
                 patience=None,
                 
                 learning_type='bacp_TS',

                 log_epochs=True,
                 enable_tqdm=True,
                 enable_mixed_precision=True,
                 db=True,
                 num_workers=24):
        self.optimizer_lr_sweep = {
            "sgd":    [0.01, 0.03, 0.05, 0.1, 0.3], 
            "adamw":  [0.0001, 0.0003, 0.001, 0.005, 0.01],
        }
        self.tau = 0.15
        self.model_name = model_name
        self.model_task = model_task
        self.batch_size = batch_size
        self.finetuned_weights = finetuned_weights
        self.finetune_epochs = finetune_epochs
        self.scheduler_type = None
        self.bacp = False

        # Pruning parameters 
        self.pruner = None
        self.pruning_type = pruning_type
        self.target_sparsity = target_sparsity
        self.sparsity_scheduler = sparsity_scheduler
        self.pruning_epochs = epochs or pruning_epochs
        self.finetune=False
        
        # Training parameters
        self.epochs = epochs
        self.recovery_epochs = recovery_epochs
        self.learning_type = learning_type
        self.patience = patience or self.epochs

        # Extra parameters
        self.log_epochs = log_epochs
        self.enable_tqdm = enable_tqdm
        self.enable_mixed_precision = enable_mixed_precision
        self.db = db
        self.num_workers = num_workers
        self.scaler = GradScaler() if self.enable_mixed_precision else None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def sweep(self):
        self.history = {}

        print("------------------------------")
        print("STARTING LEARNING RATE SWEEP")
        print("------------------------------")
        for optimizer_type in self.optimizer_lr_sweep:
            learning_rates = self.optimizer_lr_sweep[optimizer_type]
            for i, learning_rate in enumerate(learning_rates):
                print(f"TEST {i+1}/{len(learning_rates)} for {optimizer_type}")
                print(f"OPTIMIZER AND LR: {optimizer_type}({learning_rate})\n")

                self.learning_type = f"LRS_{optimizer_type}_{learning_rate}"
                self.optimizer_type = optimizer_type
                self.learning_rate = learning_rate

                training_args = TrainingArguments(
                    model_name = self.model_name,
                    model_task = self.model_task,
                    batch_size = self.batch_size,
                    optimizer_type = self.optimizer_type,
                    learning_rate = self.learning_rate,
                    pruning_type = self.pruning_type,
                    target_sparsity = self.target_sparsity,
                    scheduler_type = self.scheduler_type,
                    sparsity_scheduler = self.sparsity_scheduler,
                    pruning_epochs = self.pruning_epochs,
                    epochs = self.epochs,
                    recovery_epochs = self.recovery_epochs,
                    finetuned_weights = self.finetuned_weights,
                    learning_type = self.learning_type
                )
                trainer = Trainer(training_args)
                trainer.train()
                metrics = trainer.evaluate()
                avg_acc = metrics['average_accuracy']
                print(f"ACCURACY: {avg_acc}\n")

                if optimizer_type not in self.history:
                    self.history[optimizer_type] = {}
                self.history[optimizer_type][learning_rate] = avg_acc

        print("------------------------------")
        print("FINISHED LEARNING RATE SWEEP")
        print("------------------------------")


        print("PLOTTING RESULTS")
        # Make a subplot for both types of optimizers
        fig, axs = plt.subplots(1, 2, figsize=(10, 10))
        for i, optimizer_type in enumerate(self.optimizer_lr_sweep):
            axs[i].plot(self.optimizer_lr_sweep[optimizer_type], self.history[optimizer_type].values())
            axs[i].set_xlabel("Learning Rate")
            axs[i].set_ylabel("Accuracy")
            axs[i].set_title(f"Accuracy vs Learning Rate ({optimizer_type})")
            axs[i].grid()
        plt.tight_layout()
        plt.show()

        random_id = uuid.uuid4()
        plt.savefig(f"accuracy_vs_opt_n_lr_{random_id}.png")

class BaCPDataViewSweep():
    def __init__(self,
                 model_name,
                 model_task,
                 batch_size,
                 opt_type_and_lr,
                 finetune_opt_type_and_lr,
                 finetuned_weights,
                 
                 scheduler_type=None,
                 pruner=None,
                 pruning_type=None,
                 target_sparsity=0.0,
                 sparsity_scheduler='cubic',
                 pruning_epochs=None,

                 epochs = 5,
                 finetune_epochs=50,
                 recovery_epochs=10,
                 patience=None,
                 
                 learning_type='bacp_TS',

                 log_epochs=True,
                 enable_tqdm=True,
                 enable_mixed_precision=True,
                 db=True,
                 num_workers=24):
        self.tau = 0.15
        self.model_name = model_name
        self.model_task = model_task
        self.batch_size = batch_size
        self.optimizer_type, self.learning_rate = opt_type_and_lr
        self.optimizer_type_finetune, self.learning_rate_finetune = finetune_opt_type_and_lr
        self.finetuned_weights = finetuned_weights
        self.finetune_epochs = finetune_epochs
        self.is_bacp = True
        self.scheduler_type = None

        # Pruning parameters 
        self.pruner = None
        self.pruning_type = pruning_type
        self.target_sparsity = target_sparsity
        self.sparsity_scheduler = sparsity_scheduler
        self.pruning_epochs = epochs or pruning_epochs
        self.finetune=False
        
        # Training parameters
        self.epochs = epochs
        self.recovery_epochs = recovery_epochs
        self.learning_type = learning_type
        self.patience = patience or self.epochs

        # Extra parameters
        self.log_epochs = log_epochs
        self.enable_tqdm = enable_tqdm
        self.enable_mixed_precision = enable_mixed_precision
        self.db = db
        self.num_workers = num_workers
        self.scaler = GradScaler() if self.enable_mixed_precision else None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        _detect_model_type(self)
        _detect_num_classes(self)
        _detect_cv_image_size(self)
        _initialize_data_loaders(self)

        self.configs = ['use_different_data_view', 'use_same_data_view']        
        self.history = {}

    def sweep(self):
        print("------------------------------")
        print("STARTING LOSS FUNCTION SWEEP")
        print("------------------------------")

        for i, disable in enumerate(self.configs):
            print(f"TEST {i+1}/{len(self.configs)}: {disable}")

            self.learning_type = f"bacp_DV_{disable}"
            self.disable = disable

            _initialize_models(self)
            _initialize_optimizer(self)
            _initialize_scheduler(self)
            _initialize_contrastive_losses(self, self.tau)
            _initialize_pruner(self)
            _initialize_paths_and_logger(self)

            # Training BaCP
            bacp_trainer = BaCPTrainer(self)
            bacp_trainer.train()

            # Fine-tuning on downstream task
            bacp_trainer.generate_mask_from_model()
            training_args = TrainingArguments(
                model_name=bacp_trainer.model_name,
                model_task=bacp_trainer.model_task,
                batch_size=bacp_trainer.batch_size,
                optimizer_type=self.optimizer_type_finetune,
                learning_rate=self.learning_rate_finetune,
                pruner=bacp_trainer.get_pruner(),
                pruning_type=bacp_trainer.pruning_type,
                target_sparsity=bacp_trainer.target_sparsity,
                epochs=self.finetune_epochs,
                finetuned_weights=bacp_trainer.save_path,
                finetune=True,
                learning_type=f'{bacp_trainer.learning_type}_finetune',
            )
            trainer = Trainer(training_args)
            trainer.train()

            # Evaluating accuracy
            metrics = trainer.evaluate()
            acc = metrics['average_accuracy']
            self.history[disable] = acc
            print(f"ACCURACY: {acc:.3f}\n")

        print("------------------------------")
        print("FINISHED LOSS FUNCTION SWEEP")
        print("------------------------------")

        names = [
            'Different Batch View',
            'Same Batch View (Current)'
        ]        
        values = list(self.history.values())

        # Summary
        print("RESULT SUMMARY:")
        for name, value in zip(names, values):
            print(f"{name}: {value:.3f}%")
        print("------------------------------")

        # Printing best outcome
        max_acc = max(values)
        max_idx = values.index(max_acc)
        max_name = names[max_idx]

        print(f"\nBEST RESULT: {max_acc:.3f}% ({max_name})")
        print(f"ACCURACY DIFFERENCE FROM THE BEST RESULT:")
        for name, value in zip(names, values):
            change = max_acc - value
            if change > 0:
                print(f"{name}: -{change:.3f}%")
        print("------------------------------")

        # Plotting accuracy graphs
        print("\nPLOTTING RESULTS")
        plt.figure(figsize=(10, 6))
        plt.plot(names, values, marker='o', linestyle='--', color='b', markerfacecolor='red')
        plt.xlabel("Loss Functions")
        plt.ylabel("Accuracy")
        plt.title(f"Ablation: Effect of Using Varying Data Views (Sparsity {self.target_sparsity})")
        plt.grid()
        plt.tight_layout()
        plt.show()

        # Saving graph as picture
        random_id = uuid.uuid4()
        file_name = f"accuracy_vs_data_view_{random_id}.png"
        plt.savefig(file_name)
        print(f"Saved as: {file_name}")























            
