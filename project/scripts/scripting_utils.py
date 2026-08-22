import os
import argparse
import datetime


def wandb_login():
    # wandb is imported lazily: it is used only here, but a module-scope import
    # forced every consumer of the argument parsers -- including the test suite --
    # to have wandb installed.
    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        print("[WARNING] WANDB_API_KEY not found, skipping login.")
        return
    import wandb
    wandb.login(key=api_key)


def get_timestamp():
    timestamp = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    return timestamp

# Keep in sync with model_factory.MODELS and dataset_factory. Pinned by
# test_imports.py::test_cli_model_choices_are_all_registered and
# ::test_cli_dataset_choices_are_all_registered -- the two lists had drifted
# before, so the CLI accepted vgg11/vgg19 while both were unregistered.
models = [
    # vision (this project's backbones, DyReLU-capable)
    'resnet34', 'resnet50', 'resnet101', 'vgg11', 'vgg19',
    # vision transformers (HF hub; no DyReLU; use --image_size 224)
    'vit-tiny', 'vit-small',
    # language: sequence classification (SST-2)
    'distilbert-base-uncased', 'roberta-base',
    # language: masked language modelling (WikiText-2)
    'distilbert-base-uncased-mlm', 'roberta-base-mlm',
]
model_types = ['cv', 'llm']
datasets = [
    'cifar10', 'cifar100', 'mnist', 'fmnist', 'emnist', 'svhn', 'imagenet',
    'sst2', 'wikitext2',
]


# The parsers are built and parsed in two steps. `build_*_parser()` returns the
# parser without consuming sys.argv, so project/runner.py can
# introspect it -- which option strings exist, which are store_true flags -- and
# construct command lines from cell configs without duplicating the CLI contract
# in a second place. A duplicated contract is how a sweep ends up silently
# dropping a flag the script never learned about.

def build_baseline_parser():
    parser = argparse.ArgumentParser(
        description="Train a baseline dense model."
    )

    # Required arguments
    parser.add_argument('--model_name',     type=str, choices=models,       required=True)
    parser.add_argument('--model_type',     type=str, choices=model_types,  required=True)
    parser.add_argument('--dataset_name',   type=str, choices=datasets,     required=True)
    parser.add_argument('--num_classes',    type=int,                       required=True)

    # Optional arguments with defaults
    parser.add_argument('--batch_size',      type=int,      default=512)
    parser.add_argument('--optimizer_type',  type=str,      default='sgd')
    parser.add_argument('--learning_rate',   type=float,    default=0.1)
    parser.add_argument('--image_size',      type=int,      default=32)
    parser.add_argument('--epochs',          type=int,      default=100)
    parser.add_argument('--scheduler_type',  type=str,      default='linear_with_warmup')
    parser.add_argument('--patience',        type=int,      default=20)
    parser.add_argument('--experiment_type', type=str,      default='baseline')
    parser.add_argument('--log_epochs',          action='store_true', help="Enable detailed logging per epoch to the local logger.")
    parser.add_argument('--enable_tqdm',         action='store_true', help="Enable TQDM progress bars during training.")
    parser.add_argument('--databricks_env',      action='store_true', help="Set paths for Databricks environment (e.g., /dbfs).")

    parser.add_argument('--num_workers',     type=int,      default=os.cpu_count())

    # DyReLU phasing
    parser.add_argument('--dyrelu_en',         action='store_true')
    parser.add_argument('--dyrelu_phasing_en', action='store_true')

    # Weight Sharing
    parser.add_argument('--weight_sharing_en', action='store_true')

    parser.add_argument('--layerwise_alloc', type=str, default='erk',
                        choices=['erk', 'uniform'],
                        help="How the global sparsity budget is split across layers. "
                             "'erk' (default) matches RigL/EAST; 'uniform' gives every "
                             "layer the same density and is the floor arm of the "
                             "uniform-vs-ERK comparison. Only LocalMagnitudePrune reads it.")

    parser.add_argument('--limit_train_batches', type=int, default=0,
                        help="Truncate the train loader to N batches (0 = all). Tier-0 "
                             "smoke only; recorded on every run so a truncated result "
                             "can never be read as a measurement.")
    parser.add_argument('--limit_eval_batches',  type=int, default=0,
                        help="Truncate the val/test loaders to N batches (0 = all).")

    parser.add_argument('--wanda_group', type=str, default='output',
                        choices=['output', 'layer', 'global'],
                        help="WANDA's comparison group. 'output' ranks within each "
                             "output neuron (WANDA's headline setting); 'layer' uses "
                             "one threshold per layer. WANDA establishes 'output' for "
                             "LLMs and its Appendix A explicitly finds it does NOT "
                             "transfer to image classifiers, so on CIFAR this is a "
                             "design choice rather than a reproduction.")

    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help="L2 coefficient. EAST (TMLR 2025) uses 1e-4; GraNet uses "
                             "5e-4. Set it to match whichever published numbers the "
                             "run is meant to sit beside.")

    # Prunable scope (see pruning_factory.set_prunable_scope)
    parser.add_argument('--prune_task_head',   action='store_true',
                        help="Include the final classifier in the prunable set. "
                             "Standard in dynamic sparse training; the paper's "
                             "appendix D.4 instead keeps heads dense.")
    parser.add_argument('--prune_embeddings',  action='store_true',
                        help="Include token/position embeddings. Dangerous for "
                             "transformers: tied embeddings mean this also prunes "
                             "the LM head and dominates the sparsity denominator.")

    # Non-trainer args
    parser.add_argument('--log_to_wandb',        action='store_true', help="Log results to Weights & Biases (WANDB).")
    parser.add_argument('--seed',                type=int,       default=42)
    parser.add_argument('--val_split_seed',      type=int,       default=1234,
                        help="Train/val split seed, held fixed across runs so a 3-seed "
                             "error bar measures init and data-order variance only.")
    parser.add_argument('--deterministic',       action='store_true',
                        help="torch.use_deterministic_algorithms(True). Slower and errors "
                             "on ops lacking deterministic kernels; recorded either way.")

    return parser


def build_pruning_parser():
    parser = argparse.ArgumentParser(
        description="Prune a model using a specific pruning strategy"
    )

    # Required arguments
    parser.add_argument('--model_name',         type=str, choices=models,       required=True)
    parser.add_argument('--model_type',         type=str, choices=model_types,  required=True)
    parser.add_argument('--dataset_name',       type=str, choices=datasets,     required=True)
    parser.add_argument('--num_classes',        type=int,                       required=True)
    parser.add_argument('--trained_weights',    type=str,                       required=True)
    parser.add_argument('--pruning_type',       type=str,                       required=True)
    parser.add_argument('--target_sparsity',    type=float,                     required=True)
    parser.add_argument('--sparsity_scheduler', type=str,                       required=True)

    # Optional arguments with defaults
    parser.add_argument('--batch_size',      type=int,      default=512)
    parser.add_argument('--optimizer_type',  type=str,      default='sgd')
    parser.add_argument('--learning_rate',   type=float,    default=0.1)
    parser.add_argument('--image_size',      type=int,      default=32)
    parser.add_argument('--delta_T',         type=int,      default=100)

    parser.add_argument('--epochs',          type=int,      default=50)
    parser.add_argument('--recovery_epochs', type=int,      default=0)
    parser.add_argument('--scheduler_type',  type=str,      default=None)
    parser.add_argument('--patience',        type=int,      default=None)
    parser.add_argument('--experiment_type', type=str,      default='pruning')
    parser.add_argument('--log_epochs',      action='store_true', help="Enable detailed logging per epoch to the local logger.")
    parser.add_argument('--enable_tqdm',     action='store_true', help="Enable TQDM progress bars during training.")
    parser.add_argument('--databricks_env',  action='store_true', help="Set paths for Databricks environment (e.g., /dbfs).")
    parser.add_argument('--num_workers',     type=int,      default=os.cpu_count())

    # DyReLU phasing
    parser.add_argument('--dyrelu_en',         action='store_true')
    parser.add_argument('--dyrelu_phasing_en', action='store_true')

    # Weight Sharing
    parser.add_argument('--weight_sharing_en', action='store_true')

    parser.add_argument('--layerwise_alloc', type=str, default='erk',
                        choices=['erk', 'uniform'],
                        help="How the global sparsity budget is split across layers. "
                             "'erk' (default) matches RigL/EAST; 'uniform' gives every "
                             "layer the same density and is the floor arm of the "
                             "uniform-vs-ERK comparison. Only LocalMagnitudePrune reads it.")

    parser.add_argument('--limit_train_batches', type=int, default=0,
                        help="Truncate the train loader to N batches (0 = all). Tier-0 "
                             "smoke only; recorded on every run so a truncated result "
                             "can never be read as a measurement.")
    parser.add_argument('--limit_eval_batches',  type=int, default=0,
                        help="Truncate the val/test loaders to N batches (0 = all).")

    parser.add_argument('--wanda_group', type=str, default='output',
                        choices=['output', 'layer', 'global'],
                        help="WANDA's comparison group. 'output' ranks within each "
                             "output neuron (WANDA's headline setting); 'layer' uses "
                             "one threshold per layer. WANDA establishes 'output' for "
                             "LLMs and its Appendix A explicitly finds it does NOT "
                             "transfer to image classifiers, so on CIFAR this is a "
                             "design choice rather than a reproduction.")

    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help="L2 coefficient. EAST (TMLR 2025) uses 1e-4; GraNet uses "
                             "5e-4. Set it to match whichever published numbers the "
                             "run is meant to sit beside.")

    # Prunable scope (see pruning_factory.set_prunable_scope)
    parser.add_argument('--prune_task_head',   action='store_true',
                        help="Include the final classifier in the prunable set. "
                             "Standard in dynamic sparse training; the paper's "
                             "appendix D.4 instead keeps heads dense.")
    parser.add_argument('--prune_embeddings',  action='store_true',
                        help="Include token/position embeddings. Dangerous for "
                             "transformers: tied embeddings mean this also prunes "
                             "the LM head and dominates the sparsity denominator.")

    # Post-pruning fine-tune. Present so the I.P. baseline can be given the SAME
    # budget as BaCP: BaCP runs its contrastive phase and then a supervised
    # fine-tune, so without this the comparison is 110 epochs against 60 and the
    # objective cannot be separated from the extra training.
    parser.add_argument('--enable_finetune',     action='store_true',
                        help="Supervised fine-tune after pruning, mask frozen.")
    parser.add_argument('--optimizer_type_ft',   type=str,       default='adamw',
                        help="Optimizer for the fine-tuning phase.")
    parser.add_argument('--learning_rate_ft',    type=float,     default=1e-4,
                        help="Learning rate for the fine-tuning phase.")
    parser.add_argument('--epochs_ft',           type=int,       default=0,
                        help="Fine-tuning epochs. Match BaCP's epochs_ft for a "
                             "like-for-like comparison.")

    # Non-trainer args
    parser.add_argument('--log_to_wandb',        action='store_true', help="Log results to Weights & Biases (WANDB).")
    parser.add_argument('--seed',                type=int,       default=42)
    parser.add_argument('--val_split_seed',      type=int,       default=1234,
                        help="Train/val split seed, held fixed across runs so a 3-seed "
                             "error bar measures init and data-order variance only.")
    parser.add_argument('--deterministic',       action='store_true',
                        help="torch.use_deterministic_algorithms(True). Slower and errors "
                             "on ops lacking deterministic kernels; recorded either way.")

    return parser


def build_bacp_parser():
    parser = argparse.ArgumentParser(
        description="BaCP method for pruning a model using a specific pruning strategy"
    )

    # Required arguments
    parser.add_argument('--model_name',          type=str, choices=models,       required=True)
    parser.add_argument('--model_type',          type=str, choices=model_types,  required=True)
    parser.add_argument('--dataset_name',        type=str, choices=datasets,     required=True)
    parser.add_argument('--num_classes',         type=int,                       required=True)
    parser.add_argument('--trained_weights',     type=str,                       required=True)
    parser.add_argument('--pruning_type',        type=str,                       required=True)
    parser.add_argument('--target_sparsity',     type=float,                     required=True)
    parser.add_argument('--sparsity_scheduler',  type=str,                       required=True)
    parser.add_argument('--recovery_epochs',     type=int,                       required=True)
    
    # Optional arguments with defaults
    parser.add_argument('--batch_size',          type=int,       default=512)
    parser.add_argument('--tau',                 type=float,     default=0.07, help="Temperature parameter for InfoNCE loss (Contrastive Learning).")

    # Optimizer and lr for pretraining and fine tuning
    parser.add_argument('--optimizer_type',      type=str,       default='sgd', help="Optimizer for pre-training phase (Contrastive Loss).")
    parser.add_argument('--learning_rate',       type=float,     default=0.1, help="Learning rate for pre-training phase.")
    parser.add_argument('--epochs',              type=int,       default=50, help="Number of epochs for pre-training phase.")

    # Finetuning arguments
    parser.add_argument('--enable_finetune',     action='store_true', help="Enable supervised fine-tuning phase after pre-training.")
    parser.add_argument('--optimizer_type_ft',   type=str,       default='sgd', help="Optimizer for fine-tuning phase (Classification Loss).")
    parser.add_argument('--learning_rate_ft',    type=float,     default=0.005, help="Learning rate for fine-tuning phase.")
    parser.add_argument('--epochs_ft',           type=int,       default=100, help="Number of epochs for fine-tuning phase.")

    # Contrastive formulation (see loss_functions.cap_contrastive_loss).
    parser.add_argument('--contrastive_mode',    type=str, default='cap', choices=['cap', 'legacy'],
                        help="'cap': one CAP Eq.1 functional, rectangular student-anchor/"
                             "teacher-candidate similarities. 'legacy': the original "
                             "SupCon + NT-Xent sum over a square 2Bx2B matrix.")
    parser.add_argument('--proj_mode',           type=str, default='tied_frozen',
                        choices=['tied_frozen', 'none', 'current'],
                        help="'tied_frozen': one frozen projection head shared by student "
                             "and teachers. 'none': contrast pooled features directly "
                             "(CAP-faithful). 'current': per-model trainable heads (legacy).")
    parser.add_argument('--num_snapshots',       type=int,       default=2,
                        help="Max SnC snapshots retained; each costs a forward pass per batch.")

    # Loss weights. Order is lambda1=PrC, lambda2=FiC, lambda3=SnC, lambda4=CE --
    # see BaCPTrainer._combine_losses. Required for the CE-only control run
    # (--lambdas 0 0 0 1), which was not expressible from the CLI before.
    parser.add_argument('--lambdas',             type=float, nargs=4, default=None,
                        metavar=('PRC', 'FIC', 'SNC', 'CE'),
                        help="Four loss weights in the order PrC FiC SnC CE. "
                             "Defaults to 0.25 each. Use '0 0 0 1' for a CE-only control.")
    parser.add_argument('--learnable_lambdas',   action='store_true',
                        help="Make the four loss weights learnable parameters.")

    # Distillation controls (the KD comparison arms). Not part of BaCP -- these are
    # the arms that make its claim falsifiable, by asking whether ANY dense-teacher
    # signal helps in this regime before asking whether the contrastive one does.
    parser.add_argument('--distill_mode',   type=str, default='none',
                        choices=['none', 'kl', 'feature', 'both'],
                        help="'kl': Hinton response distillation on the fine-tuned "
                             "dense teacher's logits. 'feature': FitNets-style "
                             "matching on its embeddings. 'both': the two summed.")
    parser.add_argument('--lambda_kd',      type=float, default=0.0,
                        help="Weight on the distillation term. Deliberately OUTSIDE "
                             "the four-lambda simplex, so switching distillation on "
                             "does not also rescale the contrastive terms.")
    parser.add_argument('--kd_temperature', type=float, default=4.0)
    parser.add_argument('--feature_distill_metric', type=str, default='cosine',
                        choices=['cosine', 'mse'])
    parser.add_argument('--n_views',        type=int, default=2,
                        help="Augmented views per step. Rung C0b sets this against a "
                             "CE-only objective to separate 'contrast helps' from "
                             "'two views per step helps'.")

    parser.add_argument('--image_size',          type=int,       default=32)
    parser.add_argument('--scheduler_type',      type=str,       default=None)
    parser.add_argument('--patience',            type=int,       default=None)
    parser.add_argument('--experiment_type',     type=str,       default='bacp')
    
    # Boolean flags corrected to use action='store_true' (default=False)
    parser.add_argument('--log_epochs',          action='store_true', help="Enable detailed logging per epoch to the local logger.")
    parser.add_argument('--enable_tqdm',         action='store_true', help="Enable TQDM progress bars during training.")
    parser.add_argument('--databricks_env',      action='store_true', help="Set paths for Databricks environment (e.g., /dbfs).")
    parser.add_argument('--num_workers',         type=int,       default=os.cpu_count())

    # DyReLU phasing
    parser.add_argument('--dyrelu_en',           action='store_true', help="Enable DyReLU activation function.")
    parser.add_argument('--dyrelu_phasing_en',   action='store_true', help="Enable phasing mechanism for DyReLU.")

    # EAST weight sharing. Previously accepted by the pruning/baseline parsers but
    # not this one, even though _initialize_pruner reads args.weight_sharing_en on
    # every path -- so BaCP runs raised AttributeError during __post_init__.
    parser.add_argument('--weight_sharing_en',   action='store_true', help="Enable EAST weight sharing across residual blocks.")

    # Mask-update interval in optimizer steps. Read by _initialize_pruner and
    # _step_pruning_step; same missing-field story as above.
    parser.add_argument('--delta_T',             type=int,       default=100, help="Steps between pruner mask updates.")

    parser.add_argument('--layerwise_alloc', type=str, default='erk',
                        choices=['erk', 'uniform'],
                        help="How the global sparsity budget is split across layers. "
                             "'erk' (default) matches RigL/EAST; 'uniform' gives every "
                             "layer the same density and is the floor arm of the "
                             "uniform-vs-ERK comparison. Only LocalMagnitudePrune reads it.")

    parser.add_argument('--limit_train_batches', type=int, default=0,
                        help="Truncate the train loader to N batches (0 = all). Tier-0 "
                             "smoke only; recorded on every run so a truncated result "
                             "can never be read as a measurement.")
    parser.add_argument('--limit_eval_batches',  type=int, default=0,
                        help="Truncate the val/test loaders to N batches (0 = all).")

    parser.add_argument('--wanda_group', type=str, default='output',
                        choices=['output', 'layer', 'global'],
                        help="WANDA's comparison group. 'output' ranks within each "
                             "output neuron (WANDA's headline setting); 'layer' uses "
                             "one threshold per layer. WANDA establishes 'output' for "
                             "LLMs and its Appendix A explicitly finds it does NOT "
                             "transfer to image classifiers, so on CIFAR this is a "
                             "design choice rather than a reproduction.")

    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help="L2 coefficient. EAST (TMLR 2025) uses 1e-4; GraNet uses "
                             "5e-4. Set it to match whichever published numbers the "
                             "run is meant to sit beside.")

    # Prunable scope (see pruning_factory.set_prunable_scope)
    parser.add_argument('--prune_task_head',     action='store_true',
                        help="Include the final classifier in the prunable set. Standard "
                             "in dynamic sparse training; the paper's appendix D.4 instead "
                             "keeps heads dense. Changes what the sparsity figure denotes.")
    parser.add_argument('--prune_embeddings',    action='store_true',
                        help="Include token/position embeddings. Dangerous for transformers: "
                             "tied embeddings mean this also prunes the LM head, and the "
                             "embedding table then dominates the sparsity denominator.")

    # Non-trainer args
    parser.add_argument('--log_to_wandb',        action='store_true', help="Log results to Weights & Biases (WANDB).")
    parser.add_argument('--seed',                type=int,       default=42)
    parser.add_argument('--val_split_seed',      type=int,       default=1234,
                        help="Train/val split seed, held fixed across runs so a 3-seed "
                             "error bar measures init and data-order variance only.")
    parser.add_argument('--deterministic',       action='store_true',
                        help="torch.use_deterministic_algorithms(True). Slower and errors "
                             "on ops lacking deterministic kernels; recorded either way.")

    return parser








def baseline_parse_args(argv=None):
    return build_baseline_parser().parse_args(argv)


def pruning_parse_args(argv=None):
    return build_pruning_parser().parse_args(argv)


def bacp_parse_args(argv=None):
    return build_bacp_parser().parse_args(argv)
