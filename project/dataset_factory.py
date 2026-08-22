from torchvision.datasets import  CIFAR10, SVHN, MNIST, FashionMNIST, Food101, Flowers102, CIFAR100, EMNIST, Caltech101, ImageNet
import torchvision.transforms as T
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
import numpy as np
import torch
from utils import *
from functools import lru_cache
import inspect

# `datasets` and `transformers` are imported lazily inside the text loaders below.
# At module scope they cost ~4s of import time on every run -- including every
# pytest session -- while being unused on the entire CV path.
from copy import deepcopy


# VALID_DATASETS = {
#     'contrastive': {
#         'cifar10': lambda root_folder, size, n_views: CIFAR10(root_folder, train=True, transform=AugmentData(get_train_transform('contrastive', size, 'cifar10'), n_views), download=True),
#         'cifar100': lambda root_folder, size, n_views: CIFAR100(root_folder, train=True, transform=AugmentData(get_train_transform('contrastive', size, 'cifar100'), n_views), download=True),
#         'svhn': lambda root_folder, size, n_views: SVHN(root_folder, split='train', transform=AugmentData(get_train_transform('contrastive', size, 'svhn'), n_views), download=True),
#         'mnist': lambda root_folder, size, n_views: MNIST(root_folder, train=True, transform=AugmentData(get_train_transform('contrastive', size, 'mnist'), n_views), download=True),
#         'fmnist': lambda root_folder, size, n_views: FashionMNIST(root_folder, train=True, transform=AugmentData(get_train_transform('contrastive', size, 'fmnist'), n_views), download=True),
#         'food101': lambda root_folder, size, n_views: Food101(root_folder, split='train', transform=AugmentData(get_train_transform('contrastive', size, 'food101'), n_views), download=True),
#         'flowers102': lambda root_folder, size, n_views: Flowers102(root_folder, split='train', transform=AugmentData(get_train_transform('contrastive', size, 'flowers102'), n_views), download=True),
#         'emnist': lambda root_folder, size, n_views: EMNIST(root_folder, split='balanced', train=True, transform=AugmentData(get_train_transform('contrastive', size, 'emnist'), n_views), download=True),
#         'caltech101': lambda root_folder, size, n_views: Caltech101(root=root_folder, target_type='category', transform=AugmentData(get_train_transform('contrastive', size, 'caltech101'), n_views), download=True),
#     },
    
#     'supervised': {
#         'cifar10': lambda root_folder, size, n_views: CIFAR10(root_folder, train=True, transform=AugmentData(get_train_transform('supervised', size, 'cifar10'), n_views), download=True),
#         'cifar100': lambda root_folder, size, n_views: CIFAR100(root_folder, train=True, transform=AugmentData(get_train_transform('supervised', size, 'cifar100'), n_views), download=True),
#         'svhn': lambda root_folder, size, n_views: SVHN(root_folder, split='train', transform=AugmentData(get_train_transform('supervised', size, 'svhn'), n_views), download=True),
#         'mnist': lambda root_folder, size, n_views: MNIST(root_folder, train=True, transform=AugmentData(get_train_transform('supervised', size, 'mnist'), n_views), download=True),
#         'fmnist': lambda root_folder, size, n_views: FashionMNIST(root_folder, train=True, transform=AugmentData(get_train_transform('supervised', size, 'fmnist'), n_views), download=True),
#         'food101': lambda root_folder, size, n_views: Food101(root_folder, split='train', transform=AugmentData(get_train_transform('supervised', size, 'food101'), n_views), download=True),
#         'flowers102': lambda root_folder, size, n_views: Flowers102(root_folder, split='train', transform=AugmentData(get_train_transform('supervised', size, 'flowers102'), n_views), download=True),
#         'emnist': lambda root_folder, size, n_views: EMNIST(root_folder, split='balanced', train=True, transform=AugmentData(get_train_transform('supervised', size, 'emnist'), n_views), download=True),
#         'caltech101': lambda root_folder, size, n_views: Caltech101(root=root_folder, target_type='category', transform=AugmentData(get_train_transform('supervised', size, 'caltech101'), n_views), download=True),
#     },
    
#     'testset': {
#         'cifar10': lambda root_folder, size: CIFAR10(root_folder, train=False, transform=get_eval_transform('cifar10', size), download=True),
#         'cifar100': lambda root_folder, size: CIFAR100(root_folder, train=False, transform=get_eval_transform('cifar100', size), download=True),
#         'svhn': lambda root_folder, size: SVHN(root_folder, split='test', transform=get_eval_transform('svhn', size), download=True),
#         'mnist': lambda root_folder, size: MNIST(root_folder, train=False, transform=get_eval_transform('mnist', size), download=True),
#         'fmnist': lambda root_folder, size: FashionMNIST(root_folder, train=False, transform=get_eval_transform('fmnist', size), download=True),
#         'food101': lambda root_folder, size: Food101(root_folder, split='test', transform=get_eval_transform('food101', size), download=True),
#         'flowers102': lambda root_folder, size: Flowers102(root_folder, split='test', transform=get_eval_transform('flowers102', size), download=True),
#         'emnist': lambda root_folder, size: EMNIST(root_folder, split='balanced', train=False, transform=get_eval_transform('emnist', size), download=True),
#         'caltech101': lambda root_folder, size: Caltech101(root=root_folder, target_type='category', transform=get_eval_transform('caltech101', size), download=True),
#     },
# }

CV_DATASETS = {
    'cifar10': CIFAR10, 
    'cifar100': CIFAR100,
    'svhn': SVHN,
    'mnist': MNIST,
    'fmnist': FashionMNIST,
    'emnist': EMNIST,
    'imagenet': ImageNet,
}

GRAYSCALE_DATASETS = ["mnist", "fmnist", "emnist"]

# Fixed seed for the train/val split, independent of --seed. See load_cv_datasets
# for why this must not track the run seed.
VAL_SPLIT_SEED = 1234

# Text datasets. Kept separate from CV_DATASETS because they need a tokenizer and
# a collator rather than an image transform.
#   sst2      -- GLUE sentence-level sentiment, 2 classes
#   wikitext2 -- masked language modelling; DataCollatorForLanguageModeling
#                supplies the -100 label masking that Trainer._handle_metrics and
#                BaCPTrainer._handle_metrics both branch on.
# Dataset sources:
#   CIFAR-10/100  Krizhevsky, "Learning Multiple Layers of Features from Tiny
#                 Images", 2009 (tech report). Via torchvision.
#   MNIST         LeCun et al. 1998; FashionMNIST Xiao et al. 2017
#                 (arXiv 1708.07747); EMNIST Cohen et al. 2017
#                 (arXiv 1702.05373); SVHN Netzer et al. 2011. Via torchvision.
#   SST-2         Socher et al. 2013 (EMNLP), distributed as part of GLUE
#                 (Wang et al. 2019, ICLR, arXiv 1804.07461). NOTE: GLUE's test
#                 split is unlabelled, so the "test" loader below is the
#                 validation set -- state that wherever accuracy is reported.
#   WikiText-2    Merity et al. 2017 (ICLR, arXiv 1609.07843).
TEXT_DATASETS = {
    'sst2':      ('glue', 'sst2'),
    'wikitext2': ('wikitext', 'wikitext-2-raw-v1'),
}

# Class counts, so --num_classes cannot silently disagree with the data. EMNIST is
# loaded with split='balanced' (47 classes), matching "EMNIST (Balanced)" in the
# paper. wikitext2 is absent on purpose: MLM predicts over the tokenizer vocab, so
# there is no fixed class count.
DATASET_NUM_CLASSES = {
    'cifar10': 10,
    'cifar100': 100,
    'mnist': 10,
    'fmnist': 10,
    'emnist': 47,
    'svhn': 10,
    'imagenet': 1000,
    'sst2': 2,
}
DATASET_STATS = {
    "cifar10": ([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    "cifar100": ([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
    "svhn": ([0.4380, 0.4440, 0.4730], [0.1751, 0.1771, 0.1744]),
    "mnist": ([0.1307], [0.3081]),
    "fmnist": ([0.2860], [0.3530]),
    "emnist": ([0.1307], [0.3081]),
    "imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    } 


class AugmentData(object):
    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        if self.n_views == 1:
            return self.base_transform(x)
        else:
            return [self.base_transform(x) for _ in range(self.n_views)]


def normalize_data(dataset_name):
    mean, std = DATASET_STATS[dataset_name]
    if dataset_name in GRAYSCALE_DATASETS:
        return [
            # T.Grayscale, not T.Lambda(lambda x: x.repeat(3, 1, 1)). A lambda
            # closure cannot be pickled, so the transform could not be shipped to a
            # DataLoader worker under spawn-based multiprocessing (Windows, macOS) --
            # it only ever worked because the Databricks environment forks. Applied
            # before ToTensor so it operates on the PIL image.
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(mean*3, std*3)
        ]
    else:
        return [
            T.ToTensor(),
            T.Normalize(mean, std)
        ]
    

def get_train_transform(dataset_name: str, t_type: str, size: int):
    kernel_size = int(0.1 * size)
    if kernel_size % 2 == 0:
        kernel_size += 1

    # CIFAR-10 Augmentation
    if dataset_name == 'cifar10':
        if t_type == 'supervised':
            transform = [
                T.Resize((size, size)),
                T.RandomCrop(size, padding=4),
                T.RandomHorizontalFlip(),
            ]
        elif t_type == 'contrastive':
            transform = [
                T.Resize((size, size)), 
                T.RandomResizedCrop(size=size, scale=(0.2, 1.0)), 
                T.RandomHorizontalFlip(), 
                T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8), 
                T.RandomGrayscale(p=0.2), 
                T.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0)), 
                # AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
            ]
    elif dataset_name == 'imagenet':
        if t_type == 'supervised':
            transform = [
                T.RandomResizedCrop(size),
                T.RandomHorizontalFlip(),
            ]
        elif t_type == 'contrastive':
            transform = [
                T.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
                T.RandomHorizontalFlip(),
                T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0)),
            ]
    else:
        # Everything else reuses the CIFAR-style recipe (resize + crop + flip, plus
        # the SimCLR colour/blur stack for the contrastive variant). Previously this
        # branch raised, which made five of the seven datasets registered in
        # CV_DATASETS -- cifar100, svhn, mnist, fmnist, emnist -- unreachable, even
        # though cifar100 is accepted by the CLI and the paper reports results on
        # several of them.
        if t_type == 'supervised':
            transform = [
                T.Resize((size, size)),
                T.RandomCrop(size, padding=4),
                T.RandomHorizontalFlip(),
            ]
        elif t_type == 'contrastive':
            transform = [
                T.Resize((size, size)),
                T.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
                T.RandomHorizontalFlip(),
                T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0)),
            ]
        else:
            raise ValueError(f"Unsupported t_type: {t_type}")


    # Mean and STD normalization
    transform.extend(normalize_data(dataset_name))
    return T.Compose(transform)


def get_test_transform(dataset_name: str, size: int):
    if dataset_name == 'imagenet':
        # ImageNet standard eval: resize shorter side to 256 then center crop to size
        transform = [
            T.Resize(256),
            T.CenterCrop(size),
        ]
    else:
        transform = [T.Resize((size, size))]
    transform.extend(normalize_data(dataset_name))
    return T.Compose(transform)


def get_dataset_train_fn(
    dataset_class:type,
    dataset_name: str,
    t_type:       str,
    size:         int,
    n_views:      int,
    cache_dir:    str = './cache',
    ):
    sig = inspect.signature(dataset_class.__init__)
    dataset_args = {
        'root': cache_dir,
        'transform': AugmentData(
            get_train_transform(dataset_name, t_type, size), n_views
            ),
        'download': True
    }

    if 'train' in sig.parameters:
        dataset_args['train'] = True
    if 'split' in sig.parameters:
        dataset_args['split'] = 'train'
    if dataset_name == 'emnist':
        dataset_args['split'] = 'balanced'

    return lambda: dataset_class(**dataset_args)


def get_dataset_test_fn(
    dataset_class:type,
    dataset_name: str,
    size:         int,
    cache_dir:    str = './cache',
    ):
    sig = inspect.signature(dataset_class.__init__)
    dataset_args = {
        'root': cache_dir,
        'transform': get_test_transform(dataset_name, size),
        'download': True
    }
    if 'train' in sig.parameters:
        dataset_args['train'] = False
    if 'split' in sig.parameters:
        dataset_args['split'] = 'test'
    if dataset_name == 'emnist':
        dataset_args['split'] = 'balanced'
    if dataset_name == 'imagenet':
        dataset_args['split'] = 'val'

    return lambda: dataset_class(**dataset_args)


def get_dataset_fn(
    dataset_name: str, 
    t_type:       str, 
    size:         int, 
    n_views:      int, 
    cache_dir:    str = './cache',
    ):
    if dataset_name not in CV_DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if t_type not in ['contrastive', 'supervised']:
        raise ValueError(f"Unsupported transformation type: {t_type}")

    dataset_class = CV_DATASETS[dataset_name]

    train_dataset_fn = get_dataset_train_fn(
        dataset_class, dataset_name, t_type, size, n_views, cache_dir
        )

    test_dataset_fn = get_dataset_test_fn(
        dataset_class, dataset_name, size, cache_dir
        )

    return train_dataset_fn, test_dataset_fn


def load_cv_datasets(
    dataset_name:   str,
    t_type:         str,
    size:           int,
    n_views:        int,
    cache_dir:      str = './cache',
    val_split_seed: int = VAL_SPLIT_SEED,
    ):

    # Creating datasets
    train_dataset_fn, test_dataset_fn = get_dataset_fn(
        dataset_name, t_type, size, n_views, cache_dir
        )

    # Loading train and test data
    trainset = train_dataset_fn()
    testset = test_dataset_fn()

    if t_type == 'supervised':
        # Creating validation split
        train_size = len(trainset)
        val_size = int(0.20 * train_size)
        train_size = train_size - val_size

        # Seeded, and deliberately NOT from args.seed. Two reasons:
        #
        #   1. Without an explicit generator this consumed the *global* torch RNG,
        #      whose state at this point depends on how much randomness model
        #      construction used upstream (_initialize_models runs before
        #      _initialize_data_loaders). Changing the backbone therefore changed
        #      the validation split.
        #   2. Holding the split fixed across seeds is what makes a 3-seed error bar
        #      mean "initialisation and data-order variance". If the split moved with
        #      the seed, the spread would also contain split variance and early
        #      stopping would be selecting on different images per seed.
        #
        # Pass val_split_seed explicitly only for a deliberate split-robustness check.
        generator = torch.Generator().manual_seed(val_split_seed)
        trainset, valset = random_split(trainset, [train_size, val_size], generator=generator)

        valset = deepcopy(valset)
        valset.dataset.transform = get_test_transform(dataset_name, size)
    else:
        valset = None

    return {
        'train': trainset,
        'validation': valset,
        'test': testset,
    }


def load_cv_dataloaders(
    dataset_name:   str,
    t_type:         str,
    batch_size:     int,
    size:           int,
    n_views:        int,
    num_workers:    int,
    cache_dir:      str = './cache',
    val_split_seed: int = VAL_SPLIT_SEED,
    seed:           int = 0,
    ):
    try:
        datasets = load_cv_datasets(dataset_name, t_type, size, n_views, cache_dir,
                                    val_split_seed=val_split_seed)

        # Creating dataloaders. drop_last differs by split, which is not incidental:
        # it used to be True everywhere, so CIFAR-10's 10,000-image test set was
        # evaluated over floor(10000/512)*512 = 9,728 images. 272 test images were
        # silently never scored, and *which* 272 depended on the batch size -- so
        # changing batch size changed the reported accuracy.
        trainloader = DataLoader(datasets["train"], shuffle=True,
                                 **_loader_args(batch_size, num_workers,
                                                drop_last=True, seed=seed, salt=0))
        testloader = DataLoader(datasets["test"], shuffle=False,
                                **_loader_args(batch_size, num_workers,
                                               drop_last=False, seed=seed, salt=1))

        if datasets["validation"] is not None:
            valloader = DataLoader(datasets["validation"], shuffle=False,
                                   **_loader_args(batch_size, num_workers,
                                                  drop_last=False, seed=seed, salt=2))
        else:
            valloader = None

        return {
            "trainloader": trainloader,
            "valloader": valloader,
            "testloader": testloader,
        }

    except Exception as e:
        print(f"Error loading CV dataset {dataset_name}: {str(e)}")
        raise e


# --- Text datasets --------------------------------------------------------
#

def _seed_worker(worker_id):
    """Seed `random` and `numpy` inside each DataLoader worker.

    PyTorch derives each worker's *torch* seed from the parent generator, but leaves
    `random` and `numpy.random` unseeded. Any augmentation reaching for those is
    therefore not reproducible, and `num_workers` defaults to `os.cpu_count()`, so
    the number of workers -- and hence the sequence -- is machine-dependent.
    """
    import random as _random
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    _random.seed(worker_seed)


def _loader_args(batch_size, num_workers, drop_last=True, seed=0, salt=0):
    """DataLoader kwargs.

    drop_last defaults True for the train loader's benefit, but callers MUST pass
    drop_last=False for val and test: dropping a partial eval batch means never
    scoring those samples, and which samples get dropped depends on the batch size.

    The per-loader generator is salted so train/val/test do not share a shuffle
    stream, while staying deterministic given (seed, salt).
    """
    generator = torch.Generator()
    generator.manual_seed((int(seed) * 1000003) ^ int(salt))
    return {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
        "drop_last": drop_last,
        "worker_init_fn": _seed_worker,
        "generator": generator,
    }


class TokenizedDataset(Dataset):
    """Thin wrapper yielding the three keys the training loop expects.

    Needed because _handle_data_to_device recognises a dict batch by its keys, and
    the HF Dataset's own extra columns (idx, sentence, ...) would ride along.
    """

    def __init__(self, hf_split):
        self.data = hf_split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        return {
            "input_ids": example["input_ids"],
            "attention_mask": example["attention_mask"],
            "labels": example["labels"],
        }


def get_tokenizer(model_name, cache_dir=None):
    """Tokenizer for a registered language model.

    Strips the '-mlm' suffix: 'roberta-base-mlm' and 'roberta-base' are the same
    checkpoint with different heads, and only the former exists on the hub as
    'roberta-base'.
    """
    from transformers import AutoTokenizer
    checkpoint = model_name[:-4] if model_name.endswith('-mlm') else model_name
    return AutoTokenizer.from_pretrained(checkpoint, cache_dir=cache_dir)


def load_glue_dataloaders(tokenizer, task_name, batch_size, num_workers,
                          cache_dir=None, max_length=128, seed=0):
    """GLUE sequence classification. Only sst2 is used by this project."""
    from datasets import load_dataset

    if task_name not in ('sst2', 'mnli', 'qqp'):
        raise ValueError(f"Unsupported GLUE task: {task_name}")

    dataset = load_dataset('glue', task_name, cache_dir=cache_dir)

    text_keys = {'sst2': ('sentence',),
                 'mnli': ('premise', 'hypothesis'),
                 'qqp': ('question1', 'question2')}[task_name]

    def tokenize_fn(batch):
        return tokenizer(*[batch[k] for k in text_keys], truncation=True,
                         padding='max_length', max_length=max_length)

    dataset = dataset.map(tokenize_fn, batched=True, batch_size=512)
    dataset = dataset.rename_column('label', 'labels')
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # GLUE test splits are unlabelled (label == -1), so validation is used for both
    # validation and final evaluation -- which is also what every paper in this
    # space reports, CAP included ("results on development sets").
    # drop_last=False on the eval loader: SST-2's dev set is 872 examples, so at
    # batch 32 a drop_last=True loader would score 864 and silently discard 8.
    trainloader = DataLoader(TokenizedDataset(dataset['train']), shuffle=True,
                             **_loader_args(batch_size, num_workers,
                                            drop_last=True, seed=seed, salt=0))
    valloader = DataLoader(TokenizedDataset(dataset['validation']), shuffle=False,
                           **_loader_args(batch_size, num_workers,
                                          drop_last=False, seed=seed, salt=2))

    return {"trainloader": trainloader, "valloader": valloader, "testloader": valloader}


def load_mlm_dataloaders(tokenizer, batch_size, num_workers, cache_dir=None,
                         block_size=None, mlm_probability=0.15):
    """WikiText-2 masked language modelling."""
    from datasets import load_dataset
    from transformers import DataCollatorForLanguageModeling

    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', cache_dir=cache_dir)

    # Cap the block size: tokenizer.model_max_length is 512 for RoBERTa but a
    # sentinel ~1e30 for some tokenizers, which would make grouping allocate
    # unboundedly.
    limit = getattr(tokenizer, 'model_max_length', 512)
    block_size = block_size or (limit if limit and limit < 4096 else 512)

    def tokenize_fn(batch):
        return tokenizer(batch['text'], truncation=False,
                         return_special_tokens_mask=True)

    def group_texts(batch):
        joined = {k: sum(batch[k], []) for k in batch}
        total = (len(joined['input_ids']) // block_size) * block_size
        return {k: [v[i:i + block_size] for i in range(0, total, block_size)]
                for k, v in joined.items()}

    dataset = dataset.map(tokenize_fn, batched=True, batch_size=512,
                          remove_columns=['text'])
    dataset = dataset.map(group_texts, batched=True, batch_size=512)

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=True,
                                              mlm_probability=mlm_probability)

    def loader(split, shuffle, drop_last, salt):
        return DataLoader(dataset[split], shuffle=shuffle, collate_fn=collator,
                          **_loader_args(batch_size, num_workers,
                                         drop_last=drop_last, seed=seed, salt=salt))

    return {
        "trainloader": loader('train', True, True, 0),
        "valloader": loader('validation', False, False, 2),
        "testloader": loader('test', False, False, 1),
    }


def load_text_dataloaders(model_name, dataset_name, batch_size, num_workers,
                          cache_dir=None, seed=0):
    if dataset_name not in TEXT_DATASETS:
        raise ValueError(f"Unsupported text dataset: {dataset_name}. "
                         f"Choices: {sorted(TEXT_DATASETS)}")

    tokenizer = get_tokenizer(model_name, cache_dir)

    if dataset_name == 'wikitext2':
        return load_mlm_dataloaders(tokenizer, batch_size, num_workers, cache_dir, seed=seed)
    return load_glue_dataloaders(tokenizer, dataset_name, batch_size, num_workers,
                                 cache_dir, seed=seed)


def get_dataloaders(args, supervised_override=False):
    """Returns Dataloaders.

    supervised_override forces the single-view supervised recipe regardless of
    is_bacp. BaCPTrainer.finetune() needs that: it rebuilds loaders for the
    supervised fine-tuning phase, and previously called load_cv_dataloaders
    directly -- which raised for sst2/wikitext2, so BaCP plus a language model plus
    --enable_finetune had no working code path at all.
    """
    is_bacp = getattr(args, 'is_bacp', False) and not supervised_override

    if args.model_type == 'cv':
        return load_cv_dataloaders(
            dataset_name=args.dataset_name,
            t_type='contrastive' if is_bacp else 'supervised',
            batch_size=args.batch_size,
            size=args.image_size,
            n_views=args.n_views if is_bacp else 1,
            num_workers=args.num_workers,
            cache_dir=args.cache_dir,
            val_split_seed=getattr(args, 'val_split_seed', VAL_SPLIT_SEED),
            seed=getattr(args, 'seed', 0),
            )

    if args.model_type == 'llm':
        # No text augmentation, so there is no second view to build. That is not a
        # gap: in CAP the two "views" of a text example are the same tokens seen
        # through two different models, which is exactly what the cross-model
        # contrastive term needs. BaCPTrainer duplicates the batch accordingly.
        return load_text_dataloaders(
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            cache_dir=args.cache_dir,
            seed=getattr(args, 'seed', 0),
            )

    raise ValueError(f"Unsupported model type: {args.model_type!r}. Expected 'cv' or 'llm'.")









        


        
