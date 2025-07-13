from torchvision.datasets import  CIFAR10, SVHN, MNIST, FashionMNIST, Food101, Flowers102, CIFAR100, EMNIST, Caltech101
import torchvision.transforms as T
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
from datasets import load_dataset 
from transformers import DataCollatorForLanguageModeling
from utils import *
from constants import *
from functools import lru_cache


VALID_DATASETS = {
    'contrastive': {
        'cifar10': lambda root_folder, size, n_views: CIFAR10(root_folder, train=True, transform=AugmentData(get_train_transform('contrastive', size, 'cifar10'), n_views), download=True),
        'cifar100': lambda root_folder, size, n_views: CIFAR100(root_folder, train=True, transform=AugmentData(get_train_transform('contrastive', size, 'cifar100'), n_views), download=True),
        'svhn': lambda root_folder, size, n_views: SVHN(root_folder, split='train', transform=AugmentData(get_train_transform('contrastive', size, 'svhn'), n_views), download=True),
        'mnist': lambda root_folder, size, n_views: MNIST(root_folder, train=True, transform=AugmentData(get_train_transform('contrastive', size, 'mnist'), n_views), download=True),
        'fmnist': lambda root_folder, size, n_views: FashionMNIST(root_folder, train=True, transform=AugmentData(get_train_transform('contrastive', size, 'fmnist'), n_views), download=True),
        'food101': lambda root_folder, size, n_views: Food101(root_folder, split='train', transform=AugmentData(get_train_transform('contrastive', size, 'food101'), n_views), download=True),
        'flowers102': lambda root_folder, size, n_views: Flowers102(root_folder, split='train', transform=AugmentData(get_train_transform('contrastive', size, 'flowers102'), n_views), download=True),
        'emnist': lambda root_folder, size, n_views: EMNIST(root_folder, split='balanced', train=True, transform=AugmentData(get_train_transform('contrastive', size, 'emnist'), n_views), download=True),
        'caltech101': lambda root_folder, size, n_views: Caltech101(root=root_folder, target_type='category', transform=AugmentData(get_train_transform('contrastive', size, 'caltech101'), n_views), download=True),
    },
    
    'supervised': {
        'cifar10': lambda root_folder, size, n_views: CIFAR10(root_folder, train=True, transform=AugmentData(get_train_transform('supervised', size, 'cifar10'), n_views), download=True),
        'cifar100': lambda root_folder, size, n_views: CIFAR100(root_folder, train=True, transform=AugmentData(get_train_transform('supervised', size, 'cifar100'), n_views), download=True),
        'svhn': lambda root_folder, size, n_views: SVHN(root_folder, split='train', transform=AugmentData(get_train_transform('supervised', size, 'svhn'), n_views), download=True),
        'mnist': lambda root_folder, size, n_views: MNIST(root_folder, train=True, transform=AugmentData(get_train_transform('supervised', size, 'mnist'), n_views), download=True),
        'fmnist': lambda root_folder, size, n_views: FashionMNIST(root_folder, train=True, transform=AugmentData(get_train_transform('supervised', size, 'fmnist'), n_views), download=True),
        'food101': lambda root_folder, size, n_views: Food101(root_folder, split='train', transform=AugmentData(get_train_transform('supervised', size, 'food101'), n_views), download=True),
        'flowers102': lambda root_folder, size, n_views: Flowers102(root_folder, split='train', transform=AugmentData(get_train_transform('supervised', size, 'flowers102'), n_views), download=True),
        'emnist': lambda root_folder, size, n_views: EMNIST(root_folder, split='balanced', train=True, transform=AugmentData(get_train_transform('supervised', size, 'emnist'), n_views), download=True),
        'caltech101': lambda root_folder, size, n_views: Caltech101(root=root_folder, target_type='category', transform=AugmentData(get_train_transform('supervised', size, 'caltech101'), n_views), download=True),

    },
    
    'testset': {
        'cifar10': lambda root_folder, size: CIFAR10(root_folder, train=False, transform=get_eval_transform('cifar10', size), download=True),
        'cifar100': lambda root_folder, size: CIFAR100(root_folder, train=False, transform=get_eval_transform('cifar100', size), download=True),
        'svhn': lambda root_folder, size: SVHN(root_folder, split='test', transform=get_eval_transform('svhn', size), download=True),
        'mnist': lambda root_folder, size: MNIST(root_folder, train=False, transform=get_eval_transform('mnist', size), download=True),
        'fmnist': lambda root_folder, size: FashionMNIST(root_folder, train=False, transform=get_eval_transform('fmnist', size), download=True),
        'food101': lambda root_folder, size: Food101(root_folder, split='test', transform=get_eval_transform('food101', size), download=True),
        'flowers102': lambda root_folder, size: Flowers102(root_folder, split='test', transform=get_eval_transform('flowers102', size), download=True),
        'emnist': lambda root_folder, size: EMNIST(root_folder, split='balanced', train=False, transform=get_eval_transform('emnist', size), download=True),
        'caltech101': lambda root_folder, size: Caltech101(root=root_folder, target_type='category', transform=get_eval_transform('caltech101', size), download=True),
    },

    'unaugmented': {
        'cifar10': lambda root_folder, size: CIFAR10(root_folder, train=True, transform=get_eval_transform('cifar10', size), download=True),
        'cifar100': lambda root_folder, size: CIFAR100(root_folder, train=True, transform=get_eval_transform('cifar100', size), download=True),
        'svhn': lambda root_folder, size: SVHN(root_folder, split='train', transform=get_eval_transform('svhn', size), download=True),
        'mnist': lambda root_folder, size: MNIST(root_folder, train=True, transform=get_eval_transform('mnist', size), download=True),
        'fmnist': lambda root_folder, size: FashionMNIST(root_folder, train=True, transform=get_eval_transform('fmnist', size), download=True),
        'food101': lambda root_folder, size: Food101(root_folder, split='train', transform=get_eval_transform('food101', size), download=True),
        'flowers102': lambda root_folder, size: Flowers102(root_folder, split='train', transform=get_eval_transform('flowers102', size), download=True),
        'emnist': lambda root_folder, size: EMNIST(root_folder, split='balanced', train=True, transform=get_eval_transform('emnist', size), download=True),
        'caltech101': lambda root_folder, size: Caltech101(root=root_folder, target_type='category', transform=get_eval_transform('caltech101', size), download=True),
    },
}

CV_DATASETS = ['cifar10', 'svhn', 'mnist', 'fmnist', 'emnist', 'food101', 'flowers102', 'cifar100', 'caltech101']
DATASET_STATS = {
    "cifar10": ([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    "cifar100": ([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
    "svhn": ([0.4380, 0.4440, 0.4730], [0.1751, 0.1771, 0.1744]),
    "mnist": ([0.1307], [0.3081]),
    "fmnist": ([0.2860], [0.3530]),
    "emnist": ([0.1307], [0.3081]),
    "food101": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "flowers102": ([0.4349, 0.3836, 0.2968], [0.2963, 0.2458, 0.2686]),
    "caltech101": ([0.5459, 0.5222, 0.4782], [0.2484, 0.2431, 0.2466]),
    } 
GRAYSCALE_DATASETS = {"mnist", "fmnist", "emnist"}

class AugmentData(object):
    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views
    
    def __call__(self, x):
        if self.n_views == 1:
            return self.base_transform(x)
        else:
            return [self.base_transform(x) for _ in range(self.n_views)]

def normalize_data(dataset_name, mean, std):
    if dataset_name in GRAYSCALE_DATASETS:
        return [
            T.ToTensor(),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
            T.Normalize(mean*3, std*3)
        ]
    else:
        return [
            T.ToTensor(),
            T.Normalize(mean, std)
        ]

def get_eval_transform(dataset_name, size=32):
    mean, std = DATASET_STATS[dataset_name]
    transforms = [
        T.Resize((size, size)),
    ]
    transforms.extend(normalize_data(dataset_name, mean, std))
    return T.Compose(transforms)

def get_train_transform(learning_type, size=32, dataset_name="", s=1):
    def get_supervised_transform(dataset_name, size):
        mean, std = DATASET_STATS[dataset_name]
        if dataset_name == 'cifar10' or dataset_name == 'cifar100':
            transforms = [
                T.Resize((size, size)),
                T.RandomCrop(size, padding=4),
                T.RandomHorizontalFlip(),
            ]
        elif dataset_name == 'svhn':
            transforms = [
                T.Resize((size, size)),
                T.RandomCrop(size, padding=4),
                T.ColorJitter(0.2, 0.2, 0.2),
            ]
        elif dataset_name in ['mnist', 'fmnist', 'emnist']:
            transforms = [
                T.Resize((size, size)),
                T.RandomRotation(10)
            ]
        elif dataset_name == 'food101':
            transforms = [
                T.Resize((size, size)),
                T.RandomCrop(size, padding=4),
                T.RandomHorizontalFlip(),
                T.RandomRotation(10),
            ]
        elif dataset_name == 'flowers102':
            transforms = [
                T.Resize((size, size)),
                T.RandomResizedCrop(size),
                T.RandomHorizontalFlip(),
                T.RandomRotation(15),
            ]
        elif dataset_name == 'caltech101':
            transforms = [
                T.Resize((size, size)),
                T.RandomResizedCrop(size),
                T.RandomHorizontalFlip(),
                T.RandomRotation(15),
            ]
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        transforms.extend(normalize_data(dataset_name, mean, std))
        return T.Compose(transforms)
    
    def get_contrastive_transform(dataset_name, size, s):
        mean, std = DATASET_STATS[dataset_name]
        kernel_size = int(0.1 * size)
        if kernel_size % 2 == 0:
            kernel_size += 1

        if dataset_name == 'cifar10':
            transforms = [
                T.Resize((size, size)),
                AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
            ]
        elif dataset_name == 'cifar100':
            color_jitter = T.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
            transforms = [
                T.RandomResizedCrop(size=size),
                T.RandomHorizontalFlip(),
                T.RandomApply([color_jitter], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0)),
            ]
        elif dataset_name == 'svhn':
            color_jitter = T.ColorJitter(0.4 * s, 0.4 * s, 0.4 * s, 0.1 * s)
            
            transforms = [
                T.Resize((size, size)),
                T.RandomCrop(size, padding=4),
                T.RandomRotation(10),
                T.RandomApply([color_jitter], p=0.6),
                T.RandomGrayscale(p=0.1),
                T.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0)),
            ]
        elif dataset_name in ['mnist', 'fmnist', 'emnist']:
            transforms = [
                T.Resize((size, size)),
                T.RandomRotation(10),
                T.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 1.0))
            ]
        elif dataset_name == 'food101':
            transforms = [
                T.Resize((size, size)),
                T.RandomHorizontalFlip(),
                T.RandomApply([color_jitter], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0)),
            ]
        elif dataset_name == 'flowers102':
            transforms = [
                T.Resize((size, size)),
                T.RandomResizedCrop(size, scale=(0.2, 1.0)),
                T.RandomHorizontalFlip(),
                T.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0)),
            ]
        elif dataset_name == 'caltech101':
            transforms = [
                T.Resize((size, size)),
                T.RandomResizedCrop(size, scale=(0.2, 1.0)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.8, 0.8, 0.8, 0.2),
                T.RandomGrayscale(p=0.2),
                T.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0)),
            ]
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        transforms.extend(normalize_data(dataset_name, mean, std))
        return T.Compose(transforms)

    if dataset_name not in DATASET_STATS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    if learning_type == 'supervised':
        return get_supervised_transform(dataset_name, size)
    elif learning_type == 'contrastive':
        return get_contrastive_transform(dataset_name, size, s)
    else:
        raise ValueError(f"Unsupported learning type: {learning_type}")    

class CreateDatasets:
    def __init__(self):
        self.valid_keys = {
            'learning_types':  ['contrastive', 'supervised'],
            'dataset_names': CV_DATASETS
        }

    def get_dataset_fn(self, learning_type, dataset_name):
        assert learning_type in self.valid_keys['learning_types'], 'Learning type does not exist.'
        assert dataset_name in self.valid_keys['dataset_names'], 'dataset does not exist.'
        train_dataset_fn = VALID_DATASETS[learning_type][dataset_name]
        test_dataset_fn = VALID_DATASETS['testset'][dataset_name]
        unaugmented_dataset_fn = VALID_DATASETS['unaugmented'][dataset_name]

        val_dataset_fn = None
        if dataset_name == 'flowers102':
            val_dataset_fn = lambda root_folder, size, n_views: Flowers102(
                root_folder, 
                split='val', 
                transform=AugmentData(get_train_transform(learning_type, size, dataset_name), n_views),
                download=True,
            )

        return train_dataset_fn, test_dataset_fn, val_dataset_fn, unaugmented_dataset_fn

@lru_cache()
def load_cv_dataset(dataset_name, cache_dir, learning_type, size):
    valid_datasets = CV_DATASETS
    assert dataset_name in valid_datasets, f"Unsupported CV dataset: {dataset_name}"
    
    # Creating datasets
    dataset = CreateDatasets()
    train_dataset_fn, test_dataset_fn, val_dataset_fn, unaugmented_dataset_fn = dataset.get_dataset_fn(learning_type, dataset_name)
    n_views = 1 if learning_type == 'supervised' else 2

    # Loading train and test data
    trainset = train_dataset_fn(cache_dir, size, n_views)
    testset = test_dataset_fn(cache_dir, size)
    unaugmentedset = unaugmented_dataset_fn(cache_dir, size)

    # Creating validation set if applicable

    if val_dataset_fn is not None:
        valset = val_dataset_fn(cache_dir, size, n_views)
    else:
        train_size = len(trainset)
        val_size = int(0.15 * train_size)
        train_size = train_size - val_size
        trainset, valset = random_split(trainset, [train_size, val_size])

    return {
        'train': trainset,
        'validation': valset,
        'test': testset,
        'unaugmented': unaugmentedset
    }

def get_cv_data(dataset_name, batch_size, size=32, num_workers=24, cache_dir="/dbfs", learning_type='supervised'):
    try:
        # Loading datasets
        datasets = load_cv_dataset(dataset_name, cache_dir=cache_dir, learning_type=learning_type, size=size)
        print(f"[CV DATALOADERS] Loaded {dataset_name} with splits: {list(datasets.keys())}")
        
        loader_args = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": True,
            "persistent_workers": num_workers > 0,
            "drop_last": True,
        }
        
        # Creating dataloaders
        trainloader = DataLoader(datasets["train"], shuffle=True, **loader_args)
        valloader = DataLoader(datasets["validation"], shuffle=False, **loader_args)
        testloader = DataLoader(datasets["test"], shuffle=False, **loader_args)
        unaugmentedloader = DataLoader(datasets["unaugmented"], shuffle=False, **loader_args)
        
        data = {
            "trainloader": trainloader,
            "valloader": valloader,
            "testloader": testloader,
            "unaugmentedloader": unaugmentedloader,
        }
        return data

    except Exception as e:
        print(f"Error loading CV dataset {dataset_name}: {str(e)}")
        raise e

# LLM Datasets
class GlueDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        return {
            "input_ids": example["input_ids"],
            "attention_mask": example["attention_mask"],
            "labels": example["label"]
        }

@lru_cache()
def _load_glue_dataset(task_name, cache_dir="/dbfs/hf_datasets"):
    return load_dataset("glue", task_name, cache_dir=cache_dir)

def get_glue_data(tokenizer, task_name, batch_size, num_workers=24):
    assert task_name in ["mnli", "qqp", "sst2"], f"Unsupported task: {task_name}"
    dataset = _load_glue_dataset(task_name)
    print(f"[DATALOADERS] {[key for key in dataset]}")

    def tokenize_fn(example):
        if task_name == "mnli":
            return tokenizer(example["premise"], example["hypothesis"], truncation=True, padding="max_length")
        if task_name == "qqp":
            return tokenizer(example["question1"], example["question2"], truncation=True, padding="max_length")
        if task_name == "sst2":
            return tokenizer(example["sentence"], truncation=True, padding="max_length")
        
    dataset = dataset.map(tokenize_fn, batched=True, batch_size=512, num_proc=1)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    dataset = dataset.rename_column("label", "labels")

    trainset = GlueDataset(dataset["train"])
    valset = GlueDataset(dataset["validation"])
    testset = GlueDataset(dataset["test"])

    loader_args = {
        "batch_size" : batch_size,
        "num_workers" : num_workers,
        "pin_memory" : True,
        "persistent_workers" : num_workers > 0,
        "drop_last" : True,
    }

    trainloader = DataLoader(dataset["train"], shuffle=True, **loader_args)
    valloader = DataLoader(dataset["validation"], **loader_args)
    testloader = DataLoader(dataset["test"], **loader_args)
  
    data = {
        "trainloader": trainloader,
        "valloader": valloader,
    }
    return data

class SquadDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        return {
            "input_ids": example["input_ids"],
            "attention_mask": example["attention_mask"],
            "start_positions": example["start_positions"],
            "end_positions": example["end_positions"]
        }

@lru_cache()
def _load_squad_dataset(cache_dir="/dbfs/hf_datasets"):
    return load_dataset("squad", cache_dir=cache_dir)

def get_squad_data(tokenizer, batch_size, subset_ratio=1.0, num_workers=24, max_length=384, stride=128):
    def prepare_data(example):
        answer = example['answers']['text'][0]
        example['answer_start'] = example['answers']['answer_start'][0]
        example['answer_end'] = example['answer_start'] + len(answer)
        return example

    def tokenize_and_align_labels(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            # Use the processed answer positions
            start_char = examples["answer_start"][sample_idx]
            end_char = examples["answer_end"][sample_idx]
            sequence_ids = inputs.sequence_ids(i)

            # Find context boundaries with bounds checking
            idx = 0
            while idx < len(sequence_ids) and sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            
            while idx < len(sequence_ids) and sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # Check if answer spans are valid and within context
            if (context_start >= len(offset) or context_end >= len(offset) or
                offset[context_start][0] > start_char or offset[context_end][1] < end_char):
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Find start position
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(max(idx - 1, 0))

                # Find end position
                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(min(idx + 1, max_length - 1))

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs
    
    dataset = _load_squad_dataset()
    dataset = dataset.remove_columns(['title'])
    print(f"[DATALOADERS] {[key for key in dataset]}")
    
    # printing original dataset size
    print(f"Size of original dataset: {dataset['train'].num_rows}")

    for split_name in dataset:
        if subset_ratio < 1.0 and subset_ratio > 0.0:
            subset_size = int(len(dataset[split_name]) * subset_ratio)
            dataset[split_name] = dataset[split_name].shuffle(seed=42).select(range(subset_size))
    print(f"Size of data subset: {dataset['train'].num_rows}")

    dataset = dataset.map(prepare_data, remove_columns=['answers'])
    dataset = dataset.map(
        tokenize_and_align_labels, 
        batched=True, 
        batch_size=128, 
        num_proc=1,
        remove_columns=dataset['train'].column_names
    )
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "start_positions", "end_positions"])

    trainset = SquadDataset(dataset["train"])
    valset = SquadDataset(dataset["validation"])

    loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
        "drop_last": True,
    }

    trainloader = DataLoader(trainset, shuffle=True, **loader_args)
    valloader = DataLoader(valset, shuffle=False, **loader_args) 

    data = {
        'trainloader': trainloader,
        'valloader': valloader,
    }
    return data

@lru_cache()
def _load_wikitext2_dataset(cache_dir="/dbfs/hf_datasets"):
    return load_dataset('wikitext', 'wikitext-2-raw-v1')

def get_wikitext2_data(tokenizer, batch_size, num_workers=24):
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, return_special_tokens_mask=True)

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        block_size = tokenizer.model_max_length  
        total_length = len(concatenated_examples["input_ids"])
        total_length = (total_length // block_size) * block_size
        return {
            k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }

    dataset = _load_wikitext2_dataset()
    dataset = dataset.map(tokenize_fn, batched=True, batch_size=512, remove_columns=["text"])
    dataset = dataset.map(group_texts, batched=True, batch_size=512)

    trainset = dataset['train']
    testset = dataset['test']
    valset = dataset['validation']

    loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
        "drop_last": True,
    }

    collator = DataCollatorForLanguageModeling(tokenizer, mlm_probability=0.15)

    trainloader = DataLoader(trainset, shuffle=True, collate_fn=collator, **loader_args)
    testloader = DataLoader(testset, shuffle=False, collate_fn=collator, **loader_args)
    valloader = DataLoader(valset, shuffle=False, collate_fn=collator, **loader_args)

    data = {
        "trainloader": trainloader,
        "valloader": valloader,
        "testloader": testloader
    }
    return data


