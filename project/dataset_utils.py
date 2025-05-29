from torchvision.datasets import  CIFAR10, SVHN, MNIST, FashionMNIST, Food101, Flowers102, CIFAR100
import torchvision.transforms as T
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
from datasets import load_dataset 
from utils import *
from constants import *
from functools import lru_cache

VALID_DATASETS = {
    'contrastive': {
        'cifar10': lambda root_folder, size, n_views: CIFAR10(root_folder, train=True, transform=AugmentData(get_transform('contrastive', size, 'cifar10'), n_views), download=True),
        'svhn': lambda root_folder, size, n_views: SVHN(root_folder, split='train', transform=AugmentData(get_transform('contrastive', size, 'svhn'), n_views), download=True),
        'mnist': lambda root_folder, size, n_views: MNIST(root_folder, train=True, transform=AugmentData(get_transform('contrastive', size, 'mnist'), n_views), download=True),
        'fmnist': lambda root_folder, size, n_views: FashionMNIST(root_folder, train=True, transform=AugmentData(get_transform('contrastive', size, 'fmnist'), n_views), download=True),
        'food101': lambda root_folder, size, n_views: Food101(root_folder, split='train', transform=AugmentData(get_transform('contrastive', size), n_views), download=True),
        'flowers102': lambda root_folder, size, n_views: Flowers102(root_folder, split='train', transform=AugmentData(get_transform('contrastive', size), n_views), download=True),
        'cifar100': lambda root_folder, size, n_views: CIFAR100(root_folder, train=True, transform=AugmentData(get_transform('contrastive', size), n_views), download=True),
    },
    
    'supervised': {
        'cifar10': lambda root_folder, size, n_views: CIFAR10(root_folder, train=True, transform=AugmentData(get_transform('supervised', size, 'cifar10'), n_views), download=True),
        'svhn': lambda root_folder, size, n_views: SVHN(root_folder, split='train', transform=AugmentData(get_transform('supervised', size, 'svhn'), n_views), download=True),
        'mnist': lambda root_folder, size, n_views: MNIST(root_folder, train=True, transform=AugmentData(get_transform('supervised', size, 'mnist'), n_views), download=True),
        'fmnist': lambda root_folder, size, n_views: FashionMNIST(root_folder, train=True, transform=AugmentData(get_transform('supervised', size, 'fmnist'), n_views), download=True),
        'food101': lambda root_folder, size, n_views: Food101(root_folder, split='train', transform=AugmentData(get_transform('supervised', size), n_views), download=True),
        'flowers102': lambda root_folder, size, n_views: Flowers102(root_folder, split='train', transform=AugmentData(get_transform('supervised', size), n_views), download=True),
        'cifar100': lambda root_folder, size, n_views: CIFAR100(root_folder, train=True, transform=AugmentData(get_transform('supervised', size), n_views), download=True),
    },
    
    'testset': {
        'cifar10': lambda root_folder, size, n_views: CIFAR10(root_folder, train=False, transform=AugmentData(get_transform('supervised', size, 'cifar10'), n_views), download=True),
        'svhn': lambda root_folder, size, n_views: SVHN(root_folder, split='test', transform=AugmentData(get_transform('supervised', size, 'svhn'), n_views), download=True),
        'mnist': lambda root_folder, size, n_views: MNIST(root_folder, train=False, transform=AugmentData(get_transform('supervised', size, 'mnist'), n_views), download=True),
        'fmnist': lambda root_folder, size, n_views: FashionMNIST(root_folder, train=False, transform=AugmentData(get_transform('supervised', size, 'fmnist'), n_views), download=True),
        'food101': lambda root_folder, size, n_views: Food101(root_folder, split='test', transform=AugmentData(get_transform('supervised', size), n_views), download=True),
        'flowers102': lambda root_folder, size, n_views: Flowers102(root_folder, split='test', transform=AugmentData(get_transform('supervised', size), n_views), download=True),
        'cifar100': lambda root_folder, size, n_views: CIFAR100(root_folder, train=False, transform=AugmentData(get_transform('supervised', size), n_views), download=True),
    }
}

CV_DATASETS = ['cifar10', 'svhn', 'mnist', 'fmnist', 'food101', 'flowers102', 'cifar100']

class AugmentData(object):
    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views
    
    def __call__(self, x):
        if self.n_views == 1:
            return self.base_transform(x)
        else:
            return [self.base_transform(x) for _ in range(self.n_views)]

def get_transform(learning_type, size=32, dataset_name="", s=1):
    if learning_type == 'contrastive':   
        if dataset_name == 'cifar10':
            data_transforms = T.Compose([
                T.Resize((size, size)),
                AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
                T.ToTensor(),
                T.Normalize(mean=MEAN_CIFAR10, std=STD_CIFAR10),
                ])
        elif dataset_name == 'cifar100':
            color_jitter = T.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
            data_transforms = T.Compose([
                T.RandomResizedCrop(size=size),
                T.RandomHorizontalFlip(),
                T.RandomApply([color_jitter], p=0.8),
                T.RandomGrayscale(p=0.2), 
                T.GaussianBlur(kernel_size=int(0.1 * size)),
                T.ToTensor(),
                T.Normalize(mean=MEAN_CIFAR100, std=STD_CIFAR100),
                ])
        elif dataset_name == 'svhn':
            data_transforms = T.Compose([
                T.Resize((size, size)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                ])
        elif dataset_name in ['mnist', 'fmnist']:
            data_transforms = T.Compose([
                T.Resize((size, size)),
                T.RandomRotation(10),
                T.ToTensor(),
                ])
        else:
            color_jitter = T.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
            data_transforms = T.Compose([
                T.RandomResizedCrop(size=size),
                T.RandomHorizontalFlip(),
                T.RandomApply([color_jitter], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD),
                ])
    elif learning_type == 'supervised':
        if dataset_name == 'cifar10':
            data_transforms = T.Compose([
                T.Resize((size, size)),
                T.ToTensor(),
                T.Normalize(mean=MEAN_CIFAR10, std=STD_CIFAR10),
            ])
        elif dataset_name == 'cifar100':
            data_transforms = T.Compose([
                T.Resize((size, size)),
                T.ToTensor(),
                T.Normalize(mean=MEAN_CIFAR100, std=STD_CIFAR100),
            ])
        elif dataset_name == 'svhn':
            data_transforms = T.Compose([
                T.Resize((size, size)),
                T.ToTensor(),
            ])
        elif dataset_name in ['mnist', 'fmnist']:
            data_transforms = T.Compose([
                T.Resize((size, size)),
                T.ToTensor(),
            ])
        else:
            data_transforms = T.Compose([
                T.Resize((size, size)),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD),
            ])
    return data_transforms

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
        return train_dataset_fn, test_dataset_fn

@lru_cache()
def load_cv_dataset(dataset_name, cache_dir="./data", learning_type='supervised', size=32):
    """
    Load computer vision dataset.
    """
    valid_datasets = ['cifar10', 'svhn', 'mnist', 'fmnist', 'food101', 'flowers102', 'cifar100']
    assert dataset_name in valid_datasets, f"Unsupported CV dataset: {dataset_name}"

    # Creating datasets
    dataset = CreateDatasets()
    train_dataset_fn, test_dataset_fn = dataset.get_dataset_fn(learning_type, dataset_name)
    trainset = train_dataset_fn(cache_dir, size, 1 if learning_type == 'supervised' else 2)
    testset = test_dataset_fn(cache_dir, size, 1)
    
    # Creating validation set
    if hasattr(trainset, 'data'):
        train_size = len(trainset)
        val_size = int(0.15 * train_size)
        train_size = train_size - val_size
        trainset, valset = random_split(trainset, [train_size, val_size])
    
    return {
        'train': trainset,
        'validation': valset,
        'test': testset
    }

def get_cv_data(dataset_name, batch_size, size=32, num_workers=24, cache_dir="./data", learning_type='supervised'):
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
        
        data = {
            "trainloader": trainloader,
            "trainset": datasets["train"],
            "valloader": valloader,
            "valset": datasets["validation"],
            "testloader": testloader,
            "testset": datasets["test"]
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


def get_squad_data(tokenizer, task_name, batch_size, num_workers=24):
    dataset = load_dataset("rajpurkar/squad")
    print(f"[DATALOADERS] {[key for key in dataset]}")

    def tokenize_fn(example):
        return tokenizer(example)


def get_glue_data(tokenizer, task_name, batch_size, num_workers=24):
    assert task_name in ["mnli", "qqp", "sst2"], f"Unsupported task: {task_name}"
    dataset = load_dataset("glue", task_name, cache_dir="/dbfs/hf_datasets")
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
    validationloader = DataLoader(dataset["validation"], **loader_args)
    testloader = DataLoader(dataset["test"], **loader_args)
  
    data = {
        "trainloader": trainloader,
        "trainset": trainset,
        "valloader": validationloader,
        "valset": valset,
        "testloader": testloader,
        "testset": testset
    }
    return data
