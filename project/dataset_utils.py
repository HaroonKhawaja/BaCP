from torchvision.datasets import  CIFAR10, SVHN, MNIST, FashionMNIST, Food101, Flowers102, CIFAR100
import torchvision.transforms as T
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset 
from utils import *
from constants import *

VALID_DATASETS = {
    'supcon': {
        'cifar10': lambda: CIFAR10(self.root_folder, train=True, transform=AugmentData(self.get_transform('supcon', size, 'cifar10'), n_views), download=True),
        'svhn': lambda: SVHN(self.root_folder, split='train', transform=AugmentData(self.get_transform('supcon', size, 'svhn'), n_views), download=True),
        'mnist': lambda: MNIST(self.root_folder, train=True, transform=AugmentData(self.get_transform('supcon', size, 'mnist'), n_views), download=True),
        'fmnist': lambda: FashionMNIST(self.root_folder, train=True, transform=AugmentData(self.get_transform('supcon', size, 'fmnist'), n_views), download=True),
        'food101': lambda: Food101(self.root_folder, split='train', transform=AugmentData(self.get_transform('supcon', size), n_views), download=True),
        'flowers102': lambda: Flowers102(self.root_folder, split='train', transform=AugmentData(self.get_transform('supcon', size), n_views), download=True),
        'cifar100': lambda: CIFAR100(self.root_folder, train=True, transform=AugmentData(self.get_transform('supcon', size), n_views), download=True),
    },
    
    'supervised': {
        'cifar10': lambda: CIFAR10(self.root_folder, train=True, transform=AugmentData(self.get_transform('supervised', size, 'cifar10'), n_views=1), download=True),
        'svhn': lambda: SVHN(self.root_folder, split='train', transform=AugmentData(self.get_transform('supervised', size, 'svhn'), n_views=1), download=True),
        'mnist': lambda: MNIST(self.root_folder, train=True, transform=AugmentData(self.get_transform('supervised', size, 'mnist'), n_views=1), download=True),
        'fmnist': lambda: FashionMNIST(self.root_folder, train=True, transform=AugmentData(self.get_transform('supervised', size, 'fmnist'), n_views=1), download=True),
        'food101': lambda: Food101(self.root_folder, split='train', transform=AugmentData(self.get_transform('supervised', size), n_views=1), download=True),
        'flowers102': lambda: Flowers102(self.root_folder, split='train', transform=AugmentData(self.get_transform('supervised', size), n_views=1), download=True),
        'cifar100': lambda: CIFAR100(self.root_folder, train=True, transform=AugmentData(self.get_transform('supervised', size), n_views=1), download=True),
    },
    
    'testset': {
        'cifar10': lambda: CIFAR10(self.root_folder, train=False, transform=AugmentData(self.get_transform('supervised', size, 'cifar10'), n_views=1), download=True),
        'svhn': lambda: SVHN(self.root_folder, split='test', transform=AugmentData(self.get_transform('supervised', size, 'svhn'), n_views=1), download=True),
        'mnist': lambda: MNIST(self.root_folder, train=False, transform=AugmentData(self.get_transform('supervised', size, 'mnist'), n_views=1), download=True),
        'fmnist': lambda: FashionMNIST(self.root_folder, train=False, transform=AugmentData(self.get_transform('supervised', size, 'fmnist'), n_views=1), download=True),
        'food101': lambda: Food101(self.root_folder, split='test', transform=AugmentData(self.get_transform('supervised', size), n_views=1), download=True),
        'flowers102': lambda: Flowers102(self.root_folder, split='test', transform=AugmentData(self.get_transform('supervised', size), n_views=1), download=True),
        'cifar100': lambda: CIFAR100(self.root_folder, train=False, transform=AugmentData(self.get_transform('supervised', size), n_views=1), download=True),
    }
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
        
class CreateDatasets:
    def __init__(self, root_folder):
        self.root_folder = root_folder
        self.valid_keys = {
            'learning_types':  ['supcon', 'supervised', 'supervised_vit'],
            'dataset_names': ['cifar10', 'svhn', 'mnist', 'fmnist', 'food101', 'flowers102', 'cifar100']
        }

    def get_transform(self, learning_type, size=32, dataset_name="", s=1):
        if learning_type == 'supcon':   
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
            if dataset_name in ['mnist', 'fmnist']:
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


    def get_dataset_fn(self, learning_type, dataset_name, size=32, n_views=2):
        assert learning_type in self.valid_keys['learning_types'], 'Learning type does not exist.'
        assert dataset_name in self.valid_keys['dataset_names'], 'dataset does not exist.'
        
        train_dataset_fn = VALID_DATASETS[learning_type][dataset_name]
        test_dataset_fn = VALID_DATASETS['testset'][dataset_name]
        
        return train_dataset_fn, test_dataset_fn
    
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

def get_glue_data(model_name, tokenizer, task_name, batch_size, num_workers=24):
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


            