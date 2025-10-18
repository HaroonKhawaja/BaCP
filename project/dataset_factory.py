from torchvision.datasets import  CIFAR10, SVHN, MNIST, FashionMNIST, Food101, Flowers102, CIFAR100, EMNIST, Caltech101
import torchvision.transforms as T
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
from datasets import load_dataset 
from transformers import DataCollatorForLanguageModeling
from utils import *
from functools import lru_cache
from datasets import get_dataset_config_names
import inspect
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
}

GRAYSCALE_DATASETS = ["mnist", "fmnist", "emnist"]
DATASET_STATS = {
    "cifar10": ([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    "cifar100": ([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
    "svhn": ([0.4380, 0.4440, 0.4730], [0.1751, 0.1771, 0.1744]),
    "mnist": ([0.1307], [0.3081]),
    "fmnist": ([0.2860], [0.3530]),
    "emnist": ([0.1307], [0.3081]),
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
            T.ToTensor(),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
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
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
        
    # Mean and STD normalization
    transform.extend(normalize_data(dataset_name))
    return T.Compose(transform)
    
def get_test_transform(dataset_name: str, size: int):
    transform = [
        T.Resize((size, size)),
    ]
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
        trainset, valset = random_split(trainset, [train_size, val_size])

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
    ):
    try:
        datasets = load_cv_datasets(dataset_name, t_type, size, n_views, cache_dir)
        loader_args = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": True,
            "persistent_workers": num_workers > 0,
            "drop_last": True,
        }
        
        # Creating dataloaders
        trainloader = DataLoader(datasets["train"], shuffle=True, **loader_args)
        testloader = DataLoader(datasets["test"], shuffle=False, **loader_args)

        if datasets["validation"] is not None:
            valloader = DataLoader(datasets["validation"], shuffle=False, **loader_args)
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

def get_dataloaders(args):
    """Returns Dataloaders"""

    t_type = 'contrastive' if args.is_bacp else 'supervised'
    if args.model_type == 'cv':
        return load_cv_dataloaders(
            dataset_name=args.dataset_name, 
            t_type=t_type,
            batch_size=args.batch_size, 
            size=args.image_size,
            n_views=args.n_views,
            num_workers=args.num_workers,
            cache_dir=args.cache_dir,
            )
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")









        


        