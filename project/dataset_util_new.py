import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from datasets import load_dataset

VALID_DATASETS = {
    'cv', {'cifar10', 'cifar100'},
    'llm': {'sst2'}
}

def build_transforms(dataset_name, transform_type, train, size=32):
    common_transforms = [
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ]

    if train:
        if dataset_name == "cifar10":
            augmentations = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ]
        elif dataset_name == "cifar100":
            augmentations = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
            ]
        else:
            augmentations = []
        return transforms.Compose(augmentations + common_transforms)
    
    # Validation / test
    return transforms.Compose(common_transforms)


def load_cv_dataset(dataset_name, transform_type, cache_dir):
    if dataset_name == 'cifar10':
        train_set = datasets.CIFAR10(
            cache_dir, train=True, download=True, transform=build_transform(dataset_name, transform_type, True)
            )
        val_set = datasets.CIFAR10(
            cache_dir, train=False, download=True, transform=build_transform(dataset_name, transform_type, False)
            )
    if dataset_name == 'cifar100':
        train_set = datasets.CIFAR100(
            cache_dir, train=True, download=True, transform=build_transform(dataset_name, transform_type, True)
            )
        val_set = datasets.CIFAR100(
            cache_dir, train=False, download=True, transform=build_transform(dataset_name, transform_type, False)
            )
    else:
        raise ValueError(f"Unsupported CV dataset: {dataset_name}")
    return train_set, val_set

def get_data(dataset_name, batch_size, transform_type='contrastive', cache_dir=None):
    if dataset_name in VALID_DATASETS['cv']:
        train_set, val_set = load_cv_dataset(dataset_name, transform_type, cache_dir)
        return make_dataloaders(train_set, val_set, batch_size)
    else:
        valid_list = ", ".join(sorted(sum(VALID_DATASETS.values(), [])))
        raise ValueError(f"Unsupported dataset: {dataset_name}. Choose from: {valid_list}")