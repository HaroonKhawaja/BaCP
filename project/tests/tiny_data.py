"""In-memory synthetic image dataset -- no downloads, no disk, no /dbfs.

Deliberately builds real ``PIL.Image`` objects and pushes them through the real
``dataset_factory.get_train_transform`` + ``AugmentData``, so the tests exercise
the actual augmentation recipe and the actual ``default_collate`` behaviour
rather than a reimplementation of it. The two things worth pinning are:

  * ``AugmentData`` returns a bare tensor at ``n_views=1`` but a *list* of
    tensors at ``n_views=2`` -- a type switch, not just a length change.
  * ``default_collate`` recurses into that inner list and stacks position-wise,
    yielding ``[[Tensor[B,3,H,W], Tensor[B,3,H,W]], Tensor[B]]`` rather than a
    single ``[B,2,3,H,W]`` tensor. ``bacp.py`` unpacks ``data1, data2 = data``
    against exactly that shape.
"""

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from dataset_factory import AugmentData, get_train_transform, get_test_transform


def make_transform(t_type="supervised", size=8, n_views=1, dataset_name="cifar10"):
    """Wrap the real train transform in the real AugmentData multi-view shim.

    ``dataset_name`` defaults to cifar10 because ``get_train_transform`` only
    branches on 'cifar10' and 'imagenet' and raises ValueError otherwise --
    see test_get_train_transform_covers_all_registered_datasets.
    """
    # get_train_transform already returns a T.Compose, not a list of transforms.
    return AugmentData(get_train_transform(dataset_name, t_type, size), n_views=n_views)


def make_test_transform(size=8, dataset_name="cifar10"):
    return get_test_transform(dataset_name, size)


class SyntheticImages(Dataset):
    """``n`` deterministic PIL images with balanced labels.

    Labels are ``i % num_classes`` so every class has multiple members in any
    reasonably sized batch. That matters: ``SupConLoss`` needs same-label
    positives to exist, and an all-distinct-label batch would silently exercise
    only the ``num_pos.clamp(min=1)`` degenerate branch.
    """

    def __init__(self, transform, n=16, size=8, num_classes=4, mode="RGB", seed=0):
        self.transform = transform
        self.num_classes = num_classes
        rng = np.random.RandomState(seed)

        channels = 1 if mode == "L" else 3
        shape = (size, size) if channels == 1 else (size, size, 3)
        self.images = [
            Image.fromarray(rng.randint(0, 256, shape, dtype=np.uint8), mode=mode)
            for _ in range(n)
        ]
        self.labels = [i % num_classes for i in range(n)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.transform(self.images[idx]), self.labels[idx]


def make_loaders(n_views=1, batch_size=4, n=16, size=8, num_classes=4):
    """Return a dict shaped exactly like ``load_cv_dataloaders``' return value.

    ``valloader`` is ``None`` for the contrastive path, mirroring
    ``load_cv_datasets``, which builds no validation split when
    ``t_type == 'contrastive'``. BaCP's contrastive phase genuinely has no
    validation loader.

    ``drop_last`` mirrors production and differs by split. True on train, because
    ``SupConLoss`` does ``labels.repeat(n_views, 1)`` and would mis-align on a short
    final batch. **False on val and test**, because dropping a partial eval batch
    means never scoring those samples -- and which samples get dropped depends on
    the batch size. Keeping this fixture in step with
    ``dataset_factory.load_cv_dataloaders`` is the point: a fixture that drops eval
    samples when production does not would make the L2 assertions meaningless.
    """
    t_type = "contrastive" if n_views > 1 else "supervised"
    train = SyntheticImages(make_transform(t_type, size, n_views), n=n, size=size,
                            num_classes=num_classes)
    test = SyntheticImages(make_test_transform(size), n=n, size=size,
                           num_classes=num_classes)

    common = dict(batch_size=batch_size, num_workers=0)
    trainloader = DataLoader(train, shuffle=True, drop_last=True, **common)
    testloader = DataLoader(test, shuffle=False, drop_last=False, **common)
    valloader = (None if n_views > 1
                 else DataLoader(test, shuffle=False, drop_last=False, **common))

    return {"trainloader": trainloader, "valloader": valloader, "testloader": testloader}


def batch_of(n_views=1, batch_size=4, num_classes=4, size=8, device="cpu"):
    """One collated batch, for tests that need a batch without a DataLoader."""
    loaders = make_loaders(n_views=n_views, batch_size=batch_size,
                           n=max(batch_size * 2, 8), size=size, num_classes=num_classes)
    batch = next(iter(loaders["trainloader"]))
    data, labels = batch
    if isinstance(data, (list, tuple)):
        data = [d.to(device) for d in data]
    else:
        data = data.to(device)
    return data, labels.to(device)


def random_embeddings(batch=4, dim=8, seed=0, normalize=True):
    """L2-normalized embeddings, matching what ``get_embeddings`` produces."""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(batch, dim, generator=g)
    return torch.nn.functional.normalize(z, dim=1) if normalize else z
