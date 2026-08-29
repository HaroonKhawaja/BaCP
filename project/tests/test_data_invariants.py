"""L2 -- data pipeline invariants. No downloads, no disk.

The batch structure pinned here is not incidental: bacp.py does
``data1, data2 = data`` and feeds view 1 to the trainable model and view 2 to all
three frozen teachers. A silent change in collate shape would corrupt the view
pairing without raising anything.
"""

import pickle

import pytest
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data._utils.collate import default_collate

from dataset_factory import (
    CV_DATASETS,
    DATASET_STATS,
    AugmentData,
    get_test_transform,
    get_train_transform,
    normalize_data,
)
from training_utils import _handle_data_to_device
from types import SimpleNamespace

SIZE = 8


def _pil(mode="RGB", seed=0):
    """A non-uniform image.

    Must not be flat: a constant image is invariant to crop, flip and jitter, so
    two "independent" views of it are byte-identical and the multi-view test would
    pass vacuously in the broken case as well as the working one.
    """
    import numpy as np
    shape = (SIZE, SIZE) if mode == "L" else (SIZE, SIZE, 3)
    rng = np.random.RandomState(seed)
    return Image.fromarray(rng.randint(0, 256, shape, dtype=np.uint8), mode=mode)


# --- AugmentData multi-view semantics ------------------------------------

def test_one_view_returns_a_bare_tensor():
    """n_views=1 returns a tensor, not a 1-element list.

    A type switch, not just a length change -- which is what lets the same
    downstream unpacking serve both the supervised and contrastive paths.
    """
    out = AugmentData(get_train_transform("cifar10", "supervised", SIZE), n_views=1)(_pil())
    assert torch.is_tensor(out)
    assert out.shape == (3, SIZE, SIZE)


def test_two_views_returns_two_independently_augmented_tensors():
    """The two views must be different draws, not the same tensor twice.

    AugmentData re-enters the same Compose per view, and each RandomResizedCrop /
    ColorJitter / RandomGrayscale samples fresh parameters from the global RNG. If
    this ever collapsed to one draw, the self-supervised term would be comparing an
    image with itself and the objective would be trivially satisfiable.
    """
    out = AugmentData(get_train_transform("cifar10", "contrastive", SIZE), n_views=2)(_pil())
    assert isinstance(out, list) and len(out) == 2
    assert not torch.allclose(out[0], out[1])


# --- Collate structure ---------------------------------------------------

def test_collate_shape_supervised():
    samples = [(torch.zeros(3, SIZE, SIZE), i % 2) for i in range(4)]
    data, labels = default_collate(samples)
    assert data.shape == (4, 3, SIZE, SIZE)
    assert labels.shape == (4,)


def test_collate_shape_contrastive():
    """[[Tensor[B,3,H,W], Tensor[B,3,H,W]], Tensor[B]].

    default_collate recurses into the inner list and stacks position-wise -- so the
    result is a list of two batched tensors, NOT a single [B,2,3,H,W] tensor and not
    a list of B pairs. bacp.py's ``data1, data2 = data`` depends on exactly this.
    """
    samples = [([torch.zeros(3, SIZE, SIZE), torch.ones(3, SIZE, SIZE)], i % 2)
               for i in range(4)]
    data, labels = default_collate(samples)

    assert isinstance(data, list) and len(data) == 2
    assert data[0].shape == (4, 3, SIZE, SIZE)
    assert data[1].shape == (4, 3, SIZE, SIZE)
    assert labels.shape == (4,)


# --- _handle_data_to_device ---------------------------------------------

def _args():
    return SimpleNamespace(device="cpu")


def test_handle_data_to_device_supervised():
    batch = [torch.zeros(2, 3, SIZE, SIZE), torch.zeros(2, dtype=torch.long)]
    data, labels = _handle_data_to_device(_args(), batch)
    assert torch.is_tensor(data) and labels.shape == (2,)


def test_handle_data_to_device_contrastive():
    batch = [[torch.zeros(2, 3, SIZE, SIZE)] * 2, torch.zeros(2, dtype=torch.long)]
    data, labels = _handle_data_to_device(_args(), batch)
    assert isinstance(data, list) and len(data) == 2


def test_handle_data_to_device_dict_batch():
    batch = {"input_ids": torch.zeros(2, 4, dtype=torch.long),
             "labels": torch.zeros(2, dtype=torch.long)}
    data, labels = _handle_data_to_device(_args(), batch)
    assert set(data) == {"input_ids", "labels"}
    assert labels is not None


def test_handle_data_to_device_rejects_an_unknown_batch():
    """An unrecognised batch shape must raise, not fall through.

    Neither branch matched a 3-tuple, leaving `data` and `labels` unbound and
    producing an UnboundLocalError far from the cause.
    """
    with pytest.raises(ValueError):
        _handle_data_to_device(_args(), (torch.zeros(1), torch.zeros(1), torch.zeros(1)))


# --- Transform coverage --------------------------------------------------

@pytest.mark.parametrize("dataset", sorted(CV_DATASETS))
@pytest.mark.parametrize("t_type", ["supervised", "contrastive"])
def test_get_train_transform_covers_every_registered_dataset(dataset, t_type):
    """Every dataset in CV_DATASETS must be reachable.

    CV_DATASETS registers seven datasets but get_train_transform only branched on
    cifar10 and imagenet, raising ValueError for the rest -- so cifar100, svhn,
    mnist, fmnist and emnist were registered but unusable. The paper reports
    results on several of them, so whichever code produced those numbers took a
    different path than this one.
    """
    transform = get_train_transform(dataset, t_type, SIZE)
    mode = "L" if dataset in ("mnist", "fmnist", "emnist") else "RGB"
    assert transform(_pil(mode)).shape == (3, SIZE, SIZE)


def test_cv_datasets_all_have_normalization_stats():
    missing = sorted(set(CV_DATASETS) - set(DATASET_STATS))
    assert not missing, f"CV_DATASETS entries with no DATASET_STATS: {missing}"


def test_grayscale_transform_yields_three_channels():
    out = T.Compose(normalize_data("mnist"))(_pil("L"))
    assert out.shape == (3, SIZE, SIZE)


def test_normalize_data_is_picklable():
    """The transform must survive being sent to a DataLoader worker.

    A T.Lambda(lambda x: ...) closure cannot be pickled, which breaks
    num_workers > 0 under Windows/macOS spawn. It only ever worked because the
    Databricks environment forks.
    """
    pickle.dumps(T.Compose(normalize_data("mnist")))
    pickle.dumps(T.Compose(normalize_data("cifar10")))


def test_test_transform_is_deterministic():
    t = get_test_transform("cifar10", SIZE)
    img = _pil()
    assert torch.allclose(t(img), t(img))


# --- Loader assembly ----------------------------------------------------

def test_contrastive_path_has_no_validation_loader(loaders):
    """BaCP's contrastive phase genuinely has no validation split.

    Recorded because it explains why the contrastive phase saves on minimum
    training loss rather than on validation accuracy.
    """
    assert loaders(n_views=2)["valloader"] is None
    assert loaders(n_views=1)["valloader"] is not None


def test_loaders_drop_last(loaders):
    """drop_last=True is relied upon, not merely convenient.

    SupConLoss does labels.repeat(n_views, 1) against a concatenated embedding
    batch; a short final batch would silently misalign labels against embeddings.
    """
    assert loaders(n_views=2)["trainloader"].drop_last is True


def test_two_view_batch_matches_the_documented_structure(loaders):
    batch = next(iter(loaders(n_views=2)["trainloader"]))
    data, labels = batch
    assert isinstance(data, list) and len(data) == 2
    assert data[0].shape == data[1].shape
    assert data[0].shape[0] == labels.shape[0]
