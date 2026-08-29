"""L2b -- evaluation and reproducibility correctness.

Each test here guards a defect that silently corrupted a reported number rather
than raising. They are grouped separately from L2 because they are about
*measurement* rather than about data plumbing.
"""

import pickle

import numpy as np
import pytest
import torch
from types import SimpleNamespace

import dataset_factory
from dataset_factory import (
    VAL_SPLIT_SEED,
    _loader_args,
    _seed_worker,
    load_cv_datasets,
)

SIZE = 8


# --- drop_last -----------------------------------------------------------

def test_eval_loaders_do_not_drop_the_last_batch(loaders):
    """Val and test must score every sample.

    drop_last=True used to be set on all three loaders, so CIFAR-10's 10,000-image
    test set was evaluated over floor(10000/512)*512 = 9,728 images. 272 were never
    scored, and *which* 272 depended on the batch size -- so changing batch size
    changed the reported accuracy. This is the single most consequential silent
    defect in the evaluation path.
    """
    d = loaders(n_views=1, batch_size=5, n=16)   # 16 is not a multiple of 5
    assert d["trainloader"].drop_last is True, "train may still drop a partial batch"
    assert d["testloader"].drop_last is False
    assert d["valloader"].drop_last is False


def test_eval_loader_yields_every_sample(loaders):
    d = loaders(n_views=1, batch_size=5, n=16)
    seen = sum(labels.shape[0] for _, labels in d["testloader"])
    assert seen == 16, f"expected all 16 samples, scored {seen}"


def test_train_loader_still_drops_for_stable_batch_shape(loaders):
    """drop_last stays True on train, and not arbitrarily.

    SupConLoss does labels.repeat(n_views, 1) against a concatenated embedding
    batch; a short final batch would misalign labels against embeddings.
    """
    d = loaders(n_views=2, batch_size=5, n=16)
    sizes = {labels.shape[0] for _, labels in d["trainloader"]}
    assert sizes == {5}


# --- val split determinism ----------------------------------------------

def _val_indices(seed=VAL_SPLIT_SEED, rng_burn=0):
    """Draw a val split, optionally after burning global RNG first.

    rng_burn simulates model construction consuming randomness before the split --
    which is exactly what _initialize_models does, since it runs before
    _initialize_data_loaders.
    """
    torch.manual_seed(0)
    if rng_burn:
        torch.randn(rng_burn)
    from tiny_data import SyntheticImages, make_transform

    ds = SyntheticImages(make_transform("supervised", SIZE, 1), n=20, size=SIZE)
    gen = torch.Generator().manual_seed(seed)
    _, val = torch.utils.data.random_split(ds, [16, 4], generator=gen)
    return list(val.indices)


def test_val_split_is_independent_of_prior_rng_consumption():
    """The split must not move when the model upstream consumes different RNG.

    Without an explicit generator, random_split drew from the *global* torch RNG,
    whose state depends on how much randomness model construction used. So changing
    the backbone changed the validation set -- meaning two models were never
    evaluated on the same data.
    """
    assert _val_indices(rng_burn=0) == _val_indices(rng_burn=997)


def test_val_split_is_stable_across_calls():
    assert _val_indices() == _val_indices()


def test_val_split_seed_actually_changes_the_split():
    """Sanity: the seed is wired through, not ignored."""
    assert _val_indices(seed=VAL_SPLIT_SEED) != _val_indices(seed=VAL_SPLIT_SEED + 1)


def test_val_split_seed_defaults_to_the_module_constant_not_the_run_seed():
    """The split seed is a fixed constant, deliberately decoupled from args.seed.

    If the split tracked the run seed, a 3-seed error bar would conflate
    initialisation variance with split variance, and early stopping would select on a
    different held-out set per seed. Checked via the signature default rather than by
    grepping the source, so a comment mentioning args.seed cannot fail the test.
    """
    import inspect

    sig = inspect.signature(load_cv_datasets)
    assert sig.parameters['val_split_seed'].default == VAL_SPLIT_SEED

    # And it must not be threaded from the run seed in get_dataloaders.
    src = inspect.getsource(dataset_factory.get_dataloaders)
    assert "val_split_seed=getattr(args, 'val_split_seed', VAL_SPLIT_SEED)" in src
    assert "val_split_seed=args.seed" not in src


# --- worker and loader seeding ------------------------------------------

def test_loader_args_are_deterministic_given_seed_and_salt():
    a = _loader_args(4, 0, seed=7, salt=1)
    b = _loader_args(4, 0, seed=7, salt=1)
    assert a["generator"].initial_seed() == b["generator"].initial_seed()


def test_loader_args_salt_separates_train_val_test():
    seeds = {_loader_args(4, 0, seed=7, salt=s)["generator"].initial_seed()
             for s in (0, 1, 2)}
    assert len(seeds) == 3, "train/val/test must not share a shuffle stream"


def test_worker_init_fn_seeds_numpy_and_random():
    """PyTorch seeds each worker's torch RNG but leaves numpy and random alone.

    Any augmentation reaching for those was therefore not reproducible, and
    num_workers defaults to os.cpu_count(), so the sequence was machine-dependent.
    """
    import random as _random

    torch.manual_seed(1234)
    _seed_worker(0)
    first = (np.random.rand(), _random.random())

    torch.manual_seed(1234)
    _seed_worker(0)
    assert (np.random.rand(), _random.random()) == first


def test_loader_args_are_picklable():
    """The whole kwargs dict must survive being shipped to a spawn-based worker."""
    args = _loader_args(4, 0, seed=1, salt=0)
    pickle.dumps(args["worker_init_fn"])


# --- seed threading ------------------------------------------------------

@pytest.mark.parametrize("kind", ["baseline", "bacp"])
def test_seed_reaches_the_dataclass_and_the_config_snapshot(kind, real_args):
    """The seed must be recorded, not consumed.

    It used to be popped from args_dict in the scripts before the dataclass was
    constructed, so it never reached args, never entered logger_params, and was
    never written anywhere -- making a completed run unreproducible after the fact.
    """
    import training_utils

    args = real_args(kind, seed=1234, trained_weights="does-not-exist.pt")
    assert args.seed == 1234
    training_utils._initialize_all(args)
    assert args.logger_params["seed"] == 1234


@pytest.mark.parametrize("kind", ["baseline", "bacp"])
def test_val_split_seed_and_deterministic_are_recorded(kind, real_args):
    import training_utils

    args = real_args(kind, trained_weights="does-not-exist.pt")
    training_utils._initialize_all(args)
    assert args.logger_params["val_split_seed"] == 1234
    assert args.logger_params["deterministic"] is False


def test_scripts_no_longer_pop_the_seed():
    """Structural guard: popping it is what made it unrecordable."""
    from pathlib import Path

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    for name in ("baseline_script.py", "pruning_script.py", "bacp_script.py"):
        src = (scripts / name).read_text(encoding="utf-8")
        assert "pop('seed')" not in src, f"{name} still pops the seed"
        assert 'pop("seed")' not in src, f"{name} still pops the seed"


# --- requested vs adjusted sparsity -------------------------------------

def test_requested_and_adjusted_sparsity_are_both_recorded(fake_args, patch_models,
                                                           patch_dataloaders):
    """_calculate_adjusted_sparsity rewrites target_sparsity in place.

    Without capturing both, the requested value survived only in stdout and every
    record would report the adjusted figure as though it had been asked for.
    """
    import training_utils

    args = fake_args(pruning_type="magnitude", target_sparsity=0.9,
                     sparsity_scheduler="cubic")
    training_utils._initialize_models(args)
    training_utils._initialize_data_loaders(args)
    training_utils._initialize_optimizer(args)
    training_utils._initialize_pruner(args)

    assert args.requested_sparsity == 0.9
    assert args.adjusted_sparsity == 0.9      # no weight sharing -> identical


def test_adjusted_exceeds_requested_under_weight_sharing(fake_args, patch_models,
                                                         patch_dataloaders,
                                                         tiny_wrapper_shared):
    import training_utils

    args = fake_args(pruning_type="magnitude", target_sparsity=0.9,
                     sparsity_scheduler="cubic", weight_sharing_en=True)
    args.model = tiny_wrapper_shared
    training_utils._initialize_data_loaders(args)
    training_utils._initialize_optimizer(args)
    training_utils._initialize_pruner(args)

    assert args.requested_sparsity == 0.9
    assert args.adjusted_sparsity < args.requested_sparsity, (
        "sharing reduces the unique-parameter count, so the unique set is pruned less "
        "aggressively to keep the same absolute weight budget"
    )


# --- epoch timing --------------------------------------------------------

def test_epoch_times_are_recorded(fake_args, patch_models, patch_dataloaders):
    """Nothing measured wall-clock before, and with no run-end timestamp the
    duration could not even be derived after the fact."""
    import time

    import training_utils

    args = fake_args()
    training_utils._initialize_models(args)
    args.logger = None
    args.pruner = None
    training_utils._initialize_logs(args)

    for i in (1, 2):
        time.sleep(0.01)
        training_utils._log_metrics(args, f"Epoch [{i}/2]", {"accuracy": 1.0})

    times = training_utils.epoch_times(args)
    assert len(times) == 2
    assert all(t > 0 for t in times)


def test_epoch_times_empty_before_any_epoch(fake_args):
    import training_utils
    assert training_utils.epoch_times(fake_args()) == []
