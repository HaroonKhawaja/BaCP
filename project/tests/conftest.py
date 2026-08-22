"""Shared fixtures for the BaCP test ladder.

Read this file first. Three environment decisions below are not optional, and
two monkeypatch seams are what make the whole suite runnable on CPU in seconds
without touching production code.
"""

import os
import sys
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

# --- Path and environment seam -------------------------------------------
# Imports across this codebase are flat ("from utils import ...", not
# "from project.utils import ..."), and the scripts achieve that with
# sys.path.append(os.path.abspath('..')) plus a CWD of project/scripts/.
# Insert project/ explicitly rather than relying on pytest's rootdir insertion,
# so the mechanism is visible instead of magic.
PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

# MPLBACKEND is required, not cosmetic: pruning_factory.py imports
# matplotlib.pyplot at module scope, and check_sparsity_distribution calls
# plt.show(). Without Agg, importing the pruner can block or fail headless.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pytest  # noqa: E402
import torch  # noqa: E402

import model_factory  # noqa: E402
import training_utils  # noqa: E402
from model_factory import ClassificationAndEncoderNetwork, ModelSpec  # noqa: E402
from utils import set_seed  # noqa: E402
from weight_sharing import apply_weight_sharing_resnet  # noqa: E402

from tiny_data import make_loaders  # noqa: E402
from tiny_models import make_tiny_resnet  # noqa: E402

TINY = "tiny"
NUM_CLASSES = 4
NUM_OUT_FEATURES = 8
IMAGE_SIZE = 8
BATCH_SIZE = 4


# --- Autouse fixtures ----------------------------------------------------

@pytest.fixture(autouse=True)
def default_prunable_scope():
    """Reset the prunable scope between tests.

    pruning_factory holds it as module-level state -- deliberately, since the
    pruner and every sparsity-reporting function must agree on which parameters
    are in scope or the reported figure describes a different set of weights than
    the one being pruned. That does mean a test enabling head pruning would leak
    into later tests without this.
    """
    import pruning_factory
    pruning_factory.set_prunable_scope(prune_task_head=False, prune_embeddings=False)
    yield
    pruning_factory.set_prunable_scope(prune_task_head=False, prune_embeddings=False)


@pytest.fixture(autouse=True)
def seeded():
    """Seed every test.

    Not defensive boilerplate: RigLPruner and EASTPruner both initialise masks
    with torch.bernoulli, so their tests are genuinely stochastic without this.
    """
    set_seed(0)


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    """Force the CPU path.

    Both dataclasses pick their device with
    ``"cuda" if torch.cuda.is_available() else "cpu"``. Pinning this to False
    means a test behaves identically on a laptop and on the Databricks GPU box,
    so a green suite locally means the same thing there.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


# --- Model fixtures ------------------------------------------------------

@pytest.fixture
def tiny_backbone():
    return make_tiny_resnet(num_classes=NUM_CLASSES)


@pytest.fixture
def register_tiny_model(monkeypatch):
    """Register the tiny model in model_factory.MODELS under the name 'tiny'.

    This routes construction through the *real* ClassificationAndEncoderNetwork,
    so _get_embedded_dim_from_model, remove_last_layer, make_classification_head
    and the encoder_head wiring are all exercised rather than stubbed.
    """
    # _local_vision normalises the call signature: builders are invoked as
    # builder(num_classes=..., dyrelu_en=..., dyrelu_phasing_en=...), which is what
    # lets HuggingFace models coexist in the same registry as this project's own.
    spec = ModelSpec(builder=model_factory._local_vision(make_tiny_resnet),
                     weight="", family="resnet", type="cv")
    monkeypatch.setitem(model_factory.MODELS, TINY, spec)
    return TINY


def _build_wrapper(dyrelu_en=False, dyrelu_phasing_en=False, pretrained=False):
    # adapt=False is required: adapt_resnet_for_small_images hardcodes
    # nn.Conv2d(3, 64, ...), which would emit 64 channels into a bn1 expecting 8.
    return ClassificationAndEncoderNetwork(
        model_name=TINY,
        num_classes=NUM_CLASSES,
        num_out_features=NUM_OUT_FEATURES,
        device="cpu",
        adapt=False,
        pretrained=pretrained,
        dyrelu_en=dyrelu_en,
        dyrelu_phasing_en=dyrelu_phasing_en,
    )


@pytest.fixture
def tiny_wrapper(register_tiny_model):
    return _build_wrapper()


@pytest.fixture
def tiny_wrapper_dyrelu(register_tiny_model):
    """Wrapper whose BasicBlocks hold real DyReLUAdapter modules."""
    return _build_wrapper(dyrelu_phasing_en=True)


@pytest.fixture
def tiny_wrapper_shared(register_tiny_model):
    """Wrapper with EAST weight sharing already applied (R=2)."""
    wrapper = _build_wrapper()
    apply_weight_sharing_resnet(wrapper)
    return wrapper


# --- Data fixtures -------------------------------------------------------

@pytest.fixture
def loaders():
    """Factory: loaders(n_views=1) -> dict shaped like load_cv_dataloaders."""
    def _make(n_views=1, batch_size=BATCH_SIZE, n=16, size=IMAGE_SIZE):
        return make_loaders(n_views=n_views, batch_size=batch_size, n=n, size=size,
                            num_classes=NUM_CLASSES)
    return _make


# --- The two seams -------------------------------------------------------

@pytest.fixture
def patch_models(monkeypatch, register_tiny_model):
    """Replace _create_base_model so _initialize_models builds tiny models.

    Patch the name in training_utils, which is where it is defined and called.
    That _create_base_model was already factored out of _initialize_models is
    the reason this seam exists at all -- do not inline it back.
    """
    def _fake_create(args, is_main_model=False):
        dyrelu_en = getattr(args, "dyrelu_en", False) if is_main_model else False
        phasing = getattr(args, "dyrelu_phasing_en", False) if is_main_model else False
        return _build_wrapper(dyrelu_en=dyrelu_en, dyrelu_phasing_en=phasing)

    monkeypatch.setattr(training_utils, "_create_base_model", _fake_create)


@pytest.fixture
def patch_dataloaders(monkeypatch):
    """Replace both data-loading entry points with synthetic in-memory loaders.

    Two seams are needed, not one:

      training_utils.get_dataloaders -- used by _initialize_data_loaders. Patch the
          binding in training_utils, not dataset_factory: training_utils did
          ``from dataset_factory import get_dataloaders``, so that module-level name
          is what actually resolves.

      bacp.load_cv_dataloaders -- BaCPTrainer.finetune() bypasses get_dataloaders
          entirely and calls load_cv_dataloaders directly to rebuild supervised
          (n_views=1) loaders. Without this seam, any test touching finetune()
          downloads CIFAR-10. Worth knowing beyond the tests: the fine-tuning
          loaders take a different code path from the training ones, and that path
          is hardcoded to the CV branch.
    """
    def _fake_get(args, supervised_override=False):
        # supervised_override mirrors the real signature: finetune() uses it to force
        # single-view supervised loaders even while is_bacp is still True.
        is_bacp = getattr(args, "is_bacp", False) and not supervised_override
        n_views = (getattr(args, "n_views", 1) or 1) if is_bacp else 1
        return make_loaders(n_views=n_views, batch_size=args.batch_size, n=16,
                            size=args.image_size, num_classes=args.num_classes)

    monkeypatch.setattr(training_utils, "get_dataloaders", _fake_get)

    # bacp.py holds its own binding (used by finetune()), so both need patching.
    import bacp
    monkeypatch.setattr(bacp, "get_dataloaders", _fake_get)


# --- Argument objects ----------------------------------------------------

def _base_arg_dict(is_bacp=False, **overrides):
    """The full attribute contract the eight _initialize_* helpers read.

    Grouped by who writes each attribute. test_fake_args_covers_dataclass_fields
    asserts this stays a superset of the real dataclass fields, so the contract
    is checked against the source of truth rather than hand-maintained.
    """
    d = dict(
        # -- declared dataclass fields
        model_name=TINY, model_type="cv", dataset_name="cifar10",
        num_classes=NUM_CLASSES, num_out_features=NUM_OUT_FEATURES,
        image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, num_workers=0,
        epochs=2, recovery_epochs=0,
        # Present on BOTH dataclasses: the I.P. path gained a fine-tune phase
        # so it can be run on the same budget as BaCP.
        enable_finetune=False, optimizer_type_ft='adamw',
        learning_rate_ft=1e-4, epochs_ft=0,
        optimizer_type="sgd", learning_rate=0.1,
        scheduler_type=None, patience=None, trained_weights=None,
        experiment_type="test",
        log_epochs=False, enable_tqdm=False, enable_mixed_precision=False,
        databricks_env=False,
        pruning_type=None, target_sparsity=None, sparsity_scheduler=None,
        pruning_module=None, delta_T=1,
        dyrelu_en=False, dyrelu_phasing_en=False, weight_sharing_en=False,
        prune_task_head=False, prune_embeddings=False,
        layerwise_alloc="erk", wanda_group="output", weight_decay=5e-4,
        # Zero = no truncation. The tiny fixtures are already small; setting a
        # limit here would mean the loop tests exercise _LimitedLoader instead of
        # the real loader, which is the opposite of what they are for.
        limit_train_batches=0, limit_eval_batches=0,
        seed=0, val_split_seed=1234, deterministic=False,
        # -- set by __post_init__ before the helpers run
        scaler=None, device="cpu", retrain=False, is_bacp=is_bacp,
        # -- read by Trainer.__init__ / _optimizer_step / _log_metrics
        recover=False, current_sparsity=0.0,
    )
    if is_bacp:
        d.update(
            tau=0.07, lambdas=None, learnable_lambdas=False, n_views=2,
            enable_finetune=False, epochs_ft=1,
            optimizer_type_ft="sgd", learning_rate_ft=0.005,
            contrastive_mode="cap", proj_mode="tied_frozen", num_snapshots=2,
            distill_mode="none", lambda_kd=0.0, kd_temperature=4.0,
            feature_distill_metric="cosine",
        )
    d.update(overrides)
    return d


@pytest.fixture
def fake_args():
    """Factory for a SimpleNamespace standing in for either dataclass.

    Use this to drive the _initialize_* helpers in isolation. Use real_args when
    the test needs the genuine dataclass (field contract, __post_init__ order).
    """
    def _make(is_bacp=False, **overrides):
        return SimpleNamespace(**_base_arg_dict(is_bacp=is_bacp, **overrides))
    return _make


@pytest.fixture(autouse=True)
def _isolate_results_dir(tmp_path, monkeypatch):
    """Keep run records out of the real results/ directory.

    results.results_dir() resolves from the REPO ROOT rather than the CWD -- which
    is correct for production, because notebooks shell out to ../scripts/*.py from
    a subdirectory and a relative 'results/' would land somewhere different
    depending on who called. The consequence is that monkeypatch.chdir(tmp_path)
    does NOT redirect it, so every test that exercised record_run wrote a real
    record into the production directory. 71 tiny-model fixture records had
    accumulated there before this fixture existed.

    They were harmless to the tables (experiment_group is None, so
    ladder.load_ladder never joins them) but they polluted the
    `pytest -m results` dashboard, and a directory where fixtures and results
    are indistinguishable is one bad afternoon away from a fixture being cited.

    autouse, because opting in per-test is exactly the discipline that failed.
    """
    monkeypatch.setenv('BACP_RESULTS_DIR', str(tmp_path / 'results'))
    yield


@pytest.fixture
def real_args(tmp_path, monkeypatch, patch_models, patch_dataloaders):
    """Factory for the genuine dataclasses with auto_init=False.

    auto_init short-circuits __post_init__ after the cheap prefix (scaler,
    device, retrain, is_bacp) and before _initialize_all, which is what makes
    the dataclasses constructible without a GPU or /dbfs. chdir to tmp_path so
    Logger's os.makedirs and the './research/bacp/...' save paths stay inside
    the test sandbox.
    """
    monkeypatch.chdir(tmp_path)

    def _make(kind="baseline", **overrides):
        if kind == "bacp":
            from bacp import BaCPTrainingArguments as Cls
            defaults = _base_arg_dict(is_bacp=True)
        else:
            from trainer import TrainingArguments as Cls
            defaults = _base_arg_dict(is_bacp=False)

        allowed = {f.name for f in fields(Cls)}
        kwargs = {k: v for k, v in defaults.items() if k in allowed}
        kwargs.update({k: v for k, v in overrides.items() if k in allowed})
        kwargs["auto_init"] = False
        return Cls(**kwargs)

    return _make


@pytest.fixture
def initialized_args(real_args):
    """real_args plus the full eight-helper initialisation sequence."""
    def _make(kind="baseline", **overrides):
        args = real_args(kind, **overrides)
        training_utils._initialize_all(args)
        return args
    return _make


# --- Pruner helpers ------------------------------------------------------

@pytest.fixture
def pruner_kwargs():
    """The canonical kwargs every pruner construction in L4/L5/L6c uses.

    BasePruner raises unless scheduler_type, delta_T and total_steps are all
    provided, so there is no shorter valid form.
    """
    def _make(**overrides):
        kw = dict(scheduler_type="cubic", delta_T=1, total_steps=8,
                  Tc_ratio=0.5, s_min=0.05, prune_rate=0.1, alpha=0.3)
        kw.update(overrides)
        return kw
    return _make


class SpyPruner:
    """Duck-typed pruner recording how the training loop drives it.

    Lets the gating logic in _step_pruning_step (delta_T, end_idx, recover) be
    tested independently of any real masking behaviour.
    """

    def __init__(self, end_idx=1000):
        self.end_idx = end_idx
        self.masks = {}
        self.step_calls = []
        self.apply_calls = 0

    def step(self, current_idx, optimizer=None):
        self.step_calls.append((current_idx, optimizer is not None))

    def apply_mask(self):
        self.apply_calls += 1


@pytest.fixture
def spy_pruner():
    return SpyPruner
