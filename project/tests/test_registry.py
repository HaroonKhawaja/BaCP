"""L0b -- model and dataset registry contracts, plus the MLM tensor shapes.

Everything here runs offline. The HuggingFace builders are checked structurally
(signature, dyrelu rejection, declared metadata) rather than by construction,
since instantiating them needs a hub download; the shape handling that the
language path depends on is tested directly on synthetic tensors.
"""

import inspect
from types import SimpleNamespace

import pytest
import torch

import model_factory
from dataset_factory import CV_DATASETS, DATASET_NUM_CLASSES, TEXT_DATASETS
from model_factory import MODELS, ModelSpec

PAPER_MODELS = [
    "resnet50", "resnet101", "vgg11", "vgg19",
    "vit-tiny", "vit-small",
    "distilbert-base-uncased", "roberta-base",
    "distilbert-base-uncased-mlm", "roberta-base-mlm",
]

PAPER_DATASETS = [
    "cifar10", "cifar100", "mnist", "fmnist", "emnist", "svhn",
    "sst2", "wikitext2",
]


# --- Coverage -------------------------------------------------------------

@pytest.mark.parametrize("name", PAPER_MODELS)
def test_every_model_in_the_paper_is_registered(name):
    assert name in MODELS, f"{name} appears in the paper but is not registered"


@pytest.mark.parametrize("name", PAPER_DATASETS)
def test_every_dataset_in_the_paper_is_registered(name):
    assert name in CV_DATASETS or name in TEXT_DATASETS


def test_cv_and_text_registries_are_disjoint():
    assert not (set(CV_DATASETS) & set(TEXT_DATASETS))


# --- Spec contract --------------------------------------------------------

@pytest.mark.parametrize("name", sorted(MODELS))
def test_spec_fields_are_well_formed(name):
    spec = MODELS[name]
    assert isinstance(spec, ModelSpec)
    assert spec.type in ("cv", "llm")
    assert spec.family in ("resnet", "vgg", "mobilenet", "vit", "bert")
    assert spec.task in ("classification", "mlm")
    assert callable(spec.builder)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_every_builder_accepts_the_uniform_signature(name):
    """All builders must take the same three keyword arguments.

    This uniformity is the whole point of the builder indirection:
    initialize_model_components calls every spec identically. Previously it called
    `spec.fn(dyrelu_en=..., dyrelu_phasing_en=...)`, which no HuggingFace
    constructor accepts -- so the ViT and language entries could not be registered
    at all.
    """
    params = inspect.signature(MODELS[name].builder).parameters
    for expected in ("num_classes", "dyrelu_en", "dyrelu_phasing_en"):
        assert expected in params, f"{name}'s builder is missing {expected}"


def test_mlm_specs_are_language_models():
    for name, spec in MODELS.items():
        if spec.task == "mlm":
            assert spec.type == "llm"
            assert name.endswith("-mlm")


def test_local_vision_specs_carry_a_checkpoint_path():
    """Local backbones need an explicit weights path; HF models do not.

    HF builders arrive pretrained from the hub, so their `weight` is empty and
    initialize_model_components skips load_weights for them.
    """
    for name, spec in MODELS.items():
        if spec.family in ("resnet", "vgg", "mobilenet"):
            assert spec.weight, f"{name} has no checkpoint path"
        else:
            assert spec.weight == "", f"{name} should not carry a local path"


# --- DyReLU is vision-only -----------------------------------------------

@pytest.mark.parametrize("name", ["vit-tiny", "roberta-base", "distilbert-base-uncased-mlm"])
@pytest.mark.parametrize("flag", ["dyrelu_en", "dyrelu_phasing_en"])
def test_hf_builders_reject_dyrelu(name, flag):
    """Asking for DyReLU on a HuggingFace model must fail, not be ignored.

    DyReLU phasing is threaded into this project's own BasicBlock/VGG stacks. A
    silent no-op would train a plain-ReLU network while the logs claimed phasing was
    active -- the worst outcome, because the result looks valid.
    """
    with pytest.raises(ValueError, match="DyReLU is not supported"):
        MODELS[name].builder(num_classes=10, **{flag: True})


@pytest.mark.parametrize("name", ["resnet50", "vgg19"])
def test_local_builders_accept_dyrelu(name):
    """Sanity check the other direction: the vision path must still take the flag."""
    params = inspect.signature(MODELS[name].builder).parameters
    assert params["dyrelu_phasing_en"].default is False


# --- Class counts --------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("cifar10", 10), ("cifar100", 100), ("mnist", 10), ("fmnist", 10),
    ("emnist", 47), ("svhn", 10), ("imagenet", 1000), ("sst2", 2),
])
def test_declared_class_counts(name, expected):
    """EMNIST is 47 because it is loaded with split='balanced', matching the
    "EMNIST (Balanced)" row in the paper. Getting this wrong sizes the
    classification head incorrectly and is not otherwise detectable."""
    assert DATASET_NUM_CLASSES[name] == expected


# --- MLM tensor shapes ---------------------------------------------------

def _bacp_stub():
    """A stand-in carrying only what the two shape helpers touch."""
    from bacp import BaCPTrainer
    return SimpleNamespace(
        _task_loss=BaCPTrainer._task_loss.__get__(SimpleNamespace()),
        _task_accuracy=BaCPTrainer._task_accuracy.__get__(SimpleNamespace()),
    )


def test_task_loss_handles_classification_logits():
    stub = _bacp_stub()
    logits = torch.randn(4, 10)
    labels = torch.randint(0, 10, (4,))
    assert torch.isfinite(stub._task_loss(logits, labels))


def test_task_loss_handles_mlm_logits_and_ignores_minus_100():
    """MLM gives [B, T, V] against [B, T] with -100 on unmasked positions.

    The tensors must be flattened before cross-entropy, and -100 must be ignored --
    the convention DataCollatorForLanguageModeling writes. Passing 3-D logits
    straight to nn.CrossEntropyLoss would either raise or silently score the wrong
    axis.
    """
    stub = _bacp_stub()
    logits = torch.randn(2, 6, 50)
    labels = torch.full((2, 6), -100)
    labels[0, 1] = 7
    labels[1, 4] = 12

    loss = stub._task_loss(logits, labels)
    assert torch.isfinite(loss) and loss > 0


def test_task_loss_with_every_position_ignored_is_zero_not_nan():
    """A batch with no masked tokens must yield 0.0, not NaN.

    CrossEntropyLoss(ignore_index=-100) averages over zero elements and returns NaN,
    which propagates through backward and destroys every weight -- so this cannot be
    left to "it won't happen at 15% masking".
    """
    stub = _bacp_stub()
    loss = stub._task_loss(torch.randn(2, 6, 50), torch.full((2, 6), -100))
    assert torch.isfinite(loss), "an all-ignored batch produced a non-finite loss"
    assert loss.item() == 0.0


def test_task_accuracy_classification():
    stub = _bacp_stub()
    logits = torch.tensor([[9.0, 0.0], [0.0, 9.0], [9.0, 0.0], [9.0, 0.0]])
    labels = torch.tensor([0, 1, 0, 1])
    assert stub._task_accuracy(logits, labels).item() == pytest.approx(75.0)


def test_task_accuracy_mlm_scores_masked_positions_only():
    """Accuracy over the masked tokens, not over the whole sequence.

    Scoring all positions would report ~100% simply because most of the sequence is
    unmasked and excluded from the objective.
    """
    stub = _bacp_stub()
    logits = torch.zeros(1, 4, 3)
    logits[0, 1, 2] = 5.0   # predicts class 2 at position 1
    logits[0, 3, 0] = 5.0   # predicts class 0 at position 3

    labels = torch.full((1, 4), -100)
    labels[0, 1] = 2        # correct
    labels[0, 3] = 1        # wrong

    assert stub._task_accuracy(logits, labels).item() == pytest.approx(50.0)


def test_task_accuracy_mlm_with_no_masked_tokens_is_zero_not_nan():
    stub = _bacp_stub()
    acc = stub._task_accuracy(torch.randn(1, 4, 3), torch.full((1, 4), -100))
    assert acc.item() == 0.0


# --- View splitting ------------------------------------------------------

def test_split_views_unpacks_two_image_views():
    from bacp import BaCPTrainer
    split = BaCPTrainer._split_views.__get__(SimpleNamespace())
    a, b = split([torch.zeros(2, 3, 8, 8), torch.ones(2, 3, 8, 8)])
    assert not torch.equal(a, b)


def test_split_views_duplicates_a_text_batch():
    """A dict batch has no second view, and duplicating it is the correct behaviour.

    In CAP the two "views" of a text example are the same tokens seen through two
    different models -- the pruned student and a frozen teacher -- which is exactly
    what the cross-model term compares. No text augmentation is required.
    """
    from bacp import BaCPTrainer
    split = BaCPTrainer._split_views.__get__(SimpleNamespace())
    batch = {"input_ids": torch.zeros(2, 4, dtype=torch.long)}
    a, b = split(batch)
    assert a is batch and b is batch


def test_split_views_rejects_a_wrong_view_count():
    from bacp import BaCPTrainer
    split = BaCPTrainer._split_views.__get__(SimpleNamespace())
    with pytest.raises(ValueError, match="Expected 2 augmented views"):
        split([torch.zeros(1), torch.zeros(1), torch.zeros(1)])
