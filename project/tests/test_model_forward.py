"""L3 -- model wrapper and weight loading. Tiny model, ~200ms.

The wrapper has three output modes and they must stay distinguishable: logits for
the task, features for the backbone representation, and normalized embeddings for
the contrastive losses.
"""

import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace

from model_factory import (
    _get_embedded_dim_from_model,
    adapt_resnet_for_small_images,
    remove_last_layer,
)
from utils import load_weights

BATCH = 4
NUM_CLASSES = 4
NUM_OUT_FEATURES = 8


def _x(batch=BATCH):
    return torch.randn(batch, 3, 8, 8)


def test_wrapper_three_output_modes(tiny_wrapper):
    logits = tiny_wrapper(_x())
    feats = tiny_wrapper(_x(), return_feat=True)
    emb = tiny_wrapper(_x(), return_emb=True)

    assert logits.shape == (BATCH, NUM_CLASSES)
    assert feats.shape == (BATCH, tiny_wrapper.embedded_dim)
    assert emb.shape == (BATCH, NUM_OUT_FEATURES)


def test_get_embeddings_is_l2_normalized(tiny_wrapper):
    """The contrastive losses take z @ z.T as cosine similarity, which is only
    true if the rows are unit norm."""
    emb = tiny_wrapper(_x(), return_emb=True)
    norms = emb.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_return_feat_and_return_emb_are_not_the_same_tensor(tiny_wrapper):
    x = _x()
    assert tiny_wrapper(x, return_feat=True).shape != tiny_wrapper(x, return_emb=True).shape


def test_cls_head_and_encoder_head_are_distinct_parameters(tiny_wrapper):
    cls_ids = {id(p) for p in tiny_wrapper.cls_head.parameters()}
    enc_ids = {id(p) for p in tiny_wrapper.encoder_head.parameters()}
    assert cls_ids and enc_ids and not (cls_ids & enc_ids)


def test_embedded_dim_inferred_from_fc(tiny_backbone):
    assert _get_embedded_dim_from_model(tiny_backbone) == tiny_backbone.fc.in_features


def test_remove_last_layer_replaces_fc_with_identity(tiny_backbone):
    stripped = remove_last_layer(tiny_backbone)
    assert isinstance(stripped.fc, nn.Identity)


def test_get_embeddings_and_logits_matches_manual(tiny_wrapper):
    """Pins that logits come from cls_head and embeddings from encoder_head, both
    off the *same* forward pass -- not two independent passes."""
    from bacp import BaCPTrainer

    stand_in = SimpleNamespace(model=tiny_wrapper, model_type="cv")
    x = _x()
    tiny_wrapper.eval()
    with torch.no_grad():
        emb, logits = BaCPTrainer._get_embeddings_and_logits(stand_in, x)
        feats = tiny_wrapper(x, return_feat=True)
        assert torch.allclose(logits, tiny_wrapper.cls_head(feats), atol=1e-6)
        assert torch.allclose(emb, tiny_wrapper.get_embeddings(feats), atol=1e-6)


# --- DyReLU shape handling -----------------------------------------------

def test_dyrelu_b_forward_4d_preserves_shape():
    from dyrelu_adapter import DyReLUB
    x = torch.randn(2, 8, 4, 4)
    assert DyReLUB(8)(x).shape == x.shape


def test_dyrelu_b_forward_2d_preserves_shape():
    """The VGG classifier path: DyReLUB(4096) receives [B, 4096], not [B,C,H,W].

    get_relu_coefs assigned `thera = x` on this branch -- a typo that left `theta`
    unbound, so any 2-D input raised UnboundLocalError. VGG plus DyReLU could
    never run.
    """
    from dyrelu_adapter import DyReLUB
    x = torch.randn(2, 8)
    assert DyReLUB(8)(x).shape == x.shape


# --- load_weights --------------------------------------------------------

def test_load_weights_missing_path_reports_failure(capsys, tiny_wrapper):
    """A missing checkpoint must return False and must NOT claim success.

    The success line used to be printed *before* torch.load was called, so the
    logs read "Full load successful!" on runs that had loaded nothing. Combined
    with the disabled raise, a run asking for pretrained weights would train from
    random init while appearing to have loaded them -- and every checkpoint path
    in MODELS is a /dbfs path, so this is the default outcome off Databricks.
    """
    assert load_weights(tiny_wrapper, "no-such-file.pt") is False
    assert "successful" not in capsys.readouterr().out.lower()


def test_load_weights_none_path_returns_false(tiny_wrapper):
    """None must be rejected rather than reaching os.path.exists(None)."""
    assert load_weights(tiny_wrapper, None) is False


def test_load_weights_round_trip(tmp_path, tiny_wrapper, capsys):
    path = tmp_path / "ckpt.pt"
    torch.save(tiny_wrapper.state_dict(), path)

    with torch.no_grad():
        target = next(iter(tiny_wrapper.cls_head.parameters()))
        original = target.clone()
        target.add_(1.0)

    assert load_weights(tiny_wrapper, str(path)) is True
    assert "successful" in capsys.readouterr().out.lower()
    assert torch.allclose(next(iter(tiny_wrapper.cls_head.parameters())), original)


def test_load_weights_partial_load_reports_key_counts(tmp_path, tiny_wrapper, capsys):
    """A checkpoint missing the heads falls back to a strict=False partial load.

    That fallback is legitimate, but it must be loud: strict=False tolerates any
    key mismatch, so a silent partial load is indistinguishable from a typo in a
    parameter name.
    """
    state = {k: v for k, v in tiny_wrapper.state_dict().items()
             if "cls_head" not in k and "encoder_head" not in k}
    path = tmp_path / "headless.pt"
    torch.save(state, path)

    assert load_weights(tiny_wrapper, str(path)) is True
    out = capsys.readouterr().out
    assert "Partial load" in out
    assert "tensors loaded" in out and "missing" in out
    assert "accepted" in out


def test_load_weights_rejects_a_barely_matching_checkpoint(tmp_path, tiny_wrapper, capsys):
    """A checkpoint that matches almost nothing must NOT count as loaded.

    strict=False tolerates any number of missing keys, so an unguarded True
    here recorded init_source='imagenet_checkpoint' on an effectively random
    model. Below half the model's tensors, load_weights now returns False.
    """
    one_key = next(iter(tiny_wrapper.state_dict()))
    state = {one_key: tiny_wrapper.state_dict()[one_key]}
    path = tmp_path / "nearly_empty.pt"
    torch.save(state, path)

    assert load_weights(tiny_wrapper, str(path)) is False
    out = capsys.readouterr().out
    assert "REJECTED" in out


# --- Documented constraint ----------------------------------------------

def test_adapt_resnet_hardcodes_64_channels(tiny_backbone):
    """Records why the test wrapper is built with adapt=False.

    adapt_resnet_for_small_images replaces conv1 with a hardcoded
    nn.Conv2d(3, 64, ...) regardless of the backbone's actual width, so on any
    model whose stem is not 64 channels it desynchronises conv1 from bn1. This is
    a real limitation of the helper, recorded rather than worked around.
    """
    assert tiny_backbone.bn1.num_features == 8
    adapt_resnet_for_small_images(tiny_backbone)
    assert tiny_backbone.conv1.out_channels == 64

    with pytest.raises(RuntimeError):
        tiny_backbone(_x())
