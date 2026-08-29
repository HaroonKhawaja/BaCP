from dataclasses import dataclass

import torch.nn as nn
import torch.nn.functional as F

from utils import load_weights

from models.resnet import resnet34, resnet50, resnet101
from models.vgg import vgg11 as local_vgg11, vgg19 as local_vgg19


# --- Model registry -------------------------------------------------------
#
# Every entry carries a `builder` rather than a bare constructor. The reason is
# that the local vision factories and the HuggingFace factories have
# incompatible signatures: the former take dyrelu_en / dyrelu_phasing_en, which
# no HF constructor accepts. initialize_model_components used to call
# `spec.fn(dyrelu_en=..., dyrelu_phasing_en=...)` unconditionally, which is why
# the ViT and language-model entries could not simply be uncommented.
#
# `type` is 'cv' or 'llm', matching --model_type on the CLI and the checks in
# Trainer._handle_metrics / dataset_factory.get_dataloaders. It previously held
# 'vision', giving two vocabularies for one concept.

@dataclass(frozen=True)
class ModelSpec:
    builder: object   # callable(num_classes, dyrelu_en, dyrelu_phasing_en) -> nn.Module
    weight: str       # local checkpoint path; '' when the builder supplies pretrained weights
    family: str       # 'resnet' | 'vgg' | 'vit' | 'bert'
    type: str         # 'cv' | 'llm'
    task: str = 'classification'   # 'classification' | 'mlm'


def _local_vision(fn):
    """Wrap one of this project's own vision factories.

    num_classes is ignored: the task head is stripped by remove_last_layer and
    replaced with a fresh cls_head sized from num_classes downstream.
    """
    def build(num_classes=1000, dyrelu_en=False, dyrelu_phasing_en=False):
        return fn(dyrelu_en=dyrelu_en, dyrelu_phasing_en=dyrelu_phasing_en)
    return build


def _hf_image(checkpoint):
    def build(num_classes=1000, dyrelu_en=False, dyrelu_phasing_en=False):
        _reject_dyrelu(checkpoint, dyrelu_en, dyrelu_phasing_en)
        from transformers import AutoModelForImageClassification
        return AutoModelForImageClassification.from_pretrained(
            checkpoint, num_labels=num_classes, ignore_mismatched_sizes=True,
        )
    return build


def _hf_seqcls(checkpoint):
    def build(num_classes=2, dyrelu_en=False, dyrelu_phasing_en=False):
        _reject_dyrelu(checkpoint, dyrelu_en, dyrelu_phasing_en)
        from transformers import AutoModelForSequenceClassification
        return AutoModelForSequenceClassification.from_pretrained(
            checkpoint, num_labels=num_classes,
        )
    return build


def _hf_mlm(checkpoint):
    def build(num_classes=None, dyrelu_en=False, dyrelu_phasing_en=False):
        _reject_dyrelu(checkpoint, dyrelu_en, dyrelu_phasing_en)
        from transformers import AutoModelForMaskedLM
        return AutoModelForMaskedLM.from_pretrained(checkpoint)
    return build


def _torchvision_cv(ctor_name):
    """Wrap a stock torchvision CV builder that has no DyReLU hook.

    Unlike this project's own ResNet/VGG ports (_local_vision), torchvision's
    native constructors do not accept dyrelu_en/dyrelu_phasing_en -- there is
    no equivalent hook wired into their blocks, the same reason HF models
    reject it via _reject_dyrelu. Built with weights=None: ImageNet weights are
    loaded afterwards from the registry's checkpoint path, same as every other
    'cv' model here, not from torchvision's own weights enum.
    """
    def build(num_classes=1000, dyrelu_en=False, dyrelu_phasing_en=False):
        _reject_dyrelu(ctor_name, dyrelu_en, dyrelu_phasing_en)
        import torchvision.models as tvm
        return getattr(tvm, ctor_name)(weights=None)
    return build


def _reject_dyrelu(checkpoint, dyrelu_en, dyrelu_phasing_en):
    """Fail loudly rather than silently ignoring an EAST flag.

    DyReLU phasing is implemented by threading flags into this project's own
    BasicBlock/VGG stacks. A HuggingFace model has no such hook, so a run asking
    for it would otherwise train a plain-ReLU network while the logs claimed
    phasing was active.
    """
    if dyrelu_en or dyrelu_phasing_en:
        raise ValueError(
            f"DyReLU is not supported for '{checkpoint}'. It is wired into this "
            "project's own ResNet/VGG blocks; HuggingFace models have no equivalent "
            "hook. Drop --dyrelu_en / --dyrelu_phasing_en for this model."
        )


MODELS = {
    # Architecture sources:
    #   ResNet      He et al., CVPR 2016, arXiv 1512.03385 (models/resnet.py)
    #   VGG         Simonyan & Zisserman, ICLR 2015, arXiv 1409.1556 (models/vgg.py)
    #   ViT         Dosovitskiy et al., ICLR 2021, arXiv 2010.11929. The
    #               Tiny/Small widths come from DeiT (Touvron et al. 2021,
    #               arXiv 2012.12877); the WinKawaks checkpoints are timm
    #               conversions whose training recipe this project has NOT
    #               verified -- check the model card before citing it
    #   DistilBERT  Sanh, Debut, Chaumond & Wolf, 2019, arXiv 1910.01108
    #   RoBERTa     Liu et al., 2019, arXiv 1907.11692

    # --- Vision: this project's own backbones (DyReLU-capable) ---
    'resnet34':   ModelSpec(_local_vision(resnet34),   '/dbfs/research/bacp/resnet34_imagenet.pth',  'resnet', 'cv'),
    'resnet50':   ModelSpec(_local_vision(resnet50),   '/dbfs/research/bacp/resnet50_imagenet.pth',  'resnet', 'cv'),
    'resnet101':  ModelSpec(_local_vision(resnet101),  '/dbfs/research/bacp/resnet101_imagenet.pth', 'resnet', 'cv'),
    'vgg11':      ModelSpec(_local_vision(local_vgg11), '/dbfs/research/bacp/vgg11_imagenet.pth',    'vgg',    'cv'),
    'vgg19':      ModelSpec(_local_vision(local_vgg19), '/dbfs/research/bacp/vgg19_imagenet.pth',    'vgg',    'cv'),
    'mobilenet_v2': ModelSpec(_torchvision_cv('mobilenet_v2'),
                              '/dbfs/research/bacp/mobilenet_v2_imagenet.pth',
                              'mobilenet', 'cv'),

    # --- Vision Transformers (weights from the HF hub; no DyReLU) ---
    # Expect --image_size 224. ignore_mismatched_sizes lets the 1000-way
    # ImageNet head be replaced by a CIFAR-sized one.
    'vit-tiny':   ModelSpec(_hf_image('WinKawaks/vit-tiny-patch16-224'),  '', 'vit', 'cv'),
    'vit-small':  ModelSpec(_hf_image('WinKawaks/vit-small-patch16-224'), '', 'vit', 'cv'),

    # --- Language models: sequence classification (SST-2) ---
    'distilbert-base-uncased': ModelSpec(_hf_seqcls('distilbert-base-uncased'), '', 'bert', 'llm'),
    'roberta-base':            ModelSpec(_hf_seqcls('roberta-base'),            '', 'bert', 'llm'),

    # --- Language models: masked language modelling (WikiText-2) ---
    'distilbert-base-uncased-mlm': ModelSpec(_hf_mlm('distilbert-base-uncased'), '', 'bert', 'llm', task='mlm'),
    'roberta-base-mlm':            ModelSpec(_hf_mlm('roberta-base'),            '', 'bert', 'llm', task='mlm'),
}

PRETRAINED = True
DYRELU_ENABLED = False


def _get_embedded_dim_from_model(model):
    """Returns the model's final embedded dimension"""
    if hasattr(model, 'fc') and hasattr(model.fc, 'in_features'):
        return model.fc.in_features

    if hasattr(model, 'classifier'):
        cls_head = model.classifier
        if isinstance(cls_head, nn.Sequential):
            return cls_head[-1].in_features
        elif hasattr(cls_head, 'in_features'):
            return cls_head.in_features

    if hasattr(model, 'head') and hasattr(model.head, 'in_features'):
        return model.head.in_features

    # HF models: ViT, BERT and RoBERTa all expose hidden_size. Checked last so a
    # concrete head wins when one exists.
    if hasattr(model, 'config') and hasattr(model.config, 'hidden_size'):
        return model.config.hidden_size

    raise RuntimeError(f"Couldn't infer embedding dim for model: {model.__class__.__name__}")


def initialize_model_components(model_name: str, pretrained: bool, dyrelu_en: bool,
                                dyrelu_phasing_en: bool, num_classes: int = 1000):
    if model_name not in MODELS:
        raise ValueError(f"Unknown model {model_name}. Choices: {sorted(MODELS)}")

    spec = MODELS[model_name]
    model = spec.builder(num_classes=num_classes,
                         dyrelu_en=dyrelu_en,
                         dyrelu_phasing_en=dyrelu_phasing_en)

    # HF builders arrive already pretrained, so spec.weight is empty for them.
    #
    # `init_source` records what the weights ACTUALLY are, not what was asked
    # for. load_weights fails soft by design -- a missing checkpoint warns and
    # the run continues -- so on any machine without spec.weight on disk (a
    # fresh GPU box; every path here points into /dbfs) a run that requested
    # ImageNet init silently gets random init. Recording `pretrained=True`
    # regardless, which is what results.py did, turns that into a provenance lie
    # in the paper: the table would claim a pretrained baseline that was never
    # pretrained.
    #
    # Worth knowing which you want: RigL, GraNet and EAST all train CIFAR from
    # SCRATCH, so 'random' is the setting that makes this project's numbers
    # comparable to their published ones. 'imagenet_checkpoint' is a deviation
    # and has to be declared as one.
    init_source = 'random'
    if spec.weight == '':
        init_source = 'hf_hub'                # arrived pretrained from the hub
    elif pretrained and spec.weight:
        init_source = ('imagenet_checkpoint'
                       if load_weights(model, spec.weight) else 'random')

    return {
        'model':        model,
        'embedded_dim': _get_embedded_dim_from_model(model),
        'model_type':   spec.type,
        'model_family': spec.family,
        'model_task':   spec.task,
        'init_source':  init_source,
    }


def adapt_resnet_for_small_images(model):
    if hasattr(model, 'conv1'):
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)


def adapt_mobilenet_for_small_images(model):
    """CIFAR-scale stem fix for MobileNetV2, mirroring the ResNet one.

    features[0] is torchvision's Conv2dNormActivation -- an nn.Sequential of
    (Conv2d(3,32,k=3,s=2,p=1), BatchNorm2d, ReLU6), confirmed by inspection.
    Only the stride is wrong for 32x32 input, so only the stride changes --
    kernel size, padding and the surrounding BatchNorm/activation are left
    exactly as ImageNet pretraining produced them, same principle as the
    ResNet stem fix (kernel/channels preserved, only the resolution-discarding
    stride removed).
    """
    if hasattr(model, 'features') and len(model.features) > 0:
        conv = model.features[0][0]
        if isinstance(conv, nn.Conv2d):
            conv.stride = (1, 1)


def make_classification_head(embedded_dim: int, num_out_features: int):
    return nn.Linear(embedded_dim, num_out_features)


def adapt_head_for_model(model, head: nn.Module, model_type: str, model_family: str):
    if model_type == 'cv':
        if hasattr(model, 'fc'):
            model.fc = head
            return

        if hasattr(model, 'classifier'):
            cls_head = model.classifier
            if isinstance(cls_head, nn.Sequential):
                cls_head[-1] = head
            else:
                model.classifier = head
            return

        if hasattr(model, 'head'):
            model.head = head
            return

        raise RuntimeError("Couldn't attach head to vision model: " + model.__class__.__name__)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def remove_last_layer(model):
    """Replace the task head with Identity so forward() yields features.

    Dispatches on the head attribute rather than on `named_children()[-1]`.
    The positional version broke on HF sequence-classification models, whose last
    registered child is a Dropout rather than the classifier.
    """
    if hasattr(model, 'fc') and isinstance(model.fc, (nn.Linear, nn.Sequential)):
        model.fc = nn.Identity()
        return model

    if hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Sequential):
            layers = list(model.classifier.children())[:-1] + [nn.Identity()]
            model.classifier = nn.Sequential(*layers)
        else:
            model.classifier = nn.Identity()
        return model

    if hasattr(model, 'head') and isinstance(model.head, nn.Linear):
        model.head = nn.Identity()
        return model

    raise ValueError(f"Don't know where the task head is on {model.__class__.__name__}")


class BaseModelWrapper(nn.Module):
    def __init__(
        self,
        model_name:        str,
        device:            str,
        pretrained:        bool = True,
        adapt:             bool = True,
        dyrelu_en:         bool = False,
        dyrelu_phasing_en: bool = False,
        num_classes:       int = 1000,
        ):
        super().__init__()
        components = initialize_model_components(
            model_name, pretrained, dyrelu_en, dyrelu_phasing_en, num_classes,
        )
        self.model        = components['model']
        self.embedded_dim = components['embedded_dim']
        self.model_type   = components['model_type']
        self.model_family = components['model_family']
        self.model_task   = components['model_task']
        # What the weights actually are, not what was requested. Read by
        # results.record_run; see initialize_model_components for why the two
        # can differ without anything raising.
        self.init_source  = components['init_source']

        if adapt and self.model_family == 'resnet':
            adapt_resnet_for_small_images(self.model)
        elif adapt and self.model_family == 'mobilenet':
            adapt_mobilenet_for_small_images(self.model)

        self.to(device)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class ClassificationAndEncoderNetwork(BaseModelWrapper):
    def __init__(
        self,
        model_name:         str,
        num_classes:        int,
        num_out_features:   int = None,
        device:             str = 'cuda',
        adapt:              bool = True,
        pretrained:         bool = True,
        freeze:             bool = False,
        dyrelu_en:          bool = False,
        dyrelu_phasing_en:  bool = False
        ):
        super().__init__(model_name, device, pretrained, adapt, dyrelu_en,
                         dyrelu_phasing_en, num_classes)
        self.model_name = model_name
        self.num_classes = num_classes
        self.num_out_features = num_out_features
        self.device = device

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

        if self.model_type == 'cv':
            # Strip the backbone's head and attach our own, so features are
            # available separately from logits.
            self.model = remove_last_layer(self.model)
            self.cls_head = make_classification_head(self.embedded_dim, self.num_classes).to(self.device)
        else:
            # Language models keep their own head: a sequence classifier or, for
            # MLM, the vocabulary projection. Replacing it would discard the
            # pretrained LM head, which is the entire task. Features come from the
            # encoder's last hidden state instead.
            self.cls_head = None

        if self.num_out_features is not None:
            self.encoder_head = nn.Linear(self.embedded_dim, self.num_out_features).to(self.device)

    def get_embeddings(self, x):
        """Project backbone features into the space the contrastive losses live in.

        proj_mode controls whether a projection head is involved at all:

          'none'  -- contrast the pooled backbone features directly. This is what
                     CAP does (it uses the [CLS] hidden state), and it removes the
                     failure mode below entirely.
          else    -- project through encoder_head, as before.

        Why the mode exists: encoder_head is excluded from pruning by layer_check
        and, for a real ResNet-50, is a Linear(2048, 128) carrying ~262k dense
        parameters. All three contrastive losses are computed *only* on its output.
        With the teacher branch frozen, the student can therefore drive PrC/FiC/SnC
        down by moving those 262k unpruned parameters without ever changing the
        backbone -- the thing being pruned, and the thing the method claims to
        protect. Ordinary contrastive learning is immune because both branches
        share the encoder, so the only route to a lower loss is a better backbone.

        The 'tied_frozen' mode (wired up in training_utils._apply_proj_mode) keeps
        the head but shares one frozen weight matrix across the student and every
        teacher, so the loss again measures backbone agreement.
        """
        if getattr(self, 'proj_mode', 'current') == 'none':
            return F.normalize(x, dim=1)
        raw_emb = self.encoder_head(x)
        return F.normalize(raw_emb, dim=1)

    def _llm_forward(self, batch):
        """One pass returning (features, HF output).

        Features are the last hidden state at position 0 -- the [CLS] / <s> token --
        matching CAP, which uses the [CLS] representation and reports it beat mean
        pooling. Returning both from a single pass matters: computing them
        separately would double the cost of every language-model step.
        """
        outputs = self.model(**batch, output_hidden_states=True)
        features = outputs.hidden_states[-1][:, 0]
        return features, outputs

    def forward_all(self, x):
        """Returns (features, task_output) from a single forward pass."""
        if self.model_type == 'cv':
            features = self.model(x)
            features = features.logits if hasattr(features, 'logits') else features
            return features, self.cls_head(features)
        return self._llm_forward(x)

    def forward(self, x, return_emb=False, return_feat=False):
        features, output = self.forward_all(x)
        if return_feat:
            return features
        if return_emb:
            return self.get_embeddings(features)
        return output
