from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet101, vgg11, vgg19
from torchvision.models import ResNet50_Weights, ResNet101_Weights, VGG11_Weights, VGG19_Weights
from transformers import AutoModelForImageClassification, AutoModelForSequenceClassification, AutoModelForMaskedLM
from dataclasses import dataclass

@dataclass(frozen=True)
class ModelSpec:
    fn: object
    weight: object
    family: str
    type: str

PRETRAINED = True

MODELS = {
    'resnet50':     ModelSpec(resnet50, ResNet50_Weights.IMAGENET1K_V1,     'resnet',   'vision'),
    'resnet101':    ModelSpec(resnet101, ResNet101_Weights.IMAGENET1K_V1,   'resnet',   'vision'),
    'vgg11':    ModelSpec(vgg11, VGG11_Weights.IMAGENET1K_V1,   'vgg',    'vision'),
    'vgg19':    ModelSpec(vgg19, VGG19_Weights.IMAGENET1K_V1,   'vgg',    'vision'),
    'distilbert-base-uncased-mlm':  ModelSpec(AutoModelForMaskedLM,                 'distilbert-base-uncased',  'bert', 'language'),
    'distilbert-base-uncased':      ModelSpec(AutoModelForSequenceClassification,   'distilbert-base-uncased',  'bert', 'language'),
    'roberta-base-mlm': ModelSpec(AutoModelForMaskedLM,                 'roberta-base', 'bert', 'language'),
    'roberta-base':     ModelSpec(AutoModelForSequenceClassification,   'roberta-base', 'bert', 'language'),
}

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
    if hasattr(model, 'config') and hasattr(model.config, 'hidden_size'):
        return model.config.hidden_size
    raise RuntimeError(f"Couldn't infer embedding dim for model: {model.__class__.__name__}")
    
def initialize_model_components(model_name: str, pretrained: bool):
    if model_name not in MODELS:
        raise ValueError(f"Unknown model {model_name}. Choices: {list(MODELS.keys())}")

    spec = MODELS[model_name]
    if spec.type == 'vision':
        model = spec.fn(weights=spec.weight if pretrained else None)
    else:
        raise ValueError(f"Unsupported model type: {spec.type}")

    embedded_dim = _get_embedded_dim_from_model(model)
    return {
        'model':        model,
        'embedded_dim': embedded_dim,
        'model_type':   spec.type,
        'model_family': spec.family,
    }

def adapt_resnet_for_small_images(model):
    if hasattr(model, 'conv1'):
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

def make_classification_head(embedded_dim: int, num_out_features: int):
    return nn.Linear(embedded_dim, num_out_features)

def adapt_head_for_model(model, head: nn.Module, model_type: str, model_family: str):
    if model_type == 'vision':
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
    children = list(model.named_children())
    last_name, last_module = children[-1]

    if isinstance(last_module, nn.Sequential):
        layers = list(last_module.children())[:-1]
        layers.append(nn.Identity())
        setattr(model, last_name, nn.Sequential(*layers))
    elif isinstance(last_module, nn.Linear):
        setattr(model, last_name, nn.Identity())
    else:
        raise ValueError(f"Unsupported module type: {type(last_module)}")
    return model

class BaseModelWrapper(nn.Module):
    def __init__(self, model_name: str, device: str, pretrained: bool = True, adapt: bool = True):
        super().__init__()
        components = initialize_model_components(model_name, pretrained)
        self.model        = components['model']
        self.embedded_dim = components['embedded_dim']
        self.model_type   = components['model_type']
        self.model_family = components['model_family']

        if adapt and self.model_family == 'resnet':
            adapt_resnet_for_small_images(self.model)

        self.to(device)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

class ClassificationAndEncoderNetwork(BaseModelWrapper):
    def __init__(self, model_name, num_classes, num_out_features=None, device='cuda', adapt=True, pretrained=True, freeze=False):
        super().__init__(model_name, device, pretrained, adapt)
        self.model_name = model_name
        self.num_classes = num_classes
        self.num_out_features = num_out_features
        self.device = device

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
        
        self.model = remove_last_layer(self.model)
        self.cls_head = make_classification_head(self.embedded_dim, self.num_classes).to(self.device)

        if self.num_out_features is not None:
            self.encoder_head = nn.Linear(self.embedded_dim, self.num_out_features).to(self.device)
    
    def get_embeddings(self, x):
        raw_emb = self.encoder_head(x)
        return F.normalize(raw_emb, dim=1)

    def forward(self, x, return_emb=False, return_feat=False):
        if self.model_type == 'vision':
            x = self.model(x)
        else:
            x = self.model(**x)

        if return_feat:
            return x
        if return_emb:
            return self.get_embeddings(x)
        return self.cls_head(x)



















