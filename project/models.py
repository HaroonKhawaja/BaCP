import torch
import torch.nn as nn 
import torch.nn.functional as F
from torchvision.models import resnet50, resnet101, vgg11, vgg19, vit_b_16, vit_l_16
from torchvision.models import ResNet50_Weights, ResNet101_Weights, VGG11_Weights, VGG19_Weights, ViT_B_16_Weights, ViT_L_16_Weights
from transformers import AutoModelForImageClassification, AutoModelForSequenceClassification, AutoModelForQuestionAnswering, AutoModelForMaskedLM, AutoTokenizer
from transformers import ViTConfig, ViTModel
from utils import freeze_weights

PRETRAINED = True
# MODEL_SPECS = {
#     "resnet50":  {"fn": resnet50,  "dim": 2048, "weight": ResNet50_Weights.IMAGENET1K_V1, "type": "vision", "family": "resnet"},
#     "resnet101": {"fn": resnet101, "dim": 2048, "weight": ResNet101_Weights.IMAGENET1K_V1, "type": "vision", "family": "resnet"},
#     "vgg11":     {"fn": vgg11,     "dim": 4096, "weight": VGG11_Weights.IMAGENET1K_V1,     "type": "vision", "family": "vgg"},
#     "vgg19":     {"fn": vgg19,     "dim": 4096, "weight": VGG19_Weights.IMAGENET1K_V1,     "type": "vision", "family": "vgg"},
#     "vitb16":    {"fn": vit_b_16,  "dim": 768,  "weight": ViT_B_16_Weights.IMAGENET1K_V1,  "type": "vision", "family": "vit"},    
#     "vitl16":    {"fn": vit_l_16,  "dim": 1024, "weight": ViT_L_16_Weights.IMAGENET1K_V1,  "type": "vision", "family": "vit"},
#     "distilbert-base-uncased": {"dim": 768, "type": "language", "family": "bert"},
#     "roberta-base": {"dim": 768, "type": "language", "family": "bert"},
#     'vits16':   {'fn': timm.create_model, ''}
# }

def get_model_components(model_name, pretrained=True, num_llm_labels=2, model_task='cls'):
    if model_name not in MODEL_SPECS:
        raise ValueError(f"Unknown model: '{model_name}'. Available: {list(MODEL_SPECS.keys())}")

    spec = MODEL_SPECS[model_name]
    if spec["type"] == "vision":
        if model_name.startswith('vit'):
            model = spec["fn"](spec["weight"], pretrained=True)
        else:
            model = spec["fn"](weights=spec["weight"] if pretrained else None)
    else:
        if model_task == 'squad':
            model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        elif model_task == 'wikitext2':
            model = AutoModelForMaskedLM.from_pretrained(model_name)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_llm_labels)
    return {
        "model": model,
        "embedding_dim": spec["dim"],
        "model_type": spec["type"],
        "model_family": spec["family"]
    }


MODELS = {
    'resnet50': {
        'fn': resnet50,
        'weight': ResNet50_Weights.IMAGENET1K_V1,
        'family': 'resnet',
        'type': 'vision',
    },
    'resnet101': {
        'fn': resnet101,
        'weight': ResNet101_Weights.IMAGENET1K_V1,
        'family': 'resnet',
        'type': 'vision',
    },
    'vgg11': {
        'fn': vgg11,
        'weight': VGG11_Weights.IMAGENET1K_V1,
        'family': 'vgg',
        'type': 'vision',
    },
    'vgg19': {
        'fn': vgg19,
        'weight': VGG19_Weights.IMAGENET1K_V1,
        'family': 'vgg',
        'type': 'vision',
    },
    'vit_tiny': {
        'fn': AutoModelForImageClassification,
        'weight': 'WinKawaks/vit-tiny-patch16-224',
        'family': 'vit',
        'type': 'vision',
    },
    'vit_small': {
        'fn': AutoModelForImageClassification,
        'weight': 'WinKawaks/vit-small-patch16-224',
        'family': 'vit',
        'type': 'vision',
    },
    'distilbert-base-uncased-mlm': {
        'fn': AutoModelForMaskedLM,
        'weight': 'distilbert-base-uncased',
        'family': 'bert',
        'type': 'language',
    },
    'roberta-base-mlm': {
        'fn': AutoModelForMaskedLM,
        'weight': 'roberta-base',
        'family': 'bert',
        'type': 'language',
    }
        
}

def initialize_model_components(model_name, pretrained=True, model_task=''):
    if model_task == 'wikitext2':
        model_name += '-mlm'
    
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: '{model_name}'. Available: {list(MODELS.keys())}")
    

    spec = MODELS[model_name]

    model_fn = spec['fn']
    model_family = spec['family']
    model_type = spec['type']
    embedding_dim = None

    if model_type == 'vision':
        # Initializing model
        if model_name.startswith('vit'):
            model = model_fn.from_pretrained(spec['weight'] if pretrained else None)
        else:
            model = model_fn(weights=spec["weight"] if pretrained else None)
        
        # Initializing model embedded dimensions
        if hasattr(model, 'fc'):
            embedding_dim = model.fc.in_features
        elif hasattr(model, 'classifier'):
            if isinstance(model.classifier, nn.Sequential):
                embedding_dim = model.classifier[-1].in_features
            else:
                embedding_dim = model.classifier.in_features
        elif hasattr(model, 'head'):
            embedding_dim = model.head.in_features

    elif model_type == 'language':
        model = model_fn.from_pretrained(spec['weight'] if pretrained else None)
        embedding_dim = model.config.hidden_size
    else:
        raise ValueError(f"Model type '{model_type}' not supported.")
    
    return {
        'model': model,
        'embedding_dim': embedding_dim,
        'model_type': model_type,
        'model_family': model_family
    }

def make_projection_head(args, output_dims=128):
    if args.model_type == 'language':
        raise NotImplementedError("Language models not supported.")

    if args.model_family == 'resnet':
        args.projection_head = nn.Sequential(
            nn.Linear(args.embedding_dim, args.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.embedding_dim, output_dims)
        )
    elif args.model_family in ['vgg', 'vit']:
        args.projection_head = nn.Sequential(
            nn.Linear(args.embedding_dim, output_dims),
        )
    else:
        raise ValueError(f"Model family '{args.model_family}' not supported.")

def adapt_head_for_model(args, head):
    if args.model_type == 'vision':
        # ResNet Support
        if hasattr(args.model, 'fc'):
            args.model.fc = head
        elif hasattr(args.model, 'classifier'):
            # VGG Support
            if isinstance(args.model.classifier, nn.Sequential):
                args.model.classifier[-1] = head
            else:
                # ViT Support (HF)
                args.model.classifier = head   
        # ViT Support 
        elif hasattr(args.model, 'head'):
            args.model.head = head
        else:
            raise ValueError(f"Model '{args.model}' does not have a head.")
    elif args.model_type == 'language':
        # DistilBERT Support
        if hasattr(args.model, 'vocab_projector'):
            args.model.vocab_projector = head
        else:
            raise ValueError(f"Model '{args.model}' does not have a head.")
    else:
        raise ValueError(f"Model type '{args.model_type}' not supported.")

def adapt_resnet_for_small_images(model):
    if hasattr(model, 'conv1'):
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

class EncoderProjectionNetwork(nn.Module):
    def __init__(self, model_name, output_dims=128, adapt=True, model_task='', pretrained=True):
        super().__init__()
        
        # Loading model components
        # components = get_model_components(model_name, pretrained, num_llm_labels=output_dims, model_task=model_task)        
        components = initialize_model_components(model_name, pretrained, model_task)

        self.model = components["model"]
        self.embedding_dim = components["embedding_dim"]
        self.model_type = components["model_type"]
        self.model_family = components["model_family"]
        self.model_task = model_task

        # Adapting the model for cifar10
        if adapt and self.model_family == "resnet":
            adapt_resnet_for_small_images(self.model)

        make_projection_head(self, output_dims)
        adapt_head_for_model(self, nn.Identity())
        
    def forward(self, x, extract_raw=False):
        if self.model_type == "vision":
            raw_features = self.model(x)
            if hasattr(raw_features, 'logits'):
                raw_features = raw_features.logits

            if extract_raw:
                return raw_features
            embeddings = self.projection_head(raw_features)
            return F.normalize(embeddings, dim=1)
        else:
            x = self.model(
                **x,
                output_hidden_states=True, 
                return_dict=True
                )
            x.logits = F.normalize(x.logits, dim=1)
            return x

class ClassificationNetwork(nn.Module):
    def __init__(self, model_name, num_classes=10, adapt=True, model_task='', pretrained=True, freeze=False):
        super().__init__()
        
        # Loading model components
        # components = get_model_components(model_name, num_llm_labels=num_classes, model_task=model_task)        
        components = initialize_model_components(model_name, pretrained, model_task)

        self.model = components["model"]
        self.embedding_dim = components["embedding_dim"]
        self.model_type = components["model_type"]
        self.model_family = components["model_family"]
        self.model_task = model_task
        
        # Freezing the model if applicable
        if freeze:
            freeze_weights(self.model)

        # Adapting the model for cifar10
        if adapt and self.model_family == 'resnet':
            adapt_resnet_for_small_images(self.model)
        
        # Attaching the classification head
        classification_head = nn.Linear(self.embedding_dim, num_classes)
        adapt_head_for_model(self, classification_head)
    
    def forward(self, x):
        if self.model_type == "vision":
            return self.model(x)
        else:
            return self.model(**x)

