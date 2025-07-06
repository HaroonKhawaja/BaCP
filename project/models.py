import torch
import torch.nn as nn 
import torch.nn.functional as F
from torchvision.models import resnet50, resnet101, vgg11, vgg19, vit_b_16, vit_l_16
from torchvision.models import ResNet50_Weights, ResNet101_Weights, VGG11_Weights, VGG19_Weights, ViT_B_16_Weights, ViT_L_16_Weights
from transformers import AutoModelForSequenceClassification, AutoModelForQuestionAnswering, AutoModelForMaskedLM, AutoTokenizer
from transformers import ViTConfig, ViTModel
from utils import freeze_weights

PRETRAINED = True
MODEL_SPECS = {
    "resnet50":  {"fn": resnet50,  "dim": 2048, "weight": ResNet50_Weights.IMAGENET1K_V1, "type": "vision", "family": "resnet"},
    "resnet101": {"fn": resnet101, "dim": 2048, "weight": ResNet101_Weights.IMAGENET1K_V1, "type": "vision", "family": "resnet"},
    "vgg11":     {"fn": vgg11,     "dim": 4096, "weight": VGG11_Weights.IMAGENET1K_V1,     "type": "vision", "family": "vgg"},
    "vgg19":     {"fn": vgg19,     "dim": 4096, "weight": VGG19_Weights.IMAGENET1K_V1,     "type": "vision", "family": "vgg"},
    "vitb16":    {"fn": vit_b_16,  "dim": 768,  "weight": ViT_B_16_Weights.IMAGENET1K_V1,  "type": "vision", "family": "vit"},    
    "vitl16":    {"fn": vit_l_16,  "dim": 1024, "weight": ViT_L_16_Weights.IMAGENET1K_V1,  "type": "vision", "family": "vit"},
    "distilbert-base-uncased": {"dim": 768, "type": "language", "family": "bert"},
    "roberta-base": {"dim": 768, "type": "language", "family": "bert"},
    "vitb16_small":    {"dim": 768,  "weight": 'google/vit-base-patch16-224',  "type": "vision", "family": "vit"},    
}

def get_model_components(model_name, pretrained=True, num_llm_labels=2, model_task='cls'):
    if model_name not in MODEL_SPECS:
        raise ValueError(f"Unknown model: '{model_name}'. Available: {list(MODEL_SPECS.keys())}")

    spec = MODEL_SPECS[model_name]
    if spec["type"] == "vision":
        if model_name == 'vitb16_small':
            config = ViTConfig.from_pretrained(spec['weight'])
            config.image_size = 32
            config.patch_size = 8
            model = ViTModel(config)

            pretrained_state_dict = ViTModel.from_pretrained(spec['weight']).state_dict()
            filtered_state_dict = {
                k: v for k, v in pretrained_state_dict.items()
                if not any(p in k for p in ["embeddings.patch_embeddings.projection", "embeddings.position_embeddings"])
            }          
            model.load_state_dict(filtered_state_dict, strict=False)
        else:
            weights = spec["weight"] if pretrained else None
            model = spec["fn"](weights=weights)
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


def make_projection_head(args, output_dims=128):
    if args.model_type == 'language':
        return
    if args.model_family == 'resnet':
        args.projection_head = nn.Sequential(
            nn.Linear(args.embedding_dim, args.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.embedding_dim, output_dims)
        )
    elif args.model_family == 'vgg':
        args.projection_head = nn.Sequential(
            nn.Linear(args.embedding_dim, output_dims),
        )
    else:
        raise ValueError(f"Model family '{args.model_family}' not supported.")

def adapt_head_for_model(args, head):
    if args.model_type == 'language':
        return
    if args.model_type == 'vision':
        if args.model_family == "vit":
            args.model.heads = head
        elif args.model_family == "vgg":
            args.model.classifier[-1] = head
        elif args.model_family == "resnet":
            args.model.fc = head
    else:
        raise ValueError(f"Model type '{args.model_type}' not supported.")

def adapt_resnet_for_small_images(model):
    if hasattr(model, 'conv1'):
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

class EncoderProjectionNetwork(nn.Module):
    def __init__(self, model_name, output_dims=128, pretrained=True, adapt=True, model_task='cls'):
        super().__init__()
        
        # Loading model components
        components = get_model_components(model_name, pretrained, num_llm_labels=output_dims, model_task=model_task)
        self.model = components["model"]
        self.embedding_dim = components["embedding_dim"]
        self.model_type = components["model_type"]
        self.model_family = components["model_family"]
        self.model_task = model_task

        # Adapting the model for cifar10
        if adapt and self.model_family == "resnet":
            adapt_resnet_for_small_images(self.model)

        make_projection_head(self)
        adapt_head_for_model(self, nn.Identity())

        # if self.model_type == "vision":
        #     projection_head = nn.Sequential(
        #         nn.Linear(self.embedding_dim, self.embedding_dim),
        #         nn.ReLU(inplace=True),
        #         nn.Linear(self.embedding_dim, output_dims)
        #     )
        #     adapt_head_for_model(self.model, projection_head, self.model_family)
        # else:
        #     if self.model_task == 'wikitext2':
        #         if hasattr(self.model, 'vocab_projector'):
        #             self.model.vocab_projector = nn.Linear(self.embedding_dim, output_dims)
        #         elif hasattr(self.model, 'lm_head') and hasattr(self.model.lm_head, 'decoder'):
        #             self.model.lm_head.decoder = nn.Linear(self.embedding_dim, output_dims)
        
    def forward(self, x, extract_raw=False):
        if self.model_type == "vision":
            raw_features = self.model(x)
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
    def __init__(self, model_name, num_classes=10, freeze=False, adapt=True, model_task='cls'):
        super().__init__()
        
        # Loading model components
        components = get_model_components(model_name, num_llm_labels=num_classes, model_task=model_task)
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
        
        # Attaching the classification head if its a vision model
        if self.model_type == "vision":
            classification_head = nn.Linear(self.embedding_dim, num_classes)
            adapt_head_for_model(self, classification_head)
    
    def forward(self, x):
        if self.model_type == "vision":
            return self.model(x)
        else:
            return self.model(**x)

