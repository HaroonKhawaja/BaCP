import torch.nn as nn
import torch
from typing import Any, cast, Optional, Union
from dyrelu_adapter import DyReLUB, DyReLUAdapter
from torchvision.models.vgg import _ovewrite_named_param, WeightsEnum


class VGG(nn.Module):
    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 1000,
        init_weights: bool = True,
        dropout: float = 0.5,
        dyrelu_en: bool =False,
        dyrelu_phasing_en: bool =False,
    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.dyrelu_en = dyrelu_en
        self.dyrelu_phasing_en = dyrelu_phasing_en
        
        if self.dyrelu_en:
            self.relu_v1 = DyReLUB(4096)
            self.relu_v2 = DyReLUB(4096)
        elif self.dyrelu_phasing_en:
            self.relu_v1 = DyReLUAdapter(4096)
            self.relu_v2 = DyReLUAdapter(4096)
        else:
            self.relu_v1 = nn.ReLU(True)
            self.relu_v2 = nn.ReLU(True)
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            self.relu_v1,
            # nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            self.relu_v2,
            # nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        # if init_weights:
        #     for m in self.modules():
        #         if isinstance(m, nn.Conv2d):
        #             # do i edit this too?                    
        #             nn.init.kaiming_normal_(
        #                 m.weight, mode="fan_out", nonlinearity="relu"
        #             )
        #             if m.bias is not None:
        #                 nn.init.constant_(m.bias, 0)
        #         elif isinstance(m, nn.BatchNorm2d):
        #             nn.init.constant_(m.weight, 1)
        #             nn.init.constant_(m.bias, 0)
        #         elif isinstance(m, nn.Linear):
        #             nn.init.normal_(m.weight, 0, 0.01)
        #             nn.init.constant_(m.bias, 0)
        
        if init_weights:
        # Change self.modules() to self.named_modules() to easily skip DyReLU weights
            for name, m in self.named_modules():
                # Skip initializing the Linear networks inside your DyReLU hyperfunctions
                if 'hyperfunction' in name:
                    continue

                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def make_layers(cfg: list[Union[str, int]], batch_norm: bool = False, dyrelu_en: bool = False, dyrelu_phasing_en: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)

            if dyrelu_en:
                relu_v1 = DyReLUB(v)
            elif dyrelu_phasing_en:
                relu_v1 = DyReLUAdapter(v)
            else:
                relu_v1 = nn.ReLU(inplace=True)

            if batch_norm:
                # layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                layers += [conv2d, nn.BatchNorm2d(v), relu_v1]
            else:
                # layers += [conv2d, nn.ReLU(inplace=True)]
                layers += [conv2d, relu_v1]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: dict[str, list[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


def _vgg(
    cfg: str,
    batch_norm: bool,
    weights: Optional[WeightsEnum],
    progress: bool,
    dyrelu_en: bool = False,
    dyrelu_phasing_en: bool = False,
    **kwargs: Any
) -> VGG:
    if weights is not None:
        kwargs["init_weights"] = False
        if weights.meta["categories"] is not None:
            _ovewrite_named_param(
                kwargs, "num_classes", len(weights.meta["categories"])
            )
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm, dyrelu_en=dyrelu_en, dyrelu_phasing_en=dyrelu_phasing_en), dyrelu_en=dyrelu_en, dyrelu_phasing_en=dyrelu_phasing_en, **kwargs)
    if weights is not None:
        model.load_state_dict(
            weights.get_state_dict(progress=progress, check_hash=True),
            strict=False  # gemini told to do this so pre-trained vgg weights load successfully while letting pytorch initialize new dyrelub params from scratch (idk what this means)
        )
    return model


# _COMMON_META = {
#     "min_size": (32, 32),
#     "categories": _IMAGENET_CATEGORIES,
#     "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#alexnet-and-vgg",
#     "_docs": """These weights were trained from scratch by using a simplified training recipe.""",
# }


def vgg11(
    *, weights = None, progress: bool = True, dyrelu_en=False, dyrelu_phasing_en=False, **kwargs: Any
) -> VGG:
    """VGG-11
    """
    # weights = VGG11_Weights.verify(weights)

    return _vgg("A", False, weights, progress, dyrelu_en, dyrelu_phasing_en, **kwargs)


def vgg11_bn(
    *, weights = None, progress: bool = True, dyrelu_en=False, dyrelu_phasing_en=False, **kwargs: Any
) -> VGG:
    """VGG-11-BN
    """
    # weights = VGG11_BN_Weights.verify(weights)

    return _vgg("A", True, weights, progress, dyrelu_en, dyrelu_phasing_en, **kwargs)


def vgg13(
    *, weights = None, progress: bool = True, dyrelu_en = False, dyrelu_phasing_en = False, **kwargs: Any
) -> VGG:
    """VGG-13
    """
    # weights = VGG13_Weights.verify(weights)

    return _vgg("B", False, weights, progress, dyrelu_en, dyrelu_phasing_en, **kwargs)


def vgg13_bn(
    *, weights = None, progress: bool = True, dyrelu_en = False, dyrelu_phasing_en = False, **kwargs: Any
) -> VGG:
    """VGG-13-BN
    """
    # weights = VGG13_BN_Weights.verify(weights)

    return _vgg("B", True, weights, progress, dyrelu_en, dyrelu_phasing_en, **kwargs)


def vgg16(
    *, weights = None, progress: bool = True, dyrelu_en = False, dyrelu_phasing_en = False, **kwargs: Any
) -> VGG:
    """VGG-16
    """
    # weights = VGG16_Weights.verify(weights)

    return _vgg("D", False, weights, progress, dyrelu_en, dyrelu_phasing_en, **kwargs)


def vgg16_bn(
    *, weights = None, progress: bool = True, dyrelu_en = False, dyrelu_phasing_en = False, **kwargs: Any
) -> VGG:
    """VGG-16-BN
    """
    # weights = VGG16_BN_Weights.verify(weights)

    return _vgg("D", True, weights, progress, dyrelu_en, dyrelu_phasing_en, **kwargs)


def vgg19(
    *, weights = None, progress: bool = True, dyrelu_en = False, dyrelu_phasing_en = False, **kwargs: Any
) -> VGG:
    """VGG-19
    """
    # weights = VGG19_Weights.verify(weights)

    return _vgg("E", False, weights, progress, dyrelu_en, dyrelu_phasing_en, **kwargs)


def vgg19_bn(
    *, weights = None, progress: bool = True, dyrelu_en = False, dyrelu_phasing_en = False, **kwargs: Any
) -> VGG:
    """VGG-19_BN
    """
    # weights = VGG19_BN_Weights.verify(weights)

    return _vgg("E", True, weights, progress, dyrelu_en, dyrelu_phasing_en, **kwargs)
