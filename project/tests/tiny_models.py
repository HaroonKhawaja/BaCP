"""A tiny ResNet built from the *real* models.resnet.BasicBlock.

Why not a hand-rolled 2-layer toy? Three properties of this shape are load
bearing, and a simpler stub makes several test levels vacuously green:

1. It exposes ``.fc`` with ``.in_features``, so model_factory's real
   ``_get_embedded_dim_from_model`` and ``remove_last_layer`` paths run
   unstubbed instead of being monkeypatched around.

2. ``layer1``..``layer4`` are ``nn.Sequential`` containers holding **3** blocks
   each, whose ``conv1``/``conv2`` children have matching shapes from index 1
   onward. ``apply_weight_sharing_resnet`` skips any layer with
   ``num_blocks <= R`` (default R=2) and shares ``block[i] <- block[R-1]`` for
   ``i > R-1``, so with 3 blocks it actually shares something. With 1 block per
   layer every weight-sharing assertion would pass while testing nothing.

3. It threads ``dyrelu_en`` / ``dyrelu_phasing_en`` into the real ``BasicBlock``,
   so ``relu_v1``/``relu_v2`` become real ``DyReLUB`` / ``DyReLUAdapter``
   modules and the phasing tests exercise real wiring.

Construct the wrapper with ``adapt=False``. ``adapt_resnet_for_small_images``
hardcodes ``nn.Conv2d(3, 64, ...)``, which would leave conv1 emitting 64
channels into a ``bn1`` expecting 8. ``test_adapt_resnet_hardcodes_64_channels``
records that constraint rather than silently working around it.

Sized for 8x8 inputs: ~12k params, forward pass ~4ms on CPU.
"""

import torch
import torch.nn as nn

from models.resnet import BasicBlock

# 8x8 input -> conv1 (s1) 8x8 -> layer1 (s1) 8x8 -> layer2 (s2) 4x4
#           -> layer3 (s1) 4x4 -> layer4 (s2) 2x2 -> avgpool 1x1
WIDTHS = (8, 8, 16, 16)
BLOCKS = (3, 3, 3, 3)
STRIDES = (1, 2, 1, 2)


def _make_layer(inplanes, planes, blocks, stride, dyrelu_en, dyrelu_phasing_en):
    downsample = None
    if stride != 1 or inplanes != planes * BasicBlock.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * BasicBlock.expansion, kernel_size=1,
                      stride=stride, bias=False),
            nn.BatchNorm2d(planes * BasicBlock.expansion),
        )

    layers = [BasicBlock(inplanes, planes, stride, downsample,
                         dyrelu_en=dyrelu_en, dyrelu_phasing_en=dyrelu_phasing_en)]
    for _ in range(1, blocks):
        layers.append(BasicBlock(planes, planes,
                                 dyrelu_en=dyrelu_en,
                                 dyrelu_phasing_en=dyrelu_phasing_en))
    return nn.Sequential(*layers)


class TinyResNet(nn.Module):
    """Mirrors the real ResNet's attribute layout: conv1/bn1/layer1..4/avgpool/fc.

    Attribute assignment order matters -- ``remove_last_layer`` takes
    ``list(model.named_children())[-1]``, so ``fc`` must be registered last.
    """

    def __init__(self, num_classes=1000, widths=WIDTHS, blocks=BLOCKS,
                 strides=STRIDES, dyrelu_en=False, dyrelu_phasing_en=False):
        super().__init__()
        self.conv1 = nn.Conv2d(3, widths[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(widths[0])
        self.relu = nn.ReLU(inplace=True)

        built, inplanes = [], widths[0]
        for width, n_blocks, stride in zip(widths, blocks, strides):
            built.append(_make_layer(inplanes, width, n_blocks, stride,
                                     dyrelu_en, dyrelu_phasing_en))
            inplanes = width
        self.layer1, self.layer2, self.layer3, self.layer4 = built

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(inplanes, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def make_tiny_resnet(num_classes=1000, dyrelu_en=False, dyrelu_phasing_en=False, **kwargs):
    """Factory matching the calling convention of ``ModelSpec.fn``.

    ``initialize_model_components`` calls ``spec.fn(dyrelu_en=..., dyrelu_phasing_en=...)``,
    so this signature is what lets a ``ModelSpec`` point at the tiny model.
    """
    return TinyResNet(num_classes=num_classes,
                      dyrelu_en=dyrelu_en,
                      dyrelu_phasing_en=dyrelu_phasing_en,
                      **kwargs)
