import torch
import torch.nn as nn
import torch.nn.functional as F
from dyrelu_adapter import DyReLUB, DyReLUAdapter

class BasicBlock(nn.Module):
    """ResNet Basic block with DyReLU (for ResNet-18, 34)"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, 
                 groups=1, base_width=64, dilation=1, norm_layer=None,
                 dyrelu_en=False, dyrelu_phasing_en=False,
                 ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        self.dyrelu_en = dyrelu_en
        self.dyrelu_phasing_en = dyrelu_phasing_en

        # 3x3 conv
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        
        # 3x3 conv
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        
        self.downsample = downsample
        self.stride = stride

        if self.dyrelu_en:
            self.relu_v1 = DyReLUB(planes)
            self.relu_v2 = DyReLUB(planes)
        elif self.dyrelu_phasing_en:
            self.relu_v1 = DyReLUAdapter(planes)
            self.relu_v2 = DyReLUAdapter(planes)
        else:
            self.relu_v1 = nn.ReLU()
            self.relu_v2 = nn.ReLU()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu_v1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu_v2(out)

        return out
    
class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: None = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: None = None,
        dyrelu_en: bool = False,
        dyrelu_phasing_en: bool = False,
        ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width)

        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)
        self.bn2 = norm_layer(width)

        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)

        if dyrelu_en:
            self.relu_v1 = DyReLUB(width)
            self.relu_v2 = DyReLUB(width)
            self.relu_v3 = DyReLUB(planes * self.expansion)
        elif dyrelu_phasing_en:
            self.relu_v1 = DyReLUAdapter(width)
            self.relu_v2 = DyReLUAdapter(width)
            self.relu_v3 = DyReLUAdapter(planes * self.expansion)
        else:
            self.relu_v1 = nn.ReLU()
            self.relu_v2 = nn.ReLU()
            self.relu_v3 = nn.ReLU()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu_v1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu_v2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu_v3(out)

        return out

class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes: int = 1000,
        dyrelu_en: bool = False,
        dyrelu_phasing_en: bool = False,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation = None,
        norm_layer = None,
        ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.dyrelu_en = dyrelu_en
        self.dyrelu_phasing_en = dyrelu_phasing_en

        if dyrelu_en:
            self.relu_v1 = DyReLUB(self.inplanes)
        elif dyrelu_phasing_en:
            self.relu_v1 = DyReLUAdapter(self.inplanes)
        else:
            self.relu_v1 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(
        self,
        block,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        ):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer, self.dyrelu_en, self.dyrelu_phasing_en
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    dyrelu_en=self.dyrelu_en,
                    dyrelu_phasing_en=self.dyrelu_phasing_en,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu_v1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def resnet18(num_classes=1000, dyrelu_en=False, dyrelu_phasing_en=False, **kwargs):
    """ResNet-18"""
    return ResNet(
        BasicBlock, [2, 2, 2, 2], num_classes=num_classes,
        dyrelu_en=dyrelu_en, dyrelu_phasing_en=dyrelu_phasing_en, 
        **kwargs)

def resnet34(num_classes=1000, dyrelu_en=False, dyrelu_phasing_en=False, **kwargs):
    """ResNet-34"""
    return ResNet(
        BasicBlock, [3, 4, 6, 3], num_classes=num_classes,
        dyrelu_en=dyrelu_en, dyrelu_phasing_en=dyrelu_phasing_en, 
        **kwargs
        )
    
def resnet34_wide(num_classes=1000, dyrelu_en=False, dyrelu_phasing_en=False, **kwargs):
    """Wide ResNet-34"""
    kwargs["width_per_group"] = 64 * 2
    return ResNet(
        BasicBlock, [3, 4, 6, 3], num_classes=num_classes,
        dyrelu_en=dyrelu_en, dyrelu_phasing_en=dyrelu_phasing_en, 
        **kwargs
        )


def resnet50(num_classes=1000, dyrelu_en=False, dyrelu_phasing_en=False, **kwargs):
    """ResNet-50"""
    return ResNet(
        Bottleneck, [3, 4, 6, 3], num_classes=num_classes,
        dyrelu_en=dyrelu_en, dyrelu_phasing_en=dyrelu_phasing_en, 
        **kwargs
        )


def resnet101(num_classes=1000, dyrelu_en=False, dyrelu_phasing_en=False, **kwargs):
    """ResNet-101"""
    return ResNet(
        Bottleneck, [3, 4, 23, 3], num_classes=num_classes, 
        dyrelu_en=dyrelu_en, dyrelu_phasing_en=dyrelu_phasing_en, 
        **kwargs
        )


# ---------------------------------------------------------------------------
# Wide ResNet-22 (WRRN-22) — official architecture (pre-activation blocks)
# ---------------------------------------------------------------------------

def _wrrn22_conv(in_channels, out_channels, kernel_size=3, stride=1):
    """Conv helper with same-padding for WRN-22."""
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2)


def _wrrn22_bn_relu_conv(in_channels, out_channels,
                         dyrelu_en=False, dyrelu_phasing_en=False):
    """BN → ReLU → Conv sub-block used inside WRNBlock."""
    if dyrelu_en:
        relu = DyReLUB(in_channels)
    elif dyrelu_phasing_en:
        relu = DyReLUAdapter(in_channels)
    else:
        relu = nn.ReLU(inplace=True)

    return nn.Sequential(
        nn.BatchNorm2d(in_channels),
        relu,
        _wrrn22_conv(in_channels, out_channels),
    )


class WRNBlock(nn.Module):
    """Pre-activation residual block for WideResNet-22."""

    def __init__(self, input_channels, output_channels, stride=1,
                 dyrelu_en=False, dyrelu_phasing_en=False):
        super().__init__()

        if dyrelu_en:
            relu = DyReLUB(input_channels)
        elif dyrelu_phasing_en:
            relu = DyReLUAdapter(input_channels)
        else:
            relu = nn.ReLU(inplace=True)

        self.bn_relu = nn.Sequential(nn.BatchNorm2d(input_channels), relu)
        self.conv1 = _wrrn22_conv(input_channels, output_channels, stride=stride)
        self.conv2 = _wrrn22_bn_relu_conv(output_channels, output_channels,
                                          dyrelu_en, dyrelu_phasing_en)

        self.shortcut = nn.Identity()
        if input_channels != output_channels or stride != 1:
            self.shortcut = _wrrn22_conv(input_channels, output_channels,
                                         stride=stride, kernel_size=1)

    def forward(self, x):
        out = self.bn_relu(x)
        residual = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(out)
        return out.add_(residual)


def _wrrn22_make_group(in_ch, out_ch, n_blocks, stride,
                       dyrelu_en=False, dyrelu_phasing_en=False):
    """Construct one group of WRN residual blocks."""
    layers = [WRNBlock(in_ch, out_ch, stride, dyrelu_en, dyrelu_phasing_en)]
    for _ in range(1, n_blocks):
        layers.append(WRNBlock(out_ch, out_ch, 1, dyrelu_en, dyrelu_phasing_en))
    return layers


class WideResNet22(nn.Module):
    """Wide ResNet-22 — 3 groups × 3 blocks, channels [16, 96, 192, 384]."""

    def __init__(self, n_groups=3, num_classes=1000, channel_start=16,
                 dyrelu_en=False, dyrelu_phasing_en=False):
        super().__init__()
        n_channels = [channel_start, 96, 192, 384]

        layers = [_wrrn22_conv(3, channel_start)]

        for g in range(n_groups):
            stride = 2 if g > 0 else 1
            layers += _wrrn22_make_group(
                n_channels[g], n_channels[g + 1], n_groups, stride,
                dyrelu_en, dyrelu_phasing_en,
            )

        if dyrelu_en:
            final_relu = DyReLUB(n_channels[-1])
        elif dyrelu_phasing_en:
            final_relu = DyReLUAdapter(n_channels[-1])
        else:
            final_relu = nn.ReLU(inplace=True)

        layers += [
            nn.BatchNorm2d(n_channels[-1]),
            final_relu,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(n_channels[-1], num_classes),
        ]

        self.layers = nn.Sequential(*layers)

        # Weight initialisation (same scheme as ResNet above)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.layers(x)


def wrrn22(num_classes=1000, dyrelu_en=False, dyrelu_phasing_en=False, **kwargs):
    """Wide ResNet-22 (WRRN-22)"""
    return WideResNet22(
        n_groups=3, num_classes=num_classes,
        dyrelu_en=dyrelu_en, dyrelu_phasing_en=dyrelu_phasing_en,
    )
