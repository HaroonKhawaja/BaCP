import torch
import torch.nn as nn
import torch.nn.functional as F
from dyrelu_adapter import DyReLUB

class BasicBlock(nn.Module):
    """ResNet Basic block with DyReLU (for ResNet-18, 34)"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, 
                 groups=1, base_width=64, dilation=1, norm_layer=None,
                 dyrelu_enabled=False
                 ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        self.dyrelu_enabled = dyrelu_enabled

        # 3x3 conv
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        
        # 3x3 conv
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        
        self.downsample = downsample
        self.stride = stride

        if self.dyrelu_enabled:
            self.relu_v1 = DyReLUB(planes)
            self.relu_v2 = DyReLUB(planes)
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
        dyrelu_enabled: bool = False,
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

        if dyrelu_enabled:
            self.relu_v1 = DyReLUB(width)
            self.relu_v2 = DyReLUB(width)
            self.relu_v3 = DyReLUB(planes * self.expansion)
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
        dyrelu_enabled: bool = False,
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
        self.dyrelu_enabled = dyrelu_enabled

        if dyrelu_enabled:
            self.relu_v1 = DyReLUB(self.inplanes)
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
                if isinstance(m, DyReLUBottleneck) and m.bn3.weight is not None:
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
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer, self.dyrelu_enabled
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
                    dyrelu_enabled=self.dyrelu_enabled,
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


def resnet18(num_classes=1000, dyrelu_enabled=False, **kwargs):
    """ResNet-18"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, dyrelu_enabled=dyrelu_enabled, **kwargs)

def resnet34(num_classes=1000, dyrelu_enabled=False, **kwargs):
    """ResNet-34"""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, dyrelu_enabled=dyrelu_enabled, **kwargs)


def resnet50(num_classes=1000, dyrelu_enabled=False, **kwargs):
    """ResNet-50"""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, dyrelu_enabled=dyrelu_enabled, **kwargs)


def resnet101(num_classes=1000, dyrelu_enabled=False, **kwargs):
    """ResNet-101"""
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, dyrelu_enabled=dyrelu_enabled, **kwargs)
