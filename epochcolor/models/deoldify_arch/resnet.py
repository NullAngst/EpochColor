"""ResNet-34 and ResNet-101 bodies with torchvision's module names, so a
DeOldify checkpoint's encoder keys (layers.0.0.weight, layers.0.4.0.conv1...)
load as they are. Only the structure is here; the weights come from the
checkpoint. torchvision isn't part of the PyTorch download EpochColor makes,
and the whole of it would be needed for these forty lines.
"""

import torch.nn as nn


def _conv3(cin, cout, stride=1):
    return nn.Conv2d(cin, cout, 3, stride, 1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, cin, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = _conv3(cin, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        idt = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(out)) + idt)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, cin, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = _conv3(planes, planes, stride)  # stride on the 3x3, as torchvision does
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        idt = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return self.relu(self.bn3(self.conv3(out)) + idt)


def _layer(block, cin, planes, n, stride):
    down = None
    if stride != 1 or cin != planes * block.expansion:
        down = nn.Sequential(nn.Conv2d(cin, planes * block.expansion, 1, stride, bias=False),
                             nn.BatchNorm2d(planes * block.expansion))
    layers = [block(cin, planes, stride, down)]
    layers += [block(planes * block.expansion, planes) for _ in range(n - 1)]
    return nn.Sequential(*layers), planes * block.expansion


def body(depth: int) -> nn.Sequential:
    """conv1, bn1, relu, maxpool, layer1..4: torchvision's resnet children[:8]."""
    block, counts = {34: (BasicBlock, (3, 4, 6, 3)), 101: (Bottleneck, (3, 4, 23, 3))}[depth]
    mods = [nn.Conv2d(3, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)]
    c = 64
    for planes, n, stride in zip((64, 128, 256, 512), counts, (1, 2, 2, 2)):
        layer, c = _layer(block, c, planes, n, stride)
        mods.append(layer)
    return nn.Sequential(*mods)
