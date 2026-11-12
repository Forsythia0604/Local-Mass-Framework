r"""ResNet-56 for CIFAR (Tier B base network).

Standard CIFAR ResNet (He et al.): conv1(3->16) + 3 stages of n=9 BasicBlocks at
widths 16/32/64 + BN/ReLU + global avgpool + linear.  Depth = 6n+2 = 56.
Deterministic weights; the group gates are attached externally (gated_model.py).
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))   # conv1 = prunable inner conv (gated)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNetCIFAR(nn.Module):
    def __init__(self, n_per_stage=9, num_classes=10):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_stage(16, n_per_stage, stride=1)
        self.layer2 = self._make_stage(32, n_per_stage, stride=2)
        self.layer3 = self._make_stage(64, n_per_stage, stride=2)
        self.fc = nn.Linear(64, num_classes)

    def _make_stage(self, planes, n, stride):
        strides = [stride] + [1] * (n - 1)
        blocks = []
        for s in strides:
            blocks.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer3(self.layer2(self.layer1(out)))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


def resnet56(num_classes=10):
    return ResNetCIFAR(n_per_stage=9, num_classes=num_classes)
