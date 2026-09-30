"""
ResNet-56 (CIFAR variant) with structured pruning interface, matching
DepGraph's `reproduce/engine/models/cifar/resnet_tiny.py:resnet56`
checkpoint exactly so `cifar10_resnet56.pth` (Torch-Pruning v1.1.4
release) loads strict.

Architecture (depth=56, BasicBlock, num_filters=[16,16,32,64], n=9
blocks per stage):
  - conv1: 3 -> 16, 3x3, BN, ReLU                       (32x32)
  - layer1: 9 BasicBlocks at 16 channels                 (32x32, no downsample)
  - layer2: 9 BasicBlocks, layer2.0 has 1x1 downsample 16 -> 32 (16x16)
  - layer3: 9 BasicBlocks, layer3.0 has 1x1 downsample 32 -> 64 (8x8)
  - avgpool 8x8 -> 1x1
  - fc: 64 -> num_classes

Two prunable structure types per stage (same as resnet18/model.py,
generalised to n_blocks_per_stage):

  Stage-width  (ell 1..3):   output channels of stage 0..2.
                              N=16, 32, 64. Each removes the channel
                              from the stage's residual path:
                                - initial conv (stage 0) OR downsample
                                  (stages 1, 2)
                                - conv2 + bn2 in EVERY block of the
                                  stage (9 blocks)
                                - input channels of next stage's
                                  conv1 + downsample (or fc for stage 2)

  Block-mid    (ell 4..30):  intermediate channels of conv1 within a
                              block (27 blocks total).  Self-contained:
                                - conv1 filter j + bn1[j]
                                - conv2 input channel j slice

Total: num_hidden = 3 + 27 = 30.

Pruning interface (structure_norms / structure_dims / prune_structures
/ zero_pruned_grads) matches the engine.py contract used by VGG and
ResNet-18.

Use `load_dg_checkpoint(path, model)` to populate weights from
DepGraph's `cifar10_resnet56.pth`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Standard CIFAR ResNet-56 hyperparameters
NUM_FILTERS = (16, 16, 32, 64)
N_BLOCKS_PER_STAGE = 9   # 56 = 6n + 2 -> n = 9


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return F.relu(out)


class ResNet56(nn.Module):
    """ResNet-56 (BasicBlock) for CIFAR, matching DepGraph's checkpoint layout."""

    def __init__(self, num_classes=10, num_filters=NUM_FILTERS,
                 n_per_stage=N_BLOCKS_PER_STAGE):
        super().__init__()
        block = BasicBlock
        self.inplanes = num_filters[0]
        self.conv1 = nn.Conv2d(3, num_filters[0], kernel_size=3,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.layer1 = self._make_layer(block, num_filters[1], n_per_stage)
        self.layer2 = self._make_layer(block, num_filters[2], n_per_stage, stride=2)
        self.layer3 = self._make_layer(block, num_filters[3], n_per_stage, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(num_filters[3] * block.expansion, num_classes)

        # Pruning bookkeeping
        self.stage_planes = list(num_filters[1:])         # [16, 32, 64]
        self.stages = [self.layer1, self.layer2, self.layer3]
        self.n_per_stage = n_per_stage                    # 9
        self.n_stages = len(self.stages)                  # 3
        # num_hidden = stage-width + block-mid
        # Block-mid count = n_stages * n_per_stage = 27
        self.num_hidden = self.n_stages + self.n_stages * self.n_per_stage  # 30

        self._init_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.fc(x)

    # ------------------------------------------------------------------
    # Pruning interface
    # ------------------------------------------------------------------

    def _decode_ell(self, ell):
        """
        ell 1..n_stages          -> ('stage', stage_idx)        # 0..n_stages-1
        ell n_stages+1..num_hidden -> ('mid', flat block_idx)    # 0..n_stages*n_per_stage-1
        """
        if ell <= self.n_stages:
            return ('stage', ell - 1)
        return ('mid', ell - self.n_stages - 1)

    def _get_block(self, block_idx):
        stage_idx = block_idx // self.n_per_stage
        local_idx = block_idx % self.n_per_stage
        return self.stages[stage_idx][local_idx]

    def structure_dims(self, ell):
        typ, idx = self._decode_ell(ell)
        if typ == 'stage':
            planes = self.stage_planes[idx]
            stage = self.stages[idx]
            d = 0
            # Initial conv (stage 0) or downsample
            if idx == 0:
                d += self.conv1.weight[0].numel()
                d += 2  # bn1 weight + bias
            else:
                block0 = stage[0]
                if block0.downsample is not None:
                    d += block0.downsample[0].weight[0].numel()
                    d += 2  # downsample bn weight + bias
            # conv2 + bn2 in each block of the stage
            for block in stage:
                d += block.conv2.weight[0].numel()
                d += 2  # bn2 weight + bias
            return torch.full((planes,), d, dtype=torch.long)
        # block-mid
        block = self._get_block(idx)
        planes = block.conv1.weight.shape[0]
        d_in = block.conv1.weight[0].numel()        # one conv1 filter
        d_in += 2                                    # bn1 weight + bias
        d_out = (block.conv2.weight.shape[0] *
                 block.conv2.weight.shape[2] *
                 block.conv2.weight.shape[3])        # one conv2 input channel slice
        return torch.full((planes,), d_in + d_out, dtype=torch.long)

    def structure_norms(self, ell):
        typ, idx = self._decode_ell(ell)
        if typ == 'stage':
            return self._stage_width_norms(idx)
        return self._block_mid_norms(idx)

    def _stage_width_norms(self, stage_idx):
        stage = self.stages[stage_idx]
        planes = self.stage_planes[stage_idx]
        parts = []

        # Fan-in to this stage: initial conv (stage 0) or downsample
        if stage_idx == 0:
            parts.append(self.conv1.weight.reshape(planes, -1))
            parts.append(self.bn1.weight.unsqueeze(1))
            parts.append(self.bn1.bias.unsqueeze(1))
        else:
            block0 = stage[0]
            if block0.downsample is not None:
                ds_conv = block0.downsample[0]
                ds_bn = block0.downsample[1]
                parts.append(ds_conv.weight.reshape(planes, -1))
                parts.append(ds_bn.weight.unsqueeze(1))
                parts.append(ds_bn.bias.unsqueeze(1))

        # conv2 + bn2 in every block of this stage
        for block in stage:
            parts.append(block.conv2.weight.reshape(planes, -1))
            parts.append(block.bn2.weight.unsqueeze(1))
            parts.append(block.bn2.bias.unsqueeze(1))

        # Fan-out to the next stage (or fc for the last stage)
        if stage_idx < self.n_stages - 1:
            next_stage = self.stages[stage_idx + 1]
            next_block0 = next_stage[0]
            next_conv1 = next_block0.conv1.weight
            parts.append(next_conv1.permute(1, 0, 2, 3).reshape(planes, -1))
            if next_block0.downsample is not None:
                next_ds = next_block0.downsample[0].weight
                parts.append(next_ds.permute(1, 0, 2, 3).reshape(planes, -1))
        else:
            # fc input column j (after avgpool, fc expects exactly stage3 channels)
            parts.append(self.fc.weight.t())  # shape (planes, num_classes)

        params = torch.cat(parts, dim=1)
        return params.norm(dim=1)

    def _block_mid_norms(self, block_idx):
        block = self._get_block(block_idx)
        planes = block.conv1.weight.shape[0]
        fan_in = block.conv1.weight.reshape(planes, -1)
        bn1_w = block.bn1.weight.unsqueeze(1)
        bn1_b = block.bn1.bias.unsqueeze(1)
        fan_out = block.conv2.weight.permute(1, 0, 2, 3).reshape(planes, -1)
        params = torch.cat([fan_in, bn1_w, bn1_b, fan_out], dim=1)
        return params.norm(dim=1)

    def prune_structures(self, ell, mask):
        """mask: bool tensor, True = prune."""
        typ, idx = self._decode_ell(ell)
        if typ == 'stage':
            self._prune_stage_width(idx, mask)
        else:
            self._prune_block_mid(idx, mask)

    @torch.no_grad()
    def _prune_stage_width(self, stage_idx, mask):
        stage = self.stages[stage_idx]
        if stage_idx == 0:
            self.conv1.weight.data[mask] = 0.0
            self.bn1.weight.data[mask] = 0.0
            self.bn1.bias.data[mask] = 0.0
            self.bn1.running_mean[mask] = 0.0
            self.bn1.running_var[mask] = 1.0
        else:
            block0 = stage[0]
            if block0.downsample is not None:
                block0.downsample[0].weight.data[mask] = 0.0
                block0.downsample[1].weight.data[mask] = 0.0
                block0.downsample[1].bias.data[mask] = 0.0
                block0.downsample[1].running_mean[mask] = 0.0
                block0.downsample[1].running_var[mask] = 1.0
        for block in stage:
            block.conv2.weight.data[mask] = 0.0
            block.bn2.weight.data[mask] = 0.0
            block.bn2.bias.data[mask] = 0.0
            block.bn2.running_mean[mask] = 0.0
            block.bn2.running_var[mask] = 1.0

    @torch.no_grad()
    def _prune_block_mid(self, block_idx, mask):
        block = self._get_block(block_idx)
        block.conv1.weight.data[mask] = 0.0
        block.bn1.weight.data[mask] = 0.0
        block.bn1.bias.data[mask] = 0.0
        block.bn1.running_mean[mask] = 0.0
        block.bn1.running_var[mask] = 1.0
        block.conv2.weight.data[:, mask] = 0.0

    def zero_pruned_grads(self, ell, mask):
        typ, idx = self._decode_ell(ell)
        if typ == 'stage':
            self._zero_stage_grads(idx, mask)
        else:
            self._zero_mid_grads(idx, mask)

    def _zero_stage_grads(self, stage_idx, mask):
        stage = self.stages[stage_idx]
        if stage_idx == 0:
            if self.conv1.weight.grad is not None:
                self.conv1.weight.grad[mask] = 0.0
            if self.bn1.weight.grad is not None:
                self.bn1.weight.grad[mask] = 0.0
            if self.bn1.bias.grad is not None:
                self.bn1.bias.grad[mask] = 0.0
        else:
            block0 = stage[0]
            if block0.downsample is not None:
                ds_conv = block0.downsample[0]
                ds_bn = block0.downsample[1]
                if ds_conv.weight.grad is not None:
                    ds_conv.weight.grad[mask] = 0.0
                if ds_bn.weight.grad is not None:
                    ds_bn.weight.grad[mask] = 0.0
                if ds_bn.bias.grad is not None:
                    ds_bn.bias.grad[mask] = 0.0
        for block in stage:
            if block.conv2.weight.grad is not None:
                block.conv2.weight.grad[mask] = 0.0
            if block.bn2.weight.grad is not None:
                block.bn2.weight.grad[mask] = 0.0
            if block.bn2.bias.grad is not None:
                block.bn2.bias.grad[mask] = 0.0

    def _zero_mid_grads(self, block_idx, mask):
        block = self._get_block(block_idx)
        if block.conv1.weight.grad is not None:
            block.conv1.weight.grad[mask] = 0.0
        if block.bn1.weight.grad is not None:
            block.bn1.weight.grad[mask] = 0.0
        if block.bn1.bias.grad is not None:
            block.bn1.bias.grad[mask] = 0.0
        if block.conv2.weight.grad is not None:
            block.conv2.weight.grad[:, mask] = 0.0


# ---------------------------------------------------------------------------
# Checkpoint loader for DepGraph's cifar10_resnet56.pth
# ---------------------------------------------------------------------------

def load_dg_checkpoint(path, model, strict=True):
    """Load DepGraph's cifar10_resnet56.pth into a ResNet56.

    Their state_dict uses keys identical to ours:
        conv1.weight, bn1.{weight,bias,running_mean,running_var,num_batches_tracked}
        layer{1,2,3}.{0..8}.{conv1,bn1,conv2,bn2}.*
        layer{2,3}.0.downsample.{0,1}.*
        fc.{weight,bias}
    so a direct load_state_dict works.
    """
    sd = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    if missing or unexpected:
        print(f"  load_dg_checkpoint: missing={missing}  unexpected={unexpected}")
    return model


if __name__ == "__main__":
    import sys
    model = ResNet56(num_classes=10)
    print(f"ResNet56 built.")
    print(f"  total params (incl. BN buffers): "
          f"{sum(v.numel() for v in model.state_dict().values()):,}")
    print(f"  num_hidden = {model.num_hidden}  "
          f"(stage-width: {model.n_stages}, block-mid: {model.n_stages * model.n_per_stage})")
    print(f"  stage_planes = {model.stage_planes}")
    if len(sys.argv) >= 2:
        load_dg_checkpoint(sys.argv[1], model, strict=True)
        print("Loaded DepGraph checkpoint successfully.")
        x = torch.randn(2, 3, 32, 32)
        y = model(x)
        print(f"Forward pass OK: input {tuple(x.shape)} -> output {tuple(y.shape)}")
