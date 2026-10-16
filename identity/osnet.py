"""OSNet x1.0 person re-identification network (Zhou et al., ICCV 2019).

A minimal inference-only copy of the architecture in torchreid
(deep-person-reid, MIT licence) so the project does not depend on torchreid.
Weights: models/reid/osnet_x1_0_msmt17.pt from the torchreid model zoo
(MSMT17, 1041 identities; sha256 b7d73dc6...1cf2 as downloaded 2026-10-14).
They are loaded with ``weights_only=True`` and ``strict=True``; the identity
classifier is dropped, leaving the 512-d embedding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

CHANNELS = (64, 256, 384, 512)
LAYERS = (2, 2, 2)
FEATURE_DIM = 512
INPUT_SIZE = (256, 128)  # height, width


class _ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class _Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class _Conv1x1Linear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class _LightConv3x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False, groups=out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class _ChannelGate(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=True, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=True, padding=0)
        self.gate_activation = nn.Sigmoid()

    def forward(self, x):
        gate = self.gate_activation(self.fc2(self.relu(self.fc1(self.global_avgpool(x)))))
        return x * gate


class _OSBlock(nn.Module):
    """Omni-scale block: four streams of 1-4 stacked 3x3 convs, gated and summed."""

    def __init__(self, in_channels, out_channels, bottleneck_reduction=4):
        super().__init__()
        mid = out_channels // bottleneck_reduction
        self.conv1 = _Conv1x1(in_channels, mid)
        self.conv2a = _LightConv3x3(mid, mid)
        self.conv2b = nn.Sequential(_LightConv3x3(mid, mid), _LightConv3x3(mid, mid))
        self.conv2c = nn.Sequential(*[_LightConv3x3(mid, mid) for _ in range(3)])
        self.conv2d = nn.Sequential(*[_LightConv3x3(mid, mid) for _ in range(4)])
        self.gate = _ChannelGate(mid)
        self.conv3 = _Conv1x1Linear(mid, out_channels)
        self.downsample = _Conv1x1Linear(in_channels, out_channels) if in_channels != out_channels else None

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = (
            self.gate(self.conv2a(x1))
            + self.gate(self.conv2b(x1))
            + self.gate(self.conv2c(x1))
            + self.gate(self.conv2d(x1))
        )
        identity = self.downsample(x) if self.downsample is not None else x
        return F.relu(self.conv3(x2) + identity)


def _make_layer(layers, in_channels, out_channels, reduce_spatial_size):
    blocks = [_OSBlock(in_channels, out_channels)]
    blocks += [_OSBlock(out_channels, out_channels) for _ in range(1, layers)]
    if reduce_spatial_size:
        blocks.append(nn.Sequential(_Conv1x1(out_channels, out_channels), nn.AvgPool2d(2, stride=2)))
    return nn.Sequential(*blocks)


class OSNet(nn.Module):
    """(N, 3, 256, 128) ImageNet-normalised RGB -> (N, 512) embeddings."""

    def __init__(self):
        super().__init__()
        self.conv1 = _ConvLayer(3, CHANNELS[0], 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = _make_layer(LAYERS[0], CHANNELS[0], CHANNELS[1], True)
        self.conv3 = _make_layer(LAYERS[1], CHANNELS[1], CHANNELS[2], True)
        self.conv4 = _make_layer(LAYERS[2], CHANNELS[2], CHANNELS[3], False)
        self.conv5 = _Conv1x1(CHANNELS[3], CHANNELS[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(CHANNELS[3], FEATURE_DIM), nn.BatchNorm1d(FEATURE_DIM), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv5(self.conv4(self.conv3(self.conv2(self.maxpool(self.conv1(x))))))
        return self.fc(self.global_avgpool(x).flatten(1))


def load_osnet(weights_path: str) -> OSNet:
    model = OSNet()
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    state = {key.replace("module.", "", 1): value for key, value in state.items() if not key.startswith("classifier.")}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model
