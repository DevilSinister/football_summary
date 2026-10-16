"""The action-spotting network in models/checkpoint_088.pt.

The checkpoint is an E2E-Spot model (Hong et al., "Spotting Temporally
Precise, Fine-Grained Events in Video", ECCV 2022) in its ``rny002_gsm`` +
``gru`` configuration, trained on SoccerNet-v2 (models/config.json):

* ``_features``: timm RegNetY-200MF without its classifier (368-d per frame),
  with a Gate-Shift Module (Sudhakaran et al., CVPR 2020) wrapped around the
  first convolution of every block. The GSM gates a quarter of the channels and
  shifts them one frame forward or back, so each frame's features see their
  neighbours; it fixes the clip length the network runs on.
* ``_pred_fine``: a bidirectional GRU over the clip and a per-frame linear
  layer to 18 scores (background plus 17 SoccerNet-v2 actions).

The checkpoint is a plain state_dict and is loaded with ``weights_only=True``
and ``strict=True``: any mismatch with this definition fails loudly instead of
running a half-loaded network.
"""

import json
import math
import os
from typing import Optional

import torch
import torch.nn as nn

DEFAULT_CONFIG = {
    "clip_len": 100,
    "crop_dim": None,
    "feature_arch": "rny002_gsm",
    "temporal_arch": "gru",
    "num_classes": 17,
    "modality": "rgb",
}
FEATURE_DIM = 368          # RegNetY-200MF final width
GSM_CHANNEL_DIVISOR = 4    # E2E-Spot: a quarter of the channels are gated and shifted


class _GSM(nn.Module):
    """Gate-Shift Module over ``planes`` channels of an (N*T, C, H, W) batch."""

    def __init__(self, planes: int, num_segments: int) -> None:
        super().__init__()
        self.conv3D = nn.Conv3d(planes, 2, (3, 3, 3), stride=1, padding=(1, 1, 1), groups=2)
        self.tanh = nn.Tanh()
        self.planes = planes
        self.num_segments = num_segments
        self.bn = nn.BatchNorm3d(num_features=planes)
        self.relu = nn.ReLU()

    @staticmethod
    def _shift_left(x: torch.Tensor) -> torch.Tensor:
        return torch.cat((x[:, :, 1:], torch.zeros_like(x[:, :, :1])), dim=2)

    @staticmethod
    def _shift_right(x: torch.Tensor) -> torch.Tensor:
        return torch.cat((torch.zeros_like(x[:, :, :1]), x[:, :, :-1]), dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0) // self.num_segments
        shape = x.size(1), x.size(2), x.size(3)
        x = x.view(batch, self.num_segments, *shape).permute(0, 2, 1, 3, 4).contiguous()
        gate = self.tanh(self.conv3D(self.relu(self.bn(x))))
        gate_1, gate_2 = gate[:, 0].unsqueeze(1), gate[:, 1].unsqueeze(1)
        half = self.planes // 2
        x_1, x_2 = x[:, :half], x[:, half:]
        y_1, y_2 = gate_1 * x_1, gate_2 * x_2
        y_1 = self._shift_left(y_1) + (x_1 - y_1)
        y_2 = self._shift_right(y_2) + (x_2 - y_2)
        quarter = self.planes // 4
        y_1 = y_1.view(batch, 2, quarter, self.num_segments, *shape[1:]).permute(0, 2, 1, 3, 4, 5)
        y_2 = y_2.view(batch, 2, quarter, self.num_segments, *shape[1:]).permute(0, 2, 1, 3, 4, 5)
        y = torch.cat(
            (
                y_1.contiguous().view(batch, half, self.num_segments, *shape[1:]),
                y_2.contiguous().view(batch, half, self.num_segments, *shape[1:]),
            ),
            dim=1,
        )
        return y.permute(0, 2, 1, 3, 4).contiguous().view(batch * self.num_segments, *shape)


class _GatedShift(nn.Module):
    """Runs the GSM on the first ``fold`` channels, then the wrapped conv."""

    def __init__(self, net: nn.Module, num_segments: int) -> None:
        super().__init__()
        channels = net.conv.in_channels
        self.fold = int(math.ceil(channels // GSM_CHANNEL_DIVISOR / 4) * 4)
        self.gsm = _GSM(self.fold, num_segments)
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.cat((self.gsm(x[:, : self.fold]), x[:, self.fold:]), dim=1)
        return self.net(y)


class _FCPrediction(nn.Module):
    def __init__(self, feat_dim: int, num_classes: int) -> None:
        super().__init__()
        self._fc_out = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        return self._fc_out(x.reshape(batch * length, -1)).view(batch, length, -1)


class _GRUPrediction(nn.Module):
    def __init__(self, feat_dim: int, num_classes: int, hidden_dim: int) -> None:
        super().__init__()
        self._gru = nn.GRU(feat_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self._fc_out = _FCPrediction(2 * hidden_dim, num_classes)
        self._dropout = nn.Dropout()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self._gru(x)
        return self._fc_out(self._dropout(y))


class SpottingNet(nn.Module):
    """(batch, clip_len, 3, H, W) normalised RGB -> (batch, clip_len, classes + 1)."""

    def __init__(self, clip_len: int, num_classes: int) -> None:
        super().__init__()
        import timm  # deferred: only the spotting path needs it

        features = timm.create_model("regnety_002", pretrained=False)
        features.reset_classifier(0)
        for stage in (features.s1, features.s2, features.s3, features.s4):
            for block in stage.children():
                block.conv1 = _GatedShift(block.conv1, clip_len)
        self._features = features
        self._pred_fine = _GRUPrediction(FEATURE_DIM, num_classes + 1, FEATURE_DIM)
        self.clip_len = int(clip_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, channels, height, width = x.shape
        if length < self.clip_len:  # the GSM needs exactly clip_len frames
            pad = x.new_zeros(batch, self.clip_len - length, channels, height, width)
            x = torch.cat((x, pad), dim=1)
        feats = self._features(x.reshape(-1, channels, height, width))
        feats = feats.reshape(batch, self.clip_len, FEATURE_DIM)[:, :length]
        return self._pred_fine(feats)


def load_config(path: Optional[str]) -> dict:
    config = dict(DEFAULT_CONFIG)
    if path and os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            config.update(json.load(handle))
    if config.get("feature_arch") != "rny002_gsm" or config.get("temporal_arch") != "gru":
        raise ValueError(
            f"Unsupported spotting architecture {config.get('feature_arch')}/{config.get('temporal_arch')}; "
            "only rny002_gsm + gru is implemented"
        )
    if config.get("modality", "rgb") != "rgb":
        raise ValueError(f"Unsupported spotting modality {config.get('modality')}")
    return config


def load_spotting_model(weights_path: str, config: dict) -> SpottingNet:
    model = SpottingNet(int(config["clip_len"]), int(config["num_classes"]))
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model
