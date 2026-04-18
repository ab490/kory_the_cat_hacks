# Two small nets:
#   DBNet  — predicts a text-probability map on the full page, boxes come from
#            thresholding + contours in infer.py.
#   CRNN   — CNN feature extractor + BiLSTM, trained with CTC loss on line crops.

import torch
from torch import nn
import torch.nn.functional as F

from config import CHARSET


def conv_bn(in_c, out_c, k=3, s=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, s, k // 2, groups=groups, bias=False),
        nn.BatchNorm2d(out_c),
        nn.SiLU(inplace=True),
    )


def dw_sep(in_c, out_c, s=1):
    # depthwise 3x3 + pointwise 1x1 — cheap substitute for a plain conv
    return nn.Sequential(conv_bn(in_c, in_c, 3, s, in_c), conv_bn(in_c, out_c, 1, 1))


class DBNet(nn.Module):
    """Compact text-segmentation head. 4-stage encoder + FPN fuse -> 1-channel logits."""

    def __init__(self, inner=64):
        super().__init__()
        self.s1 = conv_bn(1, 16, 3, 2)
        self.s2 = dw_sep(16, 24, 2)
        self.s3 = dw_sep(24, 40, 2)
        self.s4 = dw_sep(40, 80, 2)
        self.l1 = nn.Conv2d(16, inner, 1)
        self.l2 = nn.Conv2d(24, inner, 1)
        self.l3 = nn.Conv2d(40, inner, 1)
        self.l4 = nn.Conv2d(80, inner, 1)
        self.fuse = nn.Sequential(
            conv_bn(inner * 4, inner, 3), dw_sep(inner, inner), nn.Conv2d(inner, 1, 1)
        )

    def forward(self, x):
        hw = x.shape[-2:]
        c1 = self.s1(x); c2 = self.s2(c1); c3 = self.s3(c2); c4 = self.s4(c3)
        t = c1.shape[-2:]
        feats = [
            self.l1(c1),
            F.interpolate(self.l2(c2), t, mode="bilinear", align_corners=False),
            F.interpolate(self.l3(c3), t, mode="bilinear", align_corners=False),
            F.interpolate(self.l4(c4), t, mode="bilinear", align_corners=False),
        ]
        logits = self.fuse(torch.cat(feats, 1))
        return F.interpolate(logits, hw, mode="bilinear", align_corners=False)


class CRNN(nn.Module):
    """CNN features -> BiLSTM -> per-timestep class logits for CTC."""

    def __init__(self, num_classes=None, hidden=128):
        super().__init__()
        self.num_classes = num_classes or (len(CHARSET) + 1)  # +1 for blank
        self.cnn = nn.Sequential(
            conv_bn(1, 32), nn.MaxPool2d(2, 2),
            conv_bn(32, 64), nn.MaxPool2d(2, 2),
            conv_bn(64, 128), conv_bn(128, 128),
            nn.MaxPool2d((2, 1), (2, 1)),        # squeeze height, keep width
            conv_bn(128, 256), conv_bn(256, 256),
            nn.MaxPool2d((2, 1), (2, 1)),
            conv_bn(256, 256),
        )
        self.lstm = nn.LSTM(256, hidden, num_layers=2, bidirectional=True, dropout=0.1)
        self.fc = nn.Linear(hidden * 2, self.num_classes)

    def forward(self, x):
        feats = self.cnn(x)                                 # [B, 256, h', W']
        feats = F.adaptive_avg_pool2d(feats, (1, feats.shape[-1])).squeeze(2)
        seq = feats.permute(2, 0, 1).contiguous()           # [T, B, 256]
        seq, _ = self.lstm(seq)
        return self.fc(seq)                                 # [T, B, C]
