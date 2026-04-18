import torch
from torch import nn
import torch.nn.functional as F

from config import CHARSET


class ConvBNAct(nn.Sequential):
    def __init__(self, in_c, out_c, k=3, s=1, groups=1):
        super().__init__(
            nn.Conv2d(in_c, out_c, k, s, k // 2, groups=groups, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True),
        )


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_c, out_c, s=1):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(in_c, in_c, 3, s, groups=in_c),
            ConvBNAct(in_c, out_c, 1, 1),
        )

    def forward(self, x):
        return self.block(x)


class DBNet(nn.Module):
    def __init__(self, inner=64):
        super().__init__()
        self.stem = ConvBNAct(1, 16, 3, 2)
        self.stage2 = DepthwiseSeparableBlock(16, 24, 2)
        self.stage3 = DepthwiseSeparableBlock(24, 40, 2)
        self.stage4 = DepthwiseSeparableBlock(40, 80, 2)

        self.lat1 = nn.Conv2d(16, inner, 1)
        self.lat2 = nn.Conv2d(24, inner, 1)
        self.lat3 = nn.Conv2d(40, inner, 1)
        self.lat4 = nn.Conv2d(80, inner, 1)

        self.fuse = nn.Sequential(
            ConvBNAct(inner * 4, inner, 3),
            DepthwiseSeparableBlock(inner, inner),
            nn.Conv2d(inner, 1, 1),
        )

    def forward(self, x):
        hw = x.shape[-2:]
        c1 = self.stem(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        t = c1.shape[-2:]
        feats = [
            self.lat1(c1),
            F.interpolate(self.lat2(c2), t, mode="bilinear", align_corners=False),
            F.interpolate(self.lat3(c3), t, mode="bilinear", align_corners=False),
            F.interpolate(self.lat4(c4), t, mode="bilinear", align_corners=False),
        ]
        logits = self.fuse(torch.cat(feats, 1))
        return F.interpolate(logits, hw, mode="bilinear", align_corners=False)


class CRNN(nn.Module):
    def __init__(self, num_classes=None, hidden=128, lstm_layers=2):
        super().__init__()
        self.num_classes = num_classes or (len(CHARSET) + 1)
        self.cnn = nn.Sequential(
            ConvBNAct(1, 32), nn.MaxPool2d(2, 2),
            ConvBNAct(32, 64), nn.MaxPool2d(2, 2),
            ConvBNAct(64, 128), ConvBNAct(128, 128),
            nn.MaxPool2d((2, 1), (2, 1)),
            ConvBNAct(128, 256), ConvBNAct(256, 256),
            nn.MaxPool2d((2, 1), (2, 1)),
            ConvBNAct(256, 256),
        )
        self.sequence = nn.LSTM(
            256, hidden, num_layers=lstm_layers,
            bidirectional=True,
            dropout=0.1 if lstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden * 2, self.num_classes)

    def forward(self, x):
        feats = self.cnn(x)
        feats = F.adaptive_avg_pool2d(feats, (1, feats.shape[-1])).squeeze(2)
        seq = feats.permute(2, 0, 1).contiguous()
        seq, _ = self.sequence(seq)
        return self.classifier(seq)
