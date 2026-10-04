from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """SE-like channel attention with average + max pooling."""

    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        hidden_channels = max(1, in_channels // reduction_ratio)
        self.shared_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, in_channels),
        )
        self.shared_mlp_max = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, in_channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.shared_mlp(x)
        max_out = self.shared_mlp_max(x)
        return self.sigmoid(avg_out + max_out).view(x.size(0), x.size(1), 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # avg_map and max_map are each single-channel, so the conv always sees 2 channels.
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_map, max_map], dim=1)
        return self.conv(combined)


class CBAMBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)
        self.channel_attention = ChannelAttention(out_channels)
        self.spatial_attention = SpatialAttention(out_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = self.conv(x)
        attention = self.channel_attention(residual) * residual
        spatial_attention = self.spatial_attention(attention)
        out = attention * spatial_attention
        out = self.relu(out)
        return out, spatial_attention


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class RespiroAttnNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.cbam = CBAMBlock(256, 256)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        features, spatial_attention = self.cbam(features)
        logits = self.classifier(features)
        return logits, spatial_attention


class PlainCNN(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 3):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        logits = self.classifier(features)
        activations = features.mean(dim=1, keepdim=True)
        attention = activations / (activations.abs().max().clamp_min(1e-8))
        return logits, attention


def resize_attention(attention_map: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
    if attention_map.dim() == 3:
        attention_map = attention_map.unsqueeze(0)
    if attention_map.shape[1] != 1:
        attention_map = attention_map.mean(dim=1, keepdim=True)
    return F.interpolate(attention_map, size=target_size, mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
