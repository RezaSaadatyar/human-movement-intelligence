# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Compact unimodal temporal classifiers and paired multimodal fusion models."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


# =========================== 01. Temporal Encoders ===========================
class ResidualTemporalBlock(nn.Module):
    """Two dilated convolutions with a channel-matching residual connection."""

    def __init__(self, in_channels: int, channels: int, dilation: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(in_channels, channels, 5, padding=2 * dilation, dilation=dilation),
            nn.BatchNorm1d(channels), nn.GELU(), nn.Dropout(0.10),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(channels),
        )
        self.residual = nn.Identity() if in_channels == channels else nn.Conv1d(in_channels, channels, 1)

    def forward(self, x):
        return F.gelu(self.network(x) + self.residual(x))


class TemporalEncoder(nn.Module):
    """Map [batch, channels, samples] to learned temporal embeddings."""

    def __init__(self, input_channels: int, architecture: str, width: int = 64):
        super().__init__()
        self.architecture = architecture
        self.width = width
        if architecture == "cnn":
            self.layers = nn.Sequential(
                nn.Conv1d(input_channels, width, 9, stride=3, padding=4),
                nn.BatchNorm1d(width), nn.GELU(),
                nn.Conv1d(width, width, 5, stride=2, padding=2),
                nn.BatchNorm1d(width), nn.GELU(),
                nn.Conv1d(width, width, 3, stride=2, padding=1),
                nn.BatchNorm1d(width), nn.GELU(),
            )
        elif architecture == "tcn":
            self.layers = nn.Sequential(
                nn.AvgPool1d(kernel_size=4, stride=4),
                ResidualTemporalBlock(input_channels, width, 1),
                ResidualTemporalBlock(width, width, 2),
                ResidualTemporalBlock(width, width, 4),
            )
        elif architecture == "bigru":
            self.rnn = nn.GRU(input_channels, width // 2,
                              batch_first=True, bidirectional=True)
        else:
            raise ValueError(f"Unsupported temporal architecture: {architecture}")

    def sequence(self, x):
        if self.architecture == "bigru":
            # Pool before recurrent processing to bound sequence length/cost.
            if x.shape[-1] >= 8:
                x = F.avg_pool1d(x, 8, 8)
            z, _ = self.rnn(x.transpose(1, 2))
            return z.transpose(1, 2)
        return self.layers(x)

    def forward(self, x):
        return self.sequence(x).mean(dim=-1)


# ======================== 02. Modality-Specific Models =======================
class UnimodalClassifier(nn.Module):
    """One modality, one temporal encoder and a fixed-class linear head."""

    def __init__(self, channels: int, classes: int, architecture: str, width: int = 64):
        super().__init__()
        self.encoder = TemporalEncoder(channels, architecture, width)
        self.head = nn.Sequential(nn.Dropout(0.15), nn.Linear(width, classes))

    def forward(self, emg, acc=None):
        return self.head(self.encoder(emg))


# ========================== 03. Multimodal Fusion ============================
class PairedFusionClassifier(nn.Module):
    """Two source-aligned branches with controlled learnable interactions.

    Fusion mode: concat, gated, cross_attention. The attention design uses
    eight pooled tokens and two heads; it is not a pretrained Transformer.
    """

    def __init__(self, classes: int, *, backbone: str = "cnn",
                 fusion: str = "concat", width: int = 64):
        super().__init__()
        self.emg_encoder = TemporalEncoder(12, backbone, width)
        self.acc_encoder = TemporalEncoder(36, backbone, width)
        self.fusion = fusion
        if fusion == "concat":
            feature_width = width * 2
        elif fusion == "gated":
            self.gate = nn.Linear(width * 2, width)
            feature_width = width
        elif fusion == "cross_attention":
            self.attention = nn.MultiheadAttention(width, num_heads=2, batch_first=True)
            self.norm = nn.LayerNorm(width)
            feature_width = width * 2
        else:
            raise ValueError(f"Unknown paired fusion mode: {fusion}")
        self.head = nn.Sequential(nn.Dropout(0.20), nn.Linear(feature_width, width),
                                  nn.GELU(), nn.Linear(width, classes))

    def forward(self, emg, acc):
        emg_seq = self.emg_encoder.sequence(emg)
        acc_seq = self.acc_encoder.sequence(acc)
        e = emg_seq.mean(dim=-1)
        a = acc_seq.mean(dim=-1)
        if self.fusion == "concat":
            fused = torch.cat([e, a], dim=1)
        elif self.fusion == "gated":
            weight = torch.sigmoid(self.gate(torch.cat([e, a], dim=1)))
            fused = weight * e + (1.0 - weight) * a
        else:
            query = F.adaptive_avg_pool1d(emg_seq, 8).transpose(1, 2)
            context = F.adaptive_avg_pool1d(acc_seq, 8).transpose(1, 2)
            attended, _ = self.attention(query, context, context, need_weights=False)
            fused = torch.cat([e, self.norm(query + attended).mean(dim=1)], dim=1)
        return self.head(fused)


# ======================= 04. Registered Study Architectures ==================
UNIMODAL_MODELS = (
    "D1_emg_cnn", "D1_acc_cnn",
    "D2_emg_tcn", "D2_acc_tcn",
    "D3_emg_bigru", "D3_acc_bigru",
)
FUSION_MODELS = (
    "F3_dual_cnn", "F3_dual_tcn", "F5_gated", "F5_cross_attention",
)


def build_neural_model(name: str, number_of_classes: int) -> nn.Module:
    """One authoritative mapping from experiment identifiers to architectures."""
    if name in UNIMODAL_MODELS:
        _, modality, architecture = name.split("_")
        channels = 12 if modality == "emg" else 36
        return UnimodalClassifier(channels, number_of_classes, architecture)
    if name == "F3_dual_cnn":
        return PairedFusionClassifier(number_of_classes, backbone="cnn", fusion="concat")
    if name == "F3_dual_tcn":
        return PairedFusionClassifier(number_of_classes, backbone="tcn", fusion="concat")
    if name == "F5_gated":
        return PairedFusionClassifier(number_of_classes, backbone="cnn", fusion="gated")
    if name == "F5_cross_attention":
        return PairedFusionClassifier(number_of_classes, backbone="cnn", fusion="cross_attention")
    raise ValueError(f"Unrecognised architecture identifier: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
