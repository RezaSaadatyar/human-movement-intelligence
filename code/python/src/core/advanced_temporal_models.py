# ======================== Presented by: Reza Saadatyar (2026) ========================
"""Four reproducible PyTorch architectures for multichannel time-series classification.

All models accept float32 windows shaped (batch, channels, time) and return
raw class logits shaped (batch, classes). No softmax appears in forward().
This module is intentionally independent of any HARTH CSV path or target names.
"""

import math

import torch
from torch import nn


ARCHITECTURES = (
    "TCN", "Multi-Scale CNN", "1D-ResNet", "Time-Series Transformer"
)


# ==================== 1. Shared Input and Model Contracts ====================
def _positive_integer(value, name, minimum=1):
    """Require genuine positive integer sizes, rejecting boolean values."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


def _validate_model_input(x, channels, minimum_steps=8):
    """Reject invalid multichannel time-series tensors before inference."""
    if x.ndim != 3 or x.shape[1] != channels or x.shape[2] < minimum_steps:
        raise ValueError(
            f"Expected (batch, {channels}, time) with time >= {minimum_steps}."
        )
    if x.shape[0] < 1:
        raise ValueError("At least one input window is required.")


def count_trainable_parameters(model):
    """
    Count genuinely trainable network parameters.

    Args:
        model (nn.Module): Instantiated PyTorch architecture.

    Returns:
        int: Number of parameters with requires_grad=True.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("Expected a PyTorch neural-network module.")
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==================== 2. Temporal Convolutional Network ====================
class _DilatedResidualBlock(nn.Module):
    """Same-length, noncausal dilated convolutions with residual projection."""

    def __init__(self, in_channels, out_channels, dilation, dropout):
        super().__init__()
        padding = 2 * dilation  # Five-point odd kernel; preserve length.
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 5,
                      padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, 5,
                      padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(dropout),
        )
        self.skip = (
            nn.Identity() if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, 1)
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(self.layers(x) + self.skip(x))


class TemporalConvNet(nn.Module):
    """
    Dilated residual TCN for fixed-window movement recognition.

    Args:
        in_channels (int): Number of original sensor channels.
        n_classes (int): Number of distinct activity target codes.
        hidden_channels (int): Number of feature maps throughout the network.
        levels (int): Residual blocks with dilations 1, 2, 4, ...
        dropout (float): Dropout probability inside each residual block.

    Input:
        torch.Tensor: (batch, channels, time).

    Output:
        torch.Tensor: Unnormalised logits (batch, n_classes).

    Notes:
        Convolutions are deliberately NONCAUSAL because whole historical
        two-second windows are classified after acquisition. Do not present
        this as an online or causal future-blind predictor.
    """

    def __init__(self, in_channels=6, n_classes=8,
                 hidden_channels=32, levels=4, dropout=0.1):
        super().__init__()
        _positive_integer(in_channels, "in_channels")
        _positive_integer(n_classes, "n_classes", minimum=2)
        _positive_integer(hidden_channels, "hidden_channels")
        _positive_integer(levels, "levels")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        self.in_channels = in_channels
        blocks = []
        for level in range(levels):
            blocks.append(_DilatedResidualBlock(
                in_channels if level == 0 else hidden_channels,
                hidden_channels, dilation=2 ** level, dropout=dropout
            ))
        self.features = nn.Sequential(*blocks, nn.AdaptiveAvgPool1d(1))
        self.classifier = nn.Linear(hidden_channels, n_classes)

    def forward(self, x):
        _validate_model_input(x, self.in_channels)
        return self.classifier(self.features(x).squeeze(-1))


# ==================== 3. Multi-Scale CNN / Inception-Style ====================
class _MultiScaleResidualBlock(nn.Module):
    """Parallel 3/7/15-point kernels, pooling branch and residual shortcut."""

    def __init__(self, in_channels, branch_channels=16):
        super().__init__()
        reduced = 16
        self.bottleneck = nn.Sequential(
            nn.Conv1d(in_channels, reduced, kernel_size=1),
            nn.BatchNorm1d(reduced), nn.GELU(),
        )
        self.branches = nn.ModuleList([
            nn.Conv1d(reduced, branch_channels, kernel_size=kernel,
                      padding=kernel // 2)
            for kernel in (3, 7, 15)
        ])
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
        )
        out_channels = 4 * branch_channels
        self.norm = nn.BatchNorm1d(out_channels)
        self.skip = (
            nn.Identity() if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )
        self.activation = nn.GELU()

    def forward(self, x):
        reduced = self.bottleneck(x)
        branches = [layer(reduced) for layer in self.branches]
        branches.append(self.pool_branch(x))
        return self.activation(
            self.norm(torch.cat(branches, dim=1)) + self.skip(x)
        )


class MultiScaleCNN(nn.Module):
    """
    Lightweight InceptionTime-inspired, multi-resolution 1D CNN.

    Args:
        in_channels (int): Original IMU channel count.
        n_classes (int): Target class count.
        branch_channels (int): Feature maps in each parallel convolution.
        blocks (int): Number of residual multi-scale extraction blocks.

    Input:
        torch.Tensor: (batch, channels, time).

    Output:
        torch.Tensor: Unnormalised logits (batch, classes).

    Notes:
        This is an InceptionTime-STYLE small research model, not a claim
        of exact replication of the full published InceptionTime ensemble.
    """

    def __init__(self, in_channels=6, n_classes=8,
                 branch_channels=16, blocks=2):
        super().__init__()
        _positive_integer(in_channels, "in_channels")
        _positive_integer(n_classes, "n_classes", 2)
        _positive_integer(branch_channels, "branch_channels")
        _positive_integer(blocks, "blocks")
        self.in_channels = in_channels
        out = 4 * branch_channels
        self.features = nn.Sequential(*[
            _MultiScaleResidualBlock(
                in_channels if i == 0 else out, branch_channels
            ) for i in range(blocks)
        ], nn.AdaptiveAvgPool1d(1))
        self.classifier = nn.Linear(out, n_classes)

    def forward(self, x):
        _validate_model_input(x, self.in_channels)
        return self.classifier(self.features(x).squeeze(-1))


# ==================== 4. One-Dimensional Residual Network ====================
class _Residual1DBlock(nn.Module):
    """Two 1D convolutions and identity/projection skip connections."""

    def __init__(self, input_channels, output_channels, kernel_size=5,
                 stride=1):
        super().__init__()
        pad = kernel_size // 2
        self.body = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, kernel_size,
                      stride=stride, padding=pad),
            nn.BatchNorm1d(output_channels),
            nn.GELU(),
            nn.Conv1d(output_channels, output_channels, kernel_size,
                      padding=pad),
            nn.BatchNorm1d(output_channels),
        )
        self.skip = (
            nn.Identity()
            if input_channels == output_channels and stride == 1
            else nn.Sequential(
                nn.Conv1d(input_channels, output_channels, 1, stride=stride),
                nn.BatchNorm1d(output_channels)
            )
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(self.body(x) + self.skip(x))


class ResNet1D(nn.Module):
    """
    Small residual convolutional classifier for HAR time-series windows.

    Args:
        in_channels (int): Original sensor axes.
        n_classes (int): Target class count.
        hidden_channels (int): Width of initial convolution and first block.

    Input:
        torch.Tensor: (batch, channels, time).

    Output:
        torch.Tensor: Raw class logits (batch, classes).

    Notes:
        Includes one stride-two feature block and global average pooling.
        It is a lightweight 1D-ResNet, not a transferred image ResNet.
    """

    def __init__(self, in_channels=6, n_classes=8, hidden_channels=32):
        super().__init__()
        _positive_integer(in_channels, "in_channels")
        _positive_integer(n_classes, "n_classes", 2)
        _positive_integer(hidden_channels, "hidden_channels")
        self.in_channels = in_channels
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 7, padding=3),
            nn.BatchNorm1d(hidden_channels), nn.GELU(),
            _Residual1DBlock(hidden_channels, hidden_channels, 7),
            _Residual1DBlock(hidden_channels, hidden_channels * 2,
                             kernel_size=5, stride=2),
            _Residual1DBlock(hidden_channels * 2, hidden_channels * 2, 5),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(hidden_channels * 2, n_classes)

    def forward(self, x):
        _validate_model_input(x, self.in_channels)
        return self.classifier(self.features(x).squeeze(-1))


# ==================== 5. Time-Series Transformer ====================
def _sinusoidal_encoding(max_steps, dimension):
    """Deterministic position encoding for chronological input samples."""
    positions = torch.arange(max_steps, dtype=torch.float32).unsqueeze(1)
    scales = torch.exp(
        torch.arange(0, dimension, 2, dtype=torch.float32) *
        (-math.log(10000.0) / dimension)
    )
    encoding = torch.zeros(1, max_steps, dimension, dtype=torch.float32)
    encoding[0, :, 0::2] = torch.sin(positions * scales)
    encoding[0, :, 1::2] = torch.cos(positions * scales[:dimension//2])
    return encoding


class TimeSeriesTransformer(nn.Module):
    """
    Full-window self-attention classifier with sinusoidal temporal encoding.

    Args:
        in_channels (int): Source accelerometer channel count.
        n_classes (int): Number of original target activity codes.
        d_model (int): Embedding size, divisible by attention heads.
        n_heads (int): Parallel self-attention heads.
        n_layers (int): Transformer encoder blocks.
        feedforward_dim (int): Hidden layer size within encoder feedforward.
        dropout (float): Dropout for training regularisation.
        max_length (int): Maximum supported source window time steps.

    Input:
        torch.Tensor: (batch, channels, time) with time <= max_length.

    Output:
        torch.Tensor: Unnormalised logits (batch, classes).

    Notes:
        Full-window bidirectional attention is NOT a causal streaming
        model. Sensor channels are projected jointly at each time step.
        Positional encoding maintains chronological information.
    """

    def __init__(self, in_channels=6, n_classes=8, d_model=48,
                 n_heads=4, n_layers=2, feedforward_dim=96, dropout=0.1,
                 max_length=256):
        super().__init__()
        for name, value, minimum in [
            ("in_channels", in_channels, 1), ("n_classes", n_classes, 2),
            ("d_model", d_model, 2), ("n_heads", n_heads, 1),
            ("n_layers", n_layers, 1), ("feedforward_dim", feedforward_dim, 1),
            ("max_length", max_length, 8),
        ]:
            _positive_integer(value, name, minimum)
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        self.in_channels = in_channels
        self.max_length = max_length
        self.input_projection = nn.Linear(in_channels, d_model)
        self.register_buffer(
            "position_encoding", _sinusoidal_encoding(max_length, d_model),
            persistent=False
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=feedforward_dim, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.output_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x):
        _validate_model_input(x, self.in_channels)
        if x.shape[-1] > self.max_length:
            raise ValueError("Window length exceeds configured max_length.")
        embeddings = self.input_projection(x.transpose(1, 2))
        embeddings = embeddings + self.position_encoding[:, :x.shape[-1]]
        features = self.encoder(embeddings).mean(dim=1)
        return self.classifier(self.output_norm(features))


# ==================== 6. Common Architecture Registry ====================
def build_temporal_model(architecture, in_channels=6, n_classes=8, **kwargs):
    """
    Construct exactly one of the four declared PyTorch time-series models.

    Args:
        architecture (str): One name from ARCHITECTURES.
        in_channels (int): Number of time-aligned source sensor channels.
        n_classes (int): Number of supported original activity codes.
        **kwargs: Architecture-specific width, kernel or attention settings.

    Returns:
        nn.Module: New, untrained network producing unnormalised class logits.

    Notes:
        The factory never loads hidden pretrained weights, trained artefacts,
        personal data, or any Test-split signals. Training is a separate step.
    """
    registry = {
        "TCN": TemporalConvNet,
        "Multi-Scale CNN": MultiScaleCNN,
        "1D-ResNet": ResNet1D,
        "Time-Series Transformer": TimeSeriesTransformer,
    }
    if architecture not in registry:
        raise ValueError(
            f"Unknown model {architecture!r}; choose from {ARCHITECTURES}."
        )
    return registry[architecture](
        in_channels=in_channels, n_classes=n_classes, **kwargs
    )
