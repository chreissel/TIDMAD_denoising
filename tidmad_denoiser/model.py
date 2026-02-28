"""
AttentionAutoencoder for 1-D time-series denoising.

Architecture
------------
Encoder:
  - Stack of 1-D strided convolution blocks (Conv1d → BatchNorm → GELU)
  - Each block halves the temporal resolution and doubles the feature channels
  - Final encoder output: (B, C_latent, T/2^n_levels)

Bottleneck:
  - Multi-Head Self-Attention over the compressed time axis
  - Allows global context across the entire (compressed) window

Decoder:
  - Mirror of the encoder using ConvTranspose1d blocks
  - Skip connections from encoder to decoder (U-Net style)
  - Final 1×1 conv maps back to a single output channel

This design gives the model both local feature extraction (convolutions) and
global dependency modelling (attention) at a compact representation level,
which is well-suited for separating a sparse sinusoidal dark-matter signal
from broadband noise.
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# --------------------------------------------------------------------------- #
#  Building blocks                                                              #
# --------------------------------------------------------------------------- #

class ConvBlock(nn.Module):
    """Conv1d → BatchNorm1d → GELU, with optional downsampling (stride=2)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7, stride: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
        self.norm = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ConvTransposeBlock(nn.Module):
    """ConvTranspose1d → BatchNorm1d → GELU for decoder upsampling."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7, stride: int = 2) -> None:
        super().__init__()
        padding = kernel_size // 2
        output_padding = stride - 1
        self.conv = nn.ConvTranspose1d(
            in_ch, out_ch, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.norm = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResidualConvBlock(nn.Module):
    """Two ConvBlocks with a residual connection (no downsampling)."""

    def __init__(self, channels: int, kernel_size: int = 7) -> None:
        super().__init__()
        self.block1 = ConvBlock(channels, channels, kernel_size, stride=1)
        self.block2 = ConvBlock(channels, channels, kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block2(self.block1(x))


class BottleneckAttention(nn.Module):
    """
    Multi-head self-attention applied to the encoder's bottleneck.

    The input has shape (B, C, T_latent).  We treat T_latent as the sequence
    length and C as the embedding dimension.
    """

    def __init__(self, channels: int, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        assert channels % num_heads == 0, (
            f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        )
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T_latent)  →  treat as (B, T_latent, C) for attention
        x_t = rearrange(x, "b c t -> b t c")

        # Self-attention sub-layer with pre-norm + residual
        h = self.norm(x_t)
        h, _ = self.attn(h, h, h)
        x_t = x_t + h

        # Feed-forward sub-layer with pre-norm + residual
        x_t = x_t + self.ff(self.norm2(x_t))

        return rearrange(x_t, "b t c -> b c t")


# --------------------------------------------------------------------------- #
#  Full model                                                                  #
# --------------------------------------------------------------------------- #

class AttentionAutoencoder(nn.Module):
    """
    1-D Autoencoder with Multi-Head Self-Attention at the bottleneck.

    Parameters
    ----------
    in_channels : int
        Number of input channels (1 for TIDMAD).
    base_channels : int
        Feature channels after the first encoder block.  Doubled at each level.
    n_levels : int
        Number of encoder downsampling stages.  The spatial resolution is
        reduced by a factor of 2^n_levels.
    n_attn_heads : int
        Number of attention heads in the bottleneck.
    attn_dropout : float
        Dropout probability inside the attention module.
    kernel_size : int
        Convolution kernel size for encoder / decoder blocks.
    n_res_blocks : int
        Number of residual blocks at each encoder / decoder level.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        n_levels: int = 4,
        n_attn_heads: int = 8,
        attn_dropout: float = 0.1,
        kernel_size: int = 7,
        n_res_blocks: int = 2,
    ) -> None:
        super().__init__()

        self.n_levels = n_levels

        # ---- Stem: in_channels → base_channels ------------------------------ #
        self.stem = ConvBlock(in_channels, base_channels, kernel_size=kernel_size, stride=1)

        # ---- Encoder -------------------------------------------------------- #
        enc_channels: List[int] = []
        self.encoder_blocks = nn.ModuleList()
        self.encoder_downs = nn.ModuleList()

        ch = base_channels
        for i in range(n_levels):
            # Residual blocks at current resolution
            res = nn.Sequential(
                *[ResidualConvBlock(ch, kernel_size) for _ in range(n_res_blocks)]
            )
            self.encoder_blocks.append(res)
            enc_channels.append(ch)

            # Downsample + double channels
            out_ch = ch * 2
            self.encoder_downs.append(
                ConvBlock(ch, out_ch, kernel_size=kernel_size, stride=2)
            )
            ch = out_ch

        self.latent_channels = ch  # channels at the bottleneck

        # ---- Bottleneck ----------------------------------------------------- #
        self.bottleneck_conv = ResidualConvBlock(ch, kernel_size)
        self.bottleneck_attn = BottleneckAttention(
            channels=ch,
            num_heads=n_attn_heads,
            dropout=attn_dropout,
        )
        self.bottleneck_conv2 = ResidualConvBlock(ch, kernel_size)

        # ---- Decoder -------------------------------------------------------- #
        self.decoder_ups = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        for i in range(n_levels - 1, -1, -1):
            skip_ch = enc_channels[i]
            out_ch = skip_ch

            self.decoder_ups.append(
                ConvTransposeBlock(ch, out_ch, kernel_size=kernel_size, stride=2)
            )

            # After concatenation with skip: out_ch + skip_ch channels
            self.decoder_blocks.append(
                nn.Sequential(
                    ConvBlock(out_ch + skip_ch, out_ch, kernel_size=kernel_size, stride=1),
                    *[ResidualConvBlock(out_ch, kernel_size) for _ in range(n_res_blocks)],
                )
            )
            ch = out_ch

        # ---- Head ----------------------------------------------------------- #
        self.head = nn.Conv1d(ch, in_channels, kernel_size=1)

    # ---------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (B, 1, T)

        Returns
        -------
        Tensor of shape (B, 1, T)
        """
        # Stem
        h = self.stem(x)                         # (B, base_ch, T)

        # Encoder – store skip connections
        skips: List[torch.Tensor] = []
        for res, down in zip(self.encoder_blocks, self.encoder_downs):
            h = res(h)
            skips.append(h)
            h = down(h)

        # Bottleneck
        h = self.bottleneck_conv(h)
        h = self.bottleneck_attn(h)
        h = self.bottleneck_conv2(h)

        # Decoder – fuse skip connections
        for up, block, skip in zip(
            self.decoder_ups, self.decoder_blocks, reversed(skips)
        ):
            h = up(h)

            # Trim or pad to match skip connection size (handles edge cases)
            if h.shape[-1] != skip.shape[-1]:
                h = F.interpolate(h, size=skip.shape[-1], mode="linear", align_corners=False)

            h = torch.cat([h, skip], dim=1)
            h = block(h)

        return self.head(h)


# --------------------------------------------------------------------------- #
#  Convenience factory                                                         #
# --------------------------------------------------------------------------- #

def build_model(cfg: dict) -> AttentionAutoencoder:
    """Instantiate an AttentionAutoencoder from a config dictionary."""
    return AttentionAutoencoder(
        in_channels=cfg.get("in_channels", 1),
        base_channels=cfg.get("base_channels", 32),
        n_levels=cfg.get("n_levels", 4),
        n_attn_heads=cfg.get("n_attn_heads", 8),
        attn_dropout=cfg.get("attn_dropout", 0.1),
        kernel_size=cfg.get("kernel_size", 7),
        n_res_blocks=cfg.get("n_res_blocks", 2),
    )
