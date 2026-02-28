"""
Metrics for evaluating the TIDMAD denoising quality.

Implemented metrics
-------------------
1. MSE          : Mean Squared Error (reconstruction loss)
2. SNR          : Signal-to-Noise Ratio (dB)
3. PSNR         : Peak Signal-to-Noise Ratio (dB)
4. Pearson r    : Linear correlation between output and ground truth
5. TIDMADScore  : Official denoising score based on PSD improvement
                  (approximation — mirrors the spirit of benchmark.py)

All functions operate on 1-D or 2-D torch Tensors:
  - 1-D: a single time series window
  - 2-D: (batch, time) – metrics are averaged over the batch dimension
"""

from __future__ import annotations

import math
from typing import Union

import torch
import torch.nn.functional as F


Tensor = torch.Tensor


# --------------------------------------------------------------------------- #
#  Utility                                                                     #
# --------------------------------------------------------------------------- #

def _flatten(y_pred: Tensor, y_true: Tensor) -> tuple[Tensor, Tensor]:
    """Ensure both tensors are 2-D (B, T), squeezing channel dim if present."""
    if y_pred.dim() == 3:
        y_pred = y_pred.squeeze(1)
    if y_true.dim() == 3:
        y_true = y_true.squeeze(1)
    return y_pred, y_true


# --------------------------------------------------------------------------- #
#  Standard metrics                                                            #
# --------------------------------------------------------------------------- #

def mse(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """Mean Squared Error, averaged over all elements."""
    y_pred, y_true = _flatten(y_pred, y_true)
    return F.mse_loss(y_pred, y_true)


def snr_db(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """
    Signal-to-Noise Ratio in dB.

    SNR = 10 * log10( E[y_true^2] / E[(y_pred - y_true)^2] )

    Averaged over batch dimension.
    """
    y_pred, y_true = _flatten(y_pred, y_true)
    signal_power = y_true.pow(2).mean(dim=-1)
    noise_power = (y_pred - y_true).pow(2).mean(dim=-1).clamp(min=1e-12)
    return 10.0 * torch.log10(signal_power / noise_power).mean()


def psnr_db(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """
    Peak Signal-to-Noise Ratio in dB.

    PSNR = 10 * log10( max(y_true)^2 / MSE )

    Averaged over batch dimension.
    """
    y_pred, y_true = _flatten(y_pred, y_true)
    peak = y_true.abs().amax(dim=-1).pow(2)
    mse_val = (y_pred - y_true).pow(2).mean(dim=-1).clamp(min=1e-12)
    return 10.0 * torch.log10(peak / mse_val).mean()


def pearson_r(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """
    Pearson correlation coefficient, averaged over batch dimension.
    """
    y_pred, y_true = _flatten(y_pred, y_true)

    y_pred_z = y_pred - y_pred.mean(dim=-1, keepdim=True)
    y_true_z = y_true - y_true.mean(dim=-1, keepdim=True)

    cov = (y_pred_z * y_true_z).sum(dim=-1)
    std_pred = y_pred_z.pow(2).sum(dim=-1).sqrt().clamp(min=1e-8)
    std_true = y_true_z.pow(2).sum(dim=-1).sqrt().clamp(min=1e-8)

    return (cov / (std_pred * std_true)).mean()


# --------------------------------------------------------------------------- #
#  TIDMAD-inspired denoising score                                             #
# --------------------------------------------------------------------------- #

def _power_spectral_density(x: Tensor, sample_rate: float = 1e7) -> tuple[Tensor, Tensor]:
    """
    Compute the one-sided PSD using Welch's method (simple FFT version).

    Returns
    -------
    freqs : (N//2,)
    psd   : (B, N//2)  or (N//2,) if x is 1-D
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    N = x.shape[-1]
    # Apply Hann window to reduce spectral leakage
    window = torch.hann_window(N, device=x.device, dtype=x.dtype)
    x_w = x * window

    fft = torch.fft.rfft(x_w, dim=-1)
    psd = (fft.abs() ** 2) / (sample_rate * N)
    # One-sided: double all bins except DC and Nyquist
    psd[..., 1:-1] *= 2.0

    freqs = torch.fft.rfftfreq(N, d=1.0 / sample_rate, device=x.device)
    return freqs, psd


def tidmad_denoising_score(
    y_pred: Tensor,
    y_true: Tensor,
    y_noisy: Tensor,
    sample_rate: float = 1e7,
    f_min: float = 0.0,
    f_max: float = 2e6,
) -> Tensor:
    """
    TIDMAD-inspired denoising score.

    Measures how much the model improves the PSD residual compared to the
    raw (un-denoised) signal, within a frequency band [f_min, f_max].

    Score = mean_freq( PSD(y_noisy - y_true) / PSD(y_pred - y_true) )
            (higher is better; 1.0 means no improvement)

    Parameters
    ----------
    y_pred   : denoised model output (B, 1, T) or (B, T)
    y_true   : injected ground-truth signal (B, 1, T) or (B, T)
    y_noisy  : raw noisy input signal (B, 1, T) or (B, T)
    sample_rate : Hz (TIDMAD default 10 MHz)
    f_min, f_max : frequency band for score evaluation (Hz)
    """
    y_pred, y_true = _flatten(y_pred, y_true)
    y_noisy, _ = _flatten(y_noisy, y_true)

    residual_pred = y_pred - y_true
    residual_noisy = y_noisy - y_true

    freqs, psd_pred = _power_spectral_density(residual_pred, sample_rate)
    _, psd_noisy = _power_spectral_density(residual_noisy, sample_rate)

    # Select frequency band
    mask = (freqs >= f_min) & (freqs <= f_max)
    psd_pred_band = psd_pred[..., mask].clamp(min=1e-30)
    psd_noisy_band = psd_noisy[..., mask].clamp(min=1e-30)

    ratio = psd_noisy_band / psd_pred_band          # > 1 means improvement
    score = ratio.mean()                             # average over freq & batch
    return score


# --------------------------------------------------------------------------- #
#  Convenience wrapper (returns a dict for Lightning logging)                  #
# --------------------------------------------------------------------------- #

def compute_all_metrics(
    y_pred: Tensor,
    y_true: Tensor,
    y_noisy: Tensor,
    sample_rate: float = 1e7,
    prefix: str = "",
) -> dict[str, Tensor]:
    """
    Compute all metrics and return them in a dict suitable for
    ``self.log_dict`` in a LightningModule.
    """
    p = prefix
    return {
        f"{p}mse": mse(y_pred, y_true),
        f"{p}snr_db": snr_db(y_pred, y_true),
        f"{p}psnr_db": psnr_db(y_pred, y_true),
        f"{p}pearson_r": pearson_r(y_pred, y_true),
        f"{p}tidmad_score": tidmad_denoising_score(y_pred, y_true, y_noisy, sample_rate),
    }
