"""
Metrics for evaluating the TIDMAD denoising quality.

Implemented metrics
-------------------
1. MSE          : Mean Squared Error (reconstruction loss)
2. SNR          : Signal-to-Noise Ratio (dB)
3. PSNR         : Peak Signal-to-Noise Ratio (dB)
4. Pearson r    : Linear correlation between output and ground truth
5. TIDMADScore  : Official benchmark score from jessicafry/TIDMAD benchmark.py
                  log₅.₂₇(mean(normalised_snr_sg × snr_squid))

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
#  TIDMAD benchmark denoising score (matching jessicafry/TIDMAD benchmark.py) #
# --------------------------------------------------------------------------- #

def _get_psd(x: Tensor, sample_rate: float) -> Tensor:
    """
    One-sided PSD for each row of x (B, T) → (B, T//2).

    Matches the reference:  dt/N * |rfft(ts)|^2  (no windowing, DC dropped).
    """
    N = x.shape[-1]
    dt = 1.0 / sample_rate
    fft = torch.fft.rfft(x, dim=-1)
    psd = (fft.abs() ** 2) * (dt / N)
    # Drop DC bin (index 0) — reference uses freq_array starting at the first non-DC bin
    return psd[:, 1:]


def _find_peak(pwr: Tensor) -> int:
    """
    Replicate benchmark.py findPeak:
      peakdiff = pwr[1:-1] - pwr[:-2] - pwr[2:]
      peakIndex = argmax(peakdiff) + 1
    Returns the integer index into `pwr`.
    """
    peakdiff = pwr[1:-1] - pwr[:-2] - pwr[2:]
    return int(peakdiff.argmax().item()) + 1


def _get_snr(pwr: Tensor, center_id: int) -> Tensor:
    """
    Replicate benchmark.py getSNR:
      sig_range   = 1
      noise_range = 50
      signal = sum(pwr[center - 1 : center + 2])
      noise  = sum(pwr[center - 50 : center + 51]) - signal
      SNR    = signal / noise
    """
    sig_range = 1
    noise_range = 50
    n = len(pwr)

    s_lo = max(0, center_id - sig_range)
    s_hi = min(n, center_id + sig_range + 1)
    n_lo = max(0, center_id - noise_range)
    n_hi = min(n, center_id + noise_range + 1)

    signal = pwr[s_lo:s_hi].sum()
    noise = (pwr[n_lo:n_hi].sum() - signal).clamp(min=1e-30)
    return signal / noise


def tidmad_denoising_score(
    y_pred: Tensor,
    y_true: Tensor,
    y_noisy: Tensor,
    sample_rate: float = 1e7,
) -> Tensor:
    """
    TIDMAD benchmark denoising score matching jessicafry/TIDMAD benchmark.py.

    Algorithm (per batch of windows)
    ---------------------------------
    1. Compute one-sided PSD of y_true  (signal generator, channel0002)
       and y_pred (denoised SQUID output).
    2. Per window: find peak frequency in y_true PSD (findPeak).
    3. Per window: compute SNR of y_true at that peak  → snr_sg.
    4. Per window: compute SNR of y_pred at same peak  → snr_squid.
    5. Normalise: snr_sg /= max(snr_sg)  (across the batch).
    6. score = mean(snr_sg_norm × snr_squid) + 1e-10
    7. Return log₅.₂₇(score).

    Parameters
    ----------
    y_pred   : denoised model output  (B, 1, T) or (B, T)
    y_true   : injected ground-truth  (B, 1, T) or (B, T)  [channel0002]
    y_noisy  : raw noisy input        (B, 1, T) or (B, T)  [unused in score,
               kept for API compatibility]
    sample_rate : Hz (TIDMAD default 10 MHz)
    """
    y_pred, y_true = _flatten(y_pred, y_true)
    B = y_true.shape[0]

    psd_sg    = _get_psd(y_true, sample_rate)   # (B, F)
    psd_squid = _get_psd(y_pred, sample_rate)   # (B, F)

    snr_sg_vals    = []
    snr_squid_vals = []

    for b in range(B):
        center = _find_peak(psd_sg[b])
        snr_sg_vals.append(_get_snr(psd_sg[b], center))
        snr_squid_vals.append(_get_snr(psd_squid[b], center))

    snr_sg    = torch.stack(snr_sg_vals)     # (B,)
    snr_squid = torch.stack(snr_squid_vals)  # (B,)

    # Normalise sg SNR by batch max — matching benchmark.py line:
    #   snr_sg = snr_sg / np.amax(snr_sg)
    snr_sg_norm = snr_sg / snr_sg.max().clamp(min=1e-30)

    # score = mean(snr_sg_norm * snr_squid) + 1e-10, then log base 5.27
    score = (snr_sg_norm * snr_squid).mean() + 1e-10
    return torch.log(score) / math.log(5.27)


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
