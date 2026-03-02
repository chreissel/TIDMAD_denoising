"""
Tests for the TIDMAD denoising metrics.
"""

import math
import pytest
import torch

from tidmad_denoiser.metrics import (
    mse,
    snr_db,
    psnr_db,
    pearson_r,
    tidmad_denoising_score,
    compute_all_metrics,
)


# --------------------------------------------------------------------------- #
#  MSE                                                                        #
# --------------------------------------------------------------------------- #

def test_mse_identical_signals():
    x = torch.randn(4, 1, 1024)
    assert mse(x, x).item() == pytest.approx(0.0, abs=1e-7)


def test_mse_positive():
    y_pred = torch.randn(4, 1, 1024)
    y_true = torch.randn(4, 1, 1024)
    assert mse(y_pred, y_true).item() > 0


def test_mse_known_value():
    y_pred = torch.ones(1, 1, 100) * 2.0
    y_true = torch.ones(1, 1, 100) * 1.0
    assert mse(y_pred, y_true).item() == pytest.approx(1.0, rel=1e-5)


# --------------------------------------------------------------------------- #
#  SNR                                                                        #
# --------------------------------------------------------------------------- #

def test_snr_perfect_reconstruction():
    x = torch.randn(2, 1, 512)
    snr = snr_db(x, x)
    assert snr.item() > 80.0, "Perfect reconstruction should have very high SNR"


def test_snr_increases_with_quality():
    signal = torch.sin(torch.linspace(0, 4 * math.pi, 1024)).unsqueeze(0).unsqueeze(0)
    high_noise = signal + 0.5 * torch.randn_like(signal)
    low_noise = signal + 0.01 * torch.randn_like(signal)

    snr_high = snr_db(high_noise, signal)
    snr_low = snr_db(low_noise, signal)
    assert snr_low.item() > snr_high.item()


# --------------------------------------------------------------------------- #
#  PSNR                                                                       #
# --------------------------------------------------------------------------- #

def test_psnr_perfect_reconstruction():
    x = torch.randn(2, 1, 512)
    psnr = psnr_db(x, x)
    assert psnr.item() > 80.0


def test_psnr_greater_than_snr():
    """PSNR ≥ SNR because peak power ≥ mean power."""
    signal = torch.randn(4, 1, 1024)
    noise = signal + 0.1 * torch.randn_like(signal)
    assert psnr_db(noise, signal).item() >= snr_db(noise, signal).item() - 1e-4


# --------------------------------------------------------------------------- #
#  Pearson r                                                                  #
# --------------------------------------------------------------------------- #

def test_pearson_perfect():
    x = torch.randn(4, 1, 512)
    r = pearson_r(x, x)
    assert r.item() == pytest.approx(1.0, abs=1e-5)


def test_pearson_anti_correlated():
    x = torch.randn(4, 1, 512)
    r = pearson_r(-x, x)
    assert r.item() == pytest.approx(-1.0, abs=1e-5)


def test_pearson_range():
    y_pred = torch.randn(8, 1, 256)
    y_true = torch.randn(8, 1, 256)
    r = pearson_r(y_pred, y_true).item()
    assert -1.0 <= r <= 1.0


# --------------------------------------------------------------------------- #
#  TIDMAD denoising score (benchmark.py formula)                              #
# --------------------------------------------------------------------------- #

def test_tidmad_score_finite():
    """Score must always be a finite number."""
    torch.manual_seed(0)
    y_true = torch.randn(4, 1, 4096)
    y_noisy = y_true + torch.randn(4, 1, 4096)
    y_pred = y_true + 0.1 * torch.randn(4, 1, 4096)

    score = tidmad_denoising_score(y_pred, y_true, y_noisy)
    assert torch.isfinite(score), f"Score is not finite: {score}"


def test_tidmad_score_better_than_no_denoising():
    """A cleaner prediction should outscore the raw noisy input."""
    torch.manual_seed(42)
    # Clear sinusoidal signal so findPeak identifies a sharp spectral peak
    signal = torch.sin(torch.linspace(0, 20 * math.pi, 4096)).unsqueeze(0).unsqueeze(0)
    noise = torch.randn(1, 1, 4096)
    y_true = signal
    y_noisy = signal + noise

    y_pred_good = signal + 0.01 * noise   # mostly denoised
    y_pred_bad = y_noisy.clone()          # no denoising

    score_good = tidmad_denoising_score(y_pred_good, y_true, y_noisy)
    score_bad = tidmad_denoising_score(y_pred_bad, y_true, y_noisy)

    assert score_good.item() > score_bad.item(), (
        f"Good prediction score ({score_good:.4f}) should exceed "
        f"no-denoising score ({score_bad:.4f})"
    )


def test_tidmad_score_perfect_better_than_noisy():
    """Perfect reconstruction (y_pred == y_true) should score above noisy input."""
    torch.manual_seed(7)
    signal = torch.sin(torch.linspace(0, 30 * math.pi, 4096)).unsqueeze(0).unsqueeze(0)
    noise = torch.randn(1, 1, 4096)
    y_true = signal
    y_noisy = signal + noise

    score_perfect = tidmad_denoising_score(y_true.clone(), y_true, y_noisy)
    score_noisy = tidmad_denoising_score(y_noisy.clone(), y_true, y_noisy)

    assert score_perfect.item() > score_noisy.item()


# --------------------------------------------------------------------------- #
#  compute_all_metrics                                                        #
# --------------------------------------------------------------------------- #

def test_compute_all_metrics_keys():
    y_pred = torch.randn(4, 1, 1024)
    y_true = torch.randn(4, 1, 1024)
    y_noisy = torch.randn(4, 1, 1024)

    metrics = compute_all_metrics(y_pred, y_true, y_noisy, prefix="test/")

    expected_keys = {
        "test/mse", "test/snr_db", "test/psnr_db",
        "test/pearson_r", "test/tidmad_score",
    }
    assert set(metrics.keys()) == expected_keys


def test_compute_all_metrics_finite():
    torch.manual_seed(0)
    y_pred = torch.randn(4, 1, 1024)
    y_true = torch.randn(4, 1, 1024)
    y_noisy = torch.randn(4, 1, 1024)

    metrics = compute_all_metrics(y_pred, y_true, y_noisy)
    for name, val in metrics.items():
        assert torch.isfinite(val), f"Metric '{name}' is not finite: {val}"
