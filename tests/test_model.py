"""
Tests for the AttentionAutoencoder model.
"""

import math
import pytest
import torch

from tidmad_denoiser.model import AttentionAutoencoder, build_model


# --------------------------------------------------------------------------- #
#  Fixtures                                                                    #
# --------------------------------------------------------------------------- #

@pytest.fixture
def default_model() -> AttentionAutoencoder:
    return AttentionAutoencoder(
        in_channels=1,
        base_channels=8,    # small for fast tests
        n_levels=3,
        n_attn_heads=4,
        attn_dropout=0.0,
        kernel_size=7,
        n_res_blocks=1,
    )


# --------------------------------------------------------------------------- #
#  Shape tests                                                                 #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("window_size", [512, 1024, 4096])
def test_output_shape_matches_input(default_model, window_size):
    x = torch.randn(4, 1, window_size)
    y = default_model(x)
    assert y.shape == x.shape, (
        f"Expected output shape {x.shape}, got {y.shape}"
    )


def test_batch_size_one(default_model):
    x = torch.randn(1, 1, 1024)
    y = default_model(x)
    assert y.shape == (1, 1, 1024)


def test_larger_batch(default_model):
    x = torch.randn(16, 1, 1024)
    y = default_model(x)
    assert y.shape == (16, 1, 1024)


# --------------------------------------------------------------------------- #
#  Gradient flow                                                               #
# --------------------------------------------------------------------------- #

def test_gradients_flow(default_model):
    x = torch.randn(2, 1, 512, requires_grad=False)
    y_hat = default_model(x)
    loss = y_hat.mean()
    loss.backward()

    for name, param in default_model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"


# --------------------------------------------------------------------------- #
#  Latent dimension checks                                                     #
# --------------------------------------------------------------------------- #

def test_latent_channels():
    model = AttentionAutoencoder(base_channels=16, n_levels=3)
    # After 3 doublings: 16 * 2^3 = 128
    assert model.latent_channels == 16 * (2 ** 3)


# --------------------------------------------------------------------------- #
#  build_model factory                                                         #
# --------------------------------------------------------------------------- #

def test_build_model_defaults():
    model = build_model({})
    x = torch.randn(1, 1, 2048)
    y = model(x)
    assert y.shape == x.shape


def test_build_model_custom():
    cfg = {"base_channels": 16, "n_levels": 2, "n_attn_heads": 4}
    model = build_model(cfg)
    x = torch.randn(2, 1, 512)
    y = model(x)
    assert y.shape == x.shape


# --------------------------------------------------------------------------- #
#  Determinism                                                                 #
# --------------------------------------------------------------------------- #

def test_deterministic_inference(default_model):
    default_model.eval()
    x = torch.randn(2, 1, 512)
    with torch.no_grad():
        y1 = default_model(x)
        y2 = default_model(x)
    assert torch.allclose(y1, y2), "Model is non-deterministic at inference"
