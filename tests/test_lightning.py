"""
Integration tests for the DenoisingModule (LightningModule).

Uses small synthetic data + a fast_dev_run Trainer to verify the full
training / validation / test loop without requiring real TIDMAD files.
"""

from __future__ import annotations

import os
import tempfile

import h5py
import numpy as np
import pytest
import torch
import pytorch_lightning as pl

from tidmad_denoiser.lightning import DenoisingModule
from tidmad_denoiser.data import TIDMADDataModule


N_SAMPLES = 8192   # small synthetic dataset


def _make_h5(path: str, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    ch1 = rng.standard_normal(N_SAMPLES).astype(np.float32)
    ch2 = rng.standard_normal(N_SAMPLES).astype(np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("channel0001", data=ch1)
        f.create_dataset("channel0002", data=ch2)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    for i in range(2):
        _make_h5(str(d / f"abra_training_{i:04d}.h5"), seed=i)
        _make_h5(str(d / f"abra_validation_{i:04d}.h5"), seed=100 + i)
    return str(d)


@pytest.fixture
def tiny_model():
    return DenoisingModule(
        model_cfg={
            "base_channels": 8,
            "n_levels": 2,
            "n_attn_heads": 2,
            "n_res_blocks": 1,
        },
        lr=1e-3,
    )


@pytest.fixture
def tiny_dm(data_dir):
    return TIDMADDataModule(
        data_dir=data_dir,
        window_size=256,
        max_windows_per_file=16,
        batch_size=4,
        num_workers=0,
    )


# --------------------------------------------------------------------------- #
#  Tests                                                                      #
# --------------------------------------------------------------------------- #

class TestDenoisingModule:
    def test_forward_pass(self, tiny_model):
        x = torch.randn(2, 1, 512)
        y = tiny_model(x)
        assert y.shape == x.shape

    def test_fast_dev_run_fit(self, tiny_model, tiny_dm, tmp_path):
        trainer = pl.Trainer(
            max_epochs=1,
            fast_dev_run=True,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        trainer.fit(tiny_model, datamodule=tiny_dm)

    def test_fast_dev_run_test(self, tiny_model, tiny_dm):
        trainer = pl.Trainer(
            fast_dev_run=True,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        results = trainer.test(tiny_model, datamodule=tiny_dm)
        assert len(results) == 1
        # All expected metric keys should be present
        for key in ["test/mse", "test/snr_db", "test/psnr_db", "test/pearson_r", "test/tidmad_score"]:
            assert key in results[0], f"Missing metric: {key}"

    def test_checkpoint_save_load(self, tiny_model, tiny_dm, tmp_path):
        ckpt_path = str(tmp_path / "test.ckpt")

        trainer = pl.Trainer(
            max_epochs=1,
            fast_dev_run=True,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        trainer.fit(tiny_model, datamodule=tiny_dm)
        trainer.save_checkpoint(ckpt_path)

        # Load and run inference
        loaded = DenoisingModule.load_from_checkpoint(ckpt_path)
        x = torch.randn(2, 1, 256)
        with torch.no_grad():
            y = loaded(x)
        assert y.shape == x.shape

    def test_hparams_saved(self, tiny_model):
        assert hasattr(tiny_model, "hparams")
        assert "lr" in tiny_model.hparams

    def test_loss_decreases_overfit(self, tmp_path):
        """Overfit on a tiny batch to confirm the optimiser works."""
        model = DenoisingModule(
            model_cfg={"base_channels": 8, "n_levels": 2, "n_attn_heads": 2, "n_res_blocks": 1},
            lr=1e-3,
        )

        x = torch.randn(4, 1, 256)
        y = torch.randn(4, 1, 256)

        opt = model.configure_optimizers()

        losses = []
        for _ in range(20):
            opt.zero_grad()
            y_hat = model(x)
            loss = torch.nn.functional.smooth_l1_loss(y_hat, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
        )
