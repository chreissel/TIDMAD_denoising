"""
Tests for the TIDMAD Dataset and DataModule.

These tests use synthetic (in-memory) HDF5 files to avoid requiring
an actual data download.
"""

from __future__ import annotations

import os
import tempfile

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from tidmad_denoiser.data import TIDMADWindowDataset, TIDMADDataModule


# --------------------------------------------------------------------------- #
#  Helpers: create synthetic HDF5 files                                       #
# --------------------------------------------------------------------------- #

N_SAMPLES = 20_000   # small enough to be fast in CI


def _make_h5(path: str, n_samples: int = N_SAMPLES, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    ch1 = rng.standard_normal(n_samples).astype(np.float32)
    ch2 = rng.standard_normal(n_samples).astype(np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("channel0001", data=ch1)
        f.create_dataset("channel0002", data=ch2)


@pytest.fixture
def h5_file(tmp_path):
    path = str(tmp_path / "abra_training_0000.h5")
    _make_h5(path)
    return path


@pytest.fixture
def data_dir_with_files(tmp_path):
    """Creates a temp dir with 2 training + 2 validation HDF5 files."""
    for i in range(2):
        _make_h5(str(tmp_path / f"abra_training_{i:04d}.h5"), seed=i)
        _make_h5(str(tmp_path / f"abra_validation_{i:04d}.h5"), seed=100 + i)
    return str(tmp_path)


# --------------------------------------------------------------------------- #
#  TIDMADWindowDataset                                                        #
# --------------------------------------------------------------------------- #

class TestTIDMADWindowDataset:
    def test_length_non_overlapping(self, h5_file):
        ds = TIDMADWindowDataset(h5_file, window_size=1000, stride=1000)
        expected = N_SAMPLES // 1000
        assert len(ds) == expected

    def test_length_overlapping(self, h5_file):
        ds = TIDMADWindowDataset(h5_file, window_size=1000, stride=500)
        # floor((20000 - 1000) / 500) + 1 = 39
        expected = (N_SAMPLES - 1000) // 500 + 1
        assert len(ds) == expected

    def test_max_windows_cap(self, h5_file):
        ds = TIDMADWindowDataset(h5_file, window_size=100, max_windows=5)
        assert len(ds) == 5

    def test_item_shapes(self, h5_file):
        window_size = 512
        ds = TIDMADWindowDataset(h5_file, window_size=window_size)
        x, y = ds[0]
        assert x.shape == (1, window_size), f"x shape: {x.shape}"
        assert y.shape == (1, window_size), f"y shape: {y.shape}"

    def test_item_dtype(self, h5_file):
        ds = TIDMADWindowDataset(h5_file, window_size=512)
        x, y = ds[0]
        assert x.dtype == torch.float32
        assert y.dtype == torch.float32

    def test_normalisation_zero_mean(self, h5_file):
        ds = TIDMADWindowDataset(h5_file, window_size=512)
        x, y = ds[0]
        assert x.mean().abs().item() < 1e-5, "x should be zero-mean after normalisation"
        assert y.mean().abs().item() < 1e-5, "y should be zero-mean after normalisation"

    def test_normalisation_unit_std(self, h5_file):
        ds = TIDMADWindowDataset(h5_file, window_size=512)
        x, y = ds[0]
        assert abs(x.std().item() - 1.0) < 0.1
        assert abs(y.std().item() - 1.0) < 0.1

    def test_different_windows_not_identical(self, h5_file):
        ds = TIDMADWindowDataset(h5_file, window_size=512)
        x0, _ = ds[0]
        x1, _ = ds[1]
        assert not torch.allclose(x0, x1), "Adjacent windows should differ"

    def test_index_out_of_range(self, h5_file):
        ds = TIDMADWindowDataset(h5_file, window_size=1000)
        with pytest.raises(IndexError):
            _ = ds[len(ds)]


# --------------------------------------------------------------------------- #
#  TIDMADDataModule                                                           #
# --------------------------------------------------------------------------- #

class TestTIDMADDataModule:
    def test_setup_creates_datasets(self, data_dir_with_files):
        dm = TIDMADDataModule(
            data_dir=data_dir_with_files,
            window_size=512,
            max_windows_per_file=10,
            batch_size=4,
            num_workers=0,
        )
        dm.setup("fit")
        assert hasattr(dm, "train_dataset")
        assert hasattr(dm, "val_dataset")
        assert len(dm.train_dataset) > 0
        assert len(dm.val_dataset) > 0

    def test_train_dataloader_batch_shape(self, data_dir_with_files):
        dm = TIDMADDataModule(
            data_dir=data_dir_with_files,
            window_size=512,
            max_windows_per_file=10,
            batch_size=4,
            num_workers=0,
        )
        dm.setup("fit")
        loader = dm.train_dataloader()
        x, y = next(iter(loader))
        assert x.shape == (4, 1, 512)
        assert y.shape == (4, 1, 512)

    def test_val_dataloader_returns_data(self, data_dir_with_files):
        dm = TIDMADDataModule(
            data_dir=data_dir_with_files,
            window_size=256,
            max_windows_per_file=8,
            batch_size=4,
            num_workers=0,
        )
        dm.setup("fit")
        loader = dm.val_dataloader()
        batch = next(iter(loader))
        assert len(batch) == 2  # (x, y)

    def test_test_dataloader_returns_data(self, data_dir_with_files):
        dm = TIDMADDataModule(
            data_dir=data_dir_with_files,
            window_size=256,
            max_windows_per_file=8,
            batch_size=4,
            num_workers=0,
        )
        dm.setup("test")
        loader = dm.test_dataloader()
        x, y = next(iter(loader))
        assert x.shape[-1] == 256

    def test_missing_files_raises(self, tmp_path):
        dm = TIDMADDataModule(data_dir=str(tmp_path), window_size=512, num_workers=0)
        with pytest.raises(FileNotFoundError):
            dm.setup("fit")
