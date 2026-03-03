"""
TIDMAD Dataset & PyTorch Lightning DataModule.

The TIDMAD HDF5 files contain:
  - channel0001 : noisy SQUID readout  (model INPUT)
  - channel0002 : injected fake signal  (model TARGET / ground truth)

Both channels are float32 time-series sampled at 10 MHz.
We slice them into non-overlapping windows of `window_size` samples
and normalise each window independently (zero-mean, unit-std).
"""

from __future__ import annotations

import glob
import os
from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pytorch_lightning as pl


# --------------------------------------------------------------------------- #
#  Low-level helpers                                                            #
# --------------------------------------------------------------------------- #

MAX_SAMPLES = 2_000_000_000  # recommended upper limit from TIDMAD README


def _load_channels(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (ch1, ch2) as float32 arrays, truncated to MAX_SAMPLES."""
    with h5py.File(path, "r") as f:
        ch1 = f["channel0001"][: MAX_SAMPLES].astype(np.float32)
        ch2 = f["channel0002"][: MAX_SAMPLES].astype(np.float32)
    # Ensure equal length
    n = min(len(ch1), len(ch2))
    return ch1[:n], ch2[:n]


# --------------------------------------------------------------------------- #
#  Dataset                                                                     #
# --------------------------------------------------------------------------- #

class TIDMADWindowDataset(Dataset):
    """
    Slices a single TIDMAD training/validation HDF5 file into fixed-length
    non-overlapping windows.

    Parameters
    ----------
    path : str
        Path to an ``abra_training_*.h5`` or ``abra_validation_*.h5`` file.
    window_size : int
        Number of time-steps per sample (must be a power of 2 for the model).
    stride : int | None
        Stride between consecutive windows.  Defaults to ``window_size``
        (non-overlapping).  Set smaller for data augmentation.
    max_windows : int | None
        Cap on the number of windows drawn from this file.  Useful to keep
        epoch length manageable when files are very large.
    """

    def __init__(
        self,
        path: str,
        window_size: int = 4096,
        stride: Optional[int] = None,
        max_windows: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.path = path
        self.window_size = window_size
        self.stride = stride if stride is not None else window_size

        ch1, ch2 = _load_channels(path)
        self.ch1 = torch.from_numpy(ch1)
        self.ch2 = torch.from_numpy(ch2)

        n = len(self.ch1)
        # Compute starting indices of all valid windows
        starts = list(range(0, n - window_size + 1, self.stride))
        if max_windows is not None:
            starts = starts[:max_windows]
        self.starts = starts

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.starts[idx]
        e = s + self.window_size

        x = self.ch1[s:e].clone()   # noisy input
        y = self.ch2[s:e].clone()   # clean target

        # Per-window standardisation using the noisy input's statistics.
        # Both channels are normalised by the same mean/std so the clean signal
        # retains its natural amplitude relative to the noise — the model learns
        # a standard denoiser rather than an amplitude-amplifying estimator.
        x_mean, x_std = x.mean(), x.std().clamp(min=1e-8)

        x = (x - x_mean) / x_std
        y = (y - x_mean) / x_std

        # Shape: (1, window_size) – single-channel 1-D signal
        return x.unsqueeze(0), y.unsqueeze(0)


# --------------------------------------------------------------------------- #
#  DataModule                                                                  #
# --------------------------------------------------------------------------- #

class TIDMADDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for TIDMAD.

    Expects the following directory layout (produced by ``download_data.py``):

        data_dir/
            abra_training_0000.h5
            abra_training_0001.h5
            ...
            abra_validation_0000.h5
            ...

    Parameters
    ----------
    data_dir : str
        Root directory containing all HDF5 files.
    window_size : int
        Window length fed to the model.
    stride : int | None
        Stride for training windows (None → non-overlapping).
    max_windows_per_file : int | None
        Limit windows per file to keep epoch length tractable.
    batch_size : int
    num_workers : int
    """

    def __init__(
        self,
        data_dir: str,
        window_size: int = 4096,
        stride: Optional[int] = None,
        max_windows_per_file: Optional[int] = 2000,
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.window_size = window_size
        self.stride = stride
        self.max_windows_per_file = max_windows_per_file
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.save_hyperparameters()

    # ------------------------------------------------------------------ #
    def _glob(self, pattern: str) -> List[str]:
        files = sorted(glob.glob(os.path.join(self.data_dir, pattern)))
        if not files:
            raise FileNotFoundError(
                f"No files matching '{pattern}' in '{self.data_dir}'. "
                "Run scripts/download_data.py first."
            )
        return files

    # ------------------------------------------------------------------ #
    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", None):
            train_files = self._glob("abra_training_*.h5")
            self.train_dataset = ConcatDataset(
                [
                    TIDMADWindowDataset(
                        f,
                        window_size=self.window_size,
                        stride=self.stride,
                        max_windows=self.max_windows_per_file,
                    )
                    for f in train_files
                ]
            )

        if stage in ("fit", "validate", None):
            val_files = self._glob("abra_validation_*.h5")
            self.val_dataset = ConcatDataset(
                [
                    TIDMADWindowDataset(
                        f,
                        window_size=self.window_size,
                        stride=None,  # non-overlapping for validation
                        max_windows=self.max_windows_per_file,
                    )
                    for f in val_files
                ]
            )

        if stage in ("test", None):
            # Re-use validation files for testing (benchmark scoring)
            val_files = self._glob("abra_validation_*.h5")
            self.test_dataset = ConcatDataset(
                [
                    TIDMADWindowDataset(
                        f,
                        window_size=self.window_size,
                        stride=None,
                        max_windows=self.max_windows_per_file,
                    )
                    for f in val_files
                ]
            )

    # ------------------------------------------------------------------ #
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
