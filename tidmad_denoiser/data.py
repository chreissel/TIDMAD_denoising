"""
TIDMAD Dataset & PyTorch Lightning DataModule.

Data loading follows the same pattern as the official TIDMAD repository:
  - The HDF5 file is opened once per worker and kept open.
  - Each __getitem__ call reads exactly one segment directly from disk
    using an integer slice — no full file is ever loaded into RAM.
  - Raw int8 values are cast to float32 on the fly.

HDF5 structure:
    timeseries/
        channel0001/timeseries   ← noisy SQUID readout  (input)
        channel0002/timeseries   ← injected signal      (target)
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
#  Constants                                                                   #
# --------------------------------------------------------------------------- #

MAX_SAMPLES  = 2_000_000_000          # recommended cap from TIDMAD README
CH1_PATH     = "timeseries/channel0001/timeseries"
CH2_PATH     = "timeseries/channel0002/timeseries"


# --------------------------------------------------------------------------- #
#  Dataset                                                                     #
# --------------------------------------------------------------------------- #

class TIDMADWindowDataset(Dataset):
    """
    Reads fixed-length segments lazily from a single TIDMAD HDF5 file.

    The HDF5 file is opened once (per worker) and individual windows are
    sliced on demand — identical to the official TIDMAD train.py approach.
    No full channel is ever loaded into RAM.

    Parameters
    ----------
    path : str
        Path to an abra_training_*.h5 or abra_validation_*.h5 file.
    window_size : int
        Number of samples per segment (segment_size in the original code).
    stride : int | None
        Step between segment start indices. Defaults to window_size
        (non-overlapping). Use window_size // 2 for 50 % overlap.
    max_windows : int | None
        Cap on total windows drawn from this file.
    """

    def __init__(
        self,
        path: str,
        window_size: int = 40_000,
        stride: Optional[int] = None,
        max_windows: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.path        = path
        self.window_size = window_size
        self.stride      = stride if stride is not None else window_size

        # Open file briefly just to read the length — then close again.
        with h5py.File(self.path, "r") as f:
            n1 = f[CH1_PATH].shape[0]
            n2 = f[CH2_PATH].shape[0]

        n = min(n1, n2, MAX_SAMPLES)
        starts = list(range(0, n - window_size + 1, self.stride))
        if max_windows is not None:
            starts = starts[:max_windows]
        self.starts = starts

        # File handle — opened lazily in _get_file() per worker
        self._file: Optional[h5py.File] = None

    # h5py File objects are not picklable, so we open them lazily
    # inside each DataLoader worker after forking.
    def _get_file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.starts[idx]
        e = s + self.window_size
        f = self._get_file()

        # Read one segment from disk — identical to the official TIDMAD code:
        #   np.array(ABRAfile['timeseries']['channel0001']['timeseries']) + 128
        # We cast to float32 directly instead of keeping uint8.
        x = f[CH1_PATH][s:e].astype(np.float32)   # noisy SQUID
        y = f[CH2_PATH][s:e].astype(np.float32)   # injected signal

        # Per-window normalisation (zero-mean, unit-std)
        x = (x - x.mean()) / (x.std() + 1e-8)
        y = (y - y.mean()) / (y.std() + 1e-8)

        return (
            torch.from_numpy(x).unsqueeze(0),   # (1, window_size)
            torch.from_numpy(y).unsqueeze(0),
        )

    def __del__(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass


# --------------------------------------------------------------------------- #
#  DataModule                                                                  #
# --------------------------------------------------------------------------- #

class TIDMADDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for TIDMAD.

    Parameters
    ----------
    data_dir : str
        Directory containing abra_training_*.h5 and abra_validation_*.h5 files.
    window_size : int
        Segment length fed to the model (default 40 000 = 4 ms at 10 MS/s,
        matching FC-Net in the original paper).
    stride : int | None
        Stride between windows. None -> non-overlapping.
    max_windows_per_file : int | None
        Cap per file to keep epoch length tractable.
    batch_size : int
    num_workers : int
    """

    def __init__(
        self,
        data_dir: str,
        window_size: int = 40_000,
        stride: Optional[int] = None,
        max_windows_per_file: Optional[int] = 2000,
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.data_dir             = data_dir
        self.window_size          = window_size
        self.stride               = stride
        self.max_windows_per_file = max_windows_per_file
        self.batch_size           = batch_size
        self.num_workers          = num_workers
        self.save_hyperparameters()

    def _glob(self, pattern: str) -> List[str]:
        files = sorted(glob.glob(os.path.join(self.data_dir, pattern)))
        if not files:
            raise FileNotFoundError(
                f"No files matching '{pattern}' in '{self.data_dir}'. "
                "Run scripts/download_data.py first."
            )
        return files

    def _make_dataset(self, files: List[str], stride: Optional[int]) -> ConcatDataset:
        return ConcatDataset([
            TIDMADWindowDataset(
                f,
                window_size=self.window_size,
                stride=stride,
                max_windows=self.max_windows_per_file,
            )
            for f in files
        ])

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", None):
            self.train_dataset = self._make_dataset(
                self._glob("abra_training_*.h5"), stride=self.stride
            )
        if stage in ("fit", "validate", None):
            self.val_dataset = self._make_dataset(
                self._glob("abra_validation_*.h5"), stride=None
            )
        if stage in ("test", None):
            self.test_dataset = self._make_dataset(
                self._glob("abra_validation_*.h5"), stride=None
            )

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
