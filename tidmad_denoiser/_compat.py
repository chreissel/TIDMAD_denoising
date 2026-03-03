"""
Compatibility shim for PyTorch Lightning.

PyTorch Lightning 2.x ships under two package names:
  - ``lightning``          → ``import lightning.pytorch as pl``  (new canonical)
  - ``pytorch-lightning``  → ``import pytorch_lightning as pl``  (legacy)

Both are supported here via try/except so the codebase works regardless of
which package the user has installed.
"""

from __future__ import annotations

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import (
        EarlyStopping,
        ModelCheckpoint,
        RichProgressBar,
    )
    from lightning.pytorch.loggers import TensorBoardLogger
except ModuleNotFoundError:
    import pytorch_lightning as pl  # type: ignore[no-redef]
    from pytorch_lightning.callbacks import (  # type: ignore[no-redef]
        EarlyStopping,
        ModelCheckpoint,
        RichProgressBar,
    )
    from pytorch_lightning.loggers import TensorBoardLogger  # type: ignore[no-redef]

__all__ = [
    "pl",
    "ModelCheckpoint",
    "EarlyStopping",
    "RichProgressBar",
    "TensorBoardLogger",
]
