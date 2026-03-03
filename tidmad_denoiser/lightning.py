"""
PyTorch Lightning Module for the TIDMAD denoising autoencoder.

Training aligned with the jessicafry/TIDMAD reference repository:
  - Loss        : SmoothL1Loss (Huber loss), matching the FCNet/AE baseline
  - Optimizer   : Adam(lr=5e-4), no weight decay
  - LR schedule : None (matches reference — no scheduler used)
  - Metrics     : benchmark.py SNR-based score (log₅.₂₇ of normalised SNR product)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from tidmad_denoiser._compat import pl

from tidmad_denoiser.model import AttentionAutoencoder, build_model
from tidmad_denoiser.metrics import compute_all_metrics


class DenoisingModule(pl.LightningModule):
    """
    LightningModule wrapping the AttentionAutoencoder denoiser.

    Parameters
    ----------
    model_cfg : dict
        Forwarded to ``build_model``.
    lr : float
        Learning rate for Adam optimizer (default 5e-4, matching reference).
    sample_rate : float
        Signal sample rate in Hz (used for metric computation).
    """

    def __init__(
        self,
        model_cfg: Optional[Dict[str, Any]] = None,
        lr: float = 5e-4,
        sample_rate: float = 1e7,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        model_cfg = model_cfg or {}
        self.model: AttentionAutoencoder = build_model(model_cfg)

        self.lr = lr
        self.sample_rate = sample_rate

    # ------------------------------------------------------------------ #
    #  Forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # ------------------------------------------------------------------ #
    #  Shared step                                                         #
    # ------------------------------------------------------------------ #

    def _shared_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str
    ) -> torch.Tensor:
        x, y = batch          # x: noisy input, y: clean target
        y_hat = self(x)       # denoised prediction

        # SmoothL1Loss (Huber loss) — matches reference repo's FCNet/AE loss
        loss = F.smooth_l1_loss(y_hat, y)

        # All metrics (logged but not back-propagated)
        metrics = compute_all_metrics(
            y_pred=y_hat,
            y_true=y,
            y_noisy=x,
            sample_rate=self.sample_rate,
            prefix=f"{stage}/",
        )
        metrics[f"{stage}/loss"] = loss

        self.log_dict(
            metrics,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=(stage != "train"),
            sync_dist=True,
        )
        return loss

    # ------------------------------------------------------------------ #
    #  Train / val / test                                                  #
    # ------------------------------------------------------------------ #

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        self._shared_step(batch, "test")

    # ------------------------------------------------------------------ #
    #  Optimiser                                                           #
    # ------------------------------------------------------------------ #

    def configure_optimizers(self):
        # Plain Adam, lr=5e-4 — matching reference repo (no weight decay, no scheduler)
        return torch.optim.Adam(self.parameters(), lr=self.lr)
