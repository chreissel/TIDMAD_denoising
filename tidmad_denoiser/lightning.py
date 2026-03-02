"""
PyTorch Lightning Module for the TIDMAD denoising autoencoder.

Handles:
  - training / validation / test steps
  - loss computation (MSE + optional spectral regularisation)
  - metric logging (all TIDMAD metrics)
  - learning-rate scheduling (cosine with warm-up)
  - checkpoint saving (best validation TIDMAD score)
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from tidmad_denoiser.model import AttentionAutoencoder, build_model
from tidmad_denoiser.metrics import compute_all_metrics


# --------------------------------------------------------------------------- #
#  Loss functions                                                              #
# --------------------------------------------------------------------------- #

def spectral_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    L1 loss in the frequency domain (encourages spectral fidelity).
    Operates on the last dimension.
    """
    # Squeeze channel dim if present
    if y_pred.dim() == 3:
        y_pred = y_pred.squeeze(1)
        y_true = y_true.squeeze(1)

    fft_pred = torch.fft.rfft(y_pred.float(), dim=-1)
    fft_true = torch.fft.rfft(y_true.float(), dim=-1)
    return F.l1_loss(fft_pred.abs(), fft_true.abs())


# --------------------------------------------------------------------------- #
#  LightningModule                                                             #
# --------------------------------------------------------------------------- #

class DenoisingModule(pl.LightningModule):
    """
    LightningModule wrapping the AttentionAutoencoder denoiser.

    Parameters (passed via ``hparams`` dict)
    -----------------------------------------
    model_cfg : dict
        Forwarded to ``build_model``.
    lr : float
        Peak learning rate.
    weight_decay : float
    warmup_epochs : int
        Number of epochs for linear LR warm-up.
    spectral_weight : float
        Weight of the spectral loss term (0 = MSE only).
    sample_rate : float
        Signal sample rate in Hz (used for metric computation).
    """

    def __init__(
        self,
        model_cfg: Optional[Dict[str, Any]] = None,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5,
        spectral_weight: float = 0.1,
        sample_rate: float = 1e7,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        model_cfg = model_cfg or {}
        self.model: AttentionAutoencoder = build_model(model_cfg)

        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.spectral_weight = spectral_weight
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
        x, y = batch                          # x: noisy, y: clean
        y_hat = self(x)                        # denoised prediction

        # Primary loss
        loss_mse = F.mse_loss(y_hat, y)

        # Optional spectral regularisation
        loss_spec = spectral_loss(y_hat, y) if self.spectral_weight > 0 else 0.0
        loss = loss_mse + self.spectral_weight * loss_spec

        # All metrics (logged but not back-propagated)
        metrics = compute_all_metrics(
            y_pred=y_hat,
            y_true=y,
            y_noisy=x,
            sample_rate=self.sample_rate,
            prefix=f"{stage}/",
        )
        metrics[f"{stage}/loss"] = loss
        metrics[f"{stage}/loss_mse"] = loss_mse
        if self.spectral_weight > 0:
            metrics[f"{stage}/loss_spec"] = loss_spec

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
    #  Optimiser / scheduler                                               #
    # ------------------------------------------------------------------ #

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Cosine annealing with linear warm-up
        def lr_lambda(current_epoch: int) -> float:
            if current_epoch < self.warmup_epochs:
                return float(current_epoch + 1) / float(self.warmup_epochs)
            # Cosine decay from warm-up end to 0
            progress = (current_epoch - self.warmup_epochs) / max(
                1, self.trainer.max_epochs - self.warmup_epochs
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "name": "lr",
            },
        }
