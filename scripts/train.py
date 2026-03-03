#!/usr/bin/env python
"""
Train the TIDMAD AttentionAutoencoder denoiser.

Usage
-----
    python scripts/train.py --data_dir data/ --config configs/default.yaml
    python scripts/train.py --data_dir data/ --config configs/default.yaml --resume checkpoints/last.ckpt
"""

from __future__ import annotations

import argparse
import os

import yaml
from tidmad_denoiser._compat import (
    pl,
    ModelCheckpoint,
    EarlyStopping,
    RichProgressBar,
    TensorBoardLogger,
)

from tidmad_denoiser.data import TIDMADDataModule
from tidmad_denoiser.lightning import DenoisingModule


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TIDMAD denoiser.")
    parser.add_argument("--data_dir", required=True, help="Directory with HDF5 files.")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config path.")
    parser.add_argument("--checkpoint_dir", default="checkpoints/", help="Where to save .ckpt files.")
    parser.add_argument("--log_dir", default="logs/", help="TensorBoard log directory.")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint path.")
    parser.add_argument("--devices", default="auto", help="Devices for PL Trainer.")
    parser.add_argument("--strategy", default="auto", help="PL strategy (ddp, fsdp, …).")
    parser.add_argument("--fast_dev_run", action="store_true", help="Quick sanity run (1 batch).")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ------------------------------------------------------------------ #
    # DataModule
    # ------------------------------------------------------------------ #
    dm = TIDMADDataModule(
        data_dir=args.data_dir,
        **cfg["data"],
    )

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    model = DenoisingModule(
        model_cfg=cfg["model"],
        lr=cfg["training"]["lr"],
        sample_rate=cfg["training"]["sample_rate"],
    )

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    cb_cfg = cfg["callbacks"]
    callbacks = [
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename="best-{epoch:03d}-{val/tidmad_score:.4f}",
            monitor=cb_cfg["checkpoint_monitor"],
            mode=cb_cfg["checkpoint_mode"],
            save_top_k=3,
            save_last=True,
            verbose=True,
        ),
        EarlyStopping(
            monitor=cb_cfg["early_stopping_monitor"],
            mode=cb_cfg["early_stopping_mode"],
            patience=cb_cfg["early_stopping_patience"],
            verbose=True,
        ),
        RichProgressBar(),
    ]

    # ------------------------------------------------------------------ #
    # Logger
    # ------------------------------------------------------------------ #
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=cfg["logging"]["project"],
    )

    # ------------------------------------------------------------------ #
    # Trainer
    # ------------------------------------------------------------------ #
    trainer = pl.Trainer(
        max_epochs=cfg["training"]["max_epochs"],
        devices=args.devices,
        strategy=args.strategy,
        gradient_clip_val=cfg["training"]["gradient_clip_val"],
        accumulate_grad_batches=cfg["training"]["accumulate_grad_batches"],
        log_every_n_steps=cfg["logging"]["log_every_n_steps"],
        callbacks=callbacks,
        logger=logger,
        fast_dev_run=args.fast_dev_run,
        precision="16-mixed",   # AMP for speed; change to "32" if needed
    )

    trainer.fit(model, datamodule=dm, ckpt_path=args.resume)
    print(f"\nTraining complete.  Best checkpoint: {callbacks[0].best_model_path}")


if __name__ == "__main__":
    main()
