#!/usr/bin/env python
"""
Run the test-set benchmark on a trained TIDMAD denoiser checkpoint.

Reports MSE, SNR, PSNR, Pearson r, and TIDMAD denoising score.

Usage
-----
    python scripts/evaluate.py --data_dir data/ --checkpoint checkpoints/best.ckpt
"""

from __future__ import annotations

import argparse

import pytorch_lightning as pl
import yaml

from tidmad_denoiser.data import TIDMADDataModule
from tidmad_denoiser.lightning import DenoisingModule


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TIDMAD denoiser.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to .ckpt file.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--devices", default="auto")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dm = TIDMADDataModule(
        data_dir=args.data_dir,
        **cfg["data"],
    )

    model = DenoisingModule.load_from_checkpoint(args.checkpoint)

    trainer = pl.Trainer(
        devices=args.devices,
        logger=False,
        enable_progress_bar=True,
    )

    results = trainer.test(model, datamodule=dm)

    print("\n─── Benchmark Results ───────────────────────────────")
    for k, v in results[0].items():
        print(f"  {k:<30s}  {v:.6f}")
    print("─────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
