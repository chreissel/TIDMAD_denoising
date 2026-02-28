"""TIDMAD Denoiser – Autoencoder with Attention for dark-matter time-series denoising."""

from tidmad_denoiser.model import AttentionAutoencoder
from tidmad_denoiser.lightning import DenoisingModule
from tidmad_denoiser.data import TIDMADDataModule

__all__ = ["AttentionAutoencoder", "DenoisingModule", "TIDMADDataModule"]
