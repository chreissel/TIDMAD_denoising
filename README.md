# TIDMAD Denoiser

A PyTorch Lightning implementation of an **Autoencoder with Attention** for denoising the [TIDMAD](https://github.com/jessicafry/TIDMAD) dark-matter time-series dataset.

## Architecture

```
Input (noisy SQUID signal)
        │
    Encoder (1-D convolutions, downsampling)
        │
   Latent space
        │
   Self-Attention (Multi-Head) over the compressed sequence
        │
    Decoder (transposed convolutions, upsampling)
        │
  Output (denoised signal)
```

The model takes windows of the noisy `channel0001` time-series and predicts the clean injected signal from `channel0002`.

## Installation

```bash
pip install -e ".[dev]"
```

## Quick start

### 1 – Download TIDMAD data

```bash
# fetch a small subset for experimentation
python scripts/download_data.py --output_dir data/ --train_files 2 --validation_files 2
```

### 2 – Train

```bash
python scripts/train.py \
    --data_dir data/ \
    --config configs/default.yaml
```

### 3 – Evaluate / Benchmark

```bash
python scripts/evaluate.py \
    --data_dir data/ \
    --checkpoint checkpoints/best.ckpt
```

### 4 – Run tests

```bash
pytest tests/ -v
```

## Project layout

```
tidmad_denoiser/
├── tidmad_denoiser/
│   ├── __init__.py
│   ├── data.py          # PyTorch Dataset & Lightning DataModule
│   ├── model.py         # Autoencoder + Attention architecture
│   ├── lightning.py     # LightningModule (training / validation / test steps)
│   └── metrics.py       # Denoising score + standard signal metrics
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_metrics.py
├── scripts/
│   ├── download_data.py
│   ├── train.py
│   └── evaluate.py
├── configs/
│   └── default.yaml
└── pyproject.toml
```

## Metrics

| Metric | Description |
|--------|-------------|
| **TIDMAD Denoising Score** | Official benchmark: measures signal-to-noise improvement over a reference frequency band |
| **MSE** | Mean squared error between denoised output and injected signal |
| **SNR** | Signal-to-noise ratio (dB) of the denoised output |
| **PSNR** | Peak signal-to-noise ratio (dB) |
| **Pearson r** | Correlation between denoised output and ground-truth signal |

## Citation

```bibtex
@article{fry2024tidmad,
  title   = {TIDMAD: Time Series Dataset for Discovering Dark Matter with AI Denoising},
  author  = {Fry, Jessica and others},
  journal = {arXiv:2406.04378},
  year    = {2024}
}
```
