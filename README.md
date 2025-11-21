# IMTS-Mixer

[![Paper](https://img.shields.io/badge/Paper-AAAI%202026-blue)](https://github.com/your-username/IMTS-Mixer)
[![Python](https://img.shields.io/badge/Python-3.11-green)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

Official implementation of the paper:  
**Mixing It Up: Exploring Mixer Networks for Irregular Multivariate Time Series Forecasting**, 
submitted to **PAKDD 2026**.

<div align="center">
  <img src="figures/imtsmixer.png" alt="IMTS-Mixer Architecture" width="600"/>
</div>

## 🔍 Overview

IMTS-Mixer is a novel approach for forecasting irregular multivariate time series using Mixer Networks. This repository contains the complete implementation and experimental setup to reproduce the results presented in our paper.

**Key Features:**
- Efficient handling of irregular time series data
- State-of-the-art performance on multiple benchmarks
- Easy-to-use implementation with comprehensive examples
- Support for multiple datasets including PhysioNet, MIMIC, Human Activity, and USHCN

## 📁 Project Structure

```
IMTS-Mixer/
├── Ablation studies/           # Ablations as discussed in the paper
├── data/                       # Dataset storage (Physionet, MIMIC, Activity, USHCN)
├── figures/                    # Architecture and result figures
├── IMTS-Mixer/
│   ├── model/                  # Model implementations
│   ├── run_imts_mixer_pmau.py  # Main training script
│   └── run_imts_mixer_physiomeode.py  # Physiome-ODE experiments
├── lib/                        # Utility functions (data loading, preprocessing, etc.)
├── Physiome-ODE/
│   ├── final/                  # Store the Physiome-ODE datasets here
│   └── saved_models/           # During training on Physiome-ODE checkpoints will be saved here
├── README.md                   # This file
└── requirements.txt            # Dependencies
```

## 📊 Datasets

Our experiments follow the setup established in [1], reusing the raw data and preprocessing code from the official [t-PatchGNN repository](https://github.com/usail-hkust/t-PatchGNN).

### Available Datasets

#### ✅ Included with Repository
- **Activity**: Human activity recognition dataset
- **USHCN**: United States Historical Climatology Network dataset

#### 🔄 Auto-downloaded
- **PhysioNet**: Automatically downloaded and preprocessed on first run

#### 📋 Requires Manual Setup
- **MIMIC**: Licensed under the PhysioNet Credentialed Health Data License
  - **Access**: [MIMIC-III preprocessed version](https://physionet.org/content/mimic-iii-ext-tpatchgnn/1.0.0/) (requires PhysioNet credentials)
  - **Requirements**: Credentialed PhysioNet user account

## 🛠️ Installation

### Prerequisites
- Python 3.11
- Conda (recommended)

### Setup Environment

```bash
# Create and activate conda environment
conda create -n imtsmixer python=3.11
conda activate imtsmixer

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import torch; print('PyTorch version:', torch.__version__)"
```

## 🚀 Quick Start

### Basic Usage

```bash
# Run on PhysioNet dataset with default parameters
python run_imts_mixer_pmau.py --dataset physionet
```

### Reproducing Paper Results

To reproduce the exact results from our paper, run the following commands with different random seeds (0, 1, 2, 3, 4):

<div align="center">
  <img src="figures/results.png" alt="Experimental Results" width="600"/>
</div>

```bash
cd IMTSMixer
```


#### PhysioNet Dataset
```bash
for seed in {0..4}; do
  python run_imts_mixer_pmau.py \
    --D 64 \
    --D_dec 128 \
    --mixer-blocks 2 \
    --w_decay 0.01 \
    --lr 0.01 \
    --patience 20 \
    --dataset physionet \
    --history 24 \
    -b 32 \
    --seed $seed
done
```

#### MIMIC Dataset
```bash
for seed in {0..4}; do
python run_imts_mixer_pmau.py \
  --D 64 \
  --D_dec 128 \
  --mixer-blocks 3 \
  --w_decay 0.001 \
  --lr 0.01 \
  --patience 20 \
  --dataset mimic \
  --history 24 \
  -b 32 \
  --seed $seed
done
```

#### Human Activity Dataset
```bash
for seed in {0..4}; do
python run_imts_mixer_pmau.py \
  --D 64 \
  --D_dec 128 \
  --mixer-blocks 2 \
  --w_decay 0.0001 \
  --lr 0.01 \
  --patience 20 \
  --dataset activity \
  --history 3000 \
  -b 32 \
  --seed $seed
done
```

#### USHCN Dataset
```bash
for seed in {0..4}; do
python run_imts_mixer_pmau.py \
  --D 64 \
  --D_dec 128 \
  --mixer-blocks 2 \
  --w_decay 0.0001 \
  --lr 0.01 \
  --patience 20 \
  --dataset ushcn \
  --history 24 \
  -b 32 \
  --seed $seed
done
```

## 🧪 Physiome-ODE Experiments

For experiments with Physiome-ODE datasets [2], additional setup is required:

### Setup
1. Download the Physiome-ODE datasets from [Zenodo](https://zenodo.org/records/11492058)
2. Extract and place in the appropriate directory
3. For detailed recreation instructions, see the [Physiome-ODE repository](https://github.com/kloetergensc/Physiome-ODE)

### Running Experiments
```bash
cd IMTSMixer
```
```bash
# Example: Borghans-Dupont-Goldbeter model
python run_imts_mixer_physiomeode.py \
  --dataset borghans_dupont_goldbeter_1997a \
  --fold 0
```

## 🔬 Ablation Studies

To conduct ablation studies as discussed in the paper, you can modify the `--model` parameter in the main training script. The following model variants are available:

### Available Models

| Model | Description | Parameter Value |
|-------|-------------|-----------------|
| **IMTS-Mixer** | Full proposed model (default) | `imtsmixer` or any other value |
| **MLP Baseline** | MLP- temporal projection | `mlp` |
| **Attention Baseline** | Replace ISCAM with MHA | `attn` |
| **TTCN Baseline** | Replace ISCAM with TTCN | `ttcn` |

### Running Ablation Studies

```bash
# Run with MLP baseline
python run_imts_mixer_pmau.py \
  --D 64 \
  --D_dec 256 \
  --mixer-blocks 1 \
  --w_decay 0.0001 \
  --lr 0.01 \
  --patience 20 \
  --dataset ushcn \
  --history 24 \
  -b 32 \
  --seed 0 \
  --model ttcn
```


## 📖 References

[1] Zhang, W., Yin, C., Liu, H., Zhou, X., & Xiong, H. (2024). Irregular Multivariate Time Series Forecasting: A Transformable Patching Graph Neural Networks Approach. In *Proceedings of the 41st International Conference on Machine Learning (ICML 2024)*.

[2] Klötergens, C., Yalavarthi, V. K., Scholz, R., Stubbemann, M., Born, S., & Schmidt-Thieme, L. (2024). Physiome-ODE: A Benchmark for Irregularly Sampled Multivariate Time-Series Forecasting Based on Biological ODEs. In *The Thirteenth International Conference on Learning Representations*.
