# IMTS-Mixer

Official implementation of the paper:  
**"IMTS-Mixer: Mixer Networks for Irregular Multivariate Time Series Forecasting"**, submitted to **SIGKDD 2025**.

<div align="center">
  <img src="./figures/imtsmixer.png" alt="IMTS-Mixer Architecture" width="600"/>
</div>

---

## 📊 Datasets

Our experiments follow the setup in [1], reusing the raw data and preprocessing code from the official [t-PatchGNN repository](https://github.com/usail-hkust/t-PatchGNN).

- **Included**:  
  - `Activity` and `USHCN` datasets (bundled with this repository)  
  - `PhysioNet`: Automatically downloaded and preprocessed on first run

- **Not included**:  
  - `MIMIC`: Licensed under the PhysioNet Credentialed Health Data License.  
    To obtain it, follow the instructions in the [t-PatchGNN repo](https://github.com/usail-hkust/t-PatchGNN).

---

## 🛠️ Installation & Requirements

We recommend setting up a fresh Conda environment (Python 3.11):

```bash
conda create -n imtsmixer python=3.11
conda activate imtsmixer
pip install -r requirements.txt
```

## 🚀 Reproducing Results
To reproduce the results on various datasets, use the following commands and add --seed  {0,1,2,3,4}:

<div align="center"> <img src="./figures/results.png" alt="Experimental Results" width="600"/> </div>

```bash
# PhysioNet
python run_imts_mixer.py --D 64 --D_dec 128 --mixer-blocks 2 --w_decay 0.01 --lr 0.01 --patience 20 --dataset physionet --history 24 -b 32

# MIMIC
python run_imts_mixer.py --D 64 --D_dec 128 --mixer-blocks 3 --w_decay 0.001 --lr 0.01 --patience 20 --dataset mimic --history 24 -b 32

# Human Activity
python run_imts_mixer.py --D 64 --D_dec 128 --mixer-blocks 2 --w_decay 0.0001 --lr 0.01 --patience 20 --dataset activity --history 3000 -b 32

# USHCN
python run_imts_mixer.py --D 64 --D_dec 128 --mixer-blocks 2 --w_decay 0.0001 --lr 0.01 --patience 20 --dataset ushcn --history 24 -b 32
```




## 📚 Reference
[1] Zhang, W., Yin, C., Liu, H., Zhou, X., & Xiong, H. (2024).
Irregular Multivariate Time Series Forecasting: A Transformable Patching Graph Neural Networks Approach.
In Forty-First International Conference on Machine Learning (ICML 2024).