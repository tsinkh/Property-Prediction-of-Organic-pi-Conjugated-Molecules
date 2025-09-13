# Property Prediction of Organic π-Conjugated Molecules

Reproduction of *Bhat et al. (2022)* on predicting **electronic, redox, and optical properties** of organic π-conjugated molecules, with additional implementation using **UniMol**.

## 📖 Project Overview

This project implements a hierarchy of machine learning models to predict molecular properties from SMILES strings.
We aim to reproduce the models described in *Bhat et al., Chem. Sci., 2022*, and extend them with UniMol-based methods.

Implemented model hierarchy:

1. **Classical ML** (Ridge Regression, SVR, Kernel Ridge Regression)
2. **Feedforward Neural Networks (FFN)**
3. **Graph Neural Networks (MPNN)**
4. **MPNN + Evidential Uncertainty**
5. **UniMol-based models** (extension beyond the original paper)

---

## 📂 Repository Structure

```
├── data/                  # Raw data (molecules.csv, descriptor info, etc.)├── processed/             # Preprocessed feature matrices (X.npy, y.csv, features.csv)
├── results/               # Model outputs (CV results, trained models)
├── src/
│   ├── featurization/     # Feature generation (RDKit descriptors, ECFP fingerprints)
│   └── models/            # Classical ML + deep learning models
├── .gitignore
├── LICENSE
├── README.md
└── requirments.txt
```

---

## ⚙️ Installation

We recommend creating a clean virtual environment first:

```bash
conda create -n molpred python=3.10 -y
conda activate molpred
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🧪 Data Preparation

Make sure your data is stored in `data/molecules.csv`.
Then build features with:

```bash
python src/featurization/build_dataset.py
```

This will generate:

* `processed/X.npy` – feature matrix (descriptors + ECFP)
* `processed/y.csv` – target properties
* `processed/features.csv` – feature names

---

## 📊 Running Models

### 1. Classical ML (with Optuna tuning)

Run Ridge regression on HOMO prediction:

```bash
python src/models/classical_ml_optuna.py --target homo --model ridge --trials 50
```

This will:

* Perform **5-fold CV** for each Optuna trial
* Save per-fold results in `results/ridge_homo.csv`
* Save final trained model on full dataset to `results/ridge_homo.pkl`

Change `--model` to `svr` or `krr` for other classical ML baselines.
Change `--target` to any property in `processed/y.csv` (e.g. `lumo`, `vie`, `aie`).

---

## 📈 Current Progress

* ✅ Data processed, features generated (RDKit + ECFP)
* ✅ Classical ML baselines implemented (Ridge, SVR, KRR)
* ✅ Optuna hyperparameter tuning integrated
* 🔄 Next: Implement FFN (2nd generation model)

---

## 📜 License

MIT License