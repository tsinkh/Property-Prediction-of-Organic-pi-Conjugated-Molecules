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
├── data/                   # Raw data (e.g., molecules.csv, descriptor info)
├── processed/              # Preprocessed feature matrices
│   ├── X.npy               # Feature matrix (RDKit + optional ECFP)
│   ├── y.csv               # Target properties
│   └── features.csv        # Feature names
├── results/
│   ├── tuning/             # Optuna tuning results (CSV per model/target)
│   ├── models/             # Trained models
│   └── best_params.json    # Best hyperparameters across models/targets
├── src/
│   ├── featurization/      # Feature generation (RDKit descriptors, ECFP fingerprints)
│   │   ├── build_dataset.py
│   │   └── rdkit_descriptors.py
│   ├── models/             # Training scripts (use best_params.json)
│   │   ├── classical_ml.py
│   │   └── ffn.py
│   └── tuning/             # Hyperparameter tuning (Optuna)
│       ├── classical_ml.py
│       └── ffn.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt        # Python dependencies
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

### 1. Classical ML (with Optuna tuning + training)

#### Step 1: Hyperparameter tuning

Run Ridge regression on HOMO prediction:

```bash
python src/tuning/classical_ml.py --target homo --model ridge --trials 50
```

This will:

* Perform **5-fold CV** for each Optuna trial
* Save per-fold and mean results in `results/tuning/ridge_homo.csv`
* Update `results/best_params.json` with the best parameters

Change `--model` to `svr` or `krr` for other classical ML baselines.
Change `--target` to any property in `processed/y.csv` (e.g. `lumo`, `vie`, `aie`).
Change `--trials` to set the number of Optuna hyperparameter search trials.

#### Step 2: Train final model

```bash
python src/models/classical_ml.py --target homo --model ridge
```

This will:

* Load the best hyperparameters from `results/best_params.json`
* Train on the **full dataset** using the best params
* Save the trained model to `results/models/ridge_homo.pkl`

Change `--model` to `svr` or `krr` for other classical ML baselines.
Change `--target` to any property in `processed/y.csv`.

---

### 2. Feed-Forward Neural Networks (FFN)

#### Step 1: Hyperparameter tuning

Run FFN tuning on HOMO prediction:

```bash
python src/tuning/ffn.py --target homo --trials 30
```

This will:

* Tune **hidden size**, **number of layers**, **dropout rate**, **learning rate**, and **batch size**
* Use **ReLU activation** and **Adam optimizer**
* Fixed training length of **50 epochs per fold**
* Perform **5-fold CV** on the training set
* Save tuning results to `results/tuning/ffn_homo.csv`
* Update `results/best_params.json` with the best configuration

Change `--target` to any property in `processed/y.csv`.
Change `--trials` to set the number of Optuna hyperparameter search trials.

#### Step 2: Train final model

```bash
python src/models/ffn.py --target homo --epochs 100
```

This will:

* Load the best hyperparameters for `homo` from `results/best_params.json`
* Train the FFN on the **full dataset** (`processed/X.npy`)
* Run for the specified number of epochs (`--epochs`, default 100)
* Save the trained model to `results/models/ffn_homo.pt`

Change `--target` to any property in `processed/y.csv`.
Change `--epochs` to set the number of training epochs for the final model.

---

## 📜 License

MIT License