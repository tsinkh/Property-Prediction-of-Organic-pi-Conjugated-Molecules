import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from rdkit import RDLogger

from featurization.rdkit_descriptors import compute_rdkit_descriptors
from featurization.fingerprints import compute_ecfp

RDLogger.DisableLog("rdApp.*") # type: ignore

DATA_PATH = os.path.join("data", "molecules.csv")
OUTPUT_DIR = "processed"
ECFP_BITS = 2048
ECFP_RADIUS = 2

def build_dataset():
    print(f"Loading molecules from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    smiles_list = df["smiles"].tolist()

    print("Computing RDKit descriptors...")
    desc_array, desc_names = compute_rdkit_descriptors(smiles_list)

    print("Computing ECFP fingerprints...")
    fp_array = compute_ecfp(smiles_list, radius=ECFP_RADIUS, n_bits=ECFP_BITS)

    print("Building feature matrix...")
    X = np.hstack([desc_array, fp_array])
    feature_names = desc_names + [f"ECFP{i}" for i in range(ECFP_BITS)]

    target_cols = [c for c in df.columns if c not in ["identifier", "smiles"]]
    y = df[target_cols]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
    pd.DataFrame(y).to_csv(os.path.join(OUTPUT_DIR, "y.csv"), index=False)
    pd.DataFrame(feature_names, columns=["feature"]).to_csv(
        os.path.join(OUTPUT_DIR, "features.csv"), index=False
    )

    print(f"Saved features to {OUTPUT_DIR}/X.npy")
    print(f"Saved targets to {OUTPUT_DIR}/y.csv")
    print(f"Saved feature names to {OUTPUT_DIR}/features.csv")


if __name__ == "__main__":
    build_dataset()