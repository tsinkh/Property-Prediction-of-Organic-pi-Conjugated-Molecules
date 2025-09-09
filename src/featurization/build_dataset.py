"""
Build dataset: combine RDKit descriptors + ECFP fingerprints as features.
"""

import os
import pandas as pd
import numpy as np
from rdkit import RDLogger

from featurization.rdkit_descriptors import compute_rdkit_descriptors
from featurization.fingerprints import compute_ecfp

RDLogger.DisableLog("rdApp.*")  # silence RDKit warnings

# ----------------------------
# Config
# ----------------------------
DATA_PATH = os.path.join("data", "molecules.csv")
OUTPUT_DIR = "processed"
ECFP_BITS = 2048
ECFP_RADIUS = 2

# ----------------------------
# Main function
# ----------------------------
def build_dataset():
    print(f"Loading molecules from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # SMILES
    smiles_list = df["smiles"].tolist()

    # 1. RDKit descriptors
    print("Computing RDKit descriptors...")
    desc_array, desc_names = compute_rdkit_descriptors(smiles_list)

    # 2. ECFP fingerprints
    print("Computing ECFP fingerprints...")
    fp_array = compute_ecfp(smiles_list, radius=ECFP_RADIUS, n_bits=ECFP_BITS)

    # 3. Concatenate features
    print("Building feature matrix...")
    X = np.hstack([desc_array, fp_array])
    feature_names = desc_names + [f"ECFP{i}" for i in range(ECFP_BITS)]

    # 4. Targets (all target properties, except identifier/smiles)
    target_cols = [c for c in df.columns if c not in ["identifier", "smiles"]]
    y = df[target_cols]

    # Save
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
