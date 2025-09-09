"""
Generate molecular fingerprints (ECFP).
"""

from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np


def compute_ecfp(smiles_list, radius=2, n_bits=2048):
    """
    Compute Morgan fingerprints (ECFP).
    
    Args:
        smiles_list (list of str): SMILES strings
        radius (int): fingerprint radius (default 2 = ECFP4)
        n_bits (int): bit vector size
    
    Returns:
        np.ndarray: shape (n_samples, n_bits)
    """
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    fps = []
    for mol in mols:
        if mol is None:
            fps.append(np.zeros(n_bits))
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            arr = np.zeros((1,), dtype=int)
            fp_array = np.array([int(fp.GetBit(i)) for i in range(n_bits)])
            fps.append(fp_array)
    return np.array(fps)


if __name__ == "__main__":
    smiles = ["C1=CC=CC=C1", "CCO", "C"]
    X = compute_ecfp(smiles, radius=2, n_bits=16)
    print(X)
