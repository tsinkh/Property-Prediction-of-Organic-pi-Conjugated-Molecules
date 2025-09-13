from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
import numpy as np


def compute_rdkit_descriptors(smiles_list):
    """
    Compute a set of RDKit descriptors for a list of SMILES.
    
    Args:
        smiles_list (list of str): SMILES strings
    
    Returns:
        np.ndarray: shape (n_samples, n_descriptors)
        list: descriptor names
    """
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]

    descriptor_names = [d[0] for d in Descriptors.descList]
    descriptors = []
    for mol in mols:
        if mol is None:
            descriptors.append([np.nan] * len(descriptor_names))
        else:
            descriptors.append([func(mol) for _, func in Descriptors.descList])
    return np.array(descriptors), descriptor_names


if __name__ == "__main__":
    smiles = ["C1=CC=CC=C1", "CCO", "C"] # test run
    X, names = compute_rdkit_descriptors(smiles)
    df = pd.DataFrame(X, columns=names)
    print(df.head())
