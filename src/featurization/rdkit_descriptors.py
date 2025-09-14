from rdkit import Chem
import rdkit.Chem.Descriptors as desc
import rdkit.Chem.rdMolDescriptors as Rdesc
from rdkit.Chem import rdmolfiles
import pandas as pd
import numpy as np

# definitions of descriptor functions
mw = desc.ExactMolWt # type: ignore
fpdensity1 = desc.FpDensityMorgan1
fpdensity2 = desc.FpDensityMorgan2
fpdensity3 = desc.FpDensityMorgan3
heavy_atom_wt = desc.HeavyAtomMolWt
max_abs_pt_chg = desc.MaxAbsPartialCharge
max_ptl_chg = desc.MaxPartialCharge
min_abs_chg = desc.MinAbsPartialCharge
min_ptl_chg = desc.MinPartialCharge
avg_mol_wt = desc.MolWt # type: ignore
num_radical = desc.NumRadicalElectrons
num_valence = desc.NumValenceElectrons
calcauto2d = Rdesc.CalcAUTOCORR2D
chi0n = Rdesc.CalcChi0n
chi0v = Rdesc.CalcChi0v
chi1n = Rdesc.CalcChi1n
chi1v = Rdesc.CalcChi1v
chi2n = Rdesc.CalcChi2n
chi2v = Rdesc.CalcChi2v
chi3n = Rdesc.CalcChi3n
chi3v = Rdesc.CalcChi3v
chi4n = Rdesc.CalcChi4n
chi4v = Rdesc.CalcChi4v
crippen = Rdesc.CalcCrippenDescriptors
frac_C_sp3 = Rdesc.CalcFractionCSP3
hall_kier = Rdesc.CalcHallKierAlpha
kappa1 = Rdesc.CalcKappa1
kappa2 = Rdesc.CalcKappa2
kappa3 = Rdesc.CalcKappa3
labute_asa = Rdesc.CalcLabuteASA
alphaptic_carbo = Rdesc.CalcNumAliphaticCarbocycles
alaphatic_hetero = Rdesc.CalcNumAliphaticHeterocycles
rings_ala = Rdesc.CalcNumAliphaticRings
amide_bonds = Rdesc.CalcNumAmideBonds
aromatic_carbo = Rdesc.CalcNumAromaticCarbocycles
aromatic_hetero = Rdesc.CalcNumAromaticHeterocycles
rings_aromatic = Rdesc.CalcNumAromaticRings
stereo_centers = Rdesc.CalcNumAtomStereoCenters
bridge_head = Rdesc.CalcNumBridgeheadAtoms
hba = Rdesc.CalcNumHBA
hbd = Rdesc.CalcNumHBD
hetero_atoms = Rdesc.CalcNumHeteroatoms
heterocycles = Rdesc.CalcNumHeterocycles
lipinski_hba  = Rdesc.CalcNumLipinskiHBA
lipinski_hbd = Rdesc.CalcNumLipinskiHBD
num_rings = Rdesc.CalcNumRings
rotatable_bonds = Rdesc.CalcNumRotatableBonds
saturated_carbo = Rdesc.CalcNumSaturatedCarbocycles
sat_hetero = Rdesc.CalcNumSaturatedHeterocycles
sat_rings = Rdesc.CalcNumSaturatedRings
unspecified_stereo = Rdesc.CalcNumUnspecifiedAtomStereoCenters
tpsa = Rdesc.CalcTPSA
mqns = Rdesc.MQNs_
peoe = Rdesc.PEOE_VSA_
color = rdmolfiles.CanonicalRankAtoms
bond_type = Chem.rdchem.Bond.GetBondType
triple = Chem.rdchem.BondType.TRIPLE
double = Chem.rdchem.BondType.DOUBLE
single = Chem.rdchem.BondType.SINGLE
aromatic = Chem.rdchem.BondType.AROMATIC
to = Chem.MolToSmiles

def all_properties(mol):
    """Extract all molecular descriptors"""
    smile = to(mol)
    count = smile.count
    prop_dict = dict()
    
    # bond type counts
    bonds = [bond_type(bond) for bond in mol.GetBonds()]
    prop_dict["num_single"] = bonds.count(single)
    prop_dict["num_double"] = bonds.count(double)
    prop_dict["num_triple"] = bonds.count(triple)
    prop_dict["aromatic_bonds"] = bonds.count(aromatic)
    
    # atom type counts
    prop_dict["num_fluorine"] = count("F")
    prop_dict["num_chlorine"] = count("Cl")
    prop_dict["num_bromine"] = count("Br")
    prop_dict["num_nitro"] = count("N") + count("n")
    prop_dict["num_oxy"] = count("O") + count("o")
    prop_dict["num_sulfur"] = count("S") + count("s") - count("Se") - count("se")
    
    # complexity of structure
    prop_dict["unique_colors"] = len(set(color(mol)))
    
    # PEOE descriptors
    m_peoe = peoe(mol)
    for i, mp in enumerate(m_peoe):
        prop_dict[f"peoe_{i}"] = mp
    
    # MQNs descriptors
    m_mqns = mqns(mol)
    for j, mq in enumerate(m_mqns):
        prop_dict[f"mqns_{j}"] = mq
    
    # molecular descriptors
    prop_dict["tpsa"] = tpsa(mol)
    prop_dict["unspecified_stero"] = unspecified_stereo(mol)
    prop_dict["sat_rings"] = sat_rings(mol)
    prop_dict["sat_het"] = sat_hetero(mol)
    prop_dict["sat_carbo"] = saturated_carbo(mol)
    prop_dict["rotable_bonds"] = rotatable_bonds(mol)
    prop_dict["num_rings"] = num_rings(mol)
    prop_dict["lipinski_hba"] = lipinski_hba(mol)
    prop_dict["lipinski_hbd"] = lipinski_hbd(mol)
    prop_dict["hetero"] = heterocycles(mol)
    prop_dict["hetero_atoms"] = hetero_atoms(mol)
    prop_dict["h_accept"] = hba(mol)
    prop_dict["donate_h"] = hbd(mol)
    prop_dict["bridge_head"] = bridge_head(mol)
    prop_dict["stereo"] = stereo_centers(mol)
    prop_dict["aromatic_rings"] = rings_aromatic(mol)
    prop_dict["hetero_aromatic"] = aromatic_hetero(mol)
    prop_dict["carbo_aromatic"] = aromatic_carbo(mol)
    prop_dict["num_amide"] = amide_bonds(mol)
    prop_dict["ala_rings"] = rings_ala(mol)
    prop_dict["alaphatic_hetero"] = alaphatic_hetero(mol)
    prop_dict["alaphatic_carbo"] = alphaptic_carbo(mol)
    prop_dict["labuta_asa"] = labute_asa(mol)
    prop_dict["kappa1"] = kappa1(mol)
    prop_dict["kappa2"] = kappa2(mol)
    prop_dict["kappa3"] = kappa3(mol)
    
    # Crippen descriptors
    crip = crippen(mol)
    for k, cri in enumerate(crip):
        prop_dict[f"crippen_{k}"] = cri
    
    # topological descriptors
    prop_dict["c_sp3"] = frac_C_sp3(mol)
    prop_dict["hall_kier"] = hall_kier(mol)
    prop_dict["chi0n"] = chi0n(mol)
    prop_dict["chi0v"] = chi0v(mol)
    prop_dict["chi1n"] = chi1n(mol)
    prop_dict["chi1v"] = chi1v(mol)
    prop_dict["chi2n"] = chi2n(mol)
    prop_dict["chi2v"] = chi2v(mol)
    prop_dict["chi3n"] = chi3n(mol)
    prop_dict["chi3v"] = chi3v(mol)
    prop_dict["chi4n"] = chi4n(mol)
    prop_dict["chi4v"] = chi4v(mol)
    
    # 2D autocorrelation descriptors
    auto_2d = calcauto2d(mol)
    for l, _2d in enumerate(auto_2d):
        prop_dict[f"auto_2d_{l}"] = _2d
    
    # basic physicochemical properties
    prop_dict["num_valence"] = num_valence(mol)
    prop_dict["num_radical"] = num_radical(mol)
    prop_dict["avg_mol_weight"] = avg_mol_wt(mol)
    prop_dict["heavy_atom"] = heavy_atom_wt(mol)
    prop_dict["fp1_densirt"] = fpdensity1(mol)
    prop_dict["fp2_density"] = fpdensity2(mol)
    prop_dict["fp3_density"] = fpdensity3(mol)
    prop_dict["weight"] = mw(mol)
    
    return prop_dict

def compute_rdkit_descriptors(smiles_list):
    """
    Compute a set of descriptors for a list of SMILES.
    
    Args:
        smiles_list (list of str): SMILES strings
    
    Returns:
        np.ndarray: shape (n_samples, n_descriptors)
        list: descriptor names
    """
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]

    descriptor_names = None
    descriptors = []
    for mol in mols:
        if mol is None:
            descriptors.append([np.nan] * len(descriptor_names)) # type: ignore
        else:
            info = all_properties(mol)
            mol_descriptors = list(info.values())
            descriptors.append(mol_descriptors)
            if descriptor_names is None:
                descriptor_names = list(info.keys())  
    return np.array(descriptors), descriptor_names


if __name__ == "__main__":
    smiles = ["C1=CC=CC=C1", "CCO", "C"]
    X, names = compute_rdkit_descriptors(smiles)
    df = pd.DataFrame(X, columns=names)
    print(df.head())
