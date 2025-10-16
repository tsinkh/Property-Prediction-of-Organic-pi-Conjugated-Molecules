import os
import sys
import json
import pandas as pd
from tqdm import tqdm

from dgl.data.utils import save_graphs
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, BASE_DIR)

DATA_PATH = os.path.join("data", "molecules.csv")
OUTPUT_DIR = "processed"

def build_mpnn_dataset():
    print(f"Loading molecules from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    smiles_list = df["smiles"].tolist()

    atom_featurizer = CanonicalAtomFeaturizer()
    bond_featurizer = CanonicalBondFeaturizer()

    graphs = []
    failed = 0

    print("Converting SMILES to DGLGraphs...")
    for smi in tqdm(smiles_list):
        try:
            g = smiles_to_bigraph(smi,
                                  node_featurizer=atom_featurizer,
                                  edge_featurizer=bond_featurizer)
            graphs.append(g)
        except Exception as e:
            failed += 1
            graphs.append(None)

    if failed > 0:
        print(f"Warning: {failed} molecules failed to convert and were skipped.")

    # 过滤失败的
    valid_idx = [i for i, g in enumerate(graphs) if g is not None]
    graphs = [g for g in graphs if g is not None]
    df = df.iloc[valid_idx].reset_index(drop=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 保存图
    graph_path = os.path.join(OUTPUT_DIR, "graphs.bin")
    save_graphs(graph_path, graphs)
    print(f"Saved {len(graphs)} graphs to {graph_path}")

    # 保存 y
    target_cols = [c for c in df.columns if c not in ["identifier", "smiles"]]
    y = df[target_cols]
    y.to_csv(os.path.join(OUTPUT_DIR, "y.csv"), index=False)
    print(f"Saved targets to {OUTPUT_DIR}/y.csv")

    # 保存图信息
    graph_info = {
        "node_in_feats": graphs[0].ndata["h"].shape[1],
        "edge_in_feats": graphs[0].edata["e"].shape[1],
        "num_graphs": len(graphs),
        "target_cols": target_cols,
    }
    with open(os.path.join(OUTPUT_DIR, "graphs_info.json"), "w") as f:
        json.dump(graph_info, f, indent=2)
    print(f"Saved graph info to {OUTPUT_DIR}/graphs_info.json")

if __name__ == "__main__":
    build_mpnn_dataset()