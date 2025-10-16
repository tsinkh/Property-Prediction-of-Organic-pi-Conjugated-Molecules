import argparse
import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import dgl
from dgl.data.utils import load_graphs
from src.featurization.mpnn_readout import mpnn_readout

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
os.chdir(BASE_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_fn(batch):
    graphs, labels = map(list, zip(*batch))
    batched_graph = dgl.batch(graphs)
    labels = torch.tensor(labels, dtype=torch.float32)
    return batched_graph, labels


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    for g, y in loader:
        g, y = g.to(device), y.to(device)
        node_feats = g.ndata["h"].to(device)
        edge_feats = g.edata["e"].to(device)
        optimizer.zero_grad()
        pred = model(g, node_feats, edge_feats)
        loss = criterion(pred.squeeze(), y)
        loss.backward()
        optimizer.step()


def main():
    parser = argparse.ArgumentParser(description="Train MPNN with best hyperparameters")
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="processed")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    # === Load data ===
    graphs, _ = load_graphs(os.path.join(args.data_dir, "graphs.bin"))
    y_df = pd.read_csv(os.path.join(args.data_dir, "y.csv"))

    if args.target not in y_df.columns:
        raise ValueError(f"Target {args.target} not found in y.csv")
    y = y_df[args.target].values.astype(np.float32)

    with open(os.path.join(args.data_dir, "graphs_info.json"), "r") as f:
        info = json.load(f)
    node_in, edge_in = info["node_in_feats"], info["edge_in_feats"]

    # === Load best hyperparameters ===
    best_params_file = os.path.join(args.results_dir, "best_params.json")
    with open(best_params_file, "r") as f:
        best_params = json.load(f)["mpnn"][args.target]

    node_out_feats = best_params["node_out_feats"]
    edge_hidden_feats = best_params["edge_hidden_feats"]
    num_step_message_passing = best_params["num_step_message_passing"]
    num_step_set2set = best_params["num_step_set2set"]
    num_layer_set2set = best_params["num_layer_set2set"]
    dropout = best_params["dropout"]
    lr = best_params["lr"]
    batch_size = best_params["batch_size"]

    dataset = list(zip(graphs, y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn) # type: ignore

    # === Build model ===
    model = mpnn_readout(
        node_in_feats=node_in,
        edge_in_feats=edge_in,
        node_out_feats=node_out_feats,
        edge_hidden_feats=edge_hidden_feats,
        num_step_message_passing=num_step_message_passing,
        num_step_set2set=num_step_set2set,
        num_layer_set2set=num_layer_set2set,
        dropout=dropout,
        n_tasks=1
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # === Train ===
    for epoch in range(args.epochs):
        train_epoch(model, loader, optimizer, criterion)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} completed")

    # === Save model ===
    model_path = os.path.join(args.results_dir, "models", f"mpnn_{args.target}.pt")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()