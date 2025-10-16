import os

# Ensure repository root is on sys.path before importing local `src` package
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
os.chdir(BASE_DIR)
import sys
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import dgl
from dgl.data.utils import load_graphs
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error
import optuna

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


def eval_epoch(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for g, y in loader:
            g, y = g.to(device), y.to(device)
            node_feats = g.ndata["h"].to(device)
            edge_feats = g.edata["e"].to(device)
            pred = model(g, node_feats, edge_feats)
            preds.append(pred.squeeze().cpu().numpy())
            targets.append(y.cpu().numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    return r2_score(targets, preds), mean_absolute_error(targets, preds)


def objective(trial, graphs, y, node_in, edge_in, results_list):
    params = {
        "node_out_feats": trial.suggest_categorical("node_out_feats", [64, 128, 256]),
        "edge_hidden_feats": trial.suggest_categorical("edge_hidden_feats", [64, 128, 256]),
        "num_step_message_passing": trial.suggest_int("num_step_message_passing", 3, 10),
        "num_step_set2set": trial.suggest_int("num_step_set2set", 3, 10),
        "num_layer_set2set": trial.suggest_int("num_layer_set2set", 1, 5),
        "dropout": trial.suggest_float("dropout", 0.0, 0.8),
        "lr": trial.suggest_loguniform("lr", 1e-4, 1e-2),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32])
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_r2, fold_mae = [], []

    for fold, (train_idx, val_idx) in enumerate(cv.split(graphs), 1):
        train_graphs = [graphs[i] for i in train_idx]
        val_graphs = [graphs[i] for i in val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_ds = list(zip(train_graphs, y_train))
        val_ds = list(zip(val_graphs, y_val))
        train_loader = DataLoader(train_ds, batch_size=params["batch_size"], # type: ignore
                                  shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=params["batch_size"], # type: ignore
                                collate_fn=collate_fn)

        model = mpnn_readout(
            node_in_feats=node_in,
            edge_in_feats=edge_in,
            node_out_feats=params["node_out_feats"],
            edge_hidden_feats=params["edge_hidden_feats"],
            num_step_message_passing=params["num_step_message_passing"],
            num_step_set2set=params["num_step_set2set"],
            num_layer_set2set=params["num_layer_set2set"],
            dropout=params["dropout"],
            n_tasks=1
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=params["lr"])
        criterion = nn.MSELoss()

        for epoch in range(40):  # 固定 epoch
            train_epoch(model, train_loader, optimizer, criterion)

        r2, mae = eval_epoch(model, val_loader)
        fold_r2.append(r2)
        fold_mae.append(mae)

        results_list.append({
            "trial": trial.number,
            **params,
            "fold": fold,
            "r2": r2,
            "mae": mae
        })

    results_list.append({
        "trial": trial.number,
        **params,
        "fold": "mean",
        "r2": np.mean(fold_r2),
        "mae": np.mean(fold_mae),
    })

    return np.mean(fold_r2)


def main():
    parser = argparse.ArgumentParser(description="MPNN hyperparameter tuning with Optuna")
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="processed")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()

    # === Load dataset ===
    graphs, _ = load_graphs(os.path.join(args.data_dir, "graphs.bin"))
    y_df = pd.read_csv(os.path.join(args.data_dir, "y.csv"))

    if args.target not in y_df.columns:
        raise ValueError(f"Target {args.target} not found in y.csv")

    y = y_df[args.target].values.astype(np.float32)
    with open(os.path.join(args.data_dir, "graphs_info.json"), "r") as f:
        info = json.load(f)
    node_in, edge_in = info["node_in_feats"], info["edge_in_feats"]

    # === Tuning ===
    results_list = []
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, graphs, y, node_in, edge_in, results_list), # type: ignore
                   n_trials=args.trials)

    print(f"Best R²: {study.best_value:.4f}")
    print("Best hyperparameters:", study.best_params)

    # === Save results ===
    os.makedirs(os.path.join(args.results_dir, "tuning"), exist_ok=True)
    results_df = pd.DataFrame(results_list)
    csv_path = os.path.join(args.results_dir, "tuning", f"mpnn_{args.target}.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Saved detailed results to {csv_path}")

    best_trial = study.best_trial.number
    best_row = results_df[(results_df["trial"] == best_trial) & (results_df["fold"] == "mean")].iloc[0]

    json_path = os.path.join(args.results_dir, "best_params.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            best_params_all = json.load(f)
    else:
        best_params_all = {}

    if "mpnn" not in best_params_all:
        best_params_all["mpnn"] = {}
    best_params_all["mpnn"][args.target] = {k: best_row[k] for k in study.best_params.keys()}

    with open(json_path, "w") as f:
        json.dump(best_params_all, f, indent=2)
    print(f"Updated best_params.json at {json_path}")


if __name__ == "__main__":
    main()