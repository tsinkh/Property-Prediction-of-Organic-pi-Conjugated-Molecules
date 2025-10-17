import os
import sys
import json
import argparse
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

# 项目根目录
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
os.chdir(BASE_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# 导入模型
from src.featurization.mpnn_evidential_readout import mpnn_evidential_readout

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======== Evidential Loss 定义 ========
class EvidentialLoss(nn.Module):
    def __init__(self, coeff=0.2):
        super().__init__()
        self.coeff = coeff

    def forward(self, outputs, targets):
        means, lambdas, alphas, betas = torch.chunk(outputs, 4, dim=1)
        twoBlambda = 2 * betas * (1 + lambdas)
        nll = 0.5 * torch.log(np.pi / lambdas) \
              - alphas * torch.log(twoBlambda) \
              + (alphas + 0.5) * torch.log(lambdas * (targets - means) ** 2 + twoBlambda) \
              + torch.lgamma(alphas) - torch.lgamma(alphas + 0.5)
        reg = torch.abs(targets - means) * (2 * lambdas + alphas)
        loss = nll + self.coeff * reg
        return torch.mean(loss)


# ======== DataLoader 的 collate 函数 ========
def collate_fn(batch):
    graphs, feats, labels = map(list, zip(*batch))
    batched_graph = dgl.batch(graphs)
    feats = torch.tensor(np.stack(feats), dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    return batched_graph, feats, labels


# ======== 单 epoch 训练 ========
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    for g, feats, y in loader:
        g, feats, y = g.to(device), feats.to(device), y.to(device)
        node_feats = g.ndata["h"].to(device)
        edge_feats = g.edata["e"].to(device)

        optimizer.zero_grad()
        pred = model(g, node_feats, edge_feats, feats)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()


# ======== 验证 ========
def eval_epoch(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for g, feats, y in loader:
            g, feats, y = g.to(device), feats.to(device), y.to(device)
            node_feats = g.ndata["h"].to(device)
            edge_feats = g.edata["e"].to(device)
            pred = model(g, node_feats, edge_feats, feats)
            means, _, _, _ = torch.chunk(pred, 4, dim=1)
            preds.append(means.squeeze().cpu().numpy())
            targets.append(y.squeeze().cpu().numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    return r2_score(targets, preds), mean_absolute_error(targets, preds)


# ======== Optuna objective ========
def objective(trial, graphs, descriptors, y, node_in, edge_in, results_list):
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
        X_train, X_val = descriptors[train_idx], descriptors[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_ds = list(zip(train_graphs, X_train, y_train))
        val_ds = list(zip(val_graphs, X_val, y_val))
        train_loader = DataLoader(train_ds, batch_size=params["batch_size"], # type: ignore
                                  shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=params["batch_size"], # type: ignore
                                collate_fn=collate_fn)

        model = mpnn_evidential_readout(
            node_in_feats=node_in,
            edge_in_feats=edge_in,
            node_out_feats=params["node_out_feats"],
            edge_hidden_feats=params["edge_hidden_feats"],
            num_step_message_passing=params["num_step_message_passing"],
            num_step_set2set=params["num_step_set2set"],
            num_layer_set2set=params["num_layer_set2set"],
            dropout=params["dropout"],
            descriptor_feats=X_train.shape[1],
            n_tasks=1
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=params["lr"])
        criterion = EvidentialLoss()

        for epoch in range(40):
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
        "mae": np.mean(fold_mae)
    })
    return np.mean(fold_r2)


# ======== main ========
def main():
    parser = argparse.ArgumentParser(description="MPNN Evidential model hyperparameter tuning with Optuna")
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="processed")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()

    graphs, _ = load_graphs(f"{args.data_dir}/graphs.bin")
    graph_info = json.load(open(f"{args.data_dir}/graphs_info.json", "r"))
    node_in = graph_info["node_in_feats"]
    edge_in = graph_info["edge_in_feats"]

    descriptors = np.load(f"{args.data_dir}/X.npy")
    y_df = pd.read_csv(f"{args.data_dir}/y.csv")

    if args.target not in y_df.columns:
        raise ValueError(f"Target {args.target} not found in y.csv")

    y = y_df[args.target].values

    results_list = []
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, graphs, descriptors, y, node_in, edge_in, results_list), # type: ignore
                   n_trials=args.trials)

    print(f"Best R²: {study.best_value:.4f}")
    print("Best hyperparameters:", study.best_params)

    os.makedirs(os.path.join(args.results_dir, "tuning"), exist_ok=True)
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(os.path.join(args.results_dir, "tuning", f"mpnn_evidential_{args.target}.csv"), index=False)

    # 更新 best_params.json
    best_path = os.path.join(args.results_dir, "best_params.json")
    if os.path.exists(best_path):
        with open(best_path, "r") as f:
            best_params = json.load(f)
    else:
        best_params = {}

    if "mpnn_evidential" not in best_params:
        best_params["mpnn_evidential"] = {}

    best_params["mpnn_evidential"][args.target] = study.best_params

    with open(best_path, "w") as f:
        json.dump(best_params, f, indent=2)

    print(f"Saved tuning results to {args.results_dir}/tuning/mpnn_evidential_{args.target}.csv")
    print(f"Updated best_params.json")

if __name__ == "__main__":
    main()