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
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ======== 路径配置 ========
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

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
    total_loss = 0
    for g, feats, y in loader:
        g, feats, y = g.to(device), feats.to(device), y.to(device)
        node_feats = g.ndata["h"].to(device)
        edge_feats = g.edata["e"].to(device)
        optimizer.zero_grad()
        pred = model(g, node_feats, edge_feats, feats)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


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


# ======== main ========
def main():
    parser = argparse.ArgumentParser(description="Train MPNN Evidential model with best hyperparameters")
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="processed")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    # 载入数据
    graphs, _ = load_graphs(f"{args.data_dir}/graphs.bin")
    graph_info = json.load(open(f"{args.data_dir}/graphs_info.json", "r"))
    node_in = graph_info["node_in_feats"]
    edge_in = graph_info["edge_in_feats"]

    descriptors = np.load(f"{args.data_dir}/X.npy")
    y_df = pd.read_csv(f"{args.data_dir}/y.csv")

    if args.target not in y_df.columns:
        raise ValueError(f"Target {args.target} not found in y.csv")

    y = y_df[args.target].values

    X_train, X_val, y_train, y_val, g_train, g_val = train_test_split(
        descriptors, y, graphs, test_size=0.1, random_state=42
    )

    train_ds = list(zip(g_train, X_train, y_train))
    val_ds = list(zip(g_val, X_val, y_val))
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn) # type: ignore
    val_loader = DataLoader(val_ds, batch_size=16, collate_fn=collate_fn) # type: ignore

    # 读取 best_params.json
    best_path = os.path.join(args.results_dir, "best_params.json")
    if not os.path.exists(best_path):
        raise FileNotFoundError("best_params.json not found. Run tuning/mpnn_evidential.py first.")
    with open(best_path, "r") as f:
        best_params = json.load(f)

    params = best_params.get("mpnn_evidential", {}).get(args.target, None)
    if params is None:
        raise ValueError(f"No best parameters found for target {args.target} in best_params.json")

    # 初始化模型
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

    print(f"Training MPNN Evidential model for {args.target} ({args.epochs} epochs)")

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion)
        if epoch % 10 == 0 or epoch == args.epochs:
            r2, mae = eval_epoch(model, val_loader)
            print(f"Epoch {epoch:3d}: loss={loss:.4f}  val_R²={r2:.4f}  val_MAE={mae:.4f}")

    os.makedirs(os.path.join(args.results_dir, "models"), exist_ok=True)
    model_path = os.path.join(args.results_dir, "models", f"mpnn_evidential_{args.target}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Saved trained model to {model_path}")


if __name__ == "__main__":
    main()