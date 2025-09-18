import argparse
import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
os.chdir(BASE_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        layers = []
        dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

def main():
    parser = argparse.ArgumentParser(description="Train FFN with best hyperparameters")
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="processed")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    X = np.load(f"{args.data_dir}/X.npy")
    y_df = pd.read_csv(f"{args.data_dir}/y.csv")
    if args.target not in y_df.columns:
        raise ValueError(f"Target {args.target} not found")
    y = y_df[args.target].values

    best_params_file = os.path.join(args.results_dir, "best_params.json")
    with open(best_params_file, "r") as f:
        best_params = json.load(f)["ffn"][args.target]

    hidden_dim = best_params["hidden_dim"]
    num_layers = best_params["num_layers"]
    dropout = best_params["dropout"]
    lr = best_params["lr"]
    batch_size = best_params["batch_size"]

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32),
                            torch.tensor(y, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = FFN(X.shape[1], hidden_dim, num_layers, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr) # type: ignore
    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        train_epoch(model, loader, optimizer, criterion)

    model_path = os.path.join(args.results_dir, "models", f"ffn_{args.target}.pt")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()