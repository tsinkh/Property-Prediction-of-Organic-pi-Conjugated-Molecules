import argparse
import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error
import optuna

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

def eval_epoch(model, loader, criterion):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            preds.append(pred.cpu().numpy())
            targets.append(yb.cpu().numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    return r2_score(targets, preds), mean_absolute_error(targets, preds)

def objective(trial, X, y, results_list):
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
    num_layers = trial.suggest_int("num_layers", 1, 4)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_r2, fold_mae = [], []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                 torch.tensor(y_train, dtype=torch.float32))
        val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                               torch.tensor(y_val, dtype=torch.float32))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        model = FFN(X.shape[1], hidden_dim, num_layers, dropout).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr) # type: ignore
        criterion = nn.MSELoss()

        for epoch in range(50):  # 固定epoch
            train_epoch(model, train_loader, optimizer, criterion)

        r2, mae = eval_epoch(model, val_loader, criterion)
        fold_r2.append(r2)
        fold_mae.append(mae)

        results_list.append({
            "trial": trial.number,
            **trial.params,
            "fold": fold,
            "r2": r2,
            "mae": mae
        })

    results_list.append({
        "trial": trial.number,
        **trial.params,
        "fold": "mean",
        "r2": np.mean(fold_r2),
        "mae": np.mean(fold_mae),
    })
    return np.mean(fold_r2)

def main():
    parser = argparse.ArgumentParser(description="FFN hyperparameter tuning with Optuna")
    parser.add_argument("--target", type=str, required=True, help="Target property")
    parser.add_argument("--data_dir", type=str, default="processed")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()

    X = np.load(f"{args.data_dir}/X.npy")
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    y_df = pd.read_csv(f"{args.data_dir}/y.csv")
    if args.target not in y_df.columns:
        raise ValueError(f"Target {args.target} not found")
    y = y_df[args.target].values

    results_list = []
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X, y, results_list), n_trials=args.trials) # type: ignore

    print(f"Best R²: {study.best_value:.4f}")
    print("Best hyperparameters:", study.best_params)

    os.makedirs(os.path.join(args.results_dir, "tuning"), exist_ok=True)
    results_df = pd.DataFrame(results_list)

    best_trial = study.best_trial.number
    best_row = results_df[(results_df["trial"] == best_trial) & (results_df["fold"] == "mean")].iloc[0]
    final_row = {
        "trial": "final",
        **study.best_params,
        "fold": "all",
        "r2": best_row["r2"],
        "mae": best_row["mae"],
    }
    results_df = pd.concat([results_df, pd.DataFrame([final_row])], ignore_index=True)

    csv_path = os.path.join(args.results_dir, "tuning", f"ffn_{args.target}.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Saved tuning results to {csv_path}")

    best_params_file = os.path.join(args.results_dir, "best_params.json")
    if os.path.exists(best_params_file):
        with open(best_params_file, "r") as f:
            all_params = json.load(f)
    else:
        all_params = {}
    if "ffn" not in all_params:
        all_params["ffn"] = {}
    all_params["ffn"][args.target] = study.best_params
    with open(best_params_file, "w") as f:
        json.dump(all_params, f, indent=2)
    print(f"Best parameters saved to {best_params_file}")

if __name__ == "__main__":
    main()