import argparse
import os, json
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
os.chdir(BASE_DIR)

def objective(trial, X, y, model_name, results_list):
    """Optuna objective function with 5-fold CV, saving each fold's results"""
    if model_name == "ridge":
        alpha = trial.suggest_loguniform("alpha", 1e-3, 1e3)
        model = Ridge(alpha=alpha)
    elif model_name == "svr":
        C = trial.suggest_float("C", 0.5, 2)
        epsilon = trial.suggest_float("epsilon", 0.01, 0.5)
        kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf"])
        model = SVR(kernel=kernel, C=C, epsilon=epsilon)
    elif model_name == "krr":
        alpha = trial.suggest_loguniform("alpha", 1e-3, 1e3)
        gamma = trial.suggest_loguniform("gamma", 1e-3, 1e1)
        model = KernelRidge(kernel="rbf", alpha=alpha, gamma=gamma)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    cv = KFold(n_splits=5, shuffle=True, random_state=None)
    fold_r2, fold_mae = [], []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)

        r2 = r2_score(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)

        fold_r2.append(r2)
        fold_mae.append(mae)

        results_list.append({
            "trial": trial.number,
            **trial.params,
            "fold": fold,
            "r2": r2,
            "mae": mae,
        })

    # mean results
    results_list.append({
        "trial": trial.number,
        **trial.params,
        "fold": "mean",
        "r2": np.mean(fold_r2),
        "mae": np.mean(fold_mae),
    })

    return np.mean(fold_r2)

def main():
    parser = argparse.ArgumentParser(description="Classical ML hyperparameter tuning with Optuna")
    parser.add_argument("--target", type=str, required=True,
                        help="Target property: homo, lumo, vie, ...")
    parser.add_argument("--model", type=str, default="ridge",
                        choices=["ridge", "svr", "krr"])
    parser.add_argument("--data_dir", type=str, default="processed")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()

    # load data
    X = np.load(f"{args.data_dir}/X.npy")
    y_df = pd.read_csv(f"{args.data_dir}/y.csv")
    if args.target not in y_df.columns:
        raise ValueError(f"Target {args.target} not found. Available: {list(y_df.columns)}")
    y = y_df[args.target].values

    results_list = []
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X, y, args.model, results_list), n_trials=args.trials) # type: ignore

    print(f"Best R²: {study.best_value:.4f}")
    print("Best hyperparameters:", study.best_params)

    # save trial results
    os.makedirs(args.results_dir, exist_ok=True)
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

    csv_path = os.path.join(args.results_dir, "tuning", f"{args.model}_{args.target}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    results_df.to_csv(csv_path, index=False)
    print(f"All trial results saved to {csv_path}")

    # update best_params.json
    best_params_file = os.path.join(args.results_dir, "best_params.json")
    if os.path.exists(best_params_file):
        with open(best_params_file, "r") as f:
            all_params = json.load(f)
    else:
        all_params = {}

    if args.model not in all_params:
        all_params[args.model] = {}

    all_params[args.model][args.target] = study.best_params

    with open(best_params_file, "w") as f:
        json.dump(all_params, f, indent=2)

    print(f"Best parameters saved to {best_params_file}")

if __name__ == "__main__":
    main()