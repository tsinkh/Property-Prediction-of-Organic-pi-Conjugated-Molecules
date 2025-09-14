import argparse
import os, json, joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
os.chdir(BASE_DIR)

def main():
    parser = argparse.ArgumentParser(description="Train classical ML model with best parameters")
    parser.add_argument("--target", type=str, required=True,
                        help="Target property: homo, lumo, vie, ...")
    parser.add_argument("--model", type=str, required=True,
                        choices=["ridge", "svr", "krr"])
    parser.add_argument("--data_dir", type=str, default="processed")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    # load data
    X = np.load(f"{args.data_dir}/X.npy")
    y_df = pd.read_csv(f"{args.data_dir}/y.csv")
    if args.target not in y_df.columns:
        raise ValueError(f"Target {args.target} not found. Available: {list(y_df.columns)}")
    y = y_df[args.target].values

    # load best params
    with open(os.path.join(args.results_dir, "best_params.json"), "r") as f:
        best_params = json.load(f)

    if args.model not in best_params or args.target not in best_params[args.model]:
        raise ValueError(f"No best params found for {args.model}-{args.target}")

    params = best_params[args.model][args.target]

    # instantiate model
    if args.model == "ridge":
        model = Ridge(**params)
    elif args.model == "svr":
        model = SVR(**params)
    elif args.model == "krr":
        model = KernelRidge(kernel="rbf", **params)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    pipeline.fit(X, y)

    os.makedirs(os.path.join(args.results_dir, "models"), exist_ok=True)
    model_path = os.path.join(args.results_dir, "models", f"{args.model}_{args.target}.pkl")
    joblib.dump(pipeline, model_path)

    print(f"Final model trained and saved to {model_path}")

if __name__ == "__main__":
    main()