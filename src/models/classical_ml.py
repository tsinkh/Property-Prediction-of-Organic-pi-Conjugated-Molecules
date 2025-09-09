"""
Classical ML models for molecular property prediction:
- Ridge Regression
- Support Vector Regression (SVR)
- Kernel Ridge Regression (KRR)

Evaluated with 5-fold cross-validation.
"""

import os
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error

# ----------------------------
# Config
# ----------------------------
DATA_PATH = os.path.join("data", "molecules.csv")
TARGET_COLUMN = "HOMO"
N_SPLITS = 5
RANDOM_SEED = 42

# ----------------------------
# Load dataset
# ----------------------------
print("Loading dataset from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)

# 假设目标列在最后一列或已知名称
y = df[TARGET_COLUMN].values

# 输入特征：去掉目标列
X = df.drop(columns=[TARGET_COLUMN]).values

print(f"Dataset shape: X={X.shape}, y={y.shape}")

# ----------------------------
# Define models
# ----------------------------
models = {
    "Ridge": Ridge(alpha=1.0, random_state=RANDOM_SEED),
    "SVR": SVR(kernel="rbf", C=1.0, epsilon=0.1),
    "KRR": KernelRidge(alpha=1.0, kernel="rbf")
}

# ----------------------------
# Cross-validation evaluation
# ----------------------------
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

def evaluate_model(name, model, X, y):
    """Run CV and return R2 and MAE scores."""
    # pipeline: 标准化 + 模型
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    # R2
    r2_scores = cross_val_score(
        pipeline, X, y,
        cv=kf, scoring="r2"
    )

    # MAE (需要自定义 scorer)
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    mae_scores = cross_val_score(
        pipeline, X, y,
        cv=kf, scoring=mae_scorer
    )

    return r2_scores, -mae_scores  # 取负号，转为正的 MAE

results = []

for name, model in models.items():
    r2_scores, mae_scores = evaluate_model(name, model, X, y)
    results.append({
        "Model": name,
        "R2_mean": np.mean(r2_scores),
        "R2_std": np.std(r2_scores),
        "MAE_mean": np.mean(mae_scores),
        "MAE_std": np.std(mae_scores)
    })
    print(f"[{name}] R2={np.mean(r2_scores):.3f}±{np.std(r2_scores):.3f}, "
          f"MAE={np.mean(mae_scores):.3f}±{np.std(mae_scores):.3f}")

# ----------------------------
# Save results
# ----------------------------
results_df = pd.DataFrame(results)
os.makedirs("results", exist_ok=True)
save_path = os.path.join("results", f"classical_ml_{TARGET_COLUMN}.csv")
results_df.to_csv(save_path, index=False)

print("Results saved to:", save_path)