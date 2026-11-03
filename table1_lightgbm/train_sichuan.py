"""
train_sichuan.py -- Cost-Sensitive LightGBM & SHAP Pipeline for Sichuan Telecom Fraud
Samsung Capstone Project - Table 1

Dataset: Sichuan Telecom Card Fraud (1,000,000 rows, 8 columns)
Features: Distance1, Distance2, Ratio, Repeat, Card, Pin, Online
Target: Fraud (0/1)
Cost-sensitive objective: C_FP=1.0, C_FN=10.0
"""

import gc
import json
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "sichuan-telecom" / "Card Fraud.csv"
ARTEFACTS_DIR = Path(__file__).resolve().parent / "artefacts"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

C_FP = 1.0
C_FN = 10.0


def cost_sensitive_objective(y_pred, dtrain):
    """Custom cost-sensitive binary cross-entropy objective."""
    y_true = dtrain.get_label().astype(np.float64)
    p = 1.0 / (1.0 + np.exp(-y_pred))
    weights = np.where(y_true == 1, C_FN, C_FP)
    grad = weights * (p - y_true)
    hess = weights * p * (1.0 - p)
    hess = np.maximum(hess, 1e-8)
    return grad, hess


def cost_sensitive_eval(y_pred, dtrain):
    """Custom eval metric: total_cost = C_FP*FP + C_FN*FN."""
    y_true = dtrain.get_label().astype(int)
    p = 1.0 / (1.0 + np.exp(-y_pred))
    y_hat = (p >= 0.5).astype(int)
    fp = int(np.sum((y_hat == 1) & (y_true == 0)))
    fn = int(np.sum((y_hat == 0) & (y_true == 1)))
    total_cost = C_FP * fp + C_FN * fn
    return "total_cost", total_cost, False


def load_and_preprocess():
    print(f"[INFO] Loading Sichuan Telecom dataset from {DATA_PATH} ...")
    if not DATA_PATH.is_file():
        print(f"[ERROR] Data file not found: {DATA_PATH}")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Loaded shape: {df.shape}")

    # Generate synthetic transaction ID for tracking
    df["TransactionID"] = [f"sichuan_{i+1:07d}" for i in range(len(df))]

    feature_cols = ["Distance1", "Distance2", "Ratio", "Repeat", "Card", "Pin", "Online"]
    for c in feature_cols:
        df[c] = df[c].astype(np.float32)
    df["Fraud"] = df["Fraud"].astype(np.int8)

    # Chronological / sequential split (80/20)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)

    print(f"[INFO] Train split: {train_df.shape}  |  Test split: {test_df.shape}")
    print(f"       Train fraud rate: {train_df['Fraud'].mean():.4f}")
    print(f"       Test fraud rate:  {test_df['Fraud'].mean():.4f}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    test_csv_path = OUTPUTS_DIR / "preprocessed_test_sichuan.csv"
    test_df.to_csv(test_csv_path, index=False)
    print(f"[INFO] Saved preprocessed test set -> {test_csv_path}")

    return train_df, test_df, feature_cols


def train_model(train_df, test_df, feature_cols):
    X_train = train_df[feature_cols].values
    y_train = train_df["Fraud"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["Fraud"].values

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols, free_raw_data=False)
    dtest = lgb.Dataset(X_test, label=y_test, feature_name=feature_cols, reference=dtrain, free_raw_data=False)

    params = {
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "objective": cost_sensitive_objective,
        "metric": "None",
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=50),
    ]

    print("\n[INFO] Training LightGBM on Sichuan Telecom Fraud ...")
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=500,
        valid_sets=[dtrain, dtest],
        valid_names=["train", "valid"],
        feval=cost_sensitive_eval,
        callbacks=callbacks,
    )

    raw_preds = booster.predict(X_test, num_iteration=booster.best_iteration)
    proba = 1.0 / (1.0 + np.exp(-raw_preds))
    y_hat = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, proba)
    cm = confusion_matrix(y_test, y_hat)
    tn, fp, fn, tp = cm.ravel()
    total_cost = C_FP * fp + C_FN * fn

    print("\n" + "=" * 60)
    print("  SICHUAN TELECOM -- EVALUATION METRICS")
    print("=" * 60)
    print(f"  AUC-ROC      : {auc:.6f}")
    print(f"  Total Cost   : {total_cost:,.0f}  (FP={fp:,}, FN={fn:,})")
    print(f"  Fraud Recall : {tp/(tp+fn):.4f} ({tp:,}/{tp+fn:,})")
    print("=" * 60)

    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTEFACTS_DIR / "lgbm_model_sichuan.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(booster, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[INFO] Saved model -> {model_path}")

    return booster, X_test, y_test, proba


def explain_and_export(booster, test_df, X_test, proba, feature_cols):
    print("\n[INFO] Computing SHAP values (TreeExplainer) ...")
    explainer = shap.TreeExplainer(booster)

    # Use first 50,000 test transactions for speedy explanation & handoff
    n_samples = min(50000, len(test_df))
    X_sample = X_test[:n_samples]
    sample_df = test_df.iloc[:n_samples]
    proba_sample = proba[:n_samples]

    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    print(f"[INFO] Building SHAP JSON payload for {n_samples:,} records ...")
    records = []
    txn_ids = sample_df["TransactionID"].values

    for i in range(n_samples):
        sv = shap_values[i]
        top_indices = np.argsort(np.abs(sv))[::-1][:3]
        top3 = {feature_cols[idx]: round(float(sv[idx]), 6) for idx in top_indices}
        records.append({
            "transaction_id": str(txn_ids[i]),
            "dataset_name": "sichuan_telecom",
            "prediction_probability": round(float(proba_sample[i]), 6),
            "top3_shap_features": top3,
        })

    json_path = OUTPUTS_DIR / "sichuan_shap_output.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"[INFO] Saved SHAP JSON -> {json_path} ({len(records):,} records)")

    # Plot summary
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, show=False)
    plot_path = OUTPUTS_DIR / "shap_summary_sichuan.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved SHAP plot -> {plot_path}")


def main():
    print("=" * 60)
    print("  TABLE 1: Sichuan Telecom Fraud - Cost-Sensitive LightGBM")
    print("=" * 60)
    train_df, test_df, feature_cols = load_and_preprocess()
    booster, X_test, y_test, proba = train_model(train_df, test_df, feature_cols)
    explain_and_export(booster, test_df, X_test, proba, feature_cols)
    print("\n[SUCCESS] Sichuan Telecom pipeline complete.\n")


if __name__ == "__main__":
    main()

