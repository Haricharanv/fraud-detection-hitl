"""
train_fifar.py -- Cost-Sensitive LightGBM & SHAP Pipeline for Financial Fraud Alert Review (FiFAR)
Samsung Capstone Project - Table 1

Dataset: FiFAR (506,118 train rows, 96,843 test rows, 37 columns)
Target: fraud_bool (train), fraud_label (test)
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
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = PROJECT_ROOT / "data" / "fifar" / "ICAIF_KAGGLE" / "testbed" / "train" / "small__regular" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "fifar" / "ICAIF_KAGGLE" / "testbed" / "test" / "test.csv"
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
    print(f"[INFO] Loading FiFAR dataset ...")
    print(f"       Train: {TRAIN_PATH}")
    print(f"       Test:  {TEST_PATH}")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print(f"[INFO] Raw train: {train_df.shape}  |  Raw test: {test_df.shape}")

    # Standardize target and ID columns
    y_train = train_df["fraud_bool"].astype(np.int8).values
    y_test = test_df["fraud_label"].astype(np.int8).values

    # Metadata columns to drop from features
    drop_train = ["case_id", "fraud_bool", "assignment", "decision", "batch"]
    drop_test = ["case_id", "fraud_label"]

    feature_cols = [c for c in train_df.columns if c not in drop_train and c in test_df.columns]
    print(f"[INFO] Selected {len(feature_cols)} aligned features.")

    # Encode categorical columns
    cat_cols = train_df[feature_cols].select_dtypes(include=["object"]).columns.tolist()
    print(f"[INFO] Encoding categorical columns: {cat_cols}")

    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        # Combine train & test values for fit to avoid unseen labels
        combined = pd.concat([train_df[col].astype(str), test_df[col].astype(str)], axis=0)
        le.fit(combined)
        train_df[col] = le.transform(train_df[col].astype(str)).astype(np.int32)
        test_df[col] = le.transform(test_df[col].astype(str)).astype(np.int32)
        encoders[col] = le

    # Cast numericals to float32
    for col in feature_cols:
        if col not in cat_cols:
            train_df[col] = pd.to_numeric(train_df[col], errors="coerce").fillna(0).astype(np.float32)
            test_df[col] = pd.to_numeric(test_df[col], errors="coerce").fillna(0).astype(np.float32)

    train_features = train_df[feature_cols].copy()
    test_features = test_df[feature_cols].copy()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    test_export = test_df[["case_id"] + feature_cols].copy()
    test_export["fraud_label"] = y_test
    test_export_path = OUTPUTS_DIR / "preprocessed_test_fifar.csv"
    test_export.to_csv(test_export_path, index=False)
    print(f"[INFO] Saved preprocessed test set -> {test_export_path}")

    return train_features, test_features, y_train, y_test, test_df["case_id"].values, feature_cols


def train_model(X_train_df, X_test_df, y_train, y_test, feature_cols):
    X_train = X_train_df.values
    X_test = X_test_df.values

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols, free_raw_data=False)
    dtest = lgb.Dataset(X_test, label=y_test, feature_name=feature_cols, reference=dtrain, free_raw_data=False)

    params = {
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
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

    print("\n[INFO] Training LightGBM on FiFAR ...")
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
    print("  FIFAR (ALERT REVIEW) -- EVALUATION METRICS")
    print("=" * 60)
    print(f"  AUC-ROC      : {auc:.6f}")
    print(f"  Total Cost   : {total_cost:,.0f}  (FP={fp:,}, FN={fn:,})")
    print(f"  Fraud Recall : {tp/(tp+fn):.4f} ({tp:,}/{tp+fn:,})")
    print("=" * 60)

    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTEFACTS_DIR / "lgbm_model_fifar.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(booster, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[INFO] Saved model -> {model_path}")

    return booster, X_test, y_test, proba


def explain_and_export(booster, case_ids, X_test, proba, feature_cols):
    print("\n[INFO] Computing SHAP values (TreeExplainer) ...")
    explainer = shap.TreeExplainer(booster)

    n_samples = min(50000, len(case_ids))
    X_sample = X_test[:n_samples]
    case_ids_sample = case_ids[:n_samples]
    proba_sample = proba[:n_samples]

    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    print(f"[INFO] Building SHAP JSON payload for {n_samples:,} records ...")
    records = []

    for i in range(n_samples):
        sv = shap_values[i]
        top_indices = np.argsort(np.abs(sv))[::-1][:3]
        top3 = {feature_cols[idx]: round(float(sv[idx]), 6) for idx in top_indices}
        records.append({
            "transaction_id": str(case_ids_sample[i]),
            "dataset_name": "fifar",
            "prediction_probability": round(float(proba_sample[i]), 6),
            "top3_shap_features": top3,
        })

    json_path = OUTPUTS_DIR / "fifar_shap_output.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"[INFO] Saved SHAP JSON -> {json_path} ({len(records):,} records)")

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, show=False)
    plot_path = OUTPUTS_DIR / "shap_summary_fifar.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved SHAP plot -> {plot_path}")


def main():
    print("=" * 60)
    print("  TABLE 1: FiFAR - Financial Fraud Alert Review - Cost-Sensitive LightGBM")
    print("=" * 60)
    train_features, test_features, y_train, y_test, case_ids, feature_cols = load_and_preprocess()
    booster, X_test, y_test, proba = train_model(train_features, test_features, y_train, y_test, feature_cols)
    explain_and_export(booster, case_ids, X_test, proba, feature_cols)
    print("\n[SUCCESS] FiFAR pipeline complete.\n")


if __name__ == "__main__":
    main()

