"""
phase2_cost_sensitive_training.py -- Cost-Sensitive LightGBM Training
Samsung Capstone Project - Table 1 (Team LightGBM)

Custom cost-sensitive objective: C_FP=1.0, C_FN=10.0
"""

import sys
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
    """Custom eval metric: total_cost = C_FP*FP + C_FN*FN. Lower is better."""
    y_true = dtrain.get_label().astype(int)
    p = 1.0 / (1.0 + np.exp(-y_pred))
    y_hat = (p >= 0.5).astype(int)
    fp = int(np.sum((y_hat == 1) & (y_true == 0)))
    fn = int(np.sum((y_hat == 0) & (y_true == 1)))
    total_cost = C_FP * fp + C_FN * fn
    return "total_cost", total_cost, False


def load_splits():
    """Load pickled train/test DataFrames from Phase 1."""
    train_pkl = ARTEFACTS_DIR / "train.pkl"
    test_pkl = ARTEFACTS_DIR / "test.pkl"

    for p in (train_pkl, test_pkl):
        if not p.is_file():
            print(f"[ERROR] Missing artefact: {p}")
            print("[HINT]  Run phase1_data_engineering.py first.")
            sys.exit(1)

    with open(train_pkl, "rb") as f:
        train_df = pickle.load(f)
    with open(test_pkl, "rb") as f:
        test_df = pickle.load(f)

    print(f"[INFO] Loaded train: {train_df.shape}  |  test: {test_df.shape}")
    return train_df, test_df


def prepare_features(train_df, test_df):
    """Separate features and labels."""
    drop_cols = ["TransactionID", "isFraud"]
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    X_train = train_df[feature_cols].values
    y_train = train_df["isFraud"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["isFraud"].values

    return X_train, X_test, y_train, y_test, feature_cols


def train_model(X_train, y_train, X_test, y_test, feature_names):
    """Train LightGBM with custom cost-sensitive objective."""
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names, free_raw_data=False)
    dtest = lgb.Dataset(X_test, label=y_test, feature_name=feature_names, reference=dtrain, free_raw_data=False)

    params = {
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 50,
        "objective": cost_sensitive_objective,
        "metric": "None",
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }

    print("\n[INFO] Training LightGBM with cost-sensitive objective ...")
    print(f"       C_FP={C_FP}  |  C_FN={C_FN}")
    print(f"       num_leaves={params['num_leaves']}  |  lr={params['learning_rate']}")
    print(f"       n_estimators=1000  |  early_stopping=100 rounds\n")

    callbacks = [
        lgb.early_stopping(stopping_rounds=100, verbose=True),
        lgb.log_evaluation(period=50),
    ]

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dtrain, dtest],
        valid_names=["train", "valid"],
        feval=cost_sensitive_eval,
        callbacks=callbacks,
    )

    print(f"\n[INFO] Best iteration: {booster.best_iteration}")
    return booster


def evaluate_model(booster, X_test, y_test, test_df):
    """Evaluate and save predictions."""
    raw_preds = booster.predict(X_test, num_iteration=booster.best_iteration)
    proba = 1.0 / (1.0 + np.exp(-raw_preds))
    y_hat = (proba >= 0.5).astype(int)

    print("\n" + "=" * 60)
    print("  MODEL EVALUATION")
    print("=" * 60)

    print("\n-- Classification Report --")
    print(classification_report(y_test, y_hat, target_names=["Legit", "Fraud"]))

    cm = confusion_matrix(y_test, y_hat)
    tn, fp, fn, tp = cm.ravel()
    print("-- Confusion Matrix --")
    print(f"       Predicted Legit   Predicted Fraud")
    print(f"  Legit     {tn:>8,}        {fp:>8,}")
    print(f"  Fraud     {fn:>8,}        {tp:>8,}")

    auc = roc_auc_score(y_test, proba)
    total_cost = C_FP * fp + C_FN * fn
    print(f"\n-- Key Metrics --")
    print(f"  AUC-ROC      : {auc:.6f}")
    print(f"  Total Cost   : {total_cost:,.0f}  (C_FP*{fp:,} + C_FN*{fn:,})")
    print(f"  FP Cost      : {C_FP * fp:,.0f}")
    print(f"  FN Cost      : {C_FN * fn:,.0f}")
    print("=" * 60 + "\n")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    preds_df = pd.DataFrame({
        "TransactionID": test_df["TransactionID"].values,
        "isFraud_actual": y_test,
        "prediction_probability": np.round(proba, 6),
        "predicted_label": y_hat,
    })
    csv_path = OUTPUTS_DIR / "lightgbm_predictions.csv"
    preds_df.to_csv(csv_path, index=False)
    print(f"[INFO] Saved predictions -> {csv_path}")


def save_model(booster):
    """Persist the trained booster."""
    model_path = ARTEFACTS_DIR / "lgbm_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(booster, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[INFO] Saved model      -> {model_path}")


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  PHASE 2 -- Cost-Sensitive LightGBM Training")
    print("=" * 60 + "\n")

    train_df, test_df = load_splits()
    X_train, X_test, y_train, y_test, feature_names = prepare_features(train_df, test_df)
    print(f"[INFO] Features: {len(feature_names)}  |  "
          f"Train fraud ratio: {y_train.mean():.4f}  |  "
          f"Test fraud ratio: {y_test.mean():.4f}")

    booster = train_model(X_train, y_train, X_test, y_test, feature_names)
    evaluate_model(booster, X_test, y_test, test_df)
    save_model(booster)

    print("\n[SUCCESS] Phase 2 complete. Ready for Phase 3.\n")


if __name__ == "__main__":
    main()
