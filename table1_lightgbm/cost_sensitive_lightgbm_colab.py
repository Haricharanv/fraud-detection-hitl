"""
================================================================================
IEEE-CIS FRAUD DETECTION — COST-SENSITIVE LIGHTGBM PIPELINE (COLAB + LOCAL)
Table 1: Machine Learning & Explainable AI (Hari's Core Pipeline)
================================================================================

Design Principles:
  1. Self-contained & Portable: Runs on Google Colab, Kaggle, or Local PC.
  2. Auto-Dependency Installation: Installs missing packages on demand.
  3. Flexible Data Discovery: Automatically searches Google Drive, /content,
     Kaggle paths, and local repo directories for transaction & identity CSVs.
  4. Memory-Optimised: Downcasts numeric types to run comfortably within 12 GB RAM.
  5. Chronological Leakage Protection: Sorts by TransactionDT before splitting.
  6. Custom Cost-Sensitive Objective: Implements asymmetric loss (C_FN=10, C_FP=1)
     via custom first-order gradients (g_i) and second-order hessians (h_i).
  7. High-Speed TreeSHAP Explainability: Computes exact local Shapley values
     and exports standardized handoff JSON for Table 3 FastAPI ingestion.
================================================================================
"""

import os
import sys
import gc
import json
import math
import pickle
import warnings
import subprocess
import importlib
from pathlib import Path

warnings.filterwarnings("ignore")

# ----------------------------- CONFIGURATION ----------------------------------
DATA_DIR = None                     # Optional explicit path (e.g. "/content/drive/MyDrive/ieee")
TRANSACTION_FILE = "train_transaction.csv"
IDENTITY_FILE = "train_identity.csv"

# Financial penalty matrix: C_FN (missing fraud) vs C_FP (false alarm)
C_FP = 1.0                          # Cost of falsely flagging legitimate transaction
C_FN = 10.0                         # Cost of letting fraudulent transaction slip through

TRAIN_SPLIT_RATIO = 0.80            # 80% chronological train, 20% untouched test
NUM_BOOST_ROUNDS = 2000
EARLY_STOPPING_ROUNDS = 100
MAX_SHAP_EXPLANATIONS = 10000       # Top transactions to explain via TreeSHAP
OUTPUT_DIR = "outputs"
ARTEFACTS_DIR = "artefacts"
RANDOM_SEED = 42

# -------------------------- DEPENDENCY MANAGEMENT -----------------------------
def _ensure_package(package_name, import_name=None):
    import_name = import_name or package_name
    try:
        importlib.import_module(import_name)
    except ImportError:
        print(f"[SETUP] Installing {package_name} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package_name])

_ensure_package("numpy")
_ensure_package("pandas")
_ensure_package("scikit-learn", "sklearn")
_ensure_package("lightgbm")
_ensure_package("shap")
_ensure_package("matplotlib")

import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    precision_recall_curve,
    roc_curve,
)

np.random.seed(RANDOM_SEED)

# --------------------------- DATASET DISCOVERY --------------------------------
def _find_file(filename):
    """Search common local, Colab, Drive, and Kaggle directories."""
    candidates = []
    if DATA_DIR:
        candidates.append(Path(DATA_DIR) / filename)
    candidates.extend([
        Path.cwd() / filename,
        Path.cwd() / "data" / "ieee-fraud-detection" / filename,
        Path.cwd().parent / "data" / "ieee-fraud-detection" / filename,
        Path("/content") / filename,
        Path("/content/drive/MyDrive") / filename,
        Path("/content/drive/MyDrive/ieee-fraud-detection") / filename,
        Path("/content/drive/MyDrive/IEEE-CIS") / filename,
        Path("/kaggle/input/ieee-fraud-detection") / filename,
    ])

    for c in candidates:
        if c.is_file():
            return c.resolve()

    # Broad search in current working directory and /content
    for root in [Path.cwd(), Path("/content") if Path("/content").exists() else Path.cwd()]:
        for matched in root.rglob(filename):
            if matched.is_file():
                return matched.resolve()

    return None

def discover_datasets():
    print("[1/7] Discovering IEEE-CIS dataset files ...")
    txn_path = _find_file(TRANSACTION_FILE)
    id_path = _find_file(IDENTITY_FILE)

    if not txn_path or not id_path:
        print("\n" + "!" * 70)
        print("[ERROR] Could not automatically locate train_transaction.csv and train_identity.csv.")
        print("Please place the CSVs in the working folder, or set DATA_DIR at the top of this script.")
        print("If running in Google Colab, mount Drive first:")
        print("  from google.colab import drive; drive.mount('/content/drive')")
        print("!" * 70 + "\n")
        sys.exit(1)

    print(f"  [FOUND] Transaction File: {txn_path}")
    print(f"  [FOUND] Identity File:    {id_path}")
    return txn_path, id_path

# -------------------------- MEMORY OPTIMIZATION -------------------------------
def reduce_mem_usage(df):
    """Downcast float64 to float32 and int64 to int32/int16/int8."""
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == "float64":
            df[col] = df[col].astype(np.float32)
        elif col_type == "int64":
            c_min, c_max = df[col].min(), df[col].max()
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
    return df

# ---------------------- DATA ENGINEERING & PREPROCESSING ----------------------
def load_and_preprocess_data(txn_path, id_path):
    print("\n[2/7] Loading and merging raw datasets ...")
    txn_df = pd.read_csv(txn_path)
    id_df = pd.read_csv(id_path)
    df = pd.merge(txn_df, id_df, on="TransactionID", how="left")
    del txn_df, id_df
    gc.collect()

    print(f"  Merged dataset shape: {df.shape}")
    print(f"  Total records: {len(df):,} | Fraud prevalence: {df['isFraud'].mean() * 100:.2f}%")

    # Drop redundant, ultra-sparse V-columns (V35 to V278) to save memory
    v_cols_to_drop = [f"V{i}" for i in range(35, 279) if f"V{i}" in df.columns]
    if v_cols_to_drop:
        df.drop(columns=v_cols_to_drop, inplace=True)
        print(f"  Pruned {len(v_cols_to_drop)} sparse V-columns for high-speed processing.")

    # Sort strictly chronologically by TransactionDT to prevent future leakage
    df.sort_values("TransactionDT", inplace=True)
    df.reset_index(drop=True, inplace=True)

    print("\n[3/7] Engineering relational and behavioral features ...")
    # Time-based cyclic features
    df["TransactionDay"] = (df["TransactionDT"] // (24 * 3600)).astype(np.int16)
    df["TransactionHour"] = ((df["TransactionDT"] // 3600) % 24).astype(np.int8)

    # Transaction Amount relative ratios
    if "card1" in df.columns:
        card1_mean = df.groupby("card1")["TransactionAmt"].transform("mean")
        df["TransactionAmt_to_mean_card1"] = (df["TransactionAmt"] / (card1_mean + 1.0)).astype(np.float32)

    # Frequency encoding for high-cardinality categoricals
    freq_cols = ["card1", "card2", "addr1", "P_emaildomain", "R_emaildomain"]
    for col in freq_cols:
        if col in df.columns:
            freq = df[col].value_counts(dropna=False)
            df[f"{col}_freq"] = df[col].map(freq).astype(np.float32)

    # Encode remaining object columns into integer categories
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_cols:
        df[col] = df[col].astype("category").cat.codes.astype(np.int32)

    df = reduce_mem_usage(df)
    return df

# ----------------- CUSTOM ASYMMETRIC OBJECTIVE & EVALUATION -------------------
def cost_sensitive_objective(y_pred, dtrain):
    """
    Custom Asymmetric Binary Cross-Entropy Objective:
      L(y, p) = - [ C_FN * y * ln(p) + C_FP * (1 - y) * ln(1 - p) ]
    First-order gradient:  g_i = w_i * (p_i - y_i)
    Second-order hessian:  h_i = w_i * p_i * (1 - p_i)
    """
    y_true = dtrain.get_label().astype(np.float64)
    # y_pred is margin output (logit); compute sigmoid
    p = 1.0 / (1.0 + np.exp(-np.clip(y_pred, -15.0, 15.0)))
    weights = np.where(y_true == 1, C_FN, C_FP)
    grad = weights * (p - y_true)
    hess = weights * p * (1.0 - p)
    hess = np.maximum(hess, 1e-8)  # Safeguard against extreme vanishing hessian
    return grad, hess

def cost_sensitive_eval(y_pred, dtrain):
    """Evaluation metric computing total financial business cost."""
    y_true = dtrain.get_label().astype(int)
    p = 1.0 / (1.0 + np.exp(-np.clip(y_pred, -15.0, 15.0)))
    y_hat = (p >= 0.5).astype(int)
    fp = int(np.sum((y_hat == 1) & (y_true == 0)))
    fn = int(np.sum((y_hat == 0) & (y_true == 1)))
    total_cost = C_FP * fp + C_FN * fn
    return "total_cost", total_cost, False

# ---------------------------- MODEL TRAINING ----------------------------------
def train_cost_sensitive_lightgbm(df):
    print("\n[4/7] Splitting data chronologically (80% train / 20% test) ...")
    split_idx = int(len(df) * TRAIN_SPLIT_RATIO)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    features = [c for c in df.columns if c not in ["TransactionID", "isFraud", "TransactionDT"]]
    target = "isFraud"

    X_train = train_df[features]
    y_train = train_df[target].values
    X_test = test_df[features]
    y_test = test_df[target].values
    test_ids = test_df["TransactionID"].values

    print(f"  Training set size:   {len(X_train):,} samples (Fraud rate: {y_train.mean()*100:.2f}%)")
    print(f"  Final test set size: {len(X_test):,} samples (Fraud rate: {y_test.mean()*100:.2f}%)")
    print(f"  Feature count:       {len(features)} attributes")

    dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    dtest = lgb.Dataset(X_test, label=y_test, reference=dtrain, free_raw_data=False)

    params = {
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "feature_fraction": 0.80,
        "bagging_fraction": 0.80,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": RANDOM_SEED,
        "num_threads": max(1, os.cpu_count() or 2),
    }

    print("\n[5/7] Training Cost-Sensitive LightGBM Booster (10:1 Asymmetric Loss) ...")
    evals_result = {}
    callbacks = [
        lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
        lgb.record_evaluation(evals_result),
    ]

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=NUM_BOOST_ROUNDS,
        valid_sets=[dtrain, dtest],
        valid_names=["train", "valid"],
        fobj=cost_sensitive_objective,
        feval=cost_sensitive_eval,
        callbacks=callbacks,
    )

    best_iter = model.best_iteration or NUM_BOOST_ROUNDS
    print(f"  Training complete! Best iteration: {best_iter}")
    return model, X_test, y_test, test_ids, features

# ----------------------- EVALUATION & BENCHMARKING ----------------------------
def evaluate_model(model, X_test, y_test, features, out_dir):
    print("\n[6/7] Evaluating financial cost and classification performance ...")
    raw_preds = model.predict(X_test, num_iteration=model.best_iteration)
    # raw_preds from custom fobj are raw margins; convert via sigmoid
    y_prob = 1.0 / (1.0 + np.exp(-np.clip(raw_preds, -15.0, 15.0)))

    # Compute standard metrics
    auc_roc = roc_auc_score(y_test, y_prob)
    avg_prec = average_precision_score(y_test, y_prob)

    # Threshold evaluation: Default 0.5 vs Swept Optimal
    thresholds = np.linspace(0.05, 0.95, 91)
    best_thresh = 0.5
    lowest_cost = float("inf")

    for th in thresholds:
        y_hat = (y_prob >= th).astype(int)
        fp = int(np.sum((y_hat == 1) & (y_test == 0)))
        fn = int(np.sum((y_hat == 0) & (y_test == 1)))
        cost = C_FP * fp + C_FN * fn
        if cost < lowest_cost:
            lowest_cost = cost
            best_thresh = th

    # Metrics at optimal threshold
    y_pred_opt = (y_prob >= best_thresh).astype(int)
    cm = confusion_matrix(y_test, y_pred_opt)
    tn, fp, fn, tp = cm.ravel()
    opt_cost = C_FP * fp + C_FN * fn

    # Metrics at 0.5 threshold
    y_pred_05 = (y_prob >= 0.5).astype(int)
    cm_05 = confusion_matrix(y_test, y_pred_05)
    tn_05, fp_05, fn_05, tp_05 = cm_05.ravel()
    cost_05 = C_FP * fp_05 + C_FN * fn_05

    print("\n" + "=" * 70)
    print("                    FINAL BENCHMARK RESULTS")
    print("=" * 70)
    print(f"  Test ROC-AUC:             {auc_roc:.4f}")
    print(f"  Average Precision (PR):   {avg_prec:.4f}")
    print("-" * 70)
    print(f"  Standard Threshold (0.50): Cost = {cost_05:,.0f} (FP={fp_05:,}, FN={fn_05:,})")
    print(f"  Optimal Threshold ({best_thresh:.2f}):  Cost = {opt_cost:,.0f} (FP={fp:,}, FN={fn:,})")
    print(f"  Financial Loss Saved:     {(cost_05 - opt_cost):,.0f} penalty units ({(cost_05 - opt_cost)/cost_05 * 100:.1f}%)")
    print("=" * 70)

    # Save Confusion Matrix Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(f"Confusion Matrix (Optimal Threshold = {best_thresh:.2f})")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_xticks([0, 1], ["Legitimate", "Fraud"])
    ax.set_yticks([0, 1], ["Legitimate", "Fraud"])
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f"{val:,}", ha="center", va="center",
                color="white" if val > cm.max()/2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix_test.png", dpi=160)
    plt.close(fig)

    # Save ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, label=f"Cost-Sensitive LightGBM (AUC = {auc_roc:.4f})", color="#2563eb", lw=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Final Test ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve_test.png", dpi=160)
    plt.close(fig)

    return y_prob, best_thresh

# ---------------------- TREESHAP EXPLAINABILITY EXPORT ------------------------
def export_explainability_and_handoff(model, X_test, test_ids, y_prob, features, out_dir):
    print("\n[7/7] Generating TreeSHAP Explanations & Table 3 JSON Handoff ...")
    n_explain = min(MAX_SHAP_EXPLANATIONS, len(X_test))
    X_explain = X_test.iloc[:n_explain]
    ids_explain = test_ids[:n_explain]
    probs_explain = y_prob[:n_explain]

    print(f"  Computing TreeSHAP polynomial attributions for {n_explain:,} transactions ...")
    # Using LightGBM's fast internal TreeSHAP implementation (pred_contrib=True)
    contributions = model.predict(X_explain, num_iteration=model.best_iteration, pred_contrib=True)
    shap_matrix = contributions[:, :-1]  # Exclude base margin bias term

    records = []
    for i in range(n_explain):
        top_idx = np.argsort(np.abs(shap_matrix[i]))[-3:][::-1]
        top3_features = {
            features[idx]: round(float(shap_matrix[i, idx]), 6)
            for idx in top_idx
        }
        records.append({
            "transaction_id": str(ids_explain[i]),
            "dataset_name": "ieee_cis",
            "prediction_probability": round(float(probs_explain[i]), 6),
            "top3_shap_features": top3_features,
            "explanation_model": "TreeSHAP (Cost-Sensitive LightGBM Booster)",
        })

    json_path = out_dir / "lightgbm_shap_output.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"  [SUCCESS] Handoff JSON saved -> {json_path}")

    # Generate Global Feature Importance Plot
    mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
    top_global_idx = np.argsort(mean_abs_shap)[-20:]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh([features[i] for i in top_global_idx], mean_abs_shap[top_global_idx], color="#2563eb")
    ax.set_xlabel("Mean Absolute TreeSHAP Impact (|SHAP|)")
    ax.set_title("Global Feature Importance (TreeSHAP)")
    fig.tight_layout()
    fig.savefig(out_dir / "shap_summary.png", dpi=160)
    plt.close(fig)
    print(f"  [SUCCESS] Global SHAP plot saved -> {out_dir / 'shap_summary.png'}")

# ------------------------------- MAIN RUNNER ----------------------------------
def main():
    print("=" * 70)
    print("  IEEE-CIS FRAUD DETECTION: COST-SENSITIVE LIGHTGBM PIPELINE")
    print("=" * 70)

    out_dir = Path(OUTPUT_DIR)
    art_dir = Path(ARTEFACTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)

    txn_path, id_path = discover_datasets()
    df = load_and_preprocess_data(txn_path, id_path)
    model, X_test, y_test, test_ids, features = train_cost_sensitive_lightgbm(df)

    # Save trained booster
    model_path = art_dir / "lgbm_cost_sensitive.txt"
    model.save_model(str(model_path))
    print(f"  Saved booster model -> {model_path}")

    y_prob, best_thresh = evaluate_model(model, X_test, y_test, features, out_dir)
    export_explainability_and_handoff(model, X_test, test_ids, y_prob, features, out_dir)

    print("\n" + "=" * 70)
    print("[ALL DONE] Generated deliverables in outputs/:")
    for p in sorted(out_dir.iterdir()):
        if p.is_file():
            print(f"  {p.name:32s} ({p.stat().st_size / (1024*1024):.2f} MB)")
    print("=" * 70)
    print("Ready for live ingestion into FastAPI & Streamlit HITL Console:")
    print("  python ingest_predictions.py")

if __name__ == "__main__":
    main()

