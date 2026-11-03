"""
phase3_explainability.py -- SHAP Explainability & JSON Handoff
Samsung Capstone Project - Table 1 (Team LightGBM)

Generates per-transaction top-3 SHAP features as JSON and a summary plot.
"""

import json
import sys
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTEFACTS_DIR = Path(__file__).resolve().parent / "artefacts"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def load_model():
    """Load trained LightGBM booster from Phase 2."""
    model_path = ARTEFACTS_DIR / "lgbm_model.pkl"
    if not model_path.is_file():
        print(f"[ERROR] Model not found: {model_path}")
        print("[HINT]  Run phase2_cost_sensitive_training.py first.")
        sys.exit(1)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"[INFO] Loaded model from {model_path}")
    return model


def load_test_data():
    """Load pickled test DataFrame from Phase 1."""
    test_pkl = ARTEFACTS_DIR / "test.pkl"
    if not test_pkl.is_file():
        print(f"[ERROR] Test data not found: {test_pkl}")
        print("[HINT]  Run phase1_data_engineering.py first.")
        sys.exit(1)
    with open(test_pkl, "rb") as f:
        test_df = pickle.load(f)
    print(f"[INFO] Loaded test data: {test_df.shape}")
    return test_df


def compute_shap_values(model, X, feature_names):
    """Compute SHAP values using TreeExplainer."""
    print("[INFO] Initialising SHAP TreeExplainer ...")
    explainer = shap.TreeExplainer(model)
    print(f"[INFO] Computing SHAP values for {X.shape[0]:,} samples ...")
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    print(f"[INFO] SHAP values shape: {shap_values.shape}")
    return shap_values, explainer


def build_shap_json(transaction_ids, prediction_probs, shap_values, feature_names):
    """Build per-transaction SHAP explanation records as flat dicts."""
    print("[INFO] Building per-transaction SHAP JSON ...")
    records = []
    for i in range(len(transaction_ids)):
        sv = shap_values[i]
        top_indices = np.argsort(np.abs(sv))[-3:][::-1]
        top3 = {
            feature_names[idx]: round(float(sv[idx]), 6)
            for idx in top_indices
        }
        records.append({
            "transaction_id": str(int(transaction_ids[i])),
            "prediction_probability": round(float(prediction_probs[i]), 6),
            "top3_shap_features": top3,
        })
    return records


def save_shap_json(records):
    """Write SHAP explanations to JSON."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUTS_DIR / "lightgbm_shap_output.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved SHAP JSON  -> {json_path}  ({len(records):,} records)")


def save_summary_plot(shap_values, X, feature_names):
    """Generate and save a global SHAP beeswarm summary plot."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUTS_DIR / "shap_summary.png"
    print("[INFO] Generating SHAP summary plot ...")
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X, feature_names=feature_names,
                      show=False, max_display=20, plot_size=(12, 10))
    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved SHAP plot  -> {png_path}")


def print_summary(records, shap_values, feature_names):
    """Print summary statistics."""
    probs = [r["prediction_probability"] for r in records]
    print("\n" + "=" * 60)
    print("  PHASE 3 -- EXPLAINABILITY SUMMARY")
    print("=" * 60)
    print(f"  Transactions explained : {len(records):,}")
    print(f"  Prediction prob range  : [{min(probs):.6f}, {max(probs):.6f}]")
    print(f"  Mean prediction prob   : {np.mean(probs):.6f}")

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    top_global = np.argsort(mean_abs_shap)[-5:][::-1]
    print(f"\n  Top-5 global features (mean |SHAP|):")
    for rank, idx in enumerate(top_global, 1):
        print(f"    {rank}. {feature_names[idx]:30s}  {mean_abs_shap[idx]:.6f}")
    print("=" * 60 + "\n")


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  PHASE 3 -- Explainability & Handoff")
    print("=" * 60 + "\n")

    model = load_model()
    test_df = load_test_data()

    drop_cols = ["TransactionID", "isFraud"]
    feature_names = [c for c in test_df.columns if c not in drop_cols]
    X_test = test_df[feature_names].values
    transaction_ids = test_df["TransactionID"].values

    raw_preds = model.predict(X_test, num_iteration=model.best_iteration)
    proba = 1.0 / (1.0 + np.exp(-raw_preds))

    shap_values, explainer = compute_shap_values(model, X_test, feature_names)

    records = build_shap_json(transaction_ids, proba, shap_values, feature_names)
    save_shap_json(records)
    save_summary_plot(shap_values, X_test, feature_names)
    print_summary(records, shap_values, feature_names)

    print("[SUCCESS] Phase 3 complete. Outputs ready for Table 3 handoff.\n")


if __name__ == "__main__":
    main()
