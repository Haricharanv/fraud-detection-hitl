# ============================================================================
# IEEE-CIS FRAUD DETECTION — TEMPORAL CATBOOST + LIGHTGBM HYBRID
# Single-cell replacement for the previous LightGBM++ pipeline.
#
# Design principles:
#   1. Chronological, leakage-controlled features (historical counts only).
#   2. Train on unweighted log-loss; do NOT force a 10x positive class weight.
#   3. Tune LightGBM for average precision on a chronological validation block.
#   4. Train CatBoost with ordered categorical handling as a complementary model.
#   5. Calibrate probabilities on a later, separate chronological block.
#   6. Select the operating threshold using C_FP*FP + C_FN*FN on another block.
#   7. Keep the final chronological test block untouched until final evaluation.
#   8. Export metrics, models, predictions, and local LightGBM TreeSHAP JSON.
#
# Expected input files (same as the uploaded notebook):
#   train_transaction.csv
#   train_identity.csv
#
# If your files are in Google Drive, mount Drive first or set DATA_DIR below.
# ============================================================================

import os
import sys
import gc
import json
import math
import warnings
import subprocess
import importlib
from pathlib import Path

warnings.filterwarnings("ignore")

# ----------------------------- USER CONFIG ----------------------------------
DATA_DIR = None                 # Example: "/content/drive/MyDrive/IEEE-CIS"
TRANSACTION_FILE = "train_transaction.csv"
IDENTITY_FILE = "train_identity.csv"

C_FP = 1.0                      # Cost of incorrectly flagging a legitimate txn
C_FN = 10.0                     # Cost of missing a fraudulent txn

TRAIN_FRACTION = 0.65           # Earliest transactions: model fitting
VALID_FRACTION_END = 0.75      # Next 10%: Optuna + early stopping
CALIBRATION_FRACTION_END = 0.80 # Next 5%: split into Platt-fit and threshold blocks
# Latest 20%: final test; never used for training/model/threshold selection

OPTUNA_TRIALS = 8               # Increase to 15-25 after the first successful run
LGB_MAX_ROUNDS = 2200
LGB_EARLY_STOPPING = 120
CATBOOST_ITERATIONS = 1800
CATBOOST_EARLY_STOPPING = 120
RANDOM_SEED = 42
MAX_EXPLANATIONS = 10000
DROP_V_COLUMNS = True           # Match the feature-pruning choice in your original notebook
MISSINGNESS_DROP_THRESHOLD = 0.995
OUTPUT_DIR = "fraud_hybrid_outputs"

# -------------------------- INSTALL DEPENDENCIES -----------------------------
def _ensure_package(package_name, import_name=None):
    import_name = import_name or package_name
    try:
        importlib.import_module(import_name)
    except ImportError:
        print(f"[SETUP] Installing {package_name}...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", package_name]
        )

_ensure_package("numpy")
_ensure_package("pandas")
_ensure_package("scikit-learn", "sklearn")
_ensure_package("lightgbm")
_ensure_package("catboost")
_ensure_package("optuna")
_ensure_package("matplotlib")

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_curve,
    auc,
)
from sklearn.linear_model import LogisticRegression

np.random.seed(RANDOM_SEED)
optuna.logging.set_verbosity(optuna.logging.WARNING)

CPU_COUNT = os.cpu_count() or 2
N_THREADS = max(1, min(8, CPU_COUNT))

# --------------------------- DATASET DISCOVERY -------------------------------
def _find_file(filename):
    candidates = []
    if DATA_DIR:
        candidates.append(Path(DATA_DIR) / filename)
    candidates.extend([
        Path.cwd() / filename,
        Path("/content") / filename,
        Path("/content/drive/MyDrive") / filename,
        Path("/kaggle/input") / filename,
    ])

    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen and candidate.is_file():
            return candidate.resolve()
        seen.add(key)

    # Search common notebook roots for files stored in subfolders.
    roots = []
    for root in [Path.cwd(), Path("/content"), Path("/kaggle/input")]:
        if root.exists() and root.is_dir() and str(root) not in [str(x) for x in roots]:
            roots.append(root)
    for root in roots:
        try:
            for path in root.rglob(filename):
                if path.is_file():
                    return path.resolve()
        except (PermissionError, OSError):
            continue
    return None

transaction_path = _find_file(TRANSACTION_FILE)
identity_path = _find_file(IDENTITY_FILE)

# In Colab, mount Drive only when the expected files were not already found.
if transaction_path is None:
    try:
        from google.colab import drive
        print("[INFO] Dataset not found in the current directory. Mounting Google Drive...")
        drive.mount("/content/drive", force_remount=False)
        transaction_path = _find_file(TRANSACTION_FILE)
        identity_path = identity_path or _find_file(IDENTITY_FILE)
    except Exception:
        pass

if transaction_path is None:
    raise FileNotFoundError(
        f"Could not find {TRANSACTION_FILE}. Put it in the notebook working folder, "
        "mount Google Drive, or set DATA_DIR at the top of this cell."
    )
if identity_path is None:
    raise FileNotFoundError(
        f"Could not find {IDENTITY_FILE}. Your original pipeline merged both "
        "transaction and identity files; provide both files for a comparable run. "
        "Set DATA_DIR if they are stored in a subfolder."
    )
print(f"[INFO] Transaction file: {transaction_path}")
print(f"[INFO] Identity file:    {identity_path}")

# ----------------------------- MEMORY UTILS ---------------------------------
def reduce_mem_usage(frame):
    """Downcast numeric columns without changing categorical text values."""
    for col in frame.columns:
        dtype = frame[col].dtype
        if pd.api.types.is_float_dtype(dtype):
            frame[col] = pd.to_numeric(frame[col], downcast="float")
        elif pd.api.types.is_integer_dtype(dtype):
            frame[col] = pd.to_numeric(frame[col], downcast="integer")
    return frame

# ------------------------------- LOAD DATA ----------------------------------
print("\n[1/9] Loading the IEEE-CIS datasets...")
transaction_header = pd.read_csv(transaction_path, nrows=0).columns.tolist()
if DROP_V_COLUMNS:
    transaction_usecols = [c for c in transaction_header if not str(c).startswith("V")]
else:
    transaction_usecols = transaction_header

transactions = pd.read_csv(
    transaction_path,
    usecols=transaction_usecols,
    low_memory=False,
)

identity = pd.read_csv(identity_path, low_memory=False)
if "TransactionID" not in identity.columns:
    raise ValueError("train_identity.csv does not contain TransactionID.")
if identity["TransactionID"].duplicated().any():
    print("[WARNING] Duplicate TransactionID values exist in train_identity.csv; dropping duplicate rows.")
    identity = identity.drop_duplicates("TransactionID", keep="first")
df = transactions.merge(identity, on="TransactionID", how="left", validate="one_to_one")
del transactions, identity

gc.collect()

required = {"TransactionID", "TransactionDT", "isFraud"}
missing_required = required.difference(df.columns)
if missing_required:
    raise ValueError(f"Required columns missing from merged data: {sorted(missing_required)}")

if df["isFraud"].isna().any():
    raise ValueError("The isFraud target contains missing values.")

df["isFraud"] = df["isFraud"].astype(np.int8)
if set(pd.unique(df["isFraud"])) - {0, 1}:
    raise ValueError("isFraud must contain only binary values 0 and 1.")
df = reduce_mem_usage(df)
print(f"[INFO] Merged data shape: {df.shape}")
print(f"[INFO] Fraud prevalence: {df['isFraud'].mean() * 100:.3f}%")

# ---------------------- TEMPORAL ORDER + SPLIT CUTS -------------------------
print("\n[2/9] Sorting chronologically and defining leakage-safe time blocks...")
df = df.sort_values("TransactionDT", kind="mergesort").reset_index(drop=True)
n_rows = len(df)
if n_rows < 10000:
    raise ValueError("Dataset is unexpectedly small. Expected the full IEEE-CIS training dataset.")

# Use unique timestamp cutoffs so transactions with exactly the same timestamp
# are not split across train/validation/calibration/test blocks.
unique_times = np.unique(df["TransactionDT"].to_numpy())
if len(unique_times) < 10:
    raise ValueError("TransactionDT has too few unique values for a chronological split.")
cut_1 = unique_times[min(int(len(unique_times) * TRAIN_FRACTION), len(unique_times) - 1)]
cut_2 = unique_times[min(int(len(unique_times) * VALID_FRACTION_END), len(unique_times) - 1)]
cut_3 = unique_times[min(int(len(unique_times) * CALIBRATION_FRACTION_END), len(unique_times) - 1)]
if not (cut_1 < cut_2 < cut_3):
    raise ValueError("Could not create distinct chronological split boundaries. Check TransactionDT.")

time_values = df["TransactionDT"].to_numpy()
train_mask = time_values < cut_1
valid_mask = (time_values >= cut_1) & (time_values < cut_2)
cal_mask = (time_values >= cut_2) & (time_values < cut_3)
test_mask = time_values >= cut_3

split_counts = {
    "train": int(train_mask.sum()),
    "validation": int(valid_mask.sum()),
    "calibration": int(cal_mask.sum()),
    "test": int(test_mask.sum()),
}
if min(split_counts.values()) == 0:
    raise ValueError(f"At least one chronological split is empty: {split_counts}")
print("[INFO] Chronological row counts:", split_counts)

# ---------------------- LEAKAGE-SAFE FEATURE ENGINEERING --------------------
print("\n[3/9] Creating strictly historical relational and time features...")
# The frame is sorted by TransactionDT before these calculations. cumcount()
# counts only rows observed before the current row; it never uses a future row's
# transaction count. This replaces the original full-dataset transform('count').
df["TransactionDay"] = (df["TransactionDT"].astype(np.int64) // 86400).astype(np.int32)
seconds_in_day = (df["TransactionDT"].astype(np.int64) % 86400).astype(np.int32)
df["tx_hour"] = (seconds_in_day // 3600).astype(np.int8)
df["tx_hour_sin"] = np.sin(2.0 * np.pi * df["tx_hour"].astype(np.float32) / 24.0).astype(np.float32)
df["tx_hour_cos"] = np.cos(2.0 * np.pi * df["tx_hour"].astype(np.float32) / 24.0).astype(np.float32)
df["tx_dayofweek_proxy"] = (df["TransactionDay"] % 7).astype(np.int8)

# Counts for the same entity based only on earlier rows.
history_entities = [
    "card1", "card2", "card3", "card5", "addr1", "addr2",
    "P_emaildomain", "R_emaildomain", "DeviceInfo", "id_30", "id_31",
]
for col in history_entities:
    if col in df.columns:
        out_col = f"hist_{col}_prior_count"
        df[out_col] = (
            df.groupby(col, sort=False, dropna=False).cumcount().astype(np.int32)
        )

# Past same-card daily activity; the current transaction is excluded.
if "card1" in df.columns:
    df["hist_card1_same_day_prior_count"] = (
        df.groupby(["card1", "TransactionDay"], sort=False, dropna=False)
          .cumcount().astype(np.int32)
    )

# A small number of pairwise relational counts; these are also past-only.
pair_groups = [
    ("card1", "P_emaildomain", "hist_card1_email_prior_count"),
    ("card1", "addr1", "hist_card1_addr1_prior_count"),
]
for left, right, out_col in pair_groups:
    if left in df.columns and right in df.columns:
        df[out_col] = (
            df.groupby([left, right], sort=False, dropna=False)
              .cumcount().astype(np.int32)
        )

if "TransactionAmt" in df.columns:
    amt = pd.to_numeric(df["TransactionAmt"], errors="coerce").astype(np.float32)
    df["TransactionAmt_log1p"] = np.log1p(amt.clip(lower=0)).astype(np.float32)
    df["TransactionAmt_round2"] = np.round(amt, 2).astype(np.float32)

for col in ["dist1", "dist2"]:
    if col in df.columns:
        val = pd.to_numeric(df[col], errors="coerce").astype(np.float32)
        df[f"{col}_log1p"] = np.log1p(val.clip(lower=0)).astype(np.float32)

# ------------------------ CHOOSE CATEGORICAL FIELDS -------------------------
# These are categorical in meaning even when the CSV stores them as integers.
known_categorical = {
    "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "P_emaildomain", "R_emaildomain", "DeviceType",
    "DeviceInfo", "id_12", "id_15", "id_16", "id_23", "id_27", "id_28",
    "id_29", "id_30", "id_31", "id_33", "id_34", "id_35", "id_36",
    "id_37", "id_38", "M1", "M2", "M3", "M4", "M5", "M6", "M7",
    "M8", "M9",
}
object_cols = [c for c in df.columns if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])]
cat_candidate_cols = sorted((set(object_cols) | known_categorical).intersection(df.columns))
cat_candidate_cols = [c for c in cat_candidate_cols if c not in {"TransactionID", "isFraud", "TransactionDT"}]

# ----------------------- TRAIN-ONLY CATEGORICAL CODES -----------------------
# Fit category dictionaries only on the earliest training period. Validation,
# calibration, and test categories not observed in training map to code 0.
print("[INFO] Encoding categorical columns using training-period vocabularies...")
cat_feature_names = []
for col in cat_candidate_cols:
    values = df[col].astype("string").fillna("__MISSING__")
    train_values = values.loc[train_mask]
    known_values = pd.Index(pd.unique(train_values))
    mapping = {value: i + 1 for i, value in enumerate(known_values)}
    df[col] = values.map(mapping).fillna(0).astype(np.int32)
    cat_feature_names.append(col)
    del values, train_values, known_values, mapping

# Ensure all model inputs are numeric, replace infinities, and use compact dtypes.
for col in df.columns:
    if col in {"TransactionID", "isFraud", "TransactionDT"} or col in cat_feature_names:
        continue
    if pd.api.types.is_numeric_dtype(df[col]):
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype(np.float32)
        elif pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")
    else:
        # Any unexpected text field is encoded with the same train-only rule.
        values = df[col].astype("string").fillna("__MISSING__")
        train_values = values.loc[train_mask]
        mapping = {value: i + 1 for i, value in enumerate(pd.unique(train_values))}
        df[col] = values.map(mapping).fillna(0).astype(np.int32)
        cat_feature_names.append(col)
        del values, train_values, mapping

df.replace([np.inf, -np.inf], np.nan, inplace=True)

# ----------------------- REMOVE UNINFORMATIVE FIELDS ------------------------
# Drop features that are almost entirely missing or constant in TRAIN only.
excluded_cols = {"TransactionID", "TransactionDT", "isFraud"}
feature_candidates = [c for c in df.columns if c not in excluded_cols]
train_view = df.loc[train_mask, feature_candidates]
drop_features = []
for col in feature_candidates:
    missing_rate = float(train_view[col].isna().mean())
    if missing_rate > MISSINGNESS_DROP_THRESHOLD:
        drop_features.append(col)
        continue
    if train_view[col].nunique(dropna=True) <= 1:
        drop_features.append(col)

del train_view
gc.collect()
features = [c for c in feature_candidates if c not in set(drop_features)]
cat_feature_names = [c for c in cat_feature_names if c in features]
cat_feature_indices = [features.index(c) for c in cat_feature_names]
if not features:
    raise ValueError("No usable model features remain after preprocessing.")
print(f"[INFO] Features retained: {len(features)} | categorical: {len(cat_feature_names)} | dropped: {len(drop_features)}")

# --------------------------- BUILD DATA MATRICES ----------------------------
print("\n[4/9] Building time-separated model matrices...")
X_all = df[features]
y_all = df["isFraud"].to_numpy(dtype=np.int8)

X_train = X_all.loc[train_mask].copy()
y_train = y_all[train_mask]
X_valid = X_all.loc[valid_mask].copy()
y_valid = y_all[valid_mask]
X_cal = X_all.loc[cal_mask].copy()
y_cal = y_all[cal_mask]
X_test = X_all.loc[test_mask].copy()
y_test = y_all[test_mask]
del X_all
gc.collect()

test_ids = df.loc[test_mask, "TransactionID"].astype(str).to_numpy()
del df
gc.collect()

# Split the 5% calibration block chronologically: first half fits Platt
# calibration/blend weight; second half selects the cost threshold.
cal_split = max(1, len(X_cal) // 2)
X_cal_fit = X_cal.iloc[:cal_split]
y_cal_fit = y_cal[:cal_split]
X_cal_threshold = X_cal.iloc[cal_split:]
y_cal_threshold = y_cal[cal_split:]

if len(np.unique(y_train)) < 2 or len(np.unique(y_valid)) < 2:
    raise ValueError("Training and validation blocks must each contain both target classes.")
if len(np.unique(y_cal_fit)) < 2 or len(np.unique(y_cal_threshold)) < 2:
    raise ValueError(
        "Calibration halves do not both contain fraud and non-fraud examples. "
        "Use a larger calibration fraction or verify the temporal data distribution."
    )

print(f"[INFO] Training fraud rate:     {y_train.mean() * 100:.3f}%")
print(f"[INFO] Validation fraud rate:   {y_valid.mean() * 100:.3f}%")
print(f"[INFO] Calibration fraud rate:  {y_cal.mean() * 100:.3f}%")
print(f"[INFO] Final test fraud rate:   {y_test.mean() * 100:.3f}%")

# --------------------------- LIGHTGBM DATASETS ------------------------------
lgb_train = lgb.Dataset(
    X_train,
    label=y_train,
    categorical_feature=cat_feature_names,
    free_raw_data=False,
)
lgb_valid = lgb.Dataset(
    X_valid,
    label=y_valid,
    categorical_feature=cat_feature_names,
    reference=lgb_train,
    free_raw_data=False,
)

def lgb_average_precision_eval(predictions, dataset):
    labels = dataset.get_label().astype(np.int8)
    score = average_precision_score(labels, predictions)
    return "average_precision", float(score), True

# ----------------------- OPTUNA LIGHTGBM SEARCH -----------------------------
print("\n[5/9] Tuning a regularized LightGBM model (average precision objective)...")

def lgb_objective(trial):
    params = {
        "objective": "binary",
        "metric": "None",
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.07, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 63),
        "max_depth": trial.suggest_int("max_depth", 4, 9),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 100, 900),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.70, 0.95),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.70, 0.95),
        "bagging_freq": 1,
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-4, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1.0, 60.0, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 0.20),
        "max_bin": 127,
        "max_cat_to_onehot": 4,
        "cat_smooth": trial.suggest_float("cat_smooth", 5.0, 100.0),
        "cat_l2": trial.suggest_float("cat_l2", 5.0, 50.0),
        "verbosity": -1,
        "num_threads": N_THREADS,
        "seed": RANDOM_SEED,
        "feature_fraction_seed": RANDOM_SEED,
        "bagging_seed": RANDOM_SEED,
        "data_random_seed": RANDOM_SEED,
        "deterministic": True,
        "force_col_wise": True,
    }
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=LGB_MAX_ROUNDS,
        valid_sets=[lgb_valid],
        feval=lgb_average_precision_eval,
        callbacks=[
            lgb.early_stopping(LGB_EARLY_STOPPING, first_metric_only=True, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    prediction = model.predict(X_valid, num_iteration=model.best_iteration)
    score = average_precision_score(y_valid, prediction)
    trial.set_user_attr("best_iteration", int(model.best_iteration or LGB_MAX_ROUNDS))
    del model, prediction
    gc.collect()
    return float(score)

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
)
study.optimize(lgb_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)

best_lgb_params = dict(study.best_params)
best_lgb_params.update({
    "objective": "binary",
    "metric": "None",
    "boosting_type": "gbdt",
    "bagging_freq": 1,
    "max_bin": 127,
    "max_cat_to_onehot": 4,
    "verbosity": -1,
    "num_threads": N_THREADS,
    "seed": RANDOM_SEED,
    "feature_fraction_seed": RANDOM_SEED,
    "bagging_seed": RANDOM_SEED,
    "data_random_seed": RANDOM_SEED,
    "deterministic": True,
    "force_col_wise": True,
})
print(f"[INFO] Best validation average precision: {study.best_value:.6f}")
print(f"[INFO] Best LightGBM parameters: {best_lgb_params}")

print("[INFO] Fitting final LightGBM model with early stopping...")
lgb_model = lgb.train(
    best_lgb_params,
    lgb_train,
    num_boost_round=LGB_MAX_ROUNDS,
    valid_sets=[lgb_valid],
    feval=lgb_average_precision_eval,
    callbacks=[
        lgb.early_stopping(LGB_EARLY_STOPPING, first_metric_only=True, verbose=True),
        lgb.log_evaluation(period=200),
    ],
)

# ---------------------------- CATBOOST MODEL --------------------------------
print("\n[6/9] Training complementary CatBoost categorical model...")
cat_model = None
catboost_available = True
try:
    cat_model = CatBoostClassifier(
        iterations=CATBOOST_ITERATIONS,
        learning_rate=0.035,
        depth=7,
        loss_function="Logloss",
        eval_metric="AUC",
        l2_leaf_reg=10.0,
        random_strength=1.0,
        bootstrap_type="Bernoulli",
        subsample=0.80,
        one_hot_max_size=10,
        random_seed=RANDOM_SEED,
        thread_count=N_THREADS,
        allow_writing_files=False,
        verbose=200,
    )
    cat_model.fit(
        X_train,
        y_train,
        cat_features=cat_feature_indices,
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=CATBOOST_EARLY_STOPPING,
        use_best_model=True,
        verbose=200,
    )
except Exception as exc:
    catboost_available = False
    cat_model = None
    print("[WARNING] CatBoost training failed; continuing with LightGBM only.")
    print(f"[WARNING] CatBoost error: {type(exc).__name__}: {exc}")

# ---------------------- PREDICTION + MODEL BLENDING -------------------------
def predict_catboost(model, X):
    if model is None:
        return None
    return model.predict_proba(X)[:, 1].astype(np.float64)

p_lgb_cal_fit = lgb_model.predict(X_cal_fit, num_iteration=lgb_model.best_iteration)
p_lgb_cal_threshold = lgb_model.predict(X_cal_threshold, num_iteration=lgb_model.best_iteration)
p_lgb_test = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)

p_cb_cal_fit = predict_catboost(cat_model, X_cal_fit)
p_cb_cal_threshold = predict_catboost(cat_model, X_cal_threshold)
p_cb_test = predict_catboost(cat_model, X_test)

# Select one blend coefficient on the probability-calibration-fit block only.
# This is a small, low-dimensional selection, not a second high-capacity learner.
if catboost_available and p_cb_cal_fit is not None:
    blend_candidates = np.linspace(0.0, 1.0, 11)  # weight assigned to LightGBM
    blend_scores = []
    for w in blend_candidates:
        blended = w * p_lgb_cal_fit + (1.0 - w) * p_cb_cal_fit
        blended = np.clip(blended, 1e-7, 1.0 - 1e-7)
        blend_scores.append(log_loss(y_cal_fit, blended, labels=[0, 1]))
    best_blend_weight = float(blend_candidates[int(np.argmin(blend_scores))])
else:
    best_blend_weight = 1.0

if catboost_available and p_cb_cal_threshold is not None:
    p_cal_threshold_raw = (
        best_blend_weight * p_lgb_cal_threshold
        + (1.0 - best_blend_weight) * p_cb_cal_threshold
    )
    p_test_raw = best_blend_weight * p_lgb_test + (1.0 - best_blend_weight) * p_cb_test
else:
    p_cal_threshold_raw = p_lgb_cal_threshold
    p_test_raw = p_lgb_test

p_cal_fit_raw = (
    best_blend_weight * p_lgb_cal_fit
    + (1.0 - best_blend_weight) * p_cb_cal_fit
    if catboost_available and p_cb_cal_fit is not None
    else p_lgb_cal_fit
)

# --------------------------- PLATT CALIBRATION -------------------------------
print("\n[7/9] Calibrating probabilities and optimizing the operating threshold...")
def probability_to_logit(probabilities):
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p)).reshape(-1, 1)

platt_model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=RANDOM_SEED)
platt_model.fit(probability_to_logit(p_cal_fit_raw), y_cal_fit)

p_cal_threshold = platt_model.predict_proba(probability_to_logit(p_cal_threshold_raw))[:, 1]
p_test = platt_model.predict_proba(probability_to_logit(p_test_raw))[:, 1]

# Pick threshold using only the second half of the calibration period.
def choose_cost_threshold(y_true, probability, cost_fp=1.0, cost_fn=10.0):
    y_true = np.asarray(y_true, dtype=np.int8)
    probability = np.asarray(probability, dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        raise ValueError("Threshold-selection block must contain both classes.")

    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    positives = int(y_true.sum())
    tp = np.rint(recall[:-1] * positives).astype(np.int64)
    # Count all predicted-positive cases directly. This remains correct even
    # when a threshold produces zero true positives (precision is then zero).
    sorted_probabilities = np.sort(probability)
    predicted_positive = len(probability) - np.searchsorted(
        sorted_probabilities, thresholds, side="left"
    )
    fp = predicted_positive.astype(np.int64) - tp
    fn = positives - tp
    costs = cost_fp * fp + cost_fn * fn

    min_cost = float(np.min(costs))
    tied = np.flatnonzero(np.isclose(costs, min_cost))
    # In a tie, take the highest threshold (fewer alerts).
    best_idx = int(tied[-1])
    best_threshold = float(thresholds[best_idx])
    best_fp = int(fp[best_idx])
    best_fn = int(fn[best_idx])
    best_cost = float(costs[best_idx])

    # Include the valid operating point that flags no transactions.
    no_alert_cost = cost_fn * positives
    if no_alert_cost <= best_cost:
        best_threshold = float(np.nextafter(np.max(probability), np.inf))
        best_fp = 0
        best_fn = positives
        best_cost = float(no_alert_cost)

    return best_threshold, {
        "calibration_fp": best_fp,
        "calibration_fn": best_fn,
        "calibration_cost": best_cost,
    }

optimal_threshold, threshold_calibration_stats = choose_cost_threshold(
    y_cal_threshold,
    p_cal_threshold,
    C_FP,
    C_FN,
)
print(f"[INFO] LightGBM blend weight: {best_blend_weight:.2f}")
print(f"[INFO] Platt calibrated threshold: {optimal_threshold:.6f}")
print(f"[INFO] Calibration-block cost: {threshold_calibration_stats['calibration_cost']:,.0f} "
      f"(FP={threshold_calibration_stats['calibration_fp']:,}, "
      f"FN={threshold_calibration_stats['calibration_fn']:,})")

# ----------------------------- FINAL EVALUATION -----------------------------
print("\n[8/9] Evaluating once on the untouched latest 20% test block...")
def calculate_metrics(y_true, probabilities, threshold, cost_fp=1.0, cost_fn=10.0):
    y_true = np.asarray(y_true, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    metrics = {
        "threshold": float(threshold),
        "transactions": int(len(y_true)),
        "actual_fraud": int(y_true.sum()),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "specificity": float(specificity),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if len(np.unique(y_true)) == 2 else None,
        "average_precision": float(average_precision_score(y_true, probabilities)) if len(np.unique(y_true)) == 2 else None,
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, np.clip(probabilities, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "total_cost": float(cost_fp * fp + cost_fn * fn),
        "cost_per_transaction": float((cost_fp * fp + cost_fn * fn) / len(y_true)),
        "cost_per_1000_transactions": float(1000.0 * (cost_fp * fp + cost_fn * fn) / len(y_true)),
    }
    return metrics, predictions

metrics_opt, yhat_test = calculate_metrics(y_test, p_test, optimal_threshold, C_FP, C_FN)
metrics_050, yhat_050 = calculate_metrics(y_test, p_test, 0.5, C_FP, C_FN)

print("\n" + "=" * 74)
print("FINAL CHRONOLOGICAL TEST RESULTS — HYBRID MODEL")
print("=" * 74)
print(f"Test transactions       : {metrics_opt['transactions']:,}")
print(f"Actual fraud transactions: {metrics_opt['actual_fraud']:,}")
print(f"Selected threshold      : {metrics_opt['threshold']:.6f}")
print(f"ROC-AUC                 : {metrics_opt['roc_auc']:.6f}")
print(f"Average Precision       : {metrics_opt['average_precision']:.6f}")
print(f"Accuracy                : {metrics_opt['accuracy'] * 100:.3f}%")
print(f"Precision               : {metrics_opt['precision'] * 100:.3f}%")
print(f"Recall                  : {metrics_opt['recall'] * 100:.3f}%")
print(f"F1-score                : {metrics_opt['f1'] * 100:.3f}%")
print(f"Specificity             : {metrics_opt['specificity'] * 100:.3f}%")
print(f"True Positives (TP)     : {metrics_opt['TP']:,}")
print(f"False Positives (FP)    : {metrics_opt['FP']:,}")
print(f"False Negatives (FN)    : {metrics_opt['FN']:,}")
print(f"True Negatives (TN)     : {metrics_opt['TN']:,}")
print(f"Total operational cost  : {metrics_opt['total_cost']:,.0f}")
print(f"Cost / transaction      : {metrics_opt['cost_per_transaction']:.6f}")
print("-" * 74)
print(f"At threshold 0.5: FP={metrics_050['FP']:,}, FN={metrics_050['FN']:,}, "
      f"cost={metrics_050['total_cost']:,.0f}")
print("=" * 74)

# Report per-class and confusion-matrix diagnostics without changing threshold.
cm = np.array([
    [metrics_opt["TN"], metrics_opt["FP"]],
    [metrics_opt["FN"], metrics_opt["TP"]],
], dtype=np.int64)

# ------------------------------- EXPORTS ------------------------------------
print("\n[9/9] Exporting models, metrics, test predictions, and explanations...")
out_dir = Path(OUTPUT_DIR)
out_dir.mkdir(parents=True, exist_ok=True)

lgb_model.save_model(str(out_dir / "fraud_lightgbm.txt"))
if cat_model is not None:
    cat_model.save_model(str(out_dir / "fraud_catboost.cbm"))

predictions_df = pd.DataFrame({
    "TransactionID": test_ids,
    "y_true": y_test.astype(int),
    "fraud_probability_calibrated": p_test.astype(float),
    "predicted_isFraud": yhat_test.astype(int),
    "decision_threshold": float(optimal_threshold),
})
predictions_df.to_csv(out_dir / "test_predictions.csv", index=False)

all_metrics = {
    "cost_definition": {"C_FP": C_FP, "C_FN": C_FN, "equation": "C_FP*FP + C_FN*FN"},
    "split_counts": split_counts,
    "split_cutoffs_TransactionDT": {
        "train_end_exclusive": int(cut_1),
        "validation_end_exclusive": int(cut_2),
        "calibration_end_exclusive": int(cut_3),
    },
    "model": {
        "lightgbm_best_iteration": int(lgb_model.best_iteration or 0),
        "catboost_best_iteration": int(cat_model.get_best_iteration()) if cat_model is not None else None,
        "lightgbm_weight_in_blend": float(best_blend_weight),
        "catboost_weight_in_blend": float(1.0 - best_blend_weight),
        "lightgbm_best_validation_average_precision": float(study.best_value),
        "lightgbm_parameters": best_lgb_params,
        "categorical_features": cat_feature_names,
        "features": features,
        "dropped_features": drop_features,
    },
    "calibration": {
        "method": "Platt scaling (logistic regression on blended probability logit)",
        "threshold_selection_block": "second half of the chronological calibration period",
        "selected_threshold": float(optimal_threshold),
        "platt_coefficient": float(platt_model.coef_[0, 0]),
        "platt_intercept": float(platt_model.intercept_[0]),
        **threshold_calibration_stats,
    },
    "final_test_optimized_threshold": metrics_opt,
    "final_test_threshold_0_5": metrics_050,
}
with open(out_dir / "fraud_hybrid_metrics.json", "w", encoding="utf-8") as f:
    json.dump(all_metrics, f, indent=2, ensure_ascii=False)

# Confusion matrix plot.
fig, ax = plt.subplots(figsize=(6, 5))
image = ax.imshow(cm, interpolation="nearest")
ax.set_title("Hybrid Model — Final Test Confusion Matrix")
ax.set_xlabel("Predicted label")
ax.set_ylabel("Actual label")
ax.set_xticks([0, 1], labels=["Legitimate", "Fraud"])
ax.set_yticks([0, 1], labels=["Legitimate", "Fraud"])
for (i, j), value in np.ndenumerate(cm):
    ax.text(j, i, f"{value:,}", ha="center", va="center")
fig.colorbar(image, ax=ax)
fig.tight_layout()
fig.savefig(out_dir / "confusion_matrix_test.png", dpi=160, bbox_inches="tight")
plt.show()
plt.close(fig)

# ROC and Precision-Recall curves.
if len(np.unique(y_test)) == 2:
    fpr_values, tpr_values, _ = roc_curve(y_test, p_test)
    pr_precision, pr_recall, _ = precision_recall_curve(y_test, p_test)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr_values, tpr_values, label=f"ROC-AUC = {metrics_opt['roc_auc']:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Final Test ROC Curve")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve_test.png", dpi=160, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(pr_recall, pr_precision, label=f"Average Precision = {metrics_opt['average_precision']:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Final Test Precision–Recall Curve")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "precision_recall_curve_test.png", dpi=160, bbox_inches="tight")
    plt.show()
    plt.close(fig)

# ------------------------------- TREE SHAP ----------------------------------
# LightGBM's pred_contrib=True returns TreeSHAP contributions without requiring
# the separate shap package. The top-three contributions explain the LightGBM
# component; the displayed probability is the calibrated hybrid probability.
n_explain = min(MAX_EXPLANATIONS, len(X_test))
X_explain = X_test.iloc[:n_explain]
explain_ids = test_ids[:n_explain]
explain_probs = p_test[:n_explain]

try:
    contributions = lgb_model.predict(
        X_explain,
        num_iteration=lgb_model.best_iteration,
        pred_contrib=True,
    )
    shap_values = contributions[:, :-1]  # final column is the expected value/base term

    records = []
    for i in range(n_explain):
        top_idx = np.argsort(np.abs(shap_values[i]))[-3:][::-1]
        top3 = {
            features[j]: round(float(shap_values[i, j]), 7)
            for j in top_idx
        }
        records.append({
            "transaction_id": str(explain_ids[i]),
            "prediction_probability": round(float(explain_probs[i]), 7),
            "top3_shap_features": top3,
            "explanation_model": "LightGBM component of calibrated CatBoost-LightGBM hybrid",
        })

    with open(out_dir / "lightgbm_shap_output.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    top_global = np.argsort(mean_abs_shap)[-20:]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh([features[i] for i in top_global], mean_abs_shap[top_global])
    ax.set_xlabel("Mean absolute TreeSHAP contribution")
    ax.set_title("LightGBM Component — Global TreeSHAP Importance")
    fig.tight_layout()
    fig.savefig(out_dir / "shap_summary.png", dpi=160, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"[SUCCESS] SHAP JSON exported for {n_explain:,} transactions.")
except Exception as exc:
    print(f"[WARNING] SHAP export could not be completed: {type(exc).__name__}: {exc}")

print("\n" + "=" * 74)
print("PIPELINE COMPLETE")
print(f"Output folder: {out_dir.resolve()}")
for file_path in sorted(out_dir.iterdir()):
    if file_path.is_file():
        print(f"  {file_path.name:38s} {file_path.stat().st_size / (1024 * 1024):8.2f} MB")
print("=" * 74)
print("IMPORTANT: Compare these final-test counts only with a baseline evaluated")
print("on the exact same chronological test rows and using the same cost ratio.")
print("Do not change the threshold after inspecting final-test results.")
import json
from pathlib import Path

path = Path("fraud_hybrid_outputs/fraud_hybrid_metrics.json")

with open(path, "r", encoding="utf-8") as f:
    results = json.load(f)

print("\nFINAL OPTIMIZED TEST RESULTS")
print(json.dumps(
    results["final_test_optimized_threshold"],
    indent=4
))

print("\nTHRESHOLD 0.5 COMPARISON")
print(json.dumps(
    results["final_test_threshold_0_5"],
    indent=4
))

print("\nCALIBRATION RESULTS")
print(json.dumps(
    results["calibration"],
    indent=4
))