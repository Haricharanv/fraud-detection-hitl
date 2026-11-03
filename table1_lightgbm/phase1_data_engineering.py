"""
phase1_data_engineering.py -- Data Engineering Pipeline
Samsung Capstone Project - Table 1 (Team LightGBM)

Loads IEEE-CIS data, merges, handles missing values, encodes categoricals,
chronological split, exports artefacts for Phase 2.

Memory-optimized for machines with ~3 GB free RAM.
"""

import sys
import gc
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "ieee-fraud-detection"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
ARTEFACTS_DIR = Path(__file__).resolve().parent / "artefacts"

TRAIN_SPLIT_RATIO = 0.80

# Drop sparse V-columns (V35-V278) to reduce memory by ~60%
_V_COLS_TO_DROP = [f"V{i}" for i in range(35, 279)]


def _ensure_dirs():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)


def _reduce_mem(df):
    """Downcast numeric columns to save memory."""
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype(np.float32)
    for col in df.select_dtypes(include=["int64"]).columns:
        c_min, c_max = df[col].min(), df[col].max()
        if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
            df[col] = df[col].astype(np.int8)
        elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
            df[col] = df[col].astype(np.int16)
        elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
            df[col] = df[col].astype(np.int32)
    return df


def load_raw_data():
    """Load and merge transaction + identity CSVs with memory optimization."""
    txn_path = DATA_DIR / "train_transaction.csv"
    id_path = DATA_DIR / "train_identity.csv"

    for p in (txn_path, id_path):
        if not p.is_file():
            print(f"[ERROR] Missing file: {p}")
            print("[HINT]  Run download_data.py first.")
            sys.exit(1)

    print("[INFO] Loading train_transaction.csv (chunked, memory-optimized) ...")
    print(f"       Dropping {len(_V_COLS_TO_DROP)} sparse V-columns to save memory ...")
    chunks = []
    for i, chunk in enumerate(pd.read_csv(txn_path, chunksize=100_000)):
        cols_to_drop = [c for c in _V_COLS_TO_DROP if c in chunk.columns]
        chunk = chunk.drop(columns=cols_to_drop)
        chunk = _reduce_mem(chunk)
        chunks.append(chunk)
        print(f"       chunk {i+1}: {chunk.shape}")
    txn = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()
    mem_mb = txn.memory_usage(deep=True).sum() / 1e6
    print(f"       Transactions shape : {txn.shape}  ({mem_mb:.0f} MB)")

    print("[INFO] Loading train_identity.csv ...")
    ident = pd.read_csv(id_path)
    ident = _reduce_mem(ident)
    mem_mb = ident.memory_usage(deep=True).sum() / 1e6
    print(f"       Identity shape     : {ident.shape}  ({mem_mb:.0f} MB)")

    print("[INFO] Merging on TransactionID (left join) ...")
    df = txn.merge(ident, on="TransactionID", how="left")
    del txn, ident
    gc.collect()
    df = _reduce_mem(df)
    mem_mb = df.memory_usage(deep=True).sum() / 1e6
    print(f"       Merged shape       : {df.shape}  ({mem_mb:.0f} MB)")

    return df


def handle_missing_values(df):
    """Fill missing values: median for numerics, mode/'unknown' for categoricals."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if numeric_cols:
        medians = df[numeric_cols].median()
        df[numeric_cols] = df[numeric_cols].fillna(medians)
        print(f"[INFO] Filled {len(numeric_cols)} numeric columns with median.")

    for col in categorical_cols:
        mode_val = df[col].mode()
        fill = mode_val.iloc[0] if not mode_val.empty else "unknown"
        df[col] = df[col].fillna(fill)
    print(f"[INFO] Filled {len(categorical_cols)} categorical columns with mode/unknown.")

    remaining_nulls = df.isnull().sum().sum()
    if remaining_nulls > 0:
        print(f"[WARN] {remaining_nulls} nulls remain -- filling with 0.")
        df = df.fillna(0)

    return df


def encode_categoricals(df):
    """Label-encode all object/category columns."""
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    print(f"[INFO] Label-encoded {len(categorical_cols)} categorical columns.")
    return df, encoders


def chronological_split(df):
    """Split data chronologically by TransactionDT (80/20)."""
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    split_idx = int(len(df) * TRAIN_SPLIT_RATIO)

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print("[INFO] Chronological split (80/20):")
    print(f"       Train : {train_df.shape}  |  TransactionDT range: "
          f"[{train_df['TransactionDT'].min()}, {train_df['TransactionDT'].max()}]")
    print(f"       Test  : {test_df.shape}  |  TransactionDT range: "
          f"[{test_df['TransactionDT'].min()}, {test_df['TransactionDT'].max()}]")

    return train_df, test_df


def print_statistics(df, train_df, test_df):
    """Print dataset statistics."""
    print("\n" + "=" * 60)
    print("  DATASET STATISTICS")
    print("=" * 60)
    print(f"  Total samples       : {len(df):,}")
    print(f"  Total features      : {df.shape[1]}")
    print(f"  Train samples       : {len(train_df):,}")
    print(f"  Test  samples       : {len(test_df):,}")
    print(f"  ---")
    total_fraud = df["isFraud"].sum()
    total_legit = len(df) - total_fraud
    print(f"  Total fraud         : {total_fraud:,} ({100*total_fraud/len(df):.2f}%)")
    print(f"  Total legit         : {total_legit:,} ({100*total_legit/len(df):.2f}%)")
    print(f"  ---")
    train_fraud = train_df["isFraud"].sum()
    test_fraud = test_df["isFraud"].sum()
    print(f"  Train fraud ratio   : {100*train_fraud/len(train_df):.2f}%")
    print(f"  Test  fraud ratio   : {100*test_fraud/len(test_df):.2f}%")
    print("=" * 60 + "\n")


def save_artefacts(train_df, test_df, encoders):
    """Persist train/test splits and encoders."""
    train_pkl = ARTEFACTS_DIR / "train.pkl"
    test_pkl = ARTEFACTS_DIR / "test.pkl"
    encoders_pkl = ARTEFACTS_DIR / "encoders.pkl"

    with open(train_pkl, "wb") as f:
        pickle.dump(train_df, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[INFO] Saved train set  -> {train_pkl}")

    with open(test_pkl, "wb") as f:
        pickle.dump(test_df, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[INFO] Saved test set   -> {test_pkl}")

    with open(encoders_pkl, "wb") as f:
        pickle.dump(encoders, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[INFO] Saved encoders   -> {encoders_pkl}")

    csv_path = OUTPUTS_DIR / "preprocessed_test.csv"
    test_df.to_csv(csv_path, index=False)
    print(f"[INFO] Exported test CSV -> {csv_path}")


def main():
    _ensure_dirs()

    print("=" * 60)
    print("  PHASE 1 -- Data Engineering")
    print("=" * 60 + "\n")

    df = load_raw_data()
    print()
    df = handle_missing_values(df)
    df, encoders = encode_categoricals(df)
    print()
    train_df, test_df = chronological_split(df)
    print_statistics(df, train_df, test_df)
    del df
    gc.collect()
    save_artefacts(train_df, test_df, encoders)
    print("\n[SUCCESS] Phase 1 complete. Ready for Phase 2.\n")


if __name__ == "__main__":
    main()
