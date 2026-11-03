"""
ingest_predictions.py -- Multi-Dataset Bridge Script: Table 1 -> Table 3
Reads SHAP JSON for all 3 datasets (IEEE-CIS, FiFAR, Sichuan) and POSTs to FastAPI (PostgreSQL).
"""

import json
import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
BACKEND_URL = "http://localhost:8000"
BATCH_SIZE = 1000

DATASETS = [
    {
        "name": "IEEE-CIS Fraud Detection",
        "tag": "ieee_cis",
        "file": OUTPUTS_DIR / "lightgbm_shap_output.json",
        "max_records": 25000,
    },
    {
        "name": "Financial Fraud Alert Review (FiFAR)",
        "tag": "fifar",
        "file": OUTPUTS_DIR / "fifar_shap_output.json",
        "max_records": 25000,
    },
    {
        "name": "Sichuan Telecom Fraud",
        "tag": "sichuan_telecom",
        "file": OUTPUTS_DIR / "sichuan_shap_output.json",
        "max_records": 25000,
    },
]


def ingest_dataset(d_info):
    file_path = d_info["file"]
    tag = d_info["tag"]
    name = d_info["name"]

    print(f"\n--- Ingesting {name} [{tag}] ---")
    if not file_path.is_file():
        print(f"[WARN] File not found: {file_path}. Skipping.")
        return 0

    with open(file_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    # Cap if needed for fast interactive loading
    max_rec = d_info.get("max_records", len(records))
    records = records[:max_rec]

    # Ensure dataset_name is set
    for r in records:
        r["dataset_name"] = tag

    total_records = len(records)
    total_batches = (total_records + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"[INFO] Ingesting {total_records:,} records in {total_batches} batches (batch_size={BATCH_SIZE}) ...")

    success_count = 0
    url = f"{BACKEND_URL}/api/ingest/lightgbm"

    for i in range(0, total_records, BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        try:
            resp = requests.post(url, json=batch, timeout=60)
            resp.raise_for_status()
            res = resp.json()
            success_count += len(batch)
            if batch_num % 5 == 0 or batch_num == total_batches:
                print(f"  Batch {batch_num}/{total_batches}: {success_count:,}/{total_records:,} records ingested")
        except Exception as exc:
            print(f"[ERROR] Failed at batch {batch_num}: {exc}")
            break

    print(f"[SUCCESS] Ingested {success_count:,} records for {tag}.")
    return success_count


def main():
    print("=" * 60)
    print("  INGEST: Multi-Dataset SHAP Outputs -> PostgreSQL (FastAPI)")
    print("=" * 60)

    try:
        health = requests.get(f"{BACKEND_URL}/health", timeout=5)
        health.raise_for_status()
        print(f"[INFO] Backend is healthy at {BACKEND_URL} ({health.json().get('database', 'DB')})")
    except Exception as exc:
        print(f"[ERROR] Backend not reachable at {BACKEND_URL}: {exc}")
        print("[HINT]  Start FastAPI: cd table3_integration && python main.py")
        sys.exit(1)

    total_all = 0
    for d in DATASETS:
        total_all += ingest_dataset(d)

    print("\n" + "=" * 60)
    print(f"  ALL INGESTION COMPLETE -- {total_all:,} total records in PostgreSQL")
    print("=" * 60)
    print("[NEXT]   Open the Streamlit HITL Console:")
    print("         cd table3_integration && streamlit run streamlit_app.py\n")


if __name__ == "__main__":
    main()
