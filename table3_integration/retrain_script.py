"""
Retraining data pull script.
Exports operator feedback from the HITL backend to CSV for model retraining.
Works with both SQLite and PostgreSQL backends via the FastAPI REST API.
"""

import sys
import logging
from pathlib import Path

import requests
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

BACKEND_URL = "http://localhost:8000"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "retrain_data.csv"


def export_retrain_data():
    """Pull labelled feedback from the FastAPI backend and write CSV."""
    logger.info("Connecting to backend at %s ...", BACKEND_URL)

    # Check health
    try:
        health = requests.get(f"{BACKEND_URL}/health", timeout=5)
        health.raise_for_status()
    except (requests.ConnectionError, requests.HTTPError) as exc:
        logger.error("Backend not reachable at %s: %s", BACKEND_URL, exc)
        logger.error("Start FastAPI first: cd table3_integration && python main.py")
        sys.exit(1)

    # Fetch all operator feedback
    try:
        resp = requests.get(f"{BACKEND_URL}/api/feedback/export", timeout=30)
        resp.raise_for_status()
        rows = resp.json()
    except requests.HTTPError as exc:
        logger.error("Failed to fetch feedback: %s", exc)
        sys.exit(1)

    if not rows:
        logger.warning("No operator feedback found. Nothing to export.")
        return

    df = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    logger.info("Exported %d rows to %s", len(df), OUTPUT_FILE)

    total = len(df)
    confirm_fraud = int((df["operator_decision"] == "confirm_fraud").sum())
    mark_legitimate = int((df["operator_decision"] == "mark_legitimate").sum())

    print("\n" + "=" * 60)
    print("  RETRAINING DATA EXPORT -- STATISTICS")
    print("=" * 60)
    print(f"  Total feedback records : {total}")
    print(f"  Confirm Fraud          : {confirm_fraud}  ({confirm_fraud/total*100:.1f}%)")
    print(f"  Mark Legitimate        : {mark_legitimate}  ({mark_legitimate/total*100:.1f}%)")
    print(f"  Output file            : {OUTPUT_FILE}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    export_retrain_data()
