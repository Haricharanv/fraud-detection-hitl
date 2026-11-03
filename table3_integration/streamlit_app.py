"""
Streamlit HITL Console for Samsung Fraud Detection Capstone Project.
Supports Multi-Dataset Review (IEEE-CIS, FiFAR, Sichuan Telecom) with PostgreSQL backend.
"""

import os
from typing import Any, Dict, List, Optional

import plotly.graph_objects as go
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Fraud Detection HITL Console", page_icon="S",
                   layout="wide", initial_sidebar_state="expanded")


def api_get(path, params=None):
    try:
        resp = requests.get(f"{BACKEND_URL}{path}", params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.error(f"Cannot connect to backend at {BACKEND_URL}. "
                 "Make sure the FastAPI server is running with PostgreSQL.")
        return None
    except requests.HTTPError as exc:
        st.error(f"API error: {exc.response.status_code} - {exc.response.text}")
        return None


def api_post(path, json_data):
    try:
        resp = requests.post(f"{BACKEND_URL}{path}", json=json_data, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.error(f"Cannot connect to backend at {BACKEND_URL}.")
        return None
    except requests.HTTPError as exc:
        st.error(f"API error: {exc.response.status_code} - {exc.response.text}")
        return None


def render_shap_chart(shap_features, title="Feature Contributions"):
    if not shap_features:
        st.info("No feature explanations available.")
        return
    sorted_items = sorted(shap_features.items(), key=lambda x: abs(x[1]), reverse=True)
    features = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    colors = ["#e53935" if v > 0 else "#43a047" for v in values]

    fig = go.Figure(go.Bar(x=values, y=features, orientation="h",
                           marker_color=colors,
                           text=[f"{v:+.4f}" for v in values],
                           textposition="outside"))
    fig.update_layout(title=dict(text=title, font=dict(size=16)),
                      xaxis_title="SHAP Value (Impact on Fraud Risk)", yaxis_title="Feature",
                      height=max(250, len(features) * 55),
                      margin=dict(l=20, r=20, t=50, b=30),
                      yaxis=dict(autorange="reversed"),
                      plot_bgcolor="#fafafa")
    st.plotly_chart(fig, width='stretch')


# --- Sidebar ---

st.sidebar.title("HITL Console Settings")

backend_url_input = st.sidebar.text_input("Backend URL", value=BACKEND_URL)
if backend_url_input != BACKEND_URL:
    BACKEND_URL = backend_url_input.rstrip("/")

dataset_choice = st.sidebar.selectbox(
    "Active Dataset",
    options=[
        "All Datasets",
        "IEEE-CIS Fraud Detection",
        "Financial Fraud Alert Review (FiFAR)",
        "Sichuan Telecom Fraud",
    ],
    index=0
)

DATASET_TAG_MAP = {
    "All Datasets": None,
    "IEEE-CIS Fraud Detection": "ieee_cis",
    "Financial Fraud Alert Review (FiFAR)": "fifar",
    "Sichuan Telecom Fraud": "sichuan_telecom",
}
selected_tag = DATASET_TAG_MAP[dataset_choice]

model_mode = st.sidebar.radio("Model Mode", options=["LightGBM Mode", "RL Mode"], index=0)
top_k = st.sidebar.slider("Top-K Transactions", min_value=1, max_value=50, value=10)

if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Samsung Capstone - HITL Console (PostgreSQL Engine)")


# --- Main Header ---

st.title("Fraud Detection HITL Console")
st.markdown(f"**Database:** PostgreSQL 15 | **Active Dataset:** `{dataset_choice}` | **Model:** `{model_mode}`")
st.markdown("---")

is_lightgbm = model_mode == "LightGBM Mode"

if is_lightgbm:
    endpoint = "/api/predictions/lightgbm"
    st.subheader(f"Top-{top_k} LightGBM Predictions (Highest Risk)")
else:
    endpoint = "/api/predictions/rl"
    st.subheader(f"Top-{top_k} RL Actions (Most Recent)")

query_params = {"top_k": top_k}
if selected_tag:
    query_params["dataset"] = selected_tag

predictions = api_get(endpoint, params=query_params)
if predictions is None:
    st.stop()
if len(predictions) == 0:
    st.warning(f"No predictions found for dataset '{dataset_choice}'. Please run ingest_predictions.py first.")
    st.stop()

table_data = []
for pred in predictions:
    d_name = pred.get("dataset_name") or "ieee_cis"
    if is_lightgbm:
        table_data.append({
            "Transaction ID": pred["transaction_id"],
            "Dataset": d_name.upper(),
            "Risk Score": f"{pred['prediction_probability']:.4f}",
            "Created At": pred["created_at"][:19],
        })
    else:
        table_data.append({
            "Transaction ID": pred["transaction_id"],
            "Dataset": d_name.upper(),
            "Action": pred["chosen_action"],
            "Created At": pred["created_at"][:19],
        })

st.dataframe(table_data, width='stretch', hide_index=True)


# --- Detail View ---

st.markdown("---")
st.subheader("Transaction Detail & Explanation")

txn_ids = [p["transaction_id"] for p in predictions]
selected_txn_id = st.selectbox("Select a transaction to inspect", options=txn_ids, index=0)

if selected_txn_id:
    # Find matching record in current list
    current_rec = next((p for p in predictions if p["transaction_id"] == selected_txn_id), None)
    detail = api_get(f"/api/transaction/{selected_txn_id}")

    if current_rec:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown(f"**Transaction ID:** `{selected_txn_id}`")
            d_name = current_rec.get("dataset_name") or (detail.get("dataset_name") if detail else "ieee_cis")
            st.markdown(f"**Dataset Source:** `{d_name}`")
            if detail and detail.get("is_fraud") is not None:
                fraud_label = "Fraud (1)" if detail["is_fraud"] else "Legitimate (0)"
                st.markdown(f"**Ground Truth:** `{fraud_label}`")

        with col2:
            if is_lightgbm:
                prob = current_rec["prediction_probability"]
                st.metric("Predicted Fraud Probability", f"{prob:.4f}")
                if prob >= 0.8:
                    st.error("RISK STATUS: CRITICAL FRAUD ALERT")
                elif prob >= 0.5:
                    st.warning("RISK STATUS: SUSPICIOUS ACTIVITY")
                else:
                    st.success("RISK STATUS: LOW RISK")

        st.markdown("#### Top-3 Feature Contributions (SHAP Values)")
        shap_data = current_rec.get("top3_shap_features")
        if shap_data:
            render_shap_chart(shap_data, title=f"SHAP Explanations for Transaction {selected_txn_id}")
        else:
            st.info("No SHAP explanations available for this record.")

        # Feedback buttons
        st.markdown("#### Human Operator Decision (Feedback Loop)")
        feedback_col1, feedback_col2, _ = st.columns([1, 1, 2])
        model_src = "lightgbm" if is_lightgbm else "rl"

        with feedback_col1:
            if st.button("Confirm Fraud", key=f"fraud_{selected_txn_id}", type="primary"):
                result = api_post("/api/feedback", {
                    "transaction_id": selected_txn_id,
                    "dataset_name": d_name,
                    "operator_decision": "confirm_fraud",
                    "model_source": model_src,
                })
                if result:
                    st.toast("Decision recorded: Confirm Fraud -> PostgreSQL updated!")
                    st.rerun()

        with feedback_col2:
            if st.button("Mark Legitimate", key=f"legit_{selected_txn_id}"):
                result = api_post("/api/feedback", {
                    "transaction_id": selected_txn_id,
                    "dataset_name": d_name,
                    "operator_decision": "mark_legitimate",
                    "model_source": model_src,
                })
                if result:
                    st.toast("Decision recorded: Mark Legitimate -> PostgreSQL updated!")
                    st.rerun()


# --- Feedback History ---

st.markdown("---")
st.subheader("Operator Feedback Audit Trail (PostgreSQL)")

feedback_params = {}
if selected_tag:
    feedback_params["dataset"] = selected_tag

feedback_data = api_get("/api/feedback/export", params=feedback_params)
if feedback_data and len(feedback_data) > 0:
    fb_table = []
    for fb in feedback_data[:25]:
        fb_table.append({
            "Transaction ID": fb["transaction_id"],
            "Dataset": (fb.get("dataset_name") or "ieee_cis").upper(),
            "Decision": fb["operator_decision"],
            "Model Source": fb["model_source"].upper(),
            "Timestamp": fb["feedback_timestamp"][:19],
        })
    st.dataframe(fb_table, width='stretch', hide_index=True)
else:
    st.info("No operator feedback recorded yet for this view.")
