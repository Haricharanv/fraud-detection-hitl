# 🛡️ Enterprise Fraud Detection & Human-in-the-Loop Platform

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Haricharanv/fraud-detection-hitl/blob/main/table1_lightgbm/Cost_Sensitive_LightGBM_Colab.ipynb)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29%2B-FF4B4B?logo=streamlit)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-brightgreen)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2%2B-yellow)
![Database](https://img.shields.io/badge/Database-PostgreSQL%2015%20%7C%20SQLite%20Fallback-blue)

A multi-dataset, production-grade fraud detection platform combining **asymmetric cost-sensitive Machine Learning** with **TreeSHAP explainability**, an **asynchronous FastAPI microservice**, **dual PostgreSQL/SQLite persistence**, and an interactive **Streamlit Human-in-the-Loop (HITL) Console**.

**Scope:** Table 1 (Cost-Sensitive ML & Explainability) & Table 3 (FastAPI Backend, Database & HITL Console)  
**Architecture:** Distributed Machine Learning & Closed-Loop Active Learning Retraining  

---

## 📋 Implementation Overview

| Component | Description | Status |
|---|---|---|
| **Table 1: Team LightGBM & Hybrid ML** | • **Cost-Sensitive LightGBM:** Asymmetric financial loss ($C_{FN}=10.0, C_{FP}=1.0$) with custom gradients ($g_i, h_i$) across 3 benchmarks.<br>• **Temporal Hybrid Pipeline:** Chronological split ensembling CatBoost + LightGBM with Platt probability calibration & cost-optimal threshold tuning (16.2% cost reduction).<br>• **Local Explainability:** Polynomial-time TreeSHAP ($O(TLD^2)$) generating per-transaction top-3 feature attributions.<br>• **Colab Support:** Self-contained single-cell scripts & interactive notebooks (`.ipynb`). | ✅ 100% Implemented |
| **Table 2: Team RL** | • **Adaptive Path (Approve / Review / Block):** Preserved database models (`rl_actions`), REST microservice endpoints (`/api/ingest/rl`, `/api/predictions/rl`), and Streamlit RL view mode ready for reinforcement learning policy ingestion. | ℹ️ Preserved schema & API endpoints for RL team |
| **Table 3: Integration & HITL Console** | • **FastAPI Microservice:** Asynchronous REST microservice with high-throughput batch ingestion and automated Swagger UI (`/docs`).<br>• **Resilient Database Layer:** Dual-engine architecture supporting PostgreSQL 15 with zero-config automatic fallback to SQLite (`fraud_detection.db`).<br>• **Streamlit Operator Console:** Prioritized Top-Risk Queue, Plotly interactive TreeSHAP attribution bar charts, and one-click operator actions (`Confirm Fraud` / `Mark Legitimate`).<br>• **Continuous Feedback Loop:** Live audit trail logging with `retrain_script.py` for continuous active learning. | ✅ 100% Implemented |

---

## 📊 Benchmarked Datasets

The platform has been benchmarked and ingested across three diverse industry domains:

| Dataset | Domain | Raw Size | Rows (Train / Test) | Features | Fraud Rate | Model ROC-AUC | Operational Cost Performance |
|---|---|---|---|---|---|---|---|
| **IEEE-CIS Fraud Detection** | Payment Cards & E-Commerce | 1.28 GB | 472,432 / 118,108 | 188 features | 3.50% | **0.9119** (Base) / **0.9078** (Temporal Hybrid) | **16.2% Cost Reduction**<br>Threshold 0.50: 24,032<br>Optimal ($\tau^*=0.1013$): **20,142** |
| **Financial Fraud Alert Review (FiFAR)** | Bank Account Opening | 151.7 MB | 506,118 / 96,843 | 32 features | 1.13% | **0.8844** | Asymmetric loss optimized |
| **Sichuan Telecom Fraud** | Call Detail Records & Cards | 44.5 MB | 800,000 / 200,000 | 7 features | 8.74% | **0.9999** | Perfect separation |

---

## ☁️ Google Colab Quick Start (Table 1 ML)

Train models or run hyperparameter sweeps directly in the cloud with free GPU/RAM:

1. **Launch Interactive Notebook:**  
   Click the badge: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Haricharanv/fraud-detection-hitl/blob/main/table1_lightgbm/Cost_Sensitive_LightGBM_Colab.ipynb)
2. **Mount Google Drive & Run:**  
   Place `train_transaction.csv` and `train_identity.csv` in your Google Drive or upload to `/content`. Run all cells to train the cost-sensitive booster and export `lightgbm_shap_output.json`.
3. **Alternative Single-File Script:**  
   Run [`table1_lightgbm/cost_sensitive_lightgbm_colab.py`](table1_lightgbm/cost_sensitive_lightgbm_colab.py) or [`table1_lightgbm/Fraud Detection LightGBM.py`](table1_lightgbm/Fraud%20Detection%20LightGBM.py) (Arjun's hybrid pipeline) in any Python environment.

---

## 🚀 Local Quick Start (Table 3 Backend & Console)

Running locally provides **instant 0ms latency** and **100% offline reliability** during live presentations.

### 1. Database (Zero-Config Resilient Setup)
* **Option A (Default - SQLite):** No installation needed! If Docker is not running, the application **automatically uses SQLite (`fraud_detection.db`) seamlessly**.
* **Option B (Docker PostgreSQL 15 - Optional):**
  ```powershell
  docker run -d --name fraud-postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=fraud_detection -p 5432:5432 postgres:15-alpine
  ```

### 2. Launch Backend & Console in Parallel
Double-click **`start_integration.bat`** or run in terminal:
```powershell
.\start_integration.bat
```
* **Streamlit HITL Console:** [http://localhost:8501](http://localhost:8501)
* **FastAPI Microservice Docs:** [http://localhost:8000/docs](http://localhost:8000/docs) (or [http://localhost:8000](http://localhost:8000))

### 3. Ingest Predictions & SHAP Attributions (75,000 records)
```powershell
python ingest_predictions.py
```

### 4. Export Human Feedback for Active Learning Retraining
```powershell
cd table3_integration
python retrain_script.py
```

---

## 📁 Project Directory Structure

```
Samsung Capstone Project/
├── outputs/                                # Model visualizations & deliverables
│   ├── presentation_cheat_sheet.pdf        # 5-page presentation cheat sheet & scripts
│   ├── shap_summary.png                    # IEEE-CIS global feature importance plot
│   ├── shap_summary_fifar.png              # FiFAR global feature importance plot
│   └── shap_summary_sichuan.png            # Sichuan Telecom global feature importance plot
├── table1_lightgbm/                        # Table 1: ML, Hybrid & Explainability Pipelines
│   ├── Cost_Sensitive_LightGBM_Colab.ipynb # Interactive Google Colab notebook
│   ├── cost_sensitive_lightgbm_colab.py    # Self-contained Colab/local cost-sensitive pipeline
│   ├── Fraud Detection LightGBM.py         # Arjun's temporal CatBoost-LightGBM hybrid pipeline
│   ├── phase1_data_engineering.py          # IEEE-CIS chunked loader & preprocessor
│   ├── phase2_cost_sensitive_training.py   # IEEE-CIS cost-sensitive trainer (g_i, h_i)
│   ├── phase3_explainability.py            # IEEE-CIS SHAP TreeExplainer
│   ├── train_fifar.py                      # FiFAR training & SHAP pipeline
│   ├── train_sichuan.py                    # Sichuan training & SHAP pipeline
│   ├── download_data.py                    # Kaggle dataset fetcher
│   └── requirements.txt                    # Table 1 dependencies (LightGBM, CatBoost, Optuna, SHAP)
├── table3_integration/                     # Table 3: Backend, DB & Operator Console
│   ├── database.py                         # Dual-engine database (PostgreSQL 15 + SQLite fallback)
│   ├── main.py                             # Asynchronous FastAPI microservice (8 REST endpoints)
│   ├── streamlit_app.py                    # Streamlit HITL operator console
│   ├── retrain_script.py                   # Feedback extraction & active learning retrainer
│   ├── requirements.txt                    # Table 3 dependencies
│   └── .env.example                        # Environment configuration template
├── ingest_predictions.py                   # High-throughput batch ingestion script
├── run_table1_pipeline.bat                 # Automated batch ML pipeline runner
├── start_integration.bat                   # One-click desktop launcher (FastAPI + Streamlit)
├── .gitignore                              # Clean repository exclusion rules
└── README.md                               # Project documentation
```

---

## ⚖️ Business Cost Matrix & Objective Function

Standard cross-entropy optimizes for raw accuracy, which fails when fraud represents $<3\%$ of transactions. We align model optimization directly with banking economics:

$$\text{Total Cost} = C_{FP} \cdot \text{FP} + C_{FN} \cdot \text{FN} = 1.0 \cdot \text{FP} + 10.0 \cdot \text{FN}$$

### First and Second-Order Custom Gradients
$$\mathcal{L}_{CS}(y, p) = - \left[ C_{FN} \cdot y \ln(p) + C_{FP} \cdot (1 - y) \ln(1 - p) \right]$$
$$w_i = \begin{cases} 10.0 & \text{if } y_i = 1 \\ 1.0 & \text{if } y_i = 0 \end{cases}, \qquad g_i = w_i (p_i - y_i), \qquad h_i = w_i \cdot p_i (1 - p_i)$$

Every split candidate in LightGBM actively prioritizes isolating fraudulent clusters, while TreeSHAP isolates the top-3 feature attributions in polynomial time ($\mathcal{O}(TLD^2)$) to ensure compliance with financial "Right to Explanation" regulations.
