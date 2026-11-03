# 🛡️ Samsung Capstone Project — Enterprise Fraud Detection Platform

A multi-dataset, cost-sensitive fraud detection platform with SHAP explainability, FastAPI microservice, PostgreSQL 15 relational backend, and an interactive Streamlit Human-in-the-Loop (HITL) Console.

**Project Scope:** Table 1 (Cost-Sensitive LightGBM + SHAP) & Table 3 (FastAPI + PostgreSQL + HITL Console)  
**Architecture:** Distributed Machine Learning & Human-in-the-Loop Integration  

---

## 📋 Implementation Overview

| Component | Description | Status |
|---|---|---|
| **Table 1: Team LightGBM** | Supervised Cost-Sensitive LightGBM ($C_{FP}=1.0$, $C_{FN}=10.0$) + SHAP TreeExplainer across 3 datasets | ✅ 100% Implemented |
| **Table 2: Team RL** | Reinforcement Learning Adaptive Path (Approve / Review / Block) | ℹ️ Preserved schema & API endpoints for RL team |
| **Table 3: Integration Developer** | Async FastAPI service + PostgreSQL 15 + Streamlit HITL Console + Continuous Active Learning Retraining | ✅ 100% Implemented |

---

## 📊 Benchmarked Datasets

| Dataset | Domain | Raw Size | Rows (Train / Test) | Features | Fraud Rate | Model AUC-ROC |
|---|---|---|---|---|---|---|
| **IEEE-CIS Fraud Detection** | Payment Cards & E-Commerce | 1.28 GB | 472,432 / 118,108 | 188 features | 3.50% | **0.9119** |
| **Financial Fraud Alert Review (FiFAR)** | Bank Account Opening | 151.7 MB | 506,118 / 96,843 | 32 features | 1.13% | **0.8844** |
| **Sichuan Telecom Fraud** | Call Detail Records & Cards | 44.5 MB | 800,000 / 200,000 | 7 features | 8.74% | **0.9999** |

---

## 🚀 One-Click Quick Start

### 1. Start PostgreSQL (Docker)
```powershell
docker run -d --name fraud-postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=fraud_detection -p 5432:5432 postgres:15-alpine
```

### 2. Launch FastAPI & Streamlit Console
Double click **`start_integration.bat`** or run:
```powershell
.\start_integration.bat
```
* **Streamlit HITL Console:** `http://localhost:8501`
* **FastAPI Swagger Docs:** `http://localhost:8000/docs`

### 3. Ingest Multi-Dataset Predictions (75,000 records)
```powershell
python ingest_predictions.py
```

### 4. Export Operator Decisions for Continuous Retraining
```powershell
cd table3_integration
python retrain_script.py
```

---

## 📁 Project Directory Structure

```
Samsung Capstone Project/
├── data/                              # Benchmark CSV files (gitignored)
├── outputs/                           # Generated outputs & deliverables
│   ├── capstone_project_report.pdf    # Full technical documentation (PDF)
│   ├── presentation_guide.pdf         # Step-by-step presentation script & Q&A
│   ├── lightgbm_shap_output.json      # IEEE-CIS SHAP explanations
│   ├── fifar_shap_output.json         # FiFAR SHAP explanations
│   ├── sichuan_shap_output.json       # Sichuan Telecom SHAP explanations
│   ├── shap_summary.png               # Global feature importance plots
│   └── retrain_data.csv               # Exported human-in-the-loop decisions
├── table1_lightgbm/                   # Table 1: ML & Explainability Pipeline
│   ├── phase1_data_engineering.py     # IEEE-CIS data preprocessor
│   ├── phase2_cost_sensitive_training.py # IEEE-CIS cost-sensitive trainer
│   ├── phase3_explainability.py       # IEEE-CIS SHAP TreeExplainer
│   ├── train_fifar.py                 # FiFAR pipeline
│   ├── train_sichuan.py               # Sichuan pipeline
│   └── artefacts/                     # Serialized models (.pkl) & encoders
├── table3_integration/                # Table 3: Production Backend & UI
│   ├── database.py                    # PostgreSQL 15 async models (SQLAlchemy)
│   ├── main.py                        # FastAPI microservice
│   ├── streamlit_app.py               # Streamlit HITL operator console
│   └── retrain_script.py              # Active learning export script
├── ingest_predictions.py              # Ingests JSON predictions to PostgreSQL
├── build_pdfs.py                      # Recompiles PDFs with KaTeX math
├── start_integration.bat              # One-click launcher
├── PRESENTATION_GUIDE.md              # Script markdown
├── PROJECT_REPORT.md                  # Report markdown
└── README.md
```
