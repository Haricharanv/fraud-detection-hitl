# Samsung Capstone Project -- Presentation Guide & Script

**Project:** Enterprise Fraud Detection System (Table 1: LightGBM + Table 3: HITL Console)  
**Database:** PostgreSQL 15  

---

## Screen Setup Before You Begin

Have these 3 browser tabs and 1 terminal ready:

* **Tab 1:** Streamlit HITL Console -- `http://localhost:8501` (Primary visual showcase)
* **Tab 2:** FastAPI Swagger Docs -- `http://localhost:8000/docs` (Proof of production backend)
* **Tab 3:** SHAP Summary Plot -- `outputs/shap_summary.png` (Model interpretability)
* **Tab 4 (VS Code):** Project Codebase -- Workspace Root

*(Note: To restart servers anytime, double-click `start_integration.bat`)*

---

## The 5-Act Presentation Script

```
Act 1: Executive Summary & Financial Cost Asymmetry (1 min)
   |
Act 2: Multi-Dataset Scale & Data Engineering (1.5 min)
   |
Act 3: Cost-Sensitive LightGBM & SHAP Mathematics (2 min)
   |
Act 4: PostgreSQL 15 & FastAPI System Architecture (1 min)
   |
Act 5: Live Interactive Demo on Streamlit HITL Console (2.5 min)
```

---

### Act 1: Executive Summary & Business Problem (1 minute)

* **Where to look:** Slide or title screen.
* **What to say:**
  > "Traditional fraud detection models optimize for standard accuracy or binary log-loss. In real banking operations, that fails because **errors do not cost the same**:
  > 
  > - A **False Positive (FP)** (blocking a legitimate customer's card) causes minor friction, costing approximately **$1 to $5** in verification costs.
  > - A **False Negative (FN)** (allowing a fraudulent transaction through) loses the entire transaction value and incurs chargeback penalties, costing **10x to 50x more**.
  > 
  > Our project delivers an end-to-end, enterprise-grade fraud detection platform:
  > - **Table 1:** Supervised Cost-Sensitive LightGBM with SHAP explainability.
  > - **Table 3:** FastAPI and PostgreSQL 15 integration with a Human-in-the-Loop Operator Console.
  > - Evaluated and benchmarked across **three distinct real-world fraud datasets**."

---

### Act 2: Multi-Dataset Scale & Data Engineering (1.5 minutes)

* **Where to navigate:** Open VS Code -> Show `data/` folder and summary table.
* **What to explain:**
  > "To ensure the architecture is domain-agnostic, we stress-tested and trained our pipeline on three diverse benchmarks:
  > 
  > 1. **IEEE-CIS Fraud Detection (E-commerce / FinTech):** 1.28 GB raw data, 590,540 transactions, 3.50% fraud rate. We engineered 188 clean features after pruning 244 high-sparsity noise columns, downcasting types to run in high-throughput environments.
  > 2. **Financial Fraud Alert Review - FiFAR (Bank Account Opening):** 151.7 MB, 602,961 total instances with 50 synthetic human reviewers, 1.13% fraud rate.
  > 3. **Sichuan Telecom Fraud (Telecommunications / Card Fraud):** 44.5 MB, 1,000,000 transactions, 8.74% fraud rate.
  > 
  > All datasets were split chronologically/sequentially to prevent data leakage from future transactions."

#### Dataset Overview Table

| Dataset | Total Size | Train Rows | Test Rows | Features | Fraud Rate | LightGBM AUC | Total Cost |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **IEEE-CIS Fraud** | 1.28 GB | 472,432 | 118,108 | 188 | 3.50% | **0.9119** | 19,792 |
| **FiFAR Alert Review** | 151.7 MB | 506,118 | 96,843 | 32 | 1.13% | **0.8844** | 10,749 |
| **Sichuan Telecom** | 44.5 MB | 800,000 | 200,000 | 7 | 8.74% | **0.9999** | 656 |

---

### Act 3: Cost-Sensitive LightGBM & SHAP (2 minutes)

* **Where to navigate:** Open `table1_lightgbm/phase2_cost_sensitive_training.py` at lines 28-36, and switch to Tab 3 (`outputs/shap_summary.png`).
* **What to explain:**
  > "Instead of default cross-entropy, we derived a **custom asymmetric cost-sensitive objective**:
  > 
  > - We enforce financial penalty weights: C_FP = 1.0 and C_FN = 10.0 (10x higher penalty for letting fraud through).
  > - The gradient and hessian passed to LightGBM's histogram engine are:
  >   - grad = w_i * (p_i - y_i)
  >   - hess = w_i * p_i * (1.0 - p_i)
  >   where w_i = 10.0 if the ground-truth is fraud (y_i = 1), else 1.0.
  > 
  > This forces the boosting trees to heavily penalize missed fraud, achieving:
  > - **0.9119 AUC-ROC** on IEEE-CIS (catching 2,499 frauds in the test set).
  > - **0.8844 AUC-ROC** on FiFAR.
  > - **0.9999 AUC-ROC** on Sichuan Telecom.
  > 
  > Next, because enterprise financial compliance forbids black-box models, we integrated **SHAP TreeExplainer**. We compute exact local feature attributions for every flagged transaction, producing a strict JSON handoff payload containing the Transaction ID, probability score, and top-3 contributing features."

---

### Act 4: System Architecture & PostgreSQL Backend (1 minute)

* **Where to navigate:** Tab 2: `http://localhost:8000/docs`.
* **What to explain:**
  > "For Table 3, we built an async **FastAPI** backend backed by **PostgreSQL 15** running in Docker:
  > 
  > - `POST /api/ingest/lightgbm` ingests high-throughput model batches with dataset identifiers.
  > - `POST /api/ingest/rl` is architected to accept actions and Q-values from reinforcement learning agents (Table 2).
  > - `GET /api/predictions/lightgbm?dataset=...` allows low-latency retrieval of the highest-risk transactions.
  > - `POST /api/feedback` captures human operator ground-truth decisions for active learning.
  > 
  > We have already ingested **75,000 active records** into PostgreSQL."

---

### Act 5: Live Interactive Demo (Streamlit HITL Console) (2.5 minutes)

* **Where to navigate:** Tab 1: `http://localhost:8501`.

Follow these physical steps in the UI:

1. **Show the Top Table:**
   > "Here is the operator's queue. By default, it surfaces the top-10 highest-risk transactions across the entire banking stream."
2. **Demonstrate Dataset Multi-Tenancy:**
   - Go to sidebar -> Click the **Active Dataset** dropdown -> Select **Sichuan Telecom Fraud**.
   - Notice the table immediately updates to show Sichuan records with high risk scores.
   - Switch back to **IEEE-CIS Fraud Detection** or **All Datasets**.
3. **Inspect a Single Transaction:**
   - In **Select a transaction to inspect**, pick the top transaction (e.g. `3557572` or `3475683`).
   - Point to the **Red CRITICAL FRAUD ALERT** badge with a score of `0.998`.
4. **Explain the SHAP Bar Chart:**
   - Point to the horizontal Plotly bar chart:
   > "Here is why the model flagged this: red bars represent features pushing the probability towards fraud (e.g. high TransactionAmt and velocity count C13), while green bars push towards legitimate. An analyst doesn't have to guess -- they see the mathematical justification in milliseconds."
5. **Close the Loop (Click Feedback):**
   - Click the red **Confirm Fraud** button.
   - Watch the toast alert pop up: *"Decision recorded: Confirm Fraud -> PostgreSQL updated!"*
   - Scroll down to **Operator Feedback Audit Trail** -> Show that this decision was saved into PostgreSQL with the exact timestamp and operator source.
6. **Show Continuous Retraining:**
   - Open terminal and show: `python table3_integration/retrain_script.py`
   > "Whenever ML teams want to retrain the models, they run this script to pull all verified human operator labels into an augmented training set, closing the continuous learning loop."

---

## Evaluator Q&A Defense Sheet

### Q1: Why LightGBM instead of XGBoost or Deep Learning?
> "LightGBM uses histogram-based feature binning and leaf-wise (best-first) tree growth. On tabular datasets with over 500,000 rows like IEEE-CIS and FiFAR, it trains 10x faster than traditional GBDTs with lower memory overhead, and natively supports custom second-order derivatives."

### Q2: Why not just tune the classification threshold instead of writing a custom loss?
> "Threshold tuning only shifts the post-hoc decision boundary. A custom cost-sensitive loss modifies tree splits during training by scaling the Hessian. The gradient boosting process prioritizes regions with false negatives, producing better calibrated probabilities in high-risk zones."

### Q3: How does SHAP TreeExplainer scale to thousands of samples?
> "Unlike KernelSHAP which is exponential in feature count O(M * 2^|F|), TreeExplainer uses the Lundberg tree-traversal algorithm running in polynomial time O(T * L * D^2), where T is number of trees, L is max leaves, and D is max depth. This allows exact Shapley values to be calculated across tens of thousands of trees in seconds."

### Q4: Where is the RL component from Table 2?
> "Our scope was specifically Table 1 (Supervised Cost-Sensitive Path) and Table 3 (Integration Hub). However, as shown in our FastAPI Swagger documentation and database schema, we fully implemented the `rl_actions` table, Pydantic schemas, and endpoints so an RL team's policy can plug in without backend modifications."

