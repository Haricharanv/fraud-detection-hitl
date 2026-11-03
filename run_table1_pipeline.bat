@echo off
REM ========================================================================
REM  Samsung Capstone Project — Full Pipeline Runner
REM  Runs Table 1 (LightGBM) phases sequentially
REM ========================================================================

echo.
echo ====================================================================
echo   Samsung Capstone Project - Full ML Pipeline
echo ====================================================================
echo.

REM Check if data exists
if not exist "data\ieee-fraud-detection\train_transaction.csv" (
    echo [STEP 1/4] Downloading IEEE-CIS dataset from Kaggle...
    python table1_lightgbm\download_data.py
    if %errorlevel% neq 0 (
        echo [FAILED] Dataset download failed. See errors above.
        pause
        exit /b 1
    )
) else (
    echo [STEP 1/4] Dataset already exists, skipping download.
)

echo.
echo [STEP 2/4] Phase 1: Data Engineering...
python table1_lightgbm\phase1_data_engineering.py
if %errorlevel% neq 0 (
    echo [FAILED] Phase 1 failed.
    pause
    exit /b 1
)

echo.
echo [STEP 3/4] Phase 2: Cost-Sensitive LightGBM Training...
python table1_lightgbm\phase2_cost_sensitive_training.py
if %errorlevel% neq 0 (
    echo [FAILED] Phase 2 failed.
    pause
    exit /b 1
)

echo.
echo [STEP 4/4] Phase 3: Explainability ^& SHAP JSON Handoff...
python table1_lightgbm\phase3_explainability.py
if %errorlevel% neq 0 (
    echo [FAILED] Phase 3 failed.
    pause
    exit /b 1
)

echo.
echo ====================================================================
echo   ALL PHASES COMPLETE!
echo ====================================================================
echo.
echo   Outputs:
echo     - outputs\lightgbm_predictions.csv
echo     - outputs\lightgbm_shap_output.json
echo     - outputs\shap_summary.png
echo     - outputs\preprocessed_test.csv
echo.
echo   Next steps:
echo     1. Start PostgreSQL and create database:
echo        psql -U postgres -c "CREATE DATABASE fraud_detection;"
echo     2. Start FastAPI: cd table3_integration ^&^& uvicorn main:app --reload
echo     3. Ingest predictions: python ingest_predictions.py
echo     4. Start Streamlit: cd table3_integration ^&^& streamlit run streamlit_app.py
echo.
pause

