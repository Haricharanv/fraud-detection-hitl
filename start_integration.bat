@echo off
REM ========================================================================
REM  Samsung Capstone Project — Start Integration Layer (Table 3)
REM  Starts FastAPI + Streamlit in parallel
REM ========================================================================

echo.
echo ====================================================================
echo   Starting Integration Layer (Table 3)
echo ====================================================================
echo.

REM Check if .env exists
if not exist "table3_integration\.env" (
    echo [INFO] Creating .env from .env.example...
    copy "table3_integration\.env.example" "table3_integration\.env"
)

echo [1/2] Starting FastAPI backend on http://localhost:8000 ...
start "FastAPI Server" cmd /c "cd table3_integration && uvicorn main:app --reload --host 0.0.0.0 --port 8000"

REM Wait for FastAPI to start
timeout /t 3 /nobreak > nul

echo [2/2] Starting Streamlit UI on http://localhost:8501 ...
start "Streamlit HITL Console" cmd /c "cd table3_integration && streamlit run streamlit_app.py"

echo.
echo ====================================================================
echo   Integration Layer is running!
echo ====================================================================
echo   FastAPI:    http://localhost:8000
echo   API Docs:   http://localhost:8000/docs
echo   Streamlit:  http://localhost:8501
echo.
echo   To ingest LightGBM predictions:
echo     python ingest_predictions.py
echo.
echo   Close the opened terminal windows to stop the servers.
echo ====================================================================
echo.
pause

