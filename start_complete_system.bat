@echo off
echo Starting Complete E-Consultation Insight Engine System
echo ======================================================

echo.
echo 1. Starting Backend API Server (Port 8000)...
echo    This may take a few minutes on first run to download models
start "API Server" /min python working_api.py

echo.
echo 2. Waiting for API to initialize...
timeout /t 10 /nobreak >nul

echo.
echo 3. Starting Dashboard (Port 8501)...
start "Dashboard" /max python -m streamlit run dashboard/main.py --server.port 8501

echo.
echo System Startup Complete!
echo ========================
echo API Server:     http://localhost:8000
echo Dashboard:      http://localhost:8501
echo API Health:     http://localhost:8000/api/v1/health
echo API Docs:       http://localhost:8000/docs
echo.
echo Press any key to close this window...
pause >nul
