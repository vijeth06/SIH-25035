@echo off
echo ============================================
echo  E-Consultation Platform - Quick Start
echo ============================================
echo.

REM Check if virtual environment exists
if not exist ".venv\" (
    echo ERROR: Virtual environment not found!
    echo Please run: python -m venv .venv
    pause
    exit /b 1
)

echo [1/3] Starting Backend API (Port 8001)...
start "Backend API" cmd /k "cd backend && ..\\.venv\\Scripts\\activate && uvicorn app.main:app --reload --port 8001 --host 0.0.0.0"

echo [2/3] Waiting for backend to initialize...
timeout /t 5 /nobreak >nul

echo [3/3] Starting Streamlit Dashboard (Port 8501)...
start "Dashboard" cmd /k ".venv\\Scripts\\activate && streamlit run dashboard\\main.py --server.port 8501"

echo.
echo ============================================
echo  All Services Started Successfully!
echo ============================================
echo.
echo  Backend API:  http://localhost:8001
echo  API Docs:     http://localhost:8001/docs
echo  Dashboard:    http://localhost:8501
echo.
echo  Press any key to view logs or Ctrl+C to exit
pause >nul
