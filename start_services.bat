@echo off
echo Starting E-Consultation Insight Engine Services...
echo =================================================

:: Check if virtual environment exists
if not exist ".venv" (
    echo Error: Virtual environment not found. Please run setup.bat first.
    pause
    exit /b 1
)

:: Activate virtual environment
call .venv\Scripts\activate
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to activate virtual environment
    pause
    exit /b 1
)

:: Start backend API in background
echo Starting backend API service...
start "Backend API" /min cmd /c "cd backend && python -m uvicorn app.main_fixed:app --host 0.0.0.0 --port 8000 --reload"

:: Wait a few seconds for backend to start
timeout /t 5 /nobreak >nul

:: Start Streamlit dashboard
echo Starting Streamlit dashboard...
cd dashboard
streamlit run main.py

:: If we get here, something went wrong
echo.
echo Services stopped.
pause