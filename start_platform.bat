@echo off
title Government Policy Feedback Analysis Platform

echo.
echo 🛠️ Government Policy Feedback Analysis Platform
echo =================================================
echo.

REM Check if virtual environment exists
if not exist ".venv" (
    echo ❌ Virtual environment not found. Please run setup_environment.bat first.
    pause
    exit /b 1
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call .venv\Scripts\activate.bat

REM Check if MongoDB is running (optional)
echo 📊 Checking database connection...
timeout /t 2 >nul

REM Start backend
echo 🚀 Starting backend server...
echo Backend will be available at: http://localhost:8000
echo API docs at: http://localhost:8000/docs
echo.

start "Backend API" cmd /k "uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000"

REM Wait for backend to start
echo ⏳ Waiting for backend to initialize...
timeout /t 8 >nul

REM Start frontend
echo 🌐 Starting frontend dashboard...
echo Dashboard will be available at: http://localhost:3000
echo.

cd frontend
start "Frontend Dashboard" cmd /k "npm start"
cd ..

echo ✅ Platform started successfully!
echo.
echo 🔗 Access URLs:
echo    Dashboard: http://localhost:3000
echo    Backend API: http://localhost:8000
echo    API Documentation: http://localhost:8000/docs
echo.
echo 📋 Default Login Credentials:
echo    Email: admin@gov.in
echo    Password: admin12345
echo.
echo 🎯 Features Available:
echo    ✓ Multilingual Sentiment Analysis (IndicBERT/MuRIL/XLM-R)
echo    ✓ Aspect-Based Sentiment Analysis (ABSA)
echo    ✓ Word Cloud Visualization (sentiment color-coded)
echo    ✓ Comment Explorer (search/filter/highlights)
echo    ✓ Explainability (LIME/SHAP highlights)
echo    ✓ Sarcasm Detection
echo    ✓ Human-in-the-Loop Moderation
echo    ✓ Summary Generation (mT5/IndicBART)
echo    ✓ PDF/Excel Export
echo    ✓ Language Representation Monitoring
echo    ✓ Role-Based Access Control
echo.
echo Press any key to open the dashboard in your browser...
pause >nul
start http://localhost:3000
echo.
echo Platform is running. Close this window to stop all services.
pause