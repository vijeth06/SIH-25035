@echo off
echo ====================================================
echo   Enhanced MCA eConsultation Platform Launcher
echo ====================================================
echo.

echo Starting services...
echo.

echo 1. Starting Backend API Server...
start "Backend API" cmd /k "cd backend && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"

echo 2. Waiting for backend to initialize...
timeout /t 5 /nobreak > nul

echo 3. Starting Enhanced Dashboard...
start "Dashboard" cmd /k "streamlit run dashboard/main.py --server.port 8501"

echo.
echo ====================================================
echo   Services are starting up...
echo   Backend API: http://localhost:8000
echo   Dashboard: http://localhost:8501
echo   API Docs: http://localhost:8000/docs
echo ====================================================
echo.
echo Press any key to continue...
pause > nul

echo.
echo Opening dashboard in browser...
start http://localhost:8501

echo.
echo ====================================================
echo   Enhanced MCA eConsultation Platform is now running!
echo   
echo   Features Available:
echo   ✓ Advanced Sentiment Analysis
echo   ✓ Stakeholder Categorization  
echo   ✓ Batch Processing
echo   ✓ Legislative Context Analysis
echo   ✓ Interactive Visualizations
echo   ✓ Complete Text Analytics
echo   ✓ Real-time API Integration
echo ====================================================