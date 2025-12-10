@echo off
setlocal enabledelayedexpansion

echo Initializing MongoDB database...

:: Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher and add it to your PATH
    pause
    exit /b 1
)

:: Read .env file if it exists
if not exist ".\.env" (
    echo Error: .env file not found in the current directory
    echo Please create a .env file with your MongoDB connection string
    pause
    exit /b 1
)

:: Set environment variables from .env file
for /f "tokens=1* delims==" %%a in (.\.env) do (
    set "%%a=%%b"
)

:: Check if MONGODB_URI is set
if "%MONGODB_URI%"=="" (
    echo Error: MONGODB_URI not found in .env file
    echo Please add MONGODB_URI to your .env file
    pause
    exit /b 1
)

echo Creating database and collections...
python backend\scripts\init_mongodb.py

if %ERRORLEVEL% neq 0 (
    echo Error initializing database
    pause
    exit /b 1
)

echo.
echo Database initialization complete!
echo.
echo You can now start the application:
echo 1. Start the backend:
echo    cd backend
echo    uvicorn app.main:app --reload
echo.
echo 2. In a new terminal, start the frontend:
echo    cd dashboard
echo    streamlit run main.py
echo.
pause
