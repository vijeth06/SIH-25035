@echo off
echo Setting up E-Consultation Insight Engine...
echo =========================================

:: Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher and add it to your PATH
    pause
    exit /b 1
)

:: Create and activate virtual environment
echo.
echo Creating Python virtual environment...
python -m venv .venv
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to create virtual environment
    pause
    exit /b 1
)

call .venv\Scripts\activate
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to activate virtual environment
    pause
    exit /b 1
)

:: Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to upgrade pip
    pause
    exit /b 1
)

:: Install dependencies
echo.
echo Installing dependencies...
pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)

:: Install spaCy model
echo.
echo Downloading spaCy language model...
python -m spacy download en_core_web_lg
if %ERRORLEVEL% neq 0 (
    echo Warning: Failed to download spaCy model. Some features may not work correctly.
)

:: Initialize database
echo.
echo Initializing database...
cd backend
python init_db.py
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to initialize database
    cd ..
    pause
    exit /b 1
)
cd ..

:: Create .streamlit directory if it doesn't exist
if not exist "dashboard\.streamlit" mkdir "dashboard\.streamlit"

:: Create secrets.toml if it doesn't exist
if not exist "dashboard\.streamlit\secrets.toml" (
    echo [server] > "dashboard\.streamlit\secrets.toml"
    echo port = 8501 >> "dashboard\.streamlit\secrets.toml"
    echo. >> "dashboard\.streamlit\secrets.toml"
    echo [api] >> "dashboard\.streamlit\secrets.toml"
    echo url = "http://localhost:8000" >> "dashboard\.streamlit\secrets.toml"
)

echo.
echo =========================================
echo Setup completed successfully!
echo.
echo To start the application:
echo 1. Start the backend API:
echo    cd backend
echo    uvicorn app.main:app --reload
echo.
echo 2. In a new terminal, start the Streamlit dashboard:
echo    cd dashboard
echo    streamlit run main.py
echo.
echo 3. Open your browser to http://localhost:8501
echo.
pause
