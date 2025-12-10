@echo off
echo Setting up Government Policy Feedback Analysis Platform...
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.11 or 3.12 from https://python.org
    pause
    exit /b 1
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv .venv
if %errorlevel% neq 0 (
    echo Error: Failed to create virtual environment
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Upgrade pip and install basic tools
echo Upgrading pip and setuptools...
python -m pip install --upgrade pip setuptools wheel

REM Install core dependencies (excluding problematic ones)
echo Installing core dependencies...
pip install fastapi uvicorn[standard] pydantic pydantic-settings python-multipart
pip install motor pymongo beanie
pip install spacy nltk scikit-learn numpy pandas
pip install sentencepiece transformers

REM Install PyTorch separately for better compatibility
echo Installing PyTorch (CPU version for compatibility)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

REM Install remaining dependencies
echo Installing visualization and utility packages...
pip install wordcloud matplotlib plotly seaborn Pillow
pip install python-dotenv aiofiles httpx jinja2
pip install pytest pytest-asyncio black isort
pip install "python-jose[cryptography]" "passlib[bcrypt]"
pip install sumy textblob
pip install reportlab openpyxl fasttext langdetect lime shap
pip install sqlalchemy psycopg2-binary

REM Download spaCy model
echo Downloading spaCy English model...
python -m spacy download en_core_web_sm

REM Download NLTK data
echo Downloading NLTK data...
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"

echo.
echo Setup complete! Virtual environment is ready.
echo To activate in the future, run: .venv\Scripts\activate.bat
echo To start the backend, run: uvicorn backend.app.main:app --reload
echo.
pause