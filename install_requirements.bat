@echo off
echo Installing Python dependencies...
pip install fastapi uvicorn python-dotenv python-multipart pydantic pydantic-settings
pip install motor pymongo dnspython
pip install pandas numpy openpyxl python-dateutil
pip install transformers torch torchvision torchaudio
pip install spacy
pip install -U spacy-lookups-data
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.0/en_core_web_lg-3.7.0-py3-none-any.whl
pip install nltk scikit-learn
pip install matplotlib seaborn wordcloud plotly
pip install streamlit streamlit-option-menu streamlit-aggrid streamlit-echarts
pip install python-jose[cryptography] passlib[bcrypt]
pip install tqdm python-slugify python-magic python-magic-bin

echo Creating necessary directories...
mkdir -p data\uploads
mkdir -p data\processed
mkdir -p models
mkdir -p logs

if not exist "backend\scripts" mkdir "backend\scripts"

if not exist "dashboard\.streamlit" mkdir "dashboard\.streamlit"

echo Installation complete!
echo.
echo To initialize the database, run:
echo    python backend\scripts\init_mongodb.py
echo.
echo Then start the backend with:
echo    cd backend
echo    uvicorn app.main:app --reload
echo.
echo In a new terminal, start the Streamlit dashboard with:
echo    cd dashboard
echo    streamlit run main.py
