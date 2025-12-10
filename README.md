# MCA e-Consultation Insight Engine

**Smart India Hackathon 2025 - Problem Statement ID: 25035**

Government Policy Feedback Analysis System with Advanced Sentiment Analysis & Visualization Suite

---

## 🎯 Problem Statement

**Title:** Sentiment analysis of comments received through E-consultation module

### Background

eConsultation module is an online platform wherein proposed amendments/draft legislations are posted on MCA's website for external users to submit their comments and suggestions pertaining to the same through the MCA21 portal. The comments are captured in a structured format for due consideration with respect to amending the draft legislation, based on the suggestions or observations received.

### Problem Description

The draft document soliciting comments is made available for a specified period, during which any stakeholder may submit their observations either on the overall amendment or on specific provisions of the draft legislation. In instances where a substantial volume of comments is received on draft legislation, there exists a risk of certain observations being inadvertently overlooked or inadequately analysed. In order to review each individual submission, leveraging AI-assisted tools will help ensure that all remarks are duly considered and systematically analysed. 

**Requirement:** Development of an AI model aimed at predicting the sentiments of the suggestions provided by stakeholders in the eConsultation module. It should also generate a visual representation in the form of a word cloud, highlighting the keywords utilised by the stakeholders within their suggestions.

### Expected Outcome

The intention is to discern the feedback received from the stakeholders through the following:
- **Sentiment Analysis** - Classify comments as positive, neutral, or negative
- **Summary Generation** - Accurate and precise summarization of comments
- **Word Cloud** - Visual representation showcasing density of keywords used

The solution should considerably reduce the effort of the end user in analysing a high volume of comments. It should be able to clearly identify the sentiments of comments individually as well as broadly overall.

---

## 📋 Solution Overview

A comprehensive AI-powered sentiment analysis platform for government policy feedback, built with FastAPI backend and Streamlit dashboard. Supports multilingual analysis including Indian languages using IndicBERT, advanced NLP processing, and interactive visualizations.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12+ (virtual environment `.venv` pre-configured)
- Windows PowerShell 5.1+

### Run the Project

**1. Backend API (Port 8001)**
```powershell
cd backend
uvicorn app.main:app --reload --port 8001
```

**2. Streamlit Dashboard (Port 8501)**
```powershell
cd dashboard
streamlit run main.py
```

**3. Alternative API Endpoint (Port 8000)**
```powershell
python working_api.py
```

All three can run in parallel in separate terminal windows.

---

## 📁 Project Structure

```
SIH-25035/
├── backend/                    # FastAPI application
│   ├── app/                   # Core FastAPI app
│   ├── models/                # Database models
│   └── main.py               # Entry point
│
├── dashboard/                 # Streamlit application
│   ├── main.py               # Main dashboard
│   ├── pages/                # Multi-page sections
│   ├── components/           # Reusable components
│   ├── assets/               # CSS, theme files
│   └── services/             # API integration
│
├── .venv/                     # Python virtual environment
├── requirements.txt           # Python dependencies
├── working_api.py            # Alternative API server
└── README.md                 # This file
```

---

## 🎯 Features

### Sentiment Analysis
- **Multilingual Support**: English, Hindi, Tamil, Telugu, Kannada, Malayalam, and more
- **Advanced NLP**: IndicBERT for Indian language accuracy
- **Real-time Processing**: Single and batch analysis
- **Contextual Understanding**: Enhanced reasoning and emotion detection

### Visualizations
- Interactive word clouds (sentiment-coded)
- Timeline sentiment trends
- Aspect-based analysis charts
- Language distribution monitoring
- Comprehensive dashboards

### Data Management
- CSV file upload and processing
- Batch analysis with progress tracking
- Comment explorer with filtering
- Export results and reports

### Security & Access Control
- Role-based permissions (Admin, Analyst, Moderator)
- User authentication
- Secure API endpoints
- Request validation with Pydantic

---

## 🛠️ Technology Stack

**Backend**
- FastAPI (async web framework)
- Uvicorn (ASGI server)
- Motor (async MongoDB driver)
- Pydantic (data validation)
- Transformers + IndicBERT (NLP)

**Frontend & Dashboard**
- Streamlit (interactive dashboard)
- Material-UI inspired theme
- D3.js visualizations
- Chart.js for graphs

**Infrastructure**
- MongoDB (database)
- Python 3.12+
- Virtual Environment for isolation

---

## 🔧 Configuration

### Environment Variables
Create `.env` file in project root:
```env
MONGODB_URL=mongodb://localhost:27017
API_PORT=8001
STREAMLIT_PORT=8501
DEBUG=false
```

### Streamlit Config
Streamlit uses `.streamlit/config.toml` for browser and server settings. The app auto-reloads on file changes.

---

## 📊 API Endpoints

### Health & Info
- `GET /api/v1/health` – System health check
- `GET /` – API info

### Sentiment Analysis
- `POST /api/analyze` – Analyze single or batch text
- `POST /api/batch-analyze` – Bulk sentiment processing
- `GET /api/sentiment/{id}` – Get analysis results

### Data Management
- `POST /api/upload` – Upload CSV for processing
- `GET /api/results` – Fetch analysis results
- `GET /api/stats` – Dashboard statistics

Full documentation available at: `http://localhost:8001/docs`

---

## 📈 Performance Tips

### Backend
- Run with workers: `uvicorn app.main:app --workers 2`
- Ensure MongoDB indexes on filter/sort fields
- Cache repeated NLP predictions
- Lazy-load heavy ML models

### Dashboard
- Use `@st.cache_data` for API/DB reads
- Paginate large data tables
- Compress images (use SVG where possible)
- Avoid full reruns with `st.session_state`

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Port already in use | Change port: `--port 8002` for uvicorn or `--server.port 8502` for streamlit |
| Module not found | Activate venv: `. .venv/Scripts/Activate.ps1` |
| Slow sentiment analysis | Preload model on startup; use quantized/ONNX variants |
| Dashboard not updating | Refresh browser; check Streamlit logs |

---

## 🔐 Security

- Keep `.env` and secrets out of version control
- Use HTTPS in production
- Restrict CORS to known origins
- Implement rate limiting on public endpoints
- Validate and sanitize all user inputs
- Use strong database credentials

---

## 📦 Dependencies

All Python packages are listed in `requirements.txt`:
```
fastapi>=0.104.1
uvicorn[standard]>=0.24.0
streamlit>=1.30.0
transformers>=4.35.0
pymongo>=4.6.0
motor>=3.3.2
# ... and more
```

Install all dependencies:
```powershell
pip install -r requirements.txt
```

---

## 🤝 Contributing

1. Keep code clean and well-documented
2. Follow PEP 8 style guidelines
3. Test changes locally before committing
4. Ensure all features work end-to-end

---

## 📝 License

Smart India Hackathon 2025 - Ministry of Consumer Affairs Project

---

## 📧 Support

For issues, errors, or feedback:
- Check logs: `logs/` directory
- Review API docs: `http://localhost:8001/docs`
- Test endpoints with Swagger UI

---

**Last Updated**: December 10, 2025  
**Version**: 1.0.0  
**Status**: Production Ready
