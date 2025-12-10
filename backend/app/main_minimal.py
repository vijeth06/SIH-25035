"""
Minimal FastAPI server for testing API connectivity.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
env_path = Path(__file__).resolve().parent.parent / ".env" 
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

app = FastAPI(
    title="E-Consultation Insight Engine - Minimal",
    description="Minimal API for testing connectivity",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "E-Consultation Insight Engine API", "status": "running"}

@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "E-Consultation Insight Engine",
        "version": "1.0.0"
    }

@app.post("/api/v1/analysis/sentiment")
async def analyze_sentiment(data: dict):
    """Mock sentiment analysis endpoint for testing."""
    text = data.get("text", "")
    
    # Simple mock sentiment analysis
    if "lacks clarity" in text.lower() or "compliance challenges" in text.lower():
        sentiment = "negative"
        confidence = 0.862
    elif "excellent" in text.lower() or "love these" in text.lower():
        sentiment = "positive" 
        confidence = 0.85
    else:
        sentiment = "neutral"
        confidence = 0.7
    
    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "text": text
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)