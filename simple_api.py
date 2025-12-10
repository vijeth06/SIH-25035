"""
Simple FastAPI server for testing API connectivity.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any

# Load environment variables
env_path = Path(__file__).resolve().parent.parent / ".env" 
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

app = FastAPI(
    title="E-Consultation Insight Engine API",
    description="Simple API for sentiment analysis",
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

class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    text: str

@app.get("/")
async def root():
    return {"message": "E-Consultation Insight Engine API", "status": "running", "port": 8002}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "E-Consultation API", "version": "1.0.0"}

@app.get("/api/v1/health")
async def api_health_check():
    return {"status": "healthy", "service": "E-Consultation API", "version": "1.0.0"}

@app.post("/api/v1/analysis/sentiment")
async def analyze_sentiment(request: SentimentRequest) -> SentimentResponse:
    """Basic sentiment analysis endpoint."""
    text = request.text.lower()
    
    # Simple keyword-based sentiment analysis for testing
    if any(word in text for word in ["lacks clarity", "compliance challenges", "disaster", "disappointed", "concerns", "disagree"]):
        sentiment = "negative"
        confidence = 0.862
    elif any(word in text for word in ["excellent", "love", "support", "great", "appreciate"]):
        sentiment = "positive"
        confidence = 0.85
    else:
        sentiment = "neutral"
        confidence = 0.7
    
    return SentimentResponse(
        sentiment=sentiment,
        confidence=confidence,
        text=request.text
    )

@app.post("/api/v1/analysis/bulk")
async def bulk_sentiment_analysis(data: Dict[str, List[str]]) -> Dict[str, Any]:
    """Bulk sentiment analysis for multiple texts."""
    texts = data.get("texts", [])
    results = []
    
    for text in texts:
        text_lower = text.lower()
        if any(word in text_lower for word in ["lacks clarity", "compliance challenges", "disaster", "disappointed", "concerns", "disagree"]):
            sentiment = "negative"
            confidence = 0.862
        elif any(word in text_lower for word in ["excellent", "love", "support", "great", "appreciate"]):
            sentiment = "positive"
            confidence = 0.85
        else:
            sentiment = "neutral"
            confidence = 0.7
        
        results.append({
            "sentiment": sentiment,
            "confidence": confidence,
            "text": text
        })
    
    return {"results": results, "count": len(results)}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting E-Consultation API on port 8002...")
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")