"""
Robust API Server that doesn't shutdown on request
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List
import uvicorn
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="E-Consultation Sentiment API",
    description="Robust API for sentiment analysis",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str

class BatchTextRequest(BaseModel):
    texts: List[str]

@app.get("/")
async def root():
    return {"message": "E-Consultation API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API is operational"}

@app.get("/api/v1/health")
async def health_check_v1():
    return {"status": "healthy", "message": "API v1 is operational", "version": "2.0.0"}

@app.post("/api/v1/analysis/sentiment")
async def analyze_sentiment(request: TextRequest):
    """Analyze sentiment of single text"""
    try:
        # Import enhanced sentiment analysis
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        from enhanced_sentiment_reasoning import analyze_text_with_enhanced_reasoning
        
        result = analyze_text_with_enhanced_reasoning(request.text)
        
        return {
            "text": request.text,
            "sentiment": result["sentiment"].lower(),
            "confidence": result["confidence"].lower() if isinstance(result["confidence"], str) else result["confidence"],
            "polarity_score": result["polarity_score"],
            "explanation": result["explanation"],
            "key_indicators": result["key_indicators"],
            "method": "enhanced_reasoning"
        }
        
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        # Fallback to basic analysis
        from textblob import TextBlob
        blob = TextBlob(request.text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            sentiment = "positive"
        elif polarity < -0.1:
            sentiment = "negative"
        else:
            sentiment = "neutral"
            
        return {
            "text": request.text,
            "sentiment": sentiment,
            "confidence": "medium",
            "polarity_score": polarity,
            "explanation": f"Basic TextBlob analysis: {sentiment} sentiment detected",
            "key_indicators": {
                "positive_words": [],
                "negative_words": [],
                "neutral_words": [],
                "intensifiers": [],
                "negations": []
            },
            "method": "textblob_fallback"
        }

@app.post("/api/v1/analysis/batch")
async def analyze_batch_sentiment(request: BatchTextRequest):
    """Analyze sentiment of multiple texts"""
    try:
        results = []
        for text in request.texts:
            single_request = TextRequest(text=text)
            result = await analyze_sentiment(single_request)
            results.append(result)
        
        return {
            "results": results,
            "total_analyzed": len(results),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Batch analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

@app.post("/api/analyze")
async def legacy_analyze(data: Dict[str, Any]):
    """Legacy analysis endpoint for backward compatibility"""
    try:
        if "comments" in data:
            # Batch analysis
            texts = data["comments"]
            batch_request = BatchTextRequest(texts=texts)
            result = await analyze_batch_sentiment(batch_request)
            return result
        elif "text" in data:
            # Single analysis
            text_request = TextRequest(text=data["text"])
            result = await analyze_sentiment(text_request)
            return result
        else:
            raise HTTPException(status_code=400, detail="Invalid request format")
            
    except Exception as e:
        logger.error(f"Legacy analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/api/status")
async def api_status():
    """Get API status information"""
    return {
        "status": "operational",
        "version": "2.0.0",
        "endpoints": [
            "/health",
            "/api/v1/health", 
            "/api/v1/analysis/sentiment",
            "/api/v1/analysis/batch",
            "/api/analyze"
        ],
        "features": [
            "Enhanced sentiment reasoning",
            "Batch processing",
            "Detailed explanations",
            "Word highlighting",
            "Legacy compatibility"
        ]
    }

# Keep-alive endpoint
@app.get("/ping")
async def ping():
    return {"message": "pong", "status": "alive"}

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    try:
        port = int(os.getenv("PORT", "8000"))
    except ValueError:
        port = 8000

    print("🚀 Starting Robust E-Consultation API...")
    print(f"📡 Server will run persistently on {host}:{port}")
    print("🔧 Enhanced with proper error handling and fallbacks")
    
    try:
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
            access_log=True,
            reload=False,
            timeout_keep_alive=30
        )
    except KeyboardInterrupt:
        print("🛑 API server stopped by user")
    except Exception as e:
        print(f"❌ API server error: {e}")
        import sys
        sys.exit(1)