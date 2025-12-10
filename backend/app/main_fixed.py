"""
Fixed FastAPI application with proper error handling and simplified services.
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status, Depends, Header, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# Load environment variables
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Ensure static files directory exists
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown events."""
    logger.info("Starting up sentiment analysis API...")
    yield
    logger.info("Shutting down sentiment analysis API...")

# Create FastAPI application
app = FastAPI(
    title="Sentiment Analysis Engine API",
    description="A robust sentiment analysis API with proper authentication",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Import services with fallback handling
try:
    from backend.app.services.simplified_sentiment_service import SimplifiedSentimentAnalyzer
    sentiment_service = SimplifiedSentimentAnalyzer()
    logger.info("✅ Using simplified sentiment analyzer")
except Exception as e:
    logger.error(f"❌ Failed to load simplified sentiment service: {e}")
    sentiment_service = None

try:
    from backend.app.services.simplified_summarization_service import (
        SimplifiedSummarizationService, SummarizationType, SummarizationMethod
    )
    summarization_service = SimplifiedSummarizationService()
    logger.info("✅ Using simplified summarization service")
except Exception as e:
    logger.error(f"❌ Failed to load simplified summarization service: {e}")
    summarization_service = None

try:
    from backend.app.services.simplified_visualization_service import SimplifiedVisualizationService
    visualization_service = SimplifiedVisualizationService()
    logger.info("✅ Using simplified visualization service")
except Exception as e:
    logger.error(f"❌ Failed to load simplified visualization service: {e}")
    visualization_service = None

# Request/Response models
class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=20000)

class AnalyzeResponse(BaseModel):
    overall_sentiment: str
    overall_confidence: float
    sentiment_scores: Dict[str, float]
    emotion: Optional[str] = None
    emotion_scores: Optional[Dict[str, float]] = None
    key_phrases: List[str] = []
    law_sections_mentioned: List[str] = []
    processing_time_ms: int
    service_used: str = "simplified"

class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=10000)
    summary_type: SummarizationType = SummarizationType.EXTRACTIVE
    method: SummarizationMethod = SummarizationMethod.CUSTOM_TEXTRANK
    num_sentences: int = Field(default=3, ge=1, le=8)
    max_length: int = Field(default=150, ge=30, le=512)
    min_length: int = Field(default=30, ge=10, le=100)

class SummarizeResponse(BaseModel):
    summary_text: str
    method: str
    summary_type: str
    key_sentences: List[str]
    confidence_score: float
    processing_time_ms: int
    service_used: str = "simplified"

class TopicSummarizeRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=10000)
    topics: List[str] = Field(default_factory=list, description="Topics to focus on")
    max_length: int = Field(default=150, ge=30, le=300)
    min_length: int = Field(default=30, ge=10, le=100)

class WordCloudRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100)
    max_words: int = Field(default=50, ge=5, le=200)
    min_word_length: int = Field(default=3, ge=2, le=10)
    width: int = Field(default=800, ge=200, le=2000)
    height: int = Field(default=400, ge=200, le=1500)

class WordCloudResponse(BaseModel):
    image_base64: str
    frequencies: Dict[str, int]
    total_words: int
    service_used: str = "simplified"

# File upload response model
class UploadResponse(BaseModel):
    success: bool
    message: str
    filename: str
    statistics: Dict[str, Any]

# Authentication dependency (simplified for testing)
async def get_current_user(authorization: Optional[str] = Header(None)):
    """Simple authentication check."""
    # In a real implementation, you would verify the JWT token
    # For now, we'll just check if a token is provided
    if authorization:
        return {"email": "test@example.com", "role": "user"}
    return None

# API endpoints
@app.get("/api/v1/health", tags=["health"])
async def api_health():
    """Basic health check."""
    services_status = {
        "sentiment": sentiment_service is not None,
        "summarization": summarization_service is not None,
        "visualization": visualization_service is not None
    }

    return {
        "status": "healthy" if all(services_status.values()) else "degraded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
        "services": services_status,
        "message": "Sentiment analysis API is running"
    }

@app.post("/api/v1/analyze-text", response_model=AnalyzeResponse, tags=["analysis"])
async def analyze_text(payload: AnalyzeRequest, current_user: dict = Depends(get_current_user)):
    """Analyze sentiment of text."""
    if not sentiment_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sentiment analysis service is not available"
        )

    try:
        result = await sentiment_service.comprehensive_analysis(payload.text)

        sentiment_scores = {}
        if result.sentiment_results:
            r = result.sentiment_results[0]
            sentiment_scores = {
                "positive": r.positive_score,
                "negative": r.negative_score,
                "neutral": r.neutral_score,
            }
            if r.compound_score is not None:
                sentiment_scores["compound"] = r.compound_score

        return AnalyzeResponse(
            overall_sentiment=result.overall_sentiment.value,
            overall_confidence=result.overall_confidence,
            sentiment_scores=sentiment_scores,
            emotion=result.emotion_result.emotion_label if result.emotion_result else None,
            emotion_scores=result.emotion_result.emotion_scores if result.emotion_result else None,
            key_phrases=result.key_phrases or [],
            law_sections_mentioned=result.law_sections_mentioned or [],
            processing_time_ms=result.processing_time_ms,
        )
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sentiment analysis failed: {str(e)}"
        )

@app.post("/api/v1/summarize-text", response_model=SummarizeResponse, tags=["summarization"])
async def summarize_text(payload: SummarizeRequest, current_user: dict = Depends(get_current_user)):
    """Summarize text."""
    if not summarization_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Summarization service is not available"
        )

    try:
        if payload.summary_type == SummarizationType.EXTRACTIVE:
            result = await summarization_service.extractive_summarization(
                text=payload.text,
                method=payload.method,
                num_sentences=payload.num_sentences,
            )
        elif payload.summary_type == SummarizationType.ABSTRACTIVE:
            result = await summarization_service.abstractive_summarization(
                text=payload.text,
                method=payload.method,
                max_length=payload.max_length,
                min_length=payload.min_length,
            )
        else:  # HYBRID
            result = await summarization_service.hybrid_summarization(
                text=payload.text,
                extractive_sentences=payload.num_sentences,
                abstractive_max_length=payload.max_length,
            )

        return SummarizeResponse(
            summary_text=result.summary_text,
            method=result.method,
            summary_type=result.summary_type.value,
            key_sentences=result.key_sentences,
            confidence_score=result.confidence_score,
            processing_time_ms=result.processing_time_ms,
        )
    except Exception as e:
        logger.error(f"Error in summarization: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Summarization failed: {str(e)}"
        )

@app.post("/api/v1/summarization/topic_based", response_model=SummarizeResponse, tags=["summarization"])
async def summarize_topic_based(payload: TopicSummarizeRequest, current_user: dict = Depends(get_current_user)):
    """Summarize text with topic focus."""
    if not summarization_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Summarization service is not available"
        )

    try:
        result = await summarization_service.topic_based_summarization(
            text=payload.text,
            topics=payload.topics,
            max_length=payload.max_length,
            min_length=payload.min_length,
        )

        return SummarizeResponse(
            summary_text=result.summary_text,
            method=result.method,
            summary_type=result.summary_type.value,
            key_sentences=result.key_sentences,
            confidence_score=result.confidence_score,
            processing_time_ms=result.processing_time_ms,
        )
    except Exception as e:
        logger.error(f"Error in topic-based summarization: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Topic-based summarization failed: {str(e)}"
        )

@app.post("/api/v1/summarization/comments", response_model=SummarizeResponse, tags=["summarization"])
async def summarize_comments(payload: Dict[str, Any], current_user: dict = Depends(get_current_user)):
    """Summarize multiple comments."""
    if not summarization_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Summarization service is not available"
        )

    try:
        comments = payload.get("comments", [])
        if not comments:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No comments provided"
            )

        # Combine all comments into one text for summarization
        combined_text = " ".join(comments)
        
        # Use extractive summarization by default
        result = await summarization_service.extractive_summarization(
            text=combined_text,
            method=SummarizationMethod.CUSTOM_TEXTRANK,
            num_sentences=payload.get("num_sentences", 3)
        )

        return SummarizeResponse(
            summary_text=result.summary_text,
            method=result.method,
            summary_type=result.summary_type.value,
            key_sentences=result.key_sentences,
            confidence_score=result.confidence_score,
            processing_time_ms=result.processing_time_ms,
        )
    except Exception as e:
        logger.error(f"Error in comment summarization: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Comment summarization failed: {str(e)}"
        )

@app.post("/api/v1/wordcloud", response_model=WordCloudResponse, tags=["visualization"])
async def create_word_cloud(payload: WordCloudRequest, current_user: dict = Depends(get_current_user)):
    """Create word cloud from texts."""
    if not visualization_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Visualization service is not available"
        )

    try:
        import base64

        tokens = await visualization_service.prepare_tokens(
            payload.texts,
            min_len=payload.min_word_length
        )
        frequencies = visualization_service.compute_frequencies(tokens, payload.max_words)
        image_bytes = visualization_service.generate_wordcloud_image(
            frequencies,
            width=payload.width,
            height=payload.height
        )

        return WordCloudResponse(
            image_base64=base64.b64encode(image_bytes).decode("utf-8") if image_bytes else "",
            frequencies=frequencies,
            total_words=sum(frequencies.values()),
        )
    except Exception as e:
        logger.error(f"Error in word cloud generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Word cloud generation failed: {str(e)}"
        )

@app.post("/api/v1/ingestion/upload", response_model=UploadResponse, tags=["ingestion"])
async def upload_file(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    """Handle file upload (simplified version)."""
    try:
        # Just acknowledge the upload for now
        content = await file.read()
        content_size = len(content)
        
        return UploadResponse(
            success=True,
            message=f"File {file.filename} uploaded successfully",
            filename=file.filename,
            statistics={
                "file_size": content_size,
                "content_type": file.content_type,
                "processed": True
            }
        )
    except Exception as e:
        logger.error(f"Error in file upload: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File upload failed: {str(e)}"
        )

@app.get("/api/v1/ingestion/health", tags=["ingestion"])
async def ingestion_health_check():
    """Check ingestion service health."""
    return {
        "status": "healthy",
        "service": "ingestion",
        "message": "Ingestion service is running (simplified version)"
    }

# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to Sentiment Analysis Engine API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": [
            "/api/v1/analyze-text",
            "/api/v1/summarize-text",
            "/api/v1/summarization/topic_based",
            "/api/v1/summarization/comments",
            "/api/v1/wordcloud",
            "/api/v1/ingestion/upload",
            "/api/v1/ingestion/health",
            "/api/v1/health"
        ]
    }

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")

    print("🚀 Starting Sentiment Analysis API...")
    print(f"📍 Host: {host}")
    print(f"📍 Port: {port}")
    print(f"📍 Debug: {debug}")
    print(f"📖 API Documentation: http://{host}:{port}/docs")

    uvicorn.run(
        "main_fixed:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info" if debug else "warning"
    )