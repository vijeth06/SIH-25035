"""
Main FastAPI application module.
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import uvicorn
from pathlib import Path
from dotenv import load_dotenv

"""Ensure package imports work regardless of working directory.

When running `uvicorn app.main:app` from within the `backend/` folder,
absolute imports like `from backend.app...` will fail because Python's
sys.path won't include the project root (the parent folder that contains
the `backend` package). Add the project root to sys.path early.
"""

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]  # .../SIH
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from .env file BEFORE importing settings
# Prefer project root .env; fallback to backend/.env if present
root_env = PROJECT_ROOT / ".env"
backend_env = CURRENT_FILE.parent.parent / ".env"
env_path = root_env if root_env.exists() else backend_env
load_dotenv(dotenv_path=env_path)

from backend.app.core.config import settings
from backend.app.core.database import MongoDB, init_db
from backend.app.routers import (
    auth, analysis, comments, visualization, 
    summarization, reports, advanced_analysis, health
)

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
    logger.info("Starting up...")
    db_disabled = os.getenv("DISABLE_DB", "").lower() in ("1", "true", "yes")
    if db_disabled:
        logger.info("Database initialization disabled via DISABLE_DB env var")
    else:
        try:
            await init_db()
            logger.info("Database connection established")
        except Exception as e:
            # Log but do not crash app in development/local runs
            logger.error(f"Failed to connect to database: {e}")

    yield

    logger.info("Shutting down...")
    try:
        await MongoDB.close_db()
        logger.info("Database connection closed")
    except Exception:
        pass

# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description="E-Consultation Insight Engine API",
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS,
)

# Mount static files
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Include routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["authentication"])
app.include_router(analysis.router, prefix="/api/v1/analysis", tags=["analysis"])
app.include_router(comments.router, prefix="/api/v1/comments", tags=["comments"])
app.include_router(visualization.router, prefix="/api/v1/visualization", tags=["visualization"])
app.include_router(summarization.router, prefix="/api/v1/summarization", tags=["summarization"])
app.include_router(reports.router, prefix="/api/v1/reports", tags=["reports"])
app.include_router(advanced_analysis.router, prefix="/api/v1/advanced", tags=["advanced"])
app.include_router(health.router, prefix="/api/v1/health", tags=["health"])

# Batch processing for high-volume comment analysis
from backend.app.routers import batch_processing
app.include_router(batch_processing.router, prefix="/api/v1/batch", tags=["batch-processing"])

# Stakeholder analysis for categorization and comparative analysis
from backend.app.routers import stakeholder_analysis
app.include_router(stakeholder_analysis.router, prefix="/api/v1/stakeholders", tags=["stakeholder-analysis"])

# Legislative context analysis for provision mapping and context-aware analysis
from backend.app.routers import legislative_context
app.include_router(legislative_context.router, prefix="/api/v1/legislative", tags=["legislative-context"])

# Legacy mongo auth for compatibility
from backend.app.routers import mongo_auth, ingestion
app.include_router(mongo_auth.router, prefix="/api/v1/mongo-auth", tags=["mongo-auth"])
app.include_router(ingestion.router, prefix="/api/v1/ingestion", tags=["ingestion"])

from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# Services for core features
from backend.app.services.sentiment_service import SentimentAnalyzer
from backend.app.services.summarization_service import (
    SummarizationService, SummarizationType, SummarizationMethod
)
from backend.app.services.visualization_service import VisualizationService

sentiment_service = SentimentAnalyzer()
summarization_service = SummarizationService()
visualization_service = VisualizationService()

# Health check endpoints compatible with tests
@app.get("/api/v1/health", tags=["health"])
async def api_health():
    """Basic health check with timestamp and version."""
    try:
        await MongoDB.connect_db()
        db_state = "connected"
        status_value = "healthy"
    except Exception:
        db_state = "disconnected"
        status_value = "unhealthy"
    return {
        "status": status_value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": settings.VERSION,
        "database": db_state,
    }

@app.get("/api/v1/health/detailed", tags=["health"])
async def api_health_detailed():
    """Detailed health providing database/services/configuration keys."""
    try:
        await MongoDB.connect_db()
        db_ok = True
    except Exception:
        db_ok = False

    return {
        "status": "healthy" if db_ok else "degraded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": settings.VERSION,
        "database": {"status": "ok" if db_ok else "error"},
        "services": {
            "api": "healthy",
            "database": "healthy" if db_ok else "unhealthy",
        },
        "configuration": {},
    }

@app.get("/api/v1/health/database", tags=["health"])
async def api_health_database():
    """Database-only health information."""
    try:
        await MongoDB.connect_db()
        db_ok = True
    except Exception:
        db_ok = False
    return {
        "database_healthy": db_ok,
        "database_info": {"name": settings.MONGODB_DB} if db_ok else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

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

class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=50)
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

class TopicSummarizeRequest(BaseModel):
    text: str = Field(..., min_length=50)
    topics: List[str] = Field(default_factory=list, description="Topics to focus on")
    max_length: int = Field(default=150, ge=30, le=300)
    min_length: int = Field(default=30, ge=10, le=100)

class WordCloudRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1)
    max_words: int = Field(default=100, ge=10, le=500)
    min_word_length: int = Field(default=3, ge=2, le=10)
    width: int = Field(default=800, ge=200, le=2000)
    height: int = Field(default=400, ge=200, le=2000)

class WordCloudResponse(BaseModel):
    image_base64: str
    frequencies: Dict[str, int]
    total_words: int

# Core feature endpoints (no DB/auth required)
@app.post("/api/v1/analyze-text", response_model=AnalyzeResponse, tags=["analysis"])
async def analyze_text(payload: AnalyzeRequest):
    result = await sentiment_service.comprehensive_analysis(payload.text)
    sentiment_scores: Dict[str, float] = {}
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
        emotion=result.emotion_result.emotion_label.value if result.emotion_result else None,
        emotion_scores=result.emotion_result.emotion_scores if result.emotion_result else None,
        key_phrases=result.key_phrases,
        law_sections_mentioned=result.law_sections_mentioned,
        processing_time_ms=result.processing_time_ms,
    )

@app.post("/api/v1/summarize-text", response_model=SummarizeResponse, tags=["summarization"])
async def summarize_text(payload: SummarizeRequest):
    if payload.summary_type == SummarizationType.EXTRACTIVE:
        res = await summarization_service.extractive_summarization(
            text=payload.text,
            method=payload.method,
            num_sentences=payload.num_sentences,
        )
    elif payload.summary_type == SummarizationType.ABSTRACTIVE:
        res = await summarization_service.abstractive_summarization(
            text=payload.text,
            method=payload.method,
            max_length=payload.max_length,
            min_length=payload.min_length,
        )
    else:
        res = await summarization_service.hybrid_summarization(
            text=payload.text,
            extractive_sentences=payload.num_sentences,
            abstractive_max_length=payload.max_length,
        )
    return SummarizeResponse(
        summary_text=res.summary_text,
        method=res.method,
        summary_type=res.summary_type.value,
        key_sentences=res.key_sentences,
        confidence_score=res.confidence_score,
        processing_time_ms=res.processing_time_ms,
    )

@app.post("/api/v1/wordcloud", response_model=WordCloudResponse, tags=["visualization"])
async def wordcloud(payload: WordCloudRequest):
    import base64
    tokens = await visualization_service.prepare_tokens(payload.texts, min_len=payload.min_word_length)
    freqs = visualization_service.compute_frequencies(tokens, max_words=payload.max_words)
    img_bytes = visualization_service.generate_wordcloud_image(freqs, width=payload.width, height=payload.height)
    return WordCloudResponse(
        image_base64=base64.b64encode(img_bytes).decode("utf-8"),
        frequencies=freqs,
        total_words=sum(freqs.values()),
    )

# Root endpoint compatible with tests
@app.get("/", tags=["root"])
async def root():
    return {
        "message": "Welcome to E-Consultation Insight Engine API",
        "version": settings.VERSION,
        "docs": "/docs",
        "redoc": "/redoc",
    }

@app.post("/api/v1/summarize-topic", response_model=SummarizeResponse, tags=["summarization"])
async def summarize_topic_based(payload: TopicSummarizeRequest):
    """Summarize text with topic-based focus using multilingual models."""
    res = await summarization_service.topic_based_summarization(
        text=payload.text,
        topics=payload.topics,
        max_length=payload.max_length,
        min_length=payload.min_length,
    )
    return SummarizeResponse(
        summary_text=res.summary_text,
        method=res.method,
        summary_type=res.summary_type.value,
        key_sentences=res.key_sentences,
        confidence_score=res.confidence_score,
        processing_time_ms=res.processing_time_ms,
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info" if settings.DEBUG else "warning"
    )