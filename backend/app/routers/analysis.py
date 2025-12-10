"""
Sentiment analysis API endpoints for comprehensive text analysis.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

# Remove SQLAlchemy dependencies and use MongoDB instead
from backend.app.core.mongo_auth import get_current_user
from backend.app.services.sentiment_service import SentimentAnalyzer, AnalysisMethod, ComprehensiveAnalysisResult
from backend.app.models.mongo_models import UserInDB

router = APIRouter()

# Initialize the sentiment analyzer (singleton)
sentiment_analyzer = SentimentAnalyzer()


class TextAnalysisRequest(BaseModel):
    """Request model for text analysis."""
    text: str = Field(..., description="Text to analyze", min_length=1, max_length=10000)
    methods: Optional[List[str]] = Field(
        default=["vader", "textblob"], 
        description="Analysis methods to use"
    )
    include_aspects: bool = Field(default=True, description="Include aspect-based sentiment analysis")
    include_emotions: bool = Field(default=True, description="Include emotion analysis")


class BatchAnalysisRequest(BaseModel):
    """Request model for batch text analysis."""
    texts: List[str] = Field(..., description="List of texts to analyze", min_items=1, max_items=50)
    methods: Optional[List[str]] = Field(
        default=["vader", "textblob"], 
        description="Analysis methods to use"
    )
    include_aspects: bool = Field(default=True, description="Include aspect-based sentiment analysis")
    include_emotions: bool = Field(default=True, description="Include emotion analysis")


class AnalysisResponse(BaseModel):
    """Response model for analysis results."""
    text: str
    sentiment: str
    confidence: float
    emotion: str
    emotion_confidence: float
    sentiment_scores: Dict[str, float]
    emotion_scores: Dict[str, float]
    aspect_sentiments: List[Dict[str, Any]]
    key_phrases: List[str]
    law_sections_mentioned: List[str]
    processing_time_ms: int
    explanation: Dict[str, Any]
    highlights: Optional[List[str]] = None  # Explainability highlights (LIME/SHAP)


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_text(
    request: TextAnalysisRequest,
    current_user: UserInDB = Depends(get_current_user)
):
    """
    Perform comprehensive sentiment and emotion analysis on text.
    
    This endpoint provides:
    - Multi-method sentiment analysis (VADER, TextBlob, ensemble)
    - Emotion classification (support, concern, suggestion, anger, etc.)
    - Aspect-based sentiment analysis for legal contexts
    - Key phrase extraction
    - Law section detection
    - Confidence scores and explanations
    """
    try:
        # Convert method names to enum values
        analysis_methods = []
        for method in request.methods:
            try:
                analysis_methods.append(AnalysisMethod(method.lower()))
            except ValueError:
                continue
        
        # Perform comprehensive analysis (transformer-enhanced via service)
        result = await sentiment_analyzer.comprehensive_analysis(request.text)
        
        # Format response
        sentiment_scores = {}
        if result.sentiment_results:
            primary_result = result.sentiment_results[0]
            sentiment_scores = {
                "positive": primary_result.positive_score,
                "negative": primary_result.negative_score,
                "neutral": primary_result.neutral_score
            }
            if primary_result.compound_score is not None:
                sentiment_scores["compound"] = primary_result.compound_score
        
        # Add explainability highlights (stub: use keywords from explanation if available)
        highlights = result.explanation.get("lime_highlights") or result.explanation.get("shap_highlights") or []
        return AnalysisResponse(
            text=result.text,
            sentiment=result.overall_sentiment.value,
            confidence=result.overall_confidence,
            emotion=result.emotion_result.emotion_label.value,
            emotion_confidence=result.emotion_result.confidence_score,
            sentiment_scores=sentiment_scores,
            emotion_scores=result.emotion_result.emotion_scores,
            aspect_sentiments=[
                {
                    "aspect": asp.aspect,
                    "sentiment": asp.sentiment.value,
                    "confidence": asp.confidence,
                    "context": asp.context,
                    "law_section": asp.law_section
                }
                for asp in result.aspect_sentiments
            ],
            key_phrases=result.key_phrases,
            law_sections_mentioned=result.law_sections_mentioned,
            processing_time_ms=result.processing_time_ms,
            explanation=result.explanation,
            highlights=highlights
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in sentiment analysis: {str(e)}"
        )


@router.post("/comprehensive")
async def analyze_text_comprehensive(
    request: TextAnalysisRequest,
    current_user: UserInDB = Depends(get_current_user)
):
    """
    Compatibility endpoint returning a richer structure expected by the dashboard.
    """
    try:
        # Perform comprehensive analysis
        result = await sentiment_analyzer.comprehensive_analysis(request.text)

        # Convert internal dataclasses to plain dicts
        sentiment_results = []
        for r in result.sentiment_results:
            sentiment_results.append({
                "method": r.method,
                "sentiment_label": r.sentiment_label.value,
                "confidence_score": r.confidence_score,
                "positive_score": r.positive_score,
                "negative_score": r.negative_score,
                "neutral_score": r.neutral_score,
                "compound_score": r.compound_score,
                "raw_scores": r.raw_scores,
            })

        emotion_result = {
            "emotion_label": result.emotion_result.emotion_label.value,
            "confidence_score": result.emotion_result.confidence_score,
            "emotion_scores": result.emotion_result.emotion_scores,
            "detected_emotions": result.emotion_result.detected_emotions,
        }

        aspect_sentiments = []
        for a in result.aspect_sentiments:
            aspect_sentiments.append({
                "aspect": a.aspect,
                "sentiment": a.sentiment.value,
                "confidence": a.confidence,
                "context": a.context,
                "law_section": a.law_section,
            })

        return {
            "text": result.text,
            "overall_sentiment": result.overall_sentiment.value,
            "overall_confidence": result.overall_confidence,
            "processing_time_ms": result.processing_time_ms,
            "sentiment_results": sentiment_results,
            "emotion_result": emotion_result,
            "aspect_sentiments": aspect_sentiments,
            "key_phrases": result.key_phrases,
            "law_sections_mentioned": result.law_sections_mentioned,
            "explanation": result.explanation,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in comprehensive analysis: {str(e)}"
        )


@router.post("/analyze-batch", response_model=List[AnalysisResponse])
async def analyze_batch(
    request: BatchAnalysisRequest,
    current_user: UserInDB = Depends(get_current_user)
):
    """
    Perform batch sentiment analysis on multiple texts.
    
    Processes up to 50 texts simultaneously with parallel processing for efficiency.
    """
    try:
        # Perform batch analysis
        results = await sentiment_analyzer.batch_analysis(request.texts)
        
        # Format responses
        responses = []
        for result in results:
            if result:
                sentiment_scores = {}
                if result.sentiment_results:
                    primary_result = result.sentiment_results[0]
                    sentiment_scores = {
                        "positive": primary_result.positive_score,
                        "negative": primary_result.negative_score,
                        "neutral": primary_result.neutral_score
                    }
                    if primary_result.compound_score is not None:
                        sentiment_scores["compound"] = primary_result.compound_score
                
                responses.append(AnalysisResponse(
                    text=result.text,
                    sentiment=result.overall_sentiment.value,
                    confidence=result.overall_confidence,
                    emotion=result.emotion_result.emotion_label.value,
                    emotion_confidence=result.emotion_result.confidence_score,
                    sentiment_scores=sentiment_scores,
                    emotion_scores=result.emotion_result.emotion_scores,
                    aspect_sentiments=[
                        {
                            "aspect": asp.aspect,
                            "sentiment": asp.sentiment.value,
                            "confidence": asp.confidence,
                            "context": asp.context,
                            "law_section": asp.law_section
                        }
                        for asp in result.aspect_sentiments
                    ],
                    key_phrases=result.key_phrases,
                    law_sections_mentioned=result.law_sections_mentioned,
                    processing_time_ms=result.processing_time_ms,
                    explanation=result.explanation
                ))
        
        return responses
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in batch sentiment analysis: {str(e)}"
        )


@router.post("/batch")
async def analyze_batch_compat(
    request: BatchAnalysisRequest,
    current_user: UserInDB = Depends(get_current_user)
):
    """
    Compatibility endpoint to match dashboard expected route and response shape.
    """
    try:
        results = await sentiment_analyzer.batch_analysis(request.texts)

        formatted = []
        for result in results:
            if not result:
                continue
            # Build sentiment scores for first method (if present)
            sentiment_results = []
            for r in result.sentiment_results:
                sentiment_results.append({
                    "method": r.method,
                    "sentiment_label": r.sentiment_label.value,
                    "confidence_score": r.confidence_score,
                    "positive_score": r.positive_score,
                    "negative_score": r.negative_score,
                    "neutral_score": r.neutral_score,
                    "compound_score": r.compound_score,
                    "raw_scores": r.raw_scores,
                })

            formatted.append({
                "text": result.text,
                "overall_sentiment": result.overall_sentiment.value,
                "overall_confidence": result.overall_confidence,
                "processing_time_ms": result.processing_time_ms,
                "sentiment_results": sentiment_results,
                "emotion_result": {
                    "emotion_label": result.emotion_result.emotion_label.value,
                    "confidence_score": result.emotion_result.confidence_score,
                    "emotion_scores": result.emotion_result.emotion_scores,
                    "detected_emotions": result.emotion_result.detected_emotions,
                },
                "aspect_sentiments": [
                    {
                        "aspect": asp.aspect,
                        "sentiment": asp.sentiment.value,
                        "confidence": asp.confidence,
                        "context": asp.context,
                        "law_section": asp.law_section,
                    }
                    for asp in result.aspect_sentiments
                ],
                "key_phrases": result.key_phrases,
                "law_sections_mentioned": result.law_sections_mentioned,
                "explanation": result.explanation,
            })

        return formatted
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in batch analysis: {str(e)}"
        )


@router.get("/methods")
async def get_analysis_methods():
    """
    Get information about available sentiment analysis methods.
    """
    return {
        "available_methods": [
            {
                "name": "vader",
                "description": "VADER (Valence Aware Dictionary and sEntiment Reasoner) - Rule-based sentiment analysis",
                "strengths": ["Fast", "Good for social media text", "Handles punctuation and capitalization"]
            },
            {
                "name": "textblob",
                "description": "TextBlob - Machine learning-based sentiment analysis",
                "strengths": ["Good for formal text", "Provides polarity and subjectivity"]
            },
            {
                "name": "spacy",
                "description": "spaCy - Token-based sentiment analysis",
                "strengths": ["Linguistically informed", "Good entity recognition"]
            },
            {
                "name": "ensemble",
                "description": "Ensemble method combining multiple approaches",
                "strengths": ["Higher accuracy", "Reduces individual method bias"]
            }
        ],
        "emotion_categories": [
            "support", "concern", "suggestion", "anger", "appreciation", "confusion", "neutral"
        ],
        "features": [
            "Multi-method sentiment analysis",
            "Emotion classification",
            "Aspect-based sentiment analysis",
            "Batch processing",
            "Explanation generation"
        ]
    }


@router.get("/health")
async def sentiment_analysis_health():
    """
    Health check for sentiment analysis service.
    """
    status_info = {
        "service": "sentiment_analysis",
        "status": "healthy",
        "components": {
            "vader_analyzer": sentiment_analyzer.vader_analyzer is not None,
            "spacy_model": sentiment_analyzer.nlp is not None,
            "preprocessor": sentiment_analyzer.preprocessor is not None
        },
        "capabilities": [
            "sentiment_analysis",
            "emotion_detection",
            "aspect_based_analysis",
            "batch_processing"
        ]
    }
    
    # Check if critical components are available
    if not sentiment_analyzer.vader_analyzer:
        status_info["status"] = "degraded"
        status_info["warnings"] = ["VADER analyzer not available"]
    
    return status_info