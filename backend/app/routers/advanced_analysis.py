"""
Advanced Analysis Router

This module provides API endpoints for advanced sentiment analysis features.
"""

from fastapi import APIRouter, HTTPException, Depends, status, Header
from typing import List, Optional
from pydantic import BaseModel, Field

from backend.app.services.advanced_sentiment_service import (
    advanced_analyzer,
    AdvancedSentimentResult
)
from backend.app.core.mongo_auth import get_current_user
from backend.app.models.mongo_models import UserInDB

router = APIRouter(
    tags=["Advanced Analysis"],
    responses={404: {"description": "Not found"}},
)

# Dependency to get current user from Authorization header
async def get_current_active_user(authorization: str = Header(...)) -> UserInDB:
    """Extract and verify current user from Authorization header."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = authorization[7:]  # Remove "Bearer " prefix
    user = await get_current_user(token)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return user

class BatchAnalysisRequest(BaseModel):
    """Request model for batch analysis."""
    texts: List[str] = Field(..., min_items=1, max_items=1000)
    include_emotions: bool = True
    include_aspects: bool = True
    detect_sarcasm: bool = True

class AnalysisResponse(BaseModel):
    """Response model for analysis results."""
    success: bool
    data: Optional[AdvancedSentimentResult] = None
    error: Optional[str] = None

class BatchAnalysisResponse(BaseModel):
    """Response model for batch analysis."""
    success: bool
    results: List[AdvancedSentimentResult]
    total: int

@router.post("/analyze", response_model=AnalysisResponse, response_model_exclude_none=True)
async def analyze_text(
    text: str,
    current_user: UserInDB = Depends(get_current_active_user)
) -> AnalysisResponse:
    """
    Perform advanced sentiment analysis on a single text.
    
    - **text**: The text to analyze
    - **returns**: Detailed sentiment analysis result
    """
    try:
        result = advanced_analyzer.analyze_sentiment(text)
        return AnalysisResponse(success=True, data=result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )

@router.post("/analyze/batch", response_model=BatchAnalysisResponse, response_model_exclude_none=True)
async def analyze_batch(
    request: BatchAnalysisRequest,
    current_user: UserInDB = Depends(get_current_active_user)
) -> BatchAnalysisResponse:
    """
    Perform advanced sentiment analysis on multiple texts in batch.
    
    - **texts**: List of texts to analyze (max 1000)
    - **include_emotions**: Whether to include emotion analysis
    - **include_aspects**: Whether to include aspect-based analysis
    - **detect_sarcasm**: Whether to detect sarcasm
    - **returns**: List of analysis results
    """
    try:
        # Configure analyzer based on request
        # (Implementation details would go here based on the analyzer's capabilities)
        
        # Process texts in batches to avoid memory issues
        batch_size = 50
        all_results = []
        
        for i in range(0, len(request.texts), batch_size):
            batch = request.texts[i:i + batch_size]
            results = await advanced_analyzer.analyze_batch(batch)
            all_results.extend(results)
        
        return BatchAnalysisResponse(
            success=True,
            results=all_results,
            total=len(all_results)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch analysis failed: {str(e)}"
        )

@router.get("/languages")
async def get_supported_languages() -> dict:
    """
    Get list of supported languages for analysis.
    """
    return {
        "success": True,
        "languages": [
            {"code": "en", "name": "English"},
            {"code": "es", "name": "Spanish"},
            {"code": "fr", "name": "French"},
            {"code": "de", "name": "German"},
            {"code": "it", "name": "Italian"},
            {"code": "pt", "name": "Portuguese"},
            {"code": "ru", "name": "Russian"},
            {"code": "zh", "name": "Chinese"},
            {"code": "ja", "name": "Japanese"},
            {"code": "hi", "name": "Hindi"},
        ]
    }