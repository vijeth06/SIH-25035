"""
Visualization API endpoints for word clouds, charts, and other visual analytics.
"""

from typing import List, Dict, Any, Optional, Tuple
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field

from backend.app.services.visualization_service import VisualizationService
from backend.app.core.mongo_auth import get_current_user, get_optional_current_user
from backend.app.models.user import User


router = APIRouter(prefix="/api/v1/visualization", tags=["visualization"])

# Initialize visualization service
visualization_service = VisualizationService()


# Request/Response Models
class WordCloudRequest(BaseModel):
    """Request model for word cloud generation."""
    texts: List[str] = Field(..., description="List of texts to analyze")
    analysis_results: List[Dict[str, Any]] = Field(..., description="Analysis results for sentiment tagging")
    max_words: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Maximum number of words in the word cloud"
    )
    width: int = Field(
        default=800,
        ge=400,
        le=2000,
        description="Width of the word cloud image"
    )
    height: int = Field(
        default=400,
        ge=300,
        le=1500,
        description="Height of the word cloud image"
    )


class WordCloudResponse(BaseModel):
    """Response model for word cloud generation."""
    image_data: str = Field(..., description="Base64 encoded PNG image data")
    word_data: Dict[str, Dict[str, Any]] = Field(..., description="Sentiment-tagged word data")
    metadata: Dict[str, Any] = Field(..., description="Metadata about the word cloud")


@router.post("/wordcloud/sentiment", response_model=WordCloudResponse)
async def create_sentiment_word_cloud(
    request: WordCloudRequest,
    current_user: Optional[User] = Depends(get_optional_current_user)
):
    """
    Create a sentiment-tagged word cloud with contextual examples.

    Generates a word cloud where words are colored by sentiment and includes
    contextual examples for hover-based exploration.
    """
    try:
        # Validate input
        if not request.texts:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No texts provided for word cloud generation"
            )

        if len(request.texts) > 1000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Too many texts (maximum 1000)"
            )

        if len(request.analysis_results) != len(request.texts):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Number of analysis results must match number of texts"
            )

        # Create sentiment-tagged word cloud
        image_bytes, word_data = await visualization_service.create_sentiment_tagged_wordcloud(
            texts=request.texts,
            analysis_results=request.analysis_results,
            width=request.width,
            height=request.height,
            max_words=request.max_words
        )

        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate word cloud"
            )

        # Convert bytes to base64 for JSON response
        import base64
        image_data = base64.b64encode(image_bytes).decode('utf-8')

        return WordCloudResponse(
            image_data=image_data,
            word_data=word_data,
            metadata={
                "total_texts": len(request.texts),
                "max_words": request.max_words,
                "image_size": f"{request.width}x{request.height}",
                "sentiment_tagged": True
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Word cloud generation failed: {str(e)}"
        )


@router.post("/wordcloud/basic", response_model=WordCloudResponse)
async def create_basic_word_cloud(
    request: WordCloudRequest,
    current_user: Optional[User] = Depends(get_optional_current_user)
):
    """
    Create a basic word cloud without sentiment tagging.

    Generates a standard word cloud based on word frequencies.
    """
    try:
        # Validate input
        if not request.texts:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No texts provided for word cloud generation"
            )

        # Prepare tokens and compute frequencies
        tokens = await visualization_service.prepare_tokens(request.texts)
        frequencies = visualization_service.compute_frequencies(tokens, request.max_words)

        if not frequencies:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid tokens found for word cloud generation"
            )

        # Generate word cloud image
        image_bytes = visualization_service.generate_wordcloud_image(
            frequencies, request.width, request.height
        )

        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate word cloud"
            )

        # Convert bytes to base64 for JSON response
        import base64
        image_data = base64.b64encode(image_bytes).decode('utf-8')

        return WordCloudResponse(
            image_data=image_data,
            word_data={},  # No detailed word data for basic word cloud
            metadata={
                "total_texts": len(request.texts),
                "max_words": request.max_words,
                "image_size": f"{request.width}x{request.height}",
                "sentiment_tagged": False
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Basic word cloud generation failed: {str(e)}"
        )


@router.get("/health")
async def visualization_health_check():
    """Check the health of visualization service."""
    try:
        # Test basic functionality
        test_texts = ["This is a test sentence for visualization health check."]
        tokens = await visualization_service.prepare_tokens(test_texts)

        return {
            "status": "healthy",
            "service": "visualization",
            "test_result": {
                "success": len(tokens) > 0,
                "tokens_found": len(tokens)
            }
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "visualization",
            "error": str(e)
        }