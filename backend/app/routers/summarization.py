"""
Summarization API endpoints for extractive, abstractive, and hybrid summarization.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi import UploadFile, File, Form
from pydantic import BaseModel, Field

from backend.app.services.summarization_service import (
    SummarizationService, SummarizationType, SummarizationMethod,
    SummaryResult, AggregateSummaryResult
)
from backend.app.core.mongo_auth import get_current_user, get_optional_current_user
from backend.app.models.mongo_models import UserInDB
from backend.app.services.document_reader import read_text_from_upload
from backend.app.services.preprocessing_service import TextPreprocessor


router = APIRouter(prefix="/api/v1/summarization", tags=["summarization"])

# Initialize summarization service
summarization_service = SummarizationService()


# Request/Response Models
class SummarizeTextRequest(BaseModel):
    """Request model for text summarization."""
    text: str = Field(..., description="Text to summarize")
    method: SummarizationMethod = Field(
        default=SummarizationMethod.CUSTOM_TEXTRANK,
        description="Summarization method to use"
    )
    summary_type: SummarizationType = Field(
        default=SummarizationType.EXTRACTIVE,
        description="Type of summarization"
    )
    num_sentences: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of sentences for extractive summarization"
    )
    max_length: int = Field(
        default=150,
        ge=30,
        le=512,
        description="Maximum length for abstractive summarization"
    )
    min_length: int = Field(
        default=30,
        ge=10,
        le=100,
        description="Minimum length for abstractive summarization"
    )


class BatchSummarizeRequest(BaseModel):
    """Request model for batch summarization."""
    texts: List[str] = Field(..., description="List of texts to summarize")
    method: SummarizationMethod = Field(
        default=SummarizationMethod.CUSTOM_TEXTRANK,
        description="Summarization method to use"
    )
    summary_type: SummarizationType = Field(
        default=SummarizationType.EXTRACTIVE,
        description="Type of summarization"
    )
    num_sentences: int = Field(default=3, ge=1, le=10)
    max_length: int = Field(default=150, ge=30, le=512)


class SummarizeCommentsRequest(BaseModel):
    """Request model for comment summarization."""
    comments: List[str] = Field(..., description="List of comments to summarize")
    method: SummarizationMethod = Field(
        default=SummarizationMethod.CUSTOM_TEXTRANK,
        description="Summarization method to use"
    )
    summary_type: SummarizationType = Field(
        default=SummarizationType.EXTRACTIVE,
        description="Type of summarization"
    )


class AggregateSummaryRequest(BaseModel):
    """Request model for aggregate summarization."""
    comments_by_section: Dict[str, List[str]] = Field(
        ...,
        description="Comments organized by law section"
    )
    sentiments_by_section: Optional[Dict[str, List[str]]] = Field(
        default=None,
        description="Sentiment labels by section"
    )


class SummaryResultResponse(BaseModel):
    """Response model for summary result."""
    method: str
    summary_type: str
    summary_text: str
    original_length: int
    summary_length: int
    compression_ratio: float
    key_sentences: List[str]
    confidence_score: float
    processing_time_ms: int
    metadata: Dict[str, Any]


class AggregateSummaryResponse(BaseModel):
    """Response model for aggregate summary."""
    section_summaries: Dict[str, SummaryResultResponse]
    overall_summary: SummaryResultResponse
    key_themes: List[str]
    sentiment_distribution: Dict[str, int]
    total_comments: int
    processing_statistics: Dict[str, Any]


class ConciseCommentsRequest(BaseModel):
    """Request model for concise summarization of comments into 1–2 sentences."""
    comments: List[str] = Field(..., description="List of stakeholder comments")
    max_length: int = Field(80, ge=40, le=160, description="Max length of concise summary")
@router.post("/file")
async def summarize_uploaded_file(
    file: UploadFile = File(...),
    max_length: int = Form(180),
    current_user: Optional[UserInDB] = Depends(get_optional_current_user)
):
    """
    Upload a PDF/DOCX/TXT administrative document and get a concise summary.
    Uses Indian-first models (IndicBART/mT5) and hierarchical summarization for long files.
    """
    try:
        text, doc_type = read_text_from_upload(file)
        if not text or len(text.strip()) < 50:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File appears empty or unsupported")

        # Cap input size for safety (approx 100k chars)
        if len(text) > 100_000:
            text = text[:100_000]

        # Detect primary language
        try:
            pre = TextPreprocessor()
            lang, lang_conf = pre._detect_language(text)
        except Exception:
            lang, lang_conf = "unknown", 0.0

        # Use hierarchical summarizer for long docs
        if len(text) > 2000:
            result = await summarization_service.summarize_long_document(text, preferred_max_length=max_length)
        else:
            # Short docs: use Indic-first single pass
            result = await summarization_service.abstractive_summarization(text, method=await summarization_service._select_indic_first_model(text), max_length=max_length, min_length=40)

        return {
            "status": "success",
            "document_type": doc_type,
            "original_length": len(text),
            "detected_language": lang,
            "language_confidence": lang_conf,
            "summary": result.summary_text,
            "summary_length": result.summary_length,
            "compression_ratio": result.compression_ratio,
            "confidence_score": result.confidence_score,
            "metadata": result.metadata
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"File summarization failed: {str(e)}")


@router.post("/concise-comments")
async def concise_comments_summary(
    request: ConciseCommentsRequest,
    current_user: Optional[UserInDB] = Depends(get_optional_current_user)
):
    """
    Generate a short, 1–2 sentence summary capturing overall sentiment and key concerns from multiple comments.
    Uses Indian-first models (IndicBART/mT5) and compresses long inputs via hierarchical summarization.
    """
    try:
        comments = [c.strip() for c in request.comments if isinstance(c, str) and c.strip()]
        if not comments:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No valid comments provided")

        combined = "\n".join(comments)

        # Detect primary language for reporting
        try:
            pre = TextPreprocessor()
            lang, lang_conf = pre._detect_language(combined)
        except Exception:
            lang, lang_conf = "unknown", 0.0

        instruction = (
            "In 1–2 sentences, summarize the overall sentiment and key concerns of the public feedback succinctly: "
        )
        text_for_summary = f"{instruction}\n{combined}"

        # Choose summarization path based on size
        if len(text_for_summary) > 2000:
            result = await summarization_service.summarize_long_document(
                text_for_summary, preferred_max_length=request.max_length
            )
        else:
            method = await summarization_service._select_indic_first_model(text_for_summary)
            result = await summarization_service.abstractive_summarization(
                text_for_summary, method=method, max_length=request.max_length, min_length=40
            )

        return {
            "status": "success",
            "summary_concise": result.summary_text,
            "detected_language": lang,
            "language_confidence": lang_conf,
            "total_comments": len(comments),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Concise comments summarization failed: {str(e)}")



# Helper function to convert service result to response
def summary_result_to_response(result: SummaryResult) -> SummaryResultResponse:
    """Convert SummaryResult to response model."""
    return SummaryResultResponse(
        method=result.method,
        summary_type=result.summary_type.value,
        summary_text=result.summary_text,
        original_length=result.original_length,
        summary_length=result.summary_length,
        compression_ratio=result.compression_ratio,
        key_sentences=result.key_sentences,
        confidence_score=result.confidence_score,
        processing_time_ms=result.processing_time_ms,
        metadata=result.metadata
    )


@router.post("/text", response_model=SummaryResultResponse)
async def summarize_text(
    request: SummarizeTextRequest,
    current_user: Optional[UserInDB] = Depends(get_optional_current_user)
):
    """
    Summarize a single text using specified method and type.
    
    Supports both extractive and abstractive summarization with multiple algorithms.
    """
    try:
        # Validate text length
        if len(request.text.strip()) < 50:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text too short for meaningful summarization (minimum 50 characters)"
            )
        
        if len(request.text) > 50000:  # ~10k words
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text too long (maximum 50,000 characters)"
            )
        
        # Perform summarization based on type
        if request.summary_type == SummarizationType.EXTRACTIVE:
            result = await summarization_service.extractive_summarization(
                text=request.text,
                method=request.method,
                num_sentences=request.num_sentences
            )
        elif request.summary_type == SummarizationType.ABSTRACTIVE:
            result = await summarization_service.abstractive_summarization(
                text=request.text,
                method=request.method,
                max_length=request.max_length,
                min_length=request.min_length
            )
        else:  # HYBRID
            result = await summarization_service.hybrid_summarization(
                text=request.text,
                extractive_sentences=request.num_sentences,
                abstractive_max_length=request.max_length
            )
        
        return summary_result_to_response(result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Summarization failed: {str(e)}"
        )


@router.post("/batch", response_model=List[SummaryResultResponse])
async def batch_summarize(
    request: BatchSummarizeRequest,
    current_user: Optional[UserInDB] = Depends(get_optional_current_user)
):
    """
    Perform batch summarization on multiple texts.
    
    Processes multiple texts concurrently for improved performance.
    """
    try:
        # Validate batch size
        if len(request.texts) > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Batch size too large (maximum 100 texts)"
            )
        
        if not request.texts:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No texts provided for summarization"
            )
        
        # Filter out very short texts
        valid_texts = [text for text in request.texts if len(text.strip()) >= 50]
        
        if not valid_texts:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="All texts too short for summarization (minimum 50 characters each)"
            )
        
        # Perform batch summarization
        results = await summarization_service.batch_summarization(
            texts=valid_texts,
            method=request.method,
            summary_type=request.summary_type
        )
        
        return [summary_result_to_response(result) for result in results]
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch summarization failed: {str(e)}"
        )


@router.post("/comments", response_model=SummaryResultResponse)
async def summarize_comments(
    request: SummarizeCommentsRequest,
    current_user: Optional[UserInDB] = Depends(get_optional_current_user)
):
    """
    Summarize multiple comments into a single comprehensive summary.
    
    Combines all comments and generates a unified summary capturing key themes.
    """
    try:
        if not request.comments:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No comments provided for summarization"
            )
        
        if len(request.comments) > 500:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Too many comments (maximum 500)"
            )
        
        # Filter out very short comments
        valid_comments = [comment for comment in request.comments if len(comment.strip()) >= 20]
        
        if not valid_comments:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="All comments too short for summarization (minimum 20 characters each)"
            )
        
        # Perform comment summarization
        result = await summarization_service.summarize_comments(
            comments=valid_comments,
            method=request.method,
            summary_type=request.summary_type
        )
        
        return summary_result_to_response(result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Comment summarization failed: {str(e)}"
        )


@router.post("/aggregate", response_model=AggregateSummaryResponse)
async def aggregate_summarization(
    request: AggregateSummaryRequest,
    current_user: UserInDB = Depends(get_current_user)  # Requires authentication for aggregate analysis
):
    """
    Create aggregate summaries per law section and overall summary.
    
    Generates section-wise summaries and an overall summary across all sections.
    Requires authentication as this is typically used for comprehensive analysis.
    """
    try:
        if not request.comments_by_section:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No comments provided for aggregate summarization"
            )
        
        # Validate section data
        total_comments = sum(len(comments) for comments in request.comments_by_section.values())
        if total_comments > 1000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Too many total comments for aggregate analysis (maximum 1000)"
            )
        
        # Filter out sections with insufficient comments
        filtered_sections = {
            section: comments 
            for section, comments in request.comments_by_section.items()
            if len(comments) > 0 and any(len(comment.strip()) >= 20 for comment in comments)
        }
        
        if not filtered_sections:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No sections have sufficient comments for summarization"
            )
        
        # Perform aggregate summarization
        result = await summarization_service.aggregate_summarization(
            comments_by_section=filtered_sections,
            sentiments_by_section=request.sentiments_by_section
        )
        
        # Convert to response format
        section_summaries_response = {
            section: summary_result_to_response(summary)
            for section, summary in result.section_summaries.items()
        }
        
        return AggregateSummaryResponse(
            section_summaries=section_summaries_response,
            overall_summary=summary_result_to_response(result.overall_summary),
            key_themes=result.key_themes,
            sentiment_distribution=result.sentiment_distribution,
            total_comments=result.total_comments,
            processing_statistics=result.processing_statistics
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Aggregate summarization failed: {str(e)}"
        )


@router.get("/methods", response_model=Dict[str, List[str]])
async def get_available_methods():
    """
    Get available summarization methods and types.
    
    Returns information about supported algorithms and approaches.
    """
    try:
        return {
            "summarization_types": [item.value for item in SummarizationType],
            "extractive_methods": [
                SummarizationMethod.TEXTRANK.value,
                SummarizationMethod.LSA.value,
                SummarizationMethod.LUHN.value,
                SummarizationMethod.EDMUNDSON.value,
                SummarizationMethod.CUSTOM_TEXTRANK.value
            ],
            "abstractive_methods": [
                SummarizationMethod.T5.value,
                SummarizationMethod.BART.value
            ],
            "recommended": {
                "general_purpose": SummarizationMethod.CUSTOM_TEXTRANK.value,
                "high_quality": SummarizationMethod.T5.value,
                "fast_processing": SummarizationMethod.TEXTRANK.value,
                "best_hybrid": "custom_textrank + t5"
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get available methods: {str(e)}"
        )


@router.get("/health")
async def summarization_health_check():
    """Check the health of summarization service."""
    try:
        # Test basic functionality
        test_text = "This is a test sentence for summarization health check. It contains multiple sentences to verify the service is working correctly. The system should be able to process this text without errors."
        
        result = await summarization_service.extractive_summarization(
            test_text, SummarizationMethod.CUSTOM_TEXTRANK, 1
        )
        
        return {
            "status": "healthy",
            "service": "summarization",
            "available_methods": len([item.value for item in SummarizationMethod]),
            "test_result": {
                "success": result is not None,
                "processing_time_ms": result.processing_time_ms if result else 0
            }
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "summarization",
            "error": str(e)
        }


# Topic-Based Summarization Request Model
class TopicBasedSummarizeRequest(BaseModel):
    """Request model for topic-based summarization."""
    text: str = Field(..., description="Text to summarize")
    topics: List[str] = Field(default_factory=list, description="Topics to focus on")
    max_length: int = Field(
        default=150,
        ge=30,
        le=300,
        description="Maximum length of the summary"
    )
    min_length: int = Field(
        default=30,
        ge=10,
        le=100,
        description="Minimum length of the summary"
    )


@router.post("/topic_based", response_model=SummaryResultResponse)
async def topic_based_summarize(
    request: TopicBasedSummarizeRequest,
    current_user: Optional[UserInDB] = Depends(get_optional_current_user)
):
    """
    Perform topic-based summarization using multilingual models.

    Uses mT5 or IndicBART models based on detected language, with topic guidance.
    """
    try:
        # Validate input
        if len(request.text.strip()) < 50:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text too short for meaningful summarization (minimum 50 characters)"
            )

        if len(request.text) > 50000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text too long (maximum 50,000 characters)"
            )

        # Perform topic-based summarization
        result = await summarization_service.topic_based_summarization(
            text=request.text,
            topics=request.topics,
            max_length=request.max_length,
            min_length=request.min_length
        )

        return summary_result_to_response(result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Topic-based summarization failed: {str(e)}"
        )