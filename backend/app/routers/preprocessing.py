"""
Text preprocessing API endpoints for cleaning and analyzing text.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from backend.app.core.database import get_db
from backend.app.core.security import get_current_active_user
from backend.app.services.preprocessing_service import TextPreprocessor, PreprocessingResult
from backend.app.models.user import User

router = APIRouter()

# Initialize the preprocessor (singleton)
text_preprocessor = TextPreprocessor()


class TextInput(BaseModel):
    """Input model for text preprocessing."""
    text: str = Field(..., description="Text to preprocess", min_length=1, max_length=10000)
    enable_spell_correction: bool = Field(default=False, description="Enable spell correction")
    remove_stopwords: bool = Field(default=True, description="Remove stopwords")
    lemmatize: bool = Field(default=True, description="Apply lemmatization")


class BatchTextInput(BaseModel):
    """Input model for batch text preprocessing."""
    texts: List[str] = Field(..., description="List of texts to preprocess", min_items=1, max_items=100)
    enable_spell_correction: bool = Field(default=False, description="Enable spell correction")
    remove_stopwords: bool = Field(default=True, description="Remove stopwords")
    lemmatize: bool = Field(default=True, description="Apply lemmatization")


class PreprocessingResponse(BaseModel):
    """Response model for preprocessing results."""
    original_text: str
    cleaned_text: str
    processed_text: str
    language: str
    language_confidence: float
    tokens: List[str]
    sentences: List[str]
    word_count: int
    character_count: int
    processing_notes: Dict[str, Any]


class TextStatsResponse(BaseModel):
    """Response model for text statistics."""
    statistics: Dict[str, Any]
    key_phrases: List[Dict[str, Any]]


@router.post("/preprocess", response_model=PreprocessingResponse)
async def preprocess_text(
    text_input: TextInput,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Preprocess a single text with comprehensive cleaning and analysis.
    
    This endpoint provides:
    - Language detection (English/Hindi support)
    - Text cleaning and normalization
    - Tokenization and lemmatization
    - Stopword removal
    - Optional spell correction
    - Detailed processing statistics
    """
    try:
        result = await text_preprocessor.preprocess_text(
            text=text_input.text,
            enable_spell_correction=text_input.enable_spell_correction,
            remove_stopwords=text_input.remove_stopwords,
            lemmatize=text_input.lemmatize
        )
        
        return PreprocessingResponse(
            original_text=result.original_text,
            cleaned_text=result.cleaned_text,
            processed_text=result.processed_text,
            language=result.language,
            language_confidence=result.language_confidence,
            tokens=result.tokens,
            sentences=result.sentences,
            word_count=result.word_count,
            character_count=result.character_count,
            processing_notes=result.processing_notes
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error preprocessing text: {str(e)}"
        )


@router.post("/preprocess-batch", response_model=List[PreprocessingResponse])
async def preprocess_batch(
    batch_input: BatchTextInput,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Preprocess multiple texts in parallel for efficiency.
    
    Processes up to 100 texts simultaneously with the same preprocessing options.
    Useful for bulk analysis of comments or documents.
    """
    try:
        results = await text_preprocessor.batch_preprocess(
            texts=batch_input.texts,
            enable_spell_correction=batch_input.enable_spell_correction,
            remove_stopwords=batch_input.remove_stopwords,
            lemmatize=batch_input.lemmatize
        )
        
        return [
            PreprocessingResponse(
                original_text=result.original_text,
                cleaned_text=result.cleaned_text,
                processed_text=result.processed_text,
                language=result.language,
                language_confidence=result.language_confidence,
                tokens=result.tokens,
                sentences=result.sentences,
                word_count=result.word_count,
                character_count=result.character_count,
                processing_notes=result.processing_notes
            )
            for result in results
        ]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in batch preprocessing: {str(e)}"
        )


class TextAnalysisRequest(BaseModel):
    """Request model for text analysis."""
    text: str = Field(..., description="Text to analyze")


@router.post("/analyze", response_model=TextStatsResponse)
async def analyze_text(
    request: TextAnalysisRequest,
    current_user: User = Depends(get_current_active_user)
):
    """
    Get comprehensive text analysis and statistics.
    
    Provides detailed statistics about the text including:
    - Character and word counts
    - Language detection
    - Key phrase extraction
    - Readability metrics
    - Content analysis (URLs, emails, etc.)
    """
    try:
        # Get text statistics
        stats = text_preprocessor.get_text_statistics(request.text)
        
        # Extract key phrases
        key_phrases = text_preprocessor.extract_key_phrases(request.text)
        
        return TextStatsResponse(
            statistics=stats,
            key_phrases=key_phrases
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing text: {str(e)}"
        )


@router.get("/supported-languages")
async def get_supported_languages():
    """
    Get information about supported languages and preprocessing capabilities.
    """
    return {
        "supported_languages": [
            {
                "code": "en",
                "name": "English",
                "features": [
                    "Advanced tokenization",
                    "Lemmatization",
                    "Named entity recognition",
                    "Spell correction",
                    "Stopword removal",
                    "Key phrase extraction"
                ]
            },
            {
                "code": "hi", 
                "name": "Hindi",
                "features": [
                    "Basic tokenization",
                    "Stopword removal",
                    "Language detection"
                ]
            }
        ],
        "preprocessing_options": {
            "spell_correction": {
                "available_for": ["en"],
                "description": "Automatic spelling correction using TextBlob"
            },
            "lemmatization": {
                "available_for": ["en"],
                "description": "Reduce words to their base form using spaCy"
            },
            "stopword_removal": {
                "available_for": ["en", "hi"],
                "description": "Remove common words that don't carry meaning"
            },
            "language_detection": {
                "available_for": ["all"],
                "description": "Automatic detection of text language"
            }
        },
        "models_loaded": {
            "english_model": text_preprocessor.nlp_en is not None,
            "hindi_model": text_preprocessor.nlp_hi is not None
        }
    }


@router.get("/health")
async def preprocessing_health_check():
    """
    Health check for preprocessing service components.
    """
    status_info = {
        "service": "preprocessing",
        "status": "healthy",
        "components": {
            "spacy_english": text_preprocessor.nlp_en is not None,
            "nltk_data": len(text_preprocessor.stopwords_en) > 0,
            "text_cleaner": True
        },
        "supported_operations": [
            "text_cleaning",
            "tokenization", 
            "language_detection",
            "stopword_removal",
            "lemmatization",
            "spell_correction",
            "key_phrase_extraction"
        ]
    }
    
    # Overall health based on critical components
    critical_components = [
        status_info["components"]["spacy_english"],
        status_info["components"]["nltk_data"]
    ]
    
    if not all(critical_components):
        status_info["status"] = "degraded"
        status_info["warnings"] = []
        
        if not status_info["components"]["spacy_english"]:
            status_info["warnings"].append("English spaCy model not loaded")
        if not status_info["components"]["nltk_data"]:
            status_info["warnings"].append("NLTK stopwords not available")
    
    return status_info