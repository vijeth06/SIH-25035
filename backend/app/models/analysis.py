"""
Analysis result model for storing sentiment analysis and NLP processing results.
"""

from sqlalchemy import Column, Integer, String, Text, Float, Boolean, DateTime, ForeignKey, JSON, Enum as SQLAlchemyEnum
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from enum import Enum
from typing import Dict, Any, Optional, List
from datetime import datetime

# Note: SQLAlchemy Base is not used in current MongoDB setup.
# To avoid ImportError in tests, define a lightweight Base placeholder.
from typing import Any
class _BasePlaceholder:
    def __init__(self, *args, **kwargs):
        pass

Base = _BasePlaceholder


class SentimentLabel(str, Enum):
    """Sentiment classification labels."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class EmotionLabel(str, Enum):
    """Emotion classification labels."""
    SUPPORT = "support"
    CONCERN = "concern"
    SUGGESTION = "suggestion"
    ANGER = "anger"
    APPRECIATION = "appreciation"
    CONFUSION = "confusion"
    NEUTRAL = "neutral"


class AnalysisType(str, Enum):
    """Types of analysis performed."""
    SENTIMENT = "sentiment"
    EMOTION = "emotion"
    SUMMARIZATION = "summarization"
    ASPECT_SENTIMENT = "aspect_sentiment"
    KEYWORD_EXTRACTION = "keyword_extraction"
    TOPIC_MODELING = "topic_modeling"


class AnalysisResult(Base):
    """
    Analysis result model for storing NLP analysis outputs.
    
    Stores various types of analysis results including sentiment analysis,
    emotion classification, summarization, and other NLP outputs.
    """
    
    __tablename__ = "analysis_results"
    
    # Primary identification
    id = Column(Integer, primary_key=True, index=True)
    comment_id = Column(Integer, ForeignKey("comments.id"), nullable=False, index=True)
    
    # Analysis type and version
    analysis_type = Column(SQLAlchemyEnum(AnalysisType), nullable=False)
    model_name = Column(String(100), nullable=True)  # Name of the model used
    model_version = Column(String(50), nullable=True)  # Version of the model
    
    # Sentiment analysis results
    sentiment_label = Column(SQLAlchemyEnum(SentimentLabel), nullable=True)
    sentiment_score = Column(Float, nullable=True)  # Confidence score (-1 to 1)
    sentiment_confidence = Column(Float, nullable=True)  # Confidence level (0 to 1)
    
    # Emotion analysis results
    emotion_label = Column(SQLAlchemyEnum(EmotionLabel), nullable=True)
    emotion_score = Column(Float, nullable=True)  # Confidence score (0 to 1)
    emotion_scores = Column(JSON, nullable=True)  # Scores for all emotions
    
    # Text analysis results
    keywords = Column(JSON, nullable=True)  # Extracted keywords with scores
    topics = Column(JSON, nullable=True)  # Topic modeling results
    summary = Column(Text, nullable=True)  # Generated summary
    
    # Aspect-based sentiment analysis
    aspects = Column(JSON, nullable=True)  # Aspect-sentiment pairs
    law_sections_mentioned = Column(JSON, nullable=True)  # Referenced law sections
    
    # Quality and reliability metrics
    confidence_score = Column(Float, nullable=True)  # Overall confidence (0 to 1)
    quality_flags = Column(JSON, nullable=True)  # Quality assessment flags
    
    # Processing metadata
    processing_time_ms = Column(Integer, nullable=True)  # Processing time in milliseconds
    error_message = Column(Text, nullable=True)  # Error message if analysis failed
    is_successful = Column(Boolean, default=True, nullable=False)
    
    # Explainability and rationale
    explanation = Column(JSON, nullable=True)  # Explanation of the analysis result
    key_phrases = Column(JSON, nullable=True)  # Key phrases that influenced the result
    
    # Human feedback and corrections
    human_verified = Column(Boolean, default=False, nullable=False)
    human_sentiment = Column(SQLAlchemyEnum(SentimentLabel), nullable=True)
    human_emotion = Column(SQLAlchemyEnum(EmotionLabel), nullable=True)
    human_feedback = Column(Text, nullable=True)
    verified_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    verified_at = Column(DateTime(timezone=True), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    comment = relationship("Comment", back_populates="analysis_results")
    verifier = relationship("User", foreign_keys=[verified_by], backref="verified_analyses")
    
    def __repr__(self):
        return f"<AnalysisResult(id={self.id}, type='{self.analysis_type}', sentiment='{self.sentiment_label}')>"
    
    def to_dict(self, include_details: bool = True) -> Dict[str, Any]:
        """
        Convert analysis result to dictionary.
        
        Args:
            include_details: Whether to include detailed analysis data
            
        Returns:
            dict: Analysis result data
        """
        result = {
            "id": self.id,
            "comment_id": self.comment_id,
            "analysis_type": self.analysis_type.value if self.analysis_type else None,
            "sentiment_label": self.sentiment_label.value if self.sentiment_label else None,
            "sentiment_score": self.sentiment_score,
            "sentiment_confidence": self.sentiment_confidence,
            "emotion_label": self.emotion_label.value if self.emotion_label else None,
            "emotion_score": self.emotion_score,
            "confidence_score": self.confidence_score,
            "is_successful": self.is_successful,
            "human_verified": self.human_verified,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
        
        if include_details:
            result.update({
                "model_name": self.model_name,
                "model_version": self.model_version,
                "emotion_scores": self.emotion_scores,
                "keywords": self.keywords,
                "topics": self.topics,
                "summary": self.summary,
                "aspects": self.aspects,
                "law_sections_mentioned": self.law_sections_mentioned,
                "quality_flags": self.quality_flags,
                "processing_time_ms": self.processing_time_ms,
                "explanation": self.explanation,
                "key_phrases": self.key_phrases,
                "human_sentiment": self.human_sentiment.value if self.human_sentiment else None,
                "human_emotion": self.human_emotion.value if self.human_emotion else None,
                "human_feedback": self.human_feedback,
                "verified_at": self.verified_at.isoformat() if self.verified_at else None,
                "error_message": self.error_message,
            })
        
        return result
    
    def get_best_sentiment(self) -> Optional[SentimentLabel]:
        """Get the best sentiment (human verified or model prediction)."""
        return self.human_sentiment if self.human_verified else self.sentiment_label
    
    def get_best_emotion(self) -> Optional[EmotionLabel]:
        """Get the best emotion (human verified or model prediction)."""
        return self.human_emotion if self.human_verified else self.emotion_label
    
    def add_quality_flag(self, flag: str, details: Any = None):
        """Add a quality assessment flag."""
        if self.quality_flags is None:
            self.quality_flags = {}
        self.quality_flags[flag] = details
    
    def has_quality_flag(self, flag: str) -> bool:
        """Check if analysis has a specific quality flag."""
        return self.quality_flags is not None and flag in self.quality_flags
    
    def get_keyword_score(self, keyword: str) -> Optional[float]:
        """Get the score for a specific keyword."""
        if self.keywords and isinstance(self.keywords, dict):
            return self.keywords.get(keyword)
        elif self.keywords and isinstance(self.keywords, list):
            for item in self.keywords:
                if isinstance(item, dict) and item.get("keyword") == keyword:
                    return item.get("score")
        return None
    
    def get_top_keywords(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get top keywords by score.
        
        Args:
            limit: Maximum number of keywords to return
            
        Returns:
            list: List of keyword dictionaries with 'keyword' and 'score' keys
        """
        if not self.keywords:
            return []
        
        if isinstance(self.keywords, dict):
            # Convert dict to list of dicts
            keywords = [{"keyword": k, "score": v} for k, v in self.keywords.items()]
        else:
            keywords = self.keywords
        
        # Sort by score and return top N
        sorted_keywords = sorted(keywords, key=lambda x: x.get("score", 0), reverse=True)
        return sorted_keywords[:limit]
    
    def is_positive_sentiment(self) -> bool:
        """Check if sentiment is positive."""
        sentiment = self.get_best_sentiment()
        return sentiment == SentimentLabel.POSITIVE
    
    def is_negative_sentiment(self) -> bool:
        """Check if sentiment is negative."""
        sentiment = self.get_best_sentiment()
        return sentiment == SentimentLabel.NEGATIVE
    
    def is_neutral_sentiment(self) -> bool:
        """Check if sentiment is neutral."""
        sentiment = self.get_best_sentiment()
        return sentiment == SentimentLabel.NEUTRAL
    
    def mark_human_verified(self, user_id: int, sentiment: Optional[SentimentLabel] = None, 
                           emotion: Optional[EmotionLabel] = None, feedback: Optional[str] = None):
        """Mark analysis as human verified with corrections."""
        self.human_verified = True
        self.verified_by = user_id
        self.verified_at = datetime.utcnow()
        
        if sentiment is not None:
            self.human_sentiment = sentiment
        if emotion is not None:
            self.human_emotion = emotion
        if feedback is not None:
            self.human_feedback = feedback