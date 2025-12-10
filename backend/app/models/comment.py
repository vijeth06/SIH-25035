"""
Comment model for storing stakeholder comments from e-Consultation.
"""

from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, Float, ForeignKey, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Optional, Dict, Any

from backend.app.core.database import Base


class Comment(Base):
    """
    Comment model for storing stakeholder comments.
    
    Represents individual comments from e-Consultation processes,
    including original text, metadata, and processing status.
    """
    
    __tablename__ = "comments"
    
    # Primary identification
    id = Column(Integer, primary_key=True, index=True)
    original_text = Column(Text, nullable=False)
    processed_text = Column(Text, nullable=True)  # Cleaned and preprocessed text
    
    # Metadata
    source_file = Column(String(255), nullable=True)  # Original file name
    source_row = Column(Integer, nullable=True)  # Row number in source file
    comment_id_external = Column(String(100), nullable=True, index=True)  # External comment ID
    
    # Content characteristics
    language = Column(String(10), nullable=True)  # Language code (e.g., 'en', 'hi')
    word_count = Column(Integer, nullable=True)
    character_count = Column(Integer, nullable=True)
    
    # Processing status
    is_processed = Column(Boolean, default=False, nullable=False)
    is_duplicate = Column(Boolean, default=False, nullable=False)
    duplicate_of_id = Column(Integer, ForeignKey("comments.id"), nullable=True)
    
    # Consultation context
    law_section = Column(String(100), nullable=True)  # Referenced law section
    consultation_id = Column(String(100), nullable=True, index=True)  # Consultation process ID
    category = Column(String(50), nullable=True)  # Comment category
    
    # Stakeholder information (anonymized)
    stakeholder_type = Column(String(50), nullable=True)  # Individual, Organization, etc.
    stakeholder_category = Column(String(50), nullable=True)  # Citizen, Expert, NGO, etc.
    location = Column(String(100), nullable=True)  # General location (city/state level only)
    
    # Quality and validation
    quality_score = Column(Float, nullable=True)  # Content quality score (0-1)
    contains_pii = Column(Boolean, default=False, nullable=False)  # Contains Personal Information
    is_valid = Column(Boolean, default=True, nullable=False)  # Valid for analysis
    
    # Processing metadata
    processing_notes = Column(JSON, nullable=True)  # Additional processing information
    tags = Column(JSON, nullable=True)  # Flexible tagging system
    
    # Timestamps
    submitted_at = Column(DateTime(timezone=True), nullable=True)  # Original submission time
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    processed_at = Column(DateTime(timezone=True), nullable=True)
    
    # User who uploaded/processed the comment
    uploaded_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    # Relationships
    duplicate_of = relationship("Comment", remote_side=[id], backref="duplicates")
    uploader = relationship("User", backref="uploaded_comments")
    analysis_results = relationship("AnalysisResult", back_populates="comment", cascade="all, delete-orphan")
    
    def __repr__(self):
        preview = self.original_text[:50] + "..." if len(self.original_text) > 50 else self.original_text
        return f"<Comment(id={self.id}, language='{self.language}', preview='{preview}')>"
    
    def to_dict(self, include_text: bool = True, include_analysis: bool = False) -> Dict[str, Any]:
        """
        Convert comment to dictionary.
        
        Args:
            include_text: Whether to include the full comment text
            include_analysis: Whether to include analysis results
            
        Returns:
            dict: Comment data
        """
        result = {
            "id": self.id,
            "language": self.language,
            "word_count": self.word_count,
            "character_count": self.character_count,
            "is_processed": self.is_processed,
            "is_duplicate": self.is_duplicate,
            "duplicate_of_id": self.duplicate_of_id,
            "law_section": self.law_section,
            "consultation_id": self.consultation_id,
            "category": self.category,
            "stakeholder_type": self.stakeholder_type,
            "stakeholder_category": self.stakeholder_category,
            "location": self.location,
            "quality_score": self.quality_score,
            "contains_pii": self.contains_pii,
            "is_valid": self.is_valid,
            "tags": self.tags,
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
        }
        
        if include_text:
            result.update({
                "original_text": self.original_text,
                "processed_text": self.processed_text,
            })
        else:
            # Just include preview
            result["text_preview"] = self.get_text_preview()
        
        if include_analysis and self.analysis_results:
            result["analysis"] = [analysis.to_dict() for analysis in self.analysis_results]
        
        return result
    
    def get_text_preview(self, max_length: int = 100) -> str:
        """
        Get a preview of the comment text.
        
        Args:
            max_length: Maximum length of preview
            
        Returns:
            str: Text preview
        """
        if not self.original_text:
            return ""
        
        if len(self.original_text) <= max_length:
            return self.original_text
        
        return self.original_text[:max_length] + "..."
    
    def mark_as_duplicate(self, original_comment_id: int):
        """Mark this comment as a duplicate of another."""
        self.is_duplicate = True
        self.duplicate_of_id = original_comment_id
    
    def add_tag(self, tag: str):
        """Add a tag to the comment."""
        if self.tags is None:
            self.tags = []
        if tag not in self.tags:
            self.tags.append(tag)
    
    def remove_tag(self, tag: str):
        """Remove a tag from the comment."""
        if self.tags and tag in self.tags:
            self.tags.remove(tag)
    
    def has_tag(self, tag: str) -> bool:
        """Check if comment has a specific tag."""
        return self.tags is not None and tag in self.tags
    
    def get_processing_note(self, key: str) -> Optional[Any]:
        """Get a specific processing note."""
        if self.processing_notes:
            return self.processing_notes.get(key)
        return None
    
    def set_processing_note(self, key: str, value: Any):
        """Set a processing note."""
        if self.processing_notes is None:
            self.processing_notes = {}
        self.processing_notes[key] = value
    
    def is_english(self) -> bool:
        """Check if comment is in English."""
        return self.language == "en"
    
    def is_hindi(self) -> bool:
        """Check if comment is in Hindi."""
        return self.language == "hi"
    
    def is_supported_language(self) -> bool:
        """Check if comment language is supported for processing."""
        return self.language in ["en", "hi"] if self.language else False