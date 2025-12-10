"""
Text processing utilities for validation, cleaning, and deduplication.
"""

import re
from typing import List, Tuple, Optional, Set
from difflib import SequenceMatcher
import hashlib
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of text validation."""
    is_valid: bool
    issues: List[str]
    quality_score: float


class TextValidator:
    """Validator for comment text quality and content."""
    
    def __init__(self):
        # Common spam patterns
        self.spam_patterns = [
            r'^(.)\1{10,}',  # Repeated characters
            r'^[^a-zA-Z0-9\s]{5,}$',  # Only special characters
            r'^\s*$',  # Empty or whitespace only
            r'^(.{1,3})\1{5,}',  # Short repeated patterns
        ]
        
        # Minimum requirements
        self.min_length = 10
        self.max_length = 10000
        self.min_words = 2
        
        # PII patterns (basic detection)
        self.pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{10}\b',  # Phone number
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{1,5}\s\w+\s(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl)\b',  # Address
        ]
    
    def is_valid_comment(self, text: str) -> bool:
        """Quick validation check."""
        result = self.validate_text(text)
        return result.is_valid
    
    def validate_text(self, text: str) -> ValidationResult:
        """
        Comprehensive text validation.
        
        Args:
            text: Text to validate
            
        Returns:
            ValidationResult: Validation result with issues and quality score
        """
        issues = []
        
        # Basic checks
        if not text or not text.strip():
            return ValidationResult(False, ["Empty text"], 0.0)
        
        text = text.strip()
        
        # Length checks
        if len(text) < self.min_length:
            issues.append(f"Text too short (minimum {self.min_length} characters)")
        
        if len(text) > self.max_length:
            issues.append(f"Text too long (maximum {self.max_length} characters)")
        
        # Word count check
        words = text.split()
        if len(words) < self.min_words:
            issues.append(f"Too few words (minimum {self.min_words} words)")
        
        # Spam detection
        for pattern in self.spam_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append("Text appears to be spam or low quality")
                break
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(text)
        
        # Consider valid if no critical issues and decent quality
        is_valid = len(issues) == 0 and quality_score >= 0.3
        
        return ValidationResult(is_valid, issues, quality_score)
    
    def contains_pii(self, text: str) -> Tuple[bool, List[str]]:
        """
        Check if text contains personally identifiable information.
        
        Args:
            text: Text to check
            
        Returns:
            tuple: (has_pii, list_of_pii_types_found)
        """
        pii_found = []
        
        pii_types = {
            r'\b\d{3}-\d{2}-\d{4}\b': 'SSN',
            r'\b\d{10}\b': 'Phone',
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b': 'Email',
            r'\b\d{1,5}\s\w+\s(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl)\b': 'Address',
        }
        
        for pattern, pii_type in pii_types.items():
            if re.search(pattern, text, re.IGNORECASE):
                pii_found.append(pii_type)
        
        return len(pii_found) > 0, pii_found
    
    def _calculate_quality_score(self, text: str) -> float:
        """Calculate a quality score for the text (0-1)."""
        score = 1.0
        
        # Penalize very short text
        if len(text) < 50:
            score *= 0.7
        
        # Penalize excessive uppercase
        uppercase_ratio = sum(1 for c in text if c.isupper()) / len(text)
        if uppercase_ratio > 0.7:
            score *= 0.8
        
        # Penalize excessive punctuation
        punct_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
        if punct_ratio > 0.3:
            score *= 0.8
        
        # Reward proper sentence structure
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) > 1:
            score *= 1.1
        
        # Penalize repetitive words
        words = text.lower().split()
        unique_words = set(words)
        if len(words) > 0:
            word_diversity = len(unique_words) / len(words)
            if word_diversity < 0.5:
                score *= 0.9
        
        return min(score, 1.0)


class DuplicationDetector:
    """Detector for duplicate and near-duplicate comments."""
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.exact_match_cache = {}
        
    def find_duplicates(self, comments: List) -> List[Tuple[int, int]]:
        """
        Find duplicate comments in a list.
        
        Args:
            comments: List of Comment objects or text strings
            
        Returns:
            list: List of (original_index, duplicate_index) tuples
        """
        duplicates = []
        
        # First pass: exact matches using hash
        hash_to_index = {}
        
        for i, comment in enumerate(comments):
            text = comment.original_text if hasattr(comment, 'original_text') else str(comment)
            text_hash = self._hash_text(text)
            
            if text_hash in hash_to_index:
                duplicates.append((hash_to_index[text_hash], i))
            else:
                hash_to_index[text_hash] = i
        
        # Second pass: near-duplicates using similarity
        processed = set()
        
        for i, comment1 in enumerate(comments):
            if i in processed:
                continue
                
            text1 = comment1.original_text if hasattr(comment1, 'original_text') else str(comment1)
            
            for j, comment2 in enumerate(comments[i+1:], i+1):
                if j in processed:
                    continue
                    
                text2 = comment2.original_text if hasattr(comment2, 'original_text') else str(comment2)
                
                if self._are_similar(text1, text2):
                    duplicates.append((i, j))
                    processed.add(j)
        
        return duplicates
    
    def is_duplicate(self, text1: str, text2: str) -> bool:
        """Check if two texts are duplicates."""
        # Exact match
        if self._normalize_text(text1) == self._normalize_text(text2):
            return True
        
        # Near-duplicate match
        return self._are_similar(text1, text2)
    
    def _hash_text(self, text: str) -> str:
        """Create hash of normalized text for exact duplicate detection."""
        normalized = self._normalize_text(text)
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove punctuation for comparison
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def _are_similar(self, text1: str, text2: str) -> bool:
        """Check if two texts are similar using sequence matching."""
        # Normalize texts
        norm1 = self._normalize_text(text1)
        norm2 = self._normalize_text(text2)
        
        # Skip if texts are too different in length
        len_ratio = min(len(norm1), len(norm2)) / max(len(norm1), len(norm2))
        if len_ratio < 0.5:
            return False
        
        # Calculate similarity
        similarity = SequenceMatcher(None, norm1, norm2).ratio()
        return similarity >= self.similarity_threshold


class TextCleaner:
    """Utility for cleaning and preprocessing text."""
    
    def __init__(self):
        # HTML tags pattern
        self.html_pattern = re.compile(r'<[^>]+>')
        
        # URL pattern
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        
        # Email pattern
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        
        # Multiple spaces pattern
        self.spaces_pattern = re.compile(r'\s+')
        
    def clean_text(self, text: str, remove_html: bool = True, 
                   remove_urls: bool = True, remove_emails: bool = False) -> str:
        """
        Clean text by removing unwanted elements.
        
        Args:
            text: Text to clean
            remove_html: Whether to remove HTML tags
            remove_urls: Whether to remove URLs
            remove_emails: Whether to remove email addresses
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Remove HTML tags
        if remove_html:
            text = self.html_pattern.sub(' ', text)
        
        # Remove URLs
        if remove_urls:
            text = self.url_pattern.sub(' [URL] ', text)
        
        # Remove or mask emails
        if remove_emails:
            text = self.email_pattern.sub(' [EMAIL] ', text)
        
        # Normalize whitespace
        text = self.spaces_pattern.sub(' ', text)
        
        # Strip and return
        return text.strip()
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def anonymize_pii(self, text: str) -> str:
        """Anonymize personally identifiable information."""
        # Replace emails
        text = re.sub(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            '[EMAIL]',
            text,
            flags=re.IGNORECASE
        )
        
        # Replace phone numbers
        text = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]', text)
        text = re.sub(r'\b\d{10}\b', '[PHONE]', text)
        
        # Replace potential SSNs
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
        
        # Replace addresses (basic pattern)
        text = re.sub(
            r'\b\d{1,5}\s\w+\s(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl)\b',
            '[ADDRESS]',
            text,
            flags=re.IGNORECASE
        )
        
        return text