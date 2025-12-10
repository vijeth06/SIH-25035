"""
Simplified summarization service with robust fallback handling.
"""

import re
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import logging
from collections import Counter
import math

logger = logging.getLogger(__name__)


class SummarizationType(str, Enum):
    """Types of summarization available."""
    EXTRACTIVE = "extractive"
    ABSTRACTIVE = "abstractive"
    HYBRID = "hybrid"


class SummarizationMethod(str, Enum):
    """Available summarization methods."""
    TEXTRANK = "textrank"
    LSA = "lsa"
    LUHN = "luhn"
    EDMUNDSON = "edmundson"
    T5 = "t5"
    CUSTOM_TEXTRANK = "custom_textrank"


@dataclass
class SummaryResult:
    """Result of summarization process."""
    method: str
    summary_type: SummarizationType
    summary_text: str
    original_length: int
    summary_length: int
    compression_ratio: float
    key_sentences: List[str]
    confidence_score: float
    processing_time_ms: int
    metadata: Dict[str, Any]


class SimplifiedSummarizationService:
    """Simplified summarization service with robust fallback handling."""

    def __init__(self):
        # Common stop words
        self.stop_words = {
            'a', 'an', 'the', 'and', 'or', 'but', 'if', 'while', 'at', 'by', 'for', 'with',
            'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
            'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
            'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
            'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'i', 'me',
            'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
            'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is',
            'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
            'do', 'does', 'did', 'doing', 'would', 'could', 'should', 'may', 'might',
            'must', 'shall', 'will', 'can'
        }

        logger.info("✅ Simplified summarization service initialized")

    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text and return sentences."""
        if not text:
            return []

        # Split into sentences (simple approach)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def _calculate_sentence_score(self, sentence: str, word_freq: Dict[str, int], total_sentences: int) -> float:
        """Calculate score for a sentence."""
        words = re.findall(r'\b\w+\b', sentence.lower())
        words = [word for word in words if word not in self.stop_words and len(word) > 2]

        if not words:
            return 0.0

        # Word frequency score
        word_score = sum(word_freq.get(word, 0) for word in words) / len(words)

        # Position score (prefer sentences at beginning and end)
        position_score = 1.0

        # Length score (prefer medium-length sentences)
        length_score = min(1.0, len(words) / 20.0)  # Optimal around 20 words

        return (word_score * 0.6) + (position_score * 0.2) + (length_score * 0.2)

    def _extract_sentences_textrank(self, sentences: List[str], num_sentences: int) -> List[str]:
        """Extract sentences using simplified TextRank approach."""
        if len(sentences) <= num_sentences:
            return sentences

        # Calculate word frequencies
        all_words = []
        for sentence in sentences:
            words = re.findall(r'\b\w+\b', sentence.lower())
            words = [word for word in words if word not in self.stop_words and len(word) > 2]
            all_words.extend(words)

        word_freq = Counter(all_words)

        # Calculate sentence scores
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            score = self._calculate_sentence_score(sentence, word_freq, len(sentences))
            sentence_scores.append((i, sentence, score))

        # Sort by score and select top sentences
        sentence_scores.sort(key=lambda x: x[2], reverse=True)
        selected_indices = sorted([idx for idx, _, _ in sentence_scores[:num_sentences]])

        return [sentences[idx] for idx in selected_indices]

    def _create_abstractive_summary(self, sentences: List[str], max_length: int, min_length: int) -> str:
        """Create abstractive summary by combining and rephrasing sentences."""
        if not sentences:
            return ""

        # Simple approach: combine top sentences and truncate
        selected_sentences = self._extract_sentences_textrank(sentences, 3)
        combined_text = ' '.join(selected_sentences)

        # Truncate to desired length
        words = combined_text.split()
        if len(words) > max_length:
            words = words[:max_length]
        elif len(words) < min_length and len(sentences) > 1:
            # Add more content if too short
            additional_sentences = self._extract_sentences_textrank(sentences, 5)[3:]
            additional_words = ' '.join(additional_sentences).split()
            words.extend(additional_words[:max_length - len(words)])

        return ' '.join(words)

    async def extractive_summarization(self, text: str,
                                     method: SummarizationMethod = SummarizationMethod.CUSTOM_TEXTRANK,
                                     num_sentences: int = 3) -> SummaryResult:
        """Perform extractive summarization."""
        start_time = time.time()

        if not text or len(text.strip()) < 50:
            return SummaryResult(
                method=method.value,
                summary_type=SummarizationType.EXTRACTIVE,
                summary_text="",
                original_length=len(text),
                summary_length=0,
                compression_ratio=0.0,
                key_sentences=[],
                confidence_score=0.0,
                processing_time_ms=int((time.time() - start_time) * 1000),
                metadata={"error": "Text too short for summarization"}
            )

        sentences = self._preprocess_text(text)
        if not sentences:
            return SummaryResult(
                method=method.value,
                summary_type=SummarizationType.EXTRACTIVE,
                summary_text="",
                original_length=len(text),
                summary_length=0,
                compression_ratio=0.0,
                key_sentences=[],
                confidence_score=0.0,
                processing_time_ms=int((time.time() - start_time) * 1000),
                metadata={"error": "No sentences found"}
            )

        # Extract key sentences
        key_sentences = self._extract_sentences_textrank(sentences, num_sentences)
        summary_text = ' '.join(key_sentences)

        original_length = len(text)
        summary_length = len(summary_text)
        compression_ratio = summary_length / original_length if original_length > 0 else 0

        processing_time = int((time.time() - start_time) * 1000)
        confidence = min(0.9, len(key_sentences) / len(sentences)) if sentences else 0.5

        return SummaryResult(
            method=method.value,
            summary_type=SummarizationType.EXTRACTIVE,
            summary_text=summary_text,
            original_length=original_length,
            summary_length=summary_length,
            compression_ratio=compression_ratio,
            key_sentences=key_sentences,
            confidence_score=confidence,
            processing_time_ms=processing_time,
            metadata={
                "num_sentences_requested": num_sentences,
                "num_sentences_returned": len(key_sentences),
                "original_sentence_count": len(sentences)
            }
        )

    async def abstractive_summarization(self, text: str,
                                      method: SummarizationMethod = SummarizationMethod.T5,
                                      max_length: int = 150,
                                      min_length: int = 30) -> SummaryResult:
        """Perform abstractive summarization."""
        start_time = time.time()

        if not text or len(text.strip()) < 50:
            return SummaryResult(
                method="fallback_abstractive",
                summary_type=SummarizationType.ABSTRACTIVE,
                summary_text="",
                original_length=len(text),
                summary_length=0,
                compression_ratio=0.0,
                key_sentences=[],
                confidence_score=0.0,
                processing_time_ms=int((time.time() - start_time) * 1000),
                metadata={"error": "Text too short for summarization"}
            )

        sentences = self._preprocess_text(text)
        summary_text = self._create_abstractive_summary(sentences, max_length, min_length)

        original_length = len(text)
        summary_length = len(summary_text)
        compression_ratio = summary_length / original_length if original_length > 0 else 0

        processing_time = int((time.time() - start_time) * 1000)

        return SummaryResult(
            method="fallback_abstractive",
            summary_type=SummarizationType.ABSTRACTIVE,
            summary_text=summary_text,
            original_length=original_length,
            summary_length=summary_length,
            compression_ratio=compression_ratio,
            key_sentences=[summary_text],  # Single abstractive summary
            confidence_score=0.7,  # Default confidence for abstractive
            processing_time_ms=processing_time,
            metadata={
                "max_length": max_length,
                "min_length": min_length,
                "original_sentence_count": len(sentences)
            }
        )

    async def hybrid_summarization(self, text: str,
                                 extractive_sentences: int = 3,
                                 abstractive_max_length: int = 150) -> SummaryResult:
        """Perform hybrid summarization."""
        start_time = time.time()

        if not text or len(text.strip()) < 50:
            return SummaryResult(
                method="hybrid_fallback",
                summary_type=SummarizationType.HYBRID,
                summary_text="",
                original_length=len(text),
                summary_length=0,
                compression_ratio=0.0,
                key_sentences=[],
                confidence_score=0.0,
                processing_time_ms=int((time.time() - start_time) * 1000),
                metadata={"error": "Text too short for summarization"}
            )

        # First do extractive summarization
        extractive_result = await self.extractive_summarization(text, num_sentences=extractive_sentences)

        # Then do abstractive on the extractive summary
        abstractive_result = await self.abstractive_summarization(
            extractive_result.summary_text,
            max_length=abstractive_max_length,
            min_length=abstractive_max_length // 3
        )

        processing_time = int((time.time() - start_time) * 1000)

        return SummaryResult(
            method="hybrid_fallback",
            summary_type=SummarizationType.HYBRID,
            summary_text=abstractive_result.summary_text,
            original_length=len(text),
            summary_length=len(abstractive_result.summary_text),
            compression_ratio=len(abstractive_result.summary_text) / len(text) if text else 0,
            key_sentences=extractive_result.key_sentences,
            confidence_score=min(extractive_result.confidence_score, abstractive_result.confidence_score),
            processing_time_ms=processing_time,
            metadata={
                "extractive_sentences": extractive_sentences,
                "abstractive_max_length": abstractive_max_length,
                "hybrid_approach": "extractive_first"
            }
        )

    async def topic_based_summarization(self, text: str,
                                      topics: List[str] = None,
                                      max_length: int = 150,
                                      min_length: int = 30) -> SummaryResult:
        """Perform topic-based summarization."""
        start_time = time.time()

        if not text or len(text.strip()) < 50:
            return SummaryResult(
                method="topic_based_fallback",
                summary_type=SummarizationType.ABSTRACTIVE,
                summary_text="",
                original_length=len(text),
                summary_length=0,
                compression_ratio=0.0,
                key_sentences=[],
                confidence_score=0.0,
                processing_time_ms=int((time.time() - start_time) * 1000),
                metadata={"error": "Text too short for summarization"}
            )

        sentences = self._preprocess_text(text)
        if not sentences:
            return SummaryResult(
                method="topic_based_fallback",
                summary_type=SummarizationType.ABSTRACTIVE,
                summary_text="",
                original_length=len(text),
                summary_length=0,
                compression_ratio=0.0,
                key_sentences=[],
                confidence_score=0.0,
                processing_time_ms=int((time.time() - start_time) * 1000),
                metadata={"error": "No sentences found"}
            )

        # If topics are provided, prioritize sentences containing those topics
        if topics:
            topic_sentences = []
            for sentence in sentences:
                sentence_lower = sentence.lower()
                for topic in topics:
                    if topic.lower() in sentence_lower:
                        topic_sentences.append(sentence)
                        break

            # If we found topic-related sentences, use them
            if len(topic_sentences) >= 2:
                sentences = topic_sentences

        # Create summary
        summary_text = self._create_abstractive_summary(sentences, max_length, min_length)

        original_length = len(text)
        summary_length = len(summary_text)
        compression_ratio = summary_length / original_length if original_length > 0 else 0

        processing_time = int((time.time() - start_time) * 1000)

        return SummaryResult(
            method="topic_based_fallback",
            summary_type=SummarizationType.ABSTRACTIVE,
            summary_text=summary_text,
            original_length=original_length,
            summary_length=summary_length,
            compression_ratio=compression_ratio,
            key_sentences=[summary_text],
            confidence_score=0.6,  # Slightly lower confidence for topic-based
            processing_time_ms=processing_time,
            metadata={
                "topics_focused": topics or [],
                "max_length": max_length,
                "min_length": min_length,
                "topic_sentences_found": len(sentences)
            }
        )