"""
Simplified sentiment analysis service with robust fallback handling.
"""

import re
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import logging

logger = logging.getLogger(__name__)


class SentimentLabel(str, Enum):
    """Sentiment labels."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class SentimentResult:
    """Result of sentiment analysis."""
    text: str
    sentiment_label: SentimentLabel
    confidence_score: float
    positive_score: float
    negative_score: float
    neutral_score: float
    compound_score: Optional[float] = None
    processing_time_ms: int = 0


@dataclass
class EmotionResult:
    """Result of emotion analysis."""
    emotion_label: str
    emotion_scores: Dict[str, float]
    detected_emotions: List[str]
    confidence_score: float


@dataclass
class ComprehensiveAnalysisResult:
    """Comprehensive analysis result."""
    overall_sentiment: SentimentLabel
    overall_confidence: float
    sentiment_results: List[SentimentResult]
    emotion_result: Optional[EmotionResult] = None
    key_phrases: List[str] = None
    law_sections_mentioned: List[str] = None
    aspect_sentiments: List[Dict[str, Any]] = None
    processing_time_ms: int = 0


class SimplifiedSentimentAnalyzer:
    """Simplified sentiment analyzer with robust fallback handling."""

    def __init__(self):
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'awesome',
            'love', 'like', 'best', 'perfect', 'brilliant', 'outstanding', 'superb',
            'support', 'agree', 'yes', 'positive', 'benefit', 'advantage', 'improve',
            'better', 'happy', 'pleased', 'satisfied', 'encouraging', 'hopeful'
        }

        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'dislike',
            'poor', 'weak', 'fail', 'failure', 'problem', 'issue', 'concern',
            'disagree', 'oppose', 'against', 'no', 'negative', 'worse', 'difficult',
            'sad', 'angry', 'frustrated', 'worried', 'disappointed', 'unhappy'
        }

        self.neutral_words = {
            'okay', 'fine', 'average', 'normal', 'standard', 'regular', 'typical',
            'moderate', 'reasonable', 'fair', 'decent', 'acceptable', 'satisfactory'
        }

        # Emotion keywords
        self.emotion_keywords = {
            'joy': ['happy', 'joy', 'pleased', 'delighted', 'excited', 'cheerful'],
            'sadness': ['sad', 'unhappy', 'depressed', 'sorrow', 'grief', 'disappointed'],
            'anger': ['angry', 'furious', 'irritated', 'annoyed', 'frustrated', 'mad'],
            'fear': ['afraid', 'scared', 'frightened', 'worried', 'anxious', 'nervous'],
            'surprise': ['surprised', 'amazed', 'shocked', 'astonished', 'unexpected'],
            'disgust': ['disgusted', 'repulsed', 'revolted', 'nauseated', 'sick'],
            'trust': ['trust', 'reliable', 'dependable', 'faithful', 'loyal'],
            'anticipation': ['hopeful', 'expectant', 'eager', 'optimistic', 'looking forward']
        }

        logger.info("✅ Simplified sentiment analyzer initialized")

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis."""
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove special characters but keep spaces and basic punctuation
        text = re.sub(r'[^\w\s.,!?]', ' ', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def _calculate_sentiment_scores(self, text: str) -> Tuple[float, float, float]:
        """Calculate sentiment scores using rule-based approach."""
        words = text.split()
        if not words:
            return 0.33, 0.33, 0.33

        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        neutral_count = sum(1 for word in words if word in self.neutral_words)

        total_sentiment_words = positive_count + negative_count + neutral_count

        if total_sentiment_words == 0:
            # If no sentiment words found, use a balanced approach
            return 0.33, 0.33, 0.34

        total_words = len(words)
        neutral_boost = max(0, (total_words - total_sentiment_words) / total_words)

        positive_score = positive_count / total_sentiment_words
        negative_score = negative_count / total_sentiment_words
        neutral_score = neutral_count / total_sentiment_words

        # Boost neutral if many non-sentiment words
        neutral_score = min(1.0, neutral_score + neutral_boost * 0.3)

        # Normalize
        total = positive_score + negative_score + neutral_score
        if total > 0:
            positive_score /= total
            negative_score /= total
            neutral_score /= total

        return positive_score, negative_score, neutral_score

    def _determine_overall_sentiment(self, positive: float, negative: float, neutral: float) -> SentimentLabel:
        """Determine overall sentiment from scores."""
        max_score = max(positive, negative, neutral)

        if max_score == positive:
            return SentimentLabel.POSITIVE
        elif max_score == negative:
            return SentimentLabel.NEGATIVE
        else:
            return SentimentLabel.NEUTRAL

    def _analyze_emotions(self, text: str) -> EmotionResult:
        """Analyze emotions in text."""
        words = text.split()
        emotion_scores = {}

        for emotion, keywords in self.emotion_keywords.items():
            count = sum(1 for word in words if word in keywords)
            emotion_scores[emotion] = count / len(words) if words else 0

        # Find top emotions
        detected_emotions = [emotion for emotion, score in emotion_scores.items() if score > 0.01]
        detected_emotions.sort(key=lambda x: emotion_scores[x], reverse=True)

        # Determine primary emotion
        if detected_emotions:
            primary_emotion = detected_emotions[0]
            confidence = emotion_scores[primary_emotion]
        else:
            primary_emotion = "neutral"
            confidence = 0.5

        return EmotionResult(
            emotion_label=primary_emotion,
            emotion_scores=emotion_scores,
            detected_emotions=detected_emotions[:3],  # Top 3 emotions
            confidence_score=confidence
        )

    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text."""
        # Simple approach: extract noun phrases or important words
        words = text.split()
        key_phrases = []

        # Look for sequences of important words
        for i, word in enumerate(words):
            if len(word) > 4:  # Longer words are likely more important
                if i > 0 and len(words[i-1]) > 2:
                    phrase = f"{words[i-1]} {word}"
                    if len(phrase) > 6:
                        key_phrases.append(phrase)
                else:
                    key_phrases.append(word)

        return key_phrases[:5]  # Return top 5 key phrases

    def _detect_law_sections(self, text: str) -> List[str]:
        """Detect law sections mentioned in text."""
        law_patterns = [
            r'section\s+\d+',
            r'article\s+\d+',
            r'clause\s+\d+',
            r'rule\s+\d+',
            r'chapter\s+\d+',
            r'part\s+\d+'
        ]

        detected_sections = []
        text_lower = text.lower()

        for pattern in law_patterns:
            matches = re.findall(pattern, text_lower)
            detected_sections.extend(matches)

        return list(set(detected_sections))  # Remove duplicates

    async def analyze_sentiment(self, text: str) -> SentimentResult:
        """Analyze sentiment of text."""
        start_time = time.time()

        if not text or not text.strip():
            return SentimentResult(
                text="",
                sentiment_label=SentimentLabel.NEUTRAL,
                confidence_score=0.0,
                positive_score=0.0,
                negative_score=0.0,
                neutral_score=0.0,
                processing_time_ms=0
            )

        processed_text = self._preprocess_text(text)
        positive_score, negative_score, neutral_score = self._calculate_sentiment_scores(processed_text)

        overall_sentiment = self._determine_overall_sentiment(positive_score, negative_score, neutral_score)
        confidence = max(positive_score, negative_score, neutral_score)

        processing_time = int((time.time() - start_time) * 1000)

        return SentimentResult(
            text=text,
            sentiment_label=overall_sentiment,
            confidence_score=confidence,
            positive_score=positive_score,
            negative_score=negative_score,
            neutral_score=neutral_score,
            compound_score=(positive_score - negative_score),  # Simple compound score
            processing_time_ms=processing_time
        )

    async def comprehensive_analysis(self, text: str) -> ComprehensiveAnalysisResult:
        """Perform comprehensive sentiment analysis."""
        start_time = time.time()

        # Analyze sentiment
        sentiment_result = await self.analyze_sentiment(text)
        sentiment_results = [sentiment_result]

        # Analyze emotions
        emotion_result = self._analyze_emotions(text)

        # Extract key phrases
        key_phrases = self._extract_key_phrases(text)

        # Detect law sections
        law_sections = self._detect_law_sections(text)

        # Determine overall sentiment and confidence
        overall_sentiment = sentiment_result.sentiment_label
        overall_confidence = sentiment_result.confidence_score

        processing_time = int((time.time() - start_time) * 1000)

        return ComprehensiveAnalysisResult(
            overall_sentiment=overall_sentiment,
            overall_confidence=overall_confidence,
            sentiment_results=sentiment_results,
            emotion_result=emotion_result,
            key_phrases=key_phrases,
            law_sections_mentioned=law_sections,
            processing_time_ms=processing_time
        )