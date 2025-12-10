"""
Sentiment analysis service with VADER, aspect-based analysis, and emotion classification.
Supports multiple analysis techniques for comprehensive sentiment insights.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Sentiment analysis libraries
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER_AVAILABLE = True
except Exception:
    SentimentIntensityAnalyzer = None  # type: ignore
    _VADER_AVAILABLE = False
try:
    from textblob import TextBlob
    _TEXTBLOB_AVAILABLE = True
except Exception:
    TextBlob = None  # type: ignore
    _TEXTBLOB_AVAILABLE = False
try:
    import spacy
    _SPACY_AVAILABLE = True
except Exception:
    spacy = None  # type: ignore
    _SPACY_AVAILABLE = False
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    _SKLEARN_AVAILABLE = True
except Exception:
    TfidfVectorizer = MultinomialNB = Pipeline = None  # type: ignore
    _SKLEARN_AVAILABLE = False
import numpy as np

from backend.app.models.analysis import SentimentLabel, EmotionLabel, AnalysisType
from backend.app.core.config import settings

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    _TRANSFORMERS_AVAILABLE = True
except Exception:
    pipeline = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    _TRANSFORMERS_AVAILABLE = False
from backend.app.services.preprocessing_service import TextPreprocessor


class AnalysisMethod(str, Enum):
    """Available sentiment analysis methods."""
    VADER = "vader"
    TEXTBLOB = "textblob"
    SPACY = "spacy"
    ENSEMBLE = "ensemble"
    TRANSFORMER = "transformer"


@dataclass
class SentimentResult:
    """Result of sentiment analysis."""
    method: str
    sentiment_label: SentimentLabel
    confidence_score: float
    positive_score: float
    negative_score: float
    neutral_score: float
    compound_score: Optional[float] = None
    raw_scores: Optional[Dict[str, float]] = None
    explanation: Optional[str] = None


@dataclass
class EmotionResult:
    """Result of emotion analysis."""
    emotion_label: EmotionLabel
    confidence_score: float
    emotion_scores: Dict[str, float]
    detected_emotions: List[str]


@dataclass
class AspectSentimentResult:
    """Result of aspect-based sentiment analysis."""
    aspect: str
    sentiment: SentimentLabel
    confidence: float
    context: str
    law_section: Optional[str] = None


@dataclass
class ComprehensiveAnalysisResult:
    """Complete analysis result combining all methods."""
    text: str
    sentiment_results: List[SentimentResult]
    emotion_result: EmotionResult
    aspect_sentiments: List[AspectSentimentResult]
    key_phrases: List[str]
    law_sections_mentioned: List[str]
    overall_sentiment: SentimentLabel
    overall_confidence: float
    processing_time_ms: int
    explanation: Dict[str, Any]


class SentimentAnalyzer:
    """Comprehensive sentiment analysis service."""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self._initialize_analyzers()
        self._initialize_emotion_patterns()
        self._initialize_aspect_patterns()
        
    def _initialize_analyzers(self):
        """Initialize sentiment analysis tools."""
        try:
            # VADER sentiment analyzer (optional, not used by default)
            self.vader_analyzer = SentimentIntensityAnalyzer() if _VADER_AVAILABLE else None
            
            # Load spaCy model if available
            try:
                self.nlp = spacy.load("en_core_web_sm") if _SPACY_AVAILABLE else None
            except Exception:
                print("⚠️ spaCy model not available for sentiment analysis")
                self.nlp = None

            # Initialize transformer pipelines lazily
            self._transformer_en = None
            self._transformer_multi = None
            self._transformer_indic = None  # For Indic languages
            self._transformer_ready = _TRANSFORMERS_AVAILABLE
            if not self._transformer_ready:
                print("⚠️ Transformers not available; install 'transformers' and 'torch' for advanced sentiment")
            
            # Legislative/Policy domain-specific keywords for enhanced analysis
            self.policy_keywords = {
                'strong_support': ['strongly support', 'fully endorse', 'completely agree', 'excellent', 'outstanding'],
                'support': ['support', 'agree', 'endorse', 'approve', 'welcome', 'appreciate', 'beneficial', 'positive', 'good'],
                'neutral_support': ['generally support', 'mostly agree', 'largely positive'],
                'neutral': ['neutral', 'balanced', 'mixed', 'unclear'],
                'neutral_oppose': ['some concerns', 'partially disagree', 'mixed feelings'],
                'oppose': ['oppose', 'disagree', 'reject', 'against', 'concerned', 'problematic', 'negative', 'bad', 'harmful'],
                'strong_oppose': ['strongly oppose', 'completely disagree', 'totally reject', 'terrible', 'awful'],
                'suggest': ['suggest', 'recommend', 'propose', 'consider', 'should', 'could', 'modification', 'amendment'],
                'concern': ['concern', 'worry', 'issue', 'problem', 'risk', 'challenge', 'difficulty', 'unclear']
            }
            
            # Stakeholder type indicators
            self.stakeholder_indicators = {
                'individual': ['citizen', 'individual', 'person', 'myself', 'i think', 'my opinion'],
                'business': ['company', 'business', 'organization', 'corporate', 'industry', 'commerce'],
                'ngo': ['ngo', 'non-profit', 'foundation', 'trust', 'society', 'association'],
                'academic': ['university', 'research', 'academic', 'professor', 'scholar', 'study'],
                'legal': ['lawyer', 'advocate', 'legal', 'bar association', 'law firm'],
                'government': ['ministry', 'department', 'government', 'official', 'authority']
            }
            
            print("✅ Sentiment analyzers initialized with policy domain enhancements")
            
        except Exception as e:
            print(f"❌ Error initializing sentiment analyzers: {e}")
            self.vader_analyzer = None
            self.nlp = None
            self._transformer_ready = False
    
    def _initialize_emotion_patterns(self):
        """Initialize emotion detection patterns and keywords."""
        self.emotion_keywords = {
            EmotionLabel.SUPPORT: [
                'support', 'agree', 'endorse', 'approve', 'favor', 'welcome', 
                'appreciate', 'excellent', 'great', 'fantastic', 'wonderful',
                'positive', 'beneficial', 'helpful', 'valuable', 'important'
            ],
            EmotionLabel.CONCERN: [
                'concern', 'worry', 'doubt', 'question', 'issue', 'problem',
                'challenge', 'difficulty', 'risk', 'danger', 'uncertain',
                'unsure', 'hesitant', 'cautious', 'careful'
            ],
            EmotionLabel.SUGGESTION: [
                'suggest', 'recommend', 'propose', 'consider', 'should',
                'could', 'might', 'perhaps', 'maybe', 'alternative',
                'improvement', 'modify', 'change', 'enhance', 'better'
            ],
            EmotionLabel.ANGER: [
                'angry', 'furious', 'outraged', 'disgusted', 'hate',
                'terrible', 'awful', 'horrible', 'ridiculous', 'absurd',
                'unacceptable', 'wrong', 'bad', 'worst', 'disaster'
            ],
            EmotionLabel.APPRECIATION: [
                'thank', 'grateful', 'appreciate', 'acknowledge', 'recognize',
                'commend', 'praise', 'admire', 'respect', 'honor',
                'value', 'treasure', 'cherish', 'pleased', 'satisfied'
            ],
            EmotionLabel.CONFUSION: [
                'confuse', 'unclear', 'understand', 'explain', 'clarify',
                'ambiguous', 'vague', 'complex', 'complicated', 'difficult',
                'what', 'how', 'why', 'when', 'where', 'help'
            ]
        }
        
        # Compile regex patterns for efficient matching
        self.emotion_patterns = {}
        for emotion, keywords in self.emotion_keywords.items():
            pattern = r'\b(?:' + '|'.join(keywords) + r')\b'
            self.emotion_patterns[emotion] = re.compile(pattern, re.IGNORECASE)
    
    def _initialize_aspect_patterns(self):
        """Initialize patterns for aspect detection in legal context."""
        self.aspect_patterns = {
            'law_sections': re.compile(
                r'section\s+(\d+(?:\.\d+)?)|article\s+(\d+)|clause\s+(\d+)|paragraph\s+(\d+)',
                re.IGNORECASE
            ),
            'legal_terms': [
                'regulation', 'compliance', 'enforcement', 'penalty', 'fine',
                'rights', 'obligations', 'procedure', 'process', 'framework',
                'implementation', 'timeline', 'deadline', 'requirement'
            ],
            'stakeholder_concerns': [
                'impact', 'effect', 'consequence', 'result', 'outcome',
                'benefit', 'cost', 'expense', 'burden', 'resource',
                'time', 'effort', 'difficulty', 'challenge'
            ]
        }
    
    async def analyze_sentiment(self, text: str, 
                              methods: List[AnalysisMethod] = None) -> List[SentimentResult]:
        """
        Perform sentiment analysis using specified methods.
        
        Args:
            text: Text to analyze
            methods: List of analysis methods to use
            
        Returns:
            list: List of sentiment results from different methods
        """
        if methods is None:
            methods = [AnalysisMethod.TRANSFORMER]
        
        results = []
        
        # Transformer analysis (preferred)
        if AnalysisMethod.TRANSFORMER in methods:
            # Run even if transformers unavailable; fall back to heuristic
            transformer_result = await self._analyze_with_transformer(text)
            if transformer_result:
                results.append(transformer_result)
        
        # VADER analysis
        if AnalysisMethod.VADER in methods and self.vader_analyzer:
            vader_result = await self._analyze_with_vader(text)
            if vader_result:
                results.append(vader_result)
        
        # TextBlob analysis
        if AnalysisMethod.TEXTBLOB in methods:
            textblob_result = await self._analyze_with_textblob(text)
            if textblob_result:
                results.append(textblob_result)
        
        # spaCy analysis (if available)
        if AnalysisMethod.SPACY in methods and self.nlp:
            spacy_result = await self._analyze_with_spacy(text)
            if spacy_result:
                results.append(spacy_result)
        
        # Ensemble method
        if AnalysisMethod.ENSEMBLE in methods and len(results) > 1:
            ensemble_result = self._create_ensemble_result(results)
            results.append(ensemble_result)
        
        return results

    def _ensure_transformers(self):
        """Lazy-load transformer pipelines for English and multilingual texts."""
        if not self._transformer_ready:
            return
        try:
            if self._transformer_en is None:
                # RoBERTa base sentiment for English
                # cardiffnlp/twitter-roberta-base-sentiment-latest returns labels: negative/neutral/positive
                self._transformer_en = pipeline(
                    "text-classification",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    top_k=None,
                    device=-1
                )
            if self._transformer_multi is None:
                # XLM-R multilingual sentiment
                self._transformer_multi = pipeline(
                    "text-classification",
                    model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
                    top_k=None,
                    device=-1
                )

                # IndicBERT for Indian languages (Hindi, etc.)
                try:
                    self._transformer_indic = pipeline(
                        "text-classification",
                        model="ai4bharat/indic-bert",
                        tokenizer="ai4bharat/indic-bert",
                        top_k=None,
                        device=-1
                    )
                except Exception as e:
                    print(f"⚠️ IndicBERT not available: {e}")
                    self._transformer_indic = None
        except Exception as e:
            print(f"⚠️ Failed to initialize transformer pipelines: {e}")
            self._transformer_ready = False

    async def _analyze_with_transformer(self, text: str) -> Optional[SentimentResult]:
        """Analyze sentiment using transformer models (English and multilingual)."""
        try:
            self._ensure_transformers()
            if not self._transformer_ready:
                return None
            # Detect language to pick pipeline
            lang, _ = self.preprocessor._detect_language(text)

            # Select appropriate model based on language
            if lang == 'en' and self._transformer_en is not None:
                clf = self._transformer_en
            elif lang in ['hi', 'bn', 'te', 'mr', 'ta', 'ur', 'gu', 'pa', 'or', 'as', 'mai', 'bho', 'awa', 'bh', 'new'] and self._transformer_indic is not None:
                # Indic languages: Hindi, Bengali, Telugu, Marathi, Tamil, Urdu, Gujarati, Punjabi, Oriya, Assamese, Maithili, Bhojpuri, Awadhi, Bihari, Nepali
                clf = self._transformer_indic
            else:
                clf = self._transformer_multi
            if clf is None:
                return None
            preds = clf(text)
            # HF may return list of dicts or list[list[dict]] depending on top_k
            scores_map: Dict[str, float] = { }
            if preds and isinstance(preds, list):
                first = preds[0]
                if isinstance(first, dict) and 'label' in first:
                    # Single best label only
                    scores_map[first['label'].lower()] = float(first['score'])
                elif isinstance(first, list):
                    for item in first:
                        scores_map[item['label'].lower()] = float(item['score'])
            # Normalize keys to positive/neutral/negative
            positive = scores_map.get('positive', scores_map.get('pos', 0.0))
            neutral = scores_map.get('neutral', scores_map.get('neu', 0.0))
            negative = scores_map.get('negative', scores_map.get('neg', 0.0))
            # Pick label and confidence
            components = {'positive': positive, 'negative': negative, 'neutral': neutral}
            label_str = max(components, key=components.get) if components else 'neutral'
            confidence = components.get(label_str, 0.0)

            sentiment_label = {
                'positive': SentimentLabel.POSITIVE,
                'negative': SentimentLabel.NEGATIVE,
                'neutral': SentimentLabel.NEUTRAL,
            }[label_str]

            # Generate explanation
            explanation = f"Analysis using transformer model for language '{lang}'. "
            explanation += f"Detected sentiment: {sentiment_label.value} with {confidence:.1%} confidence. "
            explanation += f"Scores: Positive={positive:.2f}, Negative={negative:.2f}, Neutral={neutral:.2f}."

            return SentimentResult(
                method="Transformer",
                sentiment_label=sentiment_label,
                confidence_score=confidence,
                positive_score=positive,
                negative_score=negative,
                neutral_score=neutral,
                compound_score=None,
                raw_scores={"model": "transformer", "lang": lang, **components},
                explanation=explanation
            )
        except Exception as e:
            print(f"Error in Transformer analysis: {e}")
            return None
    
    async def _analyze_with_vader(self, text: str) -> Optional[SentimentResult]:
        """Analyze sentiment using VADER."""
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            
            # Determine sentiment label based on compound score
            compound = scores['compound']
            if compound >= 0.05:
                sentiment_label = SentimentLabel.POSITIVE
                confidence = abs(compound)
            elif compound <= -0.05:
                sentiment_label = SentimentLabel.NEGATIVE
                confidence = abs(compound)
            else:
                sentiment_label = SentimentLabel.NEUTRAL
                confidence = 1 - abs(compound)
            
            return SentimentResult(
                method="VADER",
                sentiment_label=sentiment_label,
                confidence_score=confidence,
                positive_score=scores['pos'],
                negative_score=scores['neg'],
                neutral_score=scores['neu'],
                compound_score=compound,
                raw_scores=scores
            )
            
        except Exception as e:
            print(f"Error in VADER analysis: {e}")
            return None
    
    async def _analyze_with_textblob(self, text: str) -> Optional[SentimentResult]:
        """Analyze sentiment using TextBlob."""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            # Convert polarity to sentiment label
            if polarity > 0.1:
                sentiment_label = SentimentLabel.POSITIVE
            elif polarity < -0.1:
                sentiment_label = SentimentLabel.NEGATIVE
            else:
                sentiment_label = SentimentLabel.NEUTRAL
            
            # Calculate confidence and normalize scores
            confidence = abs(polarity)
            if sentiment_label == SentimentLabel.NEUTRAL:
                confidence = 1 - abs(polarity)
            
            # Normalize to 0-1 range and create component scores
            normalized_polarity = (polarity + 1) / 2  # Convert from [-1,1] to [0,1]
            
            return SentimentResult(
                method="TextBlob",
                sentiment_label=sentiment_label,
                confidence_score=confidence,
                positive_score=max(0, polarity),
                negative_score=max(0, -polarity),
                neutral_score=1 - abs(polarity),
                compound_score=polarity,
                raw_scores={"polarity": polarity, "subjectivity": blob.sentiment.subjectivity}
            )
            
        except Exception as e:
            print(f"Error in TextBlob analysis: {e}")
            return None
    
    async def _analyze_with_spacy(self, text: str) -> Optional[SentimentResult]:
        """Analyze sentiment using spaCy (basic implementation)."""
        try:
            if not self.nlp:
                return None
            
            doc = self.nlp(text)
            
            # Basic rule-based sentiment using token attributes
            positive_words = 0
            negative_words = 0
            total_words = 0
            
            for token in doc:
                if token.is_alpha and not token.is_stop:
                    total_words += 1
                    # Simple heuristic based on token sentiment (if available)
                    # This is a placeholder - in practice, you'd use a trained model
                    if any(pos_word in token.text.lower() for pos_word in ['good', 'great', 'excellent', 'support']):
                        positive_words += 1
                    elif any(neg_word in token.text.lower() for neg_word in ['bad', 'terrible', 'awful', 'oppose']):
                        negative_words += 1
            
            if total_words == 0:
                return None
            
            pos_ratio = positive_words / total_words
            neg_ratio = negative_words / total_words
            neu_ratio = 1 - pos_ratio - neg_ratio
            
            # Determine sentiment
            if pos_ratio > neg_ratio and pos_ratio > 0.1:
                sentiment_label = SentimentLabel.POSITIVE
                confidence = pos_ratio
            elif neg_ratio > pos_ratio and neg_ratio > 0.1:
                sentiment_label = SentimentLabel.NEGATIVE
                confidence = neg_ratio
            else:
                sentiment_label = SentimentLabel.NEUTRAL
                confidence = neu_ratio
            
            return SentimentResult(
                method="spaCy",
                sentiment_label=sentiment_label,
                confidence_score=confidence,
                positive_score=pos_ratio,
                negative_score=neg_ratio,
                neutral_score=neu_ratio,
                raw_scores={"pos_words": positive_words, "neg_words": negative_words, "total_words": total_words}
            )
            
        except Exception as e:
            print(f"Error in spaCy analysis: {e}")
            return None
    
    def _create_ensemble_result(self, results: List[SentimentResult]) -> SentimentResult:
        """Create ensemble result by combining multiple methods."""
        if not results:
            return None
        
        # Weight different methods
        method_weights = {
            "Transformer": 0.6,
            "VADER": 0.2,
            "TextBlob": 0.1,
            "spaCy": 0.1
        }
        
        # Weighted average of scores
        weighted_pos = sum(r.positive_score * method_weights.get(r.method, 0.33) for r in results)
        weighted_neg = sum(r.negative_score * method_weights.get(r.method, 0.33) for r in results)
        weighted_neu = sum(r.neutral_score * method_weights.get(r.method, 0.33) for r in results)
        
        # Determine overall sentiment
        max_score = max(weighted_pos, weighted_neg, weighted_neu)
        if max_score == weighted_pos:
            sentiment_label = SentimentLabel.POSITIVE
            confidence = weighted_pos
        elif max_score == weighted_neg:
            sentiment_label = SentimentLabel.NEGATIVE
            confidence = weighted_neg
        else:
            sentiment_label = SentimentLabel.NEUTRAL
            confidence = weighted_neu
        
        return SentimentResult(
            method="Ensemble",
            sentiment_label=sentiment_label,
            confidence_score=confidence,
            positive_score=weighted_pos,
            negative_score=weighted_neg,
            neutral_score=weighted_neu,
            raw_scores={
                "component_methods": [r.method for r in results],
                "individual_confidences": [r.confidence_score for r in results]
            }
        )
    
    async def analyze_emotions(self, text: str) -> EmotionResult:
        """
        Analyze emotions using keyword-based approach.
        
        Args:
            text: Text to analyze
            
        Returns:
            EmotionResult: Detected emotions with scores
        """
        text_lower = text.lower()
        emotion_scores = {}
        detected_emotions = []
        
        # Check each emotion pattern
        for emotion, pattern in self.emotion_patterns.items():
            matches = pattern.findall(text)
            score = len(matches) / len(text.split()) if text.split() else 0
            emotion_scores[emotion.value] = score
            
            if score > 0:
                detected_emotions.append(emotion.value)
        
        # Determine primary emotion
        if emotion_scores:
            primary_emotion = max(emotion_scores, key=emotion_scores.get)
            confidence = emotion_scores[primary_emotion]
            
            # If no clear emotion detected, default to neutral
            if confidence == 0:
                primary_emotion = EmotionLabel.NEUTRAL.value
                confidence = 0.8
                emotion_scores[EmotionLabel.NEUTRAL.value] = confidence
        else:
            primary_emotion = EmotionLabel.NEUTRAL.value
            confidence = 0.8
            emotion_scores = {EmotionLabel.NEUTRAL.value: confidence}
        
        return EmotionResult(
            emotion_label=EmotionLabel(primary_emotion),
            confidence_score=confidence,
            emotion_scores=emotion_scores,
            detected_emotions=detected_emotions
        )
    
    async def analyze_policy_sentiment(self, text: str) -> SentimentResult:
        """
        Enhanced sentiment analysis specifically for legislative/policy comments.
        
        Args:
            text: Comment text to analyze
            
        Returns:
            SentimentResult: Policy-specific sentiment analysis
        """
        try:
            text_lower = text.lower()
            
            # Initialize scores
            policy_scores = {
                'strong_support': 0,
                'support': 0,
                'neutral_support': 0,
                'neutral': 0,
                'neutral_oppose': 0,
                'oppose': 0,
                'strong_oppose': 0
            }
            
            # Count policy-specific keywords
            word_count = len(text.split())
            for category, keywords in self.policy_keywords.items():
                if category in policy_scores:
                    matches = sum(1 for keyword in keywords if keyword in text_lower)
                    policy_scores[category] = matches / word_count if word_count > 0 else 0
            
            # Determine sentiment based on policy keywords
            max_category = max(policy_scores, key=policy_scores.get)
            max_score = policy_scores[max_category]
            
            # Map to sentiment labels
            if max_category in ['strong_support', 'support', 'neutral_support']:
                sentiment_label = SentimentLabel.POSITIVE
                confidence = max_score * 2  # Boost confidence for clear policy language
            elif max_category in ['strong_oppose', 'oppose', 'neutral_oppose']:
                sentiment_label = SentimentLabel.NEGATIVE
                confidence = max_score * 2
            else:
                sentiment_label = SentimentLabel.NEUTRAL
                confidence = 0.7  # Default confidence for neutral
            
            # Ensure confidence is in valid range
            confidence = min(confidence, 1.0)
            if confidence < 0.3:  # If no clear policy keywords, fall back to transformer
                transformer_result = await self._analyze_with_transformer(text)
                if transformer_result:
                    return transformer_result
            
            # Calculate component scores
            positive_score = policy_scores['strong_support'] + policy_scores['support'] + policy_scores['neutral_support']
            negative_score = policy_scores['strong_oppose'] + policy_scores['oppose'] + policy_scores['neutral_oppose']
            neutral_score = policy_scores['neutral'] + (1 - positive_score - negative_score)
            
            # Normalize scores
            total = positive_score + negative_score + neutral_score
            if total > 0:
                positive_score /= total
                negative_score /= total
                neutral_score /= total
            
            return SentimentResult(
                method="Policy-Enhanced",
                sentiment_label=sentiment_label,
                confidence_score=confidence,
                positive_score=positive_score,
                negative_score=negative_score,
                neutral_score=neutral_score,
                raw_scores={
                    "policy_categories": policy_scores,
                    "max_category": max_category,
                    "stakeholder_type": self._detect_stakeholder_type(text)
                }
            )
            
        except Exception as e:
            print(f"Error in policy sentiment analysis: {e}")
            # Fall back to transformer analysis
            return await self._analyze_with_transformer(text)
    
    def _detect_stakeholder_type(self, text: str) -> str:
        """Detect the type of stakeholder based on text content."""
        text_lower = text.lower()
        stakeholder_scores = {}
        
        for stakeholder_type, indicators in self.stakeholder_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            stakeholder_scores[stakeholder_type] = score
        
        if stakeholder_scores:
            detected_type = max(stakeholder_scores, key=stakeholder_scores.get)
            if stakeholder_scores[detected_type] > 0:
                return detected_type
        
        return "unknown"
    
    async def analyze_aspect_sentiment(self, text: str) -> List[AspectSentimentResult]:
        """
        Perform aspect-based sentiment analysis to identify sentiment toward specific aspects.
        
        Args:
            text: Text to analyze
            
        Returns:
            list: List of aspect sentiment results
        """
        results = []
        
        # Find law sections mentioned
        law_sections = self._extract_law_sections(text)
        
        # Analyze sentiment for each detected aspect
        sentences = text.split('.')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue
            
            # Check if sentence mentions legal terms or sections
            aspects = self._identify_aspects_in_sentence(sentence)
            
            if aspects:
                # Analyze sentiment of this sentence
                sentiment_results = await self.analyze_sentiment(sentence, [AnalysisMethod.TRANSFORMER])
                
                if sentiment_results:
                    sentiment_result = sentiment_results[0]
                    
                    for aspect in aspects:
                        results.append(AspectSentimentResult(
                            aspect=aspect,
                            sentiment=sentiment_result.sentiment_label,
                            confidence=sentiment_result.confidence_score,
                            context=sentence,
                            law_section=self._find_law_section_in_text(sentence)
                        ))
        
        return results
    
    def _extract_law_sections(self, text: str) -> List[str]:
        """Extract mentioned law sections from text."""
        sections = []
        matches = self.aspect_patterns['law_sections'].finditer(text)
        
        for match in matches:
            # Extract the section number from any of the capture groups
            section_num = next((group for group in match.groups() if group), None)
            if section_num:
                sections.append(f"Section {section_num}")
        
        return list(set(sections))  # Remove duplicates
    
    def _identify_aspects_in_sentence(self, sentence: str) -> List[str]:
        """Identify aspects mentioned in a sentence."""
        aspects = []
        sentence_lower = sentence.lower()
        
        # Check for legal terms
        for term in self.aspect_patterns['legal_terms']:
            if term in sentence_lower:
                aspects.append(term.title())
        
        # Check for stakeholder concerns
        for concern in self.aspect_patterns['stakeholder_concerns']:
            if concern in sentence_lower:
                aspects.append(concern.title())
        
        return list(set(aspects))
    
    def _find_law_section_in_text(self, text: str) -> Optional[str]:
        """Find law section reference in text."""
        match = self.aspect_patterns['law_sections'].search(text)
        if match:
            section_num = next((group for group in match.groups() if group), None)
            return f"Section {section_num}" if section_num else None
        return None
    
    async def comprehensive_analysis(self, text: str) -> ComprehensiveAnalysisResult:
        """
        Perform comprehensive sentiment and emotion analysis.
        
        Args:
            text: Text to analyze
            
        Returns:
            ComprehensiveAnalysisResult: Complete analysis results
        """
        import time
        start_time = time.time()
        
        # Parallel analysis execution with policy-enhanced analysis
        sentiment_task = asyncio.create_task(
            self.analyze_sentiment(text, [AnalysisMethod.TRANSFORMER])
        )
        policy_task = asyncio.create_task(self.analyze_policy_sentiment(text))
        emotion_task = asyncio.create_task(self.analyze_emotions(text))
        aspect_task = asyncio.create_task(self.analyze_aspect_sentiment(text))
        
        # Wait for all analyses to complete
        sentiment_results, policy_result, emotion_result, aspect_sentiments = await asyncio.gather(
            sentiment_task, policy_task, emotion_task, aspect_task
        )
        
        # Add policy-enhanced result to sentiment results
        if policy_result:
            sentiment_results.append(policy_result)
        
        # Extract key phrases
        key_phrases = []
        if self.preprocessor:
            try:
                key_phrases = [phrase['text'] for phrase in self.preprocessor.extract_key_phrases(text, max_phrases=5)]
            except:
                pass
        
        # Extract law sections
        law_sections = self._extract_law_sections(text)
        
        # Determine overall sentiment (prefer policy-enhanced, then transformer)
        overall_sentiment = SentimentLabel.NEUTRAL
        overall_confidence = 0.5
        
        if sentiment_results:
            # Prefer policy-enhanced analysis for legislative/policy comments
            policy_result = next((r for r in sentiment_results if r.method == "Policy-Enhanced"), None)
            transformer_result = next((r for r in sentiment_results if r.method == "Transformer"), None)
            
            # Use policy result if confidence is high enough, otherwise fall back to transformer
            if policy_result and policy_result.confidence_score >= 0.6:
                best_result = policy_result
            else:
                best_result = transformer_result or policy_result or sentiment_results[0]
            
            overall_sentiment = best_result.sentiment_label
            overall_confidence = best_result.confidence_score
        
        # Create explanation
        explanation = self._create_analysis_explanation(
            sentiment_results, emotion_result, aspect_sentiments, key_phrases
        )
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return ComprehensiveAnalysisResult(
            text=text,
            sentiment_results=sentiment_results,
            emotion_result=emotion_result,
            aspect_sentiments=aspect_sentiments,
            key_phrases=key_phrases,
            law_sections_mentioned=law_sections,
            overall_sentiment=overall_sentiment,
            overall_confidence=overall_confidence,
            processing_time_ms=processing_time,
            explanation=explanation
        )
    
    def _create_analysis_explanation(self, sentiment_results: List[SentimentResult],
                                   emotion_result: EmotionResult,
                                   aspect_sentiments: List[AspectSentimentResult],
                                   key_phrases: List[str]) -> Dict[str, Any]:
        """Create human-readable explanation of analysis results."""
        explanation = {
            "sentiment_summary": "",
            "emotion_summary": "",
            "key_findings": [],
            "confidence_notes": [],
            "methodology": []
        }
        
        # Sentiment explanation
        if sentiment_results:
            primary_result = sentiment_results[0]
            explanation["sentiment_summary"] = (
                f"The text expresses {primary_result.sentiment_label.value} sentiment "
                f"with {primary_result.confidence_score:.2%} confidence."
            )
            
            explanation["methodology"].append(f"Sentiment analyzed using {len(sentiment_results)} methods")
        
        # Emotion explanation
        if emotion_result:
            explanation["emotion_summary"] = (
                f"Primary emotion detected: {emotion_result.emotion_label.value} "
                f"(confidence: {emotion_result.confidence_score:.2%})"
            )
            
            if len(emotion_result.detected_emotions) > 1:
                explanation["key_findings"].append(
                    f"Multiple emotions detected: {', '.join(emotion_result.detected_emotions)}"
                )
        
        # Aspect-based findings
        if aspect_sentiments:
            aspect_summary = {}
            for aspect_result in aspect_sentiments:
                if aspect_result.aspect not in aspect_summary:
                    aspect_summary[aspect_result.aspect] = []
                aspect_summary[aspect_result.aspect].append(aspect_result.sentiment.value)
            
            for aspect, sentiments in aspect_summary.items():
                explanation["key_findings"].append(
                    f"{aspect}: {', '.join(set(sentiments))} sentiment"
                )
        
        # Key phrases
        if key_phrases:
            explanation["key_findings"].append(f"Key phrases: {', '.join(key_phrases[:3])}")
        
        return explanation
    
    async def batch_analysis(self, texts: List[str], max_workers: int = 4) -> List[ComprehensiveAnalysisResult]:
        """
        Perform batch analysis on multiple texts.
        
        Args:
            texts: List of texts to analyze
            max_workers: Maximum number of worker threads
            
        Returns:
            list: List of comprehensive analysis results
        """
        # Create tasks for all texts
        tasks = []
        for text in texts:
            task = asyncio.create_task(self.comprehensive_analysis(text))
            tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions and return results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Error analyzing text {i}: {result}")
                # Create minimal error result
                final_results.append(None)
            else:
                final_results.append(result)
        
        return [r for r in final_results if r is not None]