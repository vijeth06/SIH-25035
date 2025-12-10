"""
Advanced Sentiment Analysis Service

This module provides advanced sentiment analysis capabilities using multiple models:
1. Transformer-based models (BERT, RoBERTa, etc.)
2. Aspect-based sentiment analysis
3. Emotion detection
4. Sarcasm detection
5. Multilingual support
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import spacy
from flair.models import TextClassifier
from flair.data import Sentence
import torch

class AdvancedSentimentResult(BaseModel):
    """Result model for advanced sentiment analysis."""
    text: str
    overall_sentiment: str  # positive, negative, neutral
    sentiment_score: float  # -1.0 to 1.0
    emotions: Dict[str, float]  # emotion: confidence
    aspects: List[Dict[str, str]]  # aspect: sentiment
    is_sarcastic: bool = False
    confidence: float
    language: str = "en"
    model_used: str

class AdvancedSentimentAnalyzer:
    """Advanced sentiment analysis using multiple models and techniques."""
    
    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        self.models = {}
        # Don't load models during initialization, load them lazily when needed
        self.models_loaded = False
    
    def _ensure_models_loaded(self):
        """Load all required models if not already loaded."""
        if self.models_loaded:
            return
            
        try:
            # Transformer-based sentiment analysis
            self.models["sentiment"] = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=self.device
            )
            
            # Emotion detection
            self.models["emotion"] = pipeline(
                "text-classification",
                model="bhadresh-savani/bert-base-uncased-emotion",
                return_all_scores=True,
                device=self.device
            )
            
            # Sarcasm detection
            self.models["sarcasm"] = pipeline(
                "text-classification",
                model="mrm8488/t5-base-finetuned-sarcasm-twitter",
                device=self.device
            )
            
            # Load spaCy for NLP tasks
            try:
                self.nlp = spacy.load("en_core_web_lg")
            except OSError:
                # If model not found, download it
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_lg"])
                self.nlp = spacy.load("en_core_web_lg")
            
            # Initialize Flair for aspect-based sentiment analysis
            self.aspect_classifier = TextClassifier.load('en-sentiment')
            
            self.models_loaded = True
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
    
    def analyze_sentiment(self, text: str) -> AdvancedSentimentResult:
        """
        Perform advanced sentiment analysis on the input text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            AdvancedSentimentResult containing detailed sentiment analysis
        """
        # Load models if not already loaded
        self._ensure_models_loaded()
        
        # Basic sentiment analysis
        sentiment_result = self.models["sentiment"](text)[0]
        
        # Emotion detection
        emotion_results = self.models["emotion"](text)[0]
        emotions = {item['label']: float(item['score']) for item in emotion_results}
        
        # Sarcasm detection
        sarcasm_result = self.models["sarcasm"](text)[0]
        is_sarcastic = sarcasm_result['label'] == 'sarcasm' and sarcasm_result['score'] > 0.7
        
        # Aspect-based sentiment analysis
        aspects = self._extract_aspects(text)
        
        # Determine overall confidence
        confidence = max(sentiment_result['score'], 
                        max(emotions.values()),
                        sarcasm_result['score'])
        
        return AdvancedSentimentResult(
            text=text,
            overall_sentiment=sentiment_result['label'].lower(),
            sentiment_score=self._convert_sentiment_to_score(sentiment_result['label'], 
                                                          sentiment_result['score']),
            emotions=emotions,
            aspects=aspects,
            is_sarcastic=is_sarcastic,
            confidence=confidence,
            model_used="distilbert-base-uncased-finetuned-sst-2-english"
        )
    
    def _extract_aspects(self, text: str) -> List[Dict[str, str]]:
        """Extract aspects and their sentiments from text."""
        # Load models if not already loaded
        self._ensure_models_loaded()
        
        doc = self.nlp(text)
        aspects = []
        
        # Extract noun phrases as potential aspects
        for chunk in doc.noun_chunks:
            # Skip very short chunks
            if len(chunk.text.split()) < 2:
                continue
                
            # Analyze sentiment of the aspect
            sentence = Sentence(chunk.text)
            self.aspect_classifier.predict(sentence)
            
            aspects.append({
                "aspect": chunk.text,
                "sentiment": sentence.labels[0].value.lower(),
                "score": float(sentence.labels[0].score)
            })
        
        return aspects
    
    def _convert_sentiment_to_score(self, label: str, score: float) -> float:
        """Convert sentiment label and score to a -1 to 1 scale."""
        if label.lower() == 'positive':
            return score
        elif label.lower() == 'negative':
            return -score
        return 0.0  # neutral

    async def analyze_batch(self, texts: List[str]) -> List[AdvancedSentimentResult]:
        """Analyze a batch of texts asynchronously."""
        # Load models if not already loaded
        self._ensure_models_loaded()
        
        import asyncio
        
        async def analyze_text(text):
            return self.analyze_sentiment(text)
            
        tasks = [analyze_text(text) for text in texts]
        return await asyncio.gather(*tasks)


# Singleton instance
advanced_analyzer = AdvancedSentimentAnalyzer()