"""
Summarization service providing both extractive and abstractive summarization.
Supports TextRank extractive summarization and transformer-based abstractive summarization.
"""

import re
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
from collections import defaultdict, Counter
import numpy as np
from functools import lru_cache

# NLP libraries
try:
    import spacy
    _SPACY_SUMMARY_AVAILABLE = True
except Exception:
    spacy = None  # type: ignore
    _SPACY_SUMMARY_AVAILABLE = False
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.cluster.util import cosine_distance

# Summarization libraries
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.edmundson import EdmundsonSummarizer

# Optional: Transformer-based abstractive summarization
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
except Exception:
    pipeline = None
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers not available for abstractive summarization")

from backend.app.core.config import settings


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
    BERT = "bert"
    T5 = "t5"
    MT5 = "mt5"  # Multilingual T5
    BART = "bart"
    INDICBART = "indicbart"  # For Indian languages
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


@dataclass
class AggregateSummaryResult:
    """Result of aggregate summarization across multiple texts."""
    section_summaries: Dict[str, SummaryResult]
    overall_summary: SummaryResult
    key_themes: List[str]
    sentiment_distribution: Dict[str, int]
    total_comments: int
    processing_statistics: Dict[str, Any]


class TextRankSummarizer:
    """Custom implementation of TextRank algorithm for extractive summarization."""
    
    def __init__(self):
        self.similarity_threshold = 0.1
        self.damping_factor = 0.85
        self.max_iterations = 100
        self.convergence_threshold = 0.0001
        
        # Initialize NLTK data if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
        self.stop_words = set(stopwords.words('english'))
    
    def sentence_similarity(self, sent1: str, sent2: str) -> float:
        """Calculate cosine similarity between two sentences."""
        # Tokenize and remove stopwords
        words1 = [word.lower() for word in word_tokenize(sent1) if word.isalpha() and word.lower() not in self.stop_words]
        words2 = [word.lower() for word in word_tokenize(sent2) if word.isalpha() and word.lower() not in self.stop_words]
        
        # Get all unique words
        all_words = list(set(words1 + words2))
        
        # Create vectors
        vector1 = [words1.count(word) for word in all_words]
        vector2 = [words2.count(word) for word in all_words]
        
        # Calculate cosine similarity
        return 1 - cosine_distance(vector1, vector2)
    
    def build_similarity_matrix(self, sentences: List[str]) -> np.ndarray:
        """Build similarity matrix for sentences."""
        n = len(sentences)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    similarity = self.sentence_similarity(sentences[i], sentences[j])
                    if similarity > self.similarity_threshold:
                        similarity_matrix[i][j] = similarity
        
        return similarity_matrix
    
    def textrank_algorithm(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """Apply TextRank algorithm to similarity matrix."""
        n = similarity_matrix.shape[0]
        
        # Initialize scores
        scores = np.ones(n) / n
        
        # Normalize similarity matrix
        for i in range(n):
            sum_row = np.sum(similarity_matrix[i])
            if sum_row > 0:
                similarity_matrix[i] = similarity_matrix[i] / sum_row
        
        # Iterative calculation
        for iteration in range(self.max_iterations):
            new_scores = np.zeros(n)
            
            for i in range(n):
                for j in range(n):
                    if i != j:
                        new_scores[i] += similarity_matrix[j][i] * scores[j]
                
                new_scores[i] = (1 - self.damping_factor) + self.damping_factor * new_scores[i]
            
            # Check convergence
            if np.sum(np.abs(scores - new_scores)) < self.convergence_threshold:
                break
                
            scores = new_scores
        
        return scores
    
    def summarize(self, text: str, num_sentences: int = 3) -> Tuple[str, List[str], float]:
        """
        Summarize text using TextRank algorithm.
        
        Args:
            text: Text to summarize
            num_sentences: Number of sentences in summary
            
        Returns:
            tuple: (summary_text, key_sentences, confidence_score)
        """
        # Tokenize into sentences
        sentences = sent_tokenize(text)
        
        if len(sentences) <= num_sentences:
            return text, sentences, 1.0
        
        # Build similarity matrix
        similarity_matrix = self.build_similarity_matrix(sentences)
        
        # Apply TextRank algorithm
        scores = self.textrank_algorithm(similarity_matrix)
        
        # Get top sentences
        ranked_sentences = [(scores[i], i, sentences[i]) for i in range(len(sentences))]
        ranked_sentences.sort(reverse=True, key=lambda x: x[0])
        
        # Select top sentences and maintain original order
        selected_indices = sorted([item[1] for item in ranked_sentences[:num_sentences]])
        summary_sentences = [sentences[i] for i in selected_indices]
        
        # Calculate confidence (average of selected sentence scores)
        confidence = np.mean([ranked_sentences[i][0] for i in range(num_sentences)])
        
        return ' '.join(summary_sentences), summary_sentences, float(confidence)


class SummarizationService:
    """Comprehensive summarization service."""
    
    def __init__(self):
        self._initialize_models()
        self.custom_textrank = TextRankSummarizer()
        self._initialize_policy_keywords()
        # Default preference: use Indian models when possible
        self.prefer_indian_models = True
    
    def _initialize_policy_keywords(self):
        """Initialize policy and legislative domain keywords for enhanced summarization."""
        self.policy_keywords = {
            'key_actions': [
                'propose', 'recommend', 'suggest', 'implement', 'establish', 'create',
                'amend', 'modify', 'change', 'update', 'revise', 'reform',
                'approve', 'reject', 'support', 'oppose', 'endorse'
            ],
            'legislative_terms': [
                'section', 'clause', 'provision', 'amendment', 'bill', 'act',
                'regulation', 'rule', 'policy', 'law', 'statute', 'ordinance',
                'notification', 'circular', 'guideline', 'framework'
            ],
            'stakeholder_concerns': [
                'concern', 'issue', 'problem', 'challenge', 'difficulty',
                'benefit', 'advantage', 'impact', 'effect', 'consequence',
                'risk', 'opportunity', 'solution', 'alternative'
            ],
            'sentiment_indicators': [
                'support', 'oppose', 'agree', 'disagree', 'welcome', 'concerned',
                'positive', 'negative', 'beneficial', 'harmful', 'effective', 'ineffective'
            ]
        }
        
        # Stakeholder type patterns for categorization
        self.stakeholder_patterns = {
            'individual': r'\b(i|my|personally|citizen|individual)\b',
            'business': r'\b(company|business|corporate|industry|firm|organization)\b',
            'ngo': r'\b(ngo|non-profit|foundation|trust|society|association)\b',
            'academic': r'\b(university|research|academic|professor|scholar|study)\b',
            'legal': r'\b(lawyer|advocate|legal|bar|law firm)\b',
            'government': r'\b(ministry|department|government|official|authority)\b'
        }
    
    def _initialize_models(self):
        """Initialize summarization models and tools."""
        try:
            # Load spaCy model if available
            try:
                self.nlp = spacy.load("en_core_web_sm") if _SPACY_SUMMARY_AVAILABLE else None
            except Exception:
                print("⚠️ spaCy model not available")
                self.nlp = None
            
            # Initialize Sumy summarizers
            self.sumy_summarizers = {
                SummarizationMethod.TEXTRANK: TextRankSummarizer(),
                SummarizationMethod.LSA: LsaSummarizer(),
                SummarizationMethod.LUHN: LuhnSummarizer(),
                SummarizationMethod.EDMUNDSON: EdmundsonSummarizer()
            }
            
            # Initialize transformer models if available
            self.transformer_summarizers = {}
            if TRANSFORMERS_AVAILABLE:
                self._initialize_transformer_models()
            
            print("✅ Summarization service initialized")
            
        except Exception as e:
            print(f"❌ Error initializing summarization service: {e}")
    
    def _initialize_transformer_models(self):
        """Initialize transformer-based summarization models."""
        try:
            if TRANSFORMERS_AVAILABLE:
                # T5 model for abstractive summarization
                self.transformer_summarizers[SummarizationMethod.T5] = pipeline(
                    "summarization",
                    model="t5-small",
                    tokenizer="t5-small",
                    framework="pt"
                )

                # mT5 multilingual model
                try:
                    self.transformer_summarizers[SummarizationMethod.MT5] = pipeline(
                        "summarization",
                        model="google/mt5-small",
                        tokenizer="google/mt5-small",
                        framework="pt"
                    )
                except Exception as e:
                    print(f"⚠️ Could not load mT5 model: {e}")

                # IndicBART for Indian languages
                try:
                    self.transformer_summarizers[SummarizationMethod.INDICBART] = pipeline(
                        "summarization",
                        model="ai4bharat/IndicBART",
                        tokenizer="ai4bharat/IndicBART",
                        framework="pt"
                    )
                except Exception as e:
                    print(f"⚠️ Could not load IndicBART model: {e}")

                # BART model (if resources allow)
                # self.transformer_summarizers[SummarizationMethod.BART] = pipeline(
                #     "summarization",
                #     model="facebook/bart-large-cnn",
                #     framework="pt"
                # )

                print("✅ Transformer models loaded")
        except Exception as e:
            print(f"⚠️ Could not load all transformer models: {e}")
    
    async def extractive_summarization(self, text: str, 
                                     method: SummarizationMethod = SummarizationMethod.TEXTRANK,
                                     num_sentences: int = 3) -> SummaryResult:
        """
        Perform extractive summarization using specified method.
        
        Args:
            text: Text to summarize
            method: Summarization method to use
            num_sentences: Number of sentences in summary
            
        Returns:
            SummaryResult: Summarization result
        """
        import time
        start_time = time.time()
        
        original_length = len(text)
        
        try:
            if method == SummarizationMethod.CUSTOM_TEXTRANK:
                # Use custom TextRank implementation
                summary_text, key_sentences, confidence = self.custom_textrank.summarize(text, num_sentences)
                
            else:
                # Use Sumy summarizers
                summarizer = self.sumy_summarizers.get(method)
                if not summarizer:
                    raise ValueError(f"Unsupported extractive method: {method}")
                
                # Parse text
                parser = PlaintextParser.from_string(text, Tokenizer("english"))
                document = parser.document
                
                # Generate summary
                summary_sentences = summarizer(document, num_sentences)
                summary_text = ' '.join([str(sentence) for sentence in summary_sentences])
                key_sentences = [str(sentence) for sentence in summary_sentences]
                confidence = 0.8  # Default confidence for Sumy methods
            
            summary_length = len(summary_text)
            compression_ratio = summary_length / original_length if original_length > 0 else 0
            
            processing_time = int((time.time() - start_time) * 1000)
            
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
                    "original_sentence_count": len(text.split('.'))
                }
            )
            
        except Exception as e:
            print(f"Error in extractive summarization: {e}")
            # Return fallback summary
            sentences = text.split('.')[:num_sentences]
            fallback_summary = '. '.join(sentences) + '.'
            
            return SummaryResult(
                method="fallback",
                summary_type=SummarizationType.EXTRACTIVE,
                summary_text=fallback_summary,
                original_length=original_length,
                summary_length=len(fallback_summary),
                compression_ratio=len(fallback_summary) / original_length,
                key_sentences=sentences,
                confidence_score=0.3,
                processing_time_ms=int((time.time() - start_time) * 1000),
                metadata={"error": str(e), "fallback_used": True}
            )
    
    async def topic_based_summarization(self, text: str,
                                       topics: List[str] = None,
                                       max_length: int = 150,
                                       min_length: int = 30) -> SummaryResult:
        """
        Perform topic-based summarization using multilingual models.

        Args:
            text: Text to summarize
            topics: List of topics to focus on (optional)
            max_length: Maximum length of summary
            min_length: Minimum length of summary

        Returns:
            SummaryResult: Summarization result with topic focus
        """
        import time
        start_time = time.time()

        original_length = len(text)

        try:
            # Detect language to choose appropriate model
            from backend.app.services.preprocessing_service import TextPreprocessor
            preprocessor = TextPreprocessor()
            lang, _ = preprocessor._detect_language(text)

            # Select model based on language
            if lang in ['hi', 'bn', 'te', 'mr', 'ta', 'ur', 'gu', 'pa', 'or', 'as'] and SummarizationMethod.INDICBART in self.transformer_summarizers:
                method = SummarizationMethod.INDICBART
            elif lang != 'en' and SummarizationMethod.MT5 in self.transformer_summarizers:
                method = SummarizationMethod.MT5
            else:
                method = SummarizationMethod.T5

            if method not in self.transformer_summarizers:
                raise ValueError(f"Required summarization method {method} not available")

            summarizer = self.transformer_summarizers[method]

            # Prepare text with topic guidance if provided
            if topics:
                topic_prefix = f"Summarize the following text focusing on these topics: {', '.join(topics)}. "
                text = topic_prefix + text

            # Truncate text if too long
            max_input_length = 512
            if len(text.split()) > max_input_length:
                text = ' '.join(text.split()[:max_input_length])

            # Generate summary
            result = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)

            summary_text = result[0]['summary_text']
            summary_length = len(summary_text)
            compression_ratio = summary_length / original_length if original_length > 0 else 0

            processing_time = int((time.time() - start_time) * 1000)

            return SummaryResult(
                method=f"{method.value}_topic_based",
                summary_type=SummarizationType.ABSTRACTIVE,
                summary_text=summary_text,
                original_length=original_length,
                summary_length=summary_length,
                compression_ratio=compression_ratio,
                key_sentences=[summary_text],
                confidence_score=0.85,
                processing_time_ms=processing_time,
                metadata={
                    "max_length": max_length,
                    "min_length": min_length,
                    "language": lang,
                    "model_used": method.value,
                    "topics_focused": topics or [],
                    "input_truncated": len(text.split()) > max_input_length
                }
            )

        except Exception as e:
            print(f"Error in topic-based summarization: {e}")
            # Fallback to extractive summarization
            fallback_result = await self.extractive_summarization(text, SummarizationMethod.CUSTOM_TEXTRANK, 2)
            fallback_result.method = "topic_based_fallback"
            fallback_result.metadata["fallback_used"] = True
            fallback_result.metadata["original_error"] = str(e)

            return fallback_result

    async def abstractive_summarization(self, text: str,
                                      method: SummarizationMethod = SummarizationMethod.T5,
                                      max_length: int = 150,
                                      min_length: int = 30) -> SummaryResult:
        """
        Perform abstractive summarization using transformer models.
        
        Args:
            text: Text to summarize
            method: Summarization method to use
            max_length: Maximum length of summary
            min_length: Minimum length of summary
            
        Returns:
            SummaryResult: Summarization result
        """
        import time
        start_time = time.time()
        
        original_length = len(text)
        
        try:
            if not TRANSFORMERS_AVAILABLE or method not in self.transformer_summarizers:
                raise ValueError(f"Abstractive method {method} not available")
            
            summarizer = self.transformer_summarizers[method]
            
            # Truncate text if too long (transformer models have input limits)
            max_input_length = 512  # Adjust based on model
            if len(text.split()) > max_input_length:
                text = ' '.join(text.split()[:max_input_length])
            
            # Generate summary
            # Simple memoization cache key
            cache_key = (method.value, text[:1024], max_length, min_length)
            result = await self._cached_summarize(summarizer, cache_key, text, max_length, min_length, method)
            
            summary_text = result[0]['summary_text']
            summary_length = len(summary_text)
            compression_ratio = summary_length / original_length if original_length > 0 else 0
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return SummaryResult(
                method=method.value,
                summary_type=SummarizationType.ABSTRACTIVE,
                summary_text=summary_text,
                original_length=original_length,
                summary_length=summary_length,
                compression_ratio=compression_ratio,
                key_sentences=[summary_text],  # Abstractive generates new sentences
                confidence_score=0.85,  # Default confidence for transformer models
                processing_time_ms=processing_time,
                metadata={
                    "max_length": max_length,
                    "min_length": min_length,
                    "input_truncated": len(text.split()) > max_input_length
                }
            )
            
        except Exception as e:
            print(f"Error in abstractive summarization: {e}")
            # Fallback to extractive summarization
            fallback_result = await self.extractive_summarization(text, SummarizationMethod.CUSTOM_TEXTRANK, 2)
            fallback_result.method = f"{method.value}_fallback"
            fallback_result.metadata["fallback_used"] = True
            fallback_result.metadata["original_error"] = str(e)
            
            return fallback_result

    async def summarize_long_document(self, text: str,
                                      preferred_max_length: int = 160,
                                      chunk_words: int = 350,
                                      overlap_words: int = 50) -> SummaryResult:
        """
        Hierarchical summarization for long documents with preference for Indian models.
        1) Detect language -> choose IndicBART (Indian), else mT5, else T5.
        2) Split into chunks -> summarize each -> synthesize final summary.
        """
        import time
        start_time = time.time()

        original_length = len(text)
        if original_length < 800:
            # For short docs, do a single-pass abstractive (Indic-first)
            method = await self._select_indic_first_model(text)
            return await self.abstractive_summarization(text, method=method, max_length=preferred_max_length, min_length=40)

        # Split into word chunks
        words = text.split()
        if not words:
            return await self.extractive_summarization(text, SummarizationMethod.CUSTOM_TEXTRANK, 3)

        chunks = []
        i = 0
        while i < len(words):
            chunk = words[i:i+chunk_words]
            chunks.append(' '.join(chunk))
            i += chunk_words - overlap_words

        # Summarize each chunk using Indic-first model
        method = await self._select_indic_first_model(text)
        partial_summaries: List[str] = []
        for ch in chunks:
            res = await self.abstractive_summarization(ch, method=method, max_length=preferred_max_length, min_length=40)
            partial_summaries.append(res.summary_text)

        # Synthesize final summary from partials
        synthesis_input = '\n'.join(partial_summaries)
        # If abstractive fails for final step, fallback to extractive over synthesis_input
        try:
            final_method = await self._select_indic_first_model(synthesis_input)
            final_res = await self.abstractive_summarization(synthesis_input, method=final_method, max_length=preferred_max_length, min_length=50)
        except Exception:
            final_res = await self.extractive_summarization(synthesis_input, SummarizationMethod.CUSTOM_TEXTRANK, 3)

        final_res.metadata.update({
            "hierarchical": True,
            "chunks": len(chunks),
            "chunk_words": chunk_words,
            "overlap_words": overlap_words,
            "partial_summaries_count": len(partial_summaries)
        })
        final_res.processing_time_ms = int((time.time() - start_time) * 1000)
        return final_res

    async def _select_indic_first_model(self, sample_text: str) -> SummarizationMethod:
        """Choose the best summarization model with an Indian-first preference."""
        try:
            from backend.app.services.preprocessing_service import TextPreprocessor
            pre = TextPreprocessor()
            lang, _ = pre._detect_language(sample_text)
        except Exception:
            lang = 'en'

        # Prefer IndicBART for key Indian languages
        if self.prefer_indian_models and lang in ['hi','bn','te','ta','mr','gu','kn','ml','pa','or','as','ur','ne'] and SummarizationMethod.INDICBART in self.transformer_summarizers:
            return SummarizationMethod.INDICBART
        # Else prefer mT5 for multilingual
        if SummarizationMethod.MT5 in self.transformer_summarizers:
            return SummarizationMethod.MT5
        # Fallback
        return SummarizationMethod.T5

    async def _cached_summarize(self, summarizer, cache_key: tuple, text: str, max_length: int, min_length: int, method: SummarizationMethod):
        """Lightweight async wrapper over a small in-memory cache for transformer summaries."""
        @lru_cache(maxsize=256)
        def _run_cached(key: tuple):
            m, t, max_l, min_l = key
            if method == SummarizationMethod.T5:
                input_text = f"summarize: {t}"
                return summarizer(input_text, max_length=max_l, min_length=min_l, do_sample=False)
            else:
                return summarizer(t, max_length=max_l, min_length=min_l, do_sample=False)

        # Execute in current thread (pipeline is I/O/CPU bound; acceptable for now)
        return _run_cached(cache_key)
    
    async def hybrid_summarization(self, text: str,
                                 extractive_sentences: int = 5,
                                 abstractive_max_length: int = 100) -> SummaryResult:
        """
        Perform hybrid summarization combining extractive and abstractive methods.
        
        Args:
            text: Text to summarize
            extractive_sentences: Number of sentences for extractive step
            abstractive_max_length: Max length for abstractive step
            
        Returns:
            SummaryResult: Combined summarization result
        """
        import time
        start_time = time.time()
        
        try:
            # Step 1: Extractive summarization to identify key content
            extractive_result = await self.extractive_summarization(
                text, SummarizationMethod.CUSTOM_TEXTRANK, extractive_sentences
            )
            
            # Step 2: Abstractive summarization on extracted content
            abstractive_result = await self.abstractive_summarization(
                extractive_result.summary_text, SummarizationMethod.T5, abstractive_max_length
            )
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return SummaryResult(
                method="hybrid_textrank_t5",
                summary_type=SummarizationType.HYBRID,
                summary_text=abstractive_result.summary_text,
                original_length=len(text),
                summary_length=len(abstractive_result.summary_text),
                compression_ratio=len(abstractive_result.summary_text) / len(text),
                key_sentences=extractive_result.key_sentences,
                confidence_score=(extractive_result.confidence_score + abstractive_result.confidence_score) / 2,
                processing_time_ms=processing_time,
                metadata={
                    "extractive_step": extractive_result.metadata,
                    "abstractive_step": abstractive_result.metadata,
                    "two_stage_process": True
                }
            )
            
        except Exception as e:
            print(f"Error in hybrid summarization: {e}")
            # Fallback to extractive only
            fallback_result = await self.extractive_summarization(text, SummarizationMethod.CUSTOM_TEXTRANK, 3)
            fallback_result.summary_type = SummarizationType.HYBRID
            fallback_result.method = "hybrid_fallback"
            fallback_result.metadata["fallback_used"] = True
            fallback_result.metadata["error"] = str(e)
            
            return fallback_result
    
    async def summarize_comments(self, comments: List[str], 
                               method: SummarizationMethod = SummarizationMethod.CUSTOM_TEXTRANK,
                               summary_type: SummarizationType = SummarizationType.EXTRACTIVE) -> SummaryResult:
        """
        Summarize multiple comments into a single summary.
        
        Args:
            comments: List of comment texts
            method: Summarization method
            summary_type: Type of summarization
            
        Returns:
            SummaryResult: Summary of all comments
        """
        # Combine all comments
        combined_text = ' '.join(comments)
        
        # Choose summarization approach based on type
        if summary_type == SummarizationType.EXTRACTIVE:
            result = await self.extractive_summarization(combined_text, method)
        elif summary_type == SummarizationType.ABSTRACTIVE:
            result = await self.abstractive_summarization(combined_text, method)
        else:  # HYBRID
            result = await self.hybrid_summarization(combined_text)
        
        # Update metadata to reflect multi-comment source
        result.metadata.update({
            "source_type": "multiple_comments",
            "comment_count": len(comments),
            "average_comment_length": len(combined_text) / len(comments) if comments else 0
        })
        
        return result
    
    async def policy_summarization(self, text: str, 
                                 stakeholder_type: Optional[str] = None,
                                 focus_areas: Optional[List[str]] = None) -> SummaryResult:
        """
        Policy-specific summarization for legislative/policy comments.
        
        Args:
            text: Text to summarize
            stakeholder_type: Type of stakeholder (individual, business, etc.)
            focus_areas: Specific areas to focus on in summary
            
        Returns:
            SummaryResult: Policy-enhanced summary
        """
        import time
        start_time = time.time()
        
        try:
            # Detect stakeholder type if not provided
            if not stakeholder_type:
                stakeholder_type = self._detect_stakeholder_type(text)
            
            # Extract policy-specific elements
            key_actions = self._extract_policy_elements(text, 'key_actions')
            legislative_terms = self._extract_policy_elements(text, 'legislative_terms')
            concerns = self._extract_policy_elements(text, 'stakeholder_concerns')
            sentiment_indicators = self._extract_policy_elements(text, 'sentiment_indicators')
            
            # Weight sentences based on policy relevance
            sentences = text.split('.')
            scored_sentences = []
            
            for i, sentence in enumerate(sentences):
                if len(sentence.strip()) < 10:
                    continue
                    
                score = self._calculate_policy_sentence_score(sentence, stakeholder_type)
                scored_sentences.append((sentence.strip(), score, i))
            
            # Sort by score and select top sentences
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            top_sentences = scored_sentences[:min(5, len(scored_sentences))]
            
            # Create structured summary
            summary_parts = []
            
            # Add stakeholder identification
            if stakeholder_type != 'unknown':
                summary_parts.append(f"Stakeholder Type: {stakeholder_type.title()}")
            
            # Add key sentiment
            if sentiment_indicators:
                dominant_sentiment = max(sentiment_indicators, key=sentiment_indicators.count)
                summary_parts.append(f"Overall Stance: {dominant_sentiment}")
            
            # Add main concerns/actions
            if key_actions:
                summary_parts.append(f"Proposed Actions: {', '.join(key_actions[:3])}")
            
            if concerns:
                summary_parts.append(f"Key Concerns: {', '.join(concerns[:3])}")
            
            # Add key sentences
            summary_sentences = [sent[0] for sent in top_sentences]
            summary_parts.extend(summary_sentences)
            
            summary_text = '. '.join(summary_parts) + '.'
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return SummaryResult(
                method="policy_enhanced",
                summary_type=SummarizationType.EXTRACTIVE,
                summary_text=summary_text,
                original_length=len(text),
                summary_length=len(summary_text),
                compression_ratio=len(summary_text) / len(text) if len(text) > 0 else 0,
                key_sentences=summary_sentences,
                confidence_score=0.85,  # High confidence for structured approach
                processing_time_ms=processing_time,
                metadata={
                    "stakeholder_type": stakeholder_type,
                    "key_actions": key_actions,
                    "legislative_terms": legislative_terms,
                    "stakeholder_concerns": concerns,
                    "sentiment_indicators": sentiment_indicators,
                    "structured_summary": True
                }
            )
            
        except Exception as e:
            print(f"Error in policy summarization: {e}")
            # Fallback to regular extractive summarization
            return await self.extractive_summarization(text, SummarizationMethod.CUSTOM_TEXTRANK)
    
    def _detect_stakeholder_type(self, text: str) -> str:
        """Detect stakeholder type from text content."""
        text_lower = text.lower()
        scores = {}
        
        for stakeholder_type, pattern in self.stakeholder_patterns.items():
            import re
            matches = len(re.findall(pattern, text_lower))
            scores[stakeholder_type] = matches
        
        if scores:
            detected_type = max(scores, key=scores.get)
            if scores[detected_type] > 0:
                return detected_type
        
        return "unknown"
    
    def _extract_policy_elements(self, text: str, category: str) -> List[str]:
        """Extract policy-specific elements from text."""
        text_lower = text.lower()
        keywords = self.policy_keywords.get(category, [])
        found_elements = []
        
        for keyword in keywords:
            if keyword in text_lower:
                found_elements.append(keyword)
        
        return found_elements
    
    def _calculate_policy_sentence_score(self, sentence: str, stakeholder_type: str) -> float:
        """Calculate relevance score for a sentence in policy context."""
        sentence_lower = sentence.lower()
        score = 0.0
        
        # Base score for sentence length (prefer moderate length)
        word_count = len(sentence.split())
        if 5 <= word_count <= 30:
            score += 0.2
        
        # Score for policy keywords
        for category, keywords in self.policy_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in sentence_lower)
            if category == 'key_actions':
                score += matches * 0.3
            elif category == 'legislative_terms':
                score += matches * 0.25
            elif category == 'stakeholder_concerns':
                score += matches * 0.2
            elif category == 'sentiment_indicators':
                score += matches * 0.15
        
        # Bonus for stakeholder-specific language
        if stakeholder_type != 'unknown':
            pattern = self.stakeholder_patterns.get(stakeholder_type, '')
            if pattern:
                import re
                if re.search(pattern, sentence_lower):
                    score += 0.1
        
        return score
    
    async def aggregate_summarization(self, 
                                    comments_by_section: Dict[str, List[str]],
                                    sentiments_by_section: Optional[Dict[str, List[str]]] = None) -> AggregateSummaryResult:
        """
        Create aggregate summaries per law section and overall summary.
        
        Args:
            comments_by_section: Comments organized by law section
            sentiments_by_section: Sentiment labels by section (optional)
            
        Returns:
            AggregateSummaryResult: Comprehensive aggregate summary
        """
        import time
        start_time = time.time()
        
        section_summaries = {}
        all_comments = []
        sentiment_counts = defaultdict(int)
        
        # Summarize each section
        for section, comments in comments_by_section.items():
            if comments:
                section_summary = await self.summarize_comments(
                    comments, SummarizationMethod.CUSTOM_TEXTRANK, SummarizationType.EXTRACTIVE
                )
                section_summaries[section] = section_summary
                all_comments.extend(comments)
        
        # Create overall summary
        if all_comments:
            overall_summary = await self.summarize_comments(
                all_comments, SummarizationMethod.CUSTOM_TEXTRANK, SummarizationType.HYBRID
            )
        else:
            overall_summary = SummaryResult(
                method="none", summary_type=SummarizationType.EXTRACTIVE,
                summary_text="No comments available for summarization.",
                original_length=0, summary_length=0, compression_ratio=0,
                key_sentences=[], confidence_score=0, processing_time_ms=0,
                metadata={}
            )
        
        # Count sentiments if provided
        if sentiments_by_section:
            for section, sentiments in sentiments_by_section.items():
                for sentiment in sentiments:
                    sentiment_counts[sentiment] += 1
        
        # Extract key themes using keyword frequency
        key_themes = self._extract_key_themes(all_comments)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return AggregateSummaryResult(
            section_summaries=section_summaries,
            overall_summary=overall_summary,
            key_themes=key_themes,
            sentiment_distribution=dict(sentiment_counts),
            total_comments=len(all_comments),
            processing_statistics={
                "total_processing_time_ms": processing_time,
                "sections_processed": len(section_summaries),
                "average_section_comments": sum(len(comments) for comments in comments_by_section.values()) / len(comments_by_section) if comments_by_section else 0
            }
        )
    
    def _extract_key_themes(self, comments: List[str], max_themes: int = 10) -> List[str]:
        """Extract key themes from comments using keyword frequency analysis."""
        if not comments:
            return []
        
        # Combine all comments
        combined_text = ' '.join(comments).lower()
        
        # Remove common words and extract meaningful terms
        if self.nlp:
            doc = self.nlp(combined_text)
            # Extract noun phrases and named entities
            themes = []
            for chunk in doc.noun_chunks:
                if len(chunk.text.strip()) > 3 and chunk.text.strip() not in ['the', 'and', 'for', 'with']:
                    themes.append(chunk.text.strip())
            
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'PERSON', 'GPE', 'LAW'] and len(ent.text.strip()) > 2:
                    themes.append(ent.text.strip())
        else:
            # Fallback: simple word frequency
            words = re.findall(r'\b\w{4,}\b', combined_text)
            theme_counts = Counter(words)
            themes = [word for word, count in theme_counts.most_common(max_themes * 2)]
        
        # Return most common themes
        theme_counts = Counter(themes)
        return [theme for theme, count in theme_counts.most_common(max_themes)]
    
    async def batch_summarization(self, texts: List[str], 
                                method: SummarizationMethod = SummarizationMethod.CUSTOM_TEXTRANK,
                                summary_type: SummarizationType = SummarizationType.EXTRACTIVE) -> List[SummaryResult]:
        """
        Perform batch summarization on multiple texts.
        
        Args:
            texts: List of texts to summarize
            method: Summarization method
            summary_type: Type of summarization
            
        Returns:
            list: List of summarization results
        """
        # Create tasks for all texts
        tasks = []
        for text in texts:
            if summary_type == SummarizationType.EXTRACTIVE:
                task = asyncio.create_task(self.extractive_summarization(text, method))
            elif summary_type == SummarizationType.ABSTRACTIVE:
                task = asyncio.create_task(self.abstractive_summarization(text, method))
            else:  # HYBRID
                task = asyncio.create_task(self.hybrid_summarization(text))
            
            tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions and return results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Error summarizing text {i}: {result}")
                # Create minimal error result
                final_results.append(None)
            else:
                final_results.append(result)
        
        return [r for r in final_results if r is not None]