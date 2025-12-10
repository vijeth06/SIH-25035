"""
Text preprocessing service for cleaning, tokenization, and language detection.
Supports English and Hindi text processing with comprehensive cleaning pipeline.
"""

import re
try:
    import spacy
    _SPACY_PREPROC_AVAILABLE = True
except Exception:
    spacy = None  # type: ignore
    _SPACY_PREPROC_AVAILABLE = False
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from textblob import TextBlob
import contractions
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import unicodedata
import html
from concurrent.futures import ThreadPoolExecutor
import asyncio

from backend.app.core.config import settings
from backend.app.utils.text_utils import TextCleaner


@dataclass
class PreprocessingResult:
    """Result of text preprocessing."""
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


class TextPreprocessor:
    """Comprehensive text preprocessing service for comment analysis."""
    
    def __init__(self):
        self.text_cleaner = TextCleaner()
        self._initialize_nlp_models()
        self._initialize_stopwords()
        
    def _initialize_nlp_models(self):
        """Initialize spaCy models for different languages."""
        try:
            # Load English model
            self.nlp_en = spacy.load("en_core_web_sm") if _SPACY_PREPROC_AVAILABLE else None
            
            # Configure pipeline - disable unnecessary components for performance
            if self.nlp_en:
                try:
                    # Try to disable components we don't need
                    self.nlp_en.disable_pipes(["parser", "ner"])
                except Exception:
                    # If some components don't exist or nlp is None, continue
                    pass
            
            print("✅ English NLP model loaded successfully")
        except IOError:
            print("❌ English spaCy model not found. Please run: python -m spacy download en_core_web_sm")
            self.nlp_en = None
        
        # For Hindi, we'll use basic processing since advanced models may not be available
        self.nlp_hi = None  # Placeholder for Hindi model
        
    def _initialize_stopwords(self):
        """Initialize stopwords for supported languages."""
        try:
            # Download NLTK data if not present
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            
            # Load English stopwords
            self.stopwords_en = set(stopwords.words('english'))
            
            # Add custom stopwords for consultation context
            custom_stopwords = {
                'would', 'could', 'should', 'may', 'might', 'will', 'shall',
                'government', 'policy', 'act', 'section', 'clause', 'bill',
                'proposed', 'draft', 'consultation', 'feedback', 'comment'
            }
            self.stopwords_en.update(custom_stopwords)
            
            # Basic Hindi stopwords (can be expanded)
            self.stopwords_hi = {
                'और', 'का', 'के', 'की', 'को', 'में', 'से', 'पर', 'है', 'हैं', 'था', 'थे', 'थी',
                'होगा', 'होगी', 'होंगे', 'यह', 'वह', 'इस', 'उस', 'एक', 'दो', 'तीन', 'अपना',
                'सरकार', 'नीति', 'कानून', 'धारा', 'बिल', 'प्रस्तावित', 'मसौदा'
            }
            
            print("✅ Stopwords initialized for English and Hindi")
            
        except Exception as e:
            print(f"❌ Error initializing NLTK data: {e}")
            self.stopwords_en = set()
            self.stopwords_hi = set()
    
    async def preprocess_text(self, text: str, 
                            enable_spell_correction: bool = False,
                            remove_stopwords: bool = True,
                            lemmatize: bool = True) -> PreprocessingResult:
        """
        Comprehensive text preprocessing pipeline.
        
        Args:
            text: Input text to preprocess
            enable_spell_correction: Whether to apply spell correction
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to apply lemmatization
            
        Returns:
            PreprocessingResult: Comprehensive preprocessing results
        """
        processing_notes = {}
        
        # Store original text
        original_text = text
        processing_notes['original_length'] = len(text)
        
        # Step 1: Language detection
        language, lang_confidence = self._detect_language(text)
        processing_notes['language_detection'] = {
            'detected': language,
            'confidence': lang_confidence
        }
        
        # Step 2: Initial cleaning
        cleaned_text = await self._clean_text(text)
        processing_notes['cleaning_applied'] = True
        
        # Step 3: Normalization
        normalized_text = self._normalize_text(cleaned_text)
        processing_notes['normalization_applied'] = True
        
        # Step 4: Spell correction (if enabled)
        if enable_spell_correction and language == 'en':
            corrected_text = self._correct_spelling(normalized_text)
            processing_notes['spell_correction'] = {
                'applied': True,
                'changes_made': corrected_text != normalized_text
            }
            normalized_text = corrected_text
        
        # Step 5: Tokenization and advanced processing
        if language == 'en' and self.nlp_en:
            tokens, sentences, processed_text = await self._process_english_text(
                normalized_text, remove_stopwords, lemmatize
            )
        elif language == 'hi':
            tokens, sentences, processed_text = await self._process_hindi_text(
                normalized_text, remove_stopwords
            )
        else:
            # Fallback to basic processing
            tokens, sentences, processed_text = await self._process_basic_text(
                normalized_text, remove_stopwords
            )
        
        processing_notes['advanced_processing'] = {
            'model_used': f"{language}_model" if language in ['en', 'hi'] else 'basic',
            'stopwords_removed': remove_stopwords,
            'lemmatization': lemmatize and language == 'en'
        }
        
        # Calculate final statistics
        word_count = len(tokens)
        character_count = len(processed_text)
        
        processing_notes['final_stats'] = {
            'word_count': word_count,
            'character_count': character_count,
            'compression_ratio': character_count / len(original_text) if len(original_text) > 0 else 0
        }
        
        return PreprocessingResult(
            original_text=original_text,
            cleaned_text=cleaned_text,
            processed_text=processed_text,
            language=language,
            language_confidence=lang_confidence,
            tokens=tokens,
            sentences=sentences,
            word_count=word_count,
            character_count=character_count,
            processing_notes=processing_notes
        )
    
    def _detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect the language of the text.
        
        Args:
            text: Text to analyze
            
        Returns:
            tuple: (language_code, confidence_score)
        """
        try:
            # Clean text for language detection
            clean_text = re.sub(r'[^\w\s]', ' ', text)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            
            if len(clean_text) < 10:
                return settings.DEFAULT_LANGUAGE, 0.5
            
            detected_lang = detect(clean_text)
            
            # Map to supported languages
            if detected_lang in settings.SUPPORTED_LANGUAGES:
                return detected_lang, 0.95
            elif detected_lang == 'hi':
                return 'hi', 0.90
            else:
                # Default to English for unsupported languages
                return 'en', 0.6
                
        except LangDetectException:
            return settings.DEFAULT_LANGUAGE, 0.3
        except Exception:
            return settings.DEFAULT_LANGUAGE, 0.1
    
    async def _clean_text(self, text: str) -> str:
        """
        Initial text cleaning pipeline.
        
        Args:
            text: Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Use the TextCleaner utility
        cleaned = self.text_cleaner.clean_text(
            text=text,
            remove_html=True,
            remove_urls=True,
            remove_emails=False  # Keep emails for now, mask later if needed
        )
        
        # Additional cleaning steps
        # Remove excessive punctuation
        cleaned = re.sub(r'[!]{2,}', '!', cleaned)
        cleaned = re.sub(r'[?]{2,}', '?', cleaned)
        cleaned = re.sub(r'[.]{3,}', '...', cleaned)
        
        # Remove excessive capitalization (more than 3 consecutive caps)
        cleaned = re.sub(r'[A-Z]{4,}', lambda m: m.group().capitalize(), cleaned)
        
        # Clean up quotation marks
        cleaned = re.sub(r'["""]', '"', cleaned)
        cleaned = re.sub(r"[''']", "'", cleaned)
        
        # Remove zero-width characters
        cleaned = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', cleaned)
        
        return cleaned.strip()
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text encoding and characters.
        
        Args:
            text: Text to normalize
            
        Returns:
            str: Normalized text
        """
        if not text:
            return ""
        
        # HTML decode
        text = html.unescape(text)
        
        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)
        
        # Expand contractions (English only)
        if self._detect_language(text)[0] == 'en':
            try:
                text = contractions.fix(text)
            except:
                pass  # Continue if contractions expansion fails
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\t+', ' ', text)
        
        return text.strip()
    
    def _correct_spelling(self, text: str) -> str:
        """
        Apply spell correction using TextBlob.
        
        Args:
            text: Text to correct
            
        Returns:
            str: Spell-corrected text
        """
        try:
            blob = TextBlob(text)
            corrected = blob.correct()
            return str(corrected)
        except Exception:
            # If spell correction fails, return original text
            return text
    
    async def _process_english_text(self, text: str, remove_stopwords: bool = True, 
                                  lemmatize: bool = True) -> Tuple[List[str], List[str], str]:
        """
        Process English text using spaCy.
        
        Args:
            text: Text to process
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to apply lemmatization
            
        Returns:
            tuple: (tokens, sentences, processed_text)
        """
        if not self.nlp_en or not text:
            return await self._process_basic_text(text, remove_stopwords)
        
        # Process with spaCy
        doc = self.nlp_en(text)
        
        # Extract sentences
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        # Extract and process tokens
        tokens = []
        processed_words = []
        
        for token in doc:
            # Skip punctuation, spaces, and very short tokens
            if token.is_punct or token.is_space or len(token.text.strip()) < 2:
                continue
            
            # Get base form
            word = token.lemma_ if lemmatize else token.text
            word = word.lower().strip()
            
            # Skip stopwords if requested
            if remove_stopwords and word in self.stopwords_en:
                continue
            
            # Skip if not alphabetic (numbers, special chars)
            if not word.isalpha():
                continue
            
            tokens.append(word)
            processed_words.append(word)
        
        processed_text = ' '.join(processed_words)
        
        return tokens, sentences, processed_text
    
    async def _process_hindi_text(self, text: str, remove_stopwords: bool = True) -> Tuple[List[str], List[str], str]:
        """
        Process Hindi text with basic tokenization.
        
        Args:
            text: Hindi text to process
            remove_stopwords: Whether to remove stopwords
            
        Returns:
            tuple: (tokens, sentences, processed_text)
        """
        if not text:
            return [], [], ""
        
        # Basic sentence splitting for Hindi
        sentences = sent_tokenize(text)
        
        # Basic word tokenization
        words = word_tokenize(text)
        
        tokens = []
        processed_words = []
        
        for word in words:
            word = word.strip().lower()
            
            # Skip punctuation and short words
            if len(word) < 2 or not word.isalpha():
                continue
            
            # Skip Hindi stopwords if requested
            if remove_stopwords and word in self.stopwords_hi:
                continue
            
            tokens.append(word)
            processed_words.append(word)
        
        processed_text = ' '.join(processed_words)
        
        return tokens, sentences, processed_text
    
    async def _process_basic_text(self, text: str, remove_stopwords: bool = True) -> Tuple[List[str], List[str], str]:
        """
        Basic text processing fallback for unsupported languages.
        
        Args:
            text: Text to process
            remove_stopwords: Whether to remove stopwords
            
        Returns:
            tuple: (tokens, sentences, processed_text)
        """
        if not text:
            return [], [], ""
        
        # Basic sentence splitting
        sentences = sent_tokenize(text)
        
        # Basic word tokenization
        words = word_tokenize(text)
        
        tokens = []
        processed_words = []
        
        for word in words:
            word = word.strip().lower()
            
            # Skip punctuation and short words
            if len(word) < 2 or not re.match(r'^[a-zA-Z\u0900-\u097F]+$', word):
                continue
            
            # Basic stopword removal (English only)
            if remove_stopwords and word in self.stopwords_en:
                continue
            
            tokens.append(word)
            processed_words.append(word)
        
        processed_text = ' '.join(processed_words)
        
        return tokens, sentences, processed_text
    
    async def batch_preprocess(self, texts: List[str], 
                             max_workers: int = 4, **kwargs) -> List[PreprocessingResult]:
        """
        Process multiple texts in parallel.
        
        Args:
            texts: List of texts to process
            max_workers: Maximum number of worker threads
            **kwargs: Additional arguments for preprocess_text
            
        Returns:
            list: List of PreprocessingResult objects
        """
        loop = asyncio.get_event_loop()
        
        # Create tasks for all texts
        tasks = []
        for text in texts:
            task = loop.create_task(self.preprocess_text(text, **kwargs))
            tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create a minimal result for failed processing
                processed_results.append(PreprocessingResult(
                    original_text=texts[i],
                    cleaned_text=texts[i],
                    processed_text=texts[i],
                    language='unknown',
                    language_confidence=0.0,
                    tokens=[],
                    sentences=[],
                    word_count=0,
                    character_count=len(texts[i]),
                    processing_notes={'error': str(result)}
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    def extract_key_phrases(self, text: str, max_phrases: int = 10) -> List[Dict[str, Any]]:
        """
        Extract key phrases from text using spaCy.
        
        Args:
            text: Input text
            max_phrases: Maximum number of phrases to extract
            
        Returns:
            list: List of key phrases with scores
        """
        if not self.nlp_en or not text:
            return []
        
        doc = self.nlp_en(text)
        
        # Extract noun phrases and named entities
        phrases = []
        
        # Noun phrases
        for chunk in doc.noun_chunks:
            if len(chunk.text.strip()) > 2:
                phrases.append({
                    'text': chunk.text.strip().lower(),
                    'type': 'noun_phrase',
                    'score': len(chunk.text.split())  # Simple scoring by length
                })
        
        # Named entities
        for ent in doc.ents:
            if len(ent.text.strip()) > 2:
                phrases.append({
                    'text': ent.text.strip().lower(),
                    'type': f'entity_{ent.label_.lower()}',
                    'score': len(ent.text.split()) * 2  # Higher score for entities
                })
        
        # Remove duplicates and sort by score
        unique_phrases = {}
        for phrase in phrases:
            text = phrase['text']
            if text not in unique_phrases or unique_phrases[text]['score'] < phrase['score']:
                unique_phrases[text] = phrase
        
        # Sort by score and return top phrases
        sorted_phrases = sorted(unique_phrases.values(), key=lambda x: x['score'], reverse=True)
        
        return sorted_phrases[:max_phrases]
    
    def get_text_statistics(self, text: str) -> Dict[str, Any]:
        """
        Get comprehensive text statistics.
        
        Args:
            text: Input text
            
        Returns:
            dict: Text statistics
        """
        stats = {
            'character_count': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(sent_tokenize(text)),
            'paragraph_count': len([p for p in text.split('\n\n') if p.strip()]),
            'average_word_length': 0,
            'average_sentence_length': 0,
            'language': 'unknown',
            'contains_urls': bool(re.search(r'http[s]?://\S+', text)),
            'contains_emails': bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)),
            'contains_numbers': bool(re.search(r'\d', text)),
            'uppercase_ratio': 0,
            'punctuation_ratio': 0
        }
        
        words = text.split()
        if words:
            stats['average_word_length'] = sum(len(word) for word in words) / len(words)
        
        sentences = sent_tokenize(text)
        if sentences:
            stats['average_sentence_length'] = sum(len(sent.split()) for sent in sentences) / len(sentences)
        
        if text:
            stats['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text)
            stats['punctuation_ratio'] = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
        
        # Language detection
        language, confidence = self._detect_language(text)
        stats['language'] = language
        stats['language_confidence'] = confidence
        
        return stats