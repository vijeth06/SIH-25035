"""
Advanced Multilingual Sentiment Analysis System
Supports all Indian languages, mixed languages, and provides highly accurate sentiment analysis
using state-of-the-art transformer models and ensemble methods.
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
import logging
from datetime import datetime

# Core ML libraries
try:
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        pipeline, AutoModel, MT5ForConditionalGeneration, MT5Tokenizer
    )
    from sentence_transformers import SentenceTransformer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Language detection
try:
    from langdetect import detect, detect_langs
    from polyglot.detect import Detector
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

# Text processing
try:
    import nltk
    from textblob import TextBlob
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import spacy
    TEXTPROCESSING_AVAILABLE = True
except ImportError:
    TEXTPROCESSING_AVAILABLE = False

# Indian language support
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    INDIC_SUPPORT = True
except ImportError:
    INDIC_SUPPORT = False

class AdvancedMultilingualSentimentAnalyzer:
    """
    Advanced sentiment analyzer supporting all Indian languages and mixed text
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models_loaded = False
        self.supported_languages = {
            'en': 'English',
            'hi': 'Hindi',
            'bn': 'Bengali', 
            'te': 'Telugu',
            'ta': 'Tamil',
            'mr': 'Marathi',
            'gu': 'Gujarati',
            'kn': 'Kannada',
            'ml': 'Malayalam',
            'pa': 'Punjabi',
            'or': 'Odia',
            'as': 'Assamese',
            'ur': 'Urdu',
            'ne': 'Nepali',
            'si': 'Sinhala',
            'my': 'Myanmar',
            'mixed': 'Mixed Languages'
        }
        
        # Initialize models
        self._initialize_models()
        
        # Enhanced keyword dictionaries for Indian languages
        self._initialize_multilingual_keywords()
        
        # Advanced sentiment patterns
        self._initialize_advanced_patterns()
    
    def _initialize_models(self):
        """Initialize all required models"""
        try:
            if TRANSFORMERS_AVAILABLE:
                # Primary multilingual sentiment model
                self.multilingual_sentiment_model = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual",
                    tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
                )
                
                # Prefer Indian model for Indian languages
                try:
                    self.indic_sentiment_model = pipeline(
                        "sentiment-analysis",
                        model="ai4bharat/indic-bert-v1-sentiment",
                        tokenizer="ai4bharat/indic-bert-v1-sentiment"
                    )
                except:
                    self.indic_sentiment_model = None
                # Expose a flag to prefer Indian models when any Indian script is detected
                self.prefer_indian_models = True
                    
                # mT5 for summarization
                self.mt5_tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
                self.mt5_model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
                
                # Sentence embeddings for semantic analysis
                self.sentence_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
                
                self.models_loaded = True
                self.logger.info("✅ Advanced multilingual models loaded successfully")
                
            else:
                self.logger.warning("⚠️ Transformers not available, using fallback methods")
                
            # VADER for English sentiment
            if TEXTPROCESSING_AVAILABLE:
                self.vader_analyzer = SentimentIntensityAnalyzer()
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            self.models_loaded = False
    
    def _initialize_multilingual_keywords(self):
        """Initialize multilingual sentiment keywords"""
        self.multilingual_keywords = {
            'positive': {
                'en': ['excellent', 'great', 'good', 'amazing', 'wonderful', 'fantastic', 'outstanding', 'superb', 'brilliant', 'impressive', 'support', 'approve', 'love', 'like', 'appreciate', 'welcome', 'benefit', 'improve', 'enhance', 'effective'],
                'hi': ['उत्कृष्ट', 'महान', 'अच्छा', 'बेहतरीन', 'शानदार', 'समर्थन', 'स्वीकार', 'पसंद', 'लाभ', 'सुधार', 'प्रभावी', 'खुशी', 'संतुष्ट'],
                'bn': ['চমৎকার', 'দুর্দান্ত', 'ভালো', 'অসাধারণ', 'চমৎকার', 'সমর্থন', 'অনুমোদন', 'পছন্দ', 'উপকার', 'উন্নতি', 'কার্যকর'],
                'te': ['అద্భుతమైన', 'గొప్ప', 'మంచి', 'అద్భుతమైన', 'మద్దతు', 'ఆమోదం', 'ఇష్టం', 'ప్రయోజనం', 'మెరుగుదల', 'ప్రభावకరమైన'],
                'ta': ['சிறந்த', 'பெரிய', 'நல்ல', 'அற்புதமான', 'ஆதரவு', 'ஒப்புதல்', 'விருப்பம்', 'நன்மை', 'மேம்பாடு', 'பயனுள்ள'],
                'mr': ['उत्कृष्ट', 'महान', 'चांगला', 'अप्रतिम', 'समर्थन', 'मंजूरी', 'आवडता', 'फायदा', 'सुधारणा', 'प्रभावी'],
                'gu': ['ઉત્કૃષ્ટ', 'મહાન', 'સારું', 'અદ્ભુત', 'સમર્થન', 'મંજૂરી', 'પસંદ', 'લાભ', 'સુધારો', 'અસરકારક'],
                'kn': ['ಅತ್ಯುತ್ತಮ', 'ಮಹಾನ್', 'ಒಳ್ಳೆಯದು', 'ಅದ್ಭುತ', 'ಬೆಂಬಲ', 'ಅನುಮೋದನೆ', 'ಇಷ್ಟ', 'ಪ್ರಯೋಜನ', 'ಸುಧಾರಣೆ', 'ಪರಿಣಾಮಕಾರಿ'],
                'ml': ['മികച്ച', 'മഹത്തായ', 'നല്ല', 'അത്ഭുതകരം', 'പിന്തുണ', 'അംഗീകാരം', 'ഇഷ്ടം', 'പ്രയോജനം', 'മെച്ചപ്പെടുത്തൽ', 'ഫലപ്രദം'],
                'pa': ['ਸ਼ਾਨਦਾਰ', 'ਮਹਾਨ', 'ਚੰਗਾ', 'ਸ਼ਾਨਦਾਰ', 'ਸਮਰਥਨ', 'ਮਨਜ਼ੂਰੀ', 'ਪਸੰਦ', 'ਫਾਇਦਾ', 'ਸੁਧਾਰ', 'ਪ੍ਰਭਾਵਸ਼ਾਲੀ']
            },
            'negative': {
                'en': ['terrible', 'awful', 'bad', 'horrible', 'disgusting', 'disappointing', 'poor', 'weak', 'fail', 'problem', 'issue', 'concern', 'oppose', 'reject', 'hate', 'dislike', 'harmful', 'damage', 'destroy', 'inadequate'],
                'hi': ['भयानक', 'बुरा', 'घटिया', 'निराशाजनक', 'कमजोर', 'असफल', 'समस्या', 'चिंता', 'विरोध', 'अस्वीकार', 'नफरत', 'हानिकारक', 'नुकसान', 'अपर्याप्त'],
                'bn': ['ভয়ানক', 'খারাপ', 'জঘন্য', 'হতাশাজনক', 'দুর্বল', 'ব্যর্থ', 'সমস্যা', 'উদ্বেগ', 'বিরোধিতা', 'প্রত্যাখ্যান', 'ঘৃণা', 'ক্ষতিকর', 'ক্ষতি', 'অপর্যাপ্ত'],
                'te': ['భయంకరమైన', 'చెడ్డ', 'దారుణమైন', 'నిరాశపరుస', 'బలహీనమైన', 'విఫలమైన', 'సమస్య', 'ఆందోళన', 'వ్యతిరేకత', 'తిరస్కరణ', 'ద్వేషం', 'హానికరమైన', 'నష్టం', 'అసంతృప్త'],
                'ta': ['பயங்கரமான', 'மோசமான', 'கொடூரமான', 'ஏமாற்றமளிக்கும்', 'பலவீனமான', 'தோல்வி', 'பிரச்சனை', 'கவலை', 'எதிர்ப்பு', 'நிராகரிப்பு', 'வெறுப்பு', 'தீங்கான', 'சேதம்', 'போதாத'],
                'mr': ['भयानक', 'वाईट', 'घृणास्पद', 'निराशाजनक', 'कमकुवत', 'अयशस्वी', 'समस्या', 'चिंता', 'विरोध', 'नाकारणे', 'द्वेष', 'हानिकारक', 'नुकसान', 'अपुरा'],
                'gu': ['ભયાનક', 'ખરાબ', 'ઘૃણાસ્પદ', 'નિરાશાજનક', 'નબળું', 'અસફળ', 'સમસ્યા', 'ચિંતા', 'વિરોધ', 'નકારવું', 'નફરત', 'નુકસાનકારક', 'નુકસાન', 'અપૂરતું'],
                'kn': ['ಭಯಾನಕ', 'ಕೆಟ್ಟ', 'ಅಸಹ್ಯಕರ', 'ನಿರಾಶಾದಾಯಕ', 'ದುರ್ಬಲ', 'ವಿಫಲ', 'ಸಮಸ್ಯೆ', 'ಚಿಂತೆ', 'ವಿರೋಧ', 'ನಿರಾಕರಣೆ', 'ದ್ವೇಷ', 'ಹಾನಿಕಾರಕ', 'ಹಾನಿ', 'ಅಸಮರ್ಪಕ'],
                'ml': ['ഭയാനകമായ', 'മോശം', 'വെറുപ്പുളവാക്കുന്ന', 'നിരാശാജനകമായ', 'ദുർബലമായ', 'പരാജയപ്പെട്ട', 'പ്രശ്നം', 'ആശങ്ക', 'എതിർപ്പ്', 'നിരസിക്കൽ', 'വെറുപ്പ്', 'ദോഷകരമായ', 'നാശം', 'അപര്യാപ്തമായ'],
                'pa': ['ਭਿਆਨਕ', 'ਬੁਰਾ', 'ਘਿਣਾਉਣਾ', 'ਨਿਰਾਸ਼ਾਜਨਕ', 'ਕਮਜ਼ੋਰ', 'ਅਸਫਲ', 'ਸਮੱਸਿਆ', 'ਚਿੰਤਾ', 'ਵਿਰੋਧ', 'ਰੱਦ', 'ਨਫ਼ਰਤ', 'ਨੁਕਸਾਨਦਾਇਕ', 'ਨੁਕਸਾਨ', 'ਨਾਕਾਫ਼ੀ']
            },
            'neutral': {
                'en': ['okay', 'fine', 'normal', 'average', 'standard', 'typical', 'regular', 'usual', 'moderate', 'balanced', 'neutral', 'mixed', 'unclear', 'uncertain', 'maybe', 'perhaps'],
                'hi': ['ठीक', 'सामान्य', 'औसत', 'मानक', 'संतुलित', 'तटस्थ', 'मिश्रित', 'अस्पष्ट', 'अनिश्चित', 'शायद'],
                'bn': ['ঠিক', 'স্বাভাবিক', 'গড়', 'মানক', 'সুষম', 'নিরপেক্ষ', 'মিশ্র', 'অস্পষ্ট', 'অনিশ্চিত', 'হয়তো'],
                'te': ['సరే', 'సాధారణ', 'సరాసరి', 'ప్రామాణిక', 'సమతుల్య', 'తటస్థ', 'మిశ్రమ', 'అస్పష్ట', 'అనిశ్చిత', 'బహుశా'],
                'ta': ['சரி', 'சாதாரண', 'சராசரி', 'நியமான', 'சமநிலை', 'நடுநிலை', 'கலந்த', 'தெளிவற்ற', 'நிச்சயமற்ற', 'ஒருவேळை'],
                'mr': ['ठीक', 'सामान्य', 'सरासरी', 'मानक', 'संतुलित', 'तटस्थ', 'मिश्रित', 'अस्पष्ट', 'अनिश्चित', 'कदाचित'],
                'gu': ['ઠીક', 'સામાન્ય', 'સરેરાશ', 'પ્રમાણભૂત', 'સંતુલિત', 'તટસ્થ', 'મિશ્ર', 'અસ્પષ્ટ', 'અનિશ્ચિત', 'કદાચ'],
                'kn': ['ಸರಿ', 'ಸಾಮಾನ್ಯ', 'ಸರಾಸರಿ', 'ಪ್ರಮಾಣಿತ', 'ಸಮತೋಲಿತ', 'ತಟಸ್ಥ', 'ಮಿಶ್ರ', 'ಅಸ್ಪಷ್ಟ', 'ಅನಿಶ್ಚಿತ', 'ಬಹುಶಃ'],
                'ml': ['ശരി', 'സാധാരണ', 'ശരാശരി', 'പ്രമാണം', 'സമതുലിത', 'നിഷ്പക്ഷ', 'മിശ്ര', 'അവ്യക്ത', 'അനിശ്ചിത', 'ഒരുപക്ഷേ'],
                'pa': ['ਠੀਕ', 'ਸਧਾਰਨ', 'ਔਸਤ', 'ਮਿਆਰੀ', 'ਸੰਤੁਲਿਤ', 'ਨਿਰਪੱਖ', 'ਮਿਸ਼ਰਤ', 'ਅਸਪਸ਼ਟ', 'ਅਨਿਸ਼ਚਿਤ', 'ਸ਼ਾਇਦ']
            }
        }
    
    def _initialize_advanced_patterns(self):
        """Initialize advanced sentiment patterns and rules"""
        self.sentiment_patterns = {
            'strong_positive': [
                r'(strongly|highly|extremely|absolutely|completely|totally)\s+(support|recommend|approve|endorse|love|like)',
                r'(excellent|outstanding|fantastic|amazing|brilliant|superb|wonderful|great|perfect)',
                r'(will\s+(greatly|significantly|substantially)\s+(benefit|improve|enhance|help))'
            ],
            'strong_negative': [
                r'(strongly|highly|extremely|absolutely|completely|totally)\s+(oppose|reject|condemn|hate|dislike|disapprove)',
                r'(terrible|awful|horrible|disgusting|appalling|dreadful|atrocious|devastating)',
                r'(will\s+(greatly|significantly|substantially)\s+(harm|damage|destroy|hurt))'
            ],
            'conditional': [
                r'(if|unless|provided|assuming|given|suppose)',
                r'(might|could|would|should|may)\s+(be|work|help|improve)'
            ],
            'uncertainty': [
                r'(not\s+sure|uncertain|unclear|ambiguous|mixed\s+feelings)',
                r'(on\s+the\s+one\s+hand|on\s+the\s+other\s+hand)',
                r'(both\s+positive\s+and\s+negative|pros\s+and\s+cons)'
            ]
        }
    
    def detect_language(self, text: str) -> Dict[str, Any]:
        """Detect language(s) in the text"""
        try:
            if LANGDETECT_AVAILABLE:
                # Primary detection
                detected_lang = detect(text)
                lang_probs = detect_langs(text)
                
                # Check for mixed languages
                is_mixed = len(lang_probs) > 1 and lang_probs[1].prob > 0.1
                
                # Check for Indian scripts
                has_devanagari = bool(re.search(r'[\u0900-\u097F]', text))
                has_tamil = bool(re.search(r'[\u0B80-\u0BFF]', text))
                has_telugu = bool(re.search(r'[\u0C00-\u0C7F]', text))
                has_kannada = bool(re.search(r'[\u0C80-\u0CFF]', text))
                has_malayalam = bool(re.search(r'[\u0D00-\u0D7F]', text))
                has_bengali = bool(re.search(r'[\u0980-\u09FF]', text))
                has_gujarati = bool(re.search(r'[\u0A80-\u0AFF]', text))
                has_punjabi = bool(re.search(r'[\u0A00-\u0A7F]', text))
                
                scripts = []
                if has_devanagari: scripts.append('devanagari')
                if has_tamil: scripts.append('tamil')
                if has_telugu: scripts.append('telugu')
                if has_kannada: scripts.append('kannada')
                if has_malayalam: scripts.append('malayalam')
                if has_bengali: scripts.append('bengali')
                if has_gujarati: scripts.append('gujarati')
                if has_punjabi: scripts.append('punjabi')
                
                return {
                    'primary_language': detected_lang,
                    'confidence': lang_probs[0].prob if lang_probs else 0.5,
                    'is_mixed': is_mixed or len(scripts) > 1,
                    'all_languages': [lang.lang for lang in lang_probs],
                    'indian_scripts': scripts,
                    'has_english': bool(re.search(r'[a-zA-Z]', text)),
                    'has_indian_script': len(scripts) > 0
                }
            else:
                # Fallback detection
                has_english = bool(re.search(r'[a-zA-Z]', text))
                has_devanagari = bool(re.search(r'[\u0900-\u097F]', text))
                
                if has_english and has_devanagari:
                    return {'primary_language': 'mixed', 'is_mixed': True, 'confidence': 0.7}
                elif has_devanagari:
                    return {'primary_language': 'hi', 'is_mixed': False, 'confidence': 0.6}
                else:
                    return {'primary_language': 'en', 'is_mixed': False, 'confidence': 0.8}
                    
        except Exception as e:
            self.logger.error(f"Language detection error: {e}")
            return {'primary_language': 'en', 'is_mixed': False, 'confidence': 0.5}
    
    def analyze_sentiment_transformer(self, text: str, lang_info: Dict) -> Dict[str, Any]:
        """Analyze sentiment using transformer models"""
        if not self.models_loaded:
            return None
            
        try:
            # Use appropriate model based on language
            # If Indian script present or primary language is Indian, prefer IndicBERT
            if (lang_info.get('has_indian_script') or lang_info.get('primary_language') in ['hi','bn','te','ta','mr','gu','kn','ml','pa','or','as','ur','ne']) and self.indic_sentiment_model:
                # Try Indic model for Indian languages
                try:
                    result = self.indic_sentiment_model(text)
                    return {
                        'model': 'indic-bert',
                        'label': result[0]['label'].lower(),
                        'score': result[0]['score'],
                        'confidence': result[0]['score']
                    }
                except:
                    pass
            
            # Use multilingual model
            result = self.multilingual_sentiment_model(text)
            
            # Map labels to standard format
            label_mapping = {
                'LABEL_0': 'negative',
                'LABEL_1': 'neutral', 
                'LABEL_2': 'positive',
                'negative': 'negative',
                'neutral': 'neutral',
                'positive': 'positive'
            }
            
            mapped_label = label_mapping.get(result[0]['label'], result[0]['label'].lower())
            
            return {
                'model': 'xlm-roberta',
                'label': mapped_label,
                'score': result[0]['score'],
                'confidence': result[0]['score']
            }
            
        except Exception as e:
            self.logger.error(f"Transformer sentiment analysis error: {e}")
            return None
    
    def analyze_sentiment_multilingual_keywords(self, text: str, lang_info: Dict) -> Dict[str, Any]:
        """Analyze sentiment using multilingual keywords"""
        text_lower = text.lower()
        
        # Get relevant languages
        languages_to_check = ['en']  # Always check English
        
        if lang_info.get('primary_language') in self.multilingual_keywords['positive']:
            languages_to_check.append(lang_info['primary_language'])
        
        if lang_info.get('has_indian_script'):
            # Add likely Indian languages based on script
            for script in lang_info.get('indian_scripts', []):
                if script == 'devanagari':
                    languages_to_check.extend(['hi', 'mr'])
                elif script == 'tamil':
                    languages_to_check.append('ta')
                elif script == 'telugu':
                    languages_to_check.append('te')
                elif script == 'kannada':
                    languages_to_check.append('kn')
                elif script == 'malayalam':
                    languages_to_check.append('ml')
                elif script == 'bengali':
                    languages_to_check.append('bn')
                elif script == 'gujarati':
                    languages_to_check.append('gu')
                elif script == 'punjabi':
                    languages_to_check.append('pa')
        
        # Count sentiment words across all relevant languages
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        found_words = {'positive': [], 'negative': [], 'neutral': []}
        
        for lang in set(languages_to_check):
            if lang in self.multilingual_keywords['positive']:
                for word in self.multilingual_keywords['positive'][lang]:
                    if word in text_lower:
                        positive_count += 1
                        found_words['positive'].append(word)
                        
                for word in self.multilingual_keywords['negative'][lang]:
                    if word in text_lower:
                        negative_count += 1
                        found_words['negative'].append(word)
                        
                for word in self.multilingual_keywords['neutral'][lang]:
                    if word in text_lower:
                        neutral_count += 1
                        found_words['neutral'].append(word)
        
        # Calculate sentiment based on keyword counts
        total_sentiment_words = positive_count + negative_count + neutral_count
        
        if total_sentiment_words == 0:
            return {
                'method': 'multilingual_keywords',
                'label': 'neutral',
                'confidence': 0.3,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'found_words': found_words
            }
        
        # Determine sentiment
        if positive_count > negative_count and positive_count > neutral_count:
            sentiment = 'positive'
            confidence = min(0.9, 0.5 + (positive_count / total_sentiment_words) * 0.4)
        elif negative_count > positive_count and negative_count > neutral_count:
            sentiment = 'negative'
            confidence = min(0.9, 0.5 + (negative_count / total_sentiment_words) * 0.4)
        else:
            sentiment = 'neutral'
            confidence = 0.5
        
        return {
            'method': 'multilingual_keywords',
            'label': sentiment,
            'confidence': confidence,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'found_words': found_words,
            'languages_checked': languages_to_check
        }
    
    def analyze_sentiment_vader(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using VADER (good for English social media text)"""
        if not TEXTPROCESSING_AVAILABLE:
            return None
            
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            
            # Determine sentiment based on compound score
            if scores['compound'] >= 0.05:
                sentiment = 'positive'
            elif scores['compound'] <= -0.05:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            return {
                'method': 'vader',
                'label': sentiment,
                'confidence': abs(scores['compound']),
                'scores': scores
            }
        except Exception as e:
            self.logger.error(f"VADER analysis error: {e}")
            return None
    
    def analyze_sentiment_textblob(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using TextBlob"""
        if not TEXTPROCESSING_AVAILABLE:
            return None
            
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                sentiment = 'positive'
            elif polarity < -0.1:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            return {
                'method': 'textblob',
                'label': sentiment,
                'confidence': abs(polarity),
                'polarity': polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
        except Exception as e:
            self.logger.error(f"TextBlob analysis error: {e}")
            return None
    
    def ensemble_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Perform ensemble sentiment analysis using multiple methods"""
        
        # Language detection
        lang_info = self.detect_language(text)
        
        # Collect results from different methods
        results = {}
        
        # Method 1: Transformer models (highest priority)
        transformer_result = self.analyze_sentiment_transformer(text, lang_info)
        if transformer_result:
            results['transformer'] = transformer_result
        
        # Method 2: Multilingual keywords
        keyword_result = self.analyze_sentiment_multilingual_keywords(text, lang_info)
        if keyword_result:
            results['keywords'] = keyword_result
        
        # Method 3: VADER (for English text)
        if lang_info.get('has_english', True):
            vader_result = self.analyze_sentiment_vader(text)
            if vader_result:
                results['vader'] = vader_result
        
        # Method 4: TextBlob (backup)
        textblob_result = self.analyze_sentiment_textblob(text)
        if textblob_result:
            results['textblob'] = textblob_result
        
        # Ensemble decision
        return self._make_ensemble_decision(results, lang_info, text)
    
    def _make_ensemble_decision(self, results: Dict, lang_info: Dict, text: str) -> Dict[str, Any]:
        """Make final sentiment decision based on ensemble results"""
        
        if not results:
            return {
                'sentiment': 'neutral',
                'confidence': 0.3,
                'polarity_score': 0.0,
                'method': 'fallback',
                'language_info': lang_info,
                'individual_results': results
            }
        
        # Weight different methods
        method_weights = {
            'transformer': 0.4,
            'keywords': 0.3,
            'vader': 0.2,
            'textblob': 0.1
        }
        
        # Calculate weighted sentiment scores
        sentiment_scores = {'positive': 0, 'negative': 0, 'neutral': 0}
        total_weight = 0
        confidence_sum = 0
        
        for method, result in results.items():
            weight = method_weights.get(method, 0.1)
            confidence = result.get('confidence', 0.5)
            
            sentiment_scores[result['label']] += weight * confidence
            total_weight += weight
            confidence_sum += confidence
        
        # Normalize scores
        if total_weight > 0:
            for sentiment in sentiment_scores:
                sentiment_scores[sentiment] /= total_weight
        
        # Determine final sentiment
        final_sentiment = max(sentiment_scores, key=sentiment_scores.get)
        final_confidence = min(0.95, confidence_sum / len(results))
        
        # Calculate polarity score
        polarity_score = sentiment_scores['positive'] - sentiment_scores['negative']
        
        # Get explanation components
        explanation_data = self._generate_detailed_explanation(text, results, lang_info, final_sentiment)
        
        return {
            'sentiment': final_sentiment,
            'confidence': final_confidence,
            'polarity_score': polarity_score,
            'method': 'ensemble',
            'language_info': lang_info,
            'individual_results': results,
            'sentiment_scores': sentiment_scores,
            'explanation': explanation_data['explanation'],
            'key_indicators': explanation_data['key_indicators'],
            'highlighted_words': explanation_data['highlighted_words']
        }
    
    def _generate_detailed_explanation(self, text: str, results: Dict, lang_info: Dict, final_sentiment: str) -> Dict[str, Any]:
        """Generate detailed explanation for sentiment classification"""
        
        explanation_parts = []
        key_indicators = {'positive': [], 'negative': [], 'neutral': [], 'languages': []}
        highlighted_words = []
        
        # Language information
        if lang_info.get('is_mixed'):
            explanation_parts.append(f"**Mixed Language Text** detected with primary language: {lang_info.get('primary_language', 'unknown')}")
            key_indicators['languages'] = lang_info.get('all_languages', [])
        else:
            explanation_parts.append(f"**Language Detected:** {self.supported_languages.get(lang_info.get('primary_language'), lang_info.get('primary_language'))}")
        
        # Method-specific explanations
        for method, result in results.items():
            if method == 'transformer':
                explanation_parts.append(f"**AI Model ({result.get('model', 'transformer')})** classified as: {result['label']} (confidence: {result.get('confidence', 0):.3f})")
            
            elif method == 'keywords':
                found_words = result.get('found_words', {})
                if found_words['positive']:
                    explanation_parts.append(f"**Positive keywords found:** {', '.join(found_words['positive'][:5])}")
                    key_indicators['positive'].extend(found_words['positive'])
                    highlighted_words.extend([(word, 'positive') for word in found_words['positive']])
                
                if found_words['negative']:
                    explanation_parts.append(f"**Negative keywords found:** {', '.join(found_words['negative'][:5])}")
                    key_indicators['negative'].extend(found_words['negative'])
                    highlighted_words.extend([(word, 'negative') for word in found_words['negative']])
                
                if found_words['neutral']:
                    explanation_parts.append(f"**Neutral keywords found:** {', '.join(found_words['neutral'][:3])}")
                    key_indicators['neutral'].extend(found_words['neutral'])
                    highlighted_words.extend([(word, 'neutral') for word in found_words['neutral']])
                
                languages_checked = result.get('languages_checked', [])
                if len(languages_checked) > 1:
                    explanation_parts.append(f"**Languages analyzed:** {', '.join(languages_checked)}")
            
            elif method == 'vader':
                scores = result.get('scores', {})
                explanation_parts.append(f"**VADER Analysis:** Compound score {scores.get('compound', 0):.3f} (pos: {scores.get('pos', 0):.2f}, neg: {scores.get('neg', 0):.2f}, neu: {scores.get('neu', 0):.2f})")
        
        # Final reasoning
        explanation_parts.append(f"**Final Classification:** {final_sentiment.upper()} based on ensemble analysis of {len(results)} methods")
        
        if lang_info.get('has_indian_script'):
            explanation_parts.append("**Multilingual Support:** Indian language content analyzed using specialized models")
        
        # Convert key_indicators to match dashboard expectations
        dashboard_indicators = {
            'positive_words': key_indicators.get('positive', []),
            'negative_words': key_indicators.get('negative', []),
            'neutral_words': key_indicators.get('neutral', []),
            'intensifiers': [],  # Will be populated later
            'negations': []      # Will be populated later
        }
        
        return {
            'explanation': '\n\n'.join(explanation_parts),
            'key_indicators': dashboard_indicators,
            'highlighted_words': highlighted_words
        }
    
    def summarize_text_mt5(self, text: str, max_length: int = 150) -> str:
        """Summarize text using mT5 model for multilingual support"""
        
        if not self.models_loaded or not hasattr(self, 'mt5_model'):
            return "Summarization not available - mT5 model not loaded"
        
        try:
            # Prepare input for mT5
            input_text = f"summarize: {text}"
            
            # Tokenize
            inputs = self.mt5_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
            
            # Generate summary
            with torch.no_grad():
                summary_ids = self.mt5_model.generate(
                    inputs, 
                    max_length=max_length,
                    min_length=30,
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True
                )
            
            # Decode summary
            summary = self.mt5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"mT5 summarization error: {e}")
            return f"Summarization failed: {str(e)}"
    
    def analyze_text_comprehensive(self, text: str) -> Dict[str, Any]:
        """Comprehensive text analysis with multilingual support"""
        
        # Main sentiment analysis
        sentiment_result = self.ensemble_sentiment_analysis(text)
        
        # Text summarization
        summary = self.summarize_text_mt5(text)
        
        # Additional analysis
        lang_info = sentiment_result.get('language_info', {})
        
        return {
            'text': text,
            'sentiment': sentiment_result['sentiment'],
            'confidence': sentiment_result['confidence'],
            'polarity_score': sentiment_result['polarity_score'],
            'language_info': lang_info,
            'explanation': sentiment_result['explanation'],
            'key_indicators': sentiment_result['key_indicators'],
            'highlighted_words': sentiment_result['highlighted_words'],
            'summary': summary,
            'analysis_methods': list(sentiment_result.get('individual_results', {}).keys()),
            'timestamp': datetime.now().isoformat(),
            'is_multilingual': lang_info.get('is_mixed', False),
            'supported_languages': list(self.supported_languages.keys())
        }

# Initialize global analyzer instance
try:
    global_analyzer = AdvancedMultilingualSentimentAnalyzer()
    ADVANCED_ANALYZER_AVAILABLE = True
    print("✅ Advanced Multilingual Sentiment Analyzer initialized successfully")
except Exception as e:
    ADVANCED_ANALYZER_AVAILABLE = False
    print(f"❌ Failed to initialize Advanced Analyzer: {e}")

# Convenience functions
def analyze_text_advanced(text: str) -> Dict[str, Any]:
    """Analyze text with advanced multilingual sentiment analysis"""
    if ADVANCED_ANALYZER_AVAILABLE:
        return global_analyzer.analyze_text_comprehensive(text)
    else:
        return {
            'text': text,
            'sentiment': 'neutral',
            'confidence': 0.3,
            'error': 'Advanced analyzer not available'
        }

def summarize_with_mt5(text: str, max_length: int = 150) -> str:
    """Summarize text using mT5"""
    if ADVANCED_ANALYZER_AVAILABLE:
        return global_analyzer.summarize_text_mt5(text, max_length)
    else:
        return "mT5 summarization not available"

if __name__ == "__main__":
    # Test with mixed language examples
    test_texts = [
        "I strongly support this policy initiative. यह बहुत अच्छी योजना है।",
        "This proposal is terrible. இது மோசமான யோசனை.",
        "The framework seems reasonable और संतुलित भी है।",
        "मुझे लगता है कि यह policy very good है and will benefit everyone.",
        "I am not sure about this. थोड़ा confusing है।"
    ]
    
    print("=== Testing Advanced Multilingual Sentiment Analysis ===\n")
    
    for i, text in enumerate(test_texts, 1):
        print(f"Example {i}: {text}")
        result = analyze_text_advanced(text)
        print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.3f})")
        print(f"Language: {result.get('language_info', {}).get('primary_language', 'unknown')}")
        print(f"Mixed: {result.get('is_multilingual', False)}")
        print(f"Summary: {result['summary'][:100]}...")
        print("-" * 80)