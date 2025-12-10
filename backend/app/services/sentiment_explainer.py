"""
Sentiment Analysis Explanation Module
Provides detailed explanations for why text is classified as positive, negative, or neutral
Enhanced with advanced multilingual support and word highlighting
"""

import re
from typing import Dict, List, Tuple
from textblob import TextBlob
import nltk
from collections import Counter
import sys
import os

# Add path for advanced analyzer
current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)

# Import advanced analyzer
try:
    from advanced_sentiment_analyzer import analyze_text_advanced, ADVANCED_ANALYZER_AVAILABLE
    ADVANCED_SUPPORT = True
except ImportError:
    ADVANCED_SUPPORT = False

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class SentimentExplainer:
    """
    Explains sentiment analysis results with detailed reasoning
    """
    
    def __init__(self):
        # Positive sentiment indicators
        self.positive_keywords = {
            'excellent', 'fantastic', 'amazing', 'wonderful', 'great', 'good', 'outstanding',
            'superb', 'brilliant', 'impressive', 'remarkable', 'terrific', 'magnificent',
            'marvelous', 'splendid', 'awesome', 'incredible', 'extraordinary', 'phenomenal',
            'support', 'endorse', 'approve', 'love', 'appreciate', 'welcome', 'favor',
            'enthusiastic', 'excited', 'thrilled', 'delighted', 'pleased', 'satisfied',
            'benefit', 'improve', 'enhance', 'boost', 'strengthen', 'advance', 'progress',
            'success', 'achievement', 'opportunity', 'advantage', 'effective', 'efficient'
        }
        
        # Negative sentiment indicators
        self.negative_keywords = {
            'terrible', 'awful', 'horrible', 'disgusting', 'appalling', 'dreadful',
            'atrocious', 'abysmal', 'deplorable', 'catastrophic', 'disastrous', 'tragic',
            'devastating', 'alarming', 'concerning', 'worrying', 'troubling', 'disturbing',
            'oppose', 'reject', 'condemn', 'criticize', 'disapprove', 'hate', 'despise',
            'disappointed', 'frustrated', 'angry', 'outraged', 'concerned', 'worried',
            'fail', 'failure', 'problem', 'issue', 'challenge', 'difficulty', 'weakness',
            'inadequate', 'insufficient', 'poor', 'weak', 'flawed', 'deficient', 'lacking'
        }
        
        # Neutral sentiment indicators
        self.neutral_keywords = {
            'adequate', 'standard', 'routine', 'conventional', 'moderate', 'balanced',
            'reasonable', 'acceptable', 'satisfactory', 'ordinary', 'typical', 'normal',
            'average', 'regular', 'basic', 'simple', 'straightforward', 'neutral',
            'uncertain', 'unsure', 'unclear', 'mixed', 'ambiguous', 'undecided',
            'maintain', 'preserve', 'continue', 'status quo', 'existing', 'current'
        }
        
        # Intensifiers that amplify sentiment
        self.intensifiers = {
            'very', 'extremely', 'incredibly', 'absolutely', 'completely', 'totally',
            'utterly', 'highly', 'deeply', 'seriously', 'significantly', 'substantially',
            'tremendously', 'enormously', 'exceptionally', 'remarkably', 'particularly',
            'especially', 'quite', 'rather', 'fairly', 'somewhat', 'definitely'
        }
        
        # Negation words that can flip sentiment
        self.negations = {
            'not', 'no', 'never', 'nothing', 'nowhere', 'nobody', 'none', 'neither',
            'nor', 'cannot', 'can\'t', 'won\'t', 'wouldn\'t', 'shouldn\'t', 'doesn\'t',
            'don\'t', 'isn\'t', 'aren\'t', 'wasn\'t', 'weren\'t', 'hasn\'t', 'haven\'t'
        }
    
    def explain_sentiment(self, text: str) -> Dict:
        """
        Provide detailed explanation for sentiment classification with advanced multilingual support
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with sentiment score, classification, detailed explanation, and word highlighting
        """
        
        # Try advanced analyzer first
        if ADVANCED_SUPPORT and ADVANCED_ANALYZER_AVAILABLE:
            try:
                advanced_result = analyze_text_advanced(text)
                
                # Extract highlighting information
                highlighted_words = advanced_result.get('highlighted_words', [])
                
                # Format for display
                highlighted_text = self._highlight_text_html(text, highlighted_words)
                
                return {
                    'sentiment': advanced_result['sentiment'].title(),
                    'polarity_score': advanced_result['polarity_score'],
                    'subjectivity_score': advanced_result.get('language_info', {}).get('subjectivity', 0.5),
                    'confidence': self._map_confidence_to_string(advanced_result['confidence']),
                    'explanation': advanced_result['explanation'],
                    'key_indicators': advanced_result['key_indicators'],
                    'highlighted_text': highlighted_text,
                    'highlighted_words': highlighted_words,
                    'language_info': advanced_result.get('language_info', {}),
                    'analysis_methods': advanced_result.get('analysis_methods', []),
                    'is_multilingual': advanced_result.get('is_multilingual', False),
                    'summary': advanced_result.get('summary', ''),
                    'method': 'advanced_multilingual'
                }
            except Exception as e:
                print(f"Advanced analyzer failed, falling back to basic: {e}")
        
        # Fallback to original TextBlob analysis
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Classify sentiment
        if polarity > 0.1:
            sentiment = "Positive"
        elif polarity < -0.1:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        
        # Analyze text components
        words = self._tokenize_text(text.lower())
        
        # Find sentiment indicators
        positive_found = self._find_sentiment_words(words, self.positive_keywords)
        negative_found = self._find_sentiment_words(words, self.negative_keywords)
        neutral_found = self._find_sentiment_words(words, self.neutral_keywords)
        intensifiers_found = self._find_sentiment_words(words, self.intensifiers)
        negations_found = self._find_sentiment_words(words, self.negations)
        
        # Create highlighting for basic analysis
        highlighted_words = []
        highlighted_words.extend([(word, 'positive') for word in positive_found])
        highlighted_words.extend([(word, 'negative') for word in negative_found])
        highlighted_words.extend([(word, 'neutral') for word in neutral_found])
        highlighted_words.extend([(word, 'intensifier') for word in intensifiers_found])
        highlighted_words.extend([(word, 'negation') for word in negations_found])
        
        highlighted_text = self._highlight_text_html(text, highlighted_words)
        
        # Analyze phrases and context
        positive_phrases = self._find_sentiment_phrases(text, 'positive')
        negative_phrases = self._find_sentiment_phrases(text, 'negative')
        neutral_phrases = self._find_sentiment_phrases(text, 'neutral')
        
        # Generate explanation
        explanation = self._generate_explanation(
            sentiment, polarity, subjectivity,
            positive_found, negative_found, neutral_found,
            intensifiers_found, negations_found,
            positive_phrases, negative_phrases, neutral_phrases
        )
        
        return {
            'sentiment': sentiment,
            'polarity_score': round(polarity, 3),
            'subjectivity_score': round(subjectivity, 3),
            'confidence': self._calculate_confidence(polarity),
            'explanation': explanation,
            'highlighted_text': highlighted_text,
            'highlighted_words': highlighted_words,
            'key_indicators': {
                'positive_words': positive_found,
                'negative_words': negative_found,
                'neutral_words': neutral_found,
                'intensifiers': intensifiers_found,
                'negations': negations_found
            },
            'phrases': {
                'positive_phrases': positive_phrases,
                'negative_phrases': negative_phrases,
                'neutral_phrases': neutral_phrases
            },
            'method': 'textblob_fallback'
        }
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into words"""
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    def _find_sentiment_words(self, words: List[str], keyword_set: set) -> List[str]:
        """Find words that match sentiment keywords"""
        found_words = []
        for word in words:
            if word in keyword_set:
                found_words.append(word)
        return found_words
    
    def _find_sentiment_phrases(self, text: str, sentiment_type: str) -> List[str]:
        """Find sentiment-bearing phrases in text"""
        text_lower = text.lower()
        phrases = []
        
        if sentiment_type == 'positive':
            positive_patterns = [
                r'strongly support', r'wholeheartedly endorse', r'excellent initiative',
                r'fantastic.*!', r'amazing.*!', r'wonderful.*!', r'love.*!',
                r'thrilled.*!', r'excited.*!', r'appreciate.*effort',
                r'will.*benefit', r'will.*improve', r'will.*enhance'
            ]
            for pattern in positive_patterns:
                matches = re.findall(pattern, text_lower)
                phrases.extend(matches)
                
        elif sentiment_type == 'negative':
            negative_patterns = [
                r'absolutely terrible', r'complete disaster', r'deeply disappointed',
                r'serious.*concern', r'significant.*problem', r'major.*issue',
                r'fail.*to', r'lack.*clarity', r'inadequate.*standard',
                r'will.*harm', r'will.*damage', r'will.*destroy'
            ]
            for pattern in negative_patterns:
                matches = re.findall(pattern, text_lower)
                phrases.extend(matches)
                
        elif sentiment_type == 'neutral':
            neutral_patterns = [
                r'not sure', r'no major.*concern', r'status quo', r'routine update',
                r'standard.*framework', r'balanced approach', r'reasonable.*balance',
                r'adequate.*framework', r'maintains.*existing'
            ]
            for pattern in neutral_patterns:
                matches = re.findall(pattern, text_lower)
                phrases.extend(matches)
        
        return phrases
    
    def _highlight_text_html(self, text: str, highlighted_words: List[Tuple[str, str]]) -> str:
        """Create HTML with highlighted sentiment words"""
        if not highlighted_words:
            return text
        
        highlighted_text = text
        
        # Color mapping for different word types
        color_map = {
            'positive': '#d4edda',      # Light green
            'negative': '#f8d7da',      # Light red
            'neutral': '#d1ecf1',       # Light blue
            'intensifier': '#fff3cd',   # Light yellow
            'negation': '#e2e3e5'       # Light gray
        }
        
        # Sort by word length (descending) to avoid partial replacements
        sorted_words = sorted(highlighted_words, key=lambda x: len(x[0]), reverse=True)
        
        for word, sentiment_type in sorted_words:
            color = color_map.get(sentiment_type, '#f8f9fa')
            
            # Create case-insensitive pattern
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            
            # Replace with highlighted version
            highlighted_text = pattern.sub(
                f'<span style="background-color: {color}; padding: 1px 3px; border-radius: 3px; margin: 1px;">{word}</span>',
                highlighted_text
            )
        
        return highlighted_text
    
    def _map_confidence_to_string(self, confidence_float: float) -> str:
        """Map confidence float to string"""
        if confidence_float >= 0.8:
            return "Very High"
        elif confidence_float >= 0.6:
            return "High"
        elif confidence_float >= 0.4:
            return "Medium"
        else:
            return "Low"
    
    def _calculate_confidence(self, polarity: float) -> str:
        """Calculate confidence level based on polarity strength"""
        abs_polarity = abs(polarity)
        if abs_polarity > 0.5:
            return "Very High"
        elif abs_polarity > 0.3:
            return "High"
        elif abs_polarity > 0.1:
            return "Medium"
        else:
            return "Low"
    
    def _generate_explanation(self, sentiment: str, polarity: float, subjectivity: float,
                            positive_words: List[str], negative_words: List[str], neutral_words: List[str],
                            intensifiers: List[str], negations: List[str],
                            positive_phrases: List[str], negative_phrases: List[str], neutral_phrases: List[str]) -> str:
        """Generate human-readable explanation for sentiment classification"""
        
        explanation_parts = []
        
        # Main classification explanation
        explanation_parts.append(f"This text is classified as **{sentiment}** with a polarity score of {polarity:.3f}.")
        
        # Explain polarity score
        if polarity > 0:
            explanation_parts.append(f"The positive polarity score ({polarity:.3f}) indicates favorable sentiment.")
        elif polarity < 0:
            explanation_parts.append(f"The negative polarity score ({polarity:.3f}) indicates unfavorable sentiment.")
        else:
            explanation_parts.append(f"The neutral polarity score ({polarity:.3f}) indicates balanced or objective sentiment.")
        
        # Explain key words found
        if positive_words:
            explanation_parts.append(f"**Positive indicators found:** {', '.join(positive_words[:5])}{'...' if len(positive_words) > 5 else ''}")
        
        if negative_words:
            explanation_parts.append(f"**Negative indicators found:** {', '.join(negative_words[:5])}{'...' if len(negative_words) > 5 else ''}")
        
        if neutral_words:
            explanation_parts.append(f"**Neutral indicators found:** {', '.join(neutral_words[:5])}{'...' if len(neutral_words) > 5 else ''}")
        
        # Explain intensifiers and modifiers
        if intensifiers:
            explanation_parts.append(f"**Intensifying words** like '{', '.join(intensifiers[:3])}' amplify the sentiment strength.")
        
        if negations:
            explanation_parts.append(f"**Negation words** like '{', '.join(negations[:3])}' may reverse or weaken sentiment.")
        
        # Explain key phrases
        if positive_phrases:
            explanation_parts.append(f"**Positive phrases** detected: '{', '.join(positive_phrases[:2])}'")
        
        if negative_phrases:
            explanation_parts.append(f"**Negative phrases** detected: '{', '.join(negative_phrases[:2])}'")
        
        if neutral_phrases:
            explanation_parts.append(f"**Neutral phrases** detected: '{', '.join(neutral_phrases[:2])}'")
        
        # Subjectivity explanation
        if subjectivity > 0.5:
            explanation_parts.append(f"The high subjectivity score ({subjectivity:.3f}) indicates this text expresses personal opinions or emotions.")
        else:
            explanation_parts.append(f"The low subjectivity score ({subjectivity:.3f}) indicates this text is more factual or objective.")
        
        # Overall reasoning
        if sentiment == "Positive":
            explanation_parts.append("**Overall:** The text expresses approval, support, or favorable opinion through positive language, supportive phrases, and optimistic tone.")
        elif sentiment == "Negative":
            explanation_parts.append("**Overall:** The text expresses disapproval, criticism, or unfavorable opinion through negative language, critical phrases, and pessimistic tone.")
        else:
            explanation_parts.append("**Overall:** The text maintains a balanced, objective, or uncertain tone without strong positive or negative bias.")
        
        return "\n\n".join(explanation_parts)

# Example usage function
def analyze_text_with_explanation(text: str) -> Dict:
    """
    Analyze text sentiment with detailed explanation
    
    Args:
        text: Input text to analyze
        
    Returns:
        Complete analysis with explanation
    """
    explainer = SentimentExplainer()
    return explainer.explain_sentiment(text)

if __name__ == "__main__":
    # Test examples
    test_texts = [
        "I strongly support the new digital governance framework. It provides excellent transparency and accountability measures that will benefit all citizens.",
        "This policy is absolutely terrible. It completely ignores environmental concerns and will cause irreversible damage to our ecosystems.",
        "The proposed amendments are reasonable and strike a good balance between regulatory oversight and business flexibility."
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n=== Example {i} ===")
        result = analyze_text_with_explanation(text)
        print(f"Text: {text[:100]}...")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Score: {result['polarity_score']}")
        print(f"Confidence: {result['confidence']}")
        print("\nExplanation:")
        print(result['explanation'])