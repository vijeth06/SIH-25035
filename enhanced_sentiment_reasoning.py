"""
Enhanced Sentiment Reasoning with Word Highlighting
"""
import re
from typing import Dict, List, Tuple, Any
from textblob import TextBlob

class EnhancedSentimentReasoner:
    """Enhanced sentiment analysis with detailed reasoning and word highlighting"""
    
    def __init__(self):
        # Comprehensive sentiment word lists with strength indicators
        self.positive_words = {
            'strong': ['excellent', 'outstanding', 'amazing', 'wonderful', 'fantastic', 'superb', 'brilliant', 'exceptional'],
            'medium': ['good', 'great', 'nice', 'pleasant', 'helpful', 'beneficial', 'effective', 'support', 'appreciate'],
            'mild': ['okay', 'fine', 'decent', 'reasonable', 'acceptable', 'satisfactory', 'adequate']
        }
        
        self.negative_words = {
            'strong': ['terrible', 'awful', 'horrible', 'devastating', 'catastrophic', 'abysmal', 'disastrous'],
            'medium': ['bad', 'poor', 'wrong', 'problem', 'issue', 'concern', 'lack', 'insufficient', 'inadequate'],
            'mild': ['somewhat', 'slightly', 'minor', 'small', 'limited']
        }
        
        self.neutral_words = {
            'balanced': ['balanced', 'neutral', 'objective', 'fair', 'reasonable', 'moderate'],
            'factual': ['according', 'states', 'indicates', 'shows', 'reports', 'mentions', 'describes'],
            'uncertain': ['maybe', 'perhaps', 'possibly', 'might', 'could', 'unclear', 'uncertain']
        }
        
        self.intensifiers = ['very', 'extremely', 'highly', 'really', 'quite', 'absolutely', 'completely', 'totally']
        self.negations = ['not', 'no', 'never', 'nothing', 'nobody', 'nowhere', 'neither', 'nor', 'none']
        
        # Contextual phrases that affect sentiment
        self.positive_phrases = [
            'strongly support', 'highly recommend', 'great initiative', 'excellent work',
            'well done', 'good job', 'appreciate the', 'thank you for', 'pleased with'
        ]
        
        self.negative_phrases = [
            'strongly oppose', 'completely disagree', 'major concern', 'serious problem',
            'lacks clarity', 'insufficient detail', 'poorly designed', 'fails to address',
            'disappointing that', 'concerned about'
        ]
        
    def analyze_with_reasoning(self, text: str) -> Dict[str, Any]:
        """
        Analyze text with detailed reasoning and word highlighting
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with sentiment analysis and detailed reasoning
        """
        # Basic sentiment analysis
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Clean and tokenize text
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        # Find sentiment indicators
        found_words = self._find_sentiment_indicators(text_lower, words)
        found_phrases = self._find_sentiment_phrases(text_lower)
        
        # Calculate sentiment weights
        sentiment_score = self._calculate_weighted_sentiment(found_words, found_phrases, polarity)
        
        # Determine final sentiment
        final_sentiment = self._classify_sentiment(sentiment_score)
        
        # Generate detailed reasoning
        reasoning = self._generate_detailed_reasoning(
            text, final_sentiment, sentiment_score, found_words, found_phrases, polarity
        )
        
        # Create highlighted text
        highlighted_text = self._create_highlighted_text(text, found_words, found_phrases)
        
        return {
            'sentiment': final_sentiment,
            'polarity_score': round(sentiment_score, 3),
            'confidence': self._calculate_confidence(sentiment_score),
            'subjectivity_score': round(subjectivity, 3),
            'explanation': reasoning,
            'sentiment_reasoning': reasoning,  # For backward compatibility
            'key_indicators': {
                'positive_words': found_words['positive'],
                'negative_words': found_words['negative'],
                'neutral_words': found_words['neutral'],
                'intensifiers': found_words['intensifiers'],
                'negations': found_words['negations']
            },
            'highlighted_text': highlighted_text,
            'found_phrases': found_phrases,
            'analysis_details': {
                'word_count': len(words),
                'positive_weight': found_words['positive_weight'],
                'negative_weight': found_words['negative_weight'],
                'phrase_impact': found_phrases['impact']
            }
        }
    
    def _find_sentiment_indicators(self, text_lower: str, words: List[str]) -> Dict[str, Any]:
        """Find sentiment indicator words with their strengths"""
        found = {
            'positive': [],
            'negative': [],
            'neutral': [],
            'intensifiers': [],
            'negations': [],
            'positive_weight': 0,
            'negative_weight': 0
        }
        
        # Find positive words
        for strength, word_list in self.positive_words.items():
            weight = {'strong': 2, 'medium': 1, 'mild': 0.5}[strength]
            for word in word_list:
                if word in text_lower:
                    found['positive'].append(f"{word} ({strength})")
                    found['positive_weight'] += weight
        
        # Find negative words
        for strength, word_list in self.negative_words.items():
            weight = {'strong': 2, 'medium': 1, 'mild': 0.5}[strength]
            for word in word_list:
                if word in text_lower:
                    found['negative'].append(f"{word} ({strength})")
                    found['negative_weight'] += weight
        
        # Find neutral words
        for category, word_list in self.neutral_words.items():
            for word in word_list:
                if word in text_lower:
                    found['neutral'].append(f"{word} ({category})")
        
        # Find intensifiers and negations
        for word in self.intensifiers:
            if word in text_lower:
                found['intensifiers'].append(word)
        
        for word in self.negations:
            if word in text_lower:
                found['negations'].append(word)
        
        return found
    
    def _find_sentiment_phrases(self, text_lower: str) -> Dict[str, Any]:
        """Find sentiment-bearing phrases"""
        found_phrases = {
            'positive': [],
            'negative': [],
            'impact': 0
        }
        
        # Check positive phrases
        for phrase in self.positive_phrases:
            if phrase in text_lower:
                found_phrases['positive'].append(phrase)
                found_phrases['impact'] += 1
        
        # Check negative phrases
        for phrase in self.negative_phrases:
            if phrase in text_lower:
                found_phrases['negative'].append(phrase)
                found_phrases['impact'] -= 1
        
        return found_phrases
    
    def _calculate_weighted_sentiment(self, found_words: Dict, found_phrases: Dict, base_polarity: float) -> float:
        """Calculate weighted sentiment score"""
        # Start with TextBlob polarity
        score = base_polarity
        
        # Adjust based on word weights
        positive_impact = found_words['positive_weight'] * 0.2
        negative_impact = found_words['negative_weight'] * -0.2
        
        # Adjust based on phrases
        phrase_impact = found_phrases['impact'] * 0.3
        
        # Apply negation effects
        if found_words['negations']:
            score *= -0.5  # Flip partial sentiment if negations present
        
        # Apply intensifier effects
        if found_words['intensifiers']:
            score *= 1.3  # Amplify sentiment if intensifiers present
        
        final_score = score + positive_impact + negative_impact + phrase_impact
        
        # Clamp to [-1, 1] range
        return max(-1.0, min(1.0, final_score))
    
    def _classify_sentiment(self, score: float) -> str:
        """Classify sentiment based on score"""
        if score > 0.15:
            return "POSITIVE"
        elif score < -0.15:
            return "NEGATIVE"
        else:
            return "NEUTRAL"
    
    def _generate_detailed_reasoning(self, text: str, sentiment: str, score: float, 
                                   found_words: Dict, found_phrases: Dict, base_polarity: float) -> str:
        """Generate detailed reasoning explanation"""
        
        reasoning_parts = []
        
        # Main classification explanation
        reasoning_parts.append(f"**Sentiment Classification: {sentiment}**")
        reasoning_parts.append(f"Overall sentiment score: {score:.3f} (TextBlob base: {base_polarity:.3f})")
        
        # Word-level analysis
        if found_words['positive']:
            reasoning_parts.append(f"**✅ Positive indicators found:** {', '.join(found_words['positive'][:5])}")
            reasoning_parts.append(f"   • Positive word weight: +{found_words['positive_weight']:.1f}")
        
        if found_words['negative']:
            reasoning_parts.append(f"**❌ Negative indicators found:** {', '.join(found_words['negative'][:5])}")
            reasoning_parts.append(f"   • Negative word weight: -{found_words['negative_weight']:.1f}")
        
        if found_words['neutral']:
            reasoning_parts.append(f"**⚪ Neutral indicators found:** {', '.join(found_words['neutral'][:3])}")
        
        # Phrase-level analysis
        if found_phrases['positive']:
            reasoning_parts.append(f"**🎯 Positive phrases detected:** {', '.join(found_phrases['positive'])}")
        
        if found_phrases['negative']:
            reasoning_parts.append(f"**🎯 Negative phrases detected:** {', '.join(found_phrases['negative'])}")
        
        # Modifiers
        if found_words['intensifiers']:
            reasoning_parts.append(f"**🔥 Intensifiers present:** {', '.join(found_words['intensifiers'])} (amplifying effect)")
        
        if found_words['negations']:
            reasoning_parts.append(f"**🔄 Negations detected:** {', '.join(found_words['negations'])} (sentiment reversal effect)")
        
        # Final reasoning
        if sentiment == "POSITIVE":
            reasoning_parts.append("**📊 Analysis Result:** Text expresses overall positive sentiment through supportive language and favorable expressions.")
        elif sentiment == "NEGATIVE":
            reasoning_parts.append("**📊 Analysis Result:** Text expresses overall negative sentiment through critical language and unfavorable expressions.")
        else:
            reasoning_parts.append("**📊 Analysis Result:** Text maintains neutral tone with balanced or factual language.")
        
        return "\n\n".join(reasoning_parts)
    
    def _create_highlighted_text(self, text: str, found_words: Dict, found_phrases: Dict) -> str:
        """Create HTML highlighted version of text"""
        highlighted = text
        
        # Highlight positive words (green)
        for word_with_strength in found_words['positive']:
            word = word_with_strength.split(' (')[0]  # Remove strength indicator
            pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
            highlighted = pattern.sub(f'<span style="background-color: #90EE90; color: #006400; font-weight: bold;">{word}</span>', highlighted)
        
        # Highlight negative words (red)
        for word_with_strength in found_words['negative']:
            word = word_with_strength.split(' (')[0]  # Remove strength indicator
            pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
            highlighted = pattern.sub(f'<span style="background-color: #FFB6C1; color: #8B0000; font-weight: bold;">{word}</span>', highlighted)
        
        # Highlight intensifiers (orange)
        for word in found_words['intensifiers']:
            pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
            highlighted = pattern.sub(f'<span style="background-color: #FFD700; color: #FF8C00; font-weight: bold;">{word}</span>', highlighted)
        
        return highlighted
    
    def _calculate_confidence(self, score: float) -> str:
        """Calculate confidence level"""
        abs_score = abs(score)
        if abs_score > 0.5:
            return "High"
        elif abs_score > 0.2:
            return "Medium"
        else:
            return "Low"

# Global instance for easy access
enhanced_reasoner = EnhancedSentimentReasoner()

def analyze_text_with_enhanced_reasoning(text: str) -> Dict[str, Any]:
    """Analyze text with enhanced reasoning - main function for external use"""
    return enhanced_reasoner.analyze_with_reasoning(text)

if __name__ == "__main__":
    # Test the enhanced reasoning
    test_texts = [
        "I strongly support this new policy as it will improve transparency.",
        "The framework lacks clarity in several key areas and may create compliance challenges.",
        "This proposal seems reasonable and balanced in its approach."
    ]
    
    for text in test_texts:
        result = analyze_text_with_enhanced_reasoning(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Reasoning: {result['explanation']}")
        print("="*50)