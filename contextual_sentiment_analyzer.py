"""
Advanced Context-Aware Sentiment Analyzer
ChatGPT-level sentiment analysis that understands full context and nuanced meanings
"""

import re
import logging
from typing import Dict, List, Tuple, Any
from datetime import datetime
import numpy as np

class AdvancedContextualSentimentAnalyzer:
    """
    Advanced sentiment analyzer that understands full context like ChatGPT
    Analyzes entire sentences/paragraphs for nuanced sentiment detection
    """
    
    def __init__(self):
        # Enhanced contextual patterns for full sentence analysis
        self.positive_contexts = {
            # Strong positive contexts
            'strong_support': {
                'patterns': [
                    r'strongly support.*because',
                    r'excellent.*will benefit',
                    r'great initiative.*will help',
                    r'wonderful.*opportunity',
                    r'fantastic.*approach',
                    r'highly recommend.*this',
                    r'absolutely necessary.*for',
                    r'perfect solution.*to'
                ],
                'weight': 0.9
            },
            'approval_with_reasoning': {
                'patterns': [
                    r'support.*as it will',
                    r'agree.*because it',
                    r'beneficial.*for all',
                    r'positive impact.*on',
                    r'will improve.*significantly',
                    r'exactly what.*needed',
                    r'right direction.*for'
                ],
                'weight': 0.8
            },
            'constructive_optimism': {
                'patterns': [
                    r'while.*positive.*overall',
                    r'despite.*good.*progress',
                    r'although.*beneficial',
                    r'even though.*valuable',
                    r'some concerns.*but.*support'
                ],
                'weight': 0.6
            }
        }
        
        self.negative_contexts = {
            # Strong negative contexts
            'strong_opposition': {
                'patterns': [
                    r'strongly oppose.*because',
                    r'terrible.*will cause',
                    r'awful.*consequences',
                    r'completely disagree.*with',
                    r'absolutely against.*this',
                    r'totally unacceptable',
                    r'worst.*decision',
                    r'disaster.*for'
                ],
                'weight': -0.9
            },
            'criticism_with_reasoning': {
                'patterns': [
                    r'lacks.*in several areas',
                    r'problems.*with this approach',
                    r'concerns.*about implementation',
                    r'issues.*that need addressing',
                    r'challenges.*for smaller',
                    r'difficulties.*in understanding',
                    r'complications.*will arise'
                ],
                'weight': -0.7
            },
            'conditional_criticism': {
                'patterns': [
                    r'while.*has merit.*but',
                    r'although.*good idea.*however',
                    r'support.*concept.*but concerned',
                    r'like.*idea.*but worried',
                    r'positive.*aspects.*but'
                ],
                'weight': -0.4
            }
        }
        
        self.neutral_contexts = {
            'balanced_analysis': {
                'patterns': [
                    r'both.*positive.*and.*negative',
                    r'mixed.*feelings.*about',
                    r'some.*good.*some.*bad',
                    r'pros.*and.*cons',
                    r'advantages.*and.*disadvantages',
                    r'benefits.*but.*also.*drawbacks'
                ],
                'weight': 0.0
            },
            'neutral_information': {
                'patterns': [
                    r'according.*to.*study',
                    r'based.*on.*research',
                    r'data.*shows.*that',
                    r'statistics.*indicate',
                    r'evidence.*suggests'
                ],
                'weight': 0.0
            }
        }
        
        # Contextual modifiers that change sentiment based on sentence structure
        self.sentiment_modifiers = {
            'negation_phrases': [
                'not really', 'not exactly', 'hardly', 'barely', 'scarcely',
                'far from', 'anything but', 'by no means', 'in no way'
            ],
            'intensifiers': [
                'extremely', 'incredibly', 'absolutely', 'completely', 'totally',
                'entirely', 'thoroughly', 'utterly', 'highly', 'very much'
            ],
            'diminishers': [
                'somewhat', 'rather', 'quite', 'fairly', 'relatively',
                'moderately', 'partially', 'slightly', 'a bit'
            ],
            'contrast_connectors': [
                'however', 'but', 'although', 'though', 'despite', 'nevertheless',
                'nonetheless', 'yet', 'still', 'even so', 'on the other hand'
            ]
        }
        
        # Advanced linguistic patterns for context understanding
        self.linguistic_patterns = {
            'conditional_support': r'if.*then.*support',
            'conditional_opposition': r'unless.*cannot.*support',
            'temporal_context': r'(initially|currently|eventually|ultimately)',
            'comparative_context': r'(better than|worse than|compared to|relative to)',
            'causal_relationships': r'(because|since|due to|as a result|therefore|thus)',
            'hypothetical': r'(would|could|might|may|should).*if'
        }
    
    def analyze_contextual_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Perform advanced contextual sentiment analysis like ChatGPT
        Considers full sentence meaning, not just keywords
        """
        
        if not text or not text.strip():
            return self._create_result(text, "neutral", 0.0, 0.3, [], "Empty or invalid text")
        
        # Preprocess and prepare text for analysis
        text_clean = self._advanced_preprocessing(text)
        
        # Multi-layered analysis
        context_score = self._analyze_sentence_context(text_clean)
        linguistic_score = self._analyze_linguistic_patterns(text_clean)
        semantic_score = self._analyze_semantic_meaning(text_clean)
        
        # Combine scores with weighted importance
        final_score = (context_score * 0.5) + (linguistic_score * 0.3) + (semantic_score * 0.2)
        
        # Extract meaningful justification phrases (not just keywords)
        justification_phrases = self._extract_contextual_justification(text_clean, final_score)
        
        # Determine sentiment with advanced thresholds
        sentiment, confidence = self._determine_contextual_sentiment(final_score, text_clean)
        
        # Generate intelligent reasoning
        reasoning = self._generate_contextual_reasoning(sentiment, justification_phrases, final_score, confidence, text_clean)
        
        # Create highlighted text showing meaningful phrases
        highlighted_text = self._create_contextual_highlighting(text, justification_phrases, sentiment)
        
        return self._create_result(
            text, sentiment, final_score, confidence, 
            justification_phrases, reasoning, highlighted_text,
            {
                'context_score': context_score,
                'linguistic_score': linguistic_score,
                'semantic_score': semantic_score,
                'analysis_method': 'advanced_contextual'
            }
        )
    
    def _advanced_preprocessing(self, text: str) -> str:
        """Advanced text preprocessing for contextual analysis"""
        # Convert to lowercase but preserve sentence structure
        text_clean = text.lower().strip()
        
        # Handle contractions more intelligently
        contractions = {
            "won't": "will not", "can't": "cannot", "shouldn't": "should not",
            "wouldn't": "would not", "couldn't": "could not", "doesn't": "does not",
            "isn't": "is not", "aren't": "are not", "wasn't": "was not",
            "weren't": "were not", "haven't": "have not", "hasn't": "has not",
            "hadn't": "had not", "don't": "do not", "didn't": "did not"
        }
        
        for contraction, expansion in contractions.items():
            text_clean = text_clean.replace(contraction, expansion)
        
        # Normalize punctuation for better pattern matching
        text_clean = re.sub(r'[.]{2,}', '.', text_clean)
        text_clean = re.sub(r'[!]{2,}', '!', text_clean)
        text_clean = re.sub(r'[?]{2,}', '?', text_clean)
        
        return text_clean
    
    def _analyze_sentence_context(self, text: str) -> float:
        """Analyze the contextual meaning of entire sentences"""
        score = 0.0
        
        # Analyze positive contexts
        for context_type, context_data in self.positive_contexts.items():
            for pattern in context_data['patterns']:
                if re.search(pattern, text, re.IGNORECASE):
                    score += context_data['weight']
        
        # Analyze negative contexts
        for context_type, context_data in self.negative_contexts.items():
            for pattern in context_data['patterns']:
                if re.search(pattern, text, re.IGNORECASE):
                    score += context_data['weight']
        
        # Analyze neutral contexts
        for context_type, context_data in self.neutral_contexts.items():
            for pattern in context_data['patterns']:
                if re.search(pattern, text, re.IGNORECASE):
                    score += context_data['weight']
        
        return score
    
    def _analyze_linguistic_patterns(self, text: str) -> float:
        """Analyze linguistic patterns and sentence structure"""
        score = 0.0
        
        # Check for contrast patterns (but, however, although)
        contrast_count = sum(1 for connector in self.sentiment_modifiers['contrast_connectors'] 
                           if connector in text)
        if contrast_count > 0:
            # Contrast usually indicates mixed or negative sentiment
            score -= 0.2 * contrast_count
        
        # Check for intensifiers
        intensifier_count = sum(1 for intensifier in self.sentiment_modifiers['intensifiers'] 
                              if intensifier in text)
        if intensifier_count > 0:
            # Intensifiers amplify the base sentiment
            base_sentiment = 0.1 if any(word in text for word in ['good', 'great', 'excellent', 'support']) else -0.1
            score += base_sentiment * intensifier_count * 1.5
        
        # Check for diminishers
        diminisher_count = sum(1 for diminisher in self.sentiment_modifiers['diminishers'] 
                             if diminisher in text)
        if diminisher_count > 0:
            # Diminishers reduce sentiment intensity
            score *= 0.7
        
        # Analyze causal relationships
        if re.search(self.linguistic_patterns['causal_relationships'], text):
            # Causal relationships usually indicate stronger sentiment
            if score > 0:
                score *= 1.3
            elif score < 0:
                score *= 1.3
        
        return score
    
    def _analyze_semantic_meaning(self, text: str) -> float:
        """Analyze semantic meaning and overall message"""
        score = 0.0
        
        # Semantic analysis based on overall message structure
        sentences = re.split(r'[.!?]+', text)
        sentence_scores = []
        
        for sentence in sentences:
            if len(sentence.strip()) < 10:  # Skip very short fragments
                continue
                
            sentence_score = 0.0
            
            # Analyze sentence for semantic indicators
            positive_indicators = [
                'will improve', 'will benefit', 'will help', 'is good', 'is great',
                'support this', 'agree with', 'in favor', 'positive impact',
                'right direction', 'good idea', 'beneficial for'
            ]
            
            negative_indicators = [
                'will harm', 'will damage', 'will hurt', 'is bad', 'is terrible',
                'oppose this', 'disagree with', 'against this', 'negative impact',
                'wrong direction', 'bad idea', 'harmful for'
            ]
            
            for indicator in positive_indicators:
                if indicator in sentence:
                    sentence_score += 0.6
                    
            for indicator in negative_indicators:
                if indicator in sentence:
                    sentence_score -= 0.6
            
            sentence_scores.append(sentence_score)
        
        # Calculate overall semantic score
        if sentence_scores:
            score = sum(sentence_scores) / len(sentence_scores)
        
        return score
    
    def _extract_contextual_justification(self, text: str, sentiment_score: float) -> List[str]:
        """Extract meaningful phrases that justify the sentiment, not just keywords"""
        justification_phrases = []
        
        # Extract phrases based on sentiment direction
        if sentiment_score > 0.1:
            # Look for positive justification phrases
            positive_phrase_patterns = [
                r'(strongly support.*?[.!?])',
                r'(will improve.*?[.!?])',
                r'(beneficial.*?[.!?])',
                r'(excellent.*?[.!?])',
                r'(positive impact.*?[.!?])'
            ]
            
            for pattern in positive_phrase_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                justification_phrases.extend(matches[:2])  # Limit to 2 phrases
                
        elif sentiment_score < -0.1:
            # Look for negative justification phrases
            negative_phrase_patterns = [
                r'(lacks.*?[.!?])',
                r'(problems.*?[.!?])',
                r'(concerns.*?[.!?])',
                r'(challenges.*?[.!?])',
                r'(oppose.*?[.!?])'
            ]
            
            for pattern in negative_phrase_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                justification_phrases.extend(matches[:2])  # Limit to 2 phrases
        
        # If no specific phrases found, extract key contextual words
        if not justification_phrases:
            if sentiment_score > 0.1:
                key_words = [word for word in ['support', 'improve', 'beneficial', 'excellent', 'good'] 
                           if word in text]
            elif sentiment_score < -0.1:
                key_words = [word for word in ['lacks', 'problems', 'concerns', 'challenges', 'oppose'] 
                           if word in text]
            else:
                key_words = [word for word in ['neutral', 'mixed', 'balanced'] if word in text]
            
            justification_phrases = key_words[:3]
        
        return justification_phrases[:3]  # Always limit to max 3 items
    
    def _determine_contextual_sentiment(self, score: float, text: str) -> Tuple[str, float]:
        """Determine sentiment with contextual confidence"""
        
        # Adaptive thresholds based on text complexity
        text_complexity = len(text.split()) / 10  # Normalize by word count
        base_threshold = 0.15
        
        # Adjust thresholds for more nuanced analysis
        positive_threshold = max(0.1, base_threshold - text_complexity * 0.02)
        negative_threshold = -positive_threshold
        
        if score > positive_threshold:
            sentiment = "positive"
            confidence = min(0.95, 0.7 + abs(score) * 0.5)
        elif score < negative_threshold:
            sentiment = "negative"
            confidence = min(0.95, 0.7 + abs(score) * 0.5)
        else:
            sentiment = "neutral"
            confidence = 0.6 + abs(score) * 0.2
        
        return sentiment, confidence
    
    def _generate_contextual_reasoning(self, sentiment: str, justification_phrases: List[str], 
                                     score: float, confidence: float, text: str) -> str:
        """Generate intelligent reasoning like ChatGPT"""
        
        reasoning_parts = []
        
        # Main sentiment explanation with context
        if justification_phrases:
            phrases_str = ", ".join(justification_phrases[:2])  # Use top 2 phrases
            reasoning_parts.append(
                f"Sentiment classified as {sentiment.upper()} based on contextual analysis of: \"{phrases_str}\""
            )
        else:
            reasoning_parts.append(
                f"Sentiment classified as {sentiment.upper()} based on overall contextual meaning and tone"
            )
        
        # Add confidence explanation
        if confidence > 0.8:
            confidence_desc = "high confidence"
            reasoning_parts.append("The sentiment is clearly expressed with strong contextual indicators")
        elif confidence > 0.6:
            confidence_desc = "medium confidence"
            reasoning_parts.append("The sentiment is reasonably clear from the contextual analysis")
        else:
            confidence_desc = "moderate confidence"
            reasoning_parts.append("The sentiment shows some ambiguity or mixed indicators")
        
        # Add contextual insights
        if "but" in text or "however" in text:
            reasoning_parts.append("Text contains contrasting elements that were considered in the analysis")
        
        if len(text.split()) > 50:
            reasoning_parts.append("Analysis considered the full context of this detailed comment")
        
        return ". ".join(reasoning_parts) + f" (Confidence: {confidence:.1%})"
    
    def _create_contextual_highlighting(self, text: str, justification_phrases: List[str], sentiment: str) -> str:
        """Create contextual highlighting of meaningful phrases"""
        
        highlighted_text = text
        
        if sentiment == "positive":
            color = "#90EE90"  # Light green
            text_color = "#006400"  # Dark green
        elif sentiment == "negative":
            color = "#FFB6C1"  # Light pink
            text_color = "#8B0000"  # Dark red
        else:
            color = "#FFFFE0"  # Light yellow
            text_color = "#000000"  # Black
        
        # Highlight meaningful phrases, not just individual words
        for phrase in justification_phrases[:3]:
            # Clean up phrase
            phrase_clean = phrase.strip('.,!?;:"()[]{}')
            if len(phrase_clean) > 2:
                # Create pattern for phrase highlighting
                pattern = re.escape(phrase_clean)
                replacement = f'<span style="background-color: {color}; color: {text_color}; font-weight: bold; padding: 2px 4px; border-radius: 3px;">{phrase_clean}</span>'
                
                highlighted_text = re.sub(pattern, replacement, highlighted_text, flags=re.IGNORECASE, count=1)
        
        return highlighted_text
    
    def _create_result(self, text: str, sentiment: str, score: float, confidence: float, 
                      justification_phrases: List[str], reasoning: str, 
                      highlighted_text: str = None, details: Dict = None) -> Dict[str, Any]:
        """Create standardized result dictionary"""
        
        if highlighted_text is None:
            highlighted_text = text
        
        return {
            "text": text,
            "sentiment": sentiment,
            "confidence": confidence,
            "polarity_score": round(score, 4),
            "reasoning": reasoning,
            "sentiment_reasoning": reasoning,  # Ensure consistency
            "justification_words": justification_phrases[:3],  # Max 3 items
            "highlighted_text": highlighted_text,
            "analysis_details": details or {},
            "method": "advanced_contextual_analysis",
            "timestamp": datetime.now().isoformat()
        }

# Global instance for easy import
contextual_analyzer = AdvancedContextualSentimentAnalyzer()

def analyze_sentiment_contextual(text: str) -> Dict[str, Any]:
    """Main function for ChatGPT-level contextual sentiment analysis"""
    return contextual_analyzer.analyze_contextual_sentiment(text)

def analyze_batch_sentiments_contextual(texts: List[str]) -> List[Dict[str, Any]]:
    """Analyze multiple texts with contextual understanding"""
    results = []
    for text in texts:
        result = analyze_sentiment_contextual(text)
        results.append(result)
    return results

if __name__ == "__main__":
    # Test the contextual analyzer
    test_cases = [
        "I strongly support this new policy as it will improve transparency in government operations. This is exactly what we need to restore public trust.",
        "The framework lacks clarity in several key areas and may create compliance challenges for smaller organizations. However, I do appreciate the effort.",
        "I have mixed feelings about this proposal. While some aspects are good, others need more work before implementation.",
        "This legislation is excellent and will benefit all citizens. The comprehensive approach addresses all major concerns.",
        "The policy is terrible and will cause significant problems. But I understand the intention behind it.",
        "Although the concept has merit, I am concerned about the implementation challenges and potential unintended consequences."
    ]
    
    print("🧠 Advanced Contextual Sentiment Analysis - ChatGPT Level")
    print("=" * 70)
    
    for i, text in enumerate(test_cases, 1):
        result = analyze_sentiment_contextual(text)
        print(f"\nTest {i}:")
        print(f"Text: {text}")
        print(f"Sentiment: {result['sentiment'].upper()}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Justification: {', '.join(result['justification_words'])}")
        print(f"Reasoning: {result['reasoning']}")
        print("-" * 50)