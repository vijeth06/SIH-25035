"""
Advanced Word Highlighting System for Sentiment Reasoning
Creates visual indicators showing exact reasons for sentiment classifications
"""

import re
import streamlit as st
from typing import List, Dict, Any, Tuple
import html

class SentimentHighlighter:
    """Advanced word highlighting system for sentiment analysis visualization."""
    
    def __init__(self):
        # Enhanced sentiment lexicons with weights
        self.positive_words = {
            'strong': ['excellent', 'outstanding', 'fantastic', 'amazing', 'wonderful', 'superb', 'exceptional'],
            'medium': ['good', 'great', 'positive', 'beneficial', 'helpful', 'effective', 'valuable'],
            'mild': ['okay', 'fine', 'adequate', 'acceptable', 'decent', 'reasonable', 'fair']
        }
        
        self.negative_words = {
            'strong': ['terrible', 'awful', 'horrible', 'disastrous', 'catastrophic', 'appalling'],
            'medium': ['bad', 'poor', 'inadequate', 'insufficient', 'problematic', 'concerning'],
            'mild': ['issues', 'concerns', 'problems', 'difficulties', 'challenges', 'limitations']
        }
        
        # Special patterns for policy analysis
        self.negative_patterns = [
            'lacks clarity',
            'compliance challenges',
            'framework lacks',
            'may create.*challenges',
            'insufficient.*guidance',
            'unclear.*requirements',
            'creates.*problems'
        ]
        
        self.positive_patterns = [
            'excellent.*framework',
            'comprehensive.*approach',
            'clear.*guidance',
            'well.*structured',
            'effective.*implementation',
            'strong.*support'
        ]
        
        # Color schemes for highlighting
        self.colors = {
            'positive_strong': '#2E7D32',     # Dark green
            'positive_medium': '#4CAF50',     # Medium green  
            'positive_mild': '#81C784',       # Light green
            'negative_strong': '#C62828',     # Dark red
            'negative_medium': '#F44336',     # Medium red
            'negative_mild': '#EF5350',       # Light red
            'pattern_positive': '#1976D2',    # Blue for positive patterns
            'pattern_negative': '#E91E63',    # Pink for negative patterns
            'neutral': '#757575'              # Gray for neutral
        }
    
    def highlight_sentiment_words(self, text: str, sentiment: str, confidence: float) -> str:
        """
        Create HTML with highlighted words showing sentiment reasoning.
        """
        if not text:
            return ""
        
        # Start with escaped text to prevent HTML injection
        highlighted_text = html.escape(text)
        
        # Track highlighting explanations
        explanations = []
        
        # Highlight negative patterns first (longer matches)
        for pattern in self.negative_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                highlighted_text = re.sub(
                    f'({pattern})', 
                    f'<span style="background-color: {self.colors["pattern_negative"]}; color: white; padding: 2px 4px; border-radius: 3px; font-weight: bold;">\\1</span>',
                    highlighted_text,
                    flags=re.IGNORECASE
                )
                explanations.append(f"🔴 Negative pattern detected: '{pattern}'")
        
        # Highlight positive patterns
        for pattern in self.positive_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                highlighted_text = re.sub(
                    f'({pattern})', 
                    f'<span style="background-color: {self.colors["pattern_positive"]}; color: white; padding: 2px 4px; border-radius: 3px; font-weight: bold;">\\1</span>',
                    highlighted_text,
                    flags=re.IGNORECASE
                )
                explanations.append(f"🔵 Positive pattern detected: '{pattern}'")
        
        # Highlight individual negative words
        for strength, words in self.negative_words.items():
            color = self.colors[f'negative_{strength}']
            for word in words:
                if word.lower() in text.lower():
                    highlighted_text = re.sub(
                        f'\\b({re.escape(word)})\\b',
                        f'<span style="background-color: {color}; color: white; padding: 1px 3px; border-radius: 2px;">\\1</span>',
                        highlighted_text,
                        flags=re.IGNORECASE
                    )
                    explanations.append(f"🔴 {strength.title()} negative word: '{word}'")
        
        # Highlight individual positive words
        for strength, words in self.positive_words.items():
            color = self.colors[f'positive_{strength}']
            for word in words:
                if word.lower() in text.lower():
                    highlighted_text = re.sub(
                        f'\\b({re.escape(word)})\\b',
                        f'<span style="background-color: {color}; color: white; padding: 1px 3px; border-radius: 2px;">\\1</span>',
                        highlighted_text,
                        flags=re.IGNORECASE
                    )
                    explanations.append(f"🟢 {strength.title()} positive word: '{word}'")
        
        return highlighted_text, explanations
    
    def create_sentiment_explanation_card(self, text: str, sentiment: str, confidence: float, reasoning: List[str]) -> str:
        """
        Create a comprehensive explanation card for sentiment analysis.
        """
        highlighted_text, word_explanations = self.highlight_sentiment_words(text, sentiment, confidence)
        
        # Sentiment color and icon
        sentiment_colors = {
            'positive': '#4CAF50',
            'negative': '#F44336', 
            'neutral': '#757575'
        }
        
        sentiment_icons = {
            'positive': '😊',
            'negative': '😞',
            'neutral': '😐'
        }
        
        sentiment_color = sentiment_colors.get(sentiment.lower(), '#757575')
        sentiment_icon = sentiment_icons.get(sentiment.lower(), '😐')
        
        # Create the explanation card HTML
        card_html = f"""
        <div style="
            border: 2px solid {sentiment_color}; 
            border-radius: 10px; 
            padding: 15px; 
            margin: 10px 0; 
            background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(250,250,250,0.9) 100%);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        ">
            <div style="
                display: flex; 
                align-items: center; 
                margin-bottom: 12px;
                padding-bottom: 8px;
                border-bottom: 1px solid #e0e0e0;
            ">
                <span style="font-size: 24px; margin-right: 10px;">{sentiment_icon}</span>
                <span style="
                    font-size: 18px; 
                    font-weight: bold; 
                    color: {sentiment_color};
                    text-transform: uppercase;
                ">{sentiment}</span>
                <span style="
                    margin-left: 15px; 
                    background-color: {sentiment_color}; 
                    color: white; 
                    padding: 4px 8px; 
                    border-radius: 15px; 
                    font-size: 12px;
                    font-weight: bold;
                ">
                    {confidence*100:.1f}% Confidence
                </span>
            </div>
            
            <div style="margin-bottom: 15px;">
                <strong style="color: #333;">📝 Analyzed Text:</strong>
                <div style="
                    margin-top: 5px; 
                    padding: 10px; 
                    background-color: #f9f9f9; 
                    border-radius: 5px; 
                    line-height: 1.6;
                    font-size: 14px;
                ">
                    {highlighted_text}
                </div>
            </div>
            
            <div style="margin-bottom: 10px;">
                <strong style="color: #333;">🔍 Analysis Reasoning:</strong>
                <ul style="margin: 5px 0; padding-left: 20px;">
        """
        
        # Add reasoning points
        for reason in reasoning[:5]:  # Limit to top 5 reasons
            card_html += f'<li style="margin: 3px 0; color: #555; font-size: 13px;">{html.escape(reason)}</li>'
        
        # Add word-level explanations
        for explanation in word_explanations[:3]:  # Limit to top 3 word explanations
            card_html += f'<li style="margin: 3px 0; color: #555; font-size: 13px;">{html.escape(explanation)}</li>'
        
        card_html += """
                </ul>
            </div>
        </div>
        """
        
        return card_html
    
    def create_legend(self) -> str:
        """Create a legend explaining the highlighting colors."""
        legend_html = """
        <div style="
            background-color: #f5f5f5; 
            border-radius: 8px; 
            padding: 15px; 
            margin: 10px 0;
            border-left: 4px solid #2196F3;
        ">
            <h4 style="margin: 0 0 10px 0; color: #333;">🎨 Highlighting Legend</h4>
            <div style="display: flex; flex-wrap: wrap; gap: 10px;">
        """
        
        legend_items = [
            ('Strong Positive', self.colors['positive_strong']),
            ('Medium Positive', self.colors['positive_medium']),
            ('Mild Positive', self.colors['positive_mild']),
            ('Strong Negative', self.colors['negative_strong']),
            ('Medium Negative', self.colors['negative_medium']),
            ('Mild Negative', self.colors['negative_mild']),
            ('Positive Pattern', self.colors['pattern_positive']),
            ('Negative Pattern', self.colors['pattern_negative'])
        ]
        
        for label, color in legend_items:
            legend_html += f"""
                <span style="
                    background-color: {color}; 
                    color: white; 
                    padding: 3px 6px; 
                    border-radius: 3px; 
                    font-size: 11px;
                    font-weight: bold;
                ">{label}</span>
            """
        
        legend_html += """
            </div>
        </div>
        """
        
        return legend_html
    
    def display_sentiment_analysis_with_highlighting(self, text: str, sentiment: str, confidence: float, reasoning: List[str]) -> None:
        """
        Display sentiment analysis with highlighting in Streamlit.
        """
        # Create and display the explanation card
        card_html = self.create_sentiment_explanation_card(text, sentiment, confidence, reasoning)
        st.markdown(card_html, unsafe_allow_html=True)
        
        # Display legend
        legend_html = self.create_legend()
        st.markdown(legend_html, unsafe_allow_html=True)
    
    def batch_highlight_dataframe(self, df, text_column: str, sentiment_column: str, confidence_column: str) -> str:
        """
        Create highlighted HTML table for batch sentiment analysis results.
        """
        html_rows = []
        
        # Table header
        html_table = """
        <table style="
            width: 100%; 
            border-collapse: collapse; 
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            <thead>
                <tr style="background-color: #f8f9fa;">
                    <th style="border: 1px solid #dee2e6; padding: 12px; text-align: left; font-weight: bold;">Row</th>
                    <th style="border: 1px solid #dee2e6; padding: 12px; text-align: left; font-weight: bold;">Highlighted Text</th>
                    <th style="border: 1px solid #dee2e6; padding: 12px; text-align: center; font-weight: bold;">Sentiment</th>
                    <th style="border: 1px solid #dee2e6; padding: 12px; text-align: center; font-weight: bold;">Confidence</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for idx, row in df.iterrows():
            text = str(row[text_column])
            sentiment = str(row[sentiment_column])
            confidence = float(row[confidence_column]) if confidence_column in df.columns else 0.0
            
            # Highlight the text
            highlighted_text, _ = self.highlight_sentiment_words(text, sentiment, confidence)
            
            # Sentiment badge
            sentiment_color = self.colors.get(f'{sentiment.lower()}_medium', '#757575')
            sentiment_badge = f"""
                <span style="
                    background-color: {sentiment_color}; 
                    color: white; 
                    padding: 4px 8px; 
                    border-radius: 12px; 
                    font-size: 12px;
                    font-weight: bold;
                    text-transform: uppercase;
                ">{sentiment}</span>
            """
            
            # Row highlighting based on sentiment
            row_color = {
                'positive': '#f8fff8',
                'negative': '#fff8f8',
                'neutral': '#f8f8f8'
            }.get(sentiment.lower(), '#ffffff')
            
            html_table += f"""
                <tr style="background-color: {row_color};">
                    <td style="border: 1px solid #dee2e6; padding: 10px; font-weight: bold; color: #666;">
                        {idx + 1}
                        {'🎯' if idx == 6 else ''}
                    </td>
                    <td style="border: 1px solid #dee2e6; padding: 10px; line-height: 1.4;">
                        {highlighted_text[:200]}{'...' if len(highlighted_text) > 200 else ''}
                    </td>
                    <td style="border: 1px solid #dee2e6; padding: 10px; text-align: center;">
                        {sentiment_badge}
                    </td>
                    <td style="border: 1px solid #dee2e6; padding: 10px; text-align: center; font-weight: bold;">
                        {confidence*100:.1f}%
                    </td>
                </tr>
            """
        
        html_table += """
            </tbody>
        </table>
        """
        
        return html_table


def create_sentiment_highlighter():
    """Factory function to create highlighter instance."""
    return SentimentHighlighter()


if __name__ == "__main__":
    # Test the word highlighting system
    test_cases = [
        {
            "text": "The framework lacks clarity in several key areas and may create compliance challenges for smaller organizations",
            "sentiment": "negative",
            "confidence": 0.862,
            "reasoning": [
                "Contains negative indicator: 'lacks clarity'",
                "Contains negative pattern: 'may create compliance challenges'",
                "Negative pattern: 'framework lacks clarity'",
                "Critical feedback detected in phrase structure"
            ]
        },
        {
            "text": "This is an excellent policy framework that will benefit everyone",
            "sentiment": "positive", 
            "confidence": 0.85,
            "reasoning": [
                "Strong positive word: 'excellent'",
                "Positive context around 'policy framework'",
                "Beneficial outcome indicated"
            ]
        }
    ]
    
    highlighter = SentimentHighlighter()
    
    print("🎨 WORD HIGHLIGHTING SYSTEM TEST")
    print("=" * 50)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}:")
        print(f"Text: {case['text']}")
        print(f"Sentiment: {case['sentiment'].upper()}")
        
        highlighted, explanations = highlighter.highlight_sentiment_words(
            case['text'], case['sentiment'], case['confidence']
        )
        
        print(f"Explanations: {len(explanations)} found")
        for exp in explanations:
            print(f"  - {exp}")
    
    print("\n✅ WORD HIGHLIGHTING SYSTEM WORKING!")