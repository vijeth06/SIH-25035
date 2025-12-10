"""
Enhanced Text Summarization with Local Fallback
"""
import re
import nltk
from collections import Counter, defaultdict
from typing import List, Dict, Any
import numpy as np
from textblob import TextBlob
import pandas as pd

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer

class EnhancedTextSummarizer:
    """Enhanced text summarization with multiple algorithms"""
    
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        try:
            self.lemmatizer = WordNetLemmatizer()
        except:
            self.lemmatizer = None
    
    def summarize_text(self, text: str, method: str = "extractive", max_sentences: int = 3) -> Dict[str, Any]:
        """
        Summarize text using various methods
        
        Args:
            text: Input text to summarize
            method: Summarization method ('extractive', 'frequency', 'sentiment')
            max_sentences: Maximum sentences in summary
            
        Returns:
            Dictionary with summary and analysis
        """
        if not text or len(text.strip()) < 50:
            return {
                'summary': text,
                'method': method,
                'original_length': len(text.split()),
                'summary_length': len(text.split()),
                'compression_ratio': 1.0,
                'key_points': [],
                'sentiment_overview': 'neutral'
            }
        
        # Clean and prepare text
        sentences = self._clean_and_split_sentences(text)
        
        if len(sentences) <= max_sentences:
            return {
                'summary': text,
                'method': method,
                'original_length': len(text.split()),
                'summary_length': len(text.split()),
                'compression_ratio': 1.0,
                'key_points': sentences,
                'sentiment_overview': self._get_overall_sentiment(text)
            }
        
        # Choose summarization method
        if method == "extractive":
            summary_sentences = self._extractive_summarization(sentences, max_sentences)
        elif method == "frequency":
            summary_sentences = self._frequency_based_summarization(sentences, max_sentences)
        elif method == "sentiment":
            summary_sentences = self._sentiment_based_summarization(sentences, max_sentences)
        else:
            summary_sentences = self._extractive_summarization(sentences, max_sentences)
        
        summary_text = ' '.join(summary_sentences)
        
        return {
            'summary': summary_text,
            'method': method,
            'original_length': len(text.split()),
            'summary_length': len(summary_text.split()),
            'compression_ratio': round(len(summary_text.split()) / len(text.split()), 2),
            'key_points': summary_sentences,
            'sentiment_overview': self._get_overall_sentiment(summary_text),
            'original_sentences': len(sentences),
            'summary_sentences': len(summary_sentences)
        }
    
    def summarize_dataframe_column(self, df: pd.DataFrame, column: str, max_sentences: int = 5) -> Dict[str, Any]:
        """
        Summarize all text in a DataFrame column
        
        Args:
            df: Pandas DataFrame
            column: Column name to summarize
            max_sentences: Maximum sentences in final summary
            
        Returns:
            Dictionary with comprehensive summary
        """
        if column not in df.columns:
            return {'error': f'Column {column} not found in DataFrame'}
        
        # Combine all text in the column
        all_text = ' '.join(df[column].astype(str).tolist())
        
        # Get basic statistics
        stats = {
            'total_entries': len(df),
            'non_empty_entries': df[column].notna().sum(),
            'avg_length': df[column].str.len().mean() if df[column].notna().any() else 0,
            'total_words': len(all_text.split())
        }
        
        # Analyze sentiment distribution
        sentiment_counts = defaultdict(int)
        sentiment_examples = defaultdict(list)
        
        for idx, text in enumerate(df[column].astype(str)):
            if pd.notna(text) and len(text.strip()) > 0:
                sentiment = self._get_text_sentiment(text)
                sentiment_counts[sentiment] += 1
                if len(sentiment_examples[sentiment]) < 2:  # Keep top 2 examples per sentiment
                    sentiment_examples[sentiment].append(text[:100] + "..." if len(text) > 100 else text)
        
        # Generate comprehensive summary
        main_summary = self.summarize_text(all_text, method="extractive", max_sentences=max_sentences)
        
        # Find key themes/topics
        key_themes = self._extract_key_themes(all_text)
        
        return {
            'column_name': column,
            'statistics': stats,
            'main_summary': main_summary['summary'],
            'sentiment_distribution': dict(sentiment_counts),
            'sentiment_examples': dict(sentiment_examples),
            'key_themes': key_themes,
            'compression_info': {
                'original_entries': stats['total_entries'],
                'original_words': stats['total_words'],
                'summary_words': main_summary['summary_length'],
                'compression_ratio': main_summary['compression_ratio']
            }
        }
    
    def _clean_and_split_sentences(self, text: str) -> List[str]:
        """Clean text and split into sentences"""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        try:
            sentences = sent_tokenize(text)
        except:
            # Fallback sentence splitting
            sentences = re.split(r'[.!?]+', text)
        
        # Clean sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Filter out very short sentences
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _extractive_summarization(self, sentences: List[str], max_sentences: int) -> List[str]:
        """Extractive summarization using sentence scoring"""
        # Calculate word frequencies
        word_freq = self._calculate_word_frequencies(' '.join(sentences))
        
        # Score sentences
        sentence_scores = {}
        for sentence in sentences:
            words = self._tokenize_sentence(sentence.lower())
            score = 0
            word_count = 0
            
            for word in words:
                if word in word_freq:
                    score += word_freq[word]
                    word_count += 1
            
            if word_count > 0:
                sentence_scores[sentence] = score / word_count
            else:
                sentence_scores[sentence] = 0
        
        # Select top sentences
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:max_sentences]
        
        # Return in original order
        selected_sentences = [sent[0] for sent in top_sentences]
        return self._order_sentences_by_appearance(sentences, selected_sentences)
    
    def _frequency_based_summarization(self, sentences: List[str], max_sentences: int) -> List[str]:
        """Frequency-based summarization"""
        # Calculate TF scores for each sentence
        sentence_scores = {}
        all_words = self._tokenize_sentence(' '.join(sentences).lower())
        word_freq = Counter(word for word in all_words if word not in self.stop_words)
        
        for sentence in sentences:
            words = self._tokenize_sentence(sentence.lower())
            score = sum(word_freq[word] for word in words if word in word_freq)
            sentence_scores[sentence] = score
        
        # Select top sentences
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:max_sentences]
        selected_sentences = [sent[0] for sent in top_sentences]
        return self._order_sentences_by_appearance(sentences, selected_sentences)
    
    def _sentiment_based_summarization(self, sentences: List[str], max_sentences: int) -> List[str]:
        """Sentiment-based summarization (select most opinionated sentences)"""
        sentence_scores = {}
        
        for sentence in sentences:
            blob = TextBlob(sentence)
            # Score based on sentiment strength (absolute polarity) and subjectivity
            sentiment_strength = abs(blob.sentiment.polarity) + blob.sentiment.subjectivity
            sentence_scores[sentence] = sentiment_strength
        
        # Select sentences with highest sentiment scores
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:max_sentences]
        selected_sentences = [sent[0] for sent in top_sentences]
        return self._order_sentences_by_appearance(sentences, selected_sentences)
    
    def _calculate_word_frequencies(self, text: str) -> Dict[str, float]:
        """Calculate normalized word frequencies"""
        words = self._tokenize_sentence(text.lower())
        word_count = Counter(word for word in words if word not in self.stop_words)
        
        # Normalize frequencies
        max_freq = max(word_count.values()) if word_count else 1
        return {word: count/max_freq for word, count in word_count.items()}
    
    def _tokenize_sentence(self, sentence: str) -> List[str]:
        """Tokenize sentence into words"""
        try:
            words = word_tokenize(sentence)
        except:
            # Fallback tokenization
            words = re.findall(r'\b\w+\b', sentence)
        
        # Lemmatize if available
        if self.lemmatizer:
            try:
                words = [self.lemmatizer.lemmatize(word) for word in words]
            except:
                pass
        
        return [word for word in words if len(word) > 2]
    
    def _order_sentences_by_appearance(self, original_sentences: List[str], selected_sentences: List[str]) -> List[str]:
        """Order selected sentences by their appearance in original text"""
        ordered = []
        for original in original_sentences:
            if original in selected_sentences:
                ordered.append(original)
        return ordered
    
    def _get_overall_sentiment(self, text: str) -> str:
        """Get overall sentiment of text"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            return 'positive'
        elif polarity < -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def _get_text_sentiment(self, text: str) -> str:
        """Get sentiment of individual text"""
        return self._get_overall_sentiment(text)
    
    def _extract_key_themes(self, text: str, max_themes: int = 5) -> List[Dict[str, Any]]:
        """Extract key themes from text"""
        words = self._tokenize_sentence(text.lower())
        word_freq = Counter(word for word in words if word not in self.stop_words and len(word) > 3)
        
        # Get top words as themes
        top_words = word_freq.most_common(max_themes)
        
        themes = []
        for word, count in top_words:
            themes.append({
                'theme': word.title(),
                'frequency': count,
                'relevance': round(count / len(words), 3)
            })
        
        return themes

# Global instance
enhanced_summarizer = EnhancedTextSummarizer()

def summarize_text_enhanced(text: str, method: str = "extractive", max_sentences: int = 3) -> Dict[str, Any]:
    """Main function for text summarization"""
    return enhanced_summarizer.summarize_text(text, method, max_sentences)

def summarize_column_data(df: pd.DataFrame, column: str, max_sentences: int = 5) -> Dict[str, Any]:
    """Main function for column summarization"""
    return enhanced_summarizer.summarize_dataframe_column(df, column, max_sentences)

if __name__ == "__main__":
    # Test the summarizer
    test_text = """
    I strongly support this new policy initiative as it provides excellent transparency measures.
    The framework addresses key concerns about government accountability and citizen participation.
    However, some aspects of the implementation plan lack clarity and may create challenges.
    Overall, this is a positive step forward for democratic governance and public engagement.
    The proposed changes will benefit all stakeholders and improve service delivery.
    """
    
    result = summarize_text_enhanced(test_text, method="extractive", max_sentences=2)
    print("Summary:", result['summary'])
    print("Compression ratio:", result['compression_ratio'])
    print("Sentiment:", result['sentiment_overview'])