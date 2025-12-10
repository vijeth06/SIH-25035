"""
Simplified visualization service with robust fallback handling.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import logging
import asyncio
import base64
import io

logger = logging.getLogger(__name__)


class SimplifiedVisualizationService:
    """Simplified visualization service with robust fallback handling."""

    def __init__(self):
        # Common stop words
        self.stop_words = {
            'a', 'an', 'the', 'and', 'or', 'but', 'if', 'while', 'at', 'by', 'for', 'with',
            'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
            'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
            'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
            'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'i', 'me',
            'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
            'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is',
            'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
            'do', 'does', 'did', 'doing', 'would', 'could', 'should', 'may', 'might',
            'must', 'shall', 'will', 'can'
        }

        logger.info("✅ Simplified visualization service initialized")

    async def prepare_tokens(self, texts: List[str], min_len: int = 3) -> List[str]:
        """Prepare tokens from texts."""
        all_tokens = []

        for text in texts:
            if not text:
                continue

            # Simple tokenization
            words = re.findall(r'\b\w+\b', text.lower())

            # Filter tokens
            tokens = [
                word for word in words
                if len(word) >= min_len
                and word not in self.stop_words
                and word.isalpha()
            ]

            all_tokens.extend(tokens)

        return all_tokens

    def compute_frequencies(self, tokens: List[str], max_words: int = 100) -> Dict[str, int]:
        """Compute word frequencies."""
        if not tokens:
            return {}

        # Count frequencies
        freq_counter = Counter(tokens)

        # Get top words
        most_common = freq_counter.most_common(max_words)

        return dict(most_common)

    def generate_wordcloud_image(self, frequencies: Dict[str, int], width: int = 800, height: int = 400) -> bytes:
        """Generate a simple word cloud image."""
        if not frequencies:
            # Return a blank image
            return self._create_blank_image(width, height)

        try:
            # Try to use wordcloud if available
            from wordcloud import WordCloud
            import matplotlib.pyplot as plt

            # Create word cloud
            wordcloud = WordCloud(
                width=width,
                height=height,
                background_color='white',
                max_words=len(frequencies),
                prefer_horizontal=0.9
            )

            wordcloud.generate_from_frequencies(frequencies)

            # Convert to image
            plt.figure(figsize=(width/100, height/100))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')

            # Save to bytes
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            plt.close()
            buf.seek(0)

            return buf.getvalue()

        except ImportError:
            # Fallback: create a simple text-based representation
            logger.warning("WordCloud library not available, using fallback")
            return self._create_text_image(frequencies, width, height)

    def _create_blank_image(self, width: int, height: int) -> bytes:
        """Create a blank image."""
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(width/100, height/100))
            plt.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=16)
            plt.axis('off')

            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            plt.close()
            buf.seek(0)

            return buf.getvalue()
        except ImportError:
            # Return empty bytes if matplotlib not available
            return b''

    def _create_text_image(self, frequencies: Dict[str, int], width: int, height: int) -> bytes:
        """Create a simple text-based image."""
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(width/100, height/100))
            plt.axis('off')

            # Display top words as text
            top_words = list(frequencies.items())[:20]
            text_content = "Top Words:\n" + "\n".join([f"{word}: {count}" for word, count in top_words])

            plt.text(0.1, 0.9, text_content, fontsize=10, verticalalignment='top',
                    fontfamily='monospace', transform=plt.gca().transAxes)

            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            plt.close()
            buf.seek(0)

            return buf.getvalue()
        except ImportError:
            return b''

    async def create_sentiment_tagged_wordcloud(self, texts: List[str],
                                               analysis_results: List[Dict[str, Any]],
                                               max_words: int = 100,
                                               width: int = 800,
                                               height: int = 400) -> Tuple[bytes, Dict[str, Dict[str, Any]]]:
        """Create sentiment-tagged word cloud."""
        if not texts or not analysis_results or len(texts) != len(analysis_results):
            return self._create_blank_image(width, height), {}

        # Prepare tokens with sentiment information
        all_tokens_with_sentiment = []

        for text, analysis in zip(texts, analysis_results):
            sentiment_label = analysis.get('sentiment_label', 'neutral')
            confidence = analysis.get('confidence_score', 0.5)

            tokens = await self.prepare_tokens([text])
            for token in tokens:
                all_tokens_with_sentiment.append({
                    'word': token,
                    'sentiment': sentiment_label,
                    'confidence': confidence
                })

        # Group by word and aggregate sentiment
        word_data = {}
        for item in all_tokens_with_sentiment:
            word = item['word']
            if word not in word_data:
                word_data[word] = {
                    'sentiment': item['sentiment'],
                    'confidence': item['confidence'],
                    'frequency': 0,
                    'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0}
                }

            word_data[word]['frequency'] += 1
            word_data[word]['sentiment_distribution'][item['sentiment']] += 1

        # Determine dominant sentiment for each word
        for word, data in word_data.items():
            distribution = data['sentiment_distribution']
            dominant_sentiment = max(distribution, key=distribution.get)
            data['sentiment'] = dominant_sentiment
            data['confidence'] = distribution[dominant_sentiment] / sum(distribution.values())

        # Create frequency dict for word cloud
        frequencies = {word: data['frequency'] for word, data in word_data.items()}

        # Generate image
        image_bytes = self.generate_wordcloud_image(frequencies, width, height)

        return image_bytes, word_data

    async def create_frequency_chart(self, frequencies: Dict[str, int], max_words: int = 20) -> bytes:
        """Create a frequency bar chart."""
        if not frequencies:
            return self._create_blank_image(800, 400)

        try:
            import matplotlib.pyplot as plt

            # Get top words
            top_words = dict(list(frequencies.items())[:max_words])
            words = list(top_words.keys())
            counts = list(top_words.values())

            plt.figure(figsize=(12, 6))
            bars = plt.bar(range(len(words)), counts)

            # Color bars based on frequency
            max_count = max(counts) if counts else 1
            for i, (bar, count) in enumerate(zip(bars, counts)):
                intensity = count / max_count
                bar.set_color((intensity, 0.5, 1-intensity))  # Blue to red gradient

            plt.xticks(range(len(words)), words, rotation=45, ha='right')
            plt.xlabel('Words')
            plt.ylabel('Frequency')
            plt.title('Top Word Frequencies')
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            plt.close()
            buf.seek(0)

            return buf.getvalue()
        except ImportError:
            return self._create_blank_image(800, 400)