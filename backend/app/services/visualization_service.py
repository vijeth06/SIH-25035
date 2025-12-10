"""
Visualization utilities: token preparation, word frequency computation, and word cloud image generation.
"""
from __future__ import annotations

from typing import List, Dict, Iterable, Tuple, Optional
import re
from collections import Counter
from io import BytesIO
from dataclasses import dataclass

import numpy as np
from wordcloud import WordCloud
from PIL import Image

# Lightweight stopwords list (avoid heavy runtime downloads). Extend as needed.
BASIC_STOPWORDS = set(
    [
        "the","a","an","and","or","but","if","while","with","without","to","of","for","in","on","at","by","from","as",
        "is","are","was","were","be","been","being","it","this","that","these","those","i","we","you","they","he","she",
        "them","his","her","their","our","your","not","no","do","does","did","done","can","could","should","would","may",
        "might","will","shall","than","then","there","here","about","into","over","under","again","further","also","such",
        # Domain-specific mild stopwords
        "government","policy","act","section","clause","bill","proposed","draft","consultation","feedback","comment",
    ]
)

TOKEN_PATTERN = re.compile(r"[A-Za-z]{2,}")


@dataclass
class SentimentTaggedWord:
    """Word with sentiment information and contextual examples."""
    word: str
    frequency: int
    sentiment_score: float  # -1 to 1
    sentiment_label: str  # positive, negative, neutral
    contextual_examples: List[str]  # Examples where this word appears
    confidence: float


class VisualizationService:
    """Helper service to prepare tokens, frequencies, and word cloud image bytes."""

    async def prepare_tokens(self, texts: List[str], min_len: int = 3) -> List[str]:
        """Clean, tokenize, and filter tokens from a list of texts."""
        tokens: List[str] = []
        for t in texts:
            if not t:
                continue
            # Basic normalize
            s = t.lower()
            # Remove URLs/emails rudimentarily
            s = re.sub(r"https?://\S+", " ", s)
            s = re.sub(r"\S+@\S+", " ", s)
            # Extract alpha tokens
            words = TOKEN_PATTERN.findall(s)
            for w in words:
                if len(w) < min_len:
                    continue
                if w in BASIC_STOPWORDS:
                    continue
                tokens.append(w)
        return tokens

    def compute_frequencies(self, tokens: Iterable[str], max_words: int = 100) -> Dict[str, int]:
        """Compute top-N word frequencies as a dict sorted by count desc."""
        counter = Counter(tokens)
        most_common = counter.most_common(max_words)
        return {k: int(v) for k, v in most_common}

    def generate_wordcloud_image(self, frequencies: Dict[str, int], width: int = 800, height: int = 400) -> bytes:
        """Generate a PNG image for a word cloud and return raw bytes."""
        if not frequencies:
            # Empty transparent image
            img = Image.new("RGBA", (width, height), (255, 255, 255, 0))
            bio = BytesIO()
            img.save(bio, format="PNG")
            return bio.getvalue()

        wc = WordCloud(width=width, height=height, background_color="white")
        wc.generate_from_frequencies(frequencies)

        img = wc.to_image()
        bio = BytesIO()
        img.save(bio, format="PNG")
        return bio.getvalue()

    async def create_sentiment_tagged_wordcloud(self,
                                                texts: List[str],
                                                analysis_results: List[Dict],
                                                width: int = 800,
                                                height: int = 400,
                                                max_words: int = 100) -> Tuple[bytes, Dict[str, SentimentTaggedWord]]:
        """
        Create a sentiment-tagged word cloud with contextual examples.

        Args:
            texts: List of original texts
            analysis_results: List of analysis results with sentiment information
            width: Width of the word cloud image
            height: Height of the word cloud image
            max_words: Maximum number of words to include

        Returns:
            Tuple of (wordcloud_image_bytes, word_sentiment_data)
        """
        # Create sentiment-tagged word data
        sentiment_words = await self._create_sentiment_word_data(texts, analysis_results)

        # Filter to top words by frequency
        top_words = dict(sorted(sentiment_words.items(),
                               key=lambda x: x[1].frequency,
                               reverse=True)[:max_words])

        # Create frequency dict for word cloud
        frequencies = {word: data.frequency for word, data in top_words.items()}

        # Generate color function based on sentiment
        def sentiment_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            if word in sentiment_words:
                sentiment = sentiment_words[word].sentiment_label
                if sentiment == 'positive':
                    return "hsl(120, 100%, 30%)"  # Green
                elif sentiment == 'negative':
                    return "hsl(0, 100%, 30%)"    # Red
                else:
                    return "hsl(60, 100%, 30%)"   # Yellow
            return "hsl(0, 0%, 50%)"  # Gray fallback

        # Generate word cloud with sentiment colors
        wc = WordCloud(
            width=width,
            height=height,
            background_color="white",
            color_func=sentiment_color_func
        )
        wc.generate_from_frequencies(frequencies)

        img = wc.to_image()
        bio = BytesIO()
        img.save(bio, format="PNG")
        image_bytes = bio.getvalue()

        return image_bytes, top_words

    async def _create_sentiment_word_data(self,
                                         texts: List[str],
                                         analysis_results: List[Dict]) -> Dict[str, SentimentTaggedWord]:
        """
        Create sentiment-tagged word data with contextual examples.
        """
        word_data = {}

        for text, result in zip(texts, analysis_results):
            # Get sentiment information from result
            sentiment_label = result.get('sentiment_label', 'neutral')
            sentiment_score = result.get('sentiment_score', 0.0)
            confidence = result.get('confidence_score', 0.5)

            # Tokenize text
            tokens = await self.prepare_tokens([text])

            # Create contextual examples (short snippets around each word)
            for i, token in enumerate(tokens):
                if token not in word_data:
                    word_data[token] = SentimentTaggedWord(
                        word=token,
                        frequency=0,
                        sentiment_score=0.0,
                        sentiment_label='neutral',
                        contextual_examples=[],
                        confidence=0.0
                    )

                # Increment frequency
                word_data[token].frequency += 1

                # Update sentiment (weighted average)
                current_freq = word_data[token].frequency
                word_data[token].sentiment_score = (
                    (word_data[token].sentiment_score * (current_freq - 1)) + sentiment_score
                ) / current_freq

                # Update sentiment label (based on average score)
                avg_sentiment = word_data[token].sentiment_score
                if avg_sentiment > 0.1:
                    word_data[token].sentiment_label = 'positive'
                elif avg_sentiment < -0.1:
                    word_data[token].sentiment_label = 'negative'
                else:
                    word_data[token].sentiment_label = 'neutral'

                # Update confidence (weighted average)
                word_data[token].confidence = (
                    (word_data[token].confidence * (current_freq - 1)) + confidence
                ) / current_freq

                # Add contextual example (if not too many already)
                if len(word_data[token].contextual_examples) < 3:
                    # Create context snippet
                    words = text.split()
                    token_index = next((j for j, w in enumerate(words) if token.lower() in w.lower()), -1)
                    if token_index >= 0:
                        start = max(0, token_index - 5)
                        end = min(len(words), token_index + 6)
                        context = ' '.join(words[start:end])
                        if context not in word_data[token].contextual_examples:
                            word_data[token].contextual_examples.append(context)

        return word_data
    
    async def generate_density_wordcloud(self, 
                                       texts: List[str],
                                       analysis_results: List[Dict],
                                       width: int = 800,
                                       height: int = 600,
                                       max_words: int = 100,
                                       density_type: str = "frequency") -> Tuple[bytes, Dict]:
        """
        Generate word cloud with density visualization.
        
        Args:
            texts: List of comment texts
            analysis_results: Analysis results with sentiment info
            width: Word cloud width
            height: Word cloud height
            max_words: Maximum words to include
            density_type: "frequency", "sentiment_intensity", or "stakeholder_spread"
            
        Returns:
            Tuple of (image_bytes, density_data)
        """
        # Create enhanced word data with density metrics
        word_data = await self._create_density_word_data(texts, analysis_results, density_type)
        
        # Filter to top words
        top_words = dict(sorted(word_data.items(),
                               key=lambda x: x[1]['density_score'],
                               reverse=True)[:max_words])
        
        # Create frequency dict for word cloud
        frequencies = {word: data['density_score'] for word, data in top_words.items()}
        
        # Enhanced color function based on density type
        def density_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            if word in word_data:
                data = word_data[word]
                
                if density_type == "sentiment_intensity":
                    # Color by sentiment intensity
                    intensity = abs(data['sentiment_score'])
                    if data['sentiment_score'] > 0:
                        return f"hsl(120, {int(intensity * 100)}%, 30%)"  # Green intensity
                    else:
                        return f"hsl(0, {int(intensity * 100)}%, 30%)"    # Red intensity
                        
                elif density_type == "stakeholder_spread":
                    # Color by stakeholder diversity
                    spread = data.get('stakeholder_spread', 0)
                    return f"hsl(240, {int(spread * 100)}%, 40%)"  # Blue intensity
                    
                else:  # frequency
                    # Color by frequency
                    freq_norm = data['frequency'] / max(d['frequency'] for d in word_data.values())
                    return f"hsl(60, {int(freq_norm * 100)}%, 30%)"  # Yellow intensity
                    
            return "hsl(0, 0%, 50%)"  # Gray fallback
        
        # Generate enhanced word cloud
        wc = WordCloud(
            width=width,
            height=height,
            background_color="white",
            color_func=density_color_func,
            relative_scaling=0.5,
            min_font_size=8
        )
        wc.generate_from_frequencies(frequencies)
        
        img = wc.to_image()
        bio = BytesIO()
        img.save(bio, format="PNG")
        image_bytes = bio.getvalue()
        
        return image_bytes, top_words
    
    async def generate_stakeholder_wordcloud(self,
                                           texts_by_stakeholder: Dict[str, List[str]],
                                           analysis_results_by_stakeholder: Dict[str, List[Dict]],
                                           width: int = 1200,
                                           height: int = 800) -> Tuple[bytes, Dict]:
        """
        Generate comparative word cloud showing different stakeholder perspectives.
        
        Args:
            texts_by_stakeholder: Comments grouped by stakeholder type
            analysis_results_by_stakeholder: Analysis results by stakeholder
            width: Word cloud width
            height: Word cloud height
            
        Returns:
            Tuple of (image_bytes, stakeholder_comparison_data)
        """
        from PIL import Image, ImageDraw, ImageFont
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        # Generate individual word clouds for each stakeholder type
        stakeholder_clouds = {}
        stakeholder_data = {}
        
        colors = {
            'individual': 'hsl(0, 70%, 50%)',      # Red
            'business': 'hsl(120, 70%, 40%)',     # Green
            'ngo': 'hsl(240, 70%, 50%)',          # Blue
            'academic': 'hsl(60, 70%, 50%)',      # Yellow
            'legal': 'hsl(300, 70%, 50%)',        # Purple
            'government': 'hsl(180, 70%, 40%)'    # Cyan
        }
        
        for stakeholder_type, texts in texts_by_stakeholder.items():
            if not texts:
                continue
                
            analysis_results = analysis_results_by_stakeholder.get(stakeholder_type, [])
            
            # Create stakeholder-specific word data
            word_data = await self._create_sentiment_word_data(texts, analysis_results)
            top_words = dict(sorted(word_data.items(),
                                   key=lambda x: x[1].frequency,
                                   reverse=True)[:30])
            
            frequencies = {word: data.frequency for word, data in top_words.items()}
            
            if frequencies:
                # Single color for stakeholder type
                base_color = colors.get(stakeholder_type, 'hsl(0, 0%, 50%)')
                
                def stakeholder_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
                    return base_color
                
                wc = WordCloud(
                    width=width//3,
                    height=height//2,
                    background_color="white",
                    color_func=stakeholder_color_func,
                    max_words=30
                )
                wc.generate_from_frequencies(frequencies)
                stakeholder_clouds[stakeholder_type] = wc.to_image()
                stakeholder_data[stakeholder_type] = top_words
        
        # Combine into single image
        combined_img = Image.new('RGB', (width, height), 'white')
        
        # Arrange stakeholder clouds in grid
        rows = 2
        cols = 3
        cell_width = width // cols
        cell_height = height // rows
        
        x, y = 0, 0
        for i, (stakeholder_type, cloud_img) in enumerate(stakeholder_clouds.items()):
            if i >= rows * cols:
                break
                
            # Resize cloud image to fit cell
            cloud_img = cloud_img.resize((cell_width - 20, cell_height - 40))
            
            # Paste image
            combined_img.paste(cloud_img, (x + 10, y + 30))
            
            # Add stakeholder label
            draw = ImageDraw.Draw(combined_img)
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            draw.text((x + 10, y + 5), stakeholder_type.title(), fill='black', font=font)
            
            # Move to next position
            x += cell_width
            if x >= width:
                x = 0
                y += cell_height
        
        # Convert to bytes
        bio = BytesIO()
        combined_img.save(bio, format="PNG")
        image_bytes = bio.getvalue()
        
        return image_bytes, stakeholder_data
    
    async def _create_density_word_data(self, 
                                      texts: List[str],
                                      analysis_results: List[Dict],
                                      density_type: str) -> Dict[str, Dict]:
        """Create word data with density metrics."""
        word_data = {}
        stakeholder_words = {}  # Track which stakeholders use which words
        
        for i, text in enumerate(texts):
            if not text or i >= len(analysis_results):
                continue
                
            analysis = analysis_results[i]
            sentiment_score = analysis.get('sentiment_score', 0)
            stakeholder_type = analysis.get('stakeholder_type', 'unknown')
            
            tokens = await self.prepare_tokens([text])
            
            for token in tokens:
                if token not in word_data:
                    word_data[token] = {
                        'frequency': 0,
                        'sentiment_score': 0,
                        'stakeholder_types': set(),
                        'contexts': []
                    }
                
                word_data[token]['frequency'] += 1
                word_data[token]['sentiment_score'] = (
                    (word_data[token]['sentiment_score'] * (word_data[token]['frequency'] - 1)) + sentiment_score
                ) / word_data[token]['frequency']
                word_data[token]['stakeholder_types'].add(stakeholder_type)
                
                # Add context
                if len(word_data[token]['contexts']) < 3:
                    words = text.split()
                    token_index = next((j for j, w in enumerate(words) if token.lower() in w.lower()), -1)
                    if token_index >= 0:
                        start = max(0, token_index - 3)
                        end = min(len(words), token_index + 4)
                        context = ' '.join(words[start:end])
                        word_data[token]['contexts'].append(context)
        
        # Calculate density scores based on type
        for word, data in word_data.items():
            if density_type == "frequency":
                data['density_score'] = data['frequency']
            elif density_type == "sentiment_intensity":
                data['density_score'] = data['frequency'] * abs(data['sentiment_score'])
            elif density_type == "stakeholder_spread":
                stakeholder_diversity = len(data['stakeholder_types'])
                data['density_score'] = data['frequency'] * stakeholder_diversity
                data['stakeholder_spread'] = stakeholder_diversity / 6  # Normalize by max stakeholder types
            else:
                data['density_score'] = data['frequency']
        
        return word_data