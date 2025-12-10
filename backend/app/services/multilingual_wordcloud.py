"""
Advanced Multilingual Word Cloud Generator
Supports all Indian languages and mixed text for generating word clouds from user comments
"""

import re
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import base64
from io import BytesIO

# Word cloud libraries
try:
    from wordcloud import WordCloud
    import matplotlib.font_manager as fm
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# Language processing
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Indian language support
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    INDIC_SUPPORT = True
except ImportError:
    INDIC_SUPPORT = False

class MultilingualWordCloudGenerator:
    """
    Advanced word cloud generator supporting all Indian languages and mixed text
    """
    
    def __init__(self):
        self.initialize_stopwords()
        self.initialize_fonts()
        
    def initialize_stopwords(self):
        """Initialize stopwords for multiple languages"""
        self.stopwords = {
            'en': set([
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
                'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall',
                'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
                'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their',
                'policy', 'proposal', 'framework', 'initiative', 'legislation', 'government',
                'implementation', 'analysis', 'comment', 'feedback', 'response', 'opinion'
            ]),
            'hi': set([
                'का', 'के', 'की', 'को', 'से', 'में', 'पर', 'और', 'या', 'लेकिन', 'है', 'हैं', 'था', 'थे', 'होना',
                'यह', 'वह', 'इस', 'उस', 'मैं', 'तुम', 'हम', 'वे', 'अपना', 'उनका', 'इसका', 'उसका',
                'नीति', 'प्रस्ताव', 'ढांचा', 'पहल', 'कानून', 'सरकार', 'कार्यान्वयन', 'विश्लेषण', 'टिप्पणी'
            ]),
            'bn': set([
                'এর', 'তার', 'তাদের', 'আমার', 'আমাদের', 'এই', 'সেই', 'যে', 'কি', 'কিন্তু', 'এবং', 'বা',
                'হয়', 'হবে', 'ছিল', 'থাকে', 'করে', 'হতে', 'দিয়ে', 'নিয়ে', 'জন্য', 'থেকে', 'পর্যন্ত',
                'নীতি', 'প্রস্তাব', 'কাঠামো', 'উদ্যোগ', 'আইন', 'সরকার', 'বাস্তবায়ন', 'বিশ্লেষণ', 'মন্তব্য'
            ]),
            'te': set([
                'యొక్క', 'కు', 'లో', 'పై', 'నుండి', 'తో', 'వద్ద', 'కోసం', 'ద్వారా', 'మరియు', 'లేదా', 'కానీ',
                'ఉంది', 'ఉన్నాయి', 'ఉంటుంది', 'ఉండవచ్చు', 'చేయవచ్చు', 'కావచ్చు', 'అవకాశం', 'సాధ్యం',
                'విధానం', 'ప్రతిపాదన', 'చట్రం', 'కార్యక్రమం', 'చట్టం', 'ప్రభుత్వం', 'అమలు', 'విశ్లేషణ', 'వ్యాఖ్య'
            ]),
            'ta': set([
                'அந்த', 'இந்த', 'அவர்', 'அவள்', 'நான்', 'நாம்', 'நீ', 'அவர்கள்', 'இது', 'அது', 'எந்த',
                'ஆனால்', 'மற்றும்', 'அல்லது', 'உள்ளது', 'இருக்கும்', 'இருந்தது', 'செய்ய', 'கொண்டு',
                'கொள்கை', 'முன்மொழிவு', 'கட்டமைப்பு', 'முயற்சி', 'சட்டம்', 'அரசு', 'செயல்படுத்துதல்', 'பகுப்பாய்வு', 'கருத்து'
            ]),
            'mr': set([
                'च्या', 'ची', 'चे', 'ला', 'ने', 'मध्ये', 'वर', 'आणि', 'किंवा', 'पण', 'आहे', 'होते', 'असेल',
                'हा', 'हे', 'ते', 'तो', 'ती', 'मी', 'तू', 'आम्ही', 'तुम्ही', 'त्यांचे', 'आमचे', 'तुमचे',
                'धोरण', 'प्रस्ताव', 'चौकट', 'उपक्रम', 'कायदा', 'सरकार', 'अंमलबजावणी', 'विश्लेषण', 'टिप्पणी'
            ]),
            'gu': set([
                'ના', 'નું', 'ની', 'ને', 'થી', 'મા', 'પર', 'અને', 'કે', 'પણ', 'છે', 'હતું', 'હશે',
                'આ', 'તે', 'એ', 'હું', 'તમે', 'આપણે', 'તેઓ', 'મારું', 'તમારું', 'તેમનું', 'આપણું',
                'નીતિ', 'દરખાસ્ત', 'માળખું', 'પહેલ', 'કાયદો', 'સરકાર', 'અમલીકરણ', 'વિશ્લેષણ', 'ટિપ્પણી'
            ]),
            'kn': set([
                'ಅದು', 'ಇದು', 'ಆ', 'ಈ', 'ಅವರು', 'ಅವಳು', 'ನಾನು', 'ನಾವು', 'ನೀವು', 'ಅವರು', 'ಮತ್ತು', 'ಅಥವಾ', 'ಆದರೆ',
                'ಇದೆ', 'ಇದ್ದರು', 'ಇರುತ್ತದೆ', 'ಮಾಡಿ', 'ಕೊಂಡು', 'ದಿಂದ', 'ಗೆ', 'ಅಲ್ಲಿ', 'ಇಲ್ಲಿ',
                'ನೀತಿ', 'ಪ್ರಸ್ತಾವನೆ', 'ಚೌಕಟ್ಟು', 'ಉಪಕ್ರಮ', 'ಕಾನೂನು', 'ಸರ್ಕಾರ', 'ಅನುಷ್ಠಾನ', 'ವಿಶ್ಲೇಷಣೆ', 'ಟಿಪ್ಪಣಿ'
            ]),
            'ml': set([
                'അത്', 'ഇത്', 'ആ', 'ഈ', 'അവൻ', 'അവൾ', 'ഞാൻ', 'നാം', 'നീ', 'അവർ', 'ഉം', 'അല്ലെങ്കിൽ', 'പക്ഷേ',
                'ആണ്', 'ആയിരുന്നു', 'ആകും', 'ചെയ്യുക', 'കൊണ്ട്', 'ൽ', 'ലേക്ക്', 'അവിടെ', 'ഇവിടെ',
                'നയം', 'നിർദ്ദേശം', 'ചട്ടക്കൂട്', 'സംരംഭം', 'നിയമം', 'സർക്കാർ', 'നടപ്പാക്കൽ', 'വിശകലനം', 'അഭിപ്രായം'
            ]),
            'pa': set([
                'ਦਾ', 'ਦੇ', 'ਦੀ', 'ਨੂੰ', 'ਤੋਂ', 'ਵਿਚ', 'ਤੇ', 'ਅਤੇ', 'ਜਾਂ', 'ਪਰ', 'ਹੈ', 'ਸੀ', 'ਹੋਵੇਗਾ',
                'ਇਹ', 'ਉਹ', 'ਮੈਂ', 'ਤੁਸੀਂ', 'ਅਸੀਂ', 'ਉਹਨਾਂ', 'ਮੇਰਾ', 'ਤੁਹਾਡਾ', 'ਸਾਡਾ', 'ਉਹਨਾਂ ਦਾ',
                'ਨੀਤੀ', 'ਪ੍ਰਸਤਾਵ', 'ਢਾਂਚਾ', 'ਪਹਿਲਕਦਮੀ', 'ਕਾਨੂੰਨ', 'ਸਰਕਾਰ', 'ਲਾਗੂਕਰਨ', 'ਵਿਸ਼ਲੇਸ਼ਣ', 'ਟਿੱਪਣੀ'
            ])
        }
        
        # Try to get NLTK stopwords if available
        if NLTK_AVAILABLE:
            try:
                english_stopwords = set(stopwords.words('english'))
                self.stopwords['en'].update(english_stopwords)
            except:
                pass
    
    def initialize_fonts(self):
        """Initialize fonts for different scripts"""
        self.fonts = {
            'devanagari': None,
            'tamil': None,
            'telugu': None,
            'kannada': None,
            'malayalam': None,
            'bengali': None,
            'gujarati': None,
            'punjabi': None,
            'english': None
        }
        
        # Try to find system fonts for Indian scripts
        try:
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            
            # Common fonts for Indian scripts
            font_mappings = {
                'devanagari': ['Noto Sans Devanagari', 'Mangal', 'Sanskrit 2003', 'Kruti Dev'],
                'tamil': ['Noto Sans Tamil', 'Tamil Sangam MN', 'Latha'],
                'telugu': ['Noto Sans Telugu', 'Gautami', 'Vani'],
                'kannada': ['Noto Sans Kannada', 'Tunga', 'Kedage'],
                'malayalam': ['Noto Sans Malayalam', 'Kartika', 'AnjaliOldLipi'],
                'bengali': ['Noto Sans Bengali', 'Vrinda', 'Akaash'],
                'gujarati': ['Noto Sans Gujarati', 'Shruti'],
                'punjabi': ['Noto Sans Gurmukhi', 'Raavi']
            }
            
            for script, font_list in font_mappings.items():
                for font_name in font_list:
                    if font_name in available_fonts:
                        self.fonts[script] = font_name
                        break
                        
        except Exception as e:
            print(f"Font initialization warning: {e}")
    
    def detect_script(self, text: str) -> Dict[str, bool]:
        """Detect which scripts are present in the text"""
        scripts = {
            'english': bool(re.search(r'[a-zA-Z]', text)),
            'devanagari': bool(re.search(r'[\u0900-\u097F]', text)),
            'tamil': bool(re.search(r'[\u0B80-\u0BFF]', text)),
            'telugu': bool(re.search(r'[\u0C00-\u0C7F]', text)),
            'kannada': bool(re.search(r'[\u0C80-\u0CFF]', text)),
            'malayalam': bool(re.search(r'[\u0D00-\u0D7F]', text)),
            'bengali': bool(re.search(r'[\u0980-\u09FF]', text)),
            'gujarati': bool(re.search(r'[\u0A80-\u0AFF]', text)),
            'punjabi': bool(re.search(r'[\u0A00-\u0A7F]', text))
        }
        return scripts
    
    def clean_and_tokenize(self, text: str, languages: List[str] = None) -> List[str]:
        """Clean and tokenize text for word cloud generation"""
        if not text or pd.isna(text):
            return []
        
        # Convert to string and normalize
        text = str(text).strip()
        
        if not text:
            return []
        
        # Detect scripts if languages not provided
        if languages is None:
            script_detection = self.detect_script(text)
            languages = [script for script, present in script_detection.items() if present]
        
        # Clean text - remove URLs, mentions, hashtags, special chars
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'[^\w\s\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F\u0A80-\u0AFF\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F]', ' ', text)
        
        # Tokenize based on script
        words = []
        
        # Simple whitespace tokenization for most cases
        tokens = text.split()
        
        for token in tokens:
            # Remove very short words (less than 2 characters) unless they are meaningful
            if len(token) < 2:
                continue
                
            # Remove very long words (likely noise)
            if len(token) > 50:
                continue
            
            # Convert to lowercase for English words
            if re.match(r'^[a-zA-Z]+$', token):
                token = token.lower()
            
            words.append(token)
        
        # Remove stopwords
        filtered_words = []
        for word in words:
            is_stopword = False
            
            # Check against stopwords for detected languages
            for lang_code in languages:
                if lang_code == 'english':
                    lang_code = 'en'
                elif lang_code == 'devanagari':
                    lang_code = 'hi'  # Default to Hindi for Devanagari
                
                if lang_code in self.stopwords and word.lower() in self.stopwords[lang_code]:
                    is_stopword = True
                    break
            
            if not is_stopword:
                filtered_words.append(word)
        
        return filtered_words
    
    def generate_word_frequencies(self, texts: List[str], min_freq: int = 2) -> Dict[str, int]:
        """Generate word frequencies from a list of texts"""
        all_words = []
        
        for text in texts:
            words = self.clean_and_tokenize(text)
            all_words.extend(words)
        
        # Count frequencies
        word_freq = Counter(all_words)
        
        # Filter by minimum frequency
        filtered_freq = {word: freq for word, freq in word_freq.items() if freq >= min_freq}
        
        return filtered_freq
    
    def create_wordcloud(self, word_frequencies: Dict[str, int], 
                        width: int = 800, height: int = 400,
                        background_color: str = 'white',
                        colormap: str = 'viridis',
                        max_words: int = 100) -> Optional[str]:
        """Create word cloud and return as base64 encoded image"""
        
        if not WORDCLOUD_AVAILABLE or not word_frequencies:
            return None
        
        try:
            # Detect primary script from words
            scripts = defaultdict(int)
            for word in word_frequencies.keys():
                script_detection = self.detect_script(word)
                for script, present in script_detection.items():
                    if present:
                        scripts[script] += word_frequencies[word]
            
            # Choose font based on most frequent script
            primary_script = max(scripts.keys(), key=scripts.get) if scripts else 'english'
            font_path = self.fonts.get(primary_script)
            
            # Create WordCloud object
            wc_params = {
                'width': width,
                'height': height,
                'background_color': background_color,
                'colormap': colormap,
                'max_words': max_words,
                'relative_scaling': 0.5,
                'min_font_size': 10,
                'max_font_size': 100,
                'prefer_horizontal': 0.7,
                'random_state': 42
            }
            
            if font_path:
                wc_params['font_path'] = font_path
            
            wordcloud = WordCloud(**wc_params)
            
            # Generate word cloud
            wordcloud.generate_from_frequencies(word_frequencies)
            
            # Convert to image
            fig, ax = plt.subplots(figsize=(width/100, height/100))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            plt.tight_layout(pad=0)
            
            # Save to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100, facecolor='white')
            buffer.seek(0)
            
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)
            
            return image_base64
            
        except Exception as e:
            print(f"Error creating word cloud: {e}")
            return None
    
    def create_sentiment_wordcloud(self, texts_with_sentiment: List[Tuple[str, str]], 
                                 sentiment_filter: str = None) -> Optional[str]:
        """Create word cloud filtered by sentiment"""
        
        if sentiment_filter:
            filtered_texts = [text for text, sentiment in texts_with_sentiment if sentiment == sentiment_filter]
        else:
            filtered_texts = [text for text, sentiment in texts_with_sentiment]
        
        if not filtered_texts:
            return None
        
        word_freq = self.generate_word_frequencies(filtered_texts)
        
        # Choose colors based on sentiment
        color_maps = {
            'positive': 'Greens',
            'negative': 'Reds', 
            'neutral': 'Blues'
        }
        
        colormap = color_maps.get(sentiment_filter, 'viridis')
        
        return self.create_wordcloud(word_freq, colormap=colormap)
    
    def analyze_comment_words(self, comments: List[str]) -> Dict[str, Any]:
        """Comprehensive analysis of words in comments"""
        
        # Generate word frequencies
        word_freq = self.generate_word_frequencies(comments)
        
        # Language detection
        script_stats = defaultdict(int)
        total_words = sum(word_freq.values())
        
        for word, freq in word_freq.items():
            scripts = self.detect_script(word)
            for script, present in scripts.items():
                if present:
                    script_stats[script] += freq
        
        # Calculate percentages
        script_percentages = {script: (count/total_words)*100 for script, count in script_stats.items()}
        
        # Top words overall
        top_words = dict(Counter(word_freq).most_common(20))
        
        # Create overall word cloud
        wordcloud_image = self.create_wordcloud(word_freq)
        
        return {
            'total_unique_words': len(word_freq),
            'total_word_occurrences': total_words,
            'top_words': top_words,
            'script_distribution': dict(script_percentages),
            'word_frequencies': word_freq,
            'wordcloud_image': wordcloud_image,
            'languages_detected': list(script_stats.keys())
        }

# Global instance
try:
    global_wordcloud_generator = MultilingualWordCloudGenerator()
    WORDCLOUD_GENERATOR_AVAILABLE = True
    print("✅ Multilingual Word Cloud Generator initialized successfully")
except Exception as e:
    WORDCLOUD_GENERATOR_AVAILABLE = False
    print(f"❌ Failed to initialize Word Cloud Generator: {e}")

# Convenience functions
def generate_comments_wordcloud(comments: List[str]) -> Dict[str, Any]:
    """Generate word cloud analysis for comments"""
    if WORDCLOUD_GENERATOR_AVAILABLE:
        return global_wordcloud_generator.analyze_comment_words(comments)
    else:
        return {'error': 'Word cloud generator not available'}

def generate_sentiment_wordcloud(texts_with_sentiment: List[Tuple[str, str]], sentiment: str = None) -> Optional[str]:
    """Generate sentiment-filtered word cloud"""
    if WORDCLOUD_GENERATOR_AVAILABLE:
        return global_wordcloud_generator.create_sentiment_wordcloud(texts_with_sentiment, sentiment)
    else:
        return None

if __name__ == "__main__":
    # Test with multilingual comments
    test_comments = [
        "I strongly support this policy initiative. यह बहुत अच्छी योजना है।",
        "This proposal is excellent. మంచి ప్రతిపాదన.",
        "नीति बहुत अच्छी है। Very good policy framework.",
        "இது சிறந்த முன்மொழிவு. Great proposal indeed.",
        "ಈ ನೀತಿ ಉತ್ತಮವಾಗಿದೆ। This policy is good.",
        "এই নীতি খুবই ভালো। Excellent initiative.",
        "આ નીતિ ખૂબ સારી છે। Very supportive of this.",
        "ਇਹ ਨੀਤੀ ਬਹੁਤ ਚੰਗੀ ਹੈ। Good framework.",
        "ही धोरण अतिशय चांगले आहे। Great policy."
    ]
    
    print("=== Testing Multilingual Word Cloud ===")
    result = generate_comments_wordcloud(test_comments)
    print(f"Total unique words: {result.get('total_unique_words', 0)}")
    print(f"Languages detected: {result.get('languages_detected', [])}")
    print(f"Script distribution: {result.get('script_distribution', {})}")
    print(f"Top words: {list(result.get('top_words', {}).keys())[:10]}")
    print(f"Word cloud generated: {'Yes' if result.get('wordcloud_image') else 'No'}")