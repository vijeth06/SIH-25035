"""
Lightweight FastAPI application for sentiment analysis API.
Only loads advanced models on-demand to prevent startup delays.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import sys
import os

# Add services to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app', 'services'))

# Global variables for lazy loading
advanced_analyzer = None
wordcloud_generator = None
basic_explainer = None

def get_basic_explainer():
    global basic_explainer
    if basic_explainer is None:
        try:
            from sentiment_explainer import analyze_text_with_explanation
            basic_explainer = analyze_text_with_explanation
            print("✅ Basic sentiment explainer loaded")
        except ImportError as e:
            print(f"❌ Basic explainer import failed: {e}")
            basic_explainer = False
    return basic_explainer

def get_advanced_analyzer():
    global advanced_analyzer
    if advanced_analyzer is None:
        try:
            print("🔄 Loading advanced multilingual analyzer (this may take a moment)...")
            from advanced_sentiment_analyzer import AdvancedMultilingualSentimentAnalyzer
            advanced_analyzer = AdvancedMultilingualSentimentAnalyzer()
            print("✅ Advanced analyzer loaded successfully")
        except Exception as e:
            print(f"❌ Advanced analyzer failed to load: {e}")
            advanced_analyzer = False
    return advanced_analyzer

def get_wordcloud_generator():
    global wordcloud_generator
    if wordcloud_generator is None:
        try:
            print("🔄 Loading multilingual wordcloud generator...")
            from multilingual_wordcloud import MultilingualWordCloudGenerator
            wordcloud_generator = MultilingualWordCloudGenerator()
            print("✅ Wordcloud generator loaded successfully")
        except Exception as e:
            print(f"❌ Wordcloud generator failed to load: {e}")
            wordcloud_generator = False
    return wordcloud_generator

app = FastAPI(
    title="MCA eConsultation Sentiment Analysis API",
    description="Lightweight API for multilingual sentiment analysis with on-demand model loading",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class TextAnalysisRequest(BaseModel):
    texts: List[str]
    include_explanation: bool = False
    use_advanced: bool = False

class SentimentResult(BaseModel):
    text: str
    sentiment: str
    confidence: float
    polarity_score: float
    explanation: Optional[Dict[str, Any]] = None

class AnalysisResponse(BaseModel):
    results: List[SentimentResult]
    summary: Dict[str, Any]

class WordCloudRequest(BaseModel):
    texts: List[str]
    width: int = 800
    height: int = 400
    max_words: int = 100
    background_color: str = "white"
    min_font_size: int = 4

class SummarizationRequest(BaseModel):
    texts: List[str]
    max_length: int = 150
    min_length: int = 50
    language: str = "auto"

class ExplanationRequest(BaseModel):
    text: str
    use_advanced: bool = False

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "MCA eConsultation Sentiment Analysis API (Lightweight)", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint with on-demand service availability"""
    basic_available = get_basic_explainer() is not False
    
    return {
        "status": "healthy",
        "basic_explainer_available": basic_available,
        "advanced_analyzer_available": "lazy_load",
        "wordcloud_available": "lazy_load",
        "services": ["sentiment_analysis", "explanation", "wordcloud", "summarization"],
        "note": "Advanced features load on first use"
    }

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_sentiment(request: TextAnalysisRequest):
    """
    Analyze sentiment for multiple texts with optional advanced analysis
    """
    results = []
    sentiments = {"positive": 0, "negative": 0, "neutral": 0}
    
    for text in request.texts:
        try:
            if request.use_advanced:
                # Use advanced analyzer
                analyzer = get_advanced_analyzer()
                if analyzer:
                    analysis = analyzer.analyze_text(text)
                    sentiment_result = SentimentResult(
                        text=text,
                        sentiment=analysis['sentiment'].lower(),
                        confidence=analysis['confidence'],
                        polarity_score=analysis['polarity_score']
                    )
                    
                    if request.include_explanation:
                        sentiment_result.explanation = {
                            "detailed_explanation": analysis['explanation'],
                            "key_indicators": analysis['key_indicators'],
                            "highlighted_words": analysis.get('highlighted_words', []),
                            "highlighted_text": analysis.get('highlighted_text', text),
                            "language_info": analysis.get('language_info', {}),
                            "summary": analysis.get('summary', ''),
                            "is_multilingual": analysis.get('is_multilingual', False),
                            "analysis_methods": analysis.get('analysis_methods', [])
                        }
                else:
                    raise Exception("Advanced analyzer not available")
            else:
                # Use basic explainer
                explainer = get_basic_explainer()
                if explainer:
                    analysis = explainer(text)
                    sentiment_result = SentimentResult(
                        text=text,
                        sentiment=analysis['sentiment'].lower(),
                        confidence=_map_confidence_to_float(analysis['confidence']),
                        polarity_score=analysis['polarity_score']
                    )
                    
                    if request.include_explanation:
                        sentiment_result.explanation = {
                            "detailed_explanation": analysis['explanation'],
                            "key_indicators": analysis['key_indicators'],
                            "highlighted_words": analysis.get('highlighted_words', []),
                            "highlighted_text": analysis.get('highlighted_text', text),
                            "method": analysis.get('method', 'textblob')
                        }
                else:
                    raise Exception("Basic explainer not available")
            
            results.append(sentiment_result)
            sentiments[sentiment_result.sentiment] += 1
            
        except Exception as e:
            # Fallback to neutral sentiment
            sentiment_result = SentimentResult(
                text=text,
                sentiment="neutral",
                confidence=0.5,
                polarity_score=0.0
            )
            if request.include_explanation:
                sentiment_result.explanation = {"error": f"Analysis failed: {str(e)}"}
            
            results.append(sentiment_result)
            sentiments["neutral"] += 1
    
    # Calculate summary statistics
    total = len(results)
    summary = {
        "total_analyzed": total,
        "sentiment_distribution": {
            "positive": {"count": sentiments["positive"], "percentage": round(sentiments["positive"]/total*100, 1)},
            "negative": {"count": sentiments["negative"], "percentage": round(sentiments["negative"]/total*100, 1)},
            "neutral": {"count": sentiments["neutral"], "percentage": round(sentiments["neutral"]/total*100, 1)}
        },
        "average_confidence": round(sum(r.confidence for r in results) / total, 3),
        "average_polarity": round(sum(r.polarity_score for r in results) / total, 3)
    }
    
    return AnalysisResponse(results=results, summary=summary)

@app.post("/api/wordcloud")
async def generate_wordcloud(request: WordCloudRequest):
    """Generate multilingual word cloud from text data"""
    try:
        generator = get_wordcloud_generator()
        if generator:
            wordcloud_data = generator.create_wordcloud(
                texts=request.texts,
                width=request.width,
                height=request.height,
                max_words=request.max_words,
                background_color=request.background_color,
                min_font_size=request.min_font_size
            )
            
            return {
                "status": "success",
                "wordcloud_data": wordcloud_data,
                "languages_detected": wordcloud_data.get('languages_detected', []),
                "total_words": wordcloud_data.get('total_words', 0),
                "scripts_detected": wordcloud_data.get('scripts_detected', [])
            }
        else:
            return {
                "status": "error",
                "message": "Multilingual wordcloud generator not available"
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Word cloud generation failed: {str(e)}"
        }

@app.post("/api/summarize")
async def summarize_texts(request: SummarizationRequest):
    """Generate multilingual summaries using mT5 model"""
    try:
        analyzer = get_advanced_analyzer()
        if analyzer:
            summaries = []
            for text in request.texts:
                try:
                    summary = analyzer.summarize_text(
                        text,
                        max_length=request.max_length,
                        min_length=request.min_length,
                        language=request.language
                    )
                    summaries.append({
                        "original_text": text[:200] + "..." if len(text) > 200 else text,
                        "summary": summary,
                        "length_reduction": f"{len(summary)}/{len(text)} chars"
                    })
                except Exception as e:
                    summaries.append({
                        "original_text": text[:200] + "..." if len(text) > 200 else text,
                        "summary": f"Summarization failed: {str(e)}",
                        "error": True
                    })
            
            return {
                "status": "success",
                "summaries": summaries,
                "total_processed": len(request.texts)
            }
        else:
            return {
                "status": "error",
                "message": "Advanced analyzer not available for summarization"
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Summarization failed: {str(e)}"
        }

@app.post("/api/explain")
async def explain_sentiment(request: ExplanationRequest):
    """
    Get detailed explanation for a single text's sentiment
    """
    try:
        if request.use_advanced:
            analyzer = get_advanced_analyzer()
            if analyzer:
                analysis = analyzer.analyze_text(request.text)
                return {
                    "text": request.text,
                    "sentiment": analysis['sentiment'],
                    "polarity_score": analysis['polarity_score'],
                    "confidence": analysis['confidence'],
                    "explanation": analysis['explanation'],
                    "key_indicators": analysis['key_indicators'],
                    "highlighted_words": analysis.get('highlighted_words', []),
                    "highlighted_text": analysis.get('highlighted_text', request.text),
                    "language_info": analysis.get('language_info', {}),
                    "summary": analysis.get('summary', ''),
                    "is_multilingual": analysis.get('is_multilingual', False),
                    "analysis_methods": analysis.get('analysis_methods', [])
                }
            else:
                raise Exception("Advanced analyzer not available")
        else:
            explainer = get_basic_explainer()
            if explainer:
                analysis = explainer(request.text)
                return {
                    "text": request.text,
                    "sentiment": analysis['sentiment'],
                    "polarity_score": analysis['polarity_score'],
                    "confidence": analysis['confidence'],
                    "explanation": analysis['explanation'],
                    "key_indicators": analysis['key_indicators'],
                    "highlighted_words": analysis.get('highlighted_words', []),
                    "highlighted_text": analysis.get('highlighted_text', request.text)
                }
            else:
                raise Exception("Basic explainer not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing text: {str(e)}")

@app.get("/api/stakeholder-analysis")
async def stakeholder_analysis():
    """
    Mock stakeholder analysis endpoint
    """
    return {
        "stakeholder_types": ["Individual", "NGO", "Corporation", "Academic", "Government"],
        "analysis": "Mock stakeholder analysis data"
    }

@app.get("/api/batch-process")
async def batch_process():
    """
    Mock batch processing endpoint
    """
    return {
        "status": "completed",
        "processed": 100,
        "results": "Mock batch processing results"
    }

def _map_confidence_to_float(confidence_str: str) -> float:
    """Map confidence string to float value"""
    mapping = {
        "Very High": 0.95,
        "High": 0.8,
        "Medium": 0.6,
        "Low": 0.4
    }
    return mapping.get(confidence_str, 0.5)

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting lightweight sentiment analysis API...")
    print("📝 Advanced models will load on first use")
    uvicorn.run(app, host="0.0.0.0", port=8001)