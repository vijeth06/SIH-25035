"""
Simplified FastAPI application for sentiment analysis API.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import sys
import os

# Add services to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app', 'services'))

try:
    from sentiment_explainer import analyze_text_with_explanation
    from advanced_sentiment_analyzer import analyze_text_advanced, ADVANCED_ANALYZER_AVAILABLE
    from multilingual_wordcloud import MultilingualWordCloudGenerator, WORDCLOUD_GENERATOR_AVAILABLE
    EXPLAINER_AVAILABLE = True
    WORDCLOUD_AVAILABLE = WORDCLOUD_GENERATOR_AVAILABLE
    print("✅ Sentiment explainer loaded successfully")
    print(f"✅ Advanced analyzer available: {ADVANCED_ANALYZER_AVAILABLE}")
    print(f"✅ Word cloud generator available: {WORDCLOUD_AVAILABLE}")
except ImportError as e:
    EXPLAINER_AVAILABLE = False
    ADVANCED_ANALYZER_AVAILABLE = False
    WORDCLOUD_AVAILABLE = False
    print(f"❌ Sentiment explainer import failed: {e}")
    
    # Individual imports as fallback
    try:
        from sentiment_explainer import analyze_text_with_explanation
        EXPLAINER_AVAILABLE = True
        print("✅ Basic sentiment explainer loaded")
    except ImportError:
        pass
        
    try:
        from advanced_sentiment_analyzer import analyze_text_advanced, ADVANCED_ANALYZER_AVAILABLE
        print(f"✅ Advanced analyzer loaded: {ADVANCED_ANALYZER_AVAILABLE}")
    except ImportError:
        ADVANCED_ANALYZER_AVAILABLE = False
        
    try:
        from multilingual_wordcloud import MultilingualWordCloudGenerator
        WORDCLOUD_AVAILABLE = True
        print("✅ Wordcloud generator loaded")
    except ImportError:
        WORDCLOUD_AVAILABLE = False

app = FastAPI(
    title="MCA eConsultation Sentiment Analysis API",
    description="API for analyzing sentiment in consultation comments with explanations",
    version="1.0.0"
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

class ExplanationRequest(BaseModel):
    text: str

class SummarizationRequest(BaseModel):
    texts: List[str]
    max_length: int = 150
    min_length: int = 50
    language: str = "auto"

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "MCA eConsultation Sentiment Analysis API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "explainer_available": EXPLAINER_AVAILABLE,
        "advanced_analyzer_available": ADVANCED_ANALYZER_AVAILABLE,
        "wordcloud_available": WORDCLOUD_AVAILABLE,
        "services": ["sentiment_analysis", "explanation", "wordcloud", "summarization"]
    }

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_sentiment(request: TextAnalysisRequest):
    """
    Analyze sentiment for multiple texts
    """
    if not EXPLAINER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Sentiment explainer service not available")
    
    results = []
    sentiments = {"positive": 0, "negative": 0, "neutral": 0}
    
    for text in request.texts:
        try:
            # Use advanced analyzer if available
            if ADVANCED_ANALYZER_AVAILABLE:
                analysis = analyze_text_advanced(text)
                
                sentiment_result = SentimentResult(
                    text=text,
                    sentiment=analysis['sentiment'].lower(),
                    confidence=analysis['confidence'],
                    polarity_score=analysis['polarity_score']
                )
                
                # Add explanation if requested
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
                # Fallback to basic explainer
                analysis = analyze_text_with_explanation(text)
                
                sentiment_result = SentimentResult(
                    text=text,
                    sentiment=analysis['sentiment'].lower(),
                    confidence=_map_confidence_to_float(analysis['confidence']),
                    polarity_score=analysis['polarity_score']
                )
                
                # Add explanation if requested
                if request.include_explanation:
                    sentiment_result.explanation = {
                        "detailed_explanation": analysis['explanation'],
                        "key_indicators": analysis['key_indicators'],
                        "highlighted_words": analysis.get('highlighted_words', []),
                        "highlighted_text": analysis.get('highlighted_text', text),
                        "method": analysis.get('method', 'textblob')
                    }
            
            results.append(sentiment_result)
            sentiments[sentiment_result.sentiment] += 1
            
        except Exception as e:
            # Fallback to basic sentiment
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
        if WORDCLOUD_AVAILABLE:
            # Try to import dynamically
            try:
                from multilingual_wordcloud import MultilingualWordCloudGenerator
                generator = MultilingualWordCloudGenerator()
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
            except ImportError:
                return {
                    "status": "error",
                    "message": "Multilingual wordcloud module not found"
                }
        else:
            return {
                "status": "error",
                "message": "Multilingual wordcloud functionality not available"
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
        if ADVANCED_ANALYZER_AVAILABLE:
            # Try to import dynamically
            try:
                from advanced_sentiment_analyzer import AdvancedMultilingualSentimentAnalyzer
                analyzer = AdvancedMultilingualSentimentAnalyzer()
                
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
            except ImportError:
                return {
                    "status": "error",
                    "message": "Advanced analyzer module not found for summarization"
                }
        else:
            return {
                "status": "error",
                "message": "Advanced multilingual summarization not available"
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
    if not EXPLAINER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Sentiment explainer service not available")
    
    try:
        analysis = analyze_text_with_explanation(request.text)
        return {
            "text": request.text,
            "sentiment": analysis['sentiment'],
            "polarity_score": analysis['polarity_score'],
            "confidence": analysis['confidence'],
            "subjectivity_score": analysis['subjectivity_score'],
            "explanation": analysis['explanation'],
            "key_indicators": analysis['key_indicators'],
            "phrases": analysis['phrases']
        }
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
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)