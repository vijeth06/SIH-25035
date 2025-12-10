"""
API Service Interface for Enhanced Dashboard
Connects Streamlit frontend to FastAPI backend endpoints
"""

import httpx
import streamlit as st
import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import os

class APIService:
    """Service class for API communication."""
    
    def __init__(self, base_url: str = None):
        # Allow override from environment; default to common local backend port
        self.base_url = base_url or os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
        self.timeout = 30.0
    
    async def health_check(self) -> Dict[str, Any]:
        """Check if backend API is healthy."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Try v1 health first then fallback to /health
                urls = [f"{self.base_url}/api/v1/health", f"{self.base_url}/health"]
                last_error = None
                for url in urls:
                    try:
                        response = await client.get(url)
                        if response.status_code == 200:
                            try:
                                data = response.json()
                            except Exception:
                                data = {"message": response.text}
                            return {"status": "healthy", "data": data}
                    except Exception as e:
                        last_error = str(e)
                        continue
                return {"status": "error", "error": last_error or "Health check failed"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def analyze_sentiment(self, texts: List[str], analysis_type: str = "basic") -> Dict[str, Any]:
        """Perform sentiment analysis on texts using advanced ML analyzer."""
        try:
            # Use our advanced API endpoints
            results = []
            for text in texts:
                payload = {
                    "text": text,
                    "use_advanced": True
                }
                
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{self.base_url}/api/explain",
                        json=payload
                    )
                    if response.status_code == 200:
                        result = response.json()
                        results.append({
                            "text": text,
                            "sentiment": result.get("sentiment", "neutral"),
                            "confidence": result.get("confidence", 0.5),
                            "reasoning": result.get("reasoning", []),
                            "key_indicators": result.get("key_indicators", {})
                        })
                    else:
                        # Fallback for failed requests
                        results.append({
                            "text": text,
                            "sentiment": "neutral",
                            "confidence": 0.5,
                            "reasoning": ["Analysis failed"],
                            "key_indicators": {"positive": [], "negative": []}
                        })
            
            return {"status": "success", "data": {"results": results}}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def policy_sentiment_analysis(self, texts: List[str]) -> Dict[str, Any]:
        """Perform policy-specific sentiment analysis."""
        try:
            payload = {"texts": texts}
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/sentiment/policy-analysis",
                    json=payload
                )
                return {"status": "success", "data": response.json()}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def stakeholder_analysis(self, texts: List[str]) -> Dict[str, Any]:
        """Perform stakeholder categorization and analysis."""
        try:
            payload = {"texts": texts}
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/stakeholder/categorize",
                    json=payload
                )
                return {"status": "success", "data": response.json()}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def legislative_context_analysis(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze legislative context and provision mapping."""
        try:
            payload = {"comments": texts}
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/legislative/provision-mapping",
                    json=payload
                )
                return {"status": "success", "data": response.json()}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def batch_processing_submit(self, texts: List[str], options: List[str]) -> Dict[str, Any]:
        """Submit batch processing job."""
        try:
            payload = {
                "texts": texts,
                "analysis_types": options,
                "priority": "normal"
            }
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/batch/submit",
                    json=payload
                )
                return {"status": "success", "data": response.json()}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def get_batch_status(self, job_id: str) -> Dict[str, Any]:
        """Get batch job status."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/api/v1/batch/status/{job_id}")
                return {"status": "success", "data": response.json()}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def generate_summary(self, texts: List[str], summary_type: str = "policy") -> Dict[str, Any]:
        """Generate text summary."""
        try:
            payload = {
                "texts": texts,
                "summary_type": summary_type,
                "max_length": 200
            }
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/summarization/policy-summarization",
                    json=payload
                )
                return {"status": "success", "data": response.json()}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def analyze_csv_data(self, csv_content: str, text_column: str) -> Dict[str, Any]:
        """Analyze CSV data using our advanced upload endpoint."""
        try:
            payload = {
                "csv_content": csv_content,
                "text_column": text_column,
                "use_advanced": True
            }
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/upload-analyze",
                    json=payload
                )
                if response.status_code == 200:
                    return {"status": "success", "data": response.json()}
                else:
                    return {"status": "error", "error": f"API returned status {response.status_code}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

# Streamlit async helpers
def run_async(coro):
    """Helper to run async functions in Streamlit."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)

# Global API service instance
@st.cache_resource
def get_api_service():
    """Get cached API service instance."""
    return APIService()

def test_api_connection():
    """Test API connection and display status."""
    api = get_api_service()
    
    with st.spinner("Testing API connection..."):
        result = run_async(api.health_check())
    
    if result["status"] == "healthy":
        st.success("✅ Backend API is connected and healthy!")
        return True
    else:
        st.error(f"❌ Backend API connection failed: {result.get('error', 'Unknown error')}")
        st.info("💡 Make sure the backend server is running on http://127.0.0.1:8000")
        return False

def perform_api_sentiment_analysis(texts: List[str], analysis_type: str = "basic"):
    """Perform sentiment analysis via API."""
    api = get_api_service()
    
    with st.spinner(f"Performing {analysis_type} sentiment analysis..."):
        if analysis_type == "policy-specific":
            result = run_async(api.policy_sentiment_analysis(texts))
        else:
            result = run_async(api.analyze_sentiment(texts, analysis_type))
    
    if result["status"] == "success":
        return result["data"]
    else:
        st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
        return None

def perform_stakeholder_analysis_api(texts: List[str]):
    """Perform stakeholder analysis via API."""
    api = get_api_service()
    
    with st.spinner("Analyzing stakeholder categories..."):
        result = run_async(api.stakeholder_analysis(texts))
    
    if result["status"] == "success":
        return result["data"]
    else:
        st.error(f"❌ Stakeholder analysis failed: {result.get('error', 'Unknown error')}")
        return None

def perform_legislative_analysis_api(texts: List[str]):
    """Perform legislative context analysis via API."""
    api = get_api_service()
    
    with st.spinner("Analyzing legislative context..."):
        result = run_async(api.legislative_context_analysis(texts))
    
    if result["status"] == "success":
        return result["data"]
    else:
        st.error(f"❌ Legislative analysis failed: {result.get('error', 'Unknown error')}")
        return None

def submit_batch_job_api(texts: List[str], options: List[str]):
    """Submit batch processing job via API."""
    api = get_api_service()
    
    with st.spinner("Submitting batch processing job..."):
        result = run_async(api.batch_processing_submit(texts, options))
    
    if result["status"] == "success":
        return result["data"]
    else:
        st.error(f"❌ Batch submission failed: {result.get('error', 'Unknown error')}")
        return None

def generate_summary_api(texts: List[str], summary_type: str = "policy"):
    """Generate summary via API."""
    api = get_api_service()
    
    with st.spinner(f"Generating {summary_type} summary..."):
        result = run_async(api.generate_summary(texts, summary_type))
    
    if result["status"] == "success":
        return result["data"]
    else:
        st.error(f"❌ Summary generation failed: {result.get('error', 'Unknown error')}")
        return None