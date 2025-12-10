"""
Advanced Sentiment Analysis Dashboard

This module provides an interactive dashboard for advanced sentiment analysis features.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from typing import List, Dict, Any
import json
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Advanced Sentiment Analysis | E-Consultation Insight",
    page_icon="🧠",
    layout="wide"
)

# This page is deprecated in the single-page experience.
st.markdown("### ℹ️ Advanced Analysis page has been removed.\nAll features are now available on the single-page dashboard.")
st.markdown("[🏠 Go to Dashboard](../main.py)")
st.stop()

# Custom CSS
st.markdown("""
    <style>
    .main {
        max-width: 1200px;
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
    }
    .stTextArea>div>div>textarea {
        min-height: 150px;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .emotion-bar {
        height: 8px;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# API URL (with fallback)
try:
    API_URL = st.secrets.get("API_URL", "http://127.0.0.1:8000")
except:
    API_URL = "http://127.0.0.1:8000"

# Import auth module to get headers
from dashboard.components.auth import get_headers

def analyze_text(text: str) -> Dict[str, Any]:
    """Send text to the advanced analysis API."""
    try:
        # Check if user is authenticated
        if not st.session_state.get('access_token'):
            st.error("❌ Authentication required. Please log in first.")
            return None
            
        response = requests.post(
            f"{API_URL}/api/v1/advanced/analyze",
            json={"text": text},
            headers=get_headers()
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        if response.status_code == 401:
            st.error("❌ Authentication failed. Please log in again.")
        elif response.status_code == 404:
            st.error("❌ Advanced analysis endpoint not found. Please check if the backend service is running.")
        else:
            st.error(f"Error analyzing text: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error analyzing text: {str(e)}")
        return None

def analyze_batch(texts: List[str]) -> Dict[str, Any]:
    """Send batch of texts to the advanced analysis API."""
    try:
        # Check if user is authenticated
        if not st.session_state.get('access_token'):
            st.error("❌ Authentication required. Please log in first.")
            return None
            
        response = requests.post(
            f"{API_URL}/api/v1/advanced/analyze/batch",
            json={"texts": texts},
            headers=get_headers()
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        if response.status_code == 401:
            st.error("❌ Authentication failed. Please log in again.")
        elif response.status_code == 404:
            st.error("❌ Batch analysis endpoint not found. Please check if the backend service is running.")
        else:
            st.error(f"Error analyzing batch: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error analyzing batch: {str(e)}")
        return None

def display_sentiment_analysis(result: Dict[str, Any]) -> None:
    """Display the sentiment analysis results in an interactive way."""
    if not result or "data" not in result:
        st.error("No analysis results to display")
        return
    
    data = result["data"]
    
    # Overall sentiment
    sentiment = data["overall_sentiment"].capitalize()
    sentiment_score = data["sentiment_score"]
    
    # Sentiment gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sentiment_score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Sentiment Score"},
        gauge={
            'axis': {'range': [-100, 100]},
            'bar': {'color': "#4e79a7"},
            'steps': [
                {'range': [-100, -50], 'color': '#e15759'},
                {'range': [-50, 0], 'color': '#f28e2b'},
                {'range': [0, 50], 'color': '#59a14f'},
                {'range': [50, 100], 'color': '#4e79a7'},
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': sentiment_score * 100
            }
        }
    ))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Emotion analysis
    st.subheader("Emotion Analysis")
    emotions = data.get("emotions", {})
    
    if emotions:
        emotion_df = pd.DataFrame({
            "Emotion": list(emotions.keys()),
            "Score": list(emotions.values())
        }).sort_values("Score", ascending=False)
        
        fig = px.bar(
            emotion_df, 
            x="Score", 
            y="Emotion", 
            orientation='h',
            title="Emotion Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Aspect analysis
    aspects = data.get("aspects", [])
    if aspects:
        st.subheader("Aspect Analysis")
        aspect_df = pd.DataFrame(aspects)
        
        if not aspect_df.empty:
            fig = px.treemap(
                aspect_df,
                path=['aspect'],
                values='score',
                color='sentiment',
                color_discrete_map={
                    'POSITIVE': '#4e79a7',
                    'NEGATIVE': '#e15759',
                    'NEUTRAL': '#f28e2b'
                },
                title="Aspect Sentiment Analysis"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Raw JSON (collapsible)
    with st.expander("View Raw Analysis"):
        st.json(data)

def main():
    """Main function for the advanced analysis dashboard."""
    st.title("🧠 Advanced Sentiment Analysis")
    st.markdown("""
        Analyze text with advanced NLP models to detect sentiment, emotions, aspects, and more.
        Enter your text below or upload a file for batch analysis.
    """)
    
    # Tab layout
    tab1, tab2 = st.tabs(["Single Text Analysis", "Batch Analysis"])
    
    with tab1:
        st.subheader("Analyze Single Text")
        text = st.text_area(
            "Enter your text here",
            placeholder="Type or paste your text here...",
            height=200
        )
        
        if st.button("Analyze Sentiment", key="analyze_single"):
            if text.strip():
                with st.spinner("Analyzing text..."):
                    result = analyze_text(text)
                    if result and result.get("success"):
                        display_sentiment_analysis(result)
                    else:
                        st.error("Failed to analyze text. Please try again.")
            else:
                st.warning("Please enter some text to analyze.")
    
    with tab2:
        st.subheader("Batch Analysis")
        uploaded_file = st.file_uploader(
            "Upload a text file (one text per line)",
            type=["txt", "csv", "json"]
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.type == "application/json":
                    data = json.load(uploaded_file)
                    texts = data.get("texts", [])
                else:
                    content = uploaded_file.getvalue().decode("utf-8")
                    texts = [line.strip() for line in content.split("\n") if line.strip()]
                
                if texts:
                    st.success(f"Loaded {len(texts)} texts for analysis.")
                    
                    if st.button("Analyze Batch", key="analyze_batch"):
                        with st.spinner(f"Analyzing {len(texts)} texts..."):
                            result = analyze_batch(texts)
                            
                            if result and result.get("success"):
                                st.subheader("Batch Analysis Results")
                                
                                # Summary statistics
                                sentiments = [r["overall_sentiment"] for r in result["results"]]
                                sentiment_counts = pd.Series(sentiments).value_counts().reset_index()
                                sentiment_counts.columns = ["Sentiment", "Count"]
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # Sentiment distribution
                                    fig = px.pie(
                                        sentiment_counts,
                                        values="Count",
                                        names="Sentiment",
                                        title="Sentiment Distribution"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                with col2:
                                    # Emotion heatmap
                                    emotions = []
                                    for r in result["results"]:
                                        emotions.extend(r["emotions"].items())
                                    
                                    if emotions:
                                        emotion_df = pd.DataFrame(emotions, columns=["Emotion", "Score"])
                                        emotion_heatmap = emotion_df.pivot_table(
                                            index=None,
                                            columns="Emotion",
                                            values="Score",
                                            aggfunc="mean"
                                        )
                                        fig = px.imshow(
                                            emotion_heatmap,
                                            title="Average Emotion Scores",
                                            color_continuous_scale="Viridis"
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                
                                # Detailed results in an expandable section
                                with st.expander("View Detailed Results"):
                                    for i, r in enumerate(result["results"][:10], 1):  # Show first 10 results
                                        with st.container():
                                            st.markdown(f"### Text {i}")
                                            st.text(r["text"][:200] + ("..." if len(r["text"]) > 200 else ""))
                                            
                                            col1, col2 = st.columns(2)
                                            with col1:
                                                st.metric("Sentiment", r["overall_sentiment"].capitalize())
                                            with col2:
                                                st.metric("Confidence", f"{r['confidence']*100:.1f}%")
                                            
                                            st.progress(r["sentiment_score"] / 2 + 0.5)
                                            st.markdown("---")
                            else:
                                st.error("Failed to analyze batch. Please try again.")
                else:
                    st.warning("No valid texts found in the uploaded file.")
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()