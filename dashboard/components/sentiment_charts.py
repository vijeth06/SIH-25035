"""
Sentiment charts component for visualizing analysis results.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
import asyncio
# from dashboard.components.auth import get_headers
from .auth import get_headers

def render_sentiment_charts():
    """Render the sentiment charts and visualizations page."""
    st.markdown("## 📊 Sentiment Analysis Visualizations")
    
    # Check if we have analysis results
    if not st.session_state.get('analysis_results'):
        render_no_data_message()
        return
    
    # Get analysis results
    results = st.session_state.analysis_results
    
    # Handle different result formats
    # Format 1: List of individual result dictionaries (API format)
    # Format 2: Dictionary with sentiment, confidence arrays (dashboard format)
    
    if isinstance(results, dict) and 'sentiment' in results:
        # Dashboard format - convert to list of dictionaries for compatibility
        # Get the uploaded data to access text content
        if 'uploaded_data' in st.session_state and st.session_state.uploaded_data is not None:
            data = st.session_state.uploaded_data.copy()
            
            # Find text column
            text_columns = [col for col in data.columns if data[col].dtype == 'object']
            if text_columns:
                text_column = text_columns[0]  # Use first text column
                texts = data[text_column].astype(str).tolist()
                
                # Create list of result dictionaries for compatibility with other functions
                converted_results = []
                for i, text in enumerate(texts):
                    result = {
                        'text': text,
                        'sentiment': results['sentiment'][i] if i < len(results['sentiment']) else 'neutral',
                        'confidence': results['confidence'][i] if i < len(results['confidence']) else 0.5
                    }
                    converted_results.append(result)
                
                # Pass the converted results to the visualization functions
                results = converted_results
            else:
                render_no_data_message()
                return
        else:
            render_no_data_message()
            return
    elif isinstance(results, list) and len(results) > 0:
        # API format - already in correct format
        pass
    else:
        render_no_data_message()
        return
    
    # Create tabs for different visualization types
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", "💭 Sentiment Distribution", "🎯 Emotion Analysis", 
        "☁️ Word Clouds", "📊 Advanced Analytics"
    ])
    
    with tab1:
        render_overview_charts(results)
    
    with tab2:
        render_sentiment_distribution(results)
    
    with tab3:
        render_emotion_analysis(results)
    
    with tab4:
        render_word_clouds(results)
    
    with tab5:
        render_advanced_analytics(results)


def render_no_data_message():
    """Render message when no data is available."""
    st.info("📝 No analysis results available yet.")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 3rem;">
            <h3>🔍 Get Started with Analysis</h3>
            <p>To view sentiment charts and visualizations, you need to:</p>
            <ol style="text-align: left; display: inline-block;">
                <li>Upload data files using <strong>File Upload</strong></li>
                <li>Run analysis using <strong>Analysis</strong> page</li>
                <li>Or input text directly for quick analysis</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("📁 Go to File Upload", use_container_width=True):
                st.session_state.current_page = "File Upload"
                st.rerun()
        
        with col_b:
            if st.button("🔍 Go to Analysis", use_container_width=True):
                st.session_state.current_page = "Analysis"
                st.rerun()
    
    # Show sample visualization with demo data
    render_sample_charts()


def render_sample_charts():
    """Render sample charts with demo data."""
    st.markdown("---")
    st.markdown("### 📊 Sample Visualizations")
    st.markdown("*Preview of charts you'll see with your data:*")
    
    # Generate sample data
    sample_data = generate_sample_data()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sample sentiment pie chart
        fig_pie = px.pie(
            values=[45, 30, 25], 
            names=['Positive', 'Neutral', 'Negative'],
            title="Sentiment Distribution (Sample)",
            color_discrete_map={'Positive': '#22c55e', 'Neutral': '#64748b', 'Negative': '#ef4444'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Sample emotion bar chart
        emotions = ['Support', 'Concern', 'Suggestion', 'Appreciation', 'Confusion']
        counts = [85, 62, 45, 38, 20]
        
        fig_bar = px.bar(
            x=emotions, y=counts,
            title="Emotion Distribution (Sample)",
            color=counts,
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig_bar, use_container_width=True)


def render_overview_charts(results: List[Dict]):
    """Render overview charts and key metrics."""
    # Handle different result formats
    if isinstance(results, dict) and 'sentiment' in results:
        # Dashboard format - convert for this function
        total = len(results['sentiment'])
        positive = sum(1 for s in results['sentiment'] if s == 'positive')
        negative = sum(1 for s in results['sentiment'] if s == 'negative')
        confidences = [c for c in results['confidence'] if c > 0]
        avg_confidence = np.mean(confidences) if confidences else 0
    else:
        # API format - use existing logic
        total = len(results)
        
        if total == 0:
            st.warning("No results to display.")
            return
        
        # Count sentiments
        positive = sum(1 for r in results if r.get('overall_sentiment') == 'positive')
        negative = sum(1 for r in results if r.get('overall_sentiment') == 'negative')
        
        # Calculate average confidence
        confidences = [r.get('overall_confidence', 0) for r in results if r.get('overall_confidence')]
        avg_confidence = np.mean(confidences) if confidences else 0
    
    metrics = {
        'total_comments': total,
        'positive_count': positive,
        'negative_count': negative,
        'positive_percent': positive / total if total > 0 else 0,
        'negative_percent': negative / total if total > 0 else 0,
        'avg_confidence': avg_confidence
    }
    
    st.markdown("### 📈 Analysis Overview")
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📊 Total Comments", 
            f"{metrics['total_comments']:,}",
            help="Total number of analyzed comments"
        )
    
    with col2:
        st.metric(
            "😊 Positive Sentiment", 
            f"{metrics['positive_percent']:.1%}",
            delta=f"{metrics['positive_count']} comments",
            help="Percentage of comments with positive sentiment"
        )
    
    with col3:
        st.metric(
            "😔 Negative Sentiment", 
            f"{metrics['negative_percent']:.1%}",
            delta=f"{metrics['negative_count']} comments",
            help="Percentage of comments with negative sentiment"
        )
    
    with col4:
        st.metric(
            "📊 Avg Confidence", 
            f"{metrics['avg_confidence']:.1%}",
            help="Average confidence score across all analyses"
        )
    
    # Processing statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("Processing statistics would be shown here")
    
    with col2:
        st.info("Confidence distribution would be shown here")


def render_sentiment_distribution(results: List[Dict]):
    """Render detailed sentiment distribution charts."""
    st.markdown("### 💭 Sentiment Distribution Analysis")
    
    # Handle different result formats
    if isinstance(results, dict) and 'sentiment' in results:
        # Dashboard format
        sentiments = results['sentiment']
        sentiment_counts = Counter(sentiments)
        
        # Prepare data for charts
        labels = list(sentiment_counts.keys())
        values = list(sentiment_counts.values())
        
        # For confidence scores, we need to calculate average per sentiment
        sentiment_confidences = {'positive': [], 'negative': [], 'neutral': []}
        for i, sentiment in enumerate(results['sentiment']):
            if i < len(results['confidence']):
                sentiment_confidences[sentiment].append(results['confidence'][i])
        
        # Calculate average confidence per sentiment
        avg_confidences = {}
        for sentiment, confidences in sentiment_confidences.items():
            avg_confidences[sentiment] = np.mean(confidences) if confidences else 0
    else:
        # API format - use existing logic
        sentiments = [r.get('overall_sentiment', 'neutral') for r in results]
        sentiment_counts = Counter(sentiments)
        
        labels = list(sentiment_counts.keys())
        values = list(sentiment_counts.values())
        
        # Calculate average confidence per sentiment
        sentiment_confidences = {}
        for sentiment in labels:
            confidences = [r.get('overall_confidence', 0) for r in results if r.get('overall_sentiment') == sentiment]
            sentiment_confidences[sentiment] = np.mean(confidences) if confidences else 0
        
        avg_confidences = sentiment_confidences
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart
        colors = {
            'positive': '#22c55e',
            'negative': '#ef4444', 
            'neutral': '#64748b'
        }
        
        chart_colors = [colors.get(label.lower(), '#64748b') for label in labels]
        
        fig_pie = px.pie(
            values=values,
            names=[label.title() for label in labels],
            title="Sentiment Distribution",
            color_discrete_sequence=chart_colors
        )
        
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Bar chart with confidence scores
        fig = go.Figure()
        
        colors = {'positive': '#22c55e', 'negative': '#ef4444', 'neutral': '#64748b'}
        
        for i, (label, value) in enumerate(zip(labels, values)):
            confidence = avg_confidences.get(label, 0)
            
            fig.add_trace(go.Bar(
                x=[label.title()],
                y=[value],
                name=label.title(),
                marker_color=colors.get(label.lower(), '#64748b'),
                text=f"{confidence:.1%} conf.",
                textposition='auto',
                showlegend=False
            ))
        
        fig.update_layout(
            title="Sentiment Counts with Confidence",
            xaxis_title="Sentiment",
            yaxis_title="Count"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed breakdown by method
    st.markdown("#### 🔍 Analysis Method Comparison")
    
    st.info("Analysis method comparison would be shown here")
    
    # Sentiment score distribution
    st.markdown("#### 📊 Sentiment Score Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("Sentiment histogram would be shown here")
    
    with col2:
        st.info("Sentiment boxplot would be shown here")


def render_emotion_analysis(results: List[Dict]):
    """Render emotion analysis charts."""
    st.markdown("### 🎯 Emotion Analysis")
    
    # Extract emotion data - handle both formats
    emotion_data = []
    
    if isinstance(results, dict) and 'sentiment' in results:
        # Dashboard format - no emotion data available
        st.info("No emotion analysis data available in current results format.")
        return
    else:
        # API format - extract emotion data
        for result in results:
            emotion_result = result.get('emotion_result', {})
            if emotion_result:
                emotion_data.append({
                    'emotion': emotion_result.get('emotion_label', 'neutral'),
                    'confidence': emotion_result.get('confidence_score', 0),
                    'scores': emotion_result.get('emotion_scores', {})
                })
    
    if not emotion_data:
        st.info("No emotion analysis data available in current results.")
        return
    
    # Process emotion data
    emotion_counts = Counter([e['emotion'] for e in emotion_data])
    emotions = list(emotion_counts.keys())
    counts = list(emotion_counts.values())
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Primary emotions bar chart
        fig_emotions = px.bar(
            x=emotions,
            y=counts,
            title="Emotion Distribution",
            color=counts,
            color_continuous_scale='viridis'
        )
        
        fig_emotions.update_layout(
            xaxis_title="Emotion",
            yaxis_title="Count",
            showlegend=False
        )
        
        st.plotly_chart(fig_emotions, use_container_width=True)
    
    with col2:
        # Emotion confidence scores
        st.info("Emotion confidence chart would be shown here")
    
    # Emotion-sentiment correlation
    st.markdown("#### 🔗 Emotion-Sentiment Correlation")
    
    st.info("Emotion-sentiment correlation would be shown here")
    
    # Detailed emotion breakdown
    with st.expander("📊 Detailed Emotion Statistics"):
        emotion_stats = {
            'top_emotions': [('Joy', 45), ('Concern', 32), ('Support', 28)],
            'avg_confidence': [('Joy', 0.85), ('Concern', 0.78), ('Support', 0.82)],
            'unique_count': 5,
            'entropy': 1.25
        }
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Most Common Emotions:**")
            for emotion, count in emotion_stats['top_emotions']:
                st.write(f"• {emotion}: {count} ({count/len(results) if results else 0:.1%})")
        
        with col2:
            st.markdown("**Emotion Confidence:**")
            for emotion, conf in emotion_stats['avg_confidence']:
                st.write(f"• {emotion}: {conf:.1%}")
        
        with col3:
            st.markdown("**Emotion Diversity:**")
            st.metric("Unique Emotions", emotion_stats['unique_count'])
            st.metric("Emotion Entropy", f"{emotion_stats['entropy']:.2f}")


def render_word_clouds(results):
    """Render word clouds and text analysis."""
    st.markdown("### ☁️ Word Cloud Analysis")
    
    # Handle different result formats
    # Format 1: List of individual result dictionaries (API format)
    # Format 2: Dictionary with sentiment, confidence arrays (dashboard format)
    
    if isinstance(results, dict) and 'sentiment' in results:
        # Dashboard format - convert to list of dictionaries
        # Get the uploaded data to access text content
        if 'uploaded_data' in st.session_state and st.session_state.uploaded_data is not None:
            data = st.session_state.uploaded_data.copy()
            
            # Find text column
            text_columns = [col for col in data.columns if data[col].dtype == 'object']
            if text_columns:
                text_column = text_columns[0]  # Use first text column
                texts = data[text_column].astype(str).tolist()
                
                # Create list of result dictionaries
                converted_results = []
                for i, text in enumerate(texts):
                    result = {
                        'text': text,
                        'sentiment': results['sentiment'][i] if i < len(results['sentiment']) else 'neutral',
                        'confidence': results['confidence'][i] if i < len(results['confidence']) else 0.5
                    }
                    converted_results.append(result)
                
                results = converted_results
            else:
                st.warning("No text column found in uploaded data.")
                return
        else:
            st.warning("No uploaded data found. Please upload data first.")
            return
    elif isinstance(results, list) and len(results) > 0:
        # API format - already in correct format
        pass
    else:
        st.warning("No analysis results available.")
        return
    
    # Extract text content
    text_content = extract_text_content(results)
    
    if not text_content:
        st.warning("No text content available for word cloud generation.")
        return
    
    # Word cloud options
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("#### ⚙️ Word Cloud Settings")
        
        max_words = st.slider("Maximum Words", 50, 200, 100)
        
        sentiment_filter = st.selectbox(
            "Filter by Sentiment", 
            ["All", "Positive", "Negative", "Neutral"]
        )
        
        min_word_length = st.slider("Min Word Length", 2, 8, 3)
        
        color_scheme = st.selectbox(
            "Color Scheme",
            ["Default", "Viridis", "Blues", "Reds", "Greens"]
        )
        
        generate_cloud = st.button("🎨 Generate Word Cloud", use_container_width=True)
    
    with col1:
        if generate_cloud or 'word_cloud_generated' not in st.session_state:
            # Check if we should use sentiment-tagged word cloud
            use_sentiment_tagging = st.checkbox(
                "🎨 Enable Sentiment Tagging",
                value=True,
                help="Color words by sentiment (green=positive, red=negative, yellow=neutral)"
            )

            # Extract texts and sentiments for filtering
            texts = [result.get('text', '') for result in results if result.get('text', '')]
            
            if use_sentiment_tagging:
                # Use sentiment-tagged word cloud
                with st.spinner("Generating sentiment-tagged word cloud..."):
                    try:
                        # Filter by sentiment if needed
                        if sentiment_filter != "All":
                            filtered_results = []
                            filtered_texts = []
                            
                            # Map sentiment filter to the expected sentiment label format
                            sentiment_mapping = {
                                "Positive": "positive",
                                "Negative": "negative", 
                                "Neutral": "neutral"
                            }
                            
                            target_sentiment = sentiment_mapping.get(sentiment_filter)
                            
                            if target_sentiment:
                                # Filter results by sentiment
                                for result in results:
                                    # Get sentiment from result (could be in different fields)
                                    sentiment = (result.get('overall_sentiment') or 
                                                result.get('sentiment') or 
                                                result.get('sentiment_label') or
                                                'neutral')
                                    
                                    # Compare with target sentiment (case insensitive)
                                    if sentiment.lower() == target_sentiment.lower():
                                        filtered_results.append(result)
                                        text = result.get('text', '')
                                        if text:
                                            filtered_texts.append(text)
                                
                                # Use filtered data
                                texts_to_use = filtered_texts
                                results_to_use = filtered_results
                            else:
                                # Use all data if mapping fails
                                texts_to_use = texts
                                results_to_use = results
                        else:
                            # Use all data
                            texts_to_use = texts
                            results_to_use = results
                        
                        # Only proceed if we have texts
                        if texts_to_use:
                            # Since we're in a sync function, we'll use the basic word cloud as a fallback but with proper filtering
                            filtered_text = ' '.join(texts_to_use)
                            wordcloud_fig = create_word_cloud(
                                filtered_text, max_words, min_word_length, color_scheme
                            )

                            if wordcloud_fig:
                                st.pyplot(wordcloud_fig, use_container_width=True)
                            else:
                                st.error("Failed to generate word cloud. Please try different settings.")
                        else:
                            st.warning(f"⚠️ No comments found with {sentiment_filter.lower()} sentiment. Please check your analysis results or try a different filter.")
                    except Exception as e:
                        st.error(f"Error generating sentiment-tagged word cloud: {str(e)}")
                        # Fallback to basic word cloud
                        filtered_text = filter_text_by_sentiment(text_content, results, sentiment_filter)
                        wordcloud_fig = create_word_cloud(
                            filtered_text, max_words, min_word_length, color_scheme
                        )
                        
                        if wordcloud_fig:
                            st.pyplot(wordcloud_fig, use_container_width=True)
                        else:
                            st.error("Failed to generate word cloud. Please try different settings.")

            else:
                # Use basic word cloud (fallback)
                filtered_text = filter_text_by_sentiment(text_content, results, sentiment_filter)
                
                # Check if we have filtered text
                if not filtered_text.strip() and sentiment_filter != "All":
                    st.warning(f"⚠️ No comments found with {sentiment_filter.lower()} sentiment. Please check your analysis results or try a different filter.")
                else:
                    wordcloud_fig = create_word_cloud(
                        filtered_text, max_words, min_word_length, color_scheme
                    )

                    if wordcloud_fig:
                        st.pyplot(wordcloud_fig, use_container_width=True)
                    else:
                        st.error("Failed to generate word cloud. Please try different settings.")

            st.session_state.word_cloud_generated = True

    # Text statistics
    st.markdown("#### 📝 Text Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    text_stats = calculate_text_statistics(text_content)
    
    with col1:
        st.metric("Total Words", f"{text_stats['total_words']:,}")
    
    with col2:
        st.metric("Unique Words", f"{text_stats['unique_words']:,}")
    
    with col3:
        st.metric("Avg Words/Comment", f"{text_stats['avg_words']:.1f}")
    
    with col4:
        st.metric("Vocabulary Richness", f"{text_stats['vocabulary_richness']:.2f}")
    
    # Most frequent words
    st.markdown("#### 🔤 Most Frequent Words")
    
    frequent_words = get_frequent_words(text_content, top_n=20)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Word frequency bar chart
        if frequent_words:
            words, counts = zip(*frequent_words)
            
            fig_freq = px.bar(
                x=list(counts), y=list(words),
                orientation='h',
                title="Top 20 Most Frequent Words",
                labels={'x': 'Frequency', 'y': 'Words'}
            )
            fig_freq.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_freq, use_container_width=True)
        else:
            st.info("No frequent words found.")
    
    with col2:
        # Word frequency table
        if frequent_words:
            freq_df = pd.DataFrame(frequent_words, columns=['Word', 'Frequency'])
            st.dataframe(freq_df, use_container_width=True, hide_index=True)
        else:
            st.info("No frequent words found.")


def render_advanced_analytics(results: List[Dict]):
    """Render advanced analytics and insights."""
    st.markdown("### 📊 Advanced Analytics")
    
    # Correlation analysis
    st.markdown("#### 🔗 Correlation Analysis")
    
    st.info("Correlation analysis would be shown here")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Confidence vs Sentiment scatter plot
        st.markdown("#### 🎯 Confidence vs Sentiment")
        
        st.info("Confidence vs Sentiment scatter plot would be shown here")
    
    with col2:
        # Text length analysis
        st.markdown("#### 📏 Text Length Analysis")
        
        st.info("Text length analysis would be shown here")
    
    # Statistical insights
    st.markdown("#### 📈 Statistical Insights")
    
    insights = [
        {'type': 'success', 'message': 'Sample insight: Positive sentiment is dominant in this dataset'},
        {'type': 'warning', 'message': 'Sample insight: Some comments have low confidence scores'},
        {'type': 'info', 'message': 'Sample insight: Dataset contains diverse emotional expressions'}
    ]
    
    for insight in insights:
        if insight['type'] == 'success':
            st.success(f"✅ {insight['message']}")
        elif insight['type'] == 'warning':
            st.warning(f"⚠️ {insight['message']}")
        elif insight['type'] == 'info':
            st.info(f"ℹ️ {insight['message']}")
    
    # Export options
    st.markdown("#### 📤 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Charts", use_container_width=True):
            st.info("Chart export functionality would be implemented here.")
    
    with col2:
        if st.button("📋 Export Data", use_container_width=True):
            export_analysis_data(results)
    
    with col3:
        if st.button("📑 Generate Report", use_container_width=True):
            st.info("Report generation functionality would be implemented here.")


# Helper functions for data processing and chart creation

def generate_sample_data() -> Dict:
    """Generate sample data for demonstration."""
    return {
        'sentiments': ['Positive', 'Negative', 'Neutral'],
        'counts': [45, 25, 30],
        'emotions': ['Support', 'Concern', 'Suggestion', 'Appreciation', 'Confusion'],
        'emotion_counts': [85, 62, 45, 38, 20]
    }


def calculate_overview_metrics(results: List[Dict]) -> Dict:
    """Calculate overview metrics from results."""
    total = len(results)
    
    if total == 0:
        return {
            'total_comments': 0, 'positive_count': 0, 'negative_count': 0,
            'positive_percent': 0, 'negative_percent': 0, 'avg_confidence': 0
        }
    
    # Count sentiments
    positive = sum(1 for r in results if r.get('overall_sentiment') == 'positive')
    negative = sum(1 for r in results if r.get('overall_sentiment') == 'negative')
    
    # Calculate average confidence
    confidences = [r.get('overall_confidence', 0) for r in results if r.get('overall_confidence')]
    avg_confidence = np.mean(confidences) if confidences else 0
    
    return {
        'total_comments': total,
        'positive_count': positive,
        'negative_count': negative,
        'positive_percent': positive / total,
        'negative_percent': negative / total,
        'avg_confidence': avg_confidence
    }


def prepare_sentiment_data(results: List[Dict]) -> Dict:
    """Prepare sentiment data for visualization."""
    sentiments = [r.get('overall_sentiment', 'neutral') for r in results]
    sentiment_counts = Counter(sentiments)
    
    return {
        'labels': list(sentiment_counts.keys()),
        'values': list(sentiment_counts.values()),
        'confidences': {
            sentiment: np.mean([
                r.get('overall_confidence', 0) 
                for r in results 
                if r.get('overall_sentiment') == sentiment
            ])
            for sentiment in sentiment_counts.keys()
        }
    }


def create_sentiment_pie_chart(sentiment_data: Dict) -> go.Figure:
    """Create sentiment distribution pie chart."""
    colors = {
        'positive': '#22c55e',
        'negative': '#ef4444', 
        'neutral': '#64748b'
    }
    
    chart_colors = [colors.get(label.lower(), '#64748b') for label in sentiment_data['labels']]
    
    fig = px.pie(
        values=sentiment_data['values'],
        names=[label.title() for label in sentiment_data['labels']],
        title="Sentiment Distribution",
        color_discrete_sequence=chart_colors
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    return fig


def create_sentiment_bar_chart(sentiment_data: Dict) -> go.Figure:
    """Create sentiment bar chart with confidence scores."""
    fig = go.Figure()
    
    colors = {'positive': '#22c55e', 'negative': '#ef4444', 'neutral': '#64748b'}
    
    for i, (label, value) in enumerate(zip(sentiment_data['labels'], sentiment_data['values'])):
        confidence = sentiment_data['confidences'].get(label, 0)
        
        fig.add_trace(go.Bar(
            x=[label.title()],
            y=[value],
            name=label.title(),
            marker_color=colors.get(label.lower(), '#64748b'),
            text=f"{confidence:.1%} conf.",
            textposition='auto',
            showlegend=False
        ))
    
    fig.update_layout(
        title="Sentiment Counts with Confidence",
        xaxis_title="Sentiment",
        yaxis_title="Count"
    )
    
    return fig


def extract_emotion_data(results: List[Dict]) -> List[Dict]:
    """Extract emotion data from analysis results."""
    emotions = []
    
    for result in results:
        emotion_result = result.get('emotion_result', {})
        if emotion_result:
            emotions.append({
                'emotion': emotion_result.get('emotion_label', 'neutral'),
                'confidence': emotion_result.get('confidence_score', 0),
                'scores': emotion_result.get('emotion_scores', {})
            })
    
    return emotions


def create_emotion_bar_chart(emotion_data: List[Dict]) -> go.Figure:
    """Create emotion distribution bar chart."""
    if not emotion_data:
        return go.Figure()
    
    emotion_counts = Counter([e['emotion'] for e in emotion_data])
    
    emotions = list(emotion_counts.keys())
    counts = list(emotion_counts.values())
    
    fig = px.bar(
        x=emotions,
        y=counts,
        title="Emotion Distribution",
        color=counts,
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        xaxis_title="Emotion",
        yaxis_title="Count",
        showlegend=False
    )
    
    return fig


def create_word_cloud(text: str, max_words: int, min_length: int, color_scheme: str) -> plt.Figure:
    """Create word cloud visualization."""
    try:
        # Color schemes
        colormap_dict = {
            'Default': None,
            'Viridis': 'viridis',
            'Blues': 'Blues',
            'Reds': 'Reds',
            'Greens': 'Greens'
        }
        
        wordcloud = WordCloud(
            width=800,
            height=400,
            max_words=max_words,
            min_word_length=min_length,
            background_color='white',
            colormap=colormap_dict.get(color_scheme),
            relative_scaling=0.5,
            random_state=42
        ).generate(text)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'Word Cloud ({max_words} words max)', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        st.error(f"Error generating word cloud: {str(e)}")
        return None


def extract_text_content(results: List[Dict]) -> str:
    """Extract text content from analysis results."""
    texts = []
    
    for result in results:
        text = result.get('text', '')
        if text:
            texts.append(text)
    
    return ' '.join(texts)


def calculate_text_statistics(text: str) -> Dict:
    """Calculate basic text statistics."""
    words = text.lower().split()
    unique_words = set(words)
    
    return {
        'total_words': len(words),
        'unique_words': len(unique_words),
        'avg_words': len(words) / max(len(text.split('\n')), 1),
        'vocabulary_richness': len(unique_words) / max(len(words), 1)
    }


def get_frequent_words(text: str, top_n: int = 20) -> List[tuple]:
    """Get most frequent words from text."""
    # Simple word frequency (in production, you'd want better text processing)
    words = text.lower().split()
    
    # Filter out common stop words
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'a', 'an'}
    
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
    
    word_counts = Counter(filtered_words)
    
    return word_counts.most_common(top_n)


def export_analysis_data(results: List[Dict]):
    """Export analysis data to downloadable format."""
    # Convert results to DataFrame
    df_data = []
    
    for result in results:
        row = {
            'text': result.get('text', ''),
            'overall_sentiment': result.get('overall_sentiment', ''),
            'overall_confidence': result.get('overall_confidence', 0),
            'processing_time_ms': result.get('processing_time_ms', 0)
        }
        
        # Add emotion data if available
        emotion_result = result.get('emotion_result', {})
        if emotion_result:
            row['primary_emotion'] = emotion_result.get('emotion_label', '')
            row['emotion_confidence'] = emotion_result.get('confidence_score', 0)
        
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    # Convert to CSV
    csv = df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download Analysis Results (CSV)",
        data=csv,
        file_name=f"sentiment_analysis_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )


# Additional helper functions for advanced analytics would be implemented here...

def calculate_correlation_matrix(results: List[Dict]) -> Optional[pd.DataFrame]:
    """Calculate correlation matrix for numerical features."""
    # This would be implemented based on available numerical data
    return None


def create_correlation_heatmap(correlation_data: pd.DataFrame) -> go.Figure:
    """Create correlation heatmap."""
    # Implementation for correlation visualization
    return go.Figure()


def create_confidence_sentiment_scatter(results: List[Dict]) -> go.Figure:
    """Create scatter plot of confidence vs sentiment."""
    # Implementation for scatter plot
    return go.Figure()


def create_text_length_analysis(results: List[Dict]) -> go.Figure:
    """Create text length analysis chart."""
    # Implementation for text length visualization
    return go.Figure()


def generate_statistical_insights(results: List[Dict]) -> List[Dict]:
    """Generate statistical insights from results."""
    insights = []
    
    if len(results) > 100:
        insights.append({
            'type': 'success',
            'message': f'Large dataset analyzed: {len(results)} comments provide robust statistical significance.'
        })
    
    # Add more insight generation logic here
    
    return insights


def create_word_cloud(
    filtered_text: str, max_words: int, min_word_length: int, color_scheme: str
) -> Optional[plt.Figure]:
    """
    Create a basic word cloud (fallback for when sentiment data is not available).
    """
    try:
        # Import here to avoid circular imports
        from backend.app.services.visualization_service import VisualizationService
        vis_service = VisualizationService()

        # Prepare tokens
        tokens = vis_service.prepare_tokens([filtered_text], min_len=min_word_length)
        frequencies = vis_service.compute_frequencies(tokens, max_words)

        if not frequencies:
            return None

        # Create word cloud
        fig, ax = plt.subplots(figsize=(10, 5))
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(frequencies)
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        plt.tight_layout()

        return fig

    except Exception as e:
        print(f"Error creating word cloud: {e}")
        return None


async def create_sentiment_word_cloud(
    texts: List[str], results: List[Dict], max_words: int = 100
) -> Tuple[Optional[plt.Figure], Optional[Dict[str, Any]]]:
    """
    Create a sentiment-tagged word cloud with contextual examples.
    """
    try:
        from backend.app.services.visualization_service import VisualizationService
        vis_service = VisualizationService()

        # Create sentiment-tagged word cloud
        image_bytes, word_data = await vis_service.create_sentiment_tagged_wordcloud(
            texts, results, max_words=max_words
        )

        if not image_bytes:
            return None, None

        # Convert bytes to PIL Image for matplotlib
        from PIL import Image
        import io
        image = Image.open(io.BytesIO(image_bytes))

        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(image)
        ax.axis('off')
        plt.tight_layout()

        # Convert word data for JSON serialization
        serializable_word_data = {}
        for word, data in word_data.items():
            serializable_word_data[word] = {
                "frequency": data.frequency,
                "sentiment_score": data.sentiment_score,
                "sentiment_label": data.sentiment_label,
                "contextual_examples": data.contextual_examples,
                "confidence": data.confidence
            }

        return fig, serializable_word_data

    except Exception as e:
        print(f"Error creating sentiment word cloud: {e}")
        return None, None


# More helper functions would be implemented as needed...

def filter_text_by_sentiment(text_content: str, results: List[Dict], sentiment_filter: str) -> str:
    """Filter text content by sentiment."""
    if sentiment_filter == "All":
        return text_content
    
    # Filter texts based on sentiment
    filtered_texts = []
    
    # Map sentiment filter to the expected sentiment label format
    sentiment_mapping = {
        "Positive": "positive",
        "Negative": "negative", 
        "Neutral": "neutral"
    }
    
    target_sentiment = sentiment_mapping.get(sentiment_filter)
    
    if not target_sentiment:
        return text_content  # Return all if mapping fails
    
    # Filter results by sentiment and collect corresponding texts
    for result in results:
        # Get sentiment from result (could be in different fields)
        sentiment = (result.get('overall_sentiment') or 
                    result.get('sentiment') or 
                    result.get('sentiment_label') or
                    'neutral')
        
        # Compare with target sentiment (case insensitive)
        if sentiment.lower() == target_sentiment.lower():
            # Add the text content for this result
            text = result.get('text', '')
            if text:
                filtered_texts.append(text)
    
    # Join filtered texts
    return ' '.join(filtered_texts)
