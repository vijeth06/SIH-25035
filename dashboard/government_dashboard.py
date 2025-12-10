"""
MCA eConsultation Sentiment Analysis Dashboard
Government-style UI with MCA21 portal aesthetic
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import base64
from io import BytesIO
import time
import numpy as np
from collections import Counter
import re

# Page configuration
st.set_page_config(
    page_title="MCA Sentiment Analysis Dashboard",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Government aesthetic
st.markdown("""
<style>
    /* Import Government fonts */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&family=Noto+Sans:wght@300;400;500;700&display=swap');
    
    /* Root variables */
    :root {
        --primary-navy: #003366;
        --secondary-white: #FFFFFF;
        --bg-light-gray: #F8F9FA;
        --sentiment-positive: #28a745;
        --sentiment-neutral: #6c757d;
        --sentiment-negative: #dc3545;
        --alert-yellow: #FFC107;
        --text-dark: #343a40;
        --hover-navy: #002244;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Main container styling */
    .main > div {
        padding-top: 0rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Government header */
    .gov-header {
        background-color: var(--primary-navy);
        color: var(--secondary-white);
        padding: 12px 20px;
        margin: -1rem -1rem 2rem -1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        font-family: 'Roboto', sans-serif;
    }
    
    .gov-header-left {
        display: flex;
        align-items: center;
        gap: 15px;
    }
    
    .gov-emblem {
        width: 50px;
        height: 50px;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="50" cy="50" r="45" fill="%23FFD700" stroke="%23FFFFFF" stroke-width="2"/><text x="50" y="55" text-anchor="middle" fill="%23003366" font-family="Arial" font-size="12" font-weight="bold">GOI</text></svg>') no-repeat center;
        background-size: contain;
    }
    
    .gov-title {
        font-size: 20px;
        font-weight: 700;
        margin: 0;
    }
    
    .gov-header-right {
        display: flex;
        align-items: center;
        gap: 15px;
        font-size: 14px;
    }
    
    .logout-btn {
        background-color: var(--sentiment-negative);
        color: var(--secondary-white);
        border: none;
        padding: 8px 16px;
        border-radius: 4px;
        cursor: pointer;
        font-size: 14px;
        transition: background-color 0.3s;
    }
    
    .logout-btn:hover {
        background-color: #c82333;
    }
    
    /* Overview cards */
    .overview-card {
        background: var(--secondary-white);
        border: 1px solid var(--primary-navy);
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s;
        text-align: center;
        height: 180px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .overview-card:hover {
        transform: scale(1.05);
        cursor: pointer;
    }
    
    .card-title {
        font-family: 'Roboto', sans-serif;
        font-size: 16px;
        font-weight: 500;
        color: var(--primary-navy);
        margin-bottom: 10px;
    }
    
    .card-value {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 8px;
    }
    
    .card-subtitle {
        font-size: 14px;
        color: var(--text-dark);
    }
    
    /* Section headers */
    .section-header {
        font-family: 'Roboto', sans-serif;
        font-size: 18px;
        font-weight: 700;
        color: var(--primary-navy);
        margin: 30px 0 15px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid var(--primary-navy);
    }
    
    /* Filters container */
    .filters-container {
        background: var(--bg-light-gray);
        padding: 20px;
        border-radius: 8px;
        margin: 20px 0;
        border: 1px solid #dee2e6;
    }
    
    /* Sentiment badges */
    .sentiment-positive { 
        background-color: var(--sentiment-positive); 
        color: white; 
        padding: 4px 8px; 
        border-radius: 12px; 
        font-size: 12px;
        font-weight: 500;
    }
    .sentiment-negative { 
        background-color: var(--sentiment-negative); 
        color: white; 
        padding: 4px 8px; 
        border-radius: 12px; 
        font-size: 12px;
        font-weight: 500;
    }
    .sentiment-neutral { 
        background-color: var(--sentiment-neutral); 
        color: white; 
        padding: 4px 8px; 
        border-radius: 12px; 
        font-size: 12px;
        font-weight: 500;
    }
    
    /* Language badges */
    .lang-english { background-color: #007bff; color: white; padding: 3px 6px; border-radius: 8px; font-size: 11px; }
    .lang-hindi { background-color: #28a745; color: white; padding: 3px 6px; border-radius: 8px; font-size: 11px; }
    .lang-tamil { background-color: #6f42c1; color: white; padding: 3px 6px; border-radius: 8px; font-size: 11px; }
    .lang-other { background-color: #6c757d; color: white; padding: 3px 6px; border-radius: 8px; font-size: 11px; }
    
    /* Progress bars */
    .confidence-bar {
        width: 100%;
        height: 8px;
        background-color: #e9ecef;
        border-radius: 4px;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        background-color: var(--sentiment-positive);
        transition: width 0.3s;
    }
    
    /* Export buttons */
    .export-btn {
        background-color: var(--primary-navy);
        color: var(--secondary-white);
        border: none;
        padding: 12px 24px;
        border-radius: 4px;
        cursor: pointer;
        font-size: 14px;
        font-weight: 500;
        margin: 5px;
        transition: background-color 0.3s;
        font-family: 'Roboto', sans-serif;
    }
    
    .export-btn:hover {
        background-color: var(--hover-navy);
    }
    
    /* Footer */
    .gov-footer {
        text-align: center;
        padding: 20px;
        color: var(--sentiment-neutral);
        font-size: 12px;
        border-top: 1px solid #dee2e6;
        margin-top: 50px;
        font-family: 'Roboto', sans-serif;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .gov-header {
            flex-direction: column;
            gap: 10px;
            text-align: center;
        }
        
        .gov-title {
            font-size: 16px;
        }
        
        .overview-card {
            height: auto;
            margin-bottom: 15px;
        }
        
        .filters-container {
            padding: 15px;
        }
    }
    
    /* Data table styling */
    .dataframe {
        font-family: 'Roboto', sans-serif;
        font-size: 14px;
    }
    
    .dataframe th {
        background-color: var(--primary-navy);
        color: var(--secondary-white);
        font-weight: 500;
        text-align: center;
    }
    
    .dataframe td {
        text-align: center;
        vertical-align: middle;
    }
    
    /* Alert styling */
    .alert-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 12px;
        border-radius: 4px;
        margin: 10px 0;
    }
    
    .alert-info {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 12px;
        border-radius: 4px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Initialize session state
if 'api_status' not in st.session_state:
    st.session_state.api_status = 'unknown'
if 'sample_data' not in st.session_state:
    st.session_state.sample_data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

def check_api_status():
    """Check if API is available"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if response.status_code == 200:
            st.session_state.api_status = 'available'
            return True
    except:
        pass
    st.session_state.api_status = 'unavailable'
    return False

def load_sample_data():
    """Load the MCA test dataset"""
    try:
        df = pd.read_csv('data/sample/mca_test_dataset.csv')
        st.session_state.sample_data = df
        return df
    except:
        # Mock data if file not found
        mock_data = {
            'stakeholder_type': ['Individual', 'NGO', 'Corporation', 'Academic'] * 15,
            'policy_area': ['Digital Governance', 'Environmental Compliance', 'Corporate Affairs', 'Financial Regulation'] * 15,
            'comment': [
                'I strongly support the new digital governance framework. It provides excellent transparency.',
                'This policy is absolutely terrible. It completely ignores environmental concerns.',
                'The proposed amendments are reasonable and strike a good balance.',
                'While the policy has merit, I have significant concerns about implementation.'
            ] * 15
        }
        df = pd.DataFrame(mock_data)
        st.session_state.sample_data = df
        return df

def analyze_comments_with_api(comments, use_advanced=False):
    """Analyze comments using the API"""
    try:
        payload = {
            "texts": comments,
            "include_explanation": True,
            "use_advanced": use_advanced
        }
        response = requests.post(f"{API_BASE_URL}/api/analyze", json=payload, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None

def create_mock_analysis(comments):
    """Create mock sentiment analysis when API is unavailable"""
    results = []
    sentiments = ['positive', 'negative', 'neutral']
    
    for comment in comments:
        # Simple keyword-based mock analysis
        comment_lower = comment.lower()
        if any(word in comment_lower for word in ['excellent', 'great', 'good', 'support', 'love', 'fantastic', 'amazing']):
            sentiment = 'positive'
            confidence = np.random.uniform(0.7, 0.95)
        elif any(word in comment_lower for word in ['terrible', 'bad', 'awful', 'disaster', 'hate', 'disappointed']):
            sentiment = 'negative'
            confidence = np.random.uniform(0.7, 0.95)
        else:
            sentiment = 'neutral'
            confidence = np.random.uniform(0.5, 0.8)
        
        results.append({
            'text': comment,
            'sentiment': sentiment,
            'confidence': confidence,
            'polarity_score': np.random.uniform(-1, 1)
        })
    
    # Calculate summary
    sentiment_counts = Counter([r['sentiment'] for r in results])
    total = len(results)
    
    summary = {
        'total_analyzed': total,
        'sentiment_distribution': {
            'positive': {'count': sentiment_counts['positive'], 'percentage': round(sentiment_counts['positive']/total*100, 1)},
            'negative': {'count': sentiment_counts['negative'], 'percentage': round(sentiment_counts['negative']/total*100, 1)},
            'neutral': {'count': sentiment_counts['neutral'], 'percentage': round(sentiment_counts['neutral']/total*100, 1)}
        },
        'average_confidence': round(np.mean([r['confidence'] for r in results]), 3)
    }
    
    return {'results': results, 'summary': summary}

def render_government_header():
    """Render the government-style header"""
    st.markdown("""
    <div class="gov-header">
        <div class="gov-header-left">
            <div class="gov-emblem" aria-label="Government of India Emblem"></div>
            <h1 class="gov-title">MCA Sentiment Analysis Dashboard</h1>
        </div>
        <div class="gov-header-right">
            <span>Welcome, Policy Analyst | Role: Senior Analyst</span>
            <button class="logout-btn" aria-label="Logout button">Logout</button>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_overview_cards(analysis_results):
    """Render overview cards"""
    if analysis_results:
        summary = analysis_results['summary']
        results = analysis_results['results']
        
        total_comments = summary['total_analyzed']
        sentiment_dist = summary['sentiment_distribution']
        avg_confidence = summary['average_confidence']
        
        # Detect languages (mock detection)
        languages = {'English': 70, 'Hindi': 20, 'Tamil': 8, 'Other': 2}
    else:
        total_comments = 60
        sentiment_dist = {'positive': {'percentage': 45}, 'neutral': {'percentage': 30}, 'negative': {'percentage': 25}}
        avg_confidence = 0.75
        languages = {'English': 70, 'Hindi': 20, 'Tamil': 8, 'Other': 2}
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="overview-card">
            <div class="card-title">Total Comments</div>
            <div class="card-value" style="color: var(--primary-navy);">{total_comments:,}</div>
            <div class="card-subtitle">Comments Analyzed</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="overview-card">
            <div class="card-title">Sentiment Breakdown</div>
            <div style="font-size: 14px; line-height: 1.6;">
                <span class="sentiment-positive">Positive: {sentiment_dist['positive']['percentage']}%</span><br>
                <span class="sentiment-neutral">Neutral: {sentiment_dist['neutral']['percentage']}%</span><br>
                <span class="sentiment-negative">Negative: {sentiment_dist['negative']['percentage']}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="overview-card">
            <div class="card-title">Language Distribution</div>
            <div style="font-size: 14px; line-height: 1.6;">
                <span class="lang-english">English: {languages['English']}%</span><br>
                <span class="lang-hindi">Hindi: {languages['Hindi']}%</span><br>
                <span class="lang-tamil">Tamil: {languages['Tamil']}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        bias_alert = ""
        if languages['Tamil'] < 10:
            bias_alert = f'<div style="color: var(--sentiment-negative); font-weight: bold;">⚠️ Tamil <10%</div>'
        
        st.markdown(f"""
        <div class="overview-card">
            <div class="card-title">Quality Check</div>
            <div class="card-value" style="color: var(--sentiment-positive);">{avg_confidence:.1%}</div>
            <div class="card-subtitle">Avg Confidence</div>
            {bias_alert}
        </div>
        """, unsafe_allow_html=True)

def render_visualizations(analysis_results, df):
    """Render sentiment timeline and word cloud"""
    st.markdown('<div class="section-header">📊 Sentiment Analysis Overview</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Sentiment Over Time**")
        
        # Mock timeline data
        dates = pd.date_range(start='2025-08-20', end='2025-09-19', freq='D')
        timeline_data = []
        
        for date in dates:
            timeline_data.append({
                'Date': date,
                'Positive': np.random.uniform(30, 60),
                'Neutral': np.random.uniform(20, 40),
                'Negative': np.random.uniform(15, 35)
            })
        
        timeline_df = pd.DataFrame(timeline_data)
        
        fig_timeline = go.Figure()
        fig_timeline.add_trace(go.Scatter(
            x=timeline_df['Date'], y=timeline_df['Positive'],
            mode='lines+markers', name='Positive',
            line=dict(color='#28a745', width=3),
            marker=dict(size=6)
        ))
        fig_timeline.add_trace(go.Scatter(
            x=timeline_df['Date'], y=timeline_df['Neutral'],
            mode='lines+markers', name='Neutral',
            line=dict(color='#6c757d', width=3),
            marker=dict(size=6)
        ))
        fig_timeline.add_trace(go.Scatter(
            x=timeline_df['Date'], y=timeline_df['Negative'],
            mode='lines+markers', name='Negative',
            line=dict(color='#dc3545', width=3),
            marker=dict(size=6)
        ))
        
        fig_timeline.update_layout(
            title="",
            xaxis_title="Date",
            yaxis_title="Sentiment Percentage (%)",
            height=400,
            showlegend=True,
            plot_bgcolor='white',
            font=dict(family="Roboto", size=12),
            margin=dict(l=0, r=0, t=20, b=0)
        )
        
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    with col2:
        st.markdown("**Key Themes Word Cloud**")
        
        # Create word frequency visualization
        if analysis_results and 'results' in analysis_results:
            # Extract words from comments
            all_text = ' '.join([r['text'] for r in analysis_results['results']])
            words = re.findall(r'\b\w+\b', all_text.lower())
            word_freq = Counter([w for w in words if len(w) > 3])
            top_words = dict(word_freq.most_common(20))
        else:
            # Mock word frequency
            top_words = {
                'policy': 25, 'governance': 20, 'digital': 18, 'compliance': 15,
                'reform': 12, 'framework': 10, 'transparency': 9, 'regulation': 8,
                'implementation': 7, 'standards': 6, 'accountability': 5, 'innovation': 4
            }
        
        # Create a simple bar chart instead of word cloud
        words_df = pd.DataFrame(list(top_words.items()), columns=['Word', 'Frequency'])
        words_df = words_df.head(10)
        
        fig_words = px.bar(
            words_df, x='Frequency', y='Word',
            orientation='h',
            color='Frequency',
            color_continuous_scale=['#003366', '#0066cc', '#3399ff'],
            title=""
        )
        
        fig_words.update_layout(
            height=400,
            showlegend=False,
            plot_bgcolor='white',
            font=dict(family="Roboto", size=12),
            margin=dict(l=0, r=0, t=20, b=0),
            coloraxis_showscale=False
        )
        
        st.plotly_chart(fig_words, use_container_width=True)

def render_comment_explorer(analysis_results, df):
    """Render the comment explorer with filters and table"""
    st.markdown('<div class="section-header">🔍 Comment Explorer</div>', unsafe_allow_html=True)
    
    # Filters
    st.markdown('<div class="filters-container">', unsafe_allow_html=True)
    st.markdown("**Filters**")
    
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
    
    with filter_col1:
        language_filter = st.selectbox("Language", ["All", "English", "Hindi", "Tamil", "Other"])
    
    with filter_col2:
        sentiment_filter = st.selectbox("Sentiment", ["All", "Positive", "Negative", "Neutral"])
    
    with filter_col3:
        stakeholder_filter = st.selectbox("Stakeholder Type", 
            ["All"] + list(df['stakeholder_type'].unique()) if df is not None else ["All", "Individual", "NGO", "Corporation"])
    
    with filter_col4:
        search_query = st.text_input("Search Keywords", placeholder="Enter keywords...")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Prepare data for table
    if analysis_results and df is not None:
        # Combine original data with analysis results
        display_data = []
        
        for i, (_, row) in enumerate(df.iterrows()):
            if i < len(analysis_results['results']):
                result = analysis_results['results'][i]
                
                # Apply filters
                if sentiment_filter != "All" and result['sentiment'].title() != sentiment_filter:
                    continue
                if stakeholder_filter != "All" and row['stakeholder_type'] != stakeholder_filter:
                    continue
                if search_query and search_query.lower() not in row['comment'].lower():
                    continue
                
                # Mock language detection
                comment_text = row['comment']
                if any(char in comment_text for char in 'हिअआई'):
                    language = 'Hindi'
                elif any(char in comment_text for char in 'தமிழ்'):
                    language = 'Tamil'
                else:
                    language = 'English'
                
                if language_filter != "All" and language != language_filter:
                    continue
                
                display_data.append({
                    'ID': i + 1,
                    'Comment': comment_text[:100] + "..." if len(comment_text) > 100 else comment_text,
                    'Stakeholder': row['stakeholder_type'],
                    'Policy Area': row['policy_area'],
                    'Language': language,
                    'Sentiment': result['sentiment'].title(),
                    'Confidence': f"{result['confidence']:.1%}",
                    'Polarity': f"{result['polarity_score']:.2f}"
                })
        
        if display_data:
            display_df = pd.DataFrame(display_data)
            
            # Style the dataframe
            def style_sentiment(val):
                if val == 'Positive':
                    return 'background-color: #d4edda; color: #155724;'
                elif val == 'Negative':
                    return 'background-color: #f8d7da; color: #721c24;'
                else:
                    return 'background-color: #e2e3e5; color: #383d41;'
            
            def style_confidence(val):
                confidence = float(val.strip('%')) / 100
                if confidence < 0.6:
                    return 'background-color: #fff3cd; color: #856404;'
                else:
                    return ''
            
            styled_df = display_df.style.applymap(style_sentiment, subset=['Sentiment'])
            styled_df = styled_df.applymap(style_confidence, subset=['Confidence'])
            
            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True,
                height=400
            )
            
            # Show summary
            st.info(f"📊 Showing {len(display_data)} comments out of {len(df)} total")
        else:
            st.warning("No comments match the selected filters.")
    else:
        st.error("⚠️ API services not available, using mock data. Please ensure the backend API is running.")

def render_summary_and_export(analysis_results):
    """Render summary and export options"""
    st.markdown('<div class="section-header">📋 Summary & Export</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Comment Summary**")
        
        if analysis_results:
            summary_text = f"""
            **Analysis Summary:**
            
            • Total comments analyzed: {analysis_results['summary']['total_analyzed']}
            • Average confidence: {analysis_results['summary']['average_confidence']:.1%}
            • Positive sentiment: {analysis_results['summary']['sentiment_distribution']['positive']['percentage']}%
            • Negative sentiment: {analysis_results['summary']['sentiment_distribution']['negative']['percentage']}%
            • Neutral sentiment: {analysis_results['summary']['sentiment_distribution']['neutral']['percentage']}%
            
            **Key Insights:**
            • Most comments express concerns about implementation timelines
            • Strong support for digital governance initiatives
            • Environmental policies show mixed reactions
            • Corporate stakeholders generally supportive of regulatory reforms
            """
        else:
            summary_text = """
            **Mock Analysis Summary:**
            
            • Total comments analyzed: 60
            • Average confidence: 75%
            • Positive sentiment: 45%
            • Negative sentiment: 25%
            • Neutral sentiment: 30%
            
            **Key Insights:**
            • Strong support for governance reforms
            • Concerns about implementation timelines
            • Mixed reactions to environmental policies
            """
        
        st.markdown(summary_text)
    
    with col2:
        st.markdown("**Export Options**")
        
        col2a, col2b = st.columns(2)
        
        with col2a:
            if st.button("📄 Download PDF Report", key="pdf_export"):
                st.success("PDF report generation initiated! (Mock)")
                st.info("Report will include charts, word cloud, and analysis summary with MCA watermark.")
        
        with col2b:
            if st.button("📊 Download Excel Data", key="excel_export"):
                st.success("Excel export initiated! (Mock)")
                st.info("Excel file will contain comment data with sentiment analysis results.")
        
        st.markdown("---")
        st.markdown("**API Status**")
        if st.session_state.api_status == 'available':
            st.success("🟢 All services operational")
        else:
            st.error("🔴 API services unavailable - using mock data")
            if st.button("🔄 Retry Connection"):
                if check_api_status():
                    st.success("✅ API connection restored!")
                    st.experimental_rerun()
                else:
                    st.error("❌ Still unable to connect to API services")

def render_footer():
    """Render government footer"""
    st.markdown("""
    <div class="gov-footer">
        <p>Developed for Ministry of Corporate Affairs | Powered by Advanced AI Analytics</p>
        <p><a href="#" style="color: var(--primary-navy);">Contact</a> | 
           <a href="#" style="color: var(--primary-navy);">Terms of Service</a> | 
           <a href="#" style="color: var(--primary-navy);">Privacy Policy</a></p>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main dashboard function"""
    # Render header
    render_government_header()
    
    # Check API status
    api_available = check_api_status()
    
    # Load data
    df = load_sample_data()
    
    # Analyze comments
    if df is not None and not df.empty:
        comments = df['comment'].tolist()
        
        if api_available:
            with st.spinner("🔄 Analyzing comments with advanced AI models..."):
                analysis_results = analyze_comments_with_api(comments, use_advanced=True)
                if analysis_results is None:
                    st.warning("API analysis failed, using mock data")
                    analysis_results = create_mock_analysis(comments)
        else:
            st.warning("⚠️ API services not available, using mock data")
            analysis_results = create_mock_analysis(comments)
        
        st.session_state.analysis_results = analysis_results
    else:
        st.error("❌ Unable to load sample data")
        analysis_results = None
    
    # Render dashboard sections
    render_overview_cards(analysis_results)
    render_visualizations(analysis_results, df)
    render_comment_explorer(analysis_results, df)
    render_summary_and_export(analysis_results)
    render_footer()

if __name__ == "__main__":
    main()