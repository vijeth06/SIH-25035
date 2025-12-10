"""
MCA Sentiment Analysis Dashboard - Government Style UI
Professional government aesthetic matching MCA21 portal style with navy blue theme.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any
import time
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="MCA eConsultation Sentiment Analysis",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Government Color Scheme
COLORS = {
    'primary': '#003366',      # Navy blue
    'secondary': '#FFFFFF',    # White
    'background': '#F8F9FA',   # Light gray
    'positive': '#28a745',     # Green
    'neutral': '#6c757d',      # Gray
    'negative': '#dc3545',     # Red
    'warning': '#FFC107',      # Yellow
    'text': '#343a40'          # Dark gray
}

# Custom CSS for Government Aesthetic
def load_custom_css():
    st.markdown(f"""
    <style>
    /* Import Government Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&family=Noto+Sans:wght@400;500;700&display=swap');
    
    /* Global Styles */
    .main {{
        background-color: {COLORS['background']};
        font-family: 'Roboto', sans-serif;
    }}
    
    /* Header Styles */
    .government-header {{
        background-color: {COLORS['primary']};
        color: {COLORS['secondary']};
        padding: 12px 20px;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    
    .header-content {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        max-width: 1200px;
        margin: 0 auto;
    }}
    
    .header-left {{
        display: flex;
        align-items: center;
        gap: 15px;
    }}
    
    .header-title {{
        font-size: 20px;
        font-weight: 700;
        color: {COLORS['secondary']};
        margin: 0;
    }}
    
    .header-right {{
        display: flex;
        align-items: center;
        gap: 20px;
        font-size: 14px;
    }}
    
    /* Card Styles */
    .metric-card {{
        background: {COLORS['secondary']};
        border: 1px solid {COLORS['primary']};
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
        height: 100%;
    }}
    
    .metric-card:hover {{
        transform: scale(1.05);
        cursor: pointer;
    }}
    
    .metric-value {{
        font-size: 24px;
        font-weight: 700;
        color: {COLORS['primary']};
        margin: 10px 0;
    }}
    
    .metric-label {{
        font-size: 16px;
        color: {COLORS['text']};
        font-weight: 500;
    }}
    
    /* Section Headers */
    .section-header {{
        font-size: 18px;
        font-weight: 700;
        color: {COLORS['primary']};
        margin: 30px 0 15px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid {COLORS['primary']};
    }}
    
    /* Filter Styles */
    .filter-container {{
        background: {COLORS['secondary']};
        border: 1px solid {COLORS['primary']};
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
    }}
    
    /* Table Styles */
    .dataframe {{
        border: 1px solid {COLORS['primary']} !important;
    }}
    
    .dataframe th {{
        background-color: {COLORS['primary']} !important;
        color: {COLORS['secondary']} !important;
        font-weight: 600 !important;
    }}
    
    /* Sentiment Badges */
    .sentiment-positive {{
        background-color: {COLORS['positive']};
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 500;
        font-size: 12px;
    }}
    
    .sentiment-negative {{
        background-color: {COLORS['negative']};
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 500;
        font-size: 12px;
    }}
    
    .sentiment-neutral {{
        background-color: {COLORS['neutral']};
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 500;
        font-size: 12px;
    }}
    
    /* Language Badges */
    .language-english {{
        background-color: #007bff;
        color: white;
        padding: 4px 8px;
        border-radius: 15px;
        font-size: 11px;
    }}
    
    .language-hindi {{
        background-color: #28a745;
        color: white;
        padding: 4px 8px;
        border-radius: 15px;
        font-size: 11px;
    }}
    
    .language-other {{
        background-color: {COLORS['neutral']};
        color: white;
        padding: 4px 8px;
        border-radius: 15px;
        font-size: 11px;
    }}
    
    /* Confidence Progress Bar */
    .confidence-bar {{
        background-color: #e9ecef;
        border-radius: 10px;
        height: 20px;
        overflow: hidden;
    }}
    
    .confidence-fill {{
        height: 100%;
        transition: width 0.3s ease;
    }}
    
    .confidence-high {{
        background-color: {COLORS['positive']};
    }}
    
    .confidence-medium {{
        background-color: {COLORS['warning']};
    }}
    
    .confidence-low {{
        background-color: {COLORS['negative']};
    }}
    
    /* Export Buttons */
    .export-button {{
        background-color: {COLORS['primary']};
        color: {COLORS['secondary']};
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: 500;
        cursor: pointer;
        margin: 5px;
        transition: background-color 0.3s ease;
    }}
    
    .export-button:hover {{
        background-color: #004080;
    }}
    
    /* Footer */
    .footer {{
        text-align: center;
        color: {COLORS['neutral']};
        font-size: 12px;
        margin-top: 40px;
        padding: 20px;
        border-top: 1px solid #dee2e6;
    }}
    
    /* Alert Styles */
    .alert-warning {{
        background-color: {COLORS['warning']};
        color: #856404;
        padding: 10px 15px;
        border-radius: 5px;
        margin: 10px 0;
        border-left: 4px solid #f0ad4e;
    }}
    
    .alert-error {{
        background-color: {COLORS['negative']};
        color: white;
        padding: 10px 15px;
        border-radius: 5px;
        margin: 10px 0;
    }}
    
    /* Responsive Design */
    @media (max-width: 768px) {{
        .header-content {{
            flex-direction: column;
            gap: 10px;
        }}
        
        .header-title {{
            font-size: 16px;
        }}
        
        .metric-card {{
            padding: 15px;
        }}
        
        .metric-value {{
            font-size: 20px;
        }}
    }}
    
    /* Accessibility */
    .sr-only {{
        position: absolute !important;
        width: 1px !important;
        height: 1px !important;
        padding: 0 !important;
        margin: -1px !important;
        overflow: hidden !important;
        clip: rect(0, 0, 0, 0) !important;
        white-space: nowrap !important;
        border: 0 !important;
    }}
    
    /* Focus States */
    .metric-card:focus,
    .export-button:focus {{
        outline: 2px solid {COLORS['warning']};
        outline-offset: 2px;
    }}
    
    /* Hide Streamlit Elements */
    #MainMenu {{visibility: hidden;}}
    .stDeployButton {{display:none;}}
    footer {{visibility: hidden;}}
    .stApp > header {{visibility: hidden;}}
    </style>
    """, unsafe_allow_html=True)

def render_government_header():
    """Render the government-style header"""
    st.markdown(f"""
    <div class="government-header">
        <div class="header-content">
            <div class="header-left">
                <svg width="50" height="50" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
                    <circle cx="50" cy="50" r="45" fill="{COLORS['secondary']}" stroke="{COLORS['primary']}" stroke-width="2"/>
                    <text x="50" y="35" text-anchor="middle" fill="{COLORS['primary']}" font-family="serif" font-size="14" font-weight="bold">भारत</text>
                    <text x="50" y="50" text-anchor="middle" fill="{COLORS['primary']}" font-family="serif" font-size="10">सरकार</text>
                    <text x="50" y="65" text-anchor="middle" fill="{COLORS['primary']}" font-family="Arial" font-size="8">GOVERNMENT</text>
                    <text x="50" y="75" text-anchor="middle" fill="{COLORS['primary']}" font-family="Arial" font-size="8">OF INDIA</text>
                </svg>
                <h1 class="header-title">MCA Sentiment Analysis Dashboard</h1>
            </div>
            <div class="header-right">
                <span>Welcome, Analyst</span>
                <span>|</span>
                <span>Role: MCA Officer</span>
                <button class="export-button" style="background-color: {COLORS['negative']}; padding: 6px 12px; font-size: 12px;">Logout</button>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# API Helper Functions
def check_api_connection():
    """Check if API is available"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except:
        return False, None

def analyze_sentiment_api(texts, use_advanced=False, include_explanation=True):
    """Call sentiment analysis API"""
    try:
        payload = {
            "texts": texts,
            "include_explanation": include_explanation,
            "use_advanced": use_advanced
        }
        response = requests.post(f"{API_BASE_URL}/api/analyze", json=payload, timeout=30)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"API Error: {response.status_code}"
    except Exception as e:
        return False, f"Connection Error: {str(e)}"

def generate_wordcloud_api(texts):
    """Generate word cloud via API"""
    try:
        payload = {
            "texts": texts,
            "width": 800,
            "height": 400,
            "max_words": 50
        }
        response = requests.post(f"{API_BASE_URL}/api/wordcloud", json=payload, timeout=30)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"API Error: {response.status_code}"
    except Exception as e:
        return False, f"Connection Error: {str(e)}"

def summarize_texts_api(texts):
    """Generate summary via API"""
    try:
        payload = {
            "texts": texts,
            "max_length": 150,
            "min_length": 50
        }
        response = requests.post(f"{API_BASE_URL}/api/summarize", json=payload, timeout=60)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"API Error: {response.status_code}"
    except Exception as e:
        return False, f"Connection Error: {str(e)}"

# Mock Data Functions
def generate_mock_data():
    """Generate mock data for testing"""
    sample_comments = [
        "I strongly support the new digital governance framework. Excellent transparency measures!",
        "यह नई नीति बहुत अच्छी है। पारदर्शिता बेहतर हो गई है।",  # Hindi: This new policy is very good. Transparency has improved.
        "This policy is terrible and will hurt small businesses significantly.",
        "The implementation timeline is reasonable and well-planned.",
        "தமிழ் மொழியில் சேவைகள் வழங்கப்படுவது மிகவும் நல்லது.",  # Tamil: Providing services in Tamil is very good.
        "I disagree with the proposed changes. They lack proper consultation.",
        "The environmental impact assessment is comprehensive and thorough.",
        "বাংলায় তথ্য পাওয়া যাচ্ছে, এটা খুব ভালো উদ্যোগ।",  # Bengali: Information is available in Bengali, this is a very good initiative.
        "Budget allocation seems insufficient for effective implementation.",
        "Great initiative for digital transformation. Highly recommended!",
    ]
    
    languages = ['English', 'Hindi', 'Tamil', 'Bengali', 'English', 'English', 'English', 'Bengali', 'English', 'English']
    sentiments = ['Positive', 'Positive', 'Negative', 'Positive', 'Positive', 'Negative', 'Positive', 'Positive', 'Negative', 'Positive']
    confidences = [0.95, 0.89, 0.87, 0.82, 0.91, 0.85, 0.88, 0.92, 0.79, 0.94]
    
    # Generate dates for the last 30 days
    dates = [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)]
    
    return {
        'comments': sample_comments,
        'languages': languages,
        'sentiments': sentiments,
        'confidences': confidences,
        'dates': dates
    }

def render_overview_cards(data):
    """Render overview metric cards"""
    col1, col2, col3, col4 = st.columns(4)
    
    total_comments = len(data['comments'])
    sentiment_counts = pd.Series(data['sentiments']).value_counts()
    language_counts = pd.Series(data['languages']).value_counts()
    
    # Calculate percentages
    pos_pct = round((sentiment_counts.get('Positive', 0) / total_comments) * 100, 1)
    neu_pct = round((sentiment_counts.get('Neutral', 0) / total_comments) * 100, 1)
    neg_pct = round((sentiment_counts.get('Negative', 0) / total_comments) * 100, 1)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card" role="button" tabindex="0" aria-label="Total comments: {total_comments}">
            <div class="metric-label">📄 Total Comments</div>
            <div class="metric-value">{total_comments:,}</div>
            <small>Last 30 days</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" role="button" tabindex="0" aria-label="Sentiment breakdown: {pos_pct}% positive, {neu_pct}% neutral, {neg_pct}% negative">
            <div class="metric-label">😊 Sentiment Breakdown</div>
            <div style="font-size: 14px; margin: 10px 0;">
                <div style="color: {COLORS['positive']};">■ Positive: {pos_pct}%</div>
                <div style="color: {COLORS['neutral']};">■ Neutral: {neu_pct}%</div>
                <div style="color: {COLORS['negative']};">■ Negative: {neg_pct}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        lang_breakdown = "<br>".join([f"■ {lang}: {round((count/total_comments)*100, 1)}%" 
                                    for lang, count in language_counts.head(3).items()])
        st.markdown(f"""
        <div class="metric-card" role="button" tabindex="0" aria-label="Language distribution">
            <div class="metric-label">🌐 Language Distribution</div>
            <div style="font-size: 14px; margin: 10px 0;">
                {lang_breakdown}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # Check for language bias (< 10% threshold)
        min_lang_pct = min(language_counts.values()) / total_comments * 100
        bias_warning = min_lang_pct < 10
        
        st.markdown(f"""
        <div class="metric-card" role="button" tabindex="0" aria-label="Bias alert status">
            <div class="metric-label">⚠️ Bias Alert</div>
            <div class="metric-value" style="color: {'#dc3545' if bias_warning else '#28a745'};">
                {'HIGH' if bias_warning else 'LOW'}
            </div>
            <small>{"Language representation < 10%" if bias_warning else "Balanced representation"}</small>
        </div>
        """, unsafe_allow_html=True)

def render_visualizations(data):
    """Render timeline and word cloud visualizations"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-header">Sentiment Over Time</div>', unsafe_allow_html=True)
        
        # Generate timeline data
        timeline_data = []
        for i, date in enumerate(data['dates']):
            # Simulate sentiment distribution over time
            pos_pct = 45 + np.random.normal(0, 10)
            neg_pct = 25 + np.random.normal(0, 8)
            neu_pct = 100 - pos_pct - neg_pct
            
            timeline_data.append({
                'Date': date,
                'Positive': max(0, min(100, pos_pct)),
                'Negative': max(0, min(100, neg_pct)),
                'Neutral': max(0, min(100, neu_pct))
            })
        
        timeline_df = pd.DataFrame(timeline_data)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=timeline_df['Date'], y=timeline_df['Positive'], 
                               mode='lines+markers', name='Positive', 
                               line=dict(color=COLORS['positive'], width=3)))
        fig.add_trace(go.Scatter(x=timeline_df['Date'], y=timeline_df['Neutral'], 
                               mode='lines+markers', name='Neutral', 
                               line=dict(color=COLORS['neutral'], width=3)))
        fig.add_trace(go.Scatter(x=timeline_df['Date'], y=timeline_df['Negative'], 
                               mode='lines+markers', name='Negative', 
                               line=dict(color=COLORS['negative'], width=3)))
        
        fig.update_layout(
            height=400,
            xaxis_title="Date",
            yaxis_title="Sentiment Percentage (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            font=dict(family="Roboto", size=12),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-header">Key Themes</div>', unsafe_allow_html=True)
        
        # Check API connection for word cloud
        api_connected, api_info = check_api_connection()
        
        if api_connected:
            with st.spinner("Generating word cloud..."):
                success, wordcloud_result = generate_wordcloud_api(data['comments'])
                
                if success and wordcloud_result.get('status') == 'success':
                    wordcloud_data = wordcloud_result.get('wordcloud_data', {})
                    if 'image_data' in wordcloud_data:
                        st.image(wordcloud_data['image_data'], use_column_width=True)
                    else:
                        st.info("Word cloud generated successfully but no image data received.")
                else:
                    st.warning("Could not generate word cloud via API. Using mock visualization.")
                    # Create mock word cloud with Plotly
                    render_mock_wordcloud()
        else:
            st.warning("⚠️ API services not available, using mock visualization")
            render_mock_wordcloud()

def render_mock_wordcloud():
    """Render a mock word cloud using Plotly"""
    words = ['governance', 'transparency', 'policy', 'implementation', 'digital', 
             'framework', 'consultation', 'stakeholder', 'compliance', 'regulatory']
    sizes = [50, 45, 40, 35, 30, 28, 25, 22, 20, 18]
    colors = [COLORS['primary'], COLORS['positive'], COLORS['negative'], COLORS['neutral']] * 3
    
    fig = go.Figure()
    for i, (word, size) in enumerate(zip(words, sizes)):
        fig.add_annotation(
            x=np.random.uniform(0.1, 0.9),
            y=np.random.uniform(0.1, 0.9),
            text=word,
            showarrow=False,
            font=dict(size=size, color=colors[i % len(colors)]),
            xanchor="center",
            yanchor="middle"
        )
    
    fig.update_layout(
        height=400,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        paper_bgcolor='white',
        title="Key Themes (Mock Data)"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_comment_explorer(data):
    """Render the comment explorer with filters and table"""
    st.markdown('<div class="section-header">Comment Explorer</div>', unsafe_allow_html=True)
    
    # Filters
    st.markdown('<div class="filter-container">', unsafe_allow_html=True)
    
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
    
    with filter_col1:
        language_filter = st.selectbox(
            "🌐 Language", 
            ["All"] + list(set(data['languages'])),
            key="lang_filter"
        )
    
    with filter_col2:
        sentiment_filter = st.selectbox(
            "😊 Sentiment", 
            ["All", "Positive", "Negative", "Neutral"],
            key="sentiment_filter"
        )
    
    with filter_col3:
        confidence_filter = st.slider(
            "📊 Min Confidence", 
            0.0, 1.0, 0.0, 0.1,
            key="confidence_filter"
        )
    
    with filter_col4:
        search_term = st.text_input(
            "🔍 Search Keywords", 
            placeholder="Enter keywords...",
            key="search_filter"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Apply filters
    filtered_data = apply_filters(data, language_filter, sentiment_filter, confidence_filter, search_term)
    
    # Display filtered results
    if filtered_data:
        render_comments_table(filtered_data)
    else:
        st.info("No comments match the selected filters.")

def apply_filters(data, language_filter, sentiment_filter, confidence_filter, search_term):
    """Apply filters to the data"""
    filtered_indices = []
    
    for i in range(len(data['comments'])):
        # Language filter
        if language_filter != "All" and data['languages'][i] != language_filter:
            continue
        
        # Sentiment filter
        if sentiment_filter != "All" and data['sentiments'][i] != sentiment_filter:
            continue
        
        # Confidence filter
        if data['confidences'][i] < confidence_filter:
            continue
        
        # Search filter
        if search_term and search_term.lower() not in data['comments'][i].lower():
            continue
        
        filtered_indices.append(i)
    
    if not filtered_indices:
        return None
    
    return {
        'comments': [data['comments'][i] for i in filtered_indices],
        'languages': [data['languages'][i] for i in filtered_indices],
        'sentiments': [data['sentiments'][i] for i in filtered_indices],
        'confidences': [data['confidences'][i] for i in filtered_indices]
    }

def render_comments_table(data):
    """Render the comments table with styling"""
    # Create DataFrame
    df = pd.DataFrame({
        'ID': range(1, len(data['comments']) + 1),
        'Comment': [comment[:100] + "..." if len(comment) > 100 else comment 
                   for comment in data['comments']],
        'Language': data['languages'],
        'Sentiment': data['sentiments'],
        'Confidence': [f"{conf:.1%}" for conf in data['confidences']],
        'Action': ['View Details'] * len(data['comments'])
    })
    
    # Style the DataFrame
    def style_sentiment(val):
        if val == 'Positive':
            return f'background-color: {COLORS["positive"]}; color: white; text-align: center; border-radius: 15px; padding: 4px;'
        elif val == 'Negative':
            return f'background-color: {COLORS["negative"]}; color: white; text-align: center; border-radius: 15px; padding: 4px;'
        else:
            return f'background-color: {COLORS["neutral"]}; color: white; text-align: center; border-radius: 15px; padding: 4px;'
    
    def style_language(val):
        if val == 'English':
            return 'background-color: #007bff; color: white; text-align: center; border-radius: 10px; padding: 2px 6px;'
        elif val == 'Hindi':
            return 'background-color: #28a745; color: white; text-align: center; border-radius: 10px; padding: 2px 6px;'
        else:
            return f'background-color: {COLORS["neutral"]}; color: white; text-align: center; border-radius: 10px; padding: 2px 6px;'
    
    # Display the table
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Comment": st.column_config.TextColumn("Comment Text", width="large"),
            "Confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1),
            "Action": st.column_config.LinkColumn("Action", display_text="View Details")
        }
    )
    
    # Add interaction note
    st.caption("💡 Click on rows to view detailed analysis with word highlighting")

def render_summary_and_export(data):
    """Render summary and export section"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-header">Comment Summary</div>', unsafe_allow_html=True)
        
        # Create tabs for different summaries
        tab1, tab2, tab3 = st.tabs(["📊 Overall", "🎯 By Topic", "🌐 By Language"])
        
        with tab1:
            # Check API connection for summary
            api_connected, api_info = check_api_connection()
            
            if api_connected:
                with st.spinner("Generating AI summary..."):
                    success, summary_result = summarize_texts_api(data['comments'][:5])  # Summarize first 5 comments
                    
                    if success and summary_result.get('status') == 'success':
                        summaries = summary_result.get('summaries', [])
                        if summaries:
                            combined_summary = " ".join([s.get('summary', '') for s in summaries[:3]])
                            st.write(combined_summary)
                        else:
                            st.write("No summary generated.")
                    else:
                        st.warning("Could not generate AI summary. Using statistical summary.")
                        render_statistical_summary(data)
            else:
                st.warning("⚠️ API services not available, using statistical summary")
                render_statistical_summary(data)
        
        with tab2:
            st.write("📈 **Policy Framework:** Mixed feedback on implementation timeline and scope.")
            st.write("🔧 **Technical Aspects:** Generally positive response to digital transformation initiatives.")
            st.write("💼 **Business Impact:** Concerns raised about compliance costs and SME effects.")
            
        with tab3:
            lang_summary = {}
            for lang in set(data['languages']):
                lang_data = [data['sentiments'][i] for i, l in enumerate(data['languages']) if l == lang]
                pos_count = lang_data.count('Positive')
                total_count = len(lang_data)
                lang_summary[lang] = f"{pos_count}/{total_count} positive ({round(pos_count/total_count*100, 1)}%)"
            
            for lang, summary in lang_summary.items():
                st.write(f"**{lang}:** {summary}")
    
    with col2:
        st.markdown('<div class="section-header">Export & Reports</div>', unsafe_allow_html=True)
        
        # Export buttons
        col2a, col2b = st.columns(2)
        
        with col2a:
            if st.button("📄 Download PDF Report", key="pdf_export", use_container_width=True):
                generate_pdf_report(data)
        
        with col2b:
            if st.button("📊 Download Excel Data", key="excel_export", use_container_width=True):
                generate_excel_report(data)
        
        # Report options
        st.subheader("Report Options")
        include_charts = st.checkbox("Include visualizations", value=True)
        include_raw_data = st.checkbox("Include raw comment data", value=True)
        include_analysis = st.checkbox("Include AI analysis", value=True)
        
        # Scheduled reports
        st.subheader("Scheduled Reports")
        st.selectbox("Report Frequency", ["Daily", "Weekly", "Monthly"])
        st.multiselect("Recipients", ["analyst@mca.gov.in", "manager@mca.gov.in"])

def render_statistical_summary(data):
    """Render statistical summary when API is not available"""
    sentiment_counts = pd.Series(data['sentiments']).value_counts()
    total = len(data['sentiments'])
    
    st.write(f"""
    **Statistical Summary:**
    
    📊 **Sentiment Distribution:** 
    - {sentiment_counts.get('Positive', 0)} positive comments ({round(sentiment_counts.get('Positive', 0)/total*100, 1)}%)
    - {sentiment_counts.get('Negative', 0)} negative comments ({round(sentiment_counts.get('Negative', 0)/total*100, 1)}%)
    - {sentiment_counts.get('Neutral', 0)} neutral comments ({round(sentiment_counts.get('Neutral', 0)/total*100, 1)}%)
    
    🌐 **Language Breakdown:**
    """)
    
    lang_counts = pd.Series(data['languages']).value_counts()
    for lang, count in lang_counts.items():
        st.write(f"- {lang}: {count} comments ({round(count/total*100, 1)}%)")
    
    avg_confidence = np.mean(data['confidences'])
    st.write(f"\n📈 **Average Confidence:** {avg_confidence:.1%}")

def generate_pdf_report(data):
    """Generate PDF report"""
    st.success("PDF report generation initiated. Download will start shortly.")
    st.info("📄 Report includes: Executive summary, sentiment analysis, visualizations, and recommendations.")

def generate_excel_report(data):
    """Generate Excel report"""
    # Create DataFrame
    df = pd.DataFrame({
        'Comment_ID': range(1, len(data['comments']) + 1),
        'Comment_Text': data['comments'],
        'Language': data['languages'],
        'Sentiment': data['sentiments'],
        'Confidence_Score': data['confidences'],
        'Date_Added': [datetime.now().strftime('%Y-%m-%d')] * len(data['comments'])
    })
    
    # Convert to Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Comment_Analysis', index=False)
    
    output.seek(0)
    
    st.download_button(
        label="📊 Download Excel File",
        data=output.getvalue(),
        file_name=f"MCA_Sentiment_Analysis_{datetime.now().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

def render_footer():
    """Render government footer"""
    st.markdown("""
    <div class="footer">
        <p>Developed for Ministry of Corporate Affairs (MCA), Government of India | Powered by Advanced AI</p>
        <p><a href="#" style="color: #003366;">Contact</a> | 
           <a href="#" style="color: #003366;">Terms of Service</a> | 
           <a href="#" style="color: #003366;">Privacy Policy</a></p>
        <p>© 2025 Government of India. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main dashboard function"""
    # Load custom CSS
    load_custom_css()
    
    # Render header
    render_government_header()
    
    # Check API connection
    api_connected, api_info = check_api_connection()
    
    if not api_connected:
        st.markdown(f"""
        <div class="alert-warning">
            <strong>⚠️ API Services Not Available</strong><br>
            Using mock data for demonstration. API services at {API_BASE_URL} are currently unavailable.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background-color: {COLORS['positive']}; color: white; padding: 10px 15px; border-radius: 5px; margin: 10px 0;">
            <strong>✅ API Services Connected</strong><br>
            Advanced multilingual sentiment analysis is available.
        </div>
        """, unsafe_allow_html=True)
    
    # Generate or load data
    if 'dashboard_data' not in st.session_state:
        st.session_state.dashboard_data = generate_mock_data()
    
    data = st.session_state.dashboard_data
    
    # Main dashboard content
    st.markdown('<div style="max-width: 1200px; margin: 0 auto;">', unsafe_allow_html=True)
    
    # Row 1: Overview Cards
    render_overview_cards(data)
    
    # Row 2: Visualizations
    render_visualizations(data)
    
    # Row 3: Comment Explorer
    render_comment_explorer(data)
    
    # Row 4: Summary & Export
    render_summary_and_export(data)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    render_footer()

if __name__ == "__main__":
    main()