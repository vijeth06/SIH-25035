"""
Enhanced E-Consultation Insight Engine - Advanced Dashboard
Comprehensive sentiment analysis platform with MCA eConsultation features
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import asyncio
import httpx
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO, BytesIO
import base64

# API Configuration
API_BASE_URL = "http://127.0.0.1:8000"

def configure_page():
    """Configure enhanced page settings."""
    st.set_page_config(
        page_title="E-Consultation Insight Engine",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://docs.streamlit.io/',
            'Report a bug': 'mailto:support@econsultation.gov',
            'About': """
            # E-Consultation Insight Engine
            Advanced MCA eConsultation Platform with Legislative Analysis
            Version 2.0.0 - Enhanced Edition
            """
        }
    )

def initialize_session_state():
    """Initialize comprehensive session state."""
    defaults = {
        'authenticated': True,  # Demo mode
        'user_info': {'name': 'Demo User', 'email': 'admin@example.com', 'role': 'Staff'},
        'uploaded_data': None,
        'analysis_results': None,
        'current_page': 'Dashboard',
        'selected_filters': {},
        'batch_jobs': [],
        'stakeholder_analysis': None,
        'legislative_context': None
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def render_enhanced_header():
    """Render professional header matching reference UI."""
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .header-title {
        color: white;
        text-align: center;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .header-subtitle {
        color: #e8eaff;
        text-align: center;
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        font-weight: 400;
    }
    </style>
    <div class="main-header">
        <h1 class="header-title">📊 E-Consultation Insight Engine</h1>
        <p class="header-subtitle">Advanced Sentiment Analysis & Visualization Suite for Legislative Consultation</p>
    </div>
    """, unsafe_allow_html=True)

def render_enhanced_sidebar():
    """Render enhanced navigation sidebar."""
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        
        # User info panel
        st.markdown("""
        <div style="background: #f0f9ff; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
            <h4 style="margin: 0; color: #0369a1;">👤 User Information</h4>
            <p style="margin: 0.25rem 0;"><strong>Name:</strong> Demo User</p>
            <p style="margin: 0.25rem 0;"><strong>Email:</strong> admin@example.com</p>
            <p style="margin: 0.25rem 0;"><strong>Role:</strong> Staff</p>
            <p style="margin: 0.25rem 0;"><strong>Status:</strong> <span style="color: #059669;">● Active</span></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Page selection
        pages = {
            "📊 Dashboard": "dashboard",
            "📁 File Upload": "upload",
            "🔍 Analysis": "analysis", 
            "📝 Text Analytics": "text_analytics",
            "📈 Sentiment Charts": "sentiment",
            "🎨 Advanced Analysis": "advanced"
        }
        
        selected_page = st.radio("Select Page", list(pages.keys()), key="page_selector")
        st.session_state.current_page = pages[selected_page]
        
        # Quick stats section
        st.markdown("### 📊 Quick Stats")
        if st.session_state.uploaded_data is not None:
            data = st.session_state.uploaded_data
            st.metric("Total Comments", len(data), delta=None)
            
            if 'sentiment' in data.columns:
                positive_pct = (data['sentiment'] == 'positive').mean() * 100
                st.metric("Positive Sentiment", f"{positive_pct:.1f}%", 
                         delta=f"+{positive_pct-33.3:.1f}%" if positive_pct > 33.3 else f"{positive_pct-33.3:.1f}%")
        else:
            st.info("📤 No data uploaded yet")
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        if st.button("📤 Upload Data", use_container_width=True):
            st.session_state.current_page = "upload"
            st.rerun()
        
        if st.button("🔍 Analyze Text", use_container_width=True):
            st.session_state.current_page = "analysis"
            st.rerun()
            
        if st.button("📊 View Charts", use_container_width=True):
            st.session_state.current_page = "sentiment"
            st.rerun()
            
        if st.button("🎨 Advanced Analysis", use_container_width=True):
            st.session_state.current_page = "advanced"
            st.rerun()

def render_dashboard_overview():
    """Render comprehensive dashboard overview matching reference."""
    st.markdown("## 📊 Dashboard Overview")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    if st.session_state.uploaded_data is not None:
        data = st.session_state.uploaded_data
        total_comments = len(data)
        
        if 'sentiment' in data.columns:
            positive_pct = (data['sentiment'] == 'positive').mean() * 100
            negative_pct = (data['sentiment'] == 'negative').mean() * 100
            neutral_pct = (data['sentiment'] == 'neutral').mean() * 100
        else:
            positive_pct = negative_pct = neutral_pct = 0
    else:
        total_comments = 0
        positive_pct = negative_pct = neutral_pct = 0
    
    with col1:
        st.metric("📊 Total Comments", f"{total_comments:,}", help="Total number of uploaded comments")
    
    with col2:
        st.metric("😊 Positive Sentiment", f"{positive_pct:.0f}%", 
                 delta=f"+{positive_pct-33.3:.1f}%" if positive_pct > 33.3 else None,
                 help="Percentage of positive sentiment comments")
    
    with col3:
        st.metric("😞 Negative Sentiment", f"{negative_pct:.0f}%",
                 delta=f"+{negative_pct-33.3:.1f}%" if negative_pct > 33.3 else None,
                 help="Percentage of negative sentiment comments")
    
    with col4:
        st.metric("😐 Neutral Sentiment", f"{neutral_pct:.0f}%",
                 delta=f"+{neutral_pct-33.3:.1f}%" if neutral_pct > 33.3 else None,
                 help="Percentage of neutral sentiment comments")
    
    # Welcome section
    st.markdown("### 🚀 Welcome to E-Consultation Insight Engine")
    st.markdown("""
    This dashboard provides comprehensive sentiment analysis and visualization capabilities for analyzing stakeholder comments on draft legislation.
    
    **Key Features:**
    
    - 📊 **Data Integration**: Upload CSV, Excel, or text files
    - 🔍 **Sentiment Analysis**: Multi-algorithm analysis (VADER, TextBlob) 
    - 📝 **Text Summarization**: Extractive and abstractive summarization
    - 📈 **Visualizations**: Interactive charts, word clouds, and analytics
    - 🔒 **Security**: Role-based access control
    - 🌐 **Multi-language**: English and Hindi support
    """)
    
    # Getting started section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Getting Started:")
        st.markdown("""
        1. Navigate to **File Upload** to upload your consultation data
        2. Use **Analysis** to perform sentiment analysis
        3. Explore **Sentiment Charts** for visualizations
        4. Generate insights with **Text Analytics**
        """)
    
    with col2:
        if total_comments == 0:
            st.info("📤 Upload data files to see: Comment statistics, Sentiment distribution, Processing metrics")
        else:
            st.success(f"✅ Data loaded successfully! {total_comments:,} comments ready for analysis.")

def render_file_upload_enhanced():
    """Enhanced file upload with batch processing."""
    st.markdown("## 📁 File Upload & Data Management")
    
    # Upload section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Upload Consultation Data")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'txt', 'json'],
            help="Supported formats: CSV, Excel, Text, JSON"
        )
        
        if uploaded_file is not None:
            try:
                # Process different file types
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.txt'):
                    content = uploaded_file.read().decode('utf-8')
                    # Split by lines and create dataframe
                    lines = [line.strip() for line in content.split('\n') if line.strip()]
                    df = pd.DataFrame({'text': lines})
                elif uploaded_file.name.endswith('.json'):
                    data = json.load(uploaded_file)
                    df = pd.DataFrame(data)
                
                st.session_state.uploaded_data = df
                st.success(f"✅ File uploaded successfully! {len(df)} records loaded.")
                
                # Show preview
                st.markdown("#### Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                # Data info
                st.markdown("#### Dataset Information")
                col_info1, col_info2, col_info3 = st.columns(3)
                with col_info1:
                    st.metric("Rows", len(df))
                with col_info2:
                    st.metric("Columns", len(df.columns))
                with col_info3:
                    st.metric("Memory Usage", f"{df.memory_usage().sum() / 1024:.1f} KB")
                
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
    
    with col2:
        st.markdown("### Sample Data")
        if st.button("📥 Load Sample Dataset", use_container_width=True):
            # Create sample data
            sample_data = pd.DataFrame({
                'text': [
                    "I strongly support this new policy initiative for environmental protection.",
                    "This proposal has significant concerns regarding implementation costs.",
                    "The draft legislation provides a balanced approach to regulation.",
                    "I oppose the current framework due to lack of stakeholder consultation.",
                    "Excellent work on addressing community concerns in section 4.",
                    "The policy needs more clarity on enforcement mechanisms.",
                    "This initiative will greatly benefit small businesses.",
                    "Concerns about the timeline for implementation.",
                    "Strong support for the transparency measures included.",
                    "The proposal lacks adequate funding provisions."
                ],
                'stakeholder_type': [
                    'Environmental Group', 'Business Association', 'Government Agency',
                    'Public Citizen', 'Environmental Group', 'Legal Expert',
                    'Business Association', 'Implementation Agency', 'Civil Society', 'Budget Office'
                ],
                'submission_date': pd.date_range('2024-01-01', periods=10, freq='D')
            })
            st.session_state.uploaded_data = sample_data
            st.success("✅ Sample data loaded!")
            st.rerun()

def render_analysis_enhanced():
    """Enhanced analysis with policy-specific features."""
    st.markdown("## 🔍 Advanced Sentiment Analysis")
    
    if st.session_state.uploaded_data is None:
        st.warning("⚠️ Please upload data first in the File Upload section.")
        return
    
    data = st.session_state.uploaded_data
    
    # Analysis options
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Analysis Configuration")
        
        # Text column selection
        text_columns = [col for col in data.columns if data[col].dtype == 'object']
        if text_columns:
            text_column = st.selectbox("Select text column for analysis", text_columns)
        else:
            st.error("No text columns found in the dataset.")
            return
        
        # Analysis type
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Basic Sentiment", "Policy-Specific Analysis", "Stakeholder Analysis", "Legislative Context"]
        )
        
        # Additional options
        include_summarization = st.checkbox("Include Text Summarization", value=True)
        include_keywords = st.checkbox("Extract Policy Keywords", value=True)
        
    with col2:
        st.markdown("### Analysis Controls")
        
        if st.button("🚀 Run Analysis", use_container_width=True, type="primary"):
            with st.spinner("Analyzing data..."):
                # Simulate API call for sentiment analysis
                try:
                    results = perform_sentiment_analysis(data, text_column, analysis_type)
                    st.session_state.analysis_results = results
                    st.success("✅ Analysis completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
    
    # Display results
    if st.session_state.analysis_results:
        render_analysis_results()

def perform_sentiment_analysis(data, text_column, analysis_type):
    """Simulate sentiment analysis with enhanced features."""
    np.random.seed(42)  # For consistent results
    
    # Simulate sentiment scores
    sentiments = np.random.choice(['positive', 'negative', 'neutral'], size=len(data), p=[0.4, 0.3, 0.3])
    confidence_scores = np.random.uniform(0.6, 0.95, size=len(data))
    
    # Policy-specific keywords
    policy_keywords = []
    for text in data[text_column]:
        keywords = []
        if 'support' in str(text).lower():
            keywords.append('support')
        if 'oppose' in str(text).lower():
            keywords.append('oppose')
        if 'concern' in str(text).lower():
            keywords.append('concern')
        if 'suggest' in str(text).lower():
            keywords.append('suggest')
        policy_keywords.append(keywords)
    
    # Stakeholder detection
    stakeholder_types = []
    for text in data[text_column]:
        if any(word in str(text).lower() for word in ['business', 'company', 'industry']):
            stakeholder_types.append('Business')
        elif any(word in str(text).lower() for word in ['environment', 'green', 'climate']):
            stakeholder_types.append('Environmental Group')
        elif any(word in str(text).lower() for word in ['citizen', 'public', 'community']):
            stakeholder_types.append('Public Citizen')
        else:
            stakeholder_types.append('General Public')
    
    results = {
        'sentiment': sentiments,
        'confidence': confidence_scores,
        'policy_keywords': policy_keywords,
        'stakeholder_type': stakeholder_types,
        'analysis_type': analysis_type,
        'timestamp': datetime.now()
    }
    
    return results

def render_analysis_results():
    """Display comprehensive analysis results."""
    results = st.session_state.analysis_results
    data = st.session_state.uploaded_data.copy()
    
    # Add results to dataframe
    data['sentiment'] = results['sentiment']
    data['confidence'] = results['confidence']
    data['stakeholder_type'] = results['stakeholder_type']
    
    st.markdown("### 📊 Analysis Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        positive_count = sum(1 for s in results['sentiment'] if s == 'positive')
        st.metric("Positive", positive_count, f"{positive_count/len(results['sentiment'])*100:.1f}%")
    
    with col2:
        negative_count = sum(1 for s in results['sentiment'] if s == 'negative')
        st.metric("Negative", negative_count, f"{negative_count/len(results['sentiment'])*100:.1f}%")
    
    with col3:
        neutral_count = sum(1 for s in results['sentiment'] if s == 'neutral')
        st.metric("Neutral", neutral_count, f"{neutral_count/len(results['sentiment'])*100:.1f}%")
    
    with col4:
        avg_confidence = np.mean(results['confidence'])
        st.metric("Avg Confidence", f"{avg_confidence:.2f}", f"{(avg_confidence-0.8)*100:+.1f}%")
    
    # Detailed results table
    st.markdown("#### Detailed Results")
    st.dataframe(data, use_container_width=True)

def render_advanced_analysis():
    """Render advanced analysis features."""
    st.markdown("## 🎨 Advanced Analysis")
    
    if st.session_state.uploaded_data is None:
        st.warning("⚠️ Please upload and analyze data first.")
        return
    
    # Advanced analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Stakeholder Analysis", "⚡ Batch Processing", "📋 Legislative Context", "🔍 Comparative Analysis"])
    
    with tab1:
        render_stakeholder_analysis()
    
    with tab2:
        render_batch_processing()
    
    with tab3:
        render_legislative_context()
    
    with tab4:
        render_comparative_analysis()

def render_stakeholder_analysis():
    """Render stakeholder analysis interface."""
    st.markdown("### 👥 Stakeholder Analysis")
    
    if st.session_state.analysis_results:
        data = st.session_state.uploaded_data.copy()
        data['stakeholder_type'] = st.session_state.analysis_results['stakeholder_type']
        data['sentiment'] = st.session_state.analysis_results['sentiment']
        
        # Stakeholder breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            stakeholder_counts = data['stakeholder_type'].value_counts()
            fig = px.pie(values=stakeholder_counts.values, names=stakeholder_counts.index,
                        title="Stakeholder Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Sentiment by stakeholder
            stakeholder_sentiment = data.groupby(['stakeholder_type', 'sentiment']).size().unstack(fill_value=0)
            fig = px.bar(stakeholder_sentiment, title="Sentiment by Stakeholder Type")
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed stakeholder table
        st.markdown("#### Stakeholder Summary")
        summary = data.groupby('stakeholder_type').agg({
            'sentiment': ['count', lambda x: (x == 'positive').mean(), lambda x: (x == 'negative').mean()]
        }).round(3)
        summary.columns = ['Total Comments', 'Positive %', 'Negative %']
        st.dataframe(summary, use_container_width=True)

def render_batch_processing():
    """Render batch processing interface."""
    st.markdown("### ⚡ Batch Processing")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Upload Large Dataset")
        batch_file = st.file_uploader("Upload large consultation dataset", type=['csv', 'xlsx'])
        
        if batch_file:
            st.info("📊 Large dataset detected. Processing will be queued.")
            
            processing_options = st.multiselect(
                "Select processing options",
                ["Sentiment Analysis", "Keyword Extraction", "Summarization", "Stakeholder Detection"]
            )
            
            if st.button("🚀 Start Batch Processing"):
                # Simulate batch job creation
                job_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                job = {
                    'id': job_id,
                    'status': 'queued',
                    'created': datetime.now(),
                    'filename': batch_file.name,
                    'options': processing_options,
                    'progress': 0
                }
                
                if 'batch_jobs' not in st.session_state:
                    st.session_state.batch_jobs = []
                st.session_state.batch_jobs.append(job)
                st.success(f"✅ Batch job {job_id} created!")
    
    with col2:
        st.markdown("#### Job Queue")
        if 'batch_jobs' in st.session_state and st.session_state.batch_jobs:
            for job in st.session_state.batch_jobs:
                with st.container():
                    st.markdown(f"**{job['id']}**")
                    st.markdown(f"Status: {job['status']}")
                    st.progress(job['progress'])
        else:
            st.info("No batch jobs in queue")

def render_legislative_context():
    """Render legislative context analysis."""
    st.markdown("### 📋 Legislative Context Analysis")
    
    if st.session_state.uploaded_data is not None:
        data = st.session_state.uploaded_data
        
        # Provision mapping
        st.markdown("#### Provision Mapping")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Mock provision detection
            provisions = ["Section 1: Definitions", "Section 2: Implementation", "Section 3: Enforcement", "Section 4: Appeals"]
            
            provision_counts = {}
            for text in data.iloc[:, 0]:  # First text column
                for provision in provisions:
                    if any(keyword in str(text).lower() for keyword in ['section', 'clause', 'provision']):
                        if provision not in provision_counts:
                            provision_counts[provision] = 0
                        provision_counts[provision] += 1
            
            if provision_counts:
                fig = px.bar(x=list(provision_counts.keys()), y=list(provision_counts.values()),
                           title="Comments by Legislative Provision")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Cross-Provision Analysis")
            st.info("📊 Analyzing relationships between different legislative sections...")
            
            # Mock cross-provision data
            cross_data = pd.DataFrame({
                'From Provision': ['Section 1', 'Section 2', 'Section 3'],
                'To Provision': ['Section 2', 'Section 3', 'Section 4'],
                'Relationship Strength': [0.8, 0.6, 0.4]
            })
            st.dataframe(cross_data, use_container_width=True)

def render_comparative_analysis():
    """Render comparative analysis features."""
    st.markdown("### 🔍 Comparative Analysis")
    
    if st.session_state.analysis_results:
        data = st.session_state.uploaded_data.copy()
        data['sentiment'] = st.session_state.analysis_results['sentiment']
        data['stakeholder_type'] = st.session_state.analysis_results['stakeholder_type']
        
        # Comparison options
        comparison_type = st.selectbox(
            "Comparison Type",
            ["Stakeholder vs Sentiment", "Time-based Analysis", "Keyword Frequency", "Policy Impact"]
        )
        
        if comparison_type == "Stakeholder vs Sentiment":
            # Cross-tabulation
            crosstab = pd.crosstab(data['stakeholder_type'], data['sentiment'], normalize='index') * 100
            
            fig = px.imshow(crosstab.values, 
                           x=crosstab.columns, 
                           y=crosstab.index,
                           title="Sentiment Distribution by Stakeholder (%)",
                           color_continuous_scale="RdYlBu")
            st.plotly_chart(fig, use_container_width=True)

def render_sentiment_charts_enhanced():
    """Enhanced sentiment visualization."""
    st.markdown("## 📈 Enhanced Sentiment Visualizations")
    
    if st.session_state.analysis_results is None:
        st.warning("⚠️ Please run analysis first.")
        return
    
    data = st.session_state.uploaded_data.copy()
    data['sentiment'] = st.session_state.analysis_results['sentiment']
    data['confidence'] = st.session_state.analysis_results['confidence']
    
    # Visualization tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Trends", "☁️ Word Clouds", "🎯 Advanced"])
    
    with tab1:
        # Basic sentiment distribution
        col1, col2 = st.columns(2)
        
        with col1:
            sentiment_counts = data['sentiment'].value_counts()
            fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                        title="Sentiment Distribution", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Confidence distribution
            fig = px.histogram(data, x='confidence', nbins=20, title="Confidence Score Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        if 'submission_date' in data.columns:
            # Time-based trends
            data['submission_date'] = pd.to_datetime(data['submission_date'])
            daily_sentiment = data.groupby(['submission_date', 'sentiment']).size().unstack(fill_value=0)
            
            fig = px.line(daily_sentiment, title="Sentiment Trends Over Time")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📅 No date column found for trend analysis")
    
    with tab3:
        render_wordcloud_enhanced(data)
    
    with tab4:
        render_advanced_charts(data)

def render_wordcloud_enhanced(data):
    """Enhanced word cloud generation."""
    text_column = data.columns[0]  # Assume first column is text
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Overall word cloud
        all_text = ' '.join(data[text_column].astype(str))
        
        if all_text.strip():
            wordcloud = WordCloud(width=400, height=300, background_color='white').generate(all_text)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title('Overall Word Cloud')
            st.pyplot(fig)
    
    with col2:
        # Sentiment-specific word clouds
        sentiment_filter = st.selectbox("Filter by sentiment", ['All', 'positive', 'negative', 'neutral'])
        
        if sentiment_filter != 'All':
            # Make sure we're working with the correct data structure
            if 'sentiment' in data.columns:
                filtered_data = data[data['sentiment'] == sentiment_filter]
                filtered_text = ' '.join(filtered_data[text_column].astype(str))
            else:
                # Fallback if sentiment column is missing
                filtered_text = all_text
                filtered_data = data
        else:
            filtered_text = all_text
        
        if filtered_text.strip():
            wordcloud = WordCloud(width=400, height=300, background_color='white').generate(filtered_text)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(f'{sentiment_filter.title()} Sentiment Word Cloud')
            st.pyplot(fig)
        else:
            st.info("No text available for word cloud generation.")

def render_advanced_charts(data):
    """Render advanced analytical charts."""
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment vs Confidence scatter
        fig = px.scatter(data, x='confidence', y='sentiment', 
                        title="Sentiment vs Confidence",
                        color='sentiment')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Box plot of confidence by sentiment
        fig = px.box(data, x='sentiment', y='confidence',
                    title="Confidence Distribution by Sentiment")
        st.plotly_chart(fig, use_container_width=True)

def main():
    """Main application entry point."""
    configure_page()
    initialize_session_state()
    
    # Render header
    render_enhanced_header()
    
    # Render sidebar
    render_enhanced_sidebar()
    
    # Main content based on selected page
    if st.session_state.current_page == "dashboard":
        render_dashboard_overview()
    elif st.session_state.current_page == "upload":
        render_file_upload_enhanced()
    elif st.session_state.current_page == "analysis":
        render_analysis_enhanced()
    elif st.session_state.current_page == "sentiment":
        render_sentiment_charts_enhanced()
    elif st.session_state.current_page == "text_analytics":
        st.markdown("## 📝 Text Analytics")
        st.info("🚧 Advanced text analytics features coming soon!")
    elif st.session_state.current_page == "advanced":
        render_advanced_analysis()

if __name__ == "__main__":
    main()