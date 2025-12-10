"""
Analysis dashboard component for running sentiment analysis.
"""

import streamlit as st
import requests
import pandas as pd
import json
from typing import Dict, List, Any, Optional
from .auth import get_headers
import time


API_BASE_URL = "http://127.0.0.1:8002"


def render_analysis_dashboard():
    """Render the main analysis dashboard."""
    st.markdown("## 🔍 Sentiment Analysis & Processing")
    
    st.markdown("""
    Perform comprehensive sentiment analysis on your consultation data using multiple 
    algorithms and analysis techniques.
    """)
    
    # Create tabs for different analysis types
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Single Text Analysis", "📊 Batch Analysis", 
        "🎯 Advanced Analysis", "📈 Analysis History"
    ])
    
    with tab1:
        render_single_text_analysis()
    
    with tab2:
        render_batch_analysis()
    
    with tab3:
        render_advanced_analysis()
    
    with tab4:
        render_analysis_history()


def render_single_text_analysis():
    """Render single text analysis interface."""
    st.markdown("### 📝 Single Text Analysis")
    st.markdown("Analyze individual comments or text snippets with detailed results.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Text input
        text_input = st.text_area(
            "Enter text to analyze:",
            height=150,
            placeholder="Type or paste your comment here for sentiment analysis...",
            help="Enter the text you want to analyze for sentiment, emotion, and other insights."
        )
        
        # Analysis options
        st.markdown("#### ⚙️ Analysis Options")
        
        col_opt1, col_opt2 = st.columns(2)
        
        with col_opt1:
            analysis_methods = st.multiselect(
                "Analysis Methods",
                ["VADER", "TextBlob", "Ensemble"],
                default=["VADER", "TextBlob"],
                help="Select which sentiment analysis methods to use"
            )
            
            include_emotions = st.checkbox(
                "Include Emotion Analysis",
                value=True,
                help="Detect emotions like support, concern, suggestion, etc."
            )
        
        with col_opt2:
            include_aspects = st.checkbox(
                "Include Aspect Analysis",
                value=True,
                help="Analyze sentiment toward specific aspects or topics"
            )
            
            include_summary = st.checkbox(
                "Include Summary",
                value=False,
                help="Generate a brief summary of the text"
            )
        
        # Analysis button
        if st.button("🔍 Analyze Text", type="primary", use_container_width=True, disabled=not text_input.strip()):
            perform_single_text_analysis(text_input, {
                'methods': analysis_methods,
                'include_emotions': include_emotions,
                'include_aspects': include_aspects,
                'include_summary': include_summary
            })
    
    with col2:
        render_analysis_guidelines()


def render_batch_analysis():
    """Render batch analysis interface."""
    st.markdown("### 📊 Batch Analysis")
    st.markdown("Process multiple texts simultaneously for comprehensive analysis.")
    
    # Data source selection
    data_source = st.radio(
        "Select Data Source:",
        ["Uploaded Files", "Manual Input", "Saved Comments"],
        horizontal=True
    )
    
    if data_source == "Uploaded Files":
        render_uploaded_files_analysis()
    elif data_source == "Manual Input":
        render_manual_batch_input()
    else:
        render_saved_comments_analysis()


def render_uploaded_files_analysis():
    """Render analysis for uploaded files."""
    st.markdown("#### 📁 Analyze Uploaded Files")
    
    # Check for uploaded data
    uploaded_data = st.session_state.get('uploaded_data')
    if uploaded_data is None or (hasattr(uploaded_data, 'empty') and uploaded_data.empty):
        st.info("📝 No uploaded files found. Please upload data first.")

        if st.button("📁 Go to File Upload"):
            st.session_state.current_page = "File Upload"
            st.rerun()

        return
    
    # Display available uploads
    st.markdown("**Available Uploads:**")
    
    # Mock uploaded files data
    uploads = [
        {"id": 1, "filename": "consultation_comments.csv", "records": 1250, "status": "Ready"},
        {"id": 2, "filename": "stakeholder_feedback.xlsx", "records": 890, "status": "Ready"},
        {"id": 3, "filename": "public_comments.txt", "records": 456, "status": "Processing"},
    ]
    
    for upload in uploads:
        with st.expander(f"📄 {upload['filename']} ({upload['records']} records)"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Records", upload['records'])
            
            with col2:
                st.metric("Status", upload['status'])
            
            with col3:
                if upload['status'] == 'Ready':
                    if st.button(f"🔍 Analyze", key=f"analyze_{upload['id']}", use_container_width=True):
                        start_batch_analysis(upload['id'])
                else:
                    st.info("Processing...")


def render_manual_batch_input():
    """Render manual batch input interface."""
    st.markdown("#### ✏️ Manual Batch Input")
    
    # Input method selection
    input_method = st.selectbox(
        "Input Method:",
        ["Paste Multiple Comments", "Upload Text File", "Enter CSV Data"]
    )
    
    if input_method == "Paste Multiple Comments":
        comments_text = st.text_area(
            "Enter comments (one per line):",
            height=200,
            placeholder="Comment 1\nComment 2\nComment 3\n...",
            help="Enter each comment on a separate line"
        )
        
        if comments_text.strip():
            comments = [line.strip() for line in comments_text.split('\n') if line.strip()]
            st.info(f"📝 Found {len(comments)} comments")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                batch_size = st.selectbox("Batch Size:", [10, 25, 50, 100], index=1)
            
            with col2:
                analysis_speed = st.selectbox("Analysis Speed:", ["Fast", "Balanced", "Comprehensive"], index=1)
            
            if st.button("🚀 Start Batch Analysis", type="primary", use_container_width=True):
                perform_batch_text_analysis(comments, batch_size, analysis_speed)


def render_saved_comments_analysis():
    """Render analysis for saved comments."""
    st.markdown("#### 💾 Analyze Saved Comments")
    
    if st.session_state.get('saved_texts'):
        saved_texts = st.session_state.saved_texts
        
        st.info(f"📝 Found {len(saved_texts)} saved comments")
        
        # Display saved texts preview
        with st.expander("📋 Preview Saved Comments"):
            for i, item in enumerate(saved_texts[:5]):  # Show first 5
                st.markdown(f"**{i+1}.** {item['text'][:100]}{'...' if len(item['text']) > 100 else ''}")
            
            if len(saved_texts) > 5:
                st.markdown(f"*... and {len(saved_texts) - 5} more comments*")
        
        if st.button("🔍 Analyze All Saved Comments", type="primary", use_container_width=True):
            texts = [item['text'] for item in saved_texts]
            perform_batch_text_analysis(texts, 25, "Balanced")
    
    else:
        st.info("💾 No saved comments found.")
        st.markdown("Save comments from the 'File Upload' page to analyze them here.")


def render_advanced_analysis():
    """Render advanced analysis options."""
    st.markdown("### 🎯 Advanced Analysis Options")
    
    st.markdown("""
    Configure advanced analysis parameters and specialized processing options.
    """)
    
    # Advanced configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔧 Analysis Configuration")
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            0.0, 1.0, 0.7,
            help="Minimum confidence score for results"
        )
        
        language_detection = st.checkbox(
            "Auto Language Detection",
            value=True,
            help="Automatically detect text language"
        )
        
        aspect_categories = st.multiselect(
            "Aspect Categories",
            ["Legal Framework", "Implementation", "Compliance", "Impact Assessment", "Stakeholder Concerns"],
            default=["Legal Framework", "Implementation"]
        )
        
        custom_keywords = st.text_area(
            "Custom Keywords (comma-separated):",
            placeholder="regulation, compliance, impact, concern, support",
            help="Add custom keywords for aspect analysis"
        )
    
    with col2:
        st.markdown("#### 📊 Output Configuration")
        
        output_format = st.selectbox(
            "Output Format",
            ["Detailed JSON", "Summary Report", "Statistical Overview", "Visualization Ready"]
        )
        
        include_raw_scores = st.checkbox(
            "Include Raw Scores",
            value=False,
            help="Include detailed numerical scores for all methods"
        )
        
        aggregation_level = st.selectbox(
            "Aggregation Level",
            ["Individual Comments", "Section-wise", "Overall Summary", "All Levels"]
        )
        
        export_results = st.checkbox(
            "Auto-export Results",
            value=False,
            help="Automatically export results after analysis"
        )
    
    # Custom analysis templates
    st.markdown("#### 📝 Analysis Templates")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Standard Legislative Analysis", use_container_width=True):
            apply_analysis_template("legislative")
    
    with col2:
        if st.button("🔍 Public Consultation Analysis", use_container_width=True):
            apply_analysis_template("consultation")
    
    with col3:
        if st.button("💼 Stakeholder Feedback Analysis", use_container_width=True):
            apply_analysis_template("stakeholder")
    
    # Save custom configuration
    st.markdown("#### 💾 Configuration Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        config_name = st.text_input("Configuration Name:", placeholder="My Custom Config")
        
        if st.button("💾 Save Configuration", use_container_width=True):
            if config_name:
                save_analysis_configuration(config_name, {
                    'confidence_threshold': confidence_threshold,
                    'language_detection': language_detection,
                    'aspect_categories': aspect_categories,
                    'custom_keywords': custom_keywords,
                    'output_format': output_format,
                    'include_raw_scores': include_raw_scores,
                    'aggregation_level': aggregation_level,
                    'export_results': export_results
                })
                st.success(f"✅ Configuration '{config_name}' saved!")
            else:
                st.error("⚠️ Please enter a configuration name.")
    
    with col2:
        saved_configs = get_saved_configurations()
        
        if saved_configs:
            selected_config = st.selectbox("Load Configuration:", [""] + saved_configs)
            
            if st.button("📂 Load Configuration", use_container_width=True):
                if selected_config:
                    load_analysis_configuration(selected_config)
                    st.success(f"✅ Configuration '{selected_config}' loaded!")
                    st.rerun()


def render_analysis_history():
    """Render analysis history and results."""
    st.markdown("### 📈 Analysis History & Results")
    
    # History filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        date_filter = st.selectbox("Time Period:", ["Today", "Last 7 days", "Last 30 days", "All time"])
    
    with col2:
        type_filter = st.selectbox("Analysis Type:", ["All", "Single Text", "Batch", "Advanced"])
    
    with col3:
        status_filter = st.selectbox("Status:", ["All", "Completed", "In Progress", "Failed"])
    
    # Mock analysis history
    history_data = [
        {
            "timestamp": "2025-01-09 14:30",
            "type": "Batch Analysis",
            "items": 1250,
            "duration": "2m 15s",
            "status": "✅ Completed",
            "confidence": "0.82"
        },
        {
            "timestamp": "2025-01-09 13:15",
            "type": "Single Text",
            "items": 1,
            "duration": "0.5s",
            "status": "✅ Completed",
            "confidence": "0.91"
        },
        {
            "timestamp": "2025-01-09 11:45",
            "type": "Advanced Analysis",
            "items": 890,
            "duration": "1m 48s",
            "status": "✅ Completed",
            "confidence": "0.78"
        },
        {
            "timestamp": "2025-01-09 10:20",
            "type": "Batch Analysis",
            "items": 2100,
            "duration": "3m 22s",
            "status": "⚠️ Partial",
            "confidence": "0.65"
        }
    ]
    
    # Display history table
    if history_data:
        df = pd.DataFrame(history_data)
        
        st.markdown("#### 📊 Recent Analysis Sessions")
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Refresh History", use_container_width=True):
                st.rerun()
        
        with col2:
            if st.button("📤 Export History", use_container_width=True):
                export_analysis_history(history_data)
        
        with col3:
            if st.button("📊 View Detailed Results", use_container_width=True):
                st.session_state.show_detailed_results = True
    
    # Recent results summary
    if st.session_state.get('analysis_results'):
        st.markdown("#### 📈 Latest Analysis Results")
        
        results = st.session_state.analysis_results
        display_results_summary(results)


def render_analysis_guidelines():
    """Render analysis guidelines and tips."""
    st.markdown("#### 📋 Analysis Guidelines")
    
    with st.expander("🎯 Analysis Methods", expanded=False):
        st.markdown("""
        **VADER Sentiment Analysis:**
        - Best for social media and informal text
        - Handles emojis and slang
        - Provides compound scores
        
        **TextBlob Analysis:**
        - Good for formal text
        - Provides polarity and subjectivity
        - Language model based
        
        **Ensemble Method:**
        - Combines multiple methods
        - More robust results
        - Higher accuracy for complex text
        """)
    
    with st.expander("💡 Best Practices"):
        st.markdown("""
        **Text Preparation:**
        - Clean text of HTML tags
        - Handle special characters
        - Ensure proper encoding
        
        **Analysis Settings:**
        - Use multiple methods for accuracy
        - Enable emotion analysis for insights
        - Include aspect analysis for detailed feedback
        
        **Result Interpretation:**
        - Check confidence scores
        - Compare method results
        - Consider context and domain
        """)
    
    with st.expander("⚡ Performance Tips"):
        st.markdown("""
        **Batch Processing:**
        - Use appropriate batch sizes (25-100)
        - Monitor processing time
        - Enable progress tracking
        
        **Resource Management:**
        - Large files: use streaming
        - Multiple files: process sequentially
        - Real-time: use caching
        """)


def perform_single_text_analysis(text: str, options: Dict):
    """Perform analysis on single text."""
    with st.spinner("🔍 Analyzing text..."):
        try:
            # Prepare API request
            payload = {
                "text": text,
                "methods": [method.lower() for method in options['methods']],
                "include_emotions": options['include_emotions'],
                "include_aspects": options['include_aspects']
            }
            
            # Make API request
            response = requests.post(
                f"{API_BASE_URL}/api/v1/analysis/comprehensive",
                json=payload,
                headers=get_headers(),
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                display_single_analysis_results(result, options)
            else:
                st.error(f"❌ Analysis failed: {response.json().get('detail', 'Unknown error')}")
        
        except requests.RequestException as e:
            st.error(f"❌ Connection error: {str(e)}")
        except Exception as e:
            st.error(f"❌ Analysis error: {str(e)}")


def display_single_analysis_results(result: Dict, options: Dict):
    """Display results from single text analysis."""
    st.markdown("#### 🎉 Analysis Complete!")
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        sentiment = result.get('overall_sentiment', 'neutral')
        st.metric("Overall Sentiment", sentiment.title())
    
    with col2:
        confidence = result.get('overall_confidence', 0)
        st.metric("Confidence", f"{confidence:.1%}")
    
    with col3:
        processing_time = result.get('processing_time_ms', 0)
        st.metric("Processing Time", f"{processing_time}ms")
    
    with col4:
        if options['include_emotions'] and result.get('emotion_result'):
            emotion = result['emotion_result'].get('emotion_label', 'neutral')
            st.metric("Primary Emotion", emotion.title())
    
    # Detailed results tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Sentiment Details", "🎭 Emotions", "🎯 Aspects", "📋 Raw Data"])
    
    with tab1:
        display_sentiment_details(result)
    
    with tab2:
        if options['include_emotions']:
            display_emotion_details(result)
        else:
            st.info("Emotion analysis not enabled for this analysis.")
    
    with tab3:
        if options['include_aspects']:
            display_aspect_details(result)
        else:
            st.info("Aspect analysis not enabled for this analysis.")
    
    with tab4:
        st.json(result)


def display_sentiment_details(result: Dict):
    """Display detailed sentiment analysis results."""
    sentiment_results = result.get('sentiment_results', [])
    
    if not sentiment_results:
        st.warning("No sentiment results available.")
        return
    
    for sentiment_result in sentiment_results:
        method = sentiment_result.get('method', 'Unknown')
        
        with st.expander(f"📈 {method} Analysis Results"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Sentiment", sentiment_result.get('sentiment_label', 'Unknown').title())
                st.metric("Confidence", f"{sentiment_result.get('confidence_score', 0):.1%}")
            
            with col2:
                st.metric("Positive Score", f"{sentiment_result.get('positive_score', 0):.3f}")
                st.metric("Negative Score", f"{sentiment_result.get('negative_score', 0):.3f}")
            
            if sentiment_result.get('compound_score') is not None:
                st.metric("Compound Score", f"{sentiment_result.get('compound_score', 0):.3f}")


def display_emotion_details(result: Dict):
    """Display emotion analysis results."""
    emotion_result = result.get('emotion_result', {})
    
    if not emotion_result:
        st.warning("No emotion results available.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Primary Emotion", emotion_result.get('emotion_label', 'Unknown').title())
        st.metric("Confidence", f"{emotion_result.get('confidence_score', 0):.1%}")
    
    with col2:
        detected_emotions = emotion_result.get('detected_emotions', [])
        if detected_emotions:
            st.markdown("**Detected Emotions:**")
            for emotion in detected_emotions:
                st.write(f"• {emotion.title()}")
    
    # Emotion scores
    emotion_scores = emotion_result.get('emotion_scores', {})
    if emotion_scores:
        st.markdown("**Emotion Scores:**")
        
        emotions = list(emotion_scores.keys())
        scores = list(emotion_scores.values())
        
        chart_data = pd.DataFrame({
            'Emotion': emotions,
            'Score': scores
        })
        
        st.bar_chart(chart_data.set_index('Emotion'))


def display_aspect_details(result: Dict):
    """Display aspect-based sentiment analysis results."""
    aspect_sentiments = result.get('aspect_sentiments', [])
    
    if not aspect_sentiments:
        st.info("No specific aspects were identified in this text.")
        return
    
    st.markdown("**Identified Aspects:**")
    
    for aspect_result in aspect_sentiments:
        aspect = aspect_result.get('aspect', 'Unknown')
        sentiment = aspect_result.get('sentiment', 'neutral')
        confidence = aspect_result.get('confidence', 0)
        context = aspect_result.get('context', '')
        
        with st.expander(f"🎯 {aspect} - {sentiment.title()} ({confidence:.1%})"):
            st.markdown(f"**Context:** {context}")
            
            if aspect_result.get('law_section'):
                st.markdown(f"**Law Section:** {aspect_result['law_section']}")


def perform_batch_text_analysis(texts: List[str], batch_size: int, speed: str):
    """Perform batch analysis on multiple texts."""
    total_texts = len(texts)
    
    with st.spinner(f"🚀 Analyzing {total_texts} texts..."):
        progress_bar = st.progress(0)
        status_container = st.empty()
        
        # Configure analysis based on speed setting
        methods = {
            "Fast": ["vader"],
            "Balanced": ["vader", "textblob"],
            "Comprehensive": ["vader", "textblob", "ensemble"]
        }
        
        try:
            # Prepare batch request
            payload = {
                "texts": texts,
                "methods": methods.get(speed, ["vader"]),
                "include_emotions": speed in ["Balanced", "Comprehensive"],
                "batch_size": batch_size
            }
            
            # Make batch API request
            response = requests.post(
                f"{API_BASE_URL}/api/v1/analysis/batch",
                json=payload,
                headers=get_headers(),
                timeout=300  # 5 minute timeout for batch processing
            )
            
            if response.status_code == 200:
                results = response.json()
                
                # Store results in session state
                st.session_state.analysis_results = results
                
                # Update progress
                progress_bar.progress(1.0)
                status_container.success(f"✅ Successfully analyzed {len(results)} texts!")
                
                # Display summary
                display_batch_results_summary(results)
                
                st.info("📊 Navigate to 'Sentiment Charts' to view detailed visualizations.")
            
            else:
                error_msg = response.json().get('detail', 'Unknown error')
                st.error(f"❌ Batch analysis failed: {error_msg}")
        
        except requests.RequestException as e:
            st.error(f"❌ Connection error: {str(e)}")
        except Exception as e:
            st.error(f"❌ Analysis error: {str(e)}")


def display_batch_results_summary(results: List[Dict]):
    """Display summary of batch analysis results."""
    st.markdown("#### 📊 Batch Analysis Results Summary")
    
    # Calculate statistics
    total_count = len(results)
    
    if total_count == 0:
        st.warning("No results to display.")
        return
    
    # Count sentiments
    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
    confidence_scores = []
    
    for result in results:
        sentiment = result.get('overall_sentiment', 'neutral')
        if sentiment in sentiment_counts:
            sentiment_counts[sentiment] += 1
        
        confidence = result.get('overall_confidence', 0)
        if confidence > 0:
            confidence_scores.append(confidence)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Analyzed", f"{total_count:,}")
    
    with col2:
        positive_pct = sentiment_counts['positive'] / total_count
        st.metric("😊 Positive", f"{positive_pct:.1%}", f"{sentiment_counts['positive']} texts")
    
    with col3:
        negative_pct = sentiment_counts['negative'] / total_count
        st.metric("😔 Negative", f"{negative_pct:.1%}", f"{sentiment_counts['negative']} texts")
    
    with col4:
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    
    # Quick visualization
    if total_count > 0:
        st.markdown("#### 📈 Quick Distribution")
        
        chart_data = pd.DataFrame({
            'Sentiment': ['Positive', 'Negative', 'Neutral'],
            'Count': [sentiment_counts['positive'], sentiment_counts['negative'], sentiment_counts['neutral']]
        })
        
        st.bar_chart(chart_data.set_index('Sentiment'))


# Helper functions for advanced analysis

def apply_analysis_template(template_type: str):
    """Apply predefined analysis template."""
    templates = {
        "legislative": {
            "confidence_threshold": 0.8,
            "aspect_categories": ["Legal Framework", "Implementation", "Compliance"],
            "methods": ["vader", "textblob", "ensemble"],
            "include_emotions": True
        },
        "consultation": {
            "confidence_threshold": 0.7,
            "aspect_categories": ["Public Impact", "Stakeholder Concerns", "Implementation"],
            "methods": ["vader", "textblob"],
            "include_emotions": True
        },
        "stakeholder": {
            "confidence_threshold": 0.75,
            "aspect_categories": ["Business Impact", "Regulatory Burden", "Compliance Costs"],
            "methods": ["ensemble"],
            "include_emotions": True
        }
    }
    
    if template_type in templates:
        st.success(f"✅ Applied {template_type} template!")
        # In a real implementation, this would update the form fields


def save_analysis_configuration(name: str, config: Dict):
    """Save analysis configuration."""
    if 'saved_configs' not in st.session_state:
        st.session_state.saved_configs = {}
    
    st.session_state.saved_configs[name] = config


def get_saved_configurations() -> List[str]:
    """Get list of saved configuration names."""
    return list(st.session_state.get('saved_configs', {}).keys())


def load_analysis_configuration(name: str):
    """Load saved analysis configuration."""
    configs = st.session_state.get('saved_configs', {})
    
    if name in configs:
        config = configs[name]
        # In a real implementation, this would populate the form fields
        return config
    
    return None


def start_batch_analysis(upload_id: int):
    """Start batch analysis for uploaded file."""
    with st.spinner(f"🚀 Starting analysis for upload ID {upload_id}..."):
        # In a real implementation, this would trigger the batch analysis
        time.sleep(2)  # Simulate processing time
        st.success("✅ Batch analysis started! Check Analysis History for progress.")


def export_analysis_history(history_data: List[Dict]):
    """Export analysis history to downloadable format."""
    df = pd.DataFrame(history_data)
    csv = df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download History (CSV)",
        data=csv,
        file_name=f"analysis_history_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )


def display_results_summary(results: List[Dict]):
    """Display summary of analysis results."""
    if not results:
        st.info("No recent results to display.")
        return
    
    st.markdown(f"**Latest analysis:** {len(results)} items processed")
    
    # Show first few results
    with st.expander("📋 Sample Results"):
        for i, result in enumerate(results[:3]):
            text_preview = result.get('text', '')[:100] + '...' if len(result.get('text', '')) > 100 else result.get('text', '')
            sentiment = result.get('overall_sentiment', 'unknown')
            confidence = result.get('overall_confidence', 0)
            
            st.markdown(f"**{i+1}.** {text_preview}")
            st.markdown(f"   *Sentiment: {sentiment.title()} ({confidence:.1%} confidence)*")
    
    if st.button("📊 View Full Visualization", use_container_width=True):
        st.session_state.current_page = "Sentiment Charts"
        st.rerun()