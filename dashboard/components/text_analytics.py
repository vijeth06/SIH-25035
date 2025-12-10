"""
Text analytics component for advanced text processing and visualization.
"""

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from typing import Dict, List, Any, Optional
# from dashboard.components.auth import get_headers
from .auth import get_headers

API_BASE_URL = "http://127.0.0.1:8000"

def perform_concise_comments_summary(comments: List[str], max_length: int = 120) -> Optional[str]:
    """Call backend concise-comments endpoint to get a 1–2 sentence summary."""
    try:
        endpoint = f"{API_BASE_URL}/api/v1/summarization/concise-comments"
        payload = {"comments": comments, "max_length": max_length}
        resp = requests.post(endpoint, json=payload, headers=get_headers(), timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("summary_concise")
        else:
            st.warning(f"Concise summary failed: {resp.status_code} - {resp.text}")
            return None
    except Exception as e:
        st.error(f"Error calling concise summary: {e}")
        return None


def render_text_analytics():
    """Render text analytics and processing interface."""
    st.markdown("## 📝 Text Analytics & Insights")
    
    st.markdown("""
    Advanced text processing, summarization, and linguistic analysis tools 
    for deeper insights into consultation data.
    """)
    
    # Check if backend is accessible
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/health", timeout=5)
        if response.status_code != 200:
            st.warning("⚠️ Backend API is not accessible. Some features may not work.")
    except Exception as e:
        st.warning("⚠️ Backend API is not accessible. Some features may not work.")
    
    # Create tabs for different analytics
    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 Text Summarization", "🔤 Language Analysis", 
        "🎯 Key Phrase Extraction", "📊 Text Statistics"
    ])
    
    with tab1:
        render_summarization_interface()
    
    with tab2:
        render_language_analysis()
    
    with tab3:
        render_key_phrase_extraction()
    
    with tab4:
        render_text_statistics()


def render_summarization_interface():
    """Render text summarization interface."""
    st.markdown("### 📄 Text Summarization")
    
    # Summarization options
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Input source selection
        input_source = st.radio(
            "Select Input Source:",
            ["Single Text", "Multiple Comments", "Analysis Results"],
            horizontal=True
        )
        
        if input_source == "Single Text":
            text_input = st.text_area(
                "Enter text to summarize:",
                height=200,
                placeholder="Paste your text here for summarization...",
                help="Enter the text you want to summarize"
            )
            
            if text_input.strip():
                if st.button("📝 Summarize Text", type="primary", use_container_width=True):
                    perform_text_summarization(text_input, "single")
        
        elif input_source == "Multiple Comments":
            comments_input = st.text_area(
                "Enter multiple comments (one per line):",
                height=200,
                placeholder="Comment 1\nComment 2\nComment 3\n...",
                help="Enter each comment on a separate line"
            )
            
            if comments_input.strip():
                comments = [line.strip() for line in comments_input.split('\n') if line.strip()]
                st.info(f"📝 Found {len(comments)} comments")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("📋 Summarize All Comments", type="primary", use_container_width=True):
                        perform_batch_summarization(comments)
                with col_b:
                    if st.button("✨ Concise Overall Summary (1–2 lines)", use_container_width=True):
                        concise = perform_concise_comments_summary(comments, max_length=120)
                        if concise:
                            st.markdown("#### 🧾 Concise Summary")
                            st.success(concise)
        
        else:  # Analysis Results
            if st.session_state.get('analysis_results'):
                results_count = len(st.session_state.analysis_results)
                st.info(f"📊 {results_count} analysis results available for summarization")
                
                if st.button("📈 Summarize Analysis Results", type="primary", use_container_width=True):
                    perform_results_summarization()
            else:
                st.warning("No analysis results found. Please run analysis first.")
    
    with col2:
        render_summarization_options()


def render_summarization_options():
    """Render summarization configuration options."""
    st.markdown("#### ⚙️ Summarization Options")
    
    with st.expander("🔧 Configuration", expanded=True):
        summarization_type = st.selectbox(
            "Summarization Type:",
            ["Extractive", "Abstractive", "Hybrid"],
            help="Choose between extractive (selecting sentences) or abstractive (generating new text)"
        )
        
        if summarization_type in ["Extractive", "Hybrid"]:
            num_sentences = st.slider(
                "Number of Sentences:",
                1, 10, 3,
                help="Number of sentences in the summary"
            )

        if summarization_type in ["Abstractive", "Hybrid"]:
            max_length = st.slider(
                "Maximum Length (words):",
                30, 200, 100,
                help="Maximum length of generated summary"
            )

            # Topic-based summarization option
            enable_topic_focus = st.checkbox(
                "🎯 Enable Topic-Based Summarization",
                value=False,
                help="Focus summary on specific topics (uses multilingual models)"
            )

            if enable_topic_focus:
                topics_input = st.text_area(
                    "Topics to focus on (one per line):",
                    height=100,
                    placeholder="government policy\nenvironmental impact\npublic consultation\n...",
                    help="Enter topics to guide the summarization"
                )
                topics = [line.strip() for line in topics_input.split('\n') if line.strip()] if topics_input else []
        
        # Update method options based on summarization type
        if summarization_type == "Abstractive" and enable_topic_focus:
            method_options = ["mT5 (Multilingual)", "IndicBART (Indian Languages)", "T5 (English)"]
        elif summarization_type == "Abstractive":
            method_options = ["T5 (English)", "mT5 (Multilingual)", "IndicBART (Indian Languages)"]
        else:
            method_options = ["TextRank", "LSA", "Luhn"]

        method = st.selectbox(
            "Summarization Method:",
            method_options,
            help="Algorithm to use for summarization"
        )
        
        include_keywords = st.checkbox(
            "Preserve Key Terms",
            value=True,
            help="Ensure important keywords are preserved in summary"
        )
    
    with st.expander("📊 Output Options"):
        show_confidence = st.checkbox(
            "Show Confidence Scores",
            value=True
        )
        
        highlight_sentences = st.checkbox(
            "Highlight Key Sentences",
            value=True
        )
        
        export_format = st.selectbox(
            "Export Format:",
            ["Text", "HTML", "Markdown", "JSON"]
        )


def render_language_analysis():
    """Render language analysis interface."""
    st.markdown("### 🔤 Language Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🌍 Language Detection")
        
        text_input = st.text_area(
            "Enter text for language analysis:",
            height=150,
            placeholder="Enter text to detect language and analyze linguistic features..."
        )
        
        if text_input.strip():
            if st.button("🔍 Analyze Language", use_container_width=True):
                perform_language_analysis(text_input)
    
    with col2:
        st.markdown("#### 📊 Language Statistics")
        
        if st.session_state.get('analysis_results'):
            display_language_statistics()
        else:
            st.info("Run analysis to see language statistics")
    
    # Multilingual support information
    st.markdown("#### 🌐 Multilingual Support")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Supported Languages:**
        - 🇬🇧 English (Full support)
        - 🇮🇳 Hindi (Basic support)
        - 🔄 Auto-detection available
        """)
    
    with col2:
        st.markdown("""
        **Language Features:**
        - Language identification
        - Script detection
        - Confidence scoring
        - Text normalization
        """)


def render_key_phrase_extraction():
    """Render key phrase extraction interface."""
    st.markdown("### 🎯 Key Phrase Extraction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Input selection
        input_type = st.radio(
            "Input Type:",
            ["Manual Text", "Analysis Results", "Uploaded Data"],
            horizontal=True
        )
        
        if input_type == "Manual Text":
            text_input = st.text_area(
                "Enter text for key phrase extraction:",
                height=200,
                placeholder="Enter text to extract key phrases and important terms..."
            )
            
            if text_input.strip():
                if st.button("🎯 Extract Key Phrases", type="primary", use_container_width=True):
                    extract_key_phrases(text_input)
        
        elif input_type == "Analysis Results":
            if st.session_state.get('analysis_results'):
                results_count = len(st.session_state.analysis_results)
                st.info(f"📊 Extracting from {results_count} analysis results")
                
                if st.button("🎯 Extract from Results", type="primary", use_container_width=True):
                    extract_phrases_from_results()
            else:
                st.warning("No analysis results available")
        
        else:  # Uploaded Data
            st.info("Key phrase extraction from uploaded files would be implemented here")
    
    with col2:
        render_key_phrase_options()


def render_key_phrase_options():
    """Render key phrase extraction options."""
    st.markdown("#### ⚙️ Extraction Options")
    
    with st.expander("🔧 Settings", expanded=True):
        max_phrases = st.slider(
            "Max Phrases:",
            5, 50, 20,
            help="Maximum number of key phrases to extract"
        )
        
        min_phrase_length = st.slider(
            "Min Phrase Length:",
            1, 5, 2,
            help="Minimum number of words in a phrase"
        )
        
        include_entities = st.checkbox(
            "Include Named Entities",
            value=True,
            help="Include people, organizations, locations"
        )
        
        phrase_types = st.multiselect(
            "Phrase Types:",
            ["Noun Phrases", "Technical Terms", "Legal Terms", "Action Items"],
            default=["Noun Phrases", "Technical Terms"]
        )


def render_text_statistics():
    """Render text statistics and metrics."""
    st.markdown("### 📊 Text Statistics & Metrics")
    
    # Data source selection
    data_source = st.selectbox(
        "Data Source:",
        ["Analysis Results", "Manual Input", "Uploaded Files"]
    )
    
    if data_source == "Analysis Results":
        if st.session_state.get('analysis_results'):
            display_comprehensive_text_stats()
        else:
            st.warning("No analysis results available. Please run analysis first.")
    
    elif data_source == "Manual Input":
        text_input = st.text_area(
            "Enter text for statistical analysis:",
            height=200,
            placeholder="Enter text to analyze its statistical properties..."
        )
        
        if text_input.strip():
            if st.button("📊 Analyze Text Statistics"):
                display_manual_text_stats(text_input)
    
    else:  # Uploaded Files
        st.info("File-based text statistics would be implemented here")


def perform_text_summarization(text: str, mode: str):
    """Perform text summarization via API."""
    with st.spinner("📝 Generating summary..."):
        try:
            # Get the current summarization settings from session state
            summarization_type = st.session_state.get('summarization_type', 'Extractive')
            method = st.session_state.get('summarization_method', 'TextRank')
            max_length = st.session_state.get('max_length', 100)
            num_sentences = st.session_state.get('num_sentences', 3)
            topics = st.session_state.get('topics', [])

            # Map method names to API values
            method_mapping = {
                "TextRank": "custom_textrank",
                "LSA": "lsa",
                "Luhn": "luhn",
                "T5 (English)": "t5",
                "mT5 (Multilingual)": "mt5",
                "IndicBART (Indian Languages)": "indicbart"
            }

            api_method = method_mapping.get(method, "custom_textrank")

            # Prepare payload based on summarization type
            if summarization_type == "Abstractive" and topics:
                # Use topic-based summarization endpoint
                payload = {
                    "text": text,
                    "topics": topics,
                    "max_length": max_length,
                    "min_length": 30
                }

                endpoint = f"{API_BASE_URL}/api/v1/summarization/topic_based"
            elif summarization_type == "Abstractive":
                # Use abstractive summarization
                payload = {
                    "text": text,
                    "method": api_method,
                    "max_length": max_length,
                    "min_length": 30
                }

                endpoint = f"{API_BASE_URL}/api/v1/summarize-text"
            else:
                # Use extractive summarization
                payload = {
                    "text": text,
                    "method": api_method,
                    "summary_type": "extractive",
                    "num_sentences": num_sentences
                }

                endpoint = f"{API_BASE_URL}/api/v1/summarize-text"

            response = requests.post(
                endpoint,
                json=payload,
                headers=get_headers(),
                timeout=60  # Longer timeout for transformer models
            )

            if response.status_code == 200:
                result = response.json()
                display_summarization_results(result)
            else:
                st.error(f"❌ Summarization failed: {response.status_code} - {response.json().get('detail', 'Unknown error')}")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


def display_summarization_results(result: Dict):
    """Display summarization results."""
    st.markdown("#### 📄 Summary Results")
    
    # Main summary
    summary_text = result.get('summary_text', '')
    
    if summary_text:
        st.markdown("**Generated Summary:**")
        st.info(summary_text)
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            original_length = result.get('original_length', 0)
            st.metric("Original Length", f"{original_length:,} chars")
        
        with col2:
            summary_length = result.get('summary_length', 0)
            st.metric("Summary Length", f"{summary_length:,} chars")
        
        with col3:
            compression_ratio = result.get('compression_ratio', 0)
            st.metric("Compression Ratio", f"{compression_ratio:.1%}")
        
        # Key sentences
        key_sentences = result.get('key_sentences', [])
        if key_sentences:
            with st.expander("🔑 Key Sentences"):
                for i, sentence in enumerate(key_sentences, 1):
                    st.markdown(f"**{i}.** {sentence}")
    else:
        st.warning("No summary generated.")


def perform_language_analysis(text: str):
    """Perform language analysis on text."""
    with st.spinner("🔍 Analyzing language..."):
        try:
            # Mock language analysis results
            detected_language = "English"
            confidence = 0.95
            script = "Latin"
            
            # Display results
            st.markdown("#### 🌍 Language Analysis Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Detected Language", detected_language)
            
            with col2:
                st.metric("Confidence", f"{confidence:.1%}")
            
            with col3:
                st.metric("Script", script)
            
            # Text characteristics
            st.markdown("#### 📊 Text Characteristics")
            
            char_count = len(text)
            word_count = len(text.split())
            sentence_count = len(text.split('.'))
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Characters", f"{char_count:,}")
            
            with col2:
                st.metric("Words", f"{word_count:,}")
            
            with col3:
                st.metric("Sentences", f"{sentence_count:,}")
        
        except Exception as e:
            st.error(f"❌ Language analysis error: {str(e)}")


def display_language_statistics():
    """Display language statistics from analysis results."""
    results = st.session_state.analysis_results
    
    # Extract language information (mock for now)
    languages = ["English"] * len(results)  # In real implementation, extract from results
    
    language_counts = Counter(languages)
    
    st.markdown("**Language Distribution:**")
    for lang, count in language_counts.most_common():
        percentage = count / len(results) * 100
        st.write(f"• {lang}: {count} texts ({percentage:.1f}%)")


def extract_key_phrases(text: str):
    """Extract key phrases from text."""
    with st.spinner("🎯 Extracting key phrases..."):
        try:
            # Mock key phrase extraction
            key_phrases = [
                {"text": "data protection regulation", "score": 0.95, "type": "Legal Term"},
                {"text": "stakeholder consultation", "score": 0.88, "type": "Technical Term"},
                {"text": "implementation timeline", "score": 0.82, "type": "Noun Phrase"},
                {"text": "compliance requirements", "score": 0.78, "type": "Legal Term"},
                {"text": "privacy rights", "score": 0.75, "type": "Technical Term"}
            ]
            
            display_key_phrases_results(key_phrases)
        
        except Exception as e:
            st.error(f"❌ Key phrase extraction error: {str(e)}")


def display_key_phrases_results(phrases: List[Dict]):
    """Display key phrase extraction results."""
    st.markdown("#### 🎯 Extracted Key Phrases")
    
    if not phrases:
        st.warning("No key phrases found.")
        return
    
    # Create DataFrame for display
    df = pd.DataFrame(phrases)
    
    # Display as table
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Display as chart
    col1, col2 = st.columns(2)
    
    with col1:
        # Phrase scores
        fig_scores = px.bar(
            df, x='score', y='text',
            orientation='h',
            title="Key Phrase Relevance Scores",
            color='score',
            color_continuous_scale='viridis'
        )
        fig_scores.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_scores, use_container_width=True)
    
    with col2:
        # Phrase types
        type_counts = df['type'].value_counts()
        
        fig_types = px.pie(
            values=type_counts.values,
            names=type_counts.index,
            title="Phrase Types Distribution"
        )
        st.plotly_chart(fig_types, use_container_width=True)


def display_comprehensive_text_stats():
    """Display comprehensive text statistics from analysis results."""
    results = st.session_state.analysis_results
    
    st.markdown("#### 📊 Comprehensive Text Statistics")
    
    # Extract all text
    all_text = ' '.join([result.get('text', '') for result in results])
    
    # Calculate statistics
    total_chars = len(all_text)
    total_words = len(all_text.split())
    unique_words = len(set(all_text.lower().split()))
    avg_words_per_comment = total_words / len(results) if results else 0
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Characters", f"{total_chars:,}")
    
    with col2:
        st.metric("Total Words", f"{total_words:,}")
    
    with col3:
        st.metric("Unique Words", f"{unique_words:,}")
    
    with col4:
        st.metric("Avg Words/Comment", f"{avg_words_per_comment:.1f}")
    
    # Word frequency analysis
    st.markdown("#### 📈 Word Frequency Analysis")
    
    # Get most common words
    words = all_text.lower().split()
    # Filter out common stop words
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
    
    word_freq = Counter(filtered_words)
    top_words = word_freq.most_common(20)
    
    if top_words:
        # Create chart
        words_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
        
        fig = px.bar(
            words_df, x='Frequency', y='Word',
            orientation='h',
            title="Top 20 Most Frequent Words"
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Text length distribution
    st.markdown("#### 📏 Text Length Distribution")
    
    text_lengths = [len(result.get('text', '').split()) for result in results]
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_hist = px.histogram(
            x=text_lengths,
            title="Distribution of Comment Lengths (words)",
            nbins=20
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Statistics
        import numpy as np
        
        st.metric("Min Length", f"{min(text_lengths)} words")
        st.metric("Max Length", f"{max(text_lengths)} words")
        st.metric("Median Length", f"{int(np.median(text_lengths))} words")
        st.metric("Std Deviation", f"{np.std(text_lengths):.1f}")


def display_manual_text_stats(text: str):
    """Display statistics for manually entered text."""
    st.markdown("#### 📊 Text Analysis Results")
    
    # Basic statistics
    char_count = len(text)
    word_count = len(text.split())
    sentence_count = len([s for s in text.split('.') if s.strip()])
    paragraph_count = len([p for p in text.split('\n\n') if p.strip()])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Characters", f"{char_count:,}")
    
    with col2:
        st.metric("Words", f"{word_count:,}")
    
    with col3:
        st.metric("Sentences", f"{sentence_count:,}")
    
    with col4:
        st.metric("Paragraphs", f"{paragraph_count:,}")
    
    # Reading metrics
    st.markdown("#### 📖 Readability Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        st.metric("Avg Sentence Length", f"{avg_sentence_length:.1f} words")
    
    with col2:
        avg_word_length = sum(len(word) for word in text.split()) / word_count if word_count > 0 else 0
        st.metric("Avg Word Length", f"{avg_word_length:.1f} chars")
    
    with col3:
        # Estimated reading time (assuming 200 words per minute)
        reading_time = word_count / 200
        st.metric("Reading Time", f"{reading_time:.1f} min")


# Helper functions for batch operations

def perform_batch_summarization(comments: List[str]):
    """Perform batch summarization on multiple comments."""
    with st.spinner(f"📝 Summarizing {len(comments)} comments..."):
        try:
            payload = {
                "comments": comments,
                "method": "custom_textrank",
                "summary_type": "extractive"
            }
            
            response = requests.post(
                f"{API_BASE_URL}/api/v1/summarization/comments",
                json=payload,
                headers=get_headers(),
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                display_summarization_results(result)
            else:
                st.error(f"❌ Batch summarization failed: {response.status_code} - {response.json().get('detail', 'Unknown error')}")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


def perform_results_summarization():
    """Perform summarization on analysis results."""
    results = st.session_state.analysis_results
    
    # Extract texts from results
    texts = [result.get('text', '') for result in results if result.get('text')]
    
    if texts:
        perform_batch_summarization(texts)
    else:
        st.warning("No text content found in analysis results.")

def perform_enhanced_column_summarization(df, column_name):
    """Perform enhanced column summarization with local fallback."""
    try:
        # Try to use enhanced summarization if available
        import sys
        import os
        parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sys.path.append(parent_dir)
        
        from enhanced_text_summarization import summarize_column_data
        
        with st.spinner("📊 Generating comprehensive column summary..."):
            summary_result = summarize_column_data(df, column_name, max_sentences=5)
            
            if 'error' in summary_result:
                st.error(f"❌ {summary_result['error']}")
                return
            
            # Display comprehensive summary
            st.markdown("#### 📄 Column Analysis Summary")
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Entries", summary_result['statistics']['total_entries'])
            with col2:
                st.metric("Non-empty Entries", summary_result['statistics']['non_empty_entries'])
            with col3:
                st.metric("Avg Length", f"{summary_result['statistics']['avg_length']:.1f}")
            with col4:
                st.metric("Total Words", summary_result['statistics']['total_words'])
            
            # Main summary
            st.markdown("##### 📝 Main Summary")
            st.info(summary_result['main_summary'])
            
            # Sentiment distribution
            if summary_result['sentiment_distribution']:
                st.markdown("##### 😊 Sentiment Distribution")
                sentiment_df = pd.DataFrame(list(summary_result['sentiment_distribution'].items()), 
                                          columns=['Sentiment', 'Count'])
                st.bar_chart(sentiment_df.set_index('Sentiment'))
                
                # Show examples for each sentiment
                for sentiment, examples in summary_result['sentiment_examples'].items():
                    if examples:
                        st.markdown(f"**{sentiment.title()} Examples:**")
                        for example in examples:
                            st.caption(f"• {example}")
            
            # Key themes
            if summary_result['key_themes']:
                st.markdown("##### 🎯 Key Themes")
                themes_df = pd.DataFrame(summary_result['key_themes'])
                st.dataframe(themes_df, use_container_width=True)
            
            # Compression info
            st.markdown("##### 📈 Summary Statistics")
            comp_info = summary_result['compression_info']
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Words", comp_info['original_words'])
            with col2:
                st.metric("Summary Words", comp_info['summary_words'])
            with col3:
                st.metric("Compression Ratio", f"{comp_info['compression_ratio']:.2f}x")
                
    except Exception as e:
        st.error(f"❌ Enhanced summarization failed: {str(e)}")
        # Fallback to basic summarization
        st.warning("🔄 Falling back to basic summarization...")
        perform_basic_column_summarization(df, column_name)

def perform_basic_column_summarization(df, column_name):
    """Basic column summarization fallback."""
    if column_name not in df.columns:
        st.error(f"Column '{column_name}' not found in data.")
        return
    
    st.markdown("#### 📄 Basic Column Summary")
    
    # Basic statistics
    col_data = df[column_name].dropna()
    total_entries = len(df)
    non_empty = len(col_data)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Entries", total_entries)
    with col2:
        st.metric("Non-empty Entries", non_empty)
    with col3:
        if non_empty > 0:
            avg_length = col_data.astype(str).str.len().mean()
            st.metric("Avg Length", f"{avg_length:.1f}")
    
    # Sample entries
    if non_empty > 0:
        st.markdown("##### 📝 Sample Entries")
        sample_size = min(5, non_empty)
        sample_data = col_data.sample(n=sample_size) if non_empty >= sample_size else col_data
        for idx, entry in enumerate(sample_data, 1):
            st.write(f"{idx}. {str(entry)[:200]}{'...' if len(str(entry)) > 200 else ''}")
    
    # Basic text analysis if it's a text column
    if col_data.dtype == 'object':
        all_text = ' '.join(col_data.astype(str).tolist())
        word_count = len(all_text.split())
        
        st.markdown("##### 📊 Text Statistics")
        st.metric("Total Words", word_count)
        
        # Simple word frequency
        from collections import Counter
        words = all_text.lower().split()
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        if filtered_words:
            word_freq = Counter(filtered_words).most_common(10)
            st.markdown("**Top 10 Most Frequent Words:**")
            for word, count in word_freq:
                st.write(f"• {word}: {count} times")

def extract_phrases_from_results():
    """Extract key phrases from analysis results."""
    results = st.session_state.analysis_results
    
    # Combine all text from results
    combined_text = ' '.join([result.get('text', '') for result in results if result.get('text')])
    
    if combined_text:
        extract_key_phrases(combined_text)
    else:
        st.warning("No text content found in analysis results.")