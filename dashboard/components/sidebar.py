"""
Sidebar component for the Streamlit dashboard.
"""

import streamlit as st
from typing import List, Dict, Any
from .auth import check_authentication, logout

def render_sidebar() -> str:
    """
    Render the sidebar navigation and return selected page.
    
    Returns:
        str: Selected page name
    """
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        
        # User info display
        if st.session_state.user_info:
            user_role = st.session_state.user_info.get('role', 'guest')
            st.markdown(f"""
            <div style="background: #f0f9ff; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                <strong>👤 {st.session_state.user_info.get('full_name', 'User')}</strong><br>
                <small style="color: #64748b;">Role: {user_role.title()}</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Main navigation
        pages = get_navigation_pages()
        
        selected_page = st.radio(
            "Select Page",
            options=list(pages.keys()),
            index=0,
            format_func=lambda x: pages[x]["display_name"],
            key="sidebar_navigation"
        )
        
        st.markdown("---")
        
        # Quick stats if data is available
        render_quick_stats()
        
        st.markdown("---")
        
        # Settings and logout
        render_sidebar_actions()
        
        return selected_page


def get_navigation_pages() -> Dict[str, Dict]:
    """
    Get navigation pages based on user role.
    
    Returns:
        dict: Dictionary of available pages
    """
    base_pages = {
        "Dashboard": {
            "display_name": "📊 Dashboard",
            "description": "Overview and quick stats"
        },
        "File Upload": {
            "display_name": "📁 File Upload",
            "description": "Upload consultation data"
        },
        "Analysis": {
            "display_name": "🔍 Analysis",
            "description": "Sentiment analysis tools"
        },
        "Text Analytics": {
            "display_name": "📝 Text Analytics",
            "description": "Text processing and insights"
        },
        "Sentiment Charts": {
            "display_name": "📊 Sentiment Charts",
            "description": "Visualization and charts"
        }
    }
    
    # Add admin-only pages
    if st.session_state.user_info and st.session_state.user_info.get('role') == 'admin':
        base_pages["Admin Panel"] = {
            "display_name": "👑 Admin Panel",
            "description": "User and system management"
        }
    
    # Add settings page
    base_pages["Settings"] = {
        "display_name": "⚙️ Settings",
        "description": "User preferences"
    }
    
    return base_pages


def render_quick_stats():
    """Render quick statistics in the sidebar."""
    st.markdown("#### 📈 Quick Stats")
    
    # Check if we have analysis results
    if st.session_state.get('analysis_results'):
        results = st.session_state.analysis_results
        
        # Calculate basic stats
        total_comments = len(results) if isinstance(results, list) else 0
        
        if total_comments > 0:
            st.metric("Total Comments", total_comments)
            
            # Calculate sentiment distribution if available
            try:
                positive = sum(1 for r in results if r.get('sentiment') == 'positive')
                negative = sum(1 for r in results if r.get('sentiment') == 'negative')
                neutral = sum(1 for r in results if r.get('sentiment') == 'neutral')
                
                st.metric("😊 Positive", f"{positive} ({positive/total_comments:.1%})")
                st.metric("😔 Negative", f"{negative} ({negative/total_comments:.1%})")
                st.metric("😐 Neutral", f"{neutral} ({neutral/total_comments:.1%})")
            except:
                pass
    else:
        st.info("📄 No data loaded yet")
        st.markdown("""
        Upload data files to see:
        - Comment statistics
        - Sentiment distribution
        - Processing metrics
        """)


def render_sidebar_actions():
    """Render sidebar action buttons."""
    st.markdown("#### 🎯 Quick Actions")
    
    # Data refresh button
    if st.button("🔄 Refresh Data", use_container_width=True):
        # Clear cached data to force refresh
        if 'analysis_results' in st.session_state:
            del st.session_state.analysis_results
        st.rerun()
    
    # Help button
    if st.button("❓ Help & Support", use_container_width=True):
        st.session_state.show_help = True
    
    # Export button (if data available)
    if st.session_state.get('analysis_results'):
        if st.button("📤 Export Results", use_container_width=True):
            st.session_state.show_export = True
    
    st.markdown("---")
    
    # System status
    render_system_status()


def render_system_status():
    """Render system status indicators."""
    st.markdown("#### 🏥 System Status")
    
    # API connection status
    try:
        import requests
        response = requests.get("http://127.0.0.1:8000/api/v1/health", timeout=5)
        if response.status_code == 200:
            st.success("🟢 API Online")
        else:
            st.warning("🟡 API Issues")
    except:
        st.error("🔴 API Offline")
    
    # Service status (mockup for now)
    services = {
        "🔍 Sentiment Analysis": True,
        "📝 Summarization": True,
        "📁 File Processing": True,
        "🔐 Authentication": True
    }
    
    for service, status in services.items():
        if status:
            st.markdown(f"✅ {service}")
        else:
            st.markdown(f"❌ {service}")


def show_help_modal():
    """Show help and support information."""
    if st.session_state.get('show_help'):
        with st.expander("❓ Help & Support", expanded=True):
            st.markdown("""
            ### 📖 User Guide
            
            #### Getting Started:
            1. **Upload Data**: Use the File Upload page to upload CSV, Excel, or text files
            2. **Run Analysis**: Navigate to Analysis to process your data
            3. **View Results**: Check Sentiment Charts for visualizations
            4. **Export Data**: Use Export options to download results
            
            #### File Formats Supported:
            - **CSV**: Comma-separated values with headers
            - **Excel**: .xlsx and .xls formats
            - **Text**: Plain text files (.txt)
            - **JSON**: Structured JSON data
            
            #### Analysis Features:
            - **Sentiment Analysis**: VADER, TextBlob, Ensemble methods
            - **Emotion Detection**: Support, concern, suggestion, anger, etc.
            - **Summarization**: Extractive and abstractive summaries
            - **Language Support**: English and Hindi
            
            #### Troubleshooting:
            - **Upload Issues**: Check file format and size (max 50MB)
            - **Analysis Errors**: Verify text data is properly formatted
            - **Login Problems**: Contact system administrator
            
            #### Contact Support:
            - **Email**: support@econsultation.gov
            - **Phone**: +1-800-CONSULT
            - **Hours**: Monday-Friday, 9 AM - 5 PM
            """)
            
            if st.button("Close Help"):
                st.session_state.show_help = False
                st.rerun()


def show_export_modal():
    """Show export options modal."""
    if st.session_state.get('show_export'):
        with st.expander("📤 Export Results", expanded=True):
            st.markdown("### Export Analysis Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                export_format = st.selectbox(
                    "Export Format",
                    ["CSV", "Excel", "JSON", "PDF Report"]
                )
                
                include_options = st.multiselect(
                    "Include Data",
                    ["Original Comments", "Sentiment Scores", "Emotions", "Summaries", "Statistics"],
                    default=["Original Comments", "Sentiment Scores"]
                )
            
            with col2:
                filename = st.text_input("Filename", value="analysis_results")
                
                date_range = st.checkbox("Filter by Date Range")
                if date_range:
                    start_date = st.date_input("Start Date")
                    end_date = st.date_input("End Date")
            
            col_export, col_cancel = st.columns(2)
            
            with col_export:
                if st.button("📥 Download", use_container_width=True):
                    st.success("Export started! Download will begin shortly.")
                    # Here you would implement the actual export functionality
                    st.session_state.show_export = False
                    st.rerun()
            
            with col_cancel:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.show_export = False
                    st.rerun()