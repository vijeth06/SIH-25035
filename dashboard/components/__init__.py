"""
Dashboard components package.
"""

# Import all components for easy access
from .auth import authenticate_user, check_authentication, get_headers
from .sidebar import render_sidebar
from .file_upload import render_file_upload
from .analysis_dashboard import render_analysis_dashboard
from .sentiment_charts import render_sentiment_charts
from .text_analytics import render_text_analytics
from .admin_panel import render_admin_panel

__all__ = [
    'authenticate_user',
    'check_authentication', 
    'get_headers',
    'render_sidebar',
    'render_file_upload',
    'render_analysis_dashboard',
    'render_sentiment_charts',
    'render_text_analytics', 
    'render_admin_panel'
]