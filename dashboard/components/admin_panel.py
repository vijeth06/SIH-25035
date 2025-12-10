"""
Admin panel component for system administration and management.
"""

import streamlit as st
import requests
import pandas as pd
from typing import Dict, List, Any, Optional
from .auth import get_headers

API_BASE_URL = "http://127.0.0.1:8000"


def render_admin_panel():
    """Render the admin panel interface."""
    st.markdown("## 👑 Admin Panel")
    
    st.markdown("""
    System administration and user management interface. 
    Only accessible to users with admin privileges.
    """)
    
    # Admin tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "👥 User Management", "📊 System Status", 
        "⚙️ Settings", "📈 Analytics"
    ])
    
    with tab1:
        render_user_management()
    
    with tab2:
        render_system_status()
    
    with tab3:
        render_system_settings()
    
    with tab4:
        render_system_analytics()


def render_user_management():
    """Render user management interface."""
    st.markdown("### 👥 User Management")
    
    # User actions
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # User list
        st.markdown("#### 📋 Current Users")
        
        # Mock user data
        users_data = [
            {"ID": 1, "Name": "John Admin", "Email": "admin@econsult.gov", "Role": "Admin", "Status": "Active", "Last Login": "2025-01-09 14:30"},
            {"ID": 2, "Name": "Jane Analyst", "Email": "jane@econsult.gov", "Role": "Staff", "Status": "Active", "Last Login": "2025-01-09 13:45"},
            {"ID": 3, "Name": "Bob Guest", "Email": "bob@example.com", "Role": "Guest", "Status": "Active", "Last Login": "2025-01-08 16:20"},
            {"ID": 4, "Name": "Alice Smith", "Email": "alice@example.com", "Role": "Staff", "Status": "Inactive", "Last Login": "2025-01-07 09:15"},
        ]
        
        df_users = pd.DataFrame(users_data)
        
        # Display user table with actions
        for _, user in df_users.iterrows():
            with st.expander(f"👤 {user['Name']} ({user['Role']})"):
                col_info, col_actions = st.columns([2, 1])
                
                with col_info:
                    st.write(f"**Email:** {user['Email']}")
                    st.write(f"**Status:** {user['Status']}")
                    st.write(f"**Last Login:** {user['Last Login']}")
                
                with col_actions:
                    if user['Status'] == 'Active':
                        if st.button(f"🚫 Deactivate", key=f"deact_{user['ID']}"):
                            st.warning(f"Deactivated user {user['Name']}")
                    else:
                        if st.button(f"✅ Activate", key=f"act_{user['ID']}"):
                            st.success(f"Activated user {user['Name']}")
                    
                    if st.button(f"✏️ Edit Role", key=f"edit_{user['ID']}"):
                        st.info(f"Edit role for {user['Name']}")
    
    with col2:
        render_user_actions()


def render_user_actions():
    """Render user management actions."""
    st.markdown("#### ⚡ Quick Actions")
    
    with st.expander("➕ Add New User", expanded=False):
        with st.form("add_user_form"):
            full_name = st.text_input("Full Name")
            email = st.text_input("Email Address")
            role = st.selectbox("Role", ["guest", "staff", "admin"])
            
            if st.form_submit_button("👤 Create User"):
                if full_name and email:
                    st.success(f"✅ User {full_name} created successfully!")
                else:
                    st.error("⚠️ Please fill all required fields.")
    
    with st.expander("📊 User Statistics"):
        st.metric("Total Users", "4")
        st.metric("Active Users", "3")
        st.metric("Admin Users", "1")
        st.metric("Staff Users", "2")
        st.metric("Guest Users", "1")
    
    with st.expander("🔍 User Search"):
        search_term = st.text_input("Search users...")
        search_by = st.selectbox("Search by", ["Name", "Email", "Role"])
        
        if st.button("🔍 Search"):
            st.info(f"Searching for '{search_term}' by {search_by}")


def render_system_status():
    """Render system status monitoring."""
    st.markdown("### 📊 System Status")
    
    # Overall system health
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("System Status", "🟢 Healthy")
    
    with col2:
        st.metric("API Response", "250ms")
    
    with col3:
        st.metric("Active Sessions", "15")
    
    with col4:
        st.metric("Uptime", "99.9%")
    
    # Service status
    st.markdown("#### 🔧 Service Status")
    
    services = [
        {"Service": "Authentication API", "Status": "🟢 Running", "Response Time": "45ms", "Last Check": "2025-01-09 14:30"},
        {"Service": "Sentiment Analysis", "Status": "🟢 Running", "Response Time": "120ms", "Last Check": "2025-01-09 14:30"},
        {"Service": "Text Summarization", "Status": "🟡 Warning", "Response Time": "2.1s", "Last Check": "2025-01-09 14:29"},
        {"Service": "File Processing", "Status": "🟢 Running", "Response Time": "300ms", "Last Check": "2025-01-09 14:30"},
        {"Service": "Database", "Status": "🟢 Running", "Response Time": "15ms", "Last Check": "2025-01-09 14:30"},
    ]
    
    df_services = pd.DataFrame(services)
    st.dataframe(df_services, use_container_width=True, hide_index=True)
    
    # System resources
    st.markdown("#### 💻 System Resources")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CPU usage (mock data)
        cpu_usage = 65
        st.metric("CPU Usage", f"{cpu_usage}%")
        st.progress(cpu_usage / 100)
        
        # Memory usage
        memory_usage = 78
        st.metric("Memory Usage", f"{memory_usage}%")
        st.progress(memory_usage / 100)
    
    with col2:
        # Disk usage
        disk_usage = 45
        st.metric("Disk Usage", f"{disk_usage}%")
        st.progress(disk_usage / 100)
        
        # Network
        st.metric("Network I/O", "125 MB/s")


def render_system_settings():
    """Render system settings configuration."""
    st.markdown("### ⚙️ System Settings")
    
    # Configuration tabs
    setting_tab1, setting_tab2, setting_tab3 = st.tabs([
        "🔧 General", "🛡️ Security", "📊 Analytics"
    ])
    
    with setting_tab1:
        render_general_settings()
    
    with setting_tab2:
        render_security_settings()
    
    with setting_tab3:
        render_analytics_settings()


def render_general_settings():
    """Render general system settings."""
    st.markdown("#### 🔧 General Configuration")
    
    with st.form("general_settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            system_name = st.text_input("System Name", value="E-Consultation Insight Engine")
            max_upload_size = st.number_input("Max Upload Size (MB)", value=50, min_value=1, max_value=500)
            session_timeout = st.number_input("Session Timeout (minutes)", value=30, min_value=5, max_value=480)
        
        with col2:
            default_language = st.selectbox("Default Language", ["English", "Hindi"], index=0)
            enable_debug = st.checkbox("Enable Debug Mode", value=False)
            auto_backup = st.checkbox("Enable Auto Backup", value=True)
        
        if st.form_submit_button("💾 Save General Settings"):
            st.success("✅ Settings saved successfully!")


def render_security_settings():
    """Render security settings."""
    st.markdown("#### 🛡️ Security Configuration")
    
    with st.form("security_settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            token_expiry = st.number_input("Token Expiry (minutes)", value=30, min_value=5, max_value=1440)
            max_login_attempts = st.number_input("Max Login Attempts", value=5, min_value=1, max_value=20)
            require_2fa = st.checkbox("Require 2FA", value=False)
        
        with col2:
            password_min_length = st.number_input("Min Password Length", value=8, min_value=6, max_value=32)
            enable_audit_log = st.checkbox("Enable Audit Logging", value=True)
            ip_whitelist_enabled = st.checkbox("Enable IP Whitelist", value=False)
        
        if st.form_submit_button("🔒 Save Security Settings"):
            st.success("✅ Security settings saved successfully!")


def render_analytics_settings():
    """Render analytics settings."""
    st.markdown("#### 📊 Analytics Configuration")
    
    with st.form("analytics_settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            enable_analytics = st.checkbox("Enable Analytics", value=True)
            data_retention_days = st.number_input("Data Retention (days)", value=90, min_value=7, max_value=365)
            enable_performance_tracking = st.checkbox("Performance Tracking", value=True)
        
        with col2:
            anonymize_data = st.checkbox("Anonymize User Data", value=True)
            export_frequency = st.selectbox("Export Frequency", ["Daily", "Weekly", "Monthly"], index=1)
            enable_alerts = st.checkbox("Enable Alerts", value=True)
        
        if st.form_submit_button("📈 Save Analytics Settings"):
            st.success("✅ Analytics settings saved successfully!")


def render_system_analytics():
    """Render system analytics and usage statistics."""
    st.markdown("### 📈 System Analytics")
    
    # Usage metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Daily Active Users", "24", delta="↑ 12%")
    
    with col2:
        st.metric("Files Processed", "156", delta="↑ 8%")
    
    with col3:
        st.metric("Analyses Run", "1,245", delta="↑ 15%")
    
    with col4:
        st.metric("API Calls", "8,924", delta="↑ 22%")
    
    # Usage charts
    st.markdown("#### 📊 Usage Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Mock daily usage data
        import plotly.express as px
        
        daily_usage = pd.DataFrame({
            'Date': pd.date_range('2025-01-01', periods=9, freq='D'),
            'Users': [12, 15, 18, 22, 19, 25, 28, 24, 26],
            'Analyses': [45, 52, 68, 73, 61, 89, 95, 82, 91]
        })
        
        fig = px.line(daily_usage, x='Date', y='Users', title="Daily Active Users")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Analysis types distribution
        analysis_types = pd.DataFrame({
            'Type': ['Sentiment Analysis', 'Text Summarization', 'Language Detection', 'Key Phrase Extraction'],
            'Count': [450, 280, 150, 120]
        })
        
        fig = px.pie(analysis_types, values='Count', names='Type', title="Analysis Types Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics
    st.markdown("#### ⚡ Performance Metrics")
    
    performance_data = [
        {"Metric": "Average Response Time", "Value": "245ms", "Target": "< 500ms", "Status": "✅ Good"},
        {"Metric": "API Success Rate", "Value": "99.2%", "Target": "> 99%", "Status": "✅ Good"},
        {"Metric": "Error Rate", "Value": "0.8%", "Target": "< 1%", "Status": "✅ Good"},
        {"Metric": "Throughput", "Value": "150 req/min", "Target": "> 100 req/min", "Status": "✅ Good"},
    ]
    
    df_performance = pd.DataFrame(performance_data)
    st.dataframe(df_performance, use_container_width=True, hide_index=True)
    
    # Export options
    st.markdown("#### 📤 Export & Reports")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Usage Report", use_container_width=True):
            st.success("Usage report exported successfully!")
    
    with col2:
        if st.button("📈 Export Performance Report", use_container_width=True):
            st.success("Performance report exported successfully!")
    
    with col3:
        if st.button("👥 Export User Activity", use_container_width=True):
            st.success("User activity report exported successfully!")