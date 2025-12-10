"""
Authentication component for Streamlit dashboard.
"""

import streamlit as st
import requests
from typing import Tuple, Optional, Dict, Any


API_BASE_URL = "http://127.0.0.1:8002"


def authenticate_user(email: str, password: str) -> Tuple[bool, Optional[Dict], Optional[str]]:
    """
    Authenticate user with the backend API.
    
    Args:
        email: User email address
        password: User password
        
    Returns:
        tuple: (success, user_info, access_token)
    """
    try:
        # Prepare login data
        login_data = {
            "username": email,
            "password": password
        }
        
        # Make login request
        response = requests.post(
            f"{API_BASE_URL}/api/v1/auth/login",
            data=login_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        if response.status_code == 200:
            data = response.json()
            return True, data["user"], data["access_token"]
        else:
            return False, None, None
            
    except requests.RequestException as e:
        st.error(f"Connection error: {str(e)}")
        return False, None, None
    except Exception as e:
        st.error(f"Authentication error: {str(e)}")
        return False, None, None


def register_user(full_name: str, email: str, password: str, role: str = "guest") -> Tuple[bool, str]:
    """
    Register a new user with the backend API.
    
    Args:
        full_name: User's full name
        email: User email address
        password: User password
        role: User role
        
    Returns:
        tuple: (success, message)
    """
    try:
        # Prepare registration data
        user_data = {
            "full_name": full_name,
            "email": email,
            "password": password,
            "role": role
        }
        
        # Make registration request
        response = requests.post(
            f"{API_BASE_URL}/api/v1/auth/register",
            json=user_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 201:
            return True, "Account created successfully!"
        else:
            error_detail = response.json().get("detail", "Registration failed")
            return False, error_detail
            
    except requests.RequestException as e:
        return False, f"Connection error: {str(e)}"
    except Exception as e:
        return False, f"Registration error: {str(e)}"


def check_authentication() -> bool:
    """
    Check if user is currently authenticated.
    
    Returns:
        bool: True if authenticated, False otherwise
    """
    if not st.session_state.get('access_token'):
        return False
    
    try:
        # Verify token by making a request to get current user
        headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
        response = requests.get(
            f"{API_BASE_URL}/api/v1/auth/me",
            headers=headers
        )
        
        return response.status_code == 200
        
    except:
        return False


def get_headers() -> Dict[str, str]:
    """
    Get HTTP headers with authentication token.
    
    Returns:
        dict: Headers dictionary with authorization
    """
    if st.session_state.get('access_token'):
        return {
            "Authorization": f"Bearer {st.session_state.access_token}",
            "Content-Type": "application/json"
        }
    return {"Content-Type": "application/json"}


def refresh_token() -> bool:
    """
    Refresh the access token if possible.
    
    Returns:
        bool: True if token refreshed successfully
    """
    if not st.session_state.get('refresh_token'):
        return False
    
    try:
        refresh_data = {"refresh_token": st.session_state.refresh_token}
        
        response = requests.post(
            f"{API_BASE_URL}/api/v1/auth/refresh",
            json=refresh_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            st.session_state.access_token = data["access_token"]
            return True
        else:
            return False
            
    except:
        return False


def logout():
    """Logout the current user."""
    try:
        if st.session_state.get('access_token'):
            headers = get_headers()
            requests.post(f"{API_BASE_URL}/api/v1/auth/logout", headers=headers)
    except:
        pass  # Ignore errors during logout
    
    # Clear session state
    st.session_state.authenticated = False
    st.session_state.user_info = None
    st.session_state.access_token = None
    st.session_state.refresh_token = None