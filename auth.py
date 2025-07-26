# auth.py - Authentication functions

import hashlib
import streamlit as st
from database import db


class Auth:
    """Authentication handler"""

    @staticmethod
    def hash_password(password):
        """Hash password using SHA256"""
        return hashlib.sha256(password.encode()).hexdigest()

    @staticmethod
    def authenticate(username, password):
        """Authenticate user"""
        password_hash = Auth.hash_password(password)
        return db.get_user(username, password_hash)

    @staticmethod
    def register(username, password, user_type, coach_id=None):
        """Register new user"""
        password_hash = Auth.hash_password(password)
        return db.create_user(username, password_hash, user_type, coach_id)

    @staticmethod
    def login_user(username, user_data):
        """Set session state for logged in user"""
        st.session_state.logged_in = True
        st.session_state.username = username
        st.session_state.user_id = user_data[0]
        st.session_state.user_type = user_data[1]
        st.session_state.coach_id = user_data[2]

    @staticmethod
    def logout_user():
        """Clear session state"""
        keys_to_delete = ['logged_in', 'username', 'user_id', 'user_type', 'coach_id']
        for key in keys_to_delete:
            if key in st.session_state:
                del st.session_state[key]

    @staticmethod
    def is_authenticated():
        """Check if user is authenticated"""
        return st.session_state.get('logged_in', False)

    @staticmethod
    def require_auth(func):
        """Decorator to require authentication"""

        def wrapper(*args, **kwargs):
            if not Auth.is_authenticated():
                st.error("Please login to access this page")
                st.stop()
            return func(*args, **kwargs)

        return wrapper

    @staticmethod
    def require_role(allowed_roles):
        """Decorator to require specific user roles"""

        def decorator(func):
            def wrapper(*args, **kwargs):
                if not Auth.is_authenticated():
                    st.error("Please login to access this page")
                    st.stop()
                if st.session_state.user_type not in allowed_roles:
                    st.error("You don't have permission to access this page")
                    st.stop()
                return func(*args, **kwargs)

            return wrapper

        return decorator


# Create default admin user if not exists
try:
    Auth.register('admin', 'admin123', 'admin')
except:
    pass  # Admin already exists