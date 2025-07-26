# app.py - Main Streamlit application

import streamlit as st
from config import Config
from auth import Auth
from pages import upload_analyze, view_reports, manage_users

# Page configuration
st.set_page_config(
    page_title=Config.APP_NAME,
    page_icon=Config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
with open('styles/custom.css', 'r') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'show_register' not in st.session_state:
    st.session_state.show_register = False


def login_page():
    """Display login page"""
    st.title(f"{Config.APP_ICON} {Config.APP_NAME}")
    st.markdown("### Please login to continue")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")

            col_a, col_b = st.columns(2)
            with col_a:
                login_button = st.form_submit_button("Login", use_container_width=True)
            with col_b:
                register_button = st.form_submit_button("Register", use_container_width=True)

            if login_button:
                user_data = Auth.authenticate(username, password)
                if user_data:
                    Auth.login_user(username, user_data)
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")

            if register_button:
                st.session_state.show_register = True
                st.rerun()


def register_page():
    """Display registration page"""
    st.title(f"{Config.APP_ICON} Register New User")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        with st.form("register_form"):
            username = st.text_input("Username", placeholder="Choose a username")
            password = st.text_input("Password", type="password", placeholder="Choose a password")
            password_confirm = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
            user_type = st.selectbox("User Type", ["runner", "coach"])

            coach_id = None
            if user_type == "runner":
                from database import db
                coaches_df = db.get_coaches()

                if not coaches_df.empty:
                    coach_selection = st.selectbox(
                        "Select Your Coach",
                        options=coaches_df['username'].tolist()
                    )
                    coach_id = coaches_df[coaches_df['username'] == coach_selection]['id'].iloc[0]
                else:
                    st.warning("No coaches available. Please ask a coach to register first.")

            col_a, col_b = st.columns(2)
            with col_a:
                register_btn = st.form_submit_button("Register", use_container_width=True)
            with col_b:
                back_btn = st.form_submit_button("Back to Login", use_container_width=True)

            if register_btn:
                if password != password_confirm:
                    st.error("Passwords do not match!")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters long!")
                elif Auth.register(username, password, user_type, coach_id):
                    st.success("Registration successful! Please login.")
                    st.session_state.show_register = False
                    st.rerun()
                else:
                    st.error("Username already exists!")

            if back_btn:
                st.session_state.show_register = False
                st.rerun()


def main_app():
    """Main application after login"""
    # Sidebar
    with st.sidebar:
        st.markdown(f"### Welcome, {st.session_state.username}!")
        st.markdown(f"**Role:** {st.session_state.user_type.title()}")
        st.markdown("---")

        # Navigation
        page = st.radio(
            "Navigation",
            ["📤 Upload & Analyze", "📊 View Reports", "👥 Manage Users"]
        )

        st.markdown("---")

        if st.button("Logout", use_container_width=True):
            Auth.logout_user()
            st.rerun()

    # Main content
    st.title(f"{Config.APP_ICON} {Config.APP_NAME}")

    # Route to appropriate page
    if page == "📤 Upload & Analyze":
        upload_analyze.show()
    elif page == "📊 View Reports":
        view_reports.show()
    elif page == "👥 Manage Users":
        if st.session_state.user_type == 'admin':
            manage_users.show()
        else:
            st.info("This section is only available for administrators.")


# Main app logic
def main():
    if st.session_state.show_register:
        register_page()
    elif not Auth.is_authenticated():
        login_page()
    else:
        main_app()


if __name__ == "__main__":
    main()