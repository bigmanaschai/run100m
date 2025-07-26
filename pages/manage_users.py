# pages/manage_users.py - User management page

import streamlit as st
import pandas as pd
from database import db
from auth import Auth
from utils import ReportGenerator
from config import Config


@Auth.require_role(['admin'])
def show():
    """Show user management page"""
    st.header("👥 User Management")

    # Tabs for different management functions
    tab1, tab2, tab3 = st.tabs(["Add User", "View Users", "User Statistics"])

    with tab1:
        add_user_section()

    with tab2:
        view_users_section()

    with tab3:
        user_statistics_section()


def add_user_section():
    """Add new user section"""
    st.subheader("➕ Add New User")

    with st.form("add_user_form"):
        col1, col2 = st.columns(2)

        with col1:
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            password_confirm = st.text_input("Confirm Password", type="password", placeholder="Confirm password")

        with col2:
            user_type = st.selectbox("User Type", Config.USER_ROLES)

            # Coach selection for runners
            coach_id = None
            if user_type == "runner":
                coaches_df = db.get_coaches()

                if not coaches_df.empty:
                    coach_selection = st.selectbox(
                        "Select Coach",
                        options=coaches_df['username'].tolist()
                    )
                    coach_id = coaches_df[coaches_df['username'] == coach_selection]['id'].iloc[0]
                else:
                    st.warning("No coaches available. Please add a coach first.")

            # Additional info
            st.text_input("Email (optional)", placeholder="user@example.com", key="email")
            st.text_input("Phone (optional)", placeholder="+1234567890", key="phone")

        submitted = st.form_submit_button("Add User", use_container_width=True, type="primary")

        if submitted:
            # Validation
            if not username or not password:
                st.error("Username and password are required!")
            elif password != password_confirm:
                st.error("Passwords do not match!")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters long!")
            elif Auth.register(username, password, user_type, coach_id):
                st.success(f"✅ User '{username}' added successfully!")
                st.balloons()
                st.rerun()
            else:
                st.error("Failed to add user. Username might already exist.")


def view_users_section():
    """View and manage existing users"""
    st.subheader("📋 Existing Users")

    # Get all users
    users_df = db.get_all_users()

    if users_df.empty:
        st.info("No users found in the system.")
        return

    # Filter options
    col1, col2, col3 = st.columns(3)

    with col1:
        user_type_filter = st.multiselect(
            "Filter by Role",
            options=Config.USER_ROLES,
            default=Config.USER_ROLES
        )

    with col2:
        search_term = st.text_input("Search Users", placeholder="Search by username...")

    with col3:
        sort_by = st.selectbox("Sort By", ["Created Date", "Username", "User Type"])

    # Apply filters
    filtered_df = users_df.copy()

    if user_type_filter:
        filtered_df = filtered_df[filtered_df['user_type'].isin(user_type_filter)]

    if search_term:
        filtered_df = filtered_df[
            filtered_df['username'].str.contains(search_term, case=False, na=False)
        ]

    # Apply sorting
    if sort_by == "Created Date":
        filtered_df = filtered_df.sort_values('created_at', ascending=False)
    elif sort_by == "Username":
        filtered_df = filtered_df.sort_values('username')
    elif sort_by == "User Type":
        filtered_df = filtered_df.sort_values('user_type')

    # Display users
    for idx, user in filtered_df.iterrows():
        with st.container():
            col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 1])

            with col1:
                st.markdown(f"**{user['username']}**")
                st.caption(f"ID: {user['id']}")

            with col2:
                role_emoji = {"admin": "👑", "coach": "🏋️", "runner": "🏃"}.get(user['user_type'], "👤")
                st.markdown(f"{role_emoji} {user['user_type'].title()}")

            with col3:
                if user['coach_name']:
                    st.markdown(f"Coach: {user['coach_name']}")
                else:
                    st.markdown("No coach")

            with col4:
                st.caption(f"Joined: {user['created_at'].strftime('%Y-%m-%d')}")

            with col5:
                if user['username'] != 'admin':  # Prevent deleting admin
                    if st.button("🗑️", key=f"delete_{user['id']}", help="Delete user"):
                        if st.checkbox(f"Confirm delete {user['username']}", key=f"confirm_{user['id']}"):
                            db.delete_user(user['id'])
                            st.success(f"User '{user['username']}' deleted.")
                            st.rerun()

            st.divider()

    # Export users
    if st.button("📥 Export User List", use_container_width=True):
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="users_export.csv",
            mime="text/csv"
        )


def user_statistics_section():
    """Display user statistics"""
    st.subheader("📊 User Statistics")

    # Get statistics
    users_df = db.get_all_users()
    performance_df = db.get_performance_data(None, 'admin')

    # User counts
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_users = len(users_df)
        st.metric("Total Users", total_users, help="Total registered users")

    with col2:
        coaches = len(users_df[users_df['user_type'] == 'coach'])
        st.metric("Coaches", coaches, help="Total number of coaches")

    with col3:
        runners = len(users_df[users_df['user_type'] == 'runner'])
        st.metric("Runners", runners, help="Total number of runners")

    with col4:
        total_tests = len(performance_df) if not performance_df.empty else 0
        st.metric("Total Tests", total_tests, help="Total performance tests conducted")

    # Charts
    if not users_df.empty:
        col1, col2 = st.columns(2)

        with col1:
            # User distribution pie chart
            user_dist = users_df['user_type'].value_counts()

            import plotly.graph_objects as go

            fig = go.Figure(data=[go.Pie(
                labels=user_dist.index.str.title(),
                values=user_dist.values,
                hole=0.3,
                marker_colors=[Config.PRIMARY_COLOR, Config.SUCCESS_COLOR, Config.INFO_COLOR]
            )])

            fig.update_layout(
                title="User Distribution",
                height=400,
                font=dict(family='Prompt')
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Registration timeline
            users_df['created_date'] = pd.to_datetime(users_df['created_at']).dt.date
            registrations = users_df.groupby('created_date').size().reset_index(name='count')

            fig = go.Figure(data=[go.Scatter(
                x=registrations['created_date'],
                y=registrations['count'],
                mode='lines+markers',
                line=dict(color=Config.PRIMARY_COLOR, width=2),
                marker=dict(size=8)
            )])

            fig.update_layout(
                title="User Registrations Over Time",
                xaxis_title="Date",
                yaxis_title="New Users",
                height=400,
                font=dict(family='Prompt')
            )

            st.plotly_chart(fig, use_container_width=True)

    # Performance statistics
    if not performance_df.empty:
        st.markdown("### 🏃 Performance Overview")

        col1, col2, col3 = st.columns(3)

        with col1:
            avg_max_speed = performance_df['max_speed'].mean()
            st.metric("Avg Max Speed", f"{avg_max_speed:.2f} m/s", help="Average maximum speed across all tests")

        with col2:
            best_speed = performance_df['max_speed'].max()
            best_runner = performance_df[performance_df['max_speed'] == best_speed]['runner_name'].iloc[0]
            st.metric("Best Speed", f"{best_speed:.2f} m/s", help=f"Achieved by {best_runner}")

        with col3:
            tests_per_runner = performance_df.groupby('runner_name').size().mean()
            st.metric("Avg Tests/Runner", f"{tests_per_runner:.1f}", help="Average number of tests per runner")

        # Top performers table
        st.markdown("### 🏆 Top Performers")

        top_performers = performance_df.groupby('runner_name').agg({
            'max_speed': ['max', 'mean', 'count']
        }).round(2)

        top_performers.columns = ['Best Speed (m/s)', 'Avg Speed (m/s)', 'Total Tests']
        top_performers = top_performers.sort_values('Best Speed (m/s)', ascending=False).head(10)

        st.dataframe(
            top_performers,
            use_container_width=True,
            height=min(400, len(top_performers) * 40 + 50)
        )