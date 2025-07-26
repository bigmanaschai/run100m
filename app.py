# app.py - Single file version for easier deployment
# Rename this to app.py if you want to use the single file version

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import sqlite3
import hashlib
import os
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Running Performance Analysis",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');

.stApp {
    font-family: 'Prompt', sans-serif;
    background-color: #f0f2f6;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Prompt', sans-serif !important;
    color: #1e3a5f !important;
}

.stButton > button {
    background-color: #ff6b35;
    color: white;
    font-family: 'Prompt', sans-serif;
    border: none;
    border-radius: 5px;
    padding: 0.5rem 1rem;
    font-weight: 500;
    transition: all 0.3s;
}

.stButton > button:hover {
    background-color: #ff5722;
    transform: translateY(-2px);
}

[data-testid="metric-container"] {
    background-color: white;
    border: 1px solid #e0e0e0;
    padding: 1rem;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.stFileUploader {
    background-color: white;
    border: 2px dashed #ff6b35;
    border-radius: 10px;
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_type = None
    st.session_state.username = None
    st.session_state.user_id = None
    st.session_state.coach_id = None


# Database functions
def init_db():
    conn = sqlite3.connect('running_performance.db')
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (
                     id
                     INTEGER
                     PRIMARY
                     KEY
                     AUTOINCREMENT,
                     username
                     TEXT
                     UNIQUE
                     NOT
                     NULL,
                     password
                     TEXT
                     NOT
                     NULL,
                     user_type
                     TEXT
                     NOT
                     NULL,
                     coach_id
                     INTEGER,
                     created_at
                     TIMESTAMP
                     DEFAULT
                     CURRENT_TIMESTAMP
                 )''')

    c.execute('''CREATE TABLE IF NOT EXISTS performance_data
                 (
                     id
                     INTEGER
                     PRIMARY
                     KEY
                     AUTOINCREMENT,
                     runner_id
                     INTEGER
                     NOT
                     NULL,
                     coach_id
                     INTEGER,
                     test_date
                     TIMESTAMP
                     DEFAULT
                     CURRENT_TIMESTAMP,
                     range_0_25_data
                     TEXT,
                     range_25_50_data
                     TEXT,
                     range_50_75_data
                     TEXT,
                     range_75_100_data
                     TEXT,
                     max_speed
                     REAL,
                     avg_speed
                     REAL,
                     total_time
                     REAL
                 )''')

    # Add default admin
    try:
        c.execute("INSERT INTO users (username, password, user_type) VALUES (?, ?, ?)",
                  ('admin', hashlib.sha256('admin123'.encode()).hexdigest(), 'admin'))
    except:
        pass

    conn.commit()
    conn.close()


init_db()


# Auth functions
def authenticate(username, password):
    conn = sqlite3.connect('running_performance.db')
    c = conn.cursor()
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    c.execute("SELECT id, user_type, coach_id FROM users WHERE username=? AND password=?",
              (username, password_hash))
    result = c.fetchone()
    conn.close()
    return result


def register_user(username, password, user_type, coach_id=None):
    conn = sqlite3.connect('running_performance.db')
    c = conn.cursor()
    try:
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        c.execute("INSERT INTO users (username, password, user_type, coach_id) VALUES (?, ?, ?, ?)",
                  (username, password_hash, user_type, coach_id))
        conn.commit()
        conn.close()
        return True
    except:
        conn.close()
        return False


# Processing functions
def process_video_mock(video_file, range_type):
    """Mock processing - replace with your actual DL model"""
    t_start = int(range_type.split('-')[0]) / 25 * 3.5 if int(range_type.split('-')[0]) > 0 else 0

    t_values = []
    x_values = []
    v_values = []

    for i in range(50):
        t = t_start + i * 0.133
        t_values.append(t)

        if i % 4 == 0:
            x = np.random.uniform(-2, 2) + i * 0.5
            x_values.append(x)
        else:
            x_values.append(np.nan)

        if i % 4 == 2:
            v = np.random.uniform(5, 10)
            v_values.append(v)
        else:
            v_values.append(np.nan)

    return pd.DataFrame({'t': t_values, 'x': x_values, 'v': v_values})


def create_position_speed_plot(data_dict):
    fig = go.Figure()

    all_x = []
    all_v = []

    for range_key, df in data_dict.items():
        if df is not None:
            clean_df = df.dropna(subset=['x', 'v'])
            if not clean_df.empty:
                all_x.extend(clean_df['x'].tolist())
                all_v.extend(clean_df['v'].tolist())

    if all_x and all_v:
        sorted_data = sorted(zip(all_x, all_v))
        x_sorted = [x for x, v in sorted_data]
        v_sorted = [v for x, v in sorted_data]

        fig.add_trace(go.Scatter(
            x=x_sorted,
            y=v_sorted,
            mode='lines',
            line=dict(color='#2196f3', width=3),
            name='Speed'
        ))

    fig.update_layout(
        title='Relationship between position (m.) and speed (m/s)',
        xaxis=dict(title='Position (m)', range=[-20, 120], dtick=20),
        yaxis=dict(title='Speed (m/s)', range=[0, 10], dtick=1),
        plot_bgcolor='white',
        height=500
    )

    return fig


def generate_excel_report(data_dict, runner_name, metrics):
    output = BytesIO()

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        summary_data = {
            'Metric': ['Runner Name', 'Test Date', 'Max Speed (m/s)', 'Avg Speed (m/s)', 'Total Distance (m)',
                       'Test Duration (s)'],
            'Value': [runner_name, datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                      f"{metrics['max_speed']:.2f}", f"{metrics['avg_speed']:.2f}", '100',
                      f"{metrics['total_time']:.2f}"]
        }

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        for range_key, df in data_dict.items():
            if df is not None:
                df.to_excel(writer, sheet_name=f'Range_{range_key}m', index=False)

    output.seek(0)
    return output


# Main app
def main():
    if not st.session_state.logged_in:
        # Login page
        st.title("🏃 Running Performance Analysis System")
        st.markdown("### Please login to continue")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")

                col_a, col_b = st.columns(2)
                with col_a:
                    if st.form_submit_button("Login", use_container_width=True):
                        result = authenticate(username, password)
                        if result:
                            st.session_state.logged_in = True
                            st.session_state.username = username
                            st.session_state.user_id = result[0]
                            st.session_state.user_type = result[1]
                            st.session_state.coach_id = result[2]
                            st.rerun()
                        else:
                            st.error("Invalid credentials")

                with col_b:
                    if st.form_submit_button("Register", use_container_width=True):
                        st.info("Please contact admin for registration")
    else:
        # Main app
        with st.sidebar:
            st.markdown(f"### Welcome, {st.session_state.username}!")
            st.markdown(f"**Role:** {st.session_state.user_type.title()}")
            st.markdown("---")

            if st.button("Logout", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

        st.title("🏃 Running Performance Analysis")

        tabs = st.tabs(["📤 Upload & Analyze", "📊 View Reports", "👥 Admin"])

        with tabs[0]:
            # Upload section
            st.header("Upload Videos for Analysis")

            col1, col2 = st.columns(2)
            video_files = {}

            with col1:
                st.markdown("#### 0-25m Range")
                video_files['0-25'] = st.file_uploader("Upload video", type=['mp4', 'avi', 'mov'], key="v1")
                st.markdown("#### 50-75m Range")
                video_files['50-75'] = st.file_uploader("Upload video", type=['mp4', 'avi', 'mov'], key="v3")

            with col2:
                st.markdown("#### 25-50m Range")
                video_files['25-50'] = st.file_uploader("Upload video", type=['mp4', 'avi', 'mov'], key="v2")
                st.markdown("#### 75-100m Range")
                video_files['75-100'] = st.file_uploader("Upload video", type=['mp4', 'avi', 'mov'], key="v4")

            if st.button("🚀 Analyze Performance", use_container_width=True):
                if not any(video_files.values()):
                    st.error("Please upload at least one video")
                else:
                    with st.spinner("Processing..."):
                        data_dict = {}
                        for range_key, video in video_files.items():
                            if video:
                                data_dict[range_key] = process_video_mock(video, range_key)
                            else:
                                data_dict[range_key] = None

                        # Calculate metrics
                        all_speeds = []
                        max_time = 0
                        for df in data_dict.values():
                            if df is not None:
                                speeds = df['v'].dropna()
                                if not speeds.empty:
                                    all_speeds.extend(speeds.tolist())
                                if not df['t'].empty:
                                    max_time = max(max_time, df['t'].max())

                        metrics = {
                            'max_speed': max(all_speeds) if all_speeds else 0,
                            'avg_speed': np.mean(all_speeds) if all_speeds else 0,
                            'total_time': max_time
                        }

                        # Save to database
                        conn = sqlite3.connect('running_performance.db')
                        c = conn.cursor()

                        db_data = {
                            'runner_id': st.session_state.user_id,
                            'coach_id': st.session_state.coach_id,
                            'max_speed': metrics['max_speed'],
                            'avg_speed': metrics['avg_speed'],
                            'total_time': metrics['total_time']
                        }

                        for range_key, df in data_dict.items():
                            if df is not None:
                                db_data[f"range_{range_key.replace('-', '_')}_data"] = df.to_json()

                        columns = ', '.join(db_data.keys())
                        placeholders = ', '.join(['?' for _ in db_data])
                        c.execute(f"INSERT INTO performance_data ({columns}) VALUES ({placeholders})",
                                  list(db_data.values()))
                        conn.commit()
                        conn.close()

                        st.success("✅ Analysis completed!")

                        # Display results
                        st.header("Results")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Max Speed", f"{metrics['max_speed']:.2f} m/s")
                        with col2:
                            st.metric("Avg Speed", f"{metrics['avg_speed']:.2f} m/s")
                        with col3:
                            st.metric("Total Time", f"{metrics['total_time']:.2f} s")

                        # Plot
                        fig = create_position_speed_plot(data_dict)
                        st.plotly_chart(fig, use_container_width=True)

                        # Download report
                        excel_data = generate_excel_report(data_dict, st.session_state.username, metrics)
                        st.download_button(
                            "📥 Download Excel Report",
                            data=excel_data,
                            file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

        with tabs[1]:
            st.header("📊 Performance Reports")

            conn = sqlite3.connect('running_performance.db')
            if st.session_state.user_type == 'admin':
                df = pd.read_sql_query("SELECT * FROM performance_data ORDER BY test_date DESC", conn)
            else:
                df = pd.read_sql_query("SELECT * FROM performance_data WHERE runner_id=? ORDER BY test_date DESC",
                                       conn, params=(st.session_state.user_id,))
            conn.close()

            if df.empty:
                st.info("No performance data available yet.")
            else:
                for idx, row in df.iterrows():
                    with st.expander(f"Test Date: {row['test_date']} - Speed: {row['max_speed']:.2f} m/s"):
                        st.metric("Max Speed", f"{row['max_speed']:.2f} m/s")
                        st.metric("Avg Speed", f"{row['avg_speed']:.2f} m/s")
                        st.metric("Total Time", f"{row['total_time']:.2f} s")

        with tabs[2]:
            if st.session_state.user_type == 'admin':
                st.header("👥 User Management")

                with st.form("add_user"):
                    st.subheader("Add New User")
                    new_username = st.text_input("Username")
                    new_password = st.text_input("Password", type="password")
                    new_user_type = st.selectbox("User Type", ["runner", "coach", "admin"])

                    if st.form_submit_button("Add User"):
                        if register_user(new_username, new_password, new_user_type):
                            st.success(f"User {new_username} added!")
                        else:
                            st.error("Failed to add user")

                st.subheader("Existing Users")
                conn = sqlite3.connect('running_performance.db')
                users_df = pd.read_sql_query("SELECT username, user_type, created_at FROM users", conn)
                conn.close()
                st.dataframe(users_df)
            else:
                st.info("Admin access only")


if __name__ == "__main__":
    main()