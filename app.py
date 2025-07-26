# app_with_yolo11.py - Complete application with YOLO 11 integration

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import sqlite3
import hashlib
import os
from io import BytesIO
import json
import tempfile

# Check if YOLO dependencies are available
try:
    from ultralytics import YOLO
    import cv2
    import torch

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    st.warning("YOLO 11 not available. Using mock data for demonstration.")

# Page configuration
st.set_page_config(
    page_title="Running Performance Analysis with YOLO 11",
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
    border-radius: 8px;
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

.yolo-info {
    background-color: #e3f2fd;
    border-left: 4px solid #2196f3;
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 0 8px 8px 0;
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
    st.session_state.yolo_model = None


# Initialize YOLO model
@st.cache_resource
def load_yolo_model(model_name='yolo11n.pt'):
    """Load YOLO 11 model"""
    if YOLO_AVAILABLE:
        try:
            model = YOLO(model_name)
            return model
        except Exception as e:
            st.error(f"Error loading YOLO model: {e}")
            return None
    return None


# Database setup
def init_db():
    conn = sqlite3.connect('running_performance_yolo.db')
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
                     REAL,
                     detection_confidence
                     REAL,
                     frames_processed
                     INTEGER
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


# YOLO processing functions
def process_video_with_yolo(video_file, range_type, progress_bar=None):
    """Process video using YOLO 11"""

    if not YOLO_AVAILABLE or st.session_state.yolo_model is None:
        return process_video_mock(video_file, range_type)

    # Save video temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_file.getbuffer())
        video_path = tmp_file.name

    try:
        # Open video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialize tracking
        track_history = {}
        runner_data = []
        frame_count = 0
        frames_processed = 0

        # Calibration (should be adjusted based on camera setup)
        pixels_per_meter = 20  # Example: 20 pixels = 1 meter
        range_offset = int(range_type.split('-')[0])

        # Process frames
        frame_skip = max(1, int(fps / 10))  # Process ~10 fps

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                # Update progress
                if progress_bar:
                    progress = frame_count / total_frames
                    progress_bar.progress(progress, f"Processing {range_type}m: {int(progress * 100)}%")

                # Run YOLO tracking
                results = st.session_state.yolo_model.track(frame, persist=True, classes=[0])

                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xywh.cpu()
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    confidences = results[0].boxes.conf.cpu().tolist()

                    # Process largest/most confident detection
                    if len(boxes) > 0:
                        # Get box with highest confidence
                        best_idx = np.argmax(confidences)
                        x_center = float(boxes[best_idx, 0])
                        track_id = track_ids[best_idx]

                        # Initialize history
                        if track_id not in track_history:
                            track_history[track_id] = []

                        # Add to history
                        track_history[track_id].append({
                            'x': x_center,
                            'frame': frame_count,
                            'time': frame_count / fps
                        })

                        # Calculate velocity if enough history
                        if len(track_history[track_id]) >= 2:
                            curr = track_history[track_id][-1]
                            prev = track_history[track_id][-2]

                            # Calculate position in meters
                            position_m = (curr['x'] / pixels_per_meter) + range_offset

                            # Calculate velocity
                            dx = (curr['x'] - prev['x']) / pixels_per_meter
                            dt = curr['time'] - prev['time']
                            velocity = abs(dx / dt) if dt > 0 else 0

                            # Reasonable bounds for running
                            velocity = np.clip(velocity, 0, 12)

                            runner_data.append({
                                't': curr['time'],
                                'x': position_m,
                                'v': velocity
                            })

                frames_processed += 1

            frame_count += 1

        cap.release()
        os.unlink(video_path)  # Clean up temp file

        # Convert to DataFrame and format
        if runner_data:
            df = pd.DataFrame(runner_data)

            # Smooth data
            if len(df) > 5:
                df['v'] = df['v'].rolling(window=5, center=True).mean().fillna(df['v'])

            # Resample to sparse format
            sparse_df = create_sparse_format(df)

            return sparse_df, frames_processed
        else:
            st.warning(f"No runner detected in {range_type}m video")
            return process_video_mock(video_file, range_type), 0

    except Exception as e:
        st.error(f"Error processing video: {e}")
        if os.path.exists(video_path):
            os.unlink(video_path)
        return process_video_mock(video_file, range_type), 0


def create_sparse_format(df):
    """Convert continuous data to sparse format"""
    if len(df) < 10:
        return df

    sparse_data = []
    time_step = 0.133
    max_time = df['t'].max()

    for i, t in enumerate(np.arange(0, max_time, time_step)):
        # Find closest time
        idx = (df['t'] - t).abs().idxmin()

        if i % 4 == 0:  # Position data
            sparse_data.append({'t': t, 'x': df.loc[idx, 'x'], 'v': np.nan})
        elif i % 4 == 2:  # Velocity data
            sparse_data.append({'t': t, 'x': np.nan, 'v': df.loc[idx, 'v']})
        else:  # Empty
            sparse_data.append({'t': t, 'x': np.nan, 'v': np.nan})

    return pd.DataFrame(sparse_data)


def process_video_mock(video_file, range_type):
    """Mock processing when YOLO is not available"""
    range_start = int(range_type.split('-')[0])
    range_end = int(range_type.split('-')[1])
    time_offset = (range_start / 25) * 3.5 if range_start > 0 else 0

    t_values = []
    x_values = []
    v_values = []

    for i in range(50):
        t = time_offset + i * 0.133
        t_values.append(t)

        if i % 4 == 0:
            progress = i / 50
            x = range_start + (range_end - range_start) * progress + np.random.normal(0, 0.5)
            x_values.append(x)
        else:
            x_values.append(np.nan)

        if i % 4 == 2:
            if range_type == "0-25":
                v_base = 6.0 + progress * 3.0
            elif range_type == "25-50":
                v_base = 8.5 + np.sin(progress * np.pi) * 0.5
            elif range_type == "50-75":
                v_base = 8.3 - progress * 0.3
            else:
                v_base = 7.5 - progress * 0.5

            v = v_base + np.random.normal(0, 0.2)
            v_values.append(max(0, v))
        else:
            v_values.append(np.nan)

    return pd.DataFrame({'t': t_values, 'x': x_values, 'v': v_values})


# Helper functions (authentication, plotting, etc.)
def authenticate_user(username, password):
    conn = sqlite3.connect('running_performance_yolo.db')
    c = conn.cursor()
    c.execute("SELECT id, user_type, coach_id FROM users WHERE username=? AND password=?",
              (username, hashlib.sha256(password.encode()).hexdigest()))
    result = c.fetchone()
    conn.close()
    return result


def register_user(username, password, user_type, coach_id=None):
    conn = sqlite3.connect('running_performance_yolo.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password, user_type, coach_id) VALUES (?, ?, ?, ?)",
                  (username, hashlib.sha256(password.encode()).hexdigest(), user_type, coach_id))
        conn.commit()
        conn.close()
        return True
    except:
        conn.close()
        return False


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
        height=500,
        font=dict(family='Prompt')
    )

    return fig


def generate_excel_report(data_dict, runner_name, metrics, detection_info=None):
    output = BytesIO()

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Summary sheet
        summary_data = {
            'Metric': ['Runner Name', 'Test Date', 'Max Speed (m/s)', 'Avg Speed (m/s)',
                       'Total Distance (m)', 'Test Duration (s)', 'Detection Method'],
            'Value': [runner_name, datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                      f"{metrics['max_speed']:.2f}", f"{metrics['avg_speed']:.2f}",
                      '100', f"{metrics['total_time']:.2f}",
                      'YOLO 11' if YOLO_AVAILABLE else 'Mock Data']
        }

        if detection_info:
            summary_data['Metric'].extend(['Frames Processed', 'Average Confidence'])
            summary_data['Value'].extend([str(detection_info.get('frames_processed', 'N/A')),
                                          f"{detection_info.get('avg_confidence', 0):.2%}"])

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # Data sheets
        for range_key, df in data_dict.items():
            if df is not None:
                df.to_excel(writer, sheet_name=f'Range_{range_key}m', index=False)

    output.seek(0)
    return output


# Main application
def main():
    if not st.session_state.logged_in:
        # Login page
        st.title("🏃 Running Performance Analysis with YOLO 11")

        # Show YOLO status
        if YOLO_AVAILABLE:
            st.success("✅ YOLO 11 is available for person detection")
        else:
            st.warning("⚠️ YOLO 11 not available. Install with: pip install ultralytics")

        st.markdown("### Please login to continue")

        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            tab1, tab2 = st.tabs(["Login", "Register"])

            with tab1:
                with st.form("login_form"):
                    username = st.text_input("Username", placeholder="Enter your username")
                    password = st.text_input("Password", type="password", placeholder="Enter your password")

                    if st.form_submit_button("Login", use_container_width=True):
                        result = authenticate_user(username, password)
                        if result:
                            st.session_state.logged_in = True
                            st.session_state.username = username
                            st.session_state.user_id = result[0]
                            st.session_state.user_type = result[1]
                            st.session_state.coach_id = result[2]

                            # Load YOLO model after login
                            if YOLO_AVAILABLE:
                                with st.spinner("Loading YOLO 11 model..."):
                                    st.session_state.yolo_model = load_yolo_model()

                            st.success("Login successful!")
                            st.rerun()
                        else:
                            st.error("Invalid username or password")

                    st.info("Default credentials: **admin** / **admin123**")

            with tab2:
                with st.form("register_form"):
                    new_username = st.text_input("Username", placeholder="Choose a username")
                    new_password = st.text_input("Password", type="password", placeholder="Choose a password")
                    new_password_confirm = st.text_input("Confirm Password", type="password")
                    user_type = st.selectbox("User Type", ["runner", "coach"])

                    coach_id = None
                    if user_type == "runner":
                        conn = sqlite3.connect('running_performance_yolo.db')
                        coaches_df = pd.read_sql_query("SELECT id, username FROM users WHERE user_type='coach'", conn)
                        conn.close()

                        if not coaches_df.empty:
                            coach_selection = st.selectbox("Select Your Coach", coaches_df['username'].tolist())
                            coach_id = coaches_df[coaches_df['username'] == coach_selection]['id'].iloc[0]

                    if st.form_submit_button("Register", use_container_width=True):
                        if new_password != new_password_confirm:
                            st.error("Passwords do not match!")
                        elif len(new_password) < 6:
                            st.error("Password must be at least 6 characters long!")
                        elif register_user(new_username, new_password, user_type, coach_id):
                            st.success("Registration successful! Please login.")
                        else:
                            st.error("Username already exists!")

    else:
        # Logged in - Main application
        with st.sidebar:
            st.markdown(f"### Welcome, {st.session_state.username}!")
            st.markdown(f"**Role:** {st.session_state.user_type.title()}")

            # YOLO Model Settings
            if YOLO_AVAILABLE:
                st.markdown("---")
                st.markdown("### 🤖 YOLO Settings")

                model_size = st.selectbox(
                    "Model Size",
                    ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"],
                    help="n=nano (fastest), s=small, m=medium"
                )

                if st.button("Reload Model"):
                    with st.spinner("Loading model..."):
                        st.session_state.yolo_model = load_yolo_model(model_size)
                    st.success("Model reloaded!")

                if torch.cuda.is_available():
                    st.success("🚀 GPU acceleration available")
                else:
                    st.info("💻 Using CPU (slower)")

            st.markdown("---")

            if st.button("🚪 Logout", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

        st.title("🏃 Running Performance Analysis with YOLO 11")

        # Show YOLO info box
        if YOLO_AVAILABLE:
            st.markdown("""
            <div class="yolo-info">
            <strong>🤖 YOLO 11 Person Detection Active</strong><br>
            Automatically detects and tracks runners in uploaded videos for accurate performance analysis.
            </div>
            """, unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["📤 Upload & Analyze", "📊 View Reports", "👥 Manage Users"])

        # Tab 1: Upload & Analyze
        with tab1:
            st.header("Upload Videos for Analysis")

            # Runner selection for coaches/admins
            runner_id = st.session_state.user_id
            runner_name = st.session_state.username

            if st.session_state.user_type in ['coach', 'admin']:
                conn = sqlite3.connect('running_performance_yolo.db')
                if st.session_state.user_type == 'coach':
                    runners_df = pd.read_sql_query(
                        "SELECT id, username FROM users WHERE user_type='runner' AND coach_id=?",
                        conn, params=(st.session_state.user_id,))
                else:
                    runners_df = pd.read_sql_query(
                        "SELECT id, username FROM users WHERE user_type='runner'", conn)
                conn.close()

                if not runners_df.empty:
                    runner_selection = st.selectbox("Select Runner", runners_df['username'].tolist())
                    runner_id = runners_df[runners_df['username'] == runner_selection]['id'].iloc[0]
                    runner_name = runner_selection

            st.markdown("### Upload videos for each range")

            col1, col2 = st.columns(2)
            video_files = {}

            with col1:
                st.markdown("#### 🎥 0-25m Range")
                video_files['0-25'] = st.file_uploader("Upload video", type=['mp4', 'avi', 'mov'], key="v1")
                st.markdown("#### 🎥 50-75m Range")
                video_files['50-75'] = st.file_uploader("Upload video", type=['mp4', 'avi', 'mov'], key="v3")

            with col2:
                st.markdown("#### 🎥 25-50m Range")
                video_files['25-50'] = st.file_uploader("Upload video", type=['mp4', 'avi', 'mov'], key="v2")
                st.markdown("#### 🎥 75-100m Range")
                video_files['75-100'] = st.file_uploader("Upload video", type=['mp4', 'avi', 'mov'], key="v4")

            # Advanced options
            with st.expander("⚙️ Advanced Options"):
                col1, col2 = st.columns(2)
                with col1:
                    confidence_threshold = st.slider("Detection Confidence", 0.0, 1.0, 0.5)
                with col2:
                    show_preview = st.checkbox("Show detection preview", value=False)

            if st.button("🚀 Analyze Performance", use_container_width=True, type="primary"):
                if not any(video_files.values()):
                    st.error("Please upload at least one video file.")
                else:
                    analysis_container = st.container()

                    with analysis_container:
                        st.markdown("### 🔄 Processing Videos...")

                        data_dict = {}
                        detection_info = {'frames_processed': 0, 'confidences': []}

                        # Process each video
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        uploaded_ranges = [(k, v) for k, v in video_files.items() if v is not None]

                        for idx, (range_key, video_file) in enumerate(uploaded_ranges):
                            status_text.text(f"Processing {range_key}m range...")

                            if YOLO_AVAILABLE and st.session_state.yolo_model:
                                df, frames = process_video_with_yolo(video_file, range_key, progress_bar)
                                detection_info['frames_processed'] += frames
                            else:
                                df = process_video_mock(video_file, range_key)

                            data_dict[range_key] = df
                            progress_bar.progress((idx + 1) / len(uploaded_ranges))

                        # Fill missing ranges with None
                        for range_key in ['0-25', '25-50', '50-75', '75-100']:
                            if range_key not in data_dict:
                                data_dict[range_key] = None

                        progress_bar.empty()
                        status_text.empty()

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
                            'min_speed': min(all_speeds) if all_speeds else 0,
                            'total_time': max_time
                        }

                        # Save to database
                        conn = sqlite3.connect('running_performance_yolo.db')
                        c = conn.cursor()

                        db_data = {
                            'runner_id': runner_id,
                            'coach_id': st.session_state.coach_id if st.session_state.user_type == 'runner' else st.session_state.user_id,
                            'max_speed': metrics['max_speed'],
                            'avg_speed': metrics['avg_speed'],
                            'total_time': metrics['total_time'],
                            'detection_confidence': confidence_threshold if YOLO_AVAILABLE else 0,
                            'frames_processed': detection_info['frames_processed']
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

                        st.success("✅ Analysis completed successfully!")

                        # Display results
                        st.markdown("---")
                        st.header("📊 Analysis Results")

                        # Metrics
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Max Speed", f"{metrics['max_speed']:.2f} m/s",
                                    help="Peak speed detected by YOLO 11" if YOLO_AVAILABLE else "Peak speed from mock data")
                        col2.metric("Avg Speed", f"{metrics['avg_speed']:.2f} m/s")
                        col3.metric("Min Speed", f"{metrics['min_speed']:.2f} m/s")
                        col4.metric("Total Time", f"{metrics['total_time']:.2f} s")

                        # Detection info
                        if YOLO_AVAILABLE and detection_info['frames_processed'] > 0:
                            st.info(
                                f"🤖 YOLO 11 processed {detection_info['frames_processed']} frames across all videos")

                        # Plot
                        fig = create_position_speed_plot(data_dict)
                        st.plotly_chart(fig, use_container_width=True)

                        # Download button
                        excel_data = generate_excel_report(data_dict, runner_name, metrics, detection_info)
                        st.download_button(
                            label="📥 Download Excel Report",
                            data=excel_data,
                            file_name=f"running_analysis_{runner_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

        # Tab 2: View Reports
        with tab2:
            st.header("📊 View Performance Reports")

            conn = sqlite3.connect('running_performance_yolo.db')

            if st.session_state.user_type == 'admin':
                query = """SELECT p.*, u.username as runner_name
                           FROM performance_data p
                                    JOIN users u ON p.runner_id = u.id
                           ORDER BY p.test_date DESC"""
                df = pd.read_sql_query(query, conn)
            elif st.session_state.user_type == 'coach':
                query = """SELECT p.*, u.username as runner_name
                           FROM performance_data p
                                    JOIN users u ON p.runner_id = u.id
                           WHERE p.coach_id = ?
                           ORDER BY p.test_date DESC"""
                df = pd.read_sql_query(query, conn, params=(st.session_state.user_id,))
            else:
                query = """SELECT p.* \
                           FROM performance_data p
                           WHERE p.runner_id = ?
                           ORDER BY p.test_date DESC"""
                df = pd.read_sql_query(query, conn, params=(st.session_state.user_id,))

            conn.close()

            if df.empty:
                st.info("No performance data available yet.")
            else:
                for idx, row in df.iterrows():
                    runner_label = f" - {row.get('runner_name', 'Me')}" if 'runner_name' in row else ""
                    detection_label = " (YOLO 11)" if row.get('detection_confidence', 0) > 0 else " (Mock)"

                    with st.expander(
                            f"📅 {row['test_date'][:19]}{runner_label}{detection_label} | Max: {row['max_speed']:.2f} m/s"):
                        # Display metrics
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Max Speed", f"{row['max_speed']:.2f} m/s")
                        col2.metric("Avg Speed", f"{row['avg_speed']:.2f} m/s")
                        col3.metric("Total Time", f"{row['total_time']:.2f} s")
                        col4.metric("Frames Analyzed", row.get('frames_processed', 'N/A'))

                        # Reconstruct and plot data
                        data_dict = {}
                        for range_key in ['0-25', '25-50', '50-75', '75-100']:
                            col_name = f"range_{range_key.replace('-', '_')}_data"
                            if pd.notna(row[col_name]):
                                data_dict[range_key] = pd.read_json(row[col_name])
                            else:
                                data_dict[range_key] = None

                        fig = create_position_speed_plot(data_dict)
                        st.plotly_chart(fig, use_container_width=True)

        # Tab 3: Manage Users (Admin only)
        with tab3:
            if st.session_state.user_type == 'admin':
                st.header("👥 User Management")

                # Add user form
                with st.form("add_user_form"):
                    st.subheader("Add New User")
                    col1, col2 = st.columns(2)

                    with col1:
                        new_username = st.text_input("Username")
                        new_password = st.text_input("Password", type="password")

                    with col2:
                        new_user_type = st.selectbox("User Type", ["runner", "coach", "admin"])

                        coach_id = None
                        if new_user_type == "runner":
                            conn = sqlite3.connect('running_performance_yolo.db')
                            coaches_df = pd.read_sql_query("SELECT id, username FROM users WHERE user_type='coach'",
                                                           conn)
                            conn.close()

                            if not coaches_df.empty:
                                coach_selection = st.selectbox("Select Coach", coaches_df['username'].tolist())
                                coach_id = coaches_df[coaches_df['username'] == coach_selection]['id'].iloc[0]

                    if st.form_submit_button("Add User", use_container_width=True):
                        if register_user(new_username, new_password, new_user_type, coach_id):
                            st.success(f"User '{new_username}' added successfully!")
                            st.balloons()
                        else:
                            st.error("Failed to add user. Username might already exist.")

                # Display users
                st.subheader("Existing Users")
                conn = sqlite3.connect('running_performance_yolo.db')
                users_df = pd.read_sql_query("""
                                             SELECT u1.id,
                                                    u1.username,
                                                    u1.user_type,
                                                    u2.username as coach_name,
                                                    u1.created_at
                                             FROM users u1
                                                      LEFT JOIN users u2 ON u1.coach_id = u2.id
                                             ORDER BY u1.created_at DESC
                                             """, conn)
                conn.close()

                display_df = users_df[['username', 'user_type', 'coach_name', 'created_at']].copy()
                display_df.columns = ['Username', 'User Type', 'Coach', 'Created At']
                display_df['User Type'] = display_df['User Type'].str.title()
                display_df['Coach'] = display_df['Coach'].fillna('-')

                st.dataframe(display_df, use_container_width=True)
            else:
                st.info("This section is only available for administrators.")


if __name__ == "__main__":
    main()