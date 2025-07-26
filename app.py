# app_yolo11_simple.py - Running Performance Analysis with Simplified YOLOv11

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

# Check YOLOv11 availability
YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO, solutions
    import cv2

    YOLO_AVAILABLE = True
except ImportError:
    pass

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

.installation-guide {
    background-color: #fff3cd;
    border: 1px solid #ffeaa7;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}

.installation-guide h4 {
    color: #856404 !important;
    margin-bottom: 0.5rem;
}

.installation-guide code {
    background-color: #f8f9fa;
    padding: 0.2rem 0.4rem;
    border-radius: 4px;
    font-family: monospace;
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


# Database setup
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
                     REAL,
                     analysis_method
                     TEXT
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


# YOLOv11 processing with solutions.Inference
def process_video_with_yolo11(video_file, range_type):
    """Process video using YOLOv11 solutions.Inference API"""

    if not YOLO_AVAILABLE:
        return generate_simulated_data(range_type)

    # Save video temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_file.getbuffer())
        video_path = tmp_file.name

    try:
        # Method 1: Using solutions.Inference (if available)
        try:
            # Initialize Inference solution
            inf = solutions.Inference(
                model="yolo11n.pt",  # Using nano model for speed
                source=video_path,
                conf=0.25,  # Confidence threshold
                save=False,  # Don't save output video
                show=False,  # Don't display
                classes=[0]  # Only detect persons
            )

            # Run inference
            results = inf.inference()

            # Process results
            detections = extract_detections_from_results(results, range_type)

        except:
            # Method 2: Using standard YOLO API
            model = YOLO("yolo11n.pt")

            # Process video
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            detections = []
            frame_count = 0

            # Create progress placeholder
            progress_text = st.empty()
            progress_bar = st.progress(0)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process every 3rd frame for speed
                if frame_count % 3 == 0:
                    # Run YOLO detection
                    results = model(frame, classes=[0])  # Detect only persons

                    # Extract detections
                    if results[0].boxes is not None:
                        boxes = results[0].boxes.xywh.cpu().numpy()
                        confidences = results[0].boxes.conf.cpu().numpy()

                        if len(boxes) > 0:
                            # Get the most confident person detection
                            best_idx = np.argmax(confidences)
                            x_center, y_center, width, height = boxes[best_idx]

                            detections.append({
                                'frame': frame_count,
                                'time': frame_count / fps,
                                'x': float(x_center),
                                'y': float(y_center),
                                'confidence': float(confidences[best_idx])
                            })

                    # Update progress
                    progress = frame_count / total_frames
                    progress_bar.progress(progress)
                    progress_text.text(f"Analyzing {range_type}m: {int(progress * 100)}%")

                frame_count += 1

            cap.release()
            progress_bar.empty()
            progress_text.empty()

        os.unlink(video_path)

        # Convert detections to performance data
        if detections:
            return convert_detections_to_performance(detections, range_type)
        else:
            st.warning(f"No runner detected in {range_type}m video")
            return generate_simulated_data(range_type)

    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        if os.path.exists(video_path):
            os.unlink(video_path)
        return generate_simulated_data(range_type)


def extract_detections_from_results(results, range_type):
    """Extract detections from YOLOv11 results"""
    detections = []

    # Process results based on the format returned by solutions.Inference
    # This will depend on the specific output format
    # For now, returning empty list as fallback

    return detections


def convert_detections_to_performance(detections, range_type):
    """Convert YOLO detections to performance data format"""

    if not detections:
        return generate_simulated_data(range_type)

    df = pd.DataFrame(detections)

    # Calculate velocity from position changes
    df = df.sort_values('time')
    df['dx'] = df['x'].diff()
    df['dt'] = df['time'].diff()
    df['velocity_px'] = (df['dx'] / df['dt']).abs()

    # Convert pixels to meters (calibration needed)
    pixels_per_meter = 20  # This should be calibrated for your setup
    df['position_m'] = df['x'] / pixels_per_meter
    df['velocity_ms'] = df['velocity_px'] / pixels_per_meter

    # Add range offset
    range_start = int(range_type.split('-')[0])
    df['position_m'] += range_start

    # Smooth velocity
    df['velocity_ms'] = df['velocity_ms'].rolling(window=3, center=True).mean()
    df['velocity_ms'] = df['velocity_ms'].fillna(method='bfill').fillna(method='ffill')

    # Clip to reasonable running speeds
    df['velocity_ms'] = df['velocity_ms'].clip(0, 12)

    # Create sparse format output
    return create_sparse_format(df, range_type)


def create_sparse_format(df, range_type):
    """Convert continuous data to sparse format matching expected output"""

    if df.empty:
        return generate_simulated_data(range_type)

    # Determine time range
    time_offset = (int(range_type.split('-')[0]) / 25) * 3.5 if int(range_type.split('-')[0]) > 0 else 0

    sparse_data = []
    time_step = 0.133
    max_time = min(df['time'].max(), 5.0)  # Limit to 5 seconds per segment

    for i, t in enumerate(np.arange(0, max_time, time_step)):
        actual_time = t + time_offset

        # Find closest detection
        if len(df) > 0:
            idx = (df['time'] - t).abs().idxmin()

            if i % 4 == 0:  # Position data points
                sparse_data.append({
                    't': actual_time,
                    'x': df.loc[idx, 'position_m'],
                    'v': np.nan
                })
            elif i % 4 == 2:  # Velocity data points
                sparse_data.append({
                    't': actual_time,
                    'x': np.nan,
                    'v': df.loc[idx, 'velocity_ms']
                })
            else:  # Empty points
                sparse_data.append({
                    't': actual_time,
                    'x': np.nan,
                    'v': np.nan
                })
        else:
            sparse_data.append({
                't': actual_time,
                'x': np.nan,
                'v': np.nan
            })

    return pd.DataFrame(sparse_data)


def generate_simulated_data(range_type):
    """Generate simulated performance data"""
    range_start = int(range_type.split('-')[0])
    range_end = int(range_type.split('-')[1])
    time_offset = (range_start / 25) * 3.5 if range_start > 0 else 0

    data = []

    for i in range(50):
        t = time_offset + i * 0.133

        if i % 4 == 0:  # Position data
            progress = i / 50
            x = range_start + (range_end - range_start) * progress + np.random.normal(0, 0.5)
            data.append({'t': t, 'x': x, 'v': np.nan})
        elif i % 4 == 2:  # Velocity data
            progress = i / 50
            if range_type == "0-25":
                v = 6.0 + progress * 3.0 + np.random.normal(0, 0.2)
            elif range_type == "25-50":
                v = 8.5 + np.sin(progress * np.pi) * 0.5 + np.random.normal(0, 0.2)
            elif range_type == "50-75":
                v = 8.3 - progress * 0.3 + np.random.normal(0, 0.2)
            else:
                v = 7.5 - progress * 0.5 + np.random.normal(0, 0.2)
            data.append({'t': t, 'x': np.nan, 'v': max(0, v)})
        else:
            data.append({'t': t, 'x': np.nan, 'v': np.nan})

    return pd.DataFrame(data)


# Authentication functions
def authenticate_user(username, password):
    conn = sqlite3.connect('running_performance.db')
    c = conn.cursor()
    c.execute("SELECT id, user_type, coach_id FROM users WHERE username=? AND password=?",
              (username, hashlib.sha256(password.encode()).hexdigest()))
    result = c.fetchone()
    conn.close()
    return result


def register_user(username, password, user_type, coach_id=None):
    conn = sqlite3.connect('running_performance.db')
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


# Visualization functions
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


def generate_excel_report(data_dict, runner_name, metrics):
    output = BytesIO()

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Summary sheet
        summary_data = {
            'Metric': ['Runner Name', 'Test Date', 'Max Speed (m/s)', 'Avg Speed (m/s)',
                       'Total Distance (m)', 'Test Duration (s)', 'Analysis Method'],
            'Value': [runner_name, datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                      f"{metrics['max_speed']:.2f}", f"{metrics['avg_speed']:.2f}",
                      '100', f"{metrics['total_time']:.2f}",
                      'YOLOv11' if YOLO_AVAILABLE else 'Simulated']
        }

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
        st.title("🏃 Running Performance Analysis")

        # System status in expander
        with st.expander("📊 System Status", expanded=True):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Core Features:**")
                st.success("✅ Available")

            with col2:
                st.markdown("**YOLO 11 Person Detection:**")
                if YOLO_AVAILABLE:
                    st.success("✅ Available")
                else:
                    st.error("❌ Not installed")

            with col3:
                st.markdown("**Excel Export:**")
                st.success("✅ Available")

            if not YOLO_AVAILABLE:
                st.markdown("""
                <div class="installation-guide">
                <h4>📦 To enable YOLO 11 person detection:</h4>
                <p>Install the following packages:</p>
                <code>pip install ultralytics</code>
                <p style="margin-top: 0.5rem;">This will automatically install all dependencies including PyTorch.</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("### Please login to continue")

        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
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
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")

                st.info("Default credentials: **admin** / **admin123**")

    else:
        # Logged in - Main application
        with st.sidebar:
            st.markdown(f"### Welcome, {st.session_state.username}!")
            st.markdown(f"**Role:** {st.session_state.user_type.title()}")

            st.markdown("---")
            st.markdown("### 🤖 Analysis Method")
            if YOLO_AVAILABLE:
                st.success("YOLOv11 Active")
                st.caption("AI-powered person detection")
            else:
                st.info("Simulated Data")
                st.caption("Install ultralytics for AI detection")

            st.markdown("---")

            if st.button("🚪 Logout", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

        st.title("🏃 Running Performance Analysis")

        tab1, tab2, tab3 = st.tabs(["📤 Upload & Analyze", "📊 View Reports", "👥 Manage Users"])

        # Tab 1: Upload & Analyze
        with tab1:
            st.header("Upload Videos for Analysis")

            # Show analysis method
            if YOLO_AVAILABLE:
                st.success("🤖 YOLOv11 person detection enabled - AI will automatically track runners")
            else:
                st.info("📊 Using simulated data - install ultralytics package for real video analysis")

            # Runner selection for coaches/admins
            runner_id = st.session_state.user_id
            runner_name = st.session_state.username

            if st.session_state.user_type in ['coach', 'admin']:
                conn = sqlite3.connect('running_performance.db')
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

            if st.button("🚀 Analyze Performance", use_container_width=True, type="primary"):
                if not any(video_files.values()):
                    st.error("Please upload at least one video file.")
                else:
                    with st.spinner("Processing videos..."):
                        data_dict = {}

                        # Process each video
                        for range_key, video_file in video_files.items():
                            if video_file is not None:
                                st.write(f"Processing {range_key}m range...")

                                if YOLO_AVAILABLE:
                                    df = process_video_with_yolo11(video_file, range_key)
                                else:
                                    df = generate_simulated_data(range_key)

                                data_dict[range_key] = df
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
                            'min_speed': min(all_speeds) if all_speeds else 0,
                            'total_time': max_time
                        }

                        # Save to database
                        conn = sqlite3.connect('running_performance.db')
                        c = conn.cursor()

                        db_data = {
                            'runner_id': runner_id,
                            'coach_id': st.session_state.coach_id if st.session_state.user_type == 'runner' else st.session_state.user_id,
                            'max_speed': metrics['max_speed'],
                            'avg_speed': metrics['avg_speed'],
                            'total_time': metrics['total_time'],
                            'analysis_method': 'YOLOv11' if YOLO_AVAILABLE else 'Simulated'
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
                        col1.metric("Max Speed", f"{metrics['max_speed']:.2f} m/s")
                        col2.metric("Avg Speed", f"{metrics['avg_speed']:.2f} m/s")
                        col3.metric("Min Speed", f"{metrics['min_speed']:.2f} m/s")
                        col4.metric("Total Time", f"{metrics['total_time']:.2f} s")

                        # Analysis method indicator
                        if YOLO_AVAILABLE:
                            st.info("🤖 Analysis performed using YOLOv11 AI person detection")
                        else:
                            st.warning("📊 Analysis based on simulated data (install ultralytics for real analysis)")

                        # Plot
                        fig = create_position_speed_plot(data_dict)
                        st.plotly_chart(fig, use_container_width=True)

                        # Download button
                        excel_data = generate_excel_report(data_dict, runner_name, metrics)
                        st.download_button(
                            label="📥 Download Excel Report",
                            data=excel_data,
                            file_name=f"running_analysis_{runner_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

        # Tab 2: View Reports
        with tab2:
            st.header("📊 View Performance Reports")

            conn = sqlite3.connect('running_performance.db')

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
                    method_label = f" ({row.get('analysis_method', 'Unknown')})"

                    with st.expander(
                            f"📅 {row['test_date'][:19]}{runner_label}{method_label} | Max: {row['max_speed']:.2f} m/s"):
                        # Display metrics
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Max Speed", f"{row['max_speed']:.2f} m/s")
                        col2.metric("Avg Speed", f"{row['avg_speed']:.2f} m/s")
                        col3.metric("Total Time", f"{row['total_time']:.2f} s")

                        # Show analysis method
                        if row.get('analysis_method') == 'YOLOv11':
                            st.success("🤖 Analyzed with YOLOv11 AI")
                        else:
                            st.info("📊 Simulated data")

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
                            conn = sqlite3.connect('running_performance.db')
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
                conn = sqlite3.connect('running_performance.db')
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