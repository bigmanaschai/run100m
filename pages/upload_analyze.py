# pages/upload_analyze.py - Upload and analyze videos page

import streamlit as st
import pandas as pd
import os
from datetime import datetime
from config import Config
from database import db
from models import model
from utils import Visualizer, ReportGenerator
from auth import Auth


@Auth.require_auth
def show():
    """Show upload and analyze page"""
    st.header("Upload Videos for Analysis")

    # Runner selection for coaches and admins
    runner_id = st.session_state.user_id
    runner_name = st.session_state.username

    if st.session_state.user_type in ['coach', 'admin']:
        if st.session_state.user_type == 'coach':
            runners_df = db.get_runners_for_coach(st.session_state.user_id)
        else:  # admin
            runners_df = db.get_all_runners()

        if not runners_df.empty:
            runner_selection = st.selectbox(
                "Select Runner",
                options=runners_df['username'].tolist()
            )
            runner_id = runners_df[runners_df['username'] == runner_selection]['id'].iloc[0]
            runner_name = runner_selection
        else:
            st.warning("No runners available.")
            return

    # Video upload sections
    st.markdown("### Upload videos for each range")
    st.info(
        "Upload one video for each running segment. Videos will be analyzed using deep learning to extract performance metrics.")

    col1, col2 = st.columns(2)
    video_files = {}

    with col1:
        st.markdown("#### 🎥 0-25m Range")
        video_files['0-25'] = st.file_uploader(
            "Upload video for 0-25m",
            type=Config.ALLOWED_VIDEO_FORMATS,
            key="video_0_25",
            help="First segment of the run"
        )

        st.markdown("#### 🎥 50-75m Range")
        video_files['50-75'] = st.file_uploader(
            "Upload video for 50-75m",
            type=Config.ALLOWED_VIDEO_FORMATS,
            key="video_50_75",
            help="Third segment of the run"
        )

    with col2:
        st.markdown("#### 🎥 25-50m Range")
        video_files['25-50'] = st.file_uploader(
            "Upload video for 25-50m",
            type=Config.ALLOWED_VIDEO_FORMATS,
            key="video_25_50",
            help="Second segment of the run"
        )

        st.markdown("#### 🎥 75-100m Range")
        video_files['75-100'] = st.file_uploader(
            "Upload video for 75-100m",
            type=Config.ALLOWED_VIDEO_FORMATS,
            key="video_75_100",
            help="Final segment of the run"
        )

    # Show upload status
    uploaded_count = sum(1 for v in video_files.values() if v is not None)
    if uploaded_count > 0:
        st.success(f"✅ {uploaded_count} video(s) uploaded successfully")

    # Analysis button
    if st.button("🚀 Analyze Performance", use_container_width=True, type="primary"):
        if not any(video_files.values()):
            st.error("Please upload at least one video file.")
            return

        with st.spinner("Processing videos with deep learning model..."):
            # Process each video
            data_dict = {}
            video_paths = {}
            progress_bar = st.progress(0)

            for i, (range_key, video_file) in enumerate(video_files.items()):
                if video_file is not None:
                    # Update progress
                    progress_bar.progress((i + 1) / 4, f"Processing {range_key}m range...")

                    # Save video temporarily
                    temp_path = os.path.join(Config.TEMP_DIR, f"{runner_id}_{range_key}_{video_file.name}")
                    with open(temp_path, "wb") as f:
                        f.write(video_file.getbuffer())
                    video_paths[f"video_{range_key.replace('-', '_')}_path"] = temp_path

                    # Process with DL model
                    df = model.process_video(temp_path, range_key)
                    data_dict[range_key] = df
                else:
                    data_dict[range_key] = None

            progress_bar.progress(1.0, "Analysis complete!")

            # Calculate metrics
            metrics = model.calculate_metrics(data_dict)

            # Prepare data for database
            db_data = {
                'runner_id': runner_id,
                'coach_id': st.session_state.coach_id if st.session_state.user_type == 'runner' else st.session_state.user_id,
                'max_speed': metrics['max_speed'],
                'avg_speed': metrics['avg_speed'],
                'total_time': metrics['total_time']
            }

            # Add processed data and video paths
            for range_key, df in data_dict.items():
                if df is not None:
                    db_data[f"range_{range_key.replace('-', '_')}_data"] = df.to_json()

            db_data.update(video_paths)

            # Save to database
            performance_id = db.save_performance_data(db_data)

            st.success("✅ Analysis completed successfully!")

            # Display results
            display_results(data_dict, metrics, runner_name)


def display_results(data_dict, metrics, runner_name):
    """Display analysis results"""
    st.markdown("---")
    st.header("📊 Analysis Results")

    # Performance metrics
    st.markdown("### 🏃 Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Max Speed",
            f"{metrics['max_speed']:.2f} m/s",
            help="Peak speed achieved during the run"
        )
    with col2:
        st.metric(
            "Avg Speed",
            f"{metrics['avg_speed']:.2f} m/s",
            help="Average speed throughout the run"
        )
    with col3:
        st.metric(
            "Min Speed",
            f"{metrics['min_speed']:.2f} m/s",
            help="Minimum speed recorded"
        )
    with col4:
        st.metric(
            "Total Time",
            f"{metrics['total_time']:.2f} s",
            help="Total time for the 100m run"
        )

    # Visualizations
    st.markdown("### 📈 Performance Visualizations")

    # Position vs Speed plot
    fig_pos_speed = Visualizer.create_position_speed_plot(data_dict)
    st.plotly_chart(fig_pos_speed, use_container_width=True)

    # Time series plots
    col1, col2 = st.columns(2)
    fig_pos, fig_vel = Visualizer.create_time_series_plots(data_dict)

    with col1:
        st.plotly_chart(fig_pos, use_container_width=True)

    with col2:
        st.plotly_chart(fig_vel, use_container_width=True)

    # Speed distribution
    fig_dist = Visualizer.create_speed_distribution_plot(data_dict)
    if fig_dist:
        st.plotly_chart(fig_dist, use_container_width=True)

    # Generate Excel report
    excel_data = ReportGenerator.generate_excel_report(data_dict, runner_name, metrics)

    st.download_button(
        label="📥 Download Excel Report",
        data=excel_data,
        file_name=f"running_analysis_{runner_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="Download detailed analysis report in Excel format"
    )