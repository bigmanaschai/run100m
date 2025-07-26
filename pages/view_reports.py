# pages/view_reports.py - View performance reports page

import streamlit as st
import pandas as pd
import json
from database import db
from models import model
from utils import Visualizer, ReportGenerator
from auth import Auth


@Auth.require_auth
def show():
    """Show performance reports page"""
    st.header("📊 View Performance Reports")

    # Get performance data based on user type
    df = db.get_performance_data(st.session_state.user_id, st.session_state.user_type)

    if df.empty:
        st.info("No performance data available yet. Upload videos to see analysis results.")
        return

    # Filter options
    col1, col2, col3 = st.columns(3)

    with col1:
        # Date range filter
        date_range = st.date_input(
            "Date Range",
            value=(df['test_date'].min(), df['test_date'].max()),
            format="YYYY-MM-DD"
        )

    with col2:
        # Runner filter (for coaches and admins)
        if st.session_state.user_type in ['coach', 'admin']:
            runners = ['All'] + df['runner_name'].unique().tolist()
            selected_runner = st.selectbox("Runner", runners)

    with col3:
        # Sort options
        sort_by = st.selectbox(
            "Sort By",
            ["Date (Newest)", "Date (Oldest)", "Max Speed", "Avg Speed"]
        )

    # Apply filters
    filtered_df = df.copy()

    # Date filter
    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df['test_date'].dt.date >= date_range[0]) &
            (filtered_df['test_date'].dt.date <= date_range[1])
            ]

    # Runner filter
    if st.session_state.user_type in ['coach', 'admin'] and selected_runner != 'All':
        filtered_df = filtered_df[filtered_df['runner_name'] == selected_runner]

    # Apply sorting
    if sort_by == "Date (Newest)":
        filtered_df = filtered_df.sort_values('test_date', ascending=False)
    elif sort_by == "Date (Oldest)":
        filtered_df = filtered_df.sort_values('test_date', ascending=True)
    elif sort_by == "Max Speed":
        filtered_df = filtered_df.sort_values('max_speed', ascending=False)
    elif sort_by == "Avg Speed":
        filtered_df = filtered_df.sort_values('avg_speed', ascending=False)

    # Display summary statistics
    if not filtered_df.empty:
        st.markdown("### 📈 Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Tests",
                len(filtered_df),
                help="Number of performance tests"
            )
        with col2:
            st.metric(
                "Best Speed",
                f"{filtered_df['max_speed'].max():.2f} m/s",
                help="Best maximum speed achieved"
            )
        with col3:
            st.metric(
                "Avg Max Speed",
                f"{filtered_df['max_speed'].mean():.2f} m/s",
                help="Average of maximum speeds"
            )
        with col4:
            st.metric(
                "Improvement",
                f"{(filtered_df['max_speed'].iloc[0] - filtered_df['max_speed'].iloc[-1]):.2f} m/s",
                help="Speed improvement from first to last test"
            )

    # Display individual reports
    st.markdown("### 📋 Performance Reports")

    for idx, row in filtered_df.iterrows():
        # Create expander for each test
        runner_label = f" - {row.get('runner_name', 'Me')}" if st.session_state.user_type != 'runner' else ""
        with st.expander(
                f"📅 {row['test_date'].strftime('%Y-%m-%d %H:%M')}{runner_label} | "
                f"Max Speed: {row['max_speed']:.2f} m/s | Avg Speed: {row['avg_speed']:.2f} m/s"
        ):
            # Reconstruct data from JSON
            data_dict = {}
            for range_key in ['0-25', '25-50', '50-75', '75-100']:
                col_name = f"range_{range_key.replace('-', '_')}_data"
                if pd.notna(row[col_name]):
                    data_dict[range_key] = pd.read_json(row[col_name])
                else:
                    data_dict[range_key] = None

            # Calculate metrics
            metrics = model.calculate_metrics(data_dict)

            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Max Speed", f"{metrics['max_speed']:.2f} m/s")
            with col2:
                st.metric("Avg Speed", f"{metrics['avg_speed']:.2f} m/s")
            with col3:
                st.metric("Total Time", f"{metrics['total_time']:.2f} s")
            with col4:
                st.metric("Speed Std Dev", f"{metrics['speed_std']:.2f} m/s")

            # Display plots
            tab1, tab2, tab3 = st.tabs(["Position vs Speed", "Time Series", "Data Table"])

            with tab1:
                fig_pos_speed = Visualizer.create_position_speed_plot(data_dict)
                st.plotly_chart(fig_pos_speed, use_container_width=True)

            with tab2:
                fig_pos, fig_vel = Visualizer.create_time_series_plots(data_dict)
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig_pos, use_container_width=True)
                with col2:
                    st.plotly_chart(fig_vel, use_container_width=True)

            with tab3:
                # Display raw data tables
                for range_key, df_range in data_dict.items():
                    if df_range is not None:
                        st.markdown(f"**Range {range_key}m**")
                        st.dataframe(
                            df_range.round(3),
                            use_container_width=True,
                            height=200
                        )

            # Generate and offer download
            runner_name = row.get('runner_name', st.session_state.username)
            excel_data = ReportGenerator.generate_excel_report(data_dict, runner_name, metrics)

            col1, col2 = st.columns([1, 4])
            with col1:
                st.download_button(
                    label="📥 Download Report",
                    data=excel_data,
                    file_name=f"report_{runner_name}_{row['test_date'].strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"download_{idx}"
                )

    # Comparison tool
    if len(filtered_df) > 1:
        st.markdown("---")
        st.markdown("### 🔄 Compare Performances")

        col1, col2 = st.columns(2)
        with col1:
            test1_idx = st.selectbox(
                "Select First Test",
                range(len(filtered_df)),
                format_func=lambda x: f"{filtered_df.iloc[x]['test_date'].strftime('%Y-%m-%d %H:%M')} - "
                                      f"Max: {filtered_df.iloc[x]['max_speed']:.2f} m/s"
            )

        with col2:
            test2_idx = st.selectbox(
                "Select Second Test",
                range(len(filtered_df)),
                format_func=lambda x: f"{filtered_df.iloc[x]['test_date'].strftime('%Y-%m-%d %H:%M')} - "
                                      f"Max: {filtered_df.iloc[x]['max_speed']:.2f} m/s",
                index=1 if len(filtered_df) > 1 else 0
            )

        if st.button("Compare", use_container_width=True):
            compare_performances(filtered_df.iloc[test1_idx], filtered_df.iloc[test2_idx])


def compare_performances(test1, test2):
    """Compare two performance tests"""
    st.markdown("#### 📊 Performance Comparison")

    # Show comparison metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Test 1**")
        st.caption(test1['test_date'].strftime('%Y-%m-%d %H:%M'))
        st.metric("Max Speed", f"{test1['max_speed']:.2f} m/s")
        st.metric("Avg Speed", f"{test1['avg_speed']:.2f} m/s")
        st.metric("Total Time", f"{test1['total_time']:.2f} s")

    with col2:
        st.markdown("**Difference**")
        st.caption("Test 2 vs Test 1")
        max_diff = test2['max_speed'] - test1['max_speed']
        avg_diff = test2['avg_speed'] - test1['avg_speed']
        time_diff = test2['total_time'] - test1['total_time']

        st.metric("Max Speed", f"{max_diff:+.2f} m/s", delta_color="normal")
        st.metric("Avg Speed", f"{avg_diff:+.2f} m/s", delta_color="normal")
        st.metric("Total Time", f"{time_diff:+.2f} s", delta_color="inverse")

    with col3:
        st.markdown("**Test 2**")
        st.caption(test2['test_date'].strftime('%Y-%m-%d %H:%M'))
        st.metric("Max Speed", f"{test2['max_speed']:.2f} m/s")
        st.metric("Avg Speed", f"{test2['avg_speed']:.2f} m/s")
        st.metric("Total Time", f"{test2['total_time']:.2f} s")

    # Reconstruct data for both tests
    data_dict1 = {}
    data_dict2 = {}

    for range_key in ['0-25', '25-50', '50-75', '75-100']:
        col_name = f"range_{range_key.replace('-', '_')}_data"

        if pd.notna(test1[col_name]):
            data_dict1[range_key] = pd.read_json(test1[col_name])
        else:
            data_dict1[range_key] = None

        if pd.notna(test2[col_name]):
            data_dict2[range_key] = pd.read_json(test2[col_name])
        else:
            data_dict2[range_key] = None

    # Create comparison plots
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Position vs Speed Comparison", "Speed Over Time Comparison")
    )

    # Position vs Speed comparison
    for data_dict, name, color in [(data_dict1, "Test 1", "#ff6b35"), (data_dict2, "Test 2", "#2196f3")]:
        all_x = []
        all_v = []

        for df in data_dict.values():
            if df is not None:
                clean_df = df.dropna(subset=['x', 'v'])
                if not clean_df.empty:
                    all_x.extend(clean_df['x'].tolist())
                    all_v.extend(clean_df['v'].tolist())

        if all_x and all_v:
            sorted_data = sorted(zip(all_x, all_v))
            x_sorted = [x for x, v in sorted_data]
            v_sorted = [v for x, v in sorted_data]

            fig.add_trace(
                go.Scatter(
                    x=x_sorted,
                    y=v_sorted,
                    mode='lines',
                    name=name,
                    line=dict(color=color, width=2),
                    showlegend=True
                ),
                row=1, col=1
            )

    # Speed over time comparison
    for data_dict, name, color in [(data_dict1, "Test 1", "#ff6b35"), (data_dict2, "Test 2", "#2196f3")]:
        all_t = []
        all_v = []

        for df in data_dict.values():
            if df is not None:
                vel_data = df.dropna(subset=['v'])
                if not vel_data.empty:
                    all_t.extend(vel_data['t'].tolist())
                    all_v.extend(vel_data['v'].tolist())

        if all_t and all_v:
            fig.add_trace(
                go.Scatter(
                    x=all_t,
                    y=all_v,
                    mode='lines+markers',
                    name=name,
                    line=dict(color=color, width=2),
                    marker=dict(size=6),
                    showlegend=False
                ),
                row=1, col=2
            )

    fig.update_xaxes(title_text="Position (m)", row=1, col=1)
    fig.update_yaxes(title_text="Speed (m/s)", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    fig.update_yaxes(title_text="Speed (m/s)", row=1, col=2)

    fig.update_layout(
        height=400,
        showlegend=True,
        legend=dict(x=0.5, y=-0.1, xanchor='center', orientation='h')
    )

    st.plotly_chart(fig, use_container_width=True)