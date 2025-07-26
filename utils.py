# utils.py - Utility functions for data processing and visualization

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from io import BytesIO
import xlsxwriter
from config import Config


class Visualizer:
    """Create visualizations for running performance data"""

    @staticmethod
    def create_position_speed_plot(data_dict):
        """Create position vs speed plot"""
        fig = go.Figure()

        # Combine data from all ranges
        all_x = []
        all_v = []

        for range_key, df in data_dict.items():
            if df is not None:
                # Clean data
                clean_df = df.dropna(subset=['x', 'v'])
                if not clean_df.empty:
                    all_x.extend(clean_df['x'].tolist())
                    all_v.extend(clean_df['v'].tolist())

        if all_x and all_v:
            # Sort by position
            sorted_data = sorted(zip(all_x, all_v))
            x_sorted = [x for x, v in sorted_data]
            v_sorted = [v for x, v in sorted_data]

            # Add trace with smoothing
            fig.add_trace(go.Scatter(
                x=x_sorted,
                y=v_sorted,
                mode='lines',
                line=dict(color=Config.INFO_COLOR, width=3, shape='spline'),
                name='Speed',
                hovertemplate='Position: %{x:.1f}m<br>Speed: %{y:.2f}m/s<extra></extra>'
            ))

        fig.update_layout(
            title=dict(
                text='Relationship between position (m.) and speed (m/s)',
                font=dict(family='Prompt', size=24, color=Config.SECONDARY_COLOR)
            ),
            xaxis=dict(
                title='Position (m)',
                gridcolor='lightgray',
                range=[-20, 120],
                dtick=20,
                showgrid=True
            ),
            yaxis=dict(
                title='Speed (m/s)',
                gridcolor='lightgray',
                range=[0, 10],
                dtick=1,
                showgrid=True
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=500,
            font=dict(family='Prompt'),
            hovermode='x unified'
        )

        return fig

    @staticmethod
    def create_time_series_plots(data_dict):
        """Create time series plots for position and velocity"""
        fig_pos = go.Figure()
        fig_vel = go.Figure()

        colors = [Config.PRIMARY_COLOR, Config.SUCCESS_COLOR, Config.INFO_COLOR, '#9c27b0']

        for i, (range_key, df) in enumerate(data_dict.items()):
            if df is not None:
                # Position plot
                pos_data = df.dropna(subset=['x'])
                if not pos_data.empty:
                    fig_pos.add_trace(go.Scatter(
                        x=pos_data['t'],
                        y=pos_data['x'],
                        mode='markers+lines',
                        name=f'Range {range_key}m',
                        line=dict(color=colors[i % len(colors)], width=2),
                        marker=dict(size=8, color=colors[i % len(colors)]),
                        hovertemplate='Time: %{x:.2f}s<br>Position: %{y:.1f}m<extra></extra>'
                    ))

                # Velocity plot
                vel_data = df.dropna(subset=['v'])
                if not vel_data.empty:
                    fig_vel.add_trace(go.Scatter(
                        x=vel_data['t'],
                        y=vel_data['v'],
                        mode='markers+lines',
                        name=f'Range {range_key}m',
                        line=dict(color=colors[i % len(colors)], width=2),
                        marker=dict(size=8, color=colors[i % len(colors)]),
                        hovertemplate='Time: %{x:.2f}s<br>Speed: %{y:.2f}m/s<extra></extra>'
                    ))

        # Update layouts
        for fig, title, ylabel in [(fig_pos, 'Position over Time', 'Position (m)'),
                                   (fig_vel, 'Velocity over Time', 'Velocity (m/s)')]:
            fig.update_layout(
                title=dict(
                    text=title,
                    font=dict(family='Prompt', size=20, color=Config.SECONDARY_COLOR)
                ),
                xaxis=dict(
                    title='Time (s)',
                    gridcolor='lightgray',
                    showgrid=True
                ),
                yaxis=dict(
                    title=ylabel,
                    gridcolor='lightgray',
                    showgrid=True
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                height=400,
                font=dict(family='Prompt'),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                hovermode='x unified'
            )

        return fig_pos, fig_vel

    @staticmethod
    def create_speed_distribution_plot(data_dict):
        """Create speed distribution histogram"""
        all_speeds = []

        for df in data_dict.values():
            if df is not None:
                speeds = df['v'].dropna()
                if not speeds.empty:
                    all_speeds.extend(speeds.tolist())

        if all_speeds:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=all_speeds,
                nbinsx=20,
                name='Speed Distribution',
                marker_color=Config.PRIMARY_COLOR,
                opacity=0.75
            ))

            fig.update_layout(
                title='Speed Distribution',
                xaxis_title='Speed (m/s)',
                yaxis_title='Frequency',
                plot_bgcolor='white',
                paper_bgcolor='white',
                height=400,
                font=dict(family='Prompt')
            )

            return fig
        return None


class ReportGenerator:
    """Generate Excel reports for running performance"""

    @staticmethod
    def generate_excel_report(data_dict, runner_name, metrics):
        """Generate comprehensive Excel report"""
        output = BytesIO()

        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            workbook = writer.book

            # Define formats
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': Config.EXCEL_HEADER_COLOR,
                'font_color': 'white',
                'align': 'center',
                'valign': 'vcenter',
                'border': 1,
                'font_name': Config.EXCEL_FONT
            })

            data_format = workbook.add_format({
                'align': 'center',
                'valign': 'vcenter',
                'border': 1,
                'font_name': Config.EXCEL_FONT
            })

            title_format = workbook.add_format({
                'bold': True,
                'font_size': 16,
                'font_color': Config.SECONDARY_COLOR,
                'font_name': Config.EXCEL_FONT
            })

            # Summary sheet
            summary_data = {
                'Metric': [
                    'Runner Name',
                    'Test Date',
                    'Max Speed (m/s)',
                    'Average Speed (m/s)',
                    'Min Speed (m/s)',
                    'Speed Std Dev (m/s)',
                    'Total Distance (m)',
                    'Test Duration (s)'
                ],
                'Value': [
                    runner_name,
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    f"{metrics['max_speed']:.2f}",
                    f"{metrics['avg_speed']:.2f}",
                    f"{metrics['min_speed']:.2f}",
                    f"{metrics['speed_std']:.2f}",
                    '100',
                    f"{metrics['total_time']:.2f}"
                ]
            }

            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False, startrow=2)

            # Format summary sheet
            worksheet = writer.sheets['Summary']
            worksheet.write('A1', 'Running Performance Analysis Report', title_format)

            # Write headers and data with formatting
            for col_num, column in enumerate(summary_df.columns):
                worksheet.write(2, col_num, column, header_format)
                for row_num, value in enumerate(summary_df[column]):
                    worksheet.write(row_num + 3, col_num, value, data_format)

            worksheet.set_column('A:A', 25)
            worksheet.set_column('B:B', 30)

            # Data sheets for each range
            for range_key, df in data_dict.items():
                if df is not None:
                    # Clean data for export
                    export_df = df.copy()
                    export_df['t'] = export_df['t'].round(3)
                    export_df['x'] = export_df['x'].round(3)
                    export_df['v'] = export_df['v'].round(3)

                    sheet_name = f'Range_{range_key}m'
                    export_df.to_excel(writer, sheet_name=sheet_name, index=False)

                    worksheet = writer.sheets[sheet_name]

                    # Format headers
                    for col_num, column in enumerate(export_df.columns):
                        worksheet.write(0, col_num, column, header_format)
                        worksheet.set_column(col_num, col_num, 15)

                    # Add chart
                    if not export_df['v'].dropna().empty:
                        chart = workbook.add_chart({'type': 'scatter', 'subtype': 'straight'})

                        # Find last row with data
                        last_row = len(export_df)

                        chart.add_series({
                            'name': 'Speed',
                            'categories': [sheet_name, 1, 0, last_row, 0],  # Time column
                            'values': [sheet_name, 1, 2, last_row, 2],  # Speed column
                            'line': {'color': Config.INFO_COLOR}
                        })

                        chart.set_title({'name': f'Speed Profile - Range {range_key}m'})
                        chart.set_x_axis({'name': 'Time (s)'})
                        chart.set_y_axis({'name': 'Speed (m/s)'})

                        worksheet.insert_chart('E2', chart)

        output.seek(0)
        return output

    @staticmethod
    def format_dataframe_for_display(df):
        """Format DataFrame for display in Streamlit"""
        display_df = df.copy()

        # Format numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if 'id' not in col.lower():
                display_df[col] = display_df[col].round(2)

        # Format datetime columns
        datetime_columns = df.select_dtypes(include=['datetime64']).columns
        for col in datetime_columns:
            display_df[col] = display_df[col].dt.strftime('%Y-%m-%d %H:%M')

        return display_df