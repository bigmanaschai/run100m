# models.py - Deep learning model integration

import numpy as np
import pandas as pd
from config import Config
import os

# Try to import cv2, but don't fail if it's not available
try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available. Video processing features will be limited.")


class RunningAnalysisModel:
    """Deep learning model for running performance analysis"""

    def __init__(self):
        """Initialize the model"""
        # In production, load your actual deep learning model here
        # self.model = load_model('path/to/model')
        pass

    def process_video(self, video_path, range_type):
        """
        Process video and extract running performance data

        Args:
            video_path: Path to the video file
            range_type: Range of the video (0-25, 25-50, 50-75, 75-100)

        Returns:
            DataFrame with columns: t (time), x (position), v (velocity)
        """
        # In production, replace this with actual model inference
        # For now, we'll generate realistic sample data

        # Determine time range based on video segment
        range_start = int(range_type.split('-')[0])
        range_end = int(range_type.split('-')[1])

        # Calculate approximate time offset
        time_offset = (range_start / 25) * 3.5 if range_start > 0 else 0

        # Generate time series
        num_frames = 50
        time_step = Config.METRICS_SAMPLE_RATE

        t_values = []
        x_values = []
        v_values = []

        for i in range(num_frames):
            t = time_offset + i * time_step
            t_values.append(t)

            # Generate position data (sparse, like in sample files)
            if i % 4 == 0:
                # Calculate position based on range and progress
                progress = i / num_frames
                x = range_start + (range_end - range_start) * progress
                # Add some realistic variation
                x += np.random.normal(0, 0.5)
                x_values.append(x)
            else:
                x_values.append(np.nan)

            # Generate velocity data (sparse, like in sample files)
            if i % 4 == 2:
                # Realistic running speeds (m/s)
                # Start slower, peak in middle ranges, slower at end
                if range_type == "0-25":
                    v_base = 6.0 + progress * 3.0  # Acceleration phase
                elif range_type == "25-50":
                    v_base = 8.5 + np.sin(progress * np.pi) * 0.5  # Peak speed
                elif range_type == "50-75":
                    v_base = 8.3 - progress * 0.3  # Maintaining speed
                else:  # 75-100
                    v_base = 7.5 - progress * 0.5  # Slight deceleration

                v = v_base + np.random.normal(0, 0.2)
                v_values.append(max(0, v))  # Ensure non-negative
            else:
                v_values.append(np.nan)

        # Create DataFrame
        df = pd.DataFrame({
            't': t_values,
            'x': x_values,
            'v': v_values
        })

        return df

    def extract_key_frames(self, video_path, num_frames=5):
        """
        Extract key frames from video for visualization

        Args:
            video_path: Path to the video file
            num_frames: Number of frames to extract

        Returns:
            List of frame images
        """
        frames = []

        if not CV2_AVAILABLE:
            print("OpenCV not available. Skipping frame extraction.")
            return frames

        if not os.path.exists(video_path):
            return frames

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames > 0:
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)

        cap.release()
        return frames

    def calculate_metrics(self, performance_data):
        """
        Calculate performance metrics from extracted data

        Args:
            performance_data: Dictionary of DataFrames for each range

        Returns:
            Dictionary of calculated metrics
        """
        metrics = {
            'max_speed': 0,
            'avg_speed': 0,
            'min_speed': float('inf'),
            'speed_std': 0,
            'total_time': 0,
            'acceleration_phases': [],
            'deceleration_phases': []
        }

        all_speeds = []
        all_times = []

        for range_key, df in performance_data.items():
            if df is not None and not df.empty:
                # Extract valid speed data
                valid_speeds = df['v'].dropna()
                if not valid_speeds.empty:
                    all_speeds.extend(valid_speeds.tolist())

                # Track time
                if not df['t'].empty:
                    all_times.extend(df['t'].tolist())

        if all_speeds:
            metrics['max_speed'] = max(all_speeds)
            metrics['avg_speed'] = np.mean(all_speeds)
            metrics['min_speed'] = min(all_speeds)
            metrics['speed_std'] = np.std(all_speeds)

        if all_times:
            metrics['total_time'] = max(all_times) - min(all_times)

        return metrics


# Global model instance
model = RunningAnalysisModel()