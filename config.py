# config.py - Configuration settings for the application

import os


class Config:
    """Application configuration settings"""

    # App settings
    APP_NAME = "Running Performance Analysis"
    APP_ICON = "🏃"

    # Database settings
    DATABASE_NAME = "running_performance.db"

    # Video settings
    ALLOWED_VIDEO_FORMATS = ['mp4', 'avi', 'mov', 'mkv']
    VIDEO_RANGES = ['0-25', '25-50', '50-75', '75-100']
    MAX_VIDEO_SIZE_MB = 500

    # Performance metrics
    METRICS_SAMPLE_RATE = 0.133  # seconds between measurements

    # Styling
    PRIMARY_COLOR = "#ff6b35"  # Orange
    SECONDARY_COLOR = "#1e3a5f"  # Navy blue
    SUCCESS_COLOR = "#4caf50"  # Green
    ERROR_COLOR = "#f44336"  # Red
    INFO_COLOR = "#2196f3"  # Blue

    # User roles
    USER_ROLES = ['admin', 'coach', 'runner']

    # Excel report settings
    EXCEL_HEADER_COLOR = "#1e3a5f"
    EXCEL_FONT = "Calibri"

    # Temporary file settings
    TEMP_DIR = "temp"

    @staticmethod
    def create_temp_dir():
        """Create temporary directory if it doesn't exist"""
        if not os.path.exists(Config.TEMP_DIR):
            os.makedirs(Config.TEMP_DIR)


# Create temp directory on import
Config.create_temp_dir()