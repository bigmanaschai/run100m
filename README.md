# 🏃 Running Performance Analysis System

A comprehensive Streamlit application for analyzing running performance using deep learning models and video analysis.

## Features

- **Multi-user Authentication**: Support for Admin, Coach, and Runner roles
- **Video Upload & Analysis**: Process videos from 4 camera angles (0-25m, 25-50m, 50-75m, 75-100m)
- **Deep Learning Integration**: Extract running performance metrics from videos
- **Beautiful Visualizations**: Position-speed relationships and time-series plots
- **Excel Reports**: Detailed performance reports with metrics
- **Role-based Access**: Coaches see only their runners, admins see all

## Demo

Access the live application at: [Your Streamlit URL]

### Default Admin Credentials
- Username: `admin`
- Password: `admin123`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/running-performance-app.git
cd running-performance-app
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

## Usage

### For Runners
1. Login with your credentials
2. Upload videos for each running segment
3. View your performance analysis and download reports

### For Coaches
1. Login to see all your runners
2. Upload videos for any of your runners
3. Track performance across multiple sessions

### For Admins
1. Manage all users in the system
2. View all performance data
3. Add new users (coaches and runners)

## Deep Learning Model Integration

To integrate your own deep learning model, modify the `process_video_with_dl_model()` function in `models.py`:

```python
def process_video_with_dl_model(video_path, range_type):
    # Your model inference code here
    # Return DataFrame with columns: t, x, v
    pass
```

## Project Structure

```
├── app.py              # Main application
├── config.py           # Configuration settings
├── database.py         # Database operations
├── auth.py             # Authentication logic
├── models.py           # ML model integration
├── utils.py            # Utility functions
├── pages/              # Page modules
│   ├── upload_analyze.py
│   ├── view_reports.py
│   └── manage_users.py
├── styles/             # Custom styling
│   └── custom.css
└── requirements.txt    # Dependencies
```

## Deployment on Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy!

## Technologies Used

- **Frontend**: Streamlit
- **Visualization**: Plotly
- **Database**: SQLite
- **Data Processing**: Pandas, NumPy
- **Video Processing**: OpenCV
- **Reports**: XlsxWriter

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License.

## Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter) - email@example.com

Project Link: [https://github.com/yourusername/running-performance-app](https://github.com/yourusername/running-performance-app)