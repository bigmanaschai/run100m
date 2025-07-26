# app_debug.py - Debug version to test imports

import streamlit as st

st.title("🏃 Running Performance Analysis - Debug Mode")

st.header("Testing Imports...")

# Test core imports
try:
    import pandas as pd
    st.success("✅ pandas imported successfully")
except Exception as e:
    st.error(f"❌ pandas import failed: {e}")

try:
    import numpy as np
    st.success("✅ numpy imported successfully")
except Exception as e:
    st.error(f"❌ numpy import failed: {e}")

try:
    import plotly
    st.success(f"✅ plotly imported successfully (version: {plotly.__version__})")
except Exception as e:
    st.error(f"❌ plotly import failed: {e}")

try:
    import plotly.graph_objects as go
    st.success("✅ plotly.graph_objects imported successfully")
except Exception as e:
    st.error(f"❌ plotly.graph_objects import failed: {e}")

try:
    import plotly.express as px
    st.success("✅ plotly.express imported successfully")
except Exception as e:
    st.error(f"❌ plotly.express import failed: {e}")

try:
    import xlsxwriter
    st.success("✅ xlsxwriter imported successfully")
except Exception as e:
    st.error(f"❌ xlsxwriter import failed: {e}")

try:
    import sqlite3
    st.success("✅ sqlite3 imported successfully")
except Exception as e:
    st.error(f"❌ sqlite3 import failed: {e}")

# Test a simple plotly chart
st.header("Testing Plotly Chart...")
try:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3, 4], y=[10, 11, 12, 13], name='Test'))
    fig.update_layout(title="Test Plot", xaxis_title="X", yaxis_title="Y")
    st.plotly_chart(fig)
    st.success("✅ Plotly chart rendered successfully")
except Exception as e:
    st.error(f"❌ Plotly chart failed: {e}")

# Show Python version and environment info
import sys
st.header("Environment Info")
st.write(f"Python version: {sys.version}")
st.write(f"Streamlit version: {st.__version__}")

# List all installed packages
st.header("Installed Packages")
try:
    import subprocess
    result = subprocess.run(['pip', 'list'], capture_output=True, text=True)
    st.code(result.stdout)
except:
    st.error("Could not list installed packages")

st.success("Debug test completed!")