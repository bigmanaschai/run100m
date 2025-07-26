# app_minimal.py - Minimal working version without complex imports

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3
import hashlib
import json
from io import BytesIO

# Page config
st.set_page_config(
    page_title="Running Performance Analysis",
    page_icon="🏃",
    layout="wide"
)

# Simple CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');
.stApp { font-family: 'Prompt', sans-serif; }
h1, h2, h3 { color: #1e3a5f !important; }
.stButton > button { background-color: #ff6b35; color: white; }
</style>
""", unsafe_allow_html=True)

# Session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.user_id = None
    st.session_state.user_type = None


# Database setup
def init_db():
    conn = sqlite3.connect('running.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (
                     id
                     INTEGER
                     PRIMARY
                     KEY,
                     username
                     TEXT
                     UNIQUE,
                     password
                     TEXT,
                     user_type
                     TEXT
                 )''')
    c.execute('''CREATE TABLE IF NOT EXISTS performance
                 (
                     id
                     INTEGER
                     PRIMARY
                     KEY,
                     user_id
                     INTEGER,
                     test_date
                     TEXT,
                     data
                     TEXT,
                     max_speed
                     REAL
                 )''')
    # Add admin
    try:
        c.execute("INSERT INTO users (username, password, user_type) VALUES (?, ?, ?)",
                  ('admin', hashlib.sha256('admin123'.encode()).hexdigest(), 'admin'))
    except:
        pass
    conn.commit()
    conn.close()


init_db()

# Main app
if not st.session_state.logged_in:
    # Login
    st.title("🏃 Running Performance Analysis")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            if st.form_submit_button("Login"):
                conn = sqlite3.connect('running.db')
                c = conn.cursor()
                pwd_hash = hashlib.sha256(password.encode()).hexdigest()
                c.execute("SELECT id, user_type FROM users WHERE username=? AND password=?",
                          (username, pwd_hash))
                result = c.fetchone()
                conn.close()

                if result:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.user_id = result[0]
                    st.session_state.user_type = result[1]
                    st.rerun()
                else:
                    st.error("Invalid credentials")

            st.info("Default: admin / admin123")
else:
    # Main interface
    with st.sidebar:
        st.write(f"Welcome, {st.session_state.username}!")
        st.write(f"Role: {st.session_state.user_type}")
        if st.button("Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    st.title("🏃 Running Performance Analysis")

    tab1, tab2, tab3 = st.tabs(["Upload", "Reports", "Admin"])

    with tab1:
        st.header("Upload Videos")

        col1, col2 = st.columns(2)
        with col1:
            v1 = st.file_uploader("0-25m", type=['mp4', 'avi'])
            v2 = st.file_uploader("50-75m", type=['mp4', 'avi'])
        with col2:
            v3 = st.file_uploader("25-50m", type=['mp4', 'avi'])
            v4 = st.file_uploader("75-100m", type=['mp4', 'avi'])

        if st.button("Analyze", type="primary", use_container_width=True):
            if any([v1, v2, v3, v4]):
                # Mock analysis
                with st.spinner("Analyzing..."):
                    import time

                    time.sleep(2)

                    # Generate mock data
                    data = {
                        'times': list(range(0, 100, 10)),
                        'speeds': np.random.uniform(6, 9, 10).tolist()
                    }
                    max_speed = max(data['speeds'])

                    # Save to DB
                    conn = sqlite3.connect('running.db')
                    c = conn.cursor()
                    c.execute("INSERT INTO performance (user_id, test_date, data, max_speed) VALUES (?, ?, ?, ?)",
                              (st.session_state.user_id, datetime.now().isoformat(), json.dumps(data), max_speed))
                    conn.commit()
                    conn.close()

                    st.success("Analysis complete!")

                    # Show results
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Max Speed", f"{max_speed:.2f} m/s")
                    col2.metric("Avg Speed", f"{np.mean(data['speeds']):.2f} m/s")
                    col3.metric("Time", "12.5 s")

                    # Simple chart using Streamlit
                    chart_data = pd.DataFrame({
                        'Position (m)': data['times'],
                        'Speed (m/s)': data['speeds']
                    })
                    st.line_chart(chart_data.set_index('Position (m)'))

                    # Excel download
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        summary = pd.DataFrame({
                            'Metric': ['Max Speed', 'Avg Speed', 'Date'],
                            'Value': [f"{max_speed:.2f} m/s",
                                      f"{np.mean(data['speeds']):.2f} m/s",
                                      datetime.now().strftime('%Y-%m-%d')]
                        })
                        summary.to_excel(writer, sheet_name='Summary', index=False)
                        chart_data.to_excel(writer, sheet_name='Data', index=False)

                    output.seek(0)
                    st.download_button(
                        "📥 Download Report",
                        data=output,
                        file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            else:
                st.error("Please upload at least one video")

    with tab2:
        st.header("Performance Reports")

        conn = sqlite3.connect('running.db')
        if st.session_state.user_type == 'admin':
            df = pd.read_sql_query("SELECT * FROM performance ORDER BY test_date DESC", conn)
        else:
            df = pd.read_sql_query("SELECT * FROM performance WHERE user_id=? ORDER BY test_date DESC",
                                   conn, params=(st.session_state.user_id,))
        conn.close()

        if not df.empty:
            for idx, row in df.iterrows():
                with st.expander(f"Test: {row['test_date'][:19]} - Max Speed: {row['max_speed']:.2f} m/s"):
                    data = json.loads(row['data'])
                    st.write(f"Max Speed: {row['max_speed']:.2f} m/s")
                    st.write(f"Avg Speed: {np.mean(data['speeds']):.2f} m/s")

                    # Chart
                    chart_df = pd.DataFrame({
                        'Position': data['times'],
                        'Speed': data['speeds']
                    })
                    st.line_chart(chart_df.set_index('Position'))
        else:
            st.info("No reports available")

    with tab3:
        if st.session_state.user_type == 'admin':
            st.header("User Management")

            # Add user
            with st.form("add_user"):
                st.subheader("Add User")
                new_user = st.text_input("Username")
                new_pass = st.text_input("Password", type="password")
                new_type = st.selectbox("Type", ["runner", "coach", "admin"])

                if st.form_submit_button("Add"):
                    conn = sqlite3.connect('running.db')
                    c = conn.cursor()
                    try:
                        c.execute("INSERT INTO users (username, password, user_type) VALUES (?, ?, ?)",
                                  (new_user, hashlib.sha256(new_pass.encode()).hexdigest(), new_type))
                        conn.commit()
                        st.success(f"Added {new_user}")
                    except:
                        st.error("Username exists")
                    conn.close()

            # List users
            st.subheader("Users")
            conn = sqlite3.connect('running.db')
            users = pd.read_sql_query("SELECT username, user_type FROM users", conn)
            conn.close()
            st.dataframe(users)
        else:
            st.info("Admin only")