import streamlit as st
import pandas as pd
import plotly.express as px
import time
import os

# Set page config
st.set_page_config(
    page_title="Caretaker Dashboard",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and Header
st.title("👶 Smart Prescription & Caretaker Assistant")
st.markdown("### Emotion Monitoring Dashboard")

# Function to load data
def load_data():
    if os.path.exists("alert_log.csv"):
        try:
            df = pd.read_csv("alert_log.csv")
            return df
        except:
            return pd.DataFrame(columns=["timestamp", "emotion", "confidence", "status"])
    return pd.DataFrame(columns=["timestamp", "emotion", "confidence", "status"])

# Sidebar
st.sidebar.header("Settings")
auto_refresh = st.sidebar.checkbox("Auto-Refresh Data", value=True)
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 1, 60, 2)

# Main Dashboard Layout
col1, col2 = st.columns(2)

# Placeholder for metrics
metrics_placeholder = st.empty()
charts_placeholder = st.container()

while True:
    df = load_data()
    
    with metrics_placeholder.container():
        # KPI Row
        kpi1, kpi2, kpi3 = st.columns(3)
        
        total_detections = len(df)
        negative_events = len(df[df['status'] == 'ALERT']) if not df.empty else 0
        latest_emotion = df.iloc[-1]['emotion'] if not df.empty else "N/A"
        
        kpi1.metric("Total Detections", total_detections)
        kpi2.metric("Distress Alerts", negative_events, delta_color="inverse")
        kpi3.metric("Last Detected", latest_emotion)

    with charts_placeholder:
        if not df.empty:
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader("Emotion Distribution")
                fig_pie = px.pie(df, names='emotion', title='Detected Emotions Breakdown', hole=0.4)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with c2:
                st.subheader("Timeline of Distress")
                # Filter for timeline
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                fig_line = px.line(df, x='timestamp', y='confidence', color='emotion', 
                                   title='Confidence over Time')
                st.plotly_chart(fig_line, use_container_width=True)
                
            st.subheader("Recent Alerts Log")
            st.dataframe(df[df['status'] == 'ALERT'].tail(10), use_container_width=True)
        else:
            st.info("Waiting for data from Camera Node...")

    if not auto_refresh:
        break
    time.sleep(refresh_rate)
    st.rerun()
