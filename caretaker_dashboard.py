
import streamlit as st
import pandas as pd
import plotly.express as px
import time
import os
import cv2
import numpy as np
from PIL import Image
from emotion_analyzer import EmotionAnalyzer

# Set page config with wide layout and dark theme
st.set_page_config(
    page_title="Caretaker Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed" # Hide sidebar by default for immersion
)

# --- CINEMATIC CSS STYLING ---
st.markdown("""
    <style>
        /* Main Background */
        .stApp {
            background-color: #050510;
            color: #e0e0e0;
            font-family: 'Consolas', 'Courier New', monospace;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #00f2ff !important;
            text-transform: uppercase;
            letter-spacing: 2px;
            text-shadow: 0 0 10px #00f2ff;
        }

        /* Metric Containers (HUD Boxes) */
        div[data-testid="stMetric"] {
            background-color: #111122;
            border: 1px solid #3333aa;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 0 15px rgba(0, 242, 255, 0.1);
        }
        
        div[data-testid="stMetricLabel"] {
            color: #8888aa !important;
        }

        div[data-testid="stMetricValue"] {
            color: #ffffff !important;
            font-size: 2em !important;
        }

        /* Alert/Sticker Box */
        .sticker-box {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border: 2px solid #00f2ff;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 0 20px rgba(0, 242, 255, 0.2);
            margin-bottom: 20px;
        }
        
        .sticker-emoji {
            font-size: 100px;
            margin: 0;
            line-height: 1.2;
            text-shadow: 0 0 20px rgba(255, 255, 255, 0.5);
            animation: pulse 2s infinite;
        }
        
        .status-text {
            font-size: 24px;
            font-weight: bold;
            color: #00f2ff;
            letter-spacing: 3px;
        }

        .alert-text {
            color: #ff0055 !important;
            text-shadow: 0 0 10px #ff0055;
            animation: blink 1s infinite alternate;
        }
        
        /* Animations */
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        @keyframes blink {
            from { opacity: 1; }
            to { opacity: 0.5; }
        }

        /* Chart Containers */
        .stPlotlyChart {
            background-color: #0a0a15;
            border: 1px solid #333;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# Emotion Emoji Map (Shared with Camera Node)
emotion_emojis = {
    "Happy": "😊", "Sad": "😢", "Angry": "😠", "Scared": "😨",
    "Love": "❤️", "Yuck": "🤢", "Wow": "😲", "Ashamed": "😳",
    "Sorry": "😓", "Proud": "😌", "Jealous": "😒", "Crying": "😭",
    "Hopeful": "🤞", "Lonely": "🙍", "Thankful": "🙏", "Worried": "😰",
    "Missing": "🌇", "N/A": "🔌"
}

# Title area
st.markdown("<h1>👁️ CARETAKER <span style='color:white'>COMMAND CENTER</span></h1>", unsafe_allow_html=True)
st.markdown("---")

# Function to load data with caching disabled for real-time feel
def load_data():
    if os.path.exists("alert_log.csv"):
        try:
            df = pd.read_csv("alert_log.csv")
            return df
        except:
            pass
    return pd.DataFrame(columns=["timestamp", "emotion", "confidence", "status"])

# Sidebar for controls (minimal)
with st.sidebar:
    st.header("⚙️ SYSTEM CONTROL")
    auto_refresh = st.checkbox("LIVE FEED SYNC", value=True)
    refresh_rate = st.slider("REFRESH RATE (s)", 0.5, 5.0, 1.0)
    st.info("System Status: OPERATIONAL")
    st.markdown("---")
    st.markdown("### TEST MODEL (Upload)")
    model_path = st.text_input("Model file", value="emotion_classifier.pkl")
    uploaded = st.file_uploader("Upload image to test model", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        try:
            # Read uploaded image and convert to BGR (OpenCV expected input path)
            pil_img = Image.open(uploaded).convert('RGB')
            arr = np.array(pil_img)
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

            analyzer_test = EmotionAnalyzer(classifier_path=model_path)
            test_results = analyzer_test.detect_and_classify(bgr)

            if not test_results:
                st.warning("No faces found in the uploaded image.")
            else:
                st.markdown("**Predictions:**")
                for i, r in enumerate(test_results):
                    lbl = r.get('label', 'Unknown')
                    conf = r.get('confidence', 0.0)
                    st.write(f"Subject {i+1}: {lbl} — {conf:.2%}")

                # Annotate and display image
                for r in test_results:
                    if 'box' in r and r['box']:
                        x, y, w, h = r['box']
                        color = (0, 0, 255) if r.get('is_alert', False) else (0, 255, 0)
                        cv2.rectangle(bgr, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(bgr, f"{r.get('label')}:{r.get('confidence'):.2f}", (x, max(y-10,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                annotated = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                st.image(annotated, use_column_width=True)

        except Exception as e:
            st.error(f"Error running model: {e}")

# Main Dashboard Layout
placeholder = st.empty()

while True:
    df = load_data()
    
    with placeholder.container():
        # Top Detection Stats
        if not df.empty:
            latest = df.iloc[-1]
            last_emotion = latest['emotion']
            last_conf = latest['confidence']
            status = latest['status']
            
            # --- ROW 1: LIVE STATUS & HUD ---
            col_live, col_stats = st.columns([1, 2])
            
            with col_live:
                # CINEMATIC STICKER CONTAINER
                sticker = emotion_emojis.get(last_emotion, "❓")
                
                # Dynamic Alert Styling
                status_class = "status-text"
                border_color = "#00f2ff" # Cyan
                if status == "ALERT":
                    status_class += " alert-text"
                    border_color = "#ff0055" # Red
                
                st.markdown(f"""
                <div class="sticker-box" style="border-color: {border_color}">
                    <div style="color: #888; letter-spacing: 2px;">SUBJECT STATUS</div>
                    <div class="sticker-emoji">{sticker}</div>
                    <div class="{status_class}">{last_emotion.upper()}</div>
                    <div style="margin-top:10px; font-size: 0.8em; color: {border_color}">CONFIDENCE: {last_conf:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_stats:
                # METRICS HUD
                m1, m2, m3 = st.columns(3)
                m1.metric("TOTAL DETECTIONS", len(df))
                m2.metric("ALERTS TRIGGERED", len(df[df['status'] == 'ALERT']))
                
                # Calculate 'Mood Stability' (Just a fun metric for the demo)
                stability = "STABLE" if len(df[df['status'] == 'ALERT'].tail(20)) < 5 else "VOLATILE"
                m3.metric("MOOD STABILITY", stability, delta_color="off")
                
                # RECENT LOG
                st.markdown("### 📝 SYSTEM LOG (LATEST ENTRIES)")
                st.dataframe(df.tail(5)[['timestamp', 'emotion', 'status']], use_container_width=True, hide_index=True)

            # --- ROW 2: ANALYTICS ---
            st.markdown("---")
            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown("### 📊 EMOTIONAL SPECTRUM")
                if not df.empty:
                    fig_pie = px.pie(df, names='emotion', hole=0.6, 
                                     color_discrete_sequence=px.colors.sequential.RdBu)
                    fig_pie.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            with c2:
                st.markdown("### 📈 CONFIDENCE TIMELINE")
                if not df.empty:
                    # Fix timestamp for plotting
                    df_plot = df.copy()
                    df_plot['timestamp'] = pd.to_datetime(df_plot['timestamp'])
                    
                    fig_line = px.line(df_plot.tail(50), x='timestamp', y='confidence', 
                                     color='status',
                                     color_discrete_map={'NORMAL': '#00f2ff', 'ALERT': '#ff0055'})
                    fig_line.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_line, use_container_width=True)

        else:
            # WAITING SCREEN
            st.warning("⚠️ WAITING FOR CAMERA LINK...")
            st.markdown("""
                <div style="text-align:center; padding: 50px;">
                    <h2>INITIALIZING LINK...</h2>
                    <div class="sticker-emoji">📡</div>
                </div>
            """, unsafe_allow_html=True)

    if not auto_refresh:
        break
    time.sleep(refresh_rate)
    st.rerun()
