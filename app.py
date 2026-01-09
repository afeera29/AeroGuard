import streamlit as st
import pandas as pd
import xgboost as xgb
import plotly.express as px
from PIL import Image, ImageDraw

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AeroGuard: Digital Twin", layout="wide", page_icon="✈️")

# --- CSS FOR STYLING ---
st.markdown("""
    <style>
    .metric-card {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
    }
    .critical { color: #FF4B4B; font-weight: bold; }
    .warning { color: #FFA500; font-weight: bold; }
    .healthy { color: #00CC96; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD & TRAIN RESOURCES ---
@st.cache_resource
def load_resources():
    # 1. Load Data
    try:
        data = pd.read_csv("aeroguard_demo_data.csv")
    except FileNotFoundError:
        try:
            data = pd.read_csv("processed_test_data.csv")
        except:
            st.error("CRITICAL: No Data File Found! Upload 'aeroguard_demo_data.csv' to GitHub.")
            st.stop()

    # --- CRITICAL FIX: FORCE INTEGERS ---
    # This fixes the "Frozen Slider" issue by ensuring strict integer matching
    data['unit_nr'] = data['unit_nr'].astype(int)
    data['time_cycles'] = data['time_cycles'].astype(int)

    # 2. TRAIN FRESH MODEL (Compatibility Fix)
    X = data.drop(columns=['unit_nr', 'time_cycles', 'RUL'], errors='ignore')
    y = data['RUL']
    
    model = xgb.XGBRegressor()
    model.fit(X, y)
    
    return model, data

# --- MAIN APP LOGIC ---
try:
    model, df = load_resources()
except Exception as e:
    st.error(f"System Error: {e}")
    st.stop()

# --- SIDEBAR: FLIGHT CONTROLLER ---
st.sidebar.title("🎮 Flight Controller")

# 1. Select Engine
engine_ids = sorted(df['unit_nr'].unique())
selected_engine = st.sidebar.selectbox("Select Engine ID", engine_ids)

# 2. Filter Data
engine_data = df[df['unit_nr'] == selected_engine]

# 3. Time Slider
# Use standard int() to be 100% sure
min_cycles = int(engine_data['time_cycles'].min()) 
max_cycles = int(engine_data['time_cycles'].max())

# Unique key ensures slider resets when engine changes
current_cycle = st.sidebar.slider(
    "Flight Cycle (Time)", 
    min_value=min_cycles, 
    max_value=max_cycles, 
    value=min_cycles,
    key=f"slider_{selected_engine}"
)

# Get specific row
# This filtering is now safe because both sides are definitely Integers
current_data = engine_data[engine_data['time_cycles'] == current_cycle]

if current_data.empty:
    st.error(f"Cycle {current_cycle} data not found for Engine {selected_engine}.")
    st.stop()

# --- PREDICTIONS ---
features = current_data.drop(columns=['unit_nr', 'time_cycles', 'RUL'], errors='ignore')
predicted_rul = model.predict(features)[0]

# --- DASHBOARD UI ---
st.title("✈️ AeroGuard: HPC-Aware Digital Twin")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Engine ID", f"#{selected_engine}")
with col2:
    st.metric("Current Cycle", f"{current_cycle}")
with col3:
    st.metric("Predicted RUL", f"{int(predicted_rul)} Cycles")
with col4:
    if predicted_rul < 20:
        st.markdown("### <span class='critical'>CRITICAL FAIL</span>", unsafe_allow_html=True)
    elif predicted_rul < 50:
        st.markdown("### <span class='warning'>MAINTENANCE NEEDED</span>", unsafe_allow_html=True)
    else:
        st.markdown("### <span class='healthy'>HEALTHY</span>", unsafe_allow_html=True)

st.markdown("---")

c_left, c_right = st.columns([1, 1])

with c_left:
    st.subheader("Digital Twin Monitor")
    try:
        img = Image.open("engine.jpg").convert("RGBA")
        overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Use .get() for safety
        temp = current_data.get('s_14_mean', pd.Series([0])).values[0]
        vib = current_data.get('s_11_mean', pd.Series([0])).values[0]

        # Draw Warning Boxes
        if temp > 0.6:  
             draw.rectangle([129, 236, 1759, 722], fill=(255, 0, 0, 100))
        if vib > 0.6:
             draw.rectangle([118, 271, 720, 1460], fill=(255, 165, 0, 100))

        combined = Image.alpha_composite(img, overlay)
        st.image(combined, use_container_width=True, caption="Real-time Thermal & Vibration Overlay")
        
    except FileNotFoundError:
        st.warning("🖼️ 'engine.jpg' not found in GitHub repository.")

with c_right:
    st.subheader("Live Telemetry")
    slope_cols = [c for c in current_data.columns if 'fslope' in c][:5] 
    if slope_cols:
        slope_data = current_data[slope_cols].T
        slope_data.columns = ["Degradation Rate"]
        fig = px.bar(slope_data, x=slope_data.index, y="Degradation Rate", 
                     title="Component Degradation (fslope)",
                     color="Degradation Rate", color_continuous_scale="RdYlGn_r")
        st.plotly_chart(fig, use_container_width=True)

    st.write("Sensor Readings:")
    cols_to_show = ['s_2_mean', 's_14_mean', 's_11_mean', 's_3_mean', 's_4_mean']
    valid_cols = [c for c in cols_to_show if c in current_data.columns]
    st.dataframe(current_data[valid_cols].T)
