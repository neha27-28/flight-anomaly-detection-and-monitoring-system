# ═══════════════════════════════════════════════════════════════════════════════
# AI-BASED FLIGHT SAFETY DASHBOARD
# Save this as: app.py
# Run with: streamlit run app.py
# ═══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="Flight Safety Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for aviation theme (dark cockpit style)
st.markdown("""
<style>
    .main {
        background-color: #0a0e27;
        color: #00ff00;
    }
    .stButton>button {
        background-color: #1a1f3a;
        color: #00ff00;
        border: 1px solid #00ff00;
    }
    .stButton>button:hover {
        background-color: #00ff00;
        color: #0a0e27;
    }
    h1, h2, h3 {
        color: #00ff00;
    }
    .alert-normal {
        background-color: #1a3d1a;
        padding: 10px;
        border-left: 5px solid #00ff00;
    }
    .alert-warning {
        background-color: #3d3d1a;
        padding: 10px;
        border-left: 5px solid #ffff00;
    }
    .alert-danger {
        background-color: #3d1a1a;
        padding: 10px;
        border-left: 5px solid #ff0000;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center;'>✈️ AI-Based Flight Safety & Pilot Performance Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888;'>Real-time anomaly detection and monitoring system</p>", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model():
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('feature_names.pkl')
    return model, scaler, features

try:
    model, scaler, feature_names = load_model()
    st.sidebar.success("✓ Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop()

# Sidebar
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("""
---
**✈️ Flight Safety ML Demo**  
---
""")

# File upload + preview (NEW)
uploaded_file = st.sidebar.file_uploader("Upload Flight Data CSV", type=['csv'])

# Load data & preview, info for sidebar
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success(f"✓ Loaded {len(df)} records")
    st.sidebar.markdown("**Preview:**")
    st.sidebar.dataframe(df.head(5), use_container_width=True)
else:
    try:
        df = pd.read_csv('ai_flight_safety.csv')
        st.sidebar.info(f"Using default dataset ({len(df)} records)")
        st.sidebar.markdown("**Preview:**")
        st.sidebar.dataframe(df.head(5), use_container_width=True)
    except:
        st.error("No data file found! Upload CSV or place 'ai_flight_safety.csv' in the same folder.")
        st.stop()

# Monitoring speed
monitor_speed = st.sidebar.slider("Monitoring Speed (seconds)", 0.1, 2.0, 0.5, 0.1)

# Auto-adjust slider for record count (NEW)
max_records = min(len(df), 50000)
if max_records <= 1000:
    step = 1
elif max_records <= 10000:
    step = 10
else:
    step = 100
default = min(100, max_records)
num_records = st.sidebar.slider(
    "Records to Monitor",
    min_value=1,
    max_value=max_records,
    value=default,
    step=step,
    help="Number of records to display and monitor (auto-adjusts to your file size!)"
)

# Prepare data
X = df[feature_names].head(num_records)
X_scaled = scaler.transform(X)
predictions = model.predict(X_scaled)
prediction_proba = model.predict_proba(X_scaled)

# Add predictions to dataframe
df_display = df.head(num_records).copy()
df_display['Prediction'] = ['Anomaly' if p == 1 else 'Normal' for p in predictions]
df_display['Confidence'] = [max(p) for p in prediction_proba]

# Main dashboard
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Records", len(df_display))
with col2:
    anomaly_count = sum(predictions)
    st.metric("Anomalies Detected", anomaly_count, delta=f"{anomaly_count/len(predictions)*100:.1f}%")
with col3:
    normal_count = len(predictions) - anomaly_count
    st.metric("Normal Flights", normal_count)
with col4:
    avg_confidence = np.mean([max(p) for p in prediction_proba])
    st.metric("Avg Confidence", f"{avg_confidence:.1%}")

st.markdown("---")

# Real-time monitoring section
st.subheader(" Real-Time Flight Monitoring")

# Create placeholder for live updates
status_placeholder = st.empty()
chart_placeholder = st.empty()
details_placeholder = st.empty()

if st.button("▶️ Start Real-Time Monitoring"):
    for i in range(min(num_records, 50)):  # Limit to 50 for demo
        row = df_display.iloc[i]
        pred = predictions[i]
        confidence = prediction_proba[i]
        
        # Status display
        if pred == 1:
            status_placeholder.markdown(
                f"<div class='alert-danger'><b>⚠️ ANOMALY DETECTED</b> - Record #{i+1}</div>",
                unsafe_allow_html=True
            )
        else:
            status_placeholder.markdown(
                f"<div class='alert-normal'><b>✓ Normal Flight</b> - Record #{i+1}</div>",
                unsafe_allow_html=True
            )
        
        # Display current readings
        details_placeholder.markdown(f"""
        **Flight Parameters (Record #{i+1}):**
        - Speed: {row['speed_knots']:.1f} knots
        - Altitude: {row['altitude_ft']:.0f} ft
        - G-Force: {row['g_force']:.2f}
        - Roll: {row['roll_deg']:.1f}°
        - Vertical Speed: {row['vertical_speed_fpm']:.0f} fpm
        - **Prediction:** {'🔴 ANOMALY' if pred == 1 else '🟢 NORMAL'} (Confidence: {max(confidence):.1%})
        """)
        
        time.sleep(monitor_speed)

st.markdown("---")

# Pilot Performance Summary
st.subheader(" Pilot Performance Summary")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Anomaly Breakdown:**")
    if 'anomaly_type' in df_display.columns:
        anomaly_types = df_display[df_display['Prediction'] == 'Anomaly']['anomaly_type'].value_counts()
        st.dataframe(anomaly_types)
    
    st.markdown("**Critical Metrics:**")
    st.write(f"- Max G-Force: {df_display['g_force'].max():.2f}")
    st.write(f"- Max Speed: {df_display['speed_knots'].max():.1f} knots")
    st.write(f"- Min Altitude: {df_display['altitude_ft'].min():.0f} ft")
    
with col2:
    # Create plotly chart for anomalies over time
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=predictions,
        mode='markers',
        marker=dict(
            color=['red' if p == 1 else 'green' for p in predictions],
            size=8
        ),
        name='Anomalies'
    ))
    fig.update_layout(
        title="Anomaly Detection Over Time",
        yaxis_title="Status (0=Normal, 1=Anomaly)",
        template="plotly_dark",
        height=300
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Data table
st.subheader(" Detailed Flight Data")
MAX_STYLE_CELLS = 250_000  # Below pandas-styler's safe limit
cell_count = df_display.shape[0] * df_display.shape[1]

if cell_count <= MAX_STYLE_CELLS:
    st.dataframe(
        df_display[['speed_knots', 'altitude_ft', 'g_force', 'roll_deg', 'Prediction', 'Confidence']].style.background_gradient(cmap='RdYlGn_r', subset=['Confidence']),
        use_container_width=True
    )
else:
    st.info(f"Table too large for colored styling ({cell_count} cells). Showing plain data for best speed & reliability.")
    st.dataframe(
        df_display[['speed_knots', 'altitude_ft', 'g_force', 'roll_deg', 'Prediction', 'Confidence']],
        use_container_width=True
    )

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #888;'>AI-Based Flight Safety System © 2025</p>", unsafe_allow_html=True)
