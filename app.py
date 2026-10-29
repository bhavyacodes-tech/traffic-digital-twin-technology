import os
import time
import urllib.request
import cv2
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from vision_engine import TrafficVisionEngine
from digital_twin import RoadDigitalTwin, generate_sumo_simulation_step

st.set_page_config(
    page_title="Traffic Digital Twin | Smart City Control",
    page_icon="🚦",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background-color: #0B1120; color: #F8FAFC; }
    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.75);
        border: 1px solid rgba(250, 204, 21, 0.35);
        border-radius: 12px;
        padding: 14px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    div[data-testid="stMetricLabel"] { color: #94A3B8 !important; font-weight: 600; }
    div[data-testid="stMetricValue"] { color: #FACC15 !important; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 🚦 AI-Powered Traffic Digital Twin & Congestion Forecasting System")
st.caption("Computer Vision (YOLOv8 + ByteTrack) • Real-Time 2D Schematic Digital Twin • Short-Term ML Traffic Buildup Predictor")

st.sidebar.header("⚙️ Control Room Settings")
mode = st.sidebar.radio(
    "Select Operation Mode:",
    [
        "🎥 CCTV Computer Vision Mode (YOLOv8 + ByteTrack)",
        "🧪 SUMO-Style Traffic Simulation Mode (Synthetic Testing)"
    ]
)

road_segment = st.sidebar.text_input("Monitored Road Segment:", "NH-24 Smart Intersection (3-Lane)")
lane_capacity = st.sidebar.slider("Max Vehicle Capacity per Lane:", 6, 25, 10)
forecast_horizon = st.sidebar.slider("Short-Term Forecast Horizon (Steps):", 5, 20, 10)

SAMPLE_VIDEO_URL = "https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection.mp4"
SAMPLE_VIDEO_PATH = "sample_traffic.mp4"

video_source_path = None
if "CCTV" in mode:
    st.sidebar.subheader("📹 Video Input Source")
    uploaded_file = st.sidebar.file_uploader("Upload Traffic CCTV Video (.mp4, .avi)", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        with open("uploaded_traffic.mp4", "wb") as f:
            f.write(uploaded_file.read())
        video_source_path = "uploaded_traffic.mp4"
    else:
        if not os.path.exists(SAMPLE_VIDEO_PATH):
            if st.sidebar.button("⬇️ Auto-Download Sample CCTV Traffic Video"):
                with st.sidebar.status("Downloading sample traffic video..."):
                    urllib.request.urlretrieve(SAMPLE_VIDEO_URL, SAMPLE_VIDEO_PATH)
                    st.sidebar.success("Sample video ready!")
        if os.path.exists(SAMPLE_VIDEO_PATH):
            video_source_path = SAMPLE_VIDEO_PATH
            st.sidebar.success("Ready: Using `sample_traffic.mp4`")

start_btn = st.sidebar.button("▶️ Start Live Digital Twin Engine", type="primary", use_container_width=True)
stop_btn = st.sidebar.button("⏹️ Stop Engine", use_container_width=True)

if "twin" not in st.session_state:
    st.session_state.twin = RoadDigitalTwin(road_name=road_segment, num_lanes=3, lane_capacity=lane_capacity)

# Top KPI Row
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
metric_active = kpi1.empty()
metric_speed = kpi2.empty()
metric_queue = kpi3.empty()
metric_status = kpi4.empty()

st.markdown("---")

# Main Split View: Left = Camera Feed, Right = Synchronized 2D Digital Twin
col_left, col_right = st.columns(2)
with col_left:
    st.subheader("📹 Live CCTV Computer Vision Feed")
    video_placeholder = st.empty()

with col_right:
    st.subheader("🛣️ Synchronized 2D Road Digital Twin")
    twin_placeholder = st.empty()

st.markdown("---")

# Bottom Analytics & Forecasting Row
col_forecast, col_classes = st.columns([1.4, 1])
with col_forecast:
    st.subheader("📈 Short-Term Traffic Buildup Forecast (Next 5 Mins)")
    forecast_banner = st.empty()
    forecast_chart_placeholder = st.empty()

with col_classes:
    st.subheader("🚗 Vehicle Classification Breakdown")
    class_chart_placeholder = st.empty()

with st.expander("🗄️ View & Download Digital Twin Historical State Database (CSV)", expanded=True):
    table_placeholder = st.empty()


def update_dashboard_ui(twin, current_state, frame_rgb, step_counter):
    """Updates KPIs, Video Feed, and 2D Twin every step; updates heavy Plotly charts smoothly."""
    metric_active.metric(
        "Active Vehicles in Twin",
        f"{current_state['total_active']} veh",
        f"Total Tracked: {current_state['total_unique']}"
    )
    metric_speed.metric(
        "Average Traffic Speed",
        f"{current_state['avg_speed']} km/h",
        f"Occupancy: {current_state['overall_occupancy']}%"
    )
    metric_queue.metric(
        "Total Intersection Queue",
        f"{current_state['total_queue']} veh",
        f"L1: {current_state['lane_1_queue']} | L2: {current_state['lane_2_queue']} | L3: {current_state['lane_3_queue']}"
    )
    metric_status.metric(
        "Digital Twin Road State",
        current_state["congestion_level"],
        f"Updated @ {current_state['timestamp']}"
    )

    # Render both Left Feed and Right Digital Twin as zero-lag synchronized images!
    video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
    twin_canvas_rgb = twin.render_twin_canvas_rgb(current_state, width=840, height=480)
    twin_placeholder.image(twin_canvas_rgb, channels="RGB", use_container_width=True)

    # Update bottom charts every 3rd step so the live feed & twin never lag
    if step_counter == 1 or step_counter % 3 == 0:
        preds, trend_msg = twin.forecast_short_term(horizon_steps=forecast_horizon)
        if "WARNING" in trend_msg:
            forecast_banner.error(f"🚨 **{trend_msg}**")
        elif "MODERATE" in trend_msg:
            forecast_banner.warning(f"⚠️ **{trend_msg}**")
        else:
            forecast_banner.success(f"✅ **{trend_msg}**")

        hist_df = pd.DataFrame(twin.history)
        f_fig = go.Figure()
        f_fig.add_trace(go.Scatter(
            x=list(range(len(hist_df))),
            y=hist_df["total_active"],
            mode="lines+markers",
            name="Observed Active Vehicles",
            line=dict(color="#38BDF8", width=3)
        ))
        future_x = list(range(len(hist_df) - 1, len(hist_df) - 1 + len(preds) + 1))
        future_y = [hist_df["total_active"].iloc[-1]] + preds
        f_fig.add_trace(go.Scatter(
            x=future_x,
            y=future_y,
            mode="lines+markers",
            name="Short-Term Forecast",
            line=dict(color="#FACC15", width=3, dash="dash")
        ))
        f_fig.update_layout(
            plot_bgcolor="#0F172A", paper_bgcolor="#0F172A", font=dict(color="#F8FAFC"),
            height=280, margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Time Step", yaxis_title="Vehicle Count"
        )
        forecast_chart_placeholder.plotly_chart(f_fig, use_container_width=True, key=f"fc_{step_counter}")

        c_fig = go.Figure(data=[
            go.Bar(
                x=["Car", "Motorcycle", "Bus", "Truck"],
                y=[current_state["Car"], current_state["Motorcycle"], current_state["Bus"], current_state["Truck"]],
                marker_color=["#FACC15", "#38BDF8", "#F97316", "#EF4444"]
            )
        ])
        c_fig.update_layout(
            plot_bgcolor="#0F172A", paper_bgcolor="#0F172A", font=dict(color="#F8FAFC"),
            height=280, margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Vehicle Class", yaxis_title="Active Count"
        )
        class_chart_placeholder.plotly_chart(c_fig, use_container_width=True, key=f"cls_{step_counter}")

        display_cols = [
            "timestamp", "total_active", "total_queue", "avg_speed",
            "overall_occupancy", "congestion_level", "lane_1_count", "lane_2_count", "lane_3_count"
        ]
        table_placeholder.dataframe(hist_df[display_cols].tail(10), use_container_width=True)


if start_btn:
    twin = RoadDigitalTwin(road_name=road_segment, num_lanes=3, lane_capacity=lane_capacity)
    st.session_state.twin = twin

    if "CCTV" in mode:
        if not video_source_path or not os.path.exists(video_source_path):
            st.error("Please click **'⬇️ Auto-Download Sample CCTV Traffic Video'** in the sidebar first!")
        else:
            engine = TrafficVisionEngine(model_name="yolov8n.pt", num_lanes=3)
            cap = cv2.VideoCapture(video_source_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

            # Skip the first 95 empty asphalt frames if using sample_traffic.mp4 so cars appear immediately!
            if video_source_path == SAMPLE_VIDEO_PATH:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 95)

            frame_idx = 0
            step_counter = 0
            while cap.isOpened() and not stop_btn:
                ret, frame = cap.read()
                if not ret:
                    # Loop video seamlessly if it reaches the end
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 95 if video_source_path == SAMPLE_VIDEO_PATH else 0)
                    continue

                frame_idx += 1
                if frame_idx % 3 != 0:
                    continue

                step_counter += 1
                frame = cv2.resize(frame, (840, 480))
                annotated_bgr, lane_data, class_counts, total_unique = engine.process_frame(frame, fps=fps)
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

                state = twin.update_twin_state(lane_data, class_counts, total_unique)
                update_dashboard_ui(twin, state, frame_rgb=annotated_rgb, step_counter=step_counter)

            cap.release()
    else:
        for step in range(1, 120):
            if stop_btn:
                break
            lane_data, class_counts, total_unique, sim_frame_rgb = generate_sumo_simulation_step(step, num_lanes=3)
            state = twin.update_twin_state(lane_data, class_counts, total_unique)
            update_dashboard_ui(twin, state, frame_rgb=sim_frame_rgb, step_counter=step)
            time.sleep(0.35)

if len(st.session_state.twin.history) > 0:
    csv_data = pd.DataFrame(st.session_state.twin.history).to_csv(index=False).encode("utf-8")
    st.sidebar.download_button(
        "📥 Download Digital Twin Logs (.CSV)",
        data=csv_data,
        file_name="traffic_digital_twin_logs.csv",
        mime="text/csv",
        use_container_width=True
    )