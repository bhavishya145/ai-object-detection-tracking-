"""
AI-Powered Real-Time Object Detection & Tracking System
Streamlit Production Web Dashboard Interface
"""

import tempfile
import cv2
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

from python_engine.config import VisionConfig, COCO_CLASSES
from python_engine.detector import YOLODetector
from python_engine.tracker import ByteTracker
from python_engine.analytics import AnalyticsManager
from python_engine.visualizer import Visualizer

# Configure Streamlit Page
st.set_page_config(
    page_title="AI Vision Tracker Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Modern CSS Styling
st.markdown("""
<style>
    .stApp { background-color: #0b0f19; color: #f1f5f9; }
    .metric-card {
        background: #151d2f;
        border: 1px solid #27354f;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .metric-title { font-size: 0.85rem; color: #94a3b8; margin-bottom: 4px; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #38bdf8; }
</style>
""", unsafe_allow_html=True)

st.title("🎯 AI-Powered Real-Time Object Detection & Tracking")
st.caption("Production Computer Vision Engine: Ultralytics YOLOv8 + ByteTrack + Real-Time Analytics")

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Pipeline Configuration")

    input_source = st.radio("Select Video Input:", ["Upload Video File", "Live Webcam", "Sample Video Simulation"])
    
    st.subheader("Model & Weights")
    model_choice = st.selectbox("YOLO Model Architecture:", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"])
    conf_thresh = st.slider("Confidence Threshold:", 0.10, 1.00, 0.40, 0.05)
    iou_thresh = st.slider("NMS IoU Threshold:", 0.10, 1.00, 0.45, 0.05)

    st.subheader("Tracking & Tripwire")
    enable_tripwire = st.checkbox("Enable Line Crossing Detection", value=True)
    show_trails = st.checkbox("Show Motion Trajectory Trails", value=True)
    
    st.subheader("Target Class Filter")
    selected_classes = st.multiselect(
        "Filter Specific Objects:",
        options=COCO_CLASSES,
        default=["person", "car", "motorcycle", "bus", "truck", "bicycle"]
    )
    
    selected_class_ids = [COCO_CLASSES.index(c) for c in selected_classes if c in COCO_CLASSES] if selected_classes else None

# Initialize config
config = VisionConfig(
    model_name=model_choice,
    confidence_threshold=conf_thresh,
    iou_threshold=iou_thresh,
    enable_line_crossing=enable_tripwire,
    show_trails=show_trails,
    target_classes=selected_class_ids
)

# Telemetry KPI row
kpi_cols = st.columns(6)
kpi_fps = kpi_cols[0].empty()
kpi_active = kpi_cols[1].empty()
kpi_unique = kpi_cols[2].empty()
kpi_in = kpi_cols[3].empty()
kpi_out = kpi_cols[4].empty()
kpi_lat = kpi_cols[5].empty()

# Layout: Video Preview (Left) & Analytics (Right)
left_col, right_col = st.columns([7, 5])

with left_col:
    video_placeholder = st.empty()
    run_btn = st.button("🚀 Start Real-Time Processing", type="primary")

with right_col:
    st.subheader("📊 Live Telemetry & Class Distribution")
    chart_placeholder = st.empty()
    st.subheader("📋 Event Log Stream")
    table_placeholder = st.empty()


if run_btn:
    # Resolve Video Source
    video_cap = None
    if input_source == "Upload Video File":
        uploaded_file = st.sidebar.file_uploader("Upload MP4 / MOV / AVI:", type=["mp4", "avi", "mov"])
        if uploaded_file is None:
            st.warning("Please upload a video file in the sidebar to begin.")
            st.stop()
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_cap = cv2.VideoCapture(tfile.name)
    elif input_source == "Live Webcam":
        video_cap = cv2.VideoCapture(0)
    else:
        # Fallback to local sample or synthetic feed
        video_cap = cv2.VideoCapture(0)

    if not video_cap.isOpened():
        st.error("Error connecting to video source. Check file or camera device.")
        st.stop()

    w = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    h = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    config.line_start = (int(w * 0.05), int(h * 0.55))
    config.line_end = (int(w * 0.95), int(h * 0.55))

    detector = YOLODetector(config)
    tracker = ByteTracker(config)
    analytics = AnalyticsManager(config)
    visualizer = Visualizer(config)

    while video_cap.isOpened():
        ret, frame = video_cap.read()
        if not ret:
            st.info("Video playback completed.")
            break

        # Process Pipeline
        detections, latency = detector.detect(frame)
        tracks = tracker.update(detections)
        analytics.process_frame(tracks, latency)
        vis_frame = visualizer.render(frame, tracks, analytics)

        # Convert to RGB for Streamlit
        frame_rgb = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # Update KPI cards
        summary = analytics.get_summary()
        kpi_fps.metric("Current FPS", f"{summary['current_fps']:.1f}")
        kpi_active.metric("Active Tracks", summary['active_objects'])
        kpi_unique.metric("Unique Total", summary['total_unique'])
        kpi_in.metric("Line Cross IN", summary['count_in'])
        kpi_out.metric("Line Cross OUT", summary['count_out'])
        kpi_lat.metric("Latency", f"{summary['inference_ms']:.1f} ms")

        # Update Chart
        if summary['class_active']:
            df_active = pd.DataFrame(list(summary['class_active'].items()), columns=["Class", "Count"])
            chart_placeholder.bar_chart(df_active.set_index("Class"))

        # Update Log Table
        if analytics.event_logs:
            df_logs = pd.DataFrame(list(analytics.event_logs)[-8:])
            table_placeholder.dataframe(df_logs, use_container_width=True, hide_index=True)

    video_cap.release()
