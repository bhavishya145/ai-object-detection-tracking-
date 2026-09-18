# AI-Powered Real-Time Object Detection & Tracking System

> Production-grade, edge-ready computer vision intelligence platform powered by **Ultralytics YOLOv8**, **ByteTrack Multi-Object Tracking**, **OpenCV**, and an interactive real-time analytics suite.

---

## 🌟 Key Features

1. **Multi-Object Real-Time Detection**:
   - Integrated with Ultralytics YOLOv8/v11 (`yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`).
   - Simultaneous detection across 80 COCO object classes (pedestrians, vehicles, animals, accessories, etc.).
   - Dynamic confidence and Non-Maximum Suppression (NMS) IoU thresholds.

2. **ByteTrack State-of-the-Art Tracking**:
   - Kalman Filter 8D state vector estimation (`[x, y, a, h, vx, vy, va, vh]`).
   - Two-stage association algorithm:
     - **Stage 1**: Matches high-confidence detections with existing tracks.
     - **Stage 2**: Recovers occluded/blurred objects by matching low-confidence detections with remaining tracks.
   - Stable, persistent unique Tracking IDs (`#001`, `#002`, ...).
   - Motion trajectory trail rendering with temporal decay.

3. **Virtual Line-Crossing Tripwire & Counting**:
   - Configurable 2D spatial line crossing boundary.
   - Vector cross-product intersection calculation determining directional flow (`IN` vs `OUT`).
   - Prevents double-counting with unique ID debounce registers.

4. **Real-Time Telemetry & Analytics Dashboard**:
   - Current FPS and rolling smoothed performance metrics.
   - AI model inference latency (milliseconds).
   - Active tracked objects count vs. historical unique IDs seen.
   - Real-time class distribution breakdown.
   - Rolling event logs and one-click CSV export.

5. **Flexible Deployment & Input Options**:
   - Live Webcam streaming (`source 0, 1, 2`).
   - Video file processing (`.mp4`, `.avi`, `.mov`, `.mkv`).
   - RTSP IP Security Camera stream support.
   - Python CLI tool with OpenCV display window.
   - Streamlit interactive browser dashboard (`app_streamlit.py`).
   - FastAPI high-throughput REST + MJPEG streaming microservice (`app_fastapi.py`).

---

## 🏛️ System Architecture

```
[ Input Video Feed ] -> (Webcam / File / RTSP)
          │
          ▼
[ Frame Preprocessor ] -> (Resize 640x640, BGR->RGB, Tensor Normalization)
          │
          ▼
[ YOLOv8 Deep Learning ] -> (CUDA / MPS / CPU Inference, NMS Filtering)
          │   Detections: [x1, y1, x2, y2, conf, class_id]
          ▼
[ ByteTrack Association ]
    ├── Kalman Filter State Prediction
    ├── Stage 1: High-Score Hungarian Association (IoU Distance)
    ├── Stage 2: Low-Score Occlusion Recovery Association
    └── Track Lifecycle Manager (New -> Tracked -> Lost -> Removed)
          │   Active Tracks: [TrackID, BoundingBox, Velocity, Trail]
          ▼
[ Analytics & Tripwire Engine ]
    ├── Ray-Casting Line Intersection (CCW Algorithm)
    ├── Directional Flow (IN / OUT Vector Product)
    ├── Class Counters & ID Registry
    └── Rolling Latency & FPS Estimator
          │
          ▼
[ HUD Visualizer & Output ]
    ├── Anti-Aliased Corner Brackets & Badges
    ├── Glowing Fading Trajectory Paths
    ├── Tripwire Boundary & Direction Indicators
    └── Top Telemetry Heads-Up Display
```

---

## 🚀 Installation & Setup

### 1. Prerequisites
- Python 3.9, 3.10, or 3.11
- NVIDIA GPU with CUDA 11.8+ or 12.0+ (optional, automatically falls back to CPU or Apple Silicon MPS)

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-org/ai-vision-tracking-system.git
cd ai-vision-tracking-system

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Model Download
YOLOv8 automatically downloads weights upon first execution. Alternatively, download manually:
```bash
# Nano (Fastest, 3.2M params)
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt

# Small (Balanced, 11.2M params)
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt
```

---

## 💻 Usage Instructions

### Mode 1: OpenCV High-Performance CLI Pipeline
Run with your default webcam:
```bash
python -m python_engine.main --source 0 --conf 0.40 --device auto
```

Run on an uploaded video file with video recording enabled:
```bash
python -m python_engine.main --source traffic_sample.mp4 --conf 0.45 --save --output tracked_output.mp4
```

**Interactive CLI Keyboard Controls:**
| Key | Action |
|---|---|
| `q` or `ESC` | Exit application safely |
| `p` or `SPACE` | Pause / Resume video processing |
| `s` | Capture high-resolution annotated screenshot |
| `r` | Reset active track IDs and line counters |

---

### Mode 2: Streamlit Interactive Web Dashboard
Launch the web interface in your browser:
```bash
streamlit run python_engine/app_streamlit.py
```
Provides:
- Interactive file uploader for local videos.
- Dynamic sliders for Confidence & IoU thresholds.
- Multi-select dropdown for target object classes.
- Live telemetry KPI cards, distribution charts, and event table.

---

### Mode 3: FastAPI Backend with WebSocket Real-Time Stream
Run the high-performance WebSocket backend that communicates directly with the React dashboard:
```bash
# Launch FastAPI WebSocket server on port 8000
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
Available Endpoints:
- `WS /ws`          : Bidirectional WebSocket stream for real-time frame ingestion, YOLOv8 inference, ByteTrack association, and live telemetry.
- `GET /health`     : Health check with device and model information.
- `GET /api/stats`  : JSON telemetry metrics.
- `POST /api/reset` : Reset tracker state and tripwire counts.
- `POST /api/detect-frame` : Single-frame REST detection.

---

## 📁 Directory Structure
```
ai-vision-tracking-system/
├── backend/                  # Production Python FastAPI & WebSocket Backend
│   ├── main.py               # FastAPI app with /ws WebSocket stream & REST API
│   ├── detector.py           # Ultralytics YOLOv8/YOLO11 detector wrapper
│   ├── tracker.py            # ByteTrack implementation with Kalman Filter
│   ├── analytics.py          # Spatial line crossing math & event registry
│   └── requirements.txt      # Backend Python dependencies
├── python_engine/            # Python CLI & Computer Vision Engine
│   ├── __init__.py
│   ├── config.py             # Global VisionConfig & COCO class mapping
│   ├── detector.py           # YOLOv8 detector with GPU auto-selection
│   ├── tracker.py            # ByteTrack algorithm with 8D Kalman filter
│   ├── analytics.py          # Line-crossing spatial math, FPS, and event logs
│   ├── visualizer.py         # Modern HUD overlays, anti-aliased boxes & trails
│   ├── main.py               # CLI runner with OpenCV window and keyboard handlers
│   ├── app_streamlit.py      # Interactive Streamlit dashboard
│   └── app_fastapi.py        # FastAPI microservice with MJPEG video feed
├── src/                      # Futuristic React & Tailwind Computer Vision Dashboard
│   ├── components/
│   │   ├── VideoPlayer.tsx        # Video canvas, draggable tripwire, HUD, controls
│   │   ├── AnalyticsDashboard.tsx # Real-time statistics, active/unique counts, class breakdown
│   │   ├── ControlsPanel.tsx      # Collapsible settings, confidence/IoU sliders, class filters
│   │   ├── DetectionLogs.tsx      # Real-time event feed terminal & table view with CSV export
│   │   └── PythonEngineModal.tsx  # Code viewer & export modal for Python files
│   ├── services/
│   │   ├── byteTrack.ts           # In-browser ByteTrack Multi-Object Tracker
│   │   ├── cocoDetector.ts        # TensorFlow.js COCO-SSD / MobileNet neural detector
│   │   └── websocketClient.ts     # Bidirectional WebSocket client bridge
│   ├── data/
│   │   └── sampleVideos.ts        # High-definition video simulation scenarios
│   ├── types.ts                   # TypeScript interfaces & configuration schemas
│   ├── App.tsx                    # Main layout and telemetry state synchronizer
│   └── main.tsx                   # React DOM entry point
├── public/
│   └── videos/                    # High-definition MP4 demo videos (Highway & Pedestrian Plaza)
├── requirements.txt          # Python dependencies for deep learning & tracking
├── package.json              # Node.js dependencies & scripts
└── README.md                 # Complete system documentation
```

---

## 🛠️ Performance Optimization Tips

- **GPU Acceleration**: For maximum throughput on NVIDIA GPUs, verify PyTorch with CUDA:
  ```python
  import torch
  print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
  ```
- **Half Precision (FP16)**: Enabled by default on CUDA devices to double frame rates with negligible accuracy loss.
- **Input Resolution**: Adjust `imgsz=640` to `imgsz=480` or `imgsz=320` in `config.py` for ultra-low latency on edge devices like Raspberry Pi 5 or Jetson Nano.
- **Skip Frames**: In high-resolution video streams, run detection every 2nd or 3rd frame and rely on Kalman filter prediction for intermediate frames.
