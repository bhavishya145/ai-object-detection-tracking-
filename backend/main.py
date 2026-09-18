"""
AI Vision — Real-Time Object Detection & Tracking
FastAPI Backend with WebSocket Real-Time Inference Stream
Powered by Ultralytics YOLOv8, ByteTrack, OpenCV, and NumPy
"""

import cv2
import json
import time
import base64
import numpy as np
from typing import Dict, Any, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .detector import YOLODetector
from .tracker import ByteTracker
from .analytics import TripwireAnalytics

app = FastAPI(
    title="AI Vision — Real-Time Object Detection & Tracking API",
    description="High-performance computer vision backend with WebSocket & REST streaming",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Core Vision Engine Singletons
detector = YOLODetector(model_name="yolov8n.pt", conf_threshold=0.4, iou_threshold=0.35)
tracker = ByteTracker(track_thresh=0.45, match_thresh=0.3, track_buffer=30)
analytics = TripwireAnalytics(tripwire_ratio=0.58)

@app.get("/")
def root():
    return {
        "title": "AI Vision — Real-Time Object Detection & Tracking API",
        "status": "online",
        "device": detector.device,
        "model": detector.model_name,
        "classes": len(detector.classes)
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "device": detector.device,
        "model": detector.model_name
    }

@app.get("/api/stats")
def get_stats():
    return analytics.get_summary()

@app.post("/api/reset")
def reset_tracker():
    tracker.reset()
    analytics.reset()
    return {"status": "reset_success"}

@app.post("/api/detect-frame")
async def detect_frame_endpoint(file: UploadFile = File(...)):
    """Single frame REST detection endpoint."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image file"})

    start_time = time.perf_counter()
    detections = detector.detect(frame)
    tracks = tracker.update(detections)
    latency_ms = (time.perf_counter() - start_time) * 1000.0
    analytics_data = analytics.process_tracks(tracks, frame.shape[0])

    return {
        "latencyMs": latency_ms,
        "detectionsCount": len(detections),
        "tracks": [t.to_dict() for t in tracks],
        "analytics": analytics_data
    }

@app.websocket("/ws")
@app.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    """
    High-throughput WebSocket endpoint for bidirectional frame streaming.
    Client sends JPEG base64 frames; Server responds with ByteTrack bounding boxes,
    tracking IDs, motion trails, tripwire crossings, and FPS telemetry.
    """
    await websocket.accept()
    print("[WebSocket] Client connected for AI Vision processing")

    last_time = time.perf_counter()
    fps_history = []

    try:
        while True:
            raw_text = await websocket.receive_text()
            data: Dict[str, Any] = json.loads(raw_text)

            msg_type = data.get("type", "frame")

            if msg_type == "ping":
                await websocket.send_text(json.dumps({"type": "pong", "time": time.time()}))
                continue

            if msg_type == "reset":
                tracker.reset()
                analytics.reset()
                await websocket.send_text(json.dumps({"type": "reset_ack"}))
                continue

            if msg_type == "config":
                conf = float(data.get("conf", 0.4))
                iou = float(data.get("iou", 0.35))
                tripwire_ratio = float(data.get("tripwireRatio", 0.58))
                detector.conf_threshold = conf
                detector.iou_threshold = iou
                analytics.tripwire_ratio = tripwire_ratio
                continue

            if msg_type == "frame":
                img_data = data.get("image", "")
                if not img_data:
                    continue

                # Strip base64 header if present
                if "," in img_data:
                    img_data = img_data.split(",", 1)[1]

                img_bytes = base64.b64decode(img_data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is None:
                    continue

                # Measure exact inference latency
                start_infer = time.perf_counter()
                detections = detector.detect(frame)
                tracks = tracker.update(detections)
                latency_ms = (time.perf_counter() - start_infer) * 1000.0

                # Spatial tripwire calculations
                analytics_summary = analytics.process_tracks(tracks, frame.shape[0])

                # FPS Calculation
                now = time.perf_counter()
                delta = now - last_time
                last_time = now
                fps = 1.0 / delta if delta > 0 else 30.0
                fps_history.append(fps)
                if len(fps_history) > 30:
                    fps_history.pop(0)
                smooth_fps = sum(fps_history) / len(fps_history)

                # Send serialized tracking telemetry
                response_payload = {
                    "type": "result",
                    "fps": round(smooth_fps, 1),
                    "latencyMs": round(latency_ms, 1),
                    "activeCount": len(tracks),
                    "uniqueCount": analytics.total_unique_tracked,
                    "countIn": analytics.count_in,
                    "countOut": analytics.count_out,
                    "classCounts": analytics.current_class_counts,
                    "events": analytics.recent_events[-5:],
                    "tracks": [t.to_dict() for t in tracks]
                }

                await websocket.send_text(json.dumps(response_payload))

    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected")
    except Exception as e:
        print(f"[WebSocket] Error during streaming: {e}")
        try:
            await websocket.close()
        except Exception:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
