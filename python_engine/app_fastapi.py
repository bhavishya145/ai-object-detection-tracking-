"""
AI-Powered Real-Time Object Detection & Tracking System
FastAPI REST & MJPEG Video Streaming Microservice
"""

import cv2
import time
from typing import Optional, Dict
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from python_engine.config import VisionConfig
from python_engine.detector import YOLODetector
from python_engine.tracker import ByteTracker
from python_engine.analytics import AnalyticsManager
from python_engine.visualizer import Visualizer

app = FastAPI(
    title="AI Vision Detection & Tracking Microservice",
    description="Production REST & MJPEG Stream API for YOLOv8 and ByteTrack",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Vision Engine Instances
config = VisionConfig()
detector = YOLODetector(config)
tracker = ByteTracker(config)
analytics = AnalyticsManager(config)
visualizer = Visualizer(config)

camera = None


def get_camera_stream():
    """Generator for live MJPEG video stream."""
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()
        if not success:
            # Reconnect or pause
            time.sleep(0.05)
            continue

        # Pipeline
        detections, latency = detector.detect(frame)
        tracks = tracker.update(detections)
        analytics.process_frame(tracks, latency)
        vis_frame = visualizer.render(frame, tracks, analytics)

        # Encode JPEG frame
        ret, buffer = cv2.imencode(".jpg", vis_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


@app.get("/health")
def health_check():
    return {"status": "online", "device": detector.device, "model": config.model_name}


@app.get("/api/stats")
def get_stats():
    """Return real-time tracking metrics and analytics summary."""
    return analytics.get_summary()


@app.get("/api/events")
def get_recent_events():
    """Return recent line-crossing and tracking events."""
    return {"events": list(analytics.event_logs)}


@app.post("/api/reset")
def reset_tracker():
    """Reset active tracks and counters."""
    tracker.reset()
    analytics.reset()
    return {"message": "Tracker and analytics counters reset successfully."}


@app.get("/video_feed")
def video_feed():
    """MJPEG streaming endpoint for HTML5 video/img tags."""
    return StreamingResponse(
        get_camera_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=8000, reload=False)
