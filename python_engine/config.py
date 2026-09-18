"""
AI-Powered Real-Time Object Detection & Tracking System
Configuration Module
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class VisionConfig:
    """System configuration parameters for YOLO detection, ByteTrack tracking, and video processing."""
    
    # Model Configuration
    model_name: str = "yolov8n.pt"  # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
    confidence_threshold: float = 0.40
    iou_threshold: float = 0.45
    device: str = "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") else "auto"  # 'cuda', 'cpu', or 'mps'
    imgsz: int = 640
    half_precision: bool = True  # FP16 inference on supported GPUs

    # Tracking Configuration (ByteTrack Algorithm)
    track_thresh: float = 0.45       # High confidence detection matching threshold
    match_thresh: float = 0.80       # Metric association threshold (IoU)
    track_buffer: int = 30           # Number of frames to retain lost tracks before dropping
    min_box_area: float = 10.0       # Ignore tiny noise bounding boxes
    frame_rate: int = 30

    # Analytics & Line Crossing
    enable_line_crossing: bool = True
    line_start: Tuple[int, int] = (100, 360)  # (x1, y1)
    line_end: Tuple[int, int] = (1180, 360)   # (x2, y2)
    trail_length: int = 40                   # History points for motion trails

    # Target Class Filter (Empty list = detect all 80 COCO classes)
    # E.g. [0, 1, 2, 3, 5, 7] -> ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']
    target_classes: Optional[List[int]] = None

    # Visual Display Settings
    show_labels: bool = True
    show_conf: bool = True
    show_trails: bool = True
    show_fps: bool = True
    show_counts: bool = True
    line_thickness: int = 2
    font_scale: float = 0.55
    hud_alpha: float = 0.65

    # Storage & Export
    save_output_video: bool = False
    output_video_path: str = "output_tracked.mp4"
    log_csv_path: str = "detection_logs.csv"


# COCO 80 Class Mapping
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
