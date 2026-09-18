"""
AI-Powered Real-Time Object Detection & Tracking System
Object Detection Engine (Ultralytics YOLO)
"""

import time
import logging
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

try:
    import torch
    from ultralytics import YOLO
except ImportError:
    torch = None
    YOLO = None

from python_engine.config import VisionConfig, COCO_CLASSES

logger = logging.getLogger("YOLODetector")
logging.basicConfig(level=logging.INFO)


class Detection:
    """Represents a single detected object in a frame."""
    def __init__(self, bbox: np.ndarray, score: float, class_id: int, class_name: str):
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.score = float(score)
        self.class_id = int(class_id)
        self.class_name = class_name

    @property
    def tlbr(self) -> np.ndarray:
        return self.bbox

    @property
    def tlwh(self) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox
        return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class YOLODetector:
    """Production YOLO detector with GPU/CPU auto-selection and batching support."""

    def __init__(self, config: Optional[VisionConfig] = None):
        self.config = config or VisionConfig()
        self.model = None
        self.device = self._resolve_device(self.config.device)
        self._load_model()

    def _resolve_device(self, requested_device: str) -> str:
        """Resolve optimal device with CUDA/MPS fallback to CPU."""
        if requested_device.startswith("cuda"):
            if torch and torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                logger.info(f"Using NVIDIA GPU Acceleration: {device_name}")
                return requested_device
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        elif requested_device == "mps":
            if torch and torch.backends.mps.is_available():
                logger.info("Using Apple Silicon MPS Acceleration.")
                return "mps"
            logger.warning("MPS not available. Falling back to CPU.")
            return "cpu"
        elif requested_device == "auto":
            if torch and torch.cuda.is_available():
                logger.info(f"Auto-selected CUDA GPU: {torch.cuda.get_device_name(0)}")
                return "cuda:0"
            elif torch and torch.backends.mps.is_available():
                logger.info("Auto-selected Apple Silicon MPS.")
                return "mps"
            logger.info("Auto-selected CPU inference mode.")
            return "cpu"
        return "cpu"

    def _load_model(self):
        """Load YOLO model weights safely with error handling."""
        if YOLO is None:
            logger.warning("Ultralytics library not installed in current environment. Running in mock/compatibility mode.")
            return

        try:
            logger.info(f"Loading YOLO model weights: {self.config.model_name} onto {self.device}...")
            self.model = YOLO(self.config.model_name)
            self.model.to(self.device)
            # Warm up
            if self.device.startswith("cuda") and self.config.half_precision:
                logger.info("Enabling FP16 half-precision tensor computation.")
            logger.info("YOLO Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise e

    def detect(self, frame: np.ndarray) -> Tuple[List[Detection], float]:
        """
        Execute object detection on a single RGB/BGR frame.
        
        Returns:
            Tuple of (list of Detection objects, inference latency in ms)
        """
        if self.model is None:
            return [], 0.0

        start_time = time.perf_counter()
        
        # Inference
        results = self.model.predict(
            source=frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            classes=self.config.target_classes,
            imgsz=self.config.imgsz,
            device=self.device,
            verbose=False,
            half=(self.device.startswith("cuda") and self.config.half_precision)
        )
        
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        detections: List[Detection] = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)

            for i in range(len(xyxy)):
                box = xyxy[i]
                score = confs[i]
                cid = clss[i]
                cname = self.model.names.get(cid, COCO_CLASSES[cid] if cid < len(COCO_CLASSES) else f"class_{cid}")
                
                # Area filter check
                width = box[2] - box[0]
                height = box[3] - box[1]
                if width * height < self.config.min_box_area:
                    continue

                detections.append(Detection(bbox=box, score=score, class_id=cid, class_name=cname))

        return detections, latency_ms
