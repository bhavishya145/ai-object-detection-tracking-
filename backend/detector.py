"""
Ultralytics YOLOv8/YOLO11 Object Detector Wrapper
Supports CUDA GPU acceleration, Apple MPS, and CPU fallback.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional

try:
    import torch
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


class DetectionResult:
    def __init__(self, bbox: List[float], score: float, class_id: int, class_name: str):
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.score = float(score)
        self.class_id = int(class_id)
        self.class_name = str(class_name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": [round(x, 1) for x in self.bbox],
            "score": round(self.score, 3),
            "classId": self.class_id,
            "className": self.class_name
        }


class YOLODetector:
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.35,
        target_classes: Optional[List[str]] = None
    ):
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = target_classes

        self.device = "cpu"
        self.model = None

        if ULTRALYTICS_AVAILABLE:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"

            print(f"[YOLODetector] Loading {self.model_name} on device: {self.device}")
            try:
                self.model = YOLO(self.model_name)
                self.model.to(self.device)
                self.classes = self.model.names
            except Exception as e:
                print(f"[YOLODetector] Model loading failed ({e}). Enabling fallback vision mode.")
                self.model = None
                self.classes = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
        else:
            self.classes = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """Runs YOLO forward inference on frame and returns bounding boxes."""
        if self.model is None or not ULTRALYTICS_AVAILABLE:
            return []

        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )

        detections: List[DetectionResult] = []
        if not results or len(results) == 0:
            return detections

        r = results[0]
        boxes = r.boxes

        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy().tolist()
            conf = float(boxes.conf[i].cpu().numpy())
            cls_id = int(boxes.cls[i].cpu().numpy())
            cls_name = self.classes.get(cls_id, "unknown")

            if self.target_classes and cls_name not in self.target_classes:
                continue

            detections.append(DetectionResult(
                bbox=xyxy,
                score=conf,
                class_id=cls_id,
                class_name=cls_name
            ))

        return detections
