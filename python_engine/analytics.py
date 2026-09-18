"""
AI-Powered Real-Time Object Detection & Tracking System
Real-Time Analytics & Virtual Line Crossing Engine
"""

import time
import os
import csv
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Set, Optional
import numpy as np

from python_engine.tracker import STrack
from python_engine.config import VisionConfig


def ccw(A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float]) -> bool:
    """Check if three points are in counterclockwise orientation."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def intersect(p1: Tuple[float, float], p2: Tuple[float, float],
              p3: Tuple[float, float], p4: Tuple[float, float]) -> bool:
    """Return True if line segment p1-p2 and p3-p4 intersect."""
    return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))


class AnalyticsManager:
    """Real-time computer vision metrics, line crossing, and event logger."""

    def __init__(self, config: Optional[VisionConfig] = None):
        self.config = config or VisionConfig()
        
        # Line Crossing Metrics
        self.count_in: int = 0
        self.count_out: int = 0
        self.crossed_ids: Set[int] = set()

        # Cumulative & Active Tracking Records
        self.unique_tracked_ids: Set[int] = set()
        self.class_distribution: Dict[str, int] = defaultdict(int)
        self.class_active_counts: Dict[str, int] = defaultdict(int)

        # Performance Monitoring
        self._fps_history = deque(maxlen=30)
        self._last_time = time.perf_counter()
        self.current_fps: float = 0.0
        self.avg_inference_ms: float = 0.0

        # Detection Event Logs
        self.event_logs = deque(maxlen=500)

    def reset(self):
        """Reset all tracking counters and logs."""
        self.count_in = 0
        self.count_out = 0
        self.crossed_ids.clear()
        self.unique_tracked_ids.clear()
        self.class_distribution.clear()
        self.class_active_counts.clear()
        self.event_logs.clear()
        self._fps_history.clear()
        self._last_time = time.perf_counter()

    def update_fps(self):
        """Compute rolling smoothed FPS."""
        now = time.perf_counter()
        delta = now - self._last_time
        self._last_time = now
        if delta > 0:
            instant_fps = 1.0 / delta
            self._fps_history.append(instant_fps)
            self.current_fps = float(np.mean(self._fps_history))

    def process_frame(self, tracks: List[STrack], inference_ms: float = 0.0):
        """Analyze active tracks for line crossing, class counts, and logs."""
        self.update_fps()
        self.avg_inference_ms = inference_ms

        self.class_active_counts.clear()
        line_p1 = self.config.line_start
        line_p2 = self.config.line_end

        for track in tracks:
            tid = track.track_id
            cname = track.class_name
            self.class_active_counts[cname] += 1

            if tid not in self.unique_tracked_ids:
                self.unique_tracked_ids.add(tid)
                self.class_distribution[cname] += 1
                self._log_event(tid, cname, track.score, "DISCOVERED")

            # Check Virtual Line Crossing
            if self.config.enable_line_crossing and len(track.trail) >= 2 and tid not in self.crossed_ids:
                prev_pos = track.trail[-2]
                curr_pos = track.trail[-1]

                if intersect(prev_pos, curr_pos, line_p1, line_p2):
                    # Determine Crossing Direction via cross-product
                    v_line = (line_p2[0] - line_p1[0], line_p2[1] - line_p1[1])
                    v_move = (curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1])
                    cross_prod = v_line[0] * v_move[1] - v_line[1] * v_move[0]

                    direction = "IN" if cross_prod > 0 else "OUT"
                    if direction == "IN":
                        self.count_in += 1
                    else:
                        self.count_out += 1

                    self.crossed_ids.add(tid)
                    self._log_event(tid, cname, track.score, f"CROSS_{direction}")

    def _log_event(self, track_id: int, class_name: str, confidence: float, action: str):
        event = {
            "timestamp": time.strftime("%H:%M:%S"),
            "track_id": track_id,
            "class_name": class_name,
            "confidence": f"{confidence * 100:.1f}%",
            "action": action
        }
        self.event_logs.append(event)

    def export_csv(self, file_path: str = "detection_logs.csv"):
        """Export logged events to CSV file."""
        if not self.event_logs:
            return
        keys = ["timestamp", "track_id", "class_name", "confidence", "action"]
        with open(file_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(list(self.event_logs))

    def get_summary(self) -> Dict:
        """Return real-time analytics summary."""
        return {
            "current_fps": round(self.current_fps, 1),
            "inference_ms": round(self.avg_inference_ms, 1),
            "active_objects": sum(self.class_active_counts.values()),
            "total_unique": len(self.unique_tracked_ids),
            "count_in": self.count_in,
            "count_out": self.count_out,
            "class_active": dict(self.class_active_counts),
            "class_total": dict(self.class_distribution),
        }
