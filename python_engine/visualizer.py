"""
AI-Powered Real-Time Object Detection & Tracking System
Computer Vision Visualizer & HUD Renderer
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional

from python_engine.tracker import STrack
from python_engine.analytics import AnalyticsManager
from python_engine.config import VisionConfig

# Distinct High-Contrast Palette for Visual Clarity
PALETTE = [
    (56, 189, 248),   # Sky Blue
    (52, 211, 153),   # Emerald Green
    (251, 146, 60),   # Bright Orange
    (244, 114, 182),  # Pink Rose
    (167, 139, 250),  # Violet Purple
    (250, 204, 21),   # Golden Amber
    (96, 165, 250),   # Indigo
    (45, 212, 191),   # Teal
    (248, 113, 113),  # Coral Red
    (192, 132, 252)   # Lavender
]


def get_color(track_id: int) -> Tuple[int, int, int]:
    """Derive distinct BGR color by tracking ID."""
    return PALETTE[track_id % len(PALETTE)]


class Visualizer:
    """Production HUD renderer for bounding boxes, IDs, trails, and tripwires."""

    def __init__(self, config: Optional[VisionConfig] = None):
        self.config = config or VisionConfig()

    def draw_rounded_rect(self, img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int],
                          color: Tuple[int, int, int], thickness: int = 1, r: int = 6):
        """Draw aesthetic rounded rectangle."""
        x1, y1 = pt1
        x2, y2 = pt2
        r = min(r, abs(x2 - x1) // 2, abs(y2 - y1) // 2)

        # Top/Bottom lines
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
        # Left/Right lines
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
        # 4 Arc corners
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)

    def draw_trails(self, frame: np.ndarray, tracks: List[STrack]):
        """Render fading motion trail paths behind moving objects."""
        overlay = frame.copy()
        for track in tracks:
            trail = list(track.trail)
            if len(trail) < 2:
                continue

            color = get_color(track.track_id)
            total_points = len(trail)

            for i in range(1, total_points):
                pt1 = (int(trail[i - 1][0]), int(trail[i - 1][1]))
                pt2 = (int(trail[i][0]), int(trail[i][1]))
                # Alpha weighting based on point recency
                thickness = max(1, int(3 * (i / total_points)))
                cv2.line(overlay, pt1, pt2, color, thickness, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    def draw_tripwire(self, frame: np.ndarray, analytics: AnalyticsManager):
        """Render virtual line-crossing boundary with directional arrows."""
        p1 = self.config.line_start
        p2 = self.config.line_end

        # Neon line
        cv2.line(frame, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, p1, 5, (0, 200, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, p2, 5, (0, 200, 255), -1, cv2.LINE_AA)

        # Midpoint label badge
        mid_x = (p1[0] + p2[0]) // 2
        mid_y = (p1[1] + p2[1]) // 2
        badge_text = f" TRIPWIRE | IN: {analytics.count_in}  OUT: {analytics.count_out} "
        
        (w, h), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (mid_x - w // 2 - 4, mid_y - h - 6), (mid_x + w // 2 + 4, mid_y + 4), (20, 24, 33), -1)
        cv2.rectangle(frame, (mid_x - w // 2 - 4, mid_y - h - 6), (mid_x + w // 2 + 4, mid_y + 4), (0, 255, 255), 1)
        cv2.putText(frame, badge_text, (mid_x - w // 2, mid_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    def draw_hud(self, frame: np.ndarray, analytics: AnalyticsManager):
        """Draw top telemetry bar (FPS, active tracks, latency)."""
        h, w = frame.shape[:2]
        hud_h = 42

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Border separator
        cv2.line(frame, (0, hud_h), (w, hud_h), (51, 65, 85), 1)

        # Telemetry Texts
        fps_str = f"FPS: {analytics.current_fps:.1f}"
        lat_str = f"LATENCY: {analytics.avg_inference_ms:.1f}ms"
        act_str = f"ACTIVE: {sum(analytics.class_active_counts.values())}"
        tot_str = f"UNIQUE TRACKED: {len(analytics.unique_tracked_ids)}"

        cv2.putText(frame, "AI VISION ENGINE // BYTE-TRACK", (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (241, 245, 249), 1, cv2.LINE_AA)
        cv2.putText(frame, fps_str, (320, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (52, 211, 153), 1, cv2.LINE_AA)
        cv2.putText(frame, lat_str, (440, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (56, 189, 248), 1, cv2.LINE_AA)
        cv2.putText(frame, act_str, (600, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (251, 146, 60), 1, cv2.LINE_AA)
        cv2.putText(frame, tot_str, (720, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (167, 139, 250), 1, cv2.LINE_AA)

    def render(self, frame: np.ndarray, tracks: List[STrack], analytics: AnalyticsManager) -> np.ndarray:
        """Render complete tracking visualization onto the video frame."""
        vis_frame = frame.copy()

        # 1. Motion trails
        if self.config.show_trails:
            self.draw_trails(vis_frame, tracks)

        # 2. Virtual tripwire
        if self.config.enable_line_crossing:
            self.draw_tripwire(vis_frame, analytics)

        # 3. Bounding Boxes and Tracking Tags
        for track in tracks:
            x1, y1, x2, y2 = track.tlbr.astype(int)
            tid = track.track_id
            cname = track.class_name
            color = get_color(tid)

            # Box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, self.config.line_thickness, cv2.LINE_AA)

            # Corner aesthetic brackets
            corner_len = min(16, (x2 - x1) // 4, (y2 - y1) // 4)
            cv2.line(vis_frame, (x1, y1), (x1 + corner_len, y1), color, 3)
            cv2.line(vis_frame, (x1, y1), (x1, y1 + corner_len), color, 3)
            cv2.line(vis_frame, (x2, y1), (x2 - corner_len, y1), color, 3)
            cv2.line(vis_frame, (x2, y1), (x2, y1 + corner_len), color, 3)
            cv2.line(vis_frame, (x1, y2), (x1 + corner_len, y2), color, 3)
            cv2.line(vis_frame, (x1, y2), (x1, y2 - corner_len), color, 3)
            cv2.line(vis_frame, (x2, y2), (x2 - corner_len, y2), color, 3)
            cv2.line(vis_frame, (x2, y2), (x2, y2 - corner_len), color, 3)

            # Center tracking crosshair
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cv2.circle(vis_frame, (cx, cy), 3, color, -1, cv2.LINE_AA)

            # Label badge: "#ID | class | conf%"
            label = f"#{tid:03d} {cname.upper()}"
            if self.config.show_conf:
                label += f" {int(track.score * 100)}%"

            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale, 1)
            badge_y1 = max(0, y1 - th - 8)
            badge_y2 = y1

            cv2.rectangle(vis_frame, (x1, badge_y1), (x1 + tw + 10, badge_y2), color, -1)
            cv2.putText(vis_frame, label, (x1 + 5, badge_y2 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        self.config.font_scale, (15, 23, 42), 1, cv2.LINE_AA)

        # 4. Top Telemetry Bar
        if self.config.show_fps:
            self.draw_hud(vis_frame, analytics)

        return vis_frame
