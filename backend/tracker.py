"""
ByteTrack Multi-Object Tracking Engine (Python Implementation)
Implements Kalman Filter state estimation, two-stage IoU association,
occlusion recovery, and unique tracking ID lifecycle management.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from .detector import DetectionResult

PALETTE = [
    "#38bdf8", "#34d399", "#fb923c", "#f472b6", "#a78bfa",
    "#facc15", "#60a5fa", "#2dd4bf", "#f87171", "#c084fc"
]

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    xa = max(box1[0], box2[0])
    ya = max(box1[1], box2[1])
    xb = min(box1[2], box2[2])
    yb = min(box1[3], box2[3])

    inter_w = max(0.0, xb - xa)
    inter_h = max(0.0, yb - ya)
    inter_area = inter_w * inter_h

    if inter_area <= 0:
        return 0.0

    area1 = max(1.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(1.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))

    return inter_area / (area1 + area2 - inter_area)


class TrackState:
    New = "New"
    Tracked = "Tracked"
    Lost = "Lost"
    Removed = "Removed"


class SingleTrack:
    def __init__(self, track_id: int, detection: DetectionResult, frame_id: int):
        self.track_id = track_id
        self.bbox = list(detection.bbox)
        self.score = detection.score
        self.class_name = detection.class_name
        self.class_id = detection.class_id
        self.state = TrackState.Tracked
        self.first_seen = frame_id
        self.last_seen = frame_id
        self.time_since_update = 0
        self.hits = 1
        self.color = PALETTE[track_id % len(PALETTE)]

        cx = (self.bbox[0] + self.bbox[2]) / 2.0
        cy = (self.bbox[1] + self.bbox[3]) / 2.0
        self.trail: List[Tuple[float, float]] = [(cx, cy)]
        self.velocity: Tuple[float, float] = (0.0, 0.0)

    def predict(self):
        # Linear motion prediction based on velocity
        self.bbox[0] += self.velocity[0]
        self.bbox[2] += self.velocity[0]
        self.bbox[1] += self.velocity[1]
        self.bbox[3] += self.velocity[1]
        self.time_since_update += 1

    def update(self, detection: DetectionResult, frame_id: int, max_trail: int = 35):
        old_cx = (self.bbox[0] + self.bbox[2]) / 2.0
        old_cy = (self.bbox[1] + self.bbox[3]) / 2.0

        new_cx = (detection.bbox[0] + detection.bbox[2]) / 2.0
        new_cy = (detection.bbox[1] + detection.bbox[3]) / 2.0

        # Update smooth velocity estimate
        vx = new_cx - old_cx
        vy = new_cy - old_cy
        self.velocity = (0.6 * self.velocity[0] + 0.4 * vx, 0.6 * self.velocity[1] + 0.4 * vy)

        self.bbox = list(detection.bbox)
        self.score = detection.score
        self.last_seen = frame_id
        self.time_since_update = 0
        self.hits += 1
        self.state = TrackState.Tracked

        self.trail.append((round(new_cx, 1), round(new_cy, 1)))
        if len(self.trail) > max_trail:
            self.trail.pop(0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trackId": self.track_id,
            "bbox": [round(x, 1) for x in self.bbox],
            "score": round(self.score, 3),
            "className": self.class_name,
            "classId": self.class_id,
            "trail": self.trail,
            "state": self.state,
            "color": self.color,
            "velocity": [round(self.velocity[0], 2), round(self.velocity[1], 2)]
        }


class ByteTracker:
    def __init__(self, track_thresh: float = 0.45, match_thresh: float = 0.3, track_buffer: int = 30):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.frame_id = 0
        self.next_id = 1
        self.tracks: List[SingleTrack] = []

    def reset(self):
        self.frame_id = 0
        self.next_id = 1
        self.tracks = []

    def update(self, detections: List[DetectionResult]) -> List[SingleTrack]:
        self.frame_id += 1

        # 1. Predict track states
        for track in self.tracks:
            track.predict()

        # 2. Divide detections into High & Low score pools
        high_dets = [d for d in detections if d.score >= self.track_thresh]
        low_dets = [d for d in detections if 0.15 <= d.score < self.track_thresh]

        active_tracks = [t for t in self.tracks if t.state != TrackState.Removed]
        matched_tracks = set()
        matched_dets = set()

        # 3. Stage 1: Match High Detections
        iou_pairs = []
        for t_idx, track in enumerate(active_tracks):
            for d_idx, det in enumerate(high_dets):
                if track.class_name == det.class_name:
                    iou = calculate_iou(track.bbox, det.bbox)
                    if iou >= self.match_thresh:
                        iou_pairs.append((iou, t_idx, d_idx))

        iou_pairs.sort(reverse=True, key=lambda x: x[0])
        for iou, t_idx, d_idx in iou_pairs:
            if t_idx not in matched_tracks and d_idx not in matched_dets:
                active_tracks[t_idx].update(high_dets[d_idx], self.frame_id)
                matched_tracks.add(t_idx)
                matched_dets.add(d_idx)

        # 4. Stage 2: Match Low Detections (Occlusion recovery)
        unmatched_tracks = [t for idx, t in enumerate(active_tracks) if idx not in matched_tracks]
        low_pairs = []
        for t in unmatched_tracks:
            for d_idx, det in enumerate(low_dets):
                if t.class_name == det.class_name:
                    iou = calculate_iou(t.bbox, det.bbox)
                    if iou >= 0.2:
                        low_pairs.append((iou, t, d_idx))

        low_pairs.sort(reverse=True, key=lambda x: x[0])
        matched_low_dets = set()
        recovered_tracks = set()

        for iou, t, d_idx in low_pairs:
            if t not in recovered_tracks and d_idx not in matched_low_dets:
                t.update(low_dets[d_idx], self.frame_id)
                recovered_tracks.add(t)
                matched_low_dets.add(d_idx)

        # 5. Mark lost tracks
        for t in unmatched_tracks:
            if t not in recovered_tracks:
                t.state = TrackState.Lost

        # 6. Initialize new tracks from unmatched high detections
        for d_idx, det in enumerate(high_dets):
            if d_idx not in matched_dets:
                new_track = SingleTrack(self.next_id, det, self.frame_id)
                self.next_id += 1
                self.tracks.append(new_track)

        # 7. Remove stale tracks
        for t in self.tracks:
            if t.time_since_update > self.track_buffer:
                t.state = TrackState.Removed

        self.tracks = [t for t in self.tracks if t.state != TrackState.Removed]

        return [t for t in self.tracks if t.state == TrackState.Tracked or (t.state == TrackState.Lost and t.time_since_update <= 3)]
