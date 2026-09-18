"""
AI-Powered Real-Time Object Detection & Tracking System
Object Tracker Engine: High-Performance ByteTrack Implementation
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import deque
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from python_engine.detector import Detection
from python_engine.config import VisionConfig


class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class KalmanFilterState:
    """
    Standard 8-dimensional Kalman filter for bounding box tracking:
    State: [x_center, y_center, aspect_ratio, height, vx, vy, va, vh]
    """
    def __init__(self):
        ndim, dt = 4, 1.0

        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        self._update_mat = np.eye(ndim, 2 * ndim)

        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        projected_mean = np.dot(self._update_mat, mean)
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        projected_cov = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T)) + np.diag(np.square(std))

        kalman_gain = np.linalg.solve(projected_cov, np.dot(self._update_mat, covariance).T).T
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance


def bbox_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Calculate IoU distance matrix between two sets of [x1, y1, x2, y2] boxes."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    boxes_a = np.ascontiguousarray(boxes_a, dtype=float)
    boxes_b = np.ascontiguousarray(boxes_b, dtype=float)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    iw = np.maximum(0, np.minimum(boxes_a[:, 2][:, None], boxes_b[:, 2]) - np.maximum(boxes_a[:, 0][:, None], boxes_b[:, 0]))
    ih = np.maximum(0, np.minimum(boxes_a[:, 3][:, None], boxes_b[:, 3]) - np.maximum(boxes_a[:, 1][:, None], boxes_b[:, 1]))
    intersection = iw * ih

    union = area_a[:, None] + area_b - intersection
    ious = np.clip(intersection / (union + 1e-7), 0.0, 1.0)
    return 1.0 - ious  # Cost matrix (lower is better match)


class STrack:
    """Single Object Track with Kalman Filter and Trajectory History."""
    _count = 0

    def __init__(self, detection: Detection, max_trail_len: int = 40):
        STrack._count += 1
        self.track_id = STrack._count
        self.is_activated = False
        self.state = TrackState.New

        self.class_id = detection.class_id
        self.class_name = detection.class_name
        self.score = detection.score

        # Bbox in tlwh: [x, y, w, h]
        x1, y1, x2, y2 = detection.bbox
        self._tlwh = np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)

        self.kalman_filter = KalmanFilterState()
        self.mean, self.covariance = self.kalman_filter.initiate(self.as_xyah(self._tlwh))

        self.frame_id = 0
        self.tracklet_len = 0
        self.trail: deque = deque(maxlen=max_trail_len)
        self.trail.append(self.center)

    @classmethod
    def reset_counter(cls):
        cls._count = 0

    @staticmethod
    def as_xyah(tlwh: np.ndarray) -> np.ndarray:
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @property
    def tlwh(self) -> np.ndarray:
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self) -> np.ndarray:
        ret = self.tlwh
        ret[2:] += ret[:2]
        return ret

    @property
    def center(self) -> Tuple[float, float]:
        box = self.tlbr
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    def update(self, new_det: Detection, frame_id: int):
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.score = new_det.score

        x1, y1, x2, y2 = new_det.bbox
        new_tlwh = np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.as_xyah(new_tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True
        self.trail.append(self.center)

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed


class ByteTracker:
    """
    ByteTrack Multi-Object Tracker.
    Associates high-confidence detections first, then recovers low-confidence
    occluded/blurred detections in the second association step.
    """
    def __init__(self, config: Optional[VisionConfig] = None):
        self.config = config or VisionConfig()
        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []
        self.frame_id = 0
        self.max_time_lost = self.config.track_buffer

    def reset(self):
        STrack.reset_counter()
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0

    def update(self, detections: List[Detection]) -> List[STrack]:
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # Split detections into High-Score and Low-Score pools
        dets_high = [d for d in detections if d.score >= self.config.track_thresh]
        dets_low = [d for d in detections if d.score < self.config.track_thresh and d.score >= self.config.confidence_threshold * 0.5]

        # Predict current locations for existing active and lost tracks
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        strack_pool = tracked_stracks + self.lost_stracks
        for strack in strack_pool:
            strack.predict()

        # Step 1: Associate High-Score Detections with Track Pool via IoU
        dists = bbox_iou(
            np.array([s.tlbr for s in strack_pool]),
            np.array([d.tlbr for d in dets_high])
        )
        matched_indices = []
        if dists.size > 0:
            row_ind, col_ind = linear_sum_assignment(dists)
            for r, c in zip(row_ind, col_ind):
                if dists[r, c] < (1.0 - self.config.match_thresh):
                    matched_indices.append((r, c))

        unmatched_tracks_1 = list(set(range(len(strack_pool))) - set([r for r, _ in matched_indices]))
        unmatched_dets_high = list(set(range(len(dets_high))) - set([c for _, c in matched_indices]))

        for r, c in matched_indices:
            track = strack_pool[r]
            det = dets_high[c]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.update(det, self.frame_id)
                refind_stracks.append(track)

        # Step 2: Associate Low-Score Detections with remaining Tracked tracks (Occlusion Recovery)
        remain_tracked = [strack_pool[i] for i in unmatched_tracks_1 if strack_pool[i].state == TrackState.Tracked]
        dists_low = bbox_iou(
            np.array([s.tlbr for s in remain_tracked]),
            np.array([d.tlbr for d in dets_low])
        )
        matched_indices_low = []
        if dists_low.size > 0:
            row_ind, col_ind = linear_sum_assignment(dists_low)
            for r, c in zip(row_ind, col_ind):
                if dists_low[r, c] < 0.5:  # Recovery threshold
                    matched_indices_low.append((r, c))

        unmatched_tracks_2 = list(set(range(len(remain_tracked))) - set([r for r, _ in matched_indices_low]))

        for r, c in matched_indices_low:
            track = remain_tracked[r]
            det = dets_low[c]
            track.update(det, self.frame_id)
            activated_stracks.append(track)

        for i in unmatched_tracks_2:
            track = remain_tracked[i]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # Step 3: Handle Unmatched High-Score Detections (Initiate New Tracks)
        for i in unmatched_dets_high:
            det = dets_high[i]
            track = STrack(det, max_trail_len=self.config.trail_length)
            track.is_activated = True
            track.state = TrackState.Tracked
            activated_stracks.append(track)

        # Step 4: Age out lost tracks
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # Update Master Track Lists
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = list(set(self.tracked_stracks + activated_stracks + refind_stracks))

        self.lost_stracks = [t for t in self.lost_stracks if t.state == TrackState.Lost]
        self.lost_stracks = list(set(self.lost_stracks + lost_stracks))
        self.lost_stracks = [t for t in self.lost_stracks if t not in self.tracked_stracks]

        self.removed_stracks.extend(removed_stracks)

        # Return currently active tracks
        return [t for t in self.tracked_stracks if t.is_activated]
