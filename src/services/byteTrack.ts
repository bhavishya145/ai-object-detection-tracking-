/**
 * AI-Powered Real-Time Object Detection & Tracking System
 * ByteTrack Multi-Object Tracker (TypeScript Implementation)
 */

import { DetectionResult, TrackedObject } from '../types';
import { KalmanBoxTracker } from './kalmanFilter';

const PALETTE = [
  '#38bdf8', // Sky Blue
  '#34d399', // Emerald Green
  '#fb923c', // Orange
  '#f472b6', // Pink Rose
  '#a78bfa', // Violet Purple
  '#facc15', // Amber Gold
  '#60a5fa', // Blue
  '#2dd4bf', // Teal
  '#f87171', // Coral Red
  '#c084fc', // Lavender
];

export function getColorForTrackId(id: number): string {
  return PALETTE[id % PALETTE.length];
}

// Calculate Intersection-Over-Union (IoU) between two bounding boxes [x1, y1, x2, y2]
export function calculateIoU(
  boxA: [number, number, number, number],
  boxB: [number, number, number, number]
): number {
  const xA = Math.max(boxA[0], boxB[0]);
  const yA = Math.max(boxA[1], boxB[1]);
  const xB = Math.min(boxA[2], boxB[2]);
  const yB = Math.min(boxA[3], boxB[3]);

  const interWidth = Math.max(0, xB - xA);
  const interHeight = Math.max(0, yB - yA);
  const interArea = interWidth * interHeight;

  if (interArea <= 0) return 0;

  const boxAArea = Math.max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]));
  const boxBArea = Math.max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]));

  return interArea / (boxAArea + boxBArea - interArea);
}

class InternalTrack {
  public id: number;
  public bbox: [number, number, number, number];
  public score: number;
  public className: string;
  public classId: number;
  public kalman: KalmanBoxTracker;
  public trail: Array<[number, number]>;
  public state: 'New' | 'Tracked' | 'Lost' | 'Removed';
  public firstSeenFrame: number;
  public lastSeenFrame: number;
  public timeSinceUpdate: number;
  public hits: number;
  public color: string;

  constructor(
    id: number,
    det: DetectionResult,
    frameId: number,
    maxTrailLen: number = 35
  ) {
    this.id = id;
    this.bbox = [...det.bbox];
    this.score = det.score;
    this.className = det.className;
    this.classId = det.classId;
    this.kalman = new KalmanBoxTracker(det.bbox);
    this.state = 'Tracked';
    this.firstSeenFrame = frameId;
    this.lastSeenFrame = frameId;
    this.timeSinceUpdate = 0;
    this.hits = 1;
    this.color = getColorForTrackId(id);

    const cx = (det.bbox[0] + det.bbox[2]) / 2;
    const cy = (det.bbox[1] + det.bbox[3]) / 2;
    this.trail = [[cx, cy]];
  }

  predict(): [number, number, number, number] {
    this.bbox = this.kalman.predict();
    this.timeSinceUpdate++;
    return this.bbox;
  }

  update(det: DetectionResult, frameId: number, maxTrailLen: number = 35): void {
    this.bbox = [...det.bbox];
    this.score = det.score;
    this.lastSeenFrame = frameId;
    this.timeSinceUpdate = 0;
    this.hits++;
    this.state = 'Tracked';

    this.kalman.update(det.bbox);

    const cx = (det.bbox[0] + det.bbox[2]) / 2;
    const cy = (det.bbox[1] + det.bbox[3]) / 2;
    this.trail.push([cx, cy]);
    if (this.trail.length > maxTrailLen) {
      this.trail.shift();
    }
  }

  toTrackedObject(): TrackedObject {
    return {
      trackId: this.id,
      bbox: [...this.bbox],
      score: this.score,
      className: this.className,
      classId: this.classId,
      trail: [...this.trail],
      state: this.state,
      firstSeenFrame: this.firstSeenFrame,
      lastSeenFrame: this.lastSeenFrame,
      color: this.color,
      velocity: this.kalman.getVelocity(),
    };
  }
}

export class ByteTrackerTS {
  private nextTrackId: number = 1;
  private frameId: number = 0;
  private tracks: InternalTrack[] = [];
  private trackBuffer: number;
  private highThresh: number;
  private matchThresh: number;
  private maxTrailLen: number;

  constructor(
    trackBuffer: number = 30,
    highThresh: number = 0.5,
    matchThresh: number = 0.25,
    maxTrailLen: number = 35
  ) {
    this.trackBuffer = trackBuffer;
    this.highThresh = highThresh;
    this.matchThresh = matchThresh;
    this.maxTrailLen = maxTrailLen;
  }

  public reset(): void {
    this.nextTrackId = 1;
    this.frameId = 0;
    this.tracks = [];
  }

  public update(detections: DetectionResult[]): TrackedObject[] {
    this.frameId++;

    // 1. Predict locations of all current tracks
    for (const track of this.tracks) {
      track.predict();
    }

    // 2. Separate detections into High-Score and Low-Score pools
    const highDets: DetectionResult[] = [];
    const lowDets: DetectionResult[] = [];

    for (const det of detections) {
      if (det.score >= this.highThresh) {
        highDets.push(det);
      } else if (det.score >= 0.15) {
        lowDets.push(det);
      }
    }

    // 3. First Association: High-Score detections with existing active/lost tracks
    const activeTracks = this.tracks.filter((t) => t.state !== 'Removed');
    const matchedTrackIndices = new Set<number>();
    const matchedDetIndices = new Set<number>();

    // Greedy IoU matching (efficient in JS)
    const iouPairs: Array<{ trackIdx: number; detIdx: number; iou: number }> = [];

    for (let t = 0; t < activeTracks.length; t++) {
      for (let d = 0; d < highDets.length; d++) {
        // Class matching check
        if (activeTracks[t].className === highDets[d].className) {
          const iou = calculateIoU(activeTracks[t].bbox, highDets[d].bbox);
          if (iou >= this.matchThresh) {
            iouPairs.push({ trackIdx: t, detIdx: d, iou });
          }
        }
      }
    }

    // Sort descending by IoU
    iouPairs.sort((a, b) => b.iou - a.iou);

    for (const pair of iouPairs) {
      if (!matchedTrackIndices.has(pair.trackIdx) && !matchedDetIndices.has(pair.detIdx)) {
        activeTracks[pair.trackIdx].update(highDets[pair.detIdx], this.frameId, this.maxTrailLen);
        matchedTrackIndices.add(pair.trackIdx);
        matchedDetIndices.add(pair.detIdx);
      }
    }

    // 4. Second Association: Low-score detections with remaining unmatched tracks (Occlusion recovery)
    const unmatchedTracks = activeTracks.filter((_, idx) => !matchedTrackIndices.has(idx));
    const lowIouPairs: Array<{ track: InternalTrack; detIdx: number; iou: number }> = [];

    for (const track of unmatchedTracks) {
      for (let d = 0; d < lowDets.length; d++) {
        if (track.className === lowDets[d].className) {
          const iou = calculateIoU(track.bbox, lowDets[d].bbox);
          if (iou >= 0.2) {
            lowIouPairs.push({ track, detIdx: d, iou });
          }
        }
      }
    }

    lowIouPairs.sort((a, b) => b.iou - a.iou);
    const matchedLowDets = new Set<number>();
    const recoveredTracks = new Set<InternalTrack>();

    for (const pair of lowIouPairs) {
      if (!recoveredTracks.has(pair.track) && !matchedLowDets.has(pair.detIdx)) {
        pair.track.update(lowDets[pair.detIdx], this.frameId, this.maxTrailLen);
        recoveredTracks.add(pair.track);
        matchedLowDets.add(pair.detIdx);
      }
    }

    // 5. Mark remaining unmatched tracks as Lost
    for (const track of unmatchedTracks) {
      if (!recoveredTracks.has(track)) {
        track.state = 'Lost';
      }
    }

    // 6. Initiate new tracks from unmatched high-score detections
    for (let d = 0; d < highDets.length; d++) {
      if (!matchedDetIndices.has(d)) {
        const newTrack = new InternalTrack(
          this.nextTrackId++,
          highDets[d],
          this.frameId,
          this.maxTrailLen
        );
        this.tracks.push(newTrack);
      }
    }

    // 7. Remove stale tracks exceeded buffer threshold
    for (const track of this.tracks) {
      if (track.timeSinceUpdate > this.trackBuffer) {
        track.state = 'Removed';
      }
    }

    this.tracks = this.tracks.filter((t) => t.state !== 'Removed');

    // Return tracks that are currently active or recently observed
    return this.tracks
      .filter((t) => t.state === 'Tracked' || (t.state === 'Lost' && t.timeSinceUpdate <= 3))
      .map((t) => t.toTrackedObject());
  }

  public getNextId(): number {
    return this.nextTrackId;
  }
}
