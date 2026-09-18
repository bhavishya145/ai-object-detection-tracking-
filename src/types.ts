/**
 * AI-Powered Real-Time Object Detection & Tracking System
 * Global TypeScript Interfaces and Enums
 */

export interface BoundingBox {
  x: number; // top-left x
  y: number; // top-left y
  width: number;
  height: number;
}

export interface DetectionResult {
  id?: number;
  bbox: [number, number, number, number]; // [x1, y1, x2, y2]
  score: number;
  classId: number;
  className: string;
}

export interface TrackedObject {
  trackId: number;
  bbox: [number, number, number, number]; // [x1, y1, x2, y2]
  score: number;
  className: string;
  classId: number;
  trail: Array<[number, number]>; // center coordinates history
  state: 'New' | 'Tracked' | 'Lost' | 'Removed';
  firstSeenFrame: number;
  lastSeenFrame: number;
  crossedLine?: 'IN' | 'OUT';
  color: string;
  velocity: [number, number]; // vx, vy
}

export interface TripwireLine {
  enabled: boolean;
  p1: { x: number; y: number }; // normalized 0.0 - 1.0 or canvas pixels
  p2: { x: number; y: number };
}

export interface DetectionEventLog {
  id: string;
  timestamp: string;
  trackId: number;
  className: string;
  confidence: number;
  action: 'DISCOVERED' | 'CROSSED_IN' | 'CROSSED_OUT' | 'OCCLUDED' | 'LOST';
  bbox?: [number, number, number, number];
}

export type VideoSourceType =
  | 'webcam'
  | 'demo-video'
  | 'highway-mp4'
  | 'pedestrian-plaza'
  | 'upload';

export type TrackingMode = 'bytetrack-standard' | 'bytetrack-high-recall' | 'deepsort';

export interface SystemConfig {
  confidenceThreshold: number;
  iouThreshold: number;
  trackBuffer: number;
  matchThreshold: number;
  maxTrailLength: number;
  trackingMode: TrackingMode;
  showBoundingBoxes: boolean;
  showLabels: boolean;
  showConfidence: boolean;
  showTrackingIds: boolean;
  showTrails: boolean;
  showTripwire: boolean;
  showHud: boolean;
  selectedClasses: string[];
}

export interface AnalyticsStats {
  currentFps: number;
  avgFps: number;
  inferenceLatencyMs: number;
  activeTracksCount: number;
  totalUniqueTracked: number;
  countIn: number;
  countOut: number;
  avgConfidence: number;
  classCounts: Record<string, number>;
  totalClassCounts: Record<string, number>;
  fpsHistory: number[];
}
