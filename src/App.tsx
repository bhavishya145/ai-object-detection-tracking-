/**
 * AI-Powered Real-Time Object Detection & Tracking System
 * Main Application Shell & Integration Orchestrator
 */

import React, { useState, useCallback } from 'react';
import {
  Cpu,
  Layers,
  Code,
  ShieldCheck,
  Video,
  Activity,
  Terminal,
  Download,
  Info,
  Sparkles,
} from 'lucide-react';
import { VideoPlayer } from './components/VideoPlayer';
import { AnalyticsDashboard } from './components/AnalyticsDashboard';
import { ControlsPanel } from './components/ControlsPanel';
import { DetectionLogs } from './components/DetectionLogs';
import { PythonEngineModal } from './components/PythonEngineModal';
import {
  SystemConfig,
  AnalyticsStats,
  DetectionEventLog,
  VideoSourceType,
} from './types';

export default function App() {
  // Video & Stream state
  const [sourceType, setSourceType] = useState<VideoSourceType>('demo-video');
  const [resetCounter, setResetCounter] = useState<number>(0);
  const [isPythonModalOpen, setIsPythonModalOpen] = useState<boolean>(false);

  // Configuration settings
  const [config, setConfig] = useState<SystemConfig>({
    confidenceThreshold: 0.4,
    iouThreshold: 0.35,
    trackBuffer: 30,
    matchThreshold: 0.3,
    maxTrailLength: 35,
    trackingMode: 'bytetrack-standard',
    showBoundingBoxes: true,
    showLabels: true,
    showConfidence: true,
    showTrackingIds: true,
    showTrails: true,
    showTripwire: true,
    showHud: true,
    selectedClasses: [], // empty = all
  });

  // Real-time Analytics Statistics
  const [stats, setStats] = useState<AnalyticsStats>({
    currentFps: 60.0,
    avgFps: 60.0,
    inferenceLatencyMs: 12.4,
    activeTracksCount: 0,
    totalUniqueTracked: 0,
    countIn: 0,
    countOut: 0,
    avgConfidence: 93.0,
    classCounts: {},
    totalClassCounts: {},
    fpsHistory: [60, 60, 60, 60, 60],
  });

  // Detection Event Logs Stream
  const [logs, setLogs] = useState<DetectionEventLog[]>([]);

  // Update telemetry stats callback
  const handleUpdateStats = useCallback(
    (newStats: {
      fps: number;
      latencyMs: number;
      activeCount: number;
      uniqueCount: number;
      classCounts: Record<string, number>;
      countIn: number;
      countOut: number;
      avgConfidence: number;
    }) => {
      setStats((prev) => ({
        ...prev,
        currentFps: newStats.fps,
        inferenceLatencyMs: newStats.latencyMs,
        activeTracksCount: newStats.activeCount,
        totalUniqueTracked: newStats.uniqueCount,
        classCounts: newStats.classCounts,
        countIn: newStats.countIn,
        countOut: newStats.countOut,
        avgConfidence: newStats.avgConfidence,
      }));
    },
    []
  );

  // Append new event log
  const handleLogEvent = useCallback(
    (event: {
      trackId: number;
      className: string;
      confidence: number;
      action: 'DISCOVERED' | 'CROSSED_IN' | 'CROSSED_OUT';
    }) => {
      const newEntry: DetectionEventLog = {
        id: `${event.trackId}_${Date.now()}_${Math.random()}`,
        timestamp: new Date().toTimeString().split(' ')[0],
        trackId: event.trackId,
        className: event.className,
        confidence: event.confidence,
        action: event.action,
      };

      setLogs((prev) => [newEntry, ...prev.slice(0, 199)]);
    },
    []
  );

  const handleClearLogs = () => {
    setLogs([]);
  };

  const handleFullReset = () => {
    setResetCounter((prev) => prev + 1);
    setLogs([]);
    setStats((prev) => ({
      ...prev,
      activeTracksCount: 0,
      totalUniqueTracked: 0,
      countIn: 0,
      countOut: 0,
      classCounts: {},
    }));
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-sky-500/30">
      {/* Top Application Navigation Bar */}
      <header className="border-b border-slate-800/80 bg-slate-950/80 backdrop-blur-md sticky top-0 z-30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-3 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-gradient-to-br from-sky-500/20 to-purple-500/20 border border-sky-500/30 text-sky-400">
              <Cpu className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-base sm:text-lg font-extrabold tracking-tight text-white font-mono">
                  AI Vision — Real-Time Object Detection & Tracking
                </h1>
                <span className="text-[10px] font-mono font-bold bg-sky-500/10 text-sky-400 border border-sky-500/30 px-2 py-0.5 rounded-full uppercase">
                  YOLOv8 + ByteTrack
                </span>
              </div>
              <p className="text-xs text-slate-400 hidden sm:block font-mono">
                Real-Time Computer Vision • 8D Kalman State Estimation • Spatial Tripwire Analytics
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2.5">
            {/* Open Python Engine Suite Modal */}
            <button
              id="python-suite-btn"
              onClick={() => setIsPythonModalOpen(true)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold text-sky-300 bg-sky-950/40 hover:bg-sky-900/50 border border-sky-800/70 transition-all shadow-sm"
            >
              <Code className="w-4 h-4 text-sky-400" />
              <span>Python & OpenCV Suite</span>
            </button>
          </div>
        </div>
      </header>

      {/* Main Applet Body Grid */}
      <main className="flex-1 max-w-7xl mx-auto w-full p-4 sm:p-6 flex flex-col gap-6">
        {/* Top KPI Analytics Ribbon */}
        <AnalyticsDashboard stats={stats} />

        {/* 2-Column Responsive Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Column: Live Video Canvas & Settings Panel */}
          <div className="lg:col-span-7 flex flex-col gap-5">
            <VideoPlayer
              sourceType={sourceType}
              setSourceType={setSourceType}
              config={config}
              onUpdateStats={handleUpdateStats}
              onLogEvent={handleLogEvent}
              onResetSignal={resetCounter}
            />

            <ControlsPanel config={config} onChangeConfig={setConfig} />
          </div>

          {/* Right Column: Telemetry Event Logs & System Architecture Breakdown */}
          <div className="lg:col-span-5 flex flex-col gap-5">
            <DetectionLogs logs={logs} onClearLogs={handleClearLogs} />

            {/* Architecture Card */}
            <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 flex flex-col gap-3">
              <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                <div className="flex items-center gap-2">
                  <Layers className="w-4 h-4 text-purple-400" />
                  <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
                    ByteTrack Computer Vision Pipeline
                  </h3>
                </div>
                <span className="text-[10px] font-mono text-slate-400">Production Architecture</span>
              </div>

              <div className="flex flex-col gap-2.5 text-xs font-mono text-slate-300">
                <div className="flex items-start gap-2.5 p-2 rounded-lg bg-slate-950 border border-slate-800/80">
                  <div className="w-5 h-5 rounded-full bg-sky-500/20 text-sky-400 flex items-center justify-center shrink-0 text-[11px] font-bold">
                    1
                  </div>
                  <div>
                    <span className="font-bold text-sky-400">YOLO Object Detection:</span> Ingests raw video frame, predicts bounding boxes, confidence probabilities, and COCO class IDs with GPU/CPU inference.
                  </div>
                </div>

                <div className="flex items-start gap-2.5 p-2 rounded-lg bg-slate-950 border border-slate-800/80">
                  <div className="w-5 h-5 rounded-full bg-purple-500/20 text-purple-400 flex items-center justify-center shrink-0 text-[11px] font-bold">
                    2
                  </div>
                  <div>
                    <span className="font-bold text-purple-400">Kalman Filter Prediction:</span> Forecasts 8D state vector <code className="text-purple-300">[x, y, a, h, vx, vy, va, vh]</code> for each active trajectory.
                  </div>
                </div>

                <div className="flex items-start gap-2.5 p-2 rounded-lg bg-slate-950 border border-slate-800/80">
                  <div className="w-5 h-5 rounded-full bg-emerald-500/20 text-emerald-400 flex items-center justify-center shrink-0 text-[11px] font-bold">
                    3
                  </div>
                  <div>
                    <span className="font-bold text-emerald-400">2-Stage ByteTrack Association:</span> Matches high-score detections first, then matches low-score detections with unmatched tracks to recover occluded objects.
                  </div>
                </div>

                <div className="flex items-start gap-2.5 p-2 rounded-lg bg-slate-950 border border-slate-800/80">
                  <div className="w-5 h-5 rounded-full bg-amber-500/20 text-amber-400 flex items-center justify-center shrink-0 text-[11px] font-bold">
                    4
                  </div>
                  <div>
                    <span className="font-bold text-amber-400">Virtual Spatial Tripwire:</span> Computes 2D vector line segment intersections to trigger IN/OUT directional counting without duplicates.
                  </div>
                </div>
              </div>

              {/* Quick Action to Open Code Viewer */}
              <button
                onClick={() => setIsPythonModalOpen(true)}
                className="mt-1 w-full py-2 px-3 rounded-lg text-xs font-semibold text-slate-950 bg-sky-400 hover:bg-sky-300 transition-colors flex items-center justify-center gap-1.5 shadow-sm"
              >
                <Code className="w-3.5 h-3.5" />
                View Python Codebase & Download Files
              </button>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-slate-950/80 py-4 px-6 text-center text-xs text-slate-500 font-mono">
        AI-Powered Real-Time Object Detection & Tracking System • Ultralytics YOLOv8 & ByteTrack Architecture
      </footer>

      {/* Python Engine Code Viewer & Exporter Modal */}
      <PythonEngineModal
        isOpen={isPythonModalOpen}
        onClose={() => setIsPythonModalOpen(false)}
      />
    </div>
  );
}
