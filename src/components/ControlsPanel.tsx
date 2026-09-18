/**
 * AI Vision — Real-Time Object Detection & Tracking
 * Collapsible Configuration Settings Panel
 */

import React, { useState } from 'react';
import {
  Sliders,
  Eye,
  Filter,
  CheckSquare,
  Square,
  ChevronDown,
  ChevronUp,
  Cpu,
  Zap,
} from 'lucide-react';
import { SystemConfig, TrackingMode } from '../types';

interface ControlsPanelProps {
  config: SystemConfig;
  onChangeConfig: (newConfig: SystemConfig) => void;
}

const COMMON_CLASSES = [
  'person',
  'car',
  'truck',
  'bus',
  'motorcycle',
  'bicycle',
  'dog',
  'cat',
  'backpack',
  'cell phone',
  'traffic light',
];

export const ControlsPanel: React.FC<ControlsPanelProps> = ({ config, onChangeConfig }) => {
  const [isCollapsed, setIsCollapsed] = useState<boolean>(false);

  const handleToggleClass = (cls: string) => {
    let updated: string[];
    if (config.selectedClasses.includes(cls)) {
      updated = config.selectedClasses.filter((c) => c !== cls);
    } else {
      updated = [...config.selectedClasses, cls];
    }
    onChangeConfig({ ...config, selectedClasses: updated });
  };

  const handleSelectPreset = (preset: 'all' | 'traffic' | 'pedestrians') => {
    if (preset === 'all') {
      onChangeConfig({ ...config, selectedClasses: [] });
    } else if (preset === 'traffic') {
      onChangeConfig({ ...config, selectedClasses: ['car', 'truck', 'bus', 'motorcycle', 'bicycle'] });
    } else if (preset === 'pedestrians') {
      onChangeConfig({ ...config, selectedClasses: ['person', 'dog', 'bicycle', 'backpack'] });
    }
  };

  return (
    <div id="controls-panel" className="flex flex-col bg-slate-900/90 border border-slate-800 rounded-xl overflow-hidden shadow-lg">
      {/* Collapsible Header */}
      <button
        onClick={() => setIsCollapsed(!isCollapsed)}
        className="w-full flex items-center justify-between p-4 bg-slate-900 hover:bg-slate-850 transition-colors text-left"
      >
        <div className="flex items-center gap-2">
          <Sliders className="w-4 h-4 text-sky-400" />
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
            Settings & Hyperparameters
          </h3>
          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-sky-950/60 border border-sky-800/60 text-sky-400">
            {config.trackingMode}
          </span>
        </div>
        <div className="flex items-center gap-1.5 text-xs text-slate-400 font-mono">
          <span>{isCollapsed ? 'Expand' : 'Collapse'}</span>
          {isCollapsed ? <ChevronDown className="w-4 h-4" /> : <ChevronUp className="w-4 h-4" />}
        </div>
      </button>

      {!isCollapsed && (
        <div className="flex flex-col gap-4 p-4 pt-1 border-t border-slate-800/80">
          {/* Sliders: Detection Confidence, IoU Threshold & Tracking Mode */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
            {/* 1. Detection Confidence */}
            <div className="flex flex-col gap-1.5">
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="text-slate-400">Detection Confidence</span>
                <span className="text-sky-400 font-bold">{Math.round(config.confidenceThreshold * 100)}%</span>
              </div>
              <input
                id="confidence-slider"
                type="range"
                min="0.10"
                max="0.95"
                step="0.05"
                value={config.confidenceThreshold}
                onChange={(e) =>
                  onChangeConfig({ ...config, confidenceThreshold: parseFloat(e.target.value) })
                }
                className="w-full accent-sky-500 cursor-pointer h-1.5 bg-slate-950 rounded-lg"
              />
              <span className="text-[10px] text-slate-500">Filter low-probability object detections</span>
            </div>

            {/* 2. IoU Threshold */}
            <div className="flex flex-col gap-1.5">
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="text-slate-400">IoU Threshold</span>
                <span className="text-purple-400 font-bold">{Math.round(config.iouThreshold * 100)}%</span>
              </div>
              <input
                id="iou-slider"
                type="range"
                min="0.15"
                max="0.85"
                step="0.05"
                value={config.iouThreshold}
                onChange={(e) =>
                  onChangeConfig({ ...config, iouThreshold: parseFloat(e.target.value) })
                }
                className="w-full accent-purple-500 cursor-pointer h-1.5 bg-slate-950 rounded-lg"
              />
              <span className="text-[10px] text-slate-500">Bounding-box overlap matching cutoff</span>
            </div>

            {/* 3. Tracking Mode */}
            <div className="flex flex-col gap-1.5">
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="text-slate-400">Tracking Mode</span>
                <span className="text-emerald-400 font-bold capitalize">
                  {config.trackingMode === 'bytetrack-standard'
                    ? 'ByteTrack Standard'
                    : config.trackingMode === 'bytetrack-high-recall'
                    ? 'ByteTrack High-Recall'
                    : 'Deep SORT'}
                </span>
              </div>
              <select
                id="tracking-mode-select"
                value={config.trackingMode}
                onChange={(e) =>
                  onChangeConfig({ ...config, trackingMode: e.target.value as TrackingMode })
                }
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs font-mono text-slate-200 focus:outline-none focus:border-sky-500"
              >
                <option value="bytetrack-standard">ByteTrack (Standard Kalman)</option>
                <option value="bytetrack-high-recall">ByteTrack (High Recall / Dense Crowds)</option>
                <option value="deepsort">Deep SORT (Appearance Mode)</option>
              </select>
              <span className="text-[10px] text-slate-500">Multi-object association algorithm</span>
            </div>
          </div>

          {/* Toggle Switches: Labels, Confidence, Tracking IDs, Trails, Tripwire */}
          <div className="flex flex-col gap-2 pt-2 border-t border-slate-800">
            <div className="flex items-center gap-2 mb-1">
              <Eye className="w-3.5 h-3.5 text-slate-400" />
              <span className="text-[11px] font-bold uppercase tracking-wider text-slate-300">
                Visual Overlays & Tripwire
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
              {/* Show Labels */}
              <button
                onClick={() => onChangeConfig({ ...config, showLabels: !config.showLabels })}
                className={`flex items-center gap-2 p-2 rounded-lg text-xs font-mono transition-all border ${
                  config.showLabels
                    ? 'bg-slate-800 text-sky-400 border-sky-500/40'
                    : 'bg-slate-950 text-slate-500 border-slate-800 hover:text-slate-300'
                }`}
              >
                {config.showLabels ? <CheckSquare className="w-3.5 h-3.5 text-sky-400" /> : <Square className="w-3.5 h-3.5" />}
                Show Labels
              </button>

              {/* Show Confidence */}
              <button
                onClick={() => onChangeConfig({ ...config, showConfidence: !config.showConfidence })}
                className={`flex items-center gap-2 p-2 rounded-lg text-xs font-mono transition-all border ${
                  config.showConfidence
                    ? 'bg-slate-800 text-emerald-400 border-emerald-500/40'
                    : 'bg-slate-950 text-slate-500 border-slate-800 hover:text-slate-300'
                }`}
              >
                {config.showConfidence ? <CheckSquare className="w-3.5 h-3.5 text-emerald-400" /> : <Square className="w-3.5 h-3.5" />}
                Show Confidence
              </button>

              {/* Show Tracking IDs */}
              <button
                onClick={() => onChangeConfig({ ...config, showTrackingIds: !config.showTrackingIds })}
                className={`flex items-center gap-2 p-2 rounded-lg text-xs font-mono transition-all border ${
                  config.showTrackingIds
                    ? 'bg-slate-800 text-purple-400 border-purple-500/40'
                    : 'bg-slate-950 text-slate-500 border-slate-800 hover:text-slate-300'
                }`}
              >
                {config.showTrackingIds ? <CheckSquare className="w-3.5 h-3.5 text-purple-400" /> : <Square className="w-3.5 h-3.5" />}
                Show Tracking IDs
              </button>

              {/* Show Trails */}
              <button
                onClick={() => onChangeConfig({ ...config, showTrails: !config.showTrails })}
                className={`flex items-center gap-2 p-2 rounded-lg text-xs font-mono transition-all border ${
                  config.showTrails
                    ? 'bg-slate-800 text-amber-400 border-amber-500/40'
                    : 'bg-slate-950 text-slate-500 border-slate-800 hover:text-slate-300'
                }`}
              >
                {config.showTrails ? <CheckSquare className="w-3.5 h-3.5 text-amber-400" /> : <Square className="w-3.5 h-3.5" />}
                Show Trails
              </button>

              {/* Tripwire Enable/Disable */}
              <button
                onClick={() => onChangeConfig({ ...config, showTripwire: !config.showTripwire })}
                className={`flex items-center gap-2 p-2 rounded-lg text-xs font-mono transition-all border ${
                  config.showTripwire
                    ? 'bg-slate-800 text-yellow-400 border-yellow-500/40'
                    : 'bg-slate-950 text-slate-500 border-slate-800 hover:text-slate-300'
                }`}
              >
                {config.showTripwire ? <CheckSquare className="w-3.5 h-3.5 text-yellow-400" /> : <Square className="w-3.5 h-3.5" />}
                Tripwire Line
              </button>

              {/* Bounding Boxes */}
              <button
                onClick={() => onChangeConfig({ ...config, showBoundingBoxes: !config.showBoundingBoxes })}
                className={`flex items-center gap-2 p-2 rounded-lg text-xs font-mono transition-all border ${
                  config.showBoundingBoxes
                    ? 'bg-slate-800 text-teal-400 border-teal-500/40'
                    : 'bg-slate-950 text-slate-500 border-slate-800 hover:text-slate-300'
                }`}
              >
                {config.showBoundingBoxes ? <CheckSquare className="w-3.5 h-3.5 text-teal-400" /> : <Square className="w-3.5 h-3.5" />}
                Bounding Boxes
              </button>
            </div>
          </div>

          {/* Object Class Filters */}
          <div className="flex flex-col gap-2 pt-2 border-t border-slate-800">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Filter className="w-3.5 h-3.5 text-slate-400" />
                <span className="text-[11px] font-bold uppercase tracking-wider text-slate-300">
                  Target Object Classes
                </span>
              </div>

              <div className="flex items-center gap-1 text-[11px] font-mono">
                <button
                  onClick={() => handleSelectPreset('all')}
                  className={`px-2 py-0.5 rounded transition-colors ${
                    config.selectedClasses.length === 0
                      ? 'bg-sky-500 text-slate-950 font-bold'
                      : 'bg-slate-800 text-slate-400 hover:text-slate-200'
                  }`}
                >
                  All 80 COCO
                </button>
                <button
                  onClick={() => handleSelectPreset('traffic')}
                  className="px-2 py-0.5 rounded bg-slate-800 text-slate-400 hover:text-slate-200"
                >
                  Traffic Only
                </button>
                <button
                  onClick={() => handleSelectPreset('pedestrians')}
                  className="px-2 py-0.5 rounded bg-slate-800 text-slate-400 hover:text-slate-200"
                >
                  Pedestrians
                </button>
              </div>
            </div>

            <div className="flex flex-wrap gap-1.5">
              {COMMON_CLASSES.map((cls) => {
                const isSelected =
                  config.selectedClasses.length === 0 || config.selectedClasses.includes(cls);

                return (
                  <button
                    key={cls}
                    onClick={() => handleToggleClass(cls)}
                    className={`px-2.5 py-1 rounded-md text-xs font-mono transition-all border capitalize ${
                      isSelected
                        ? 'bg-slate-800 text-sky-300 border-sky-500/50 shadow-sm'
                        : 'bg-slate-950 text-slate-500 border-slate-800 hover:text-slate-400 opacity-60'
                    }`}
                  >
                    {cls}
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
