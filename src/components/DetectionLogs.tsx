/**
 * AI Vision — Real-Time Object Detection & Tracking
 * Real-Time Event Log Terminal Feed & Telemetry Table
 */

import React, { useState } from 'react';
import {
  FileSpreadsheet,
  Terminal,
  Search,
  ArrowDownRight,
  ArrowUpRight,
  Sparkles,
  Trash2,
  Table as TableIcon,
  Code2,
} from 'lucide-react';
import { DetectionEventLog } from '../types';

interface DetectionLogsProps {
  logs: DetectionEventLog[];
  onClearLogs: () => void;
}

export const DetectionLogs: React.FC<DetectionLogsProps> = ({ logs, onClearLogs }) => {
  const [viewMode, setViewMode] = useState<'feed' | 'table'>('feed');
  const [searchTerm, setSearchTerm] = useState('');
  const [actionFilter, setActionFilter] = useState<'ALL' | 'DISCOVERED' | 'CROSSED'>('ALL');

  const filteredLogs = logs.filter((log) => {
    const matchesSearch =
      log.className.toLowerCase().includes(searchTerm.toLowerCase()) ||
      String(log.trackId).includes(searchTerm);

    if (actionFilter === 'ALL') return matchesSearch;
    if (actionFilter === 'DISCOVERED') return matchesSearch && log.action === 'DISCOVERED';
    if (actionFilter === 'CROSSED') return matchesSearch && log.action.startsWith('CROSSED');
    return matchesSearch;
  });

  const handleExportCSV = () => {
    if (logs.length === 0) return;

    const headers = ['Timestamp', 'Track_ID', 'Class_Name', 'Confidence', 'Action'];
    const rows = logs.map((l) => [
      l.timestamp,
      `#${String(l.trackId).padStart(2, '0')}`,
      l.className,
      `${Math.round(l.confidence * 100)}%`,
      l.action,
    ]);

    const csvContent =
      'data:text/csv;charset=utf-8,' +
      [headers.join(','), ...rows.map((e) => e.join(','))].join('\n');

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `ai_vision_events_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Convert log action to user-specified string format
  const formatFeedAction = (action: string) => {
    switch (action) {
      case 'CROSSED_IN':
        return 'entered';
      case 'CROSSED_OUT':
        return 'exited';
      case 'DISCOVERED':
        return 'detected';
      default:
        return 'crossed tripwire';
    }
  };

  return (
    <div id="detection-logs-panel" className="flex flex-col gap-3 bg-slate-900/90 border border-slate-800 rounded-xl p-4 shadow-lg">
      {/* Header with Search & Filter */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <Terminal className="w-4 h-4 text-emerald-400" />
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
            Live Event Log
          </h3>
          <span className="text-[11px] font-mono bg-slate-950 border border-slate-800 px-2 py-0.5 rounded text-emerald-400 font-bold">
            {logs.length} events
          </span>
        </div>

        {/* View Mode Toggle & Controls */}
        <div className="flex flex-wrap items-center gap-2">
          {/* Feed / Table View Toggle */}
          <div className="flex items-center bg-slate-950 p-1 rounded-lg border border-slate-800 text-xs font-mono">
            <button
              onClick={() => setViewMode('feed')}
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded transition-colors ${
                viewMode === 'feed'
                  ? 'bg-slate-800 text-emerald-400 font-bold'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Code2 className="w-3 h-3" />
              Event Feed
            </button>
            <button
              onClick={() => setViewMode('table')}
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded transition-colors ${
                viewMode === 'table'
                  ? 'bg-slate-800 text-sky-400 font-bold'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <TableIcon className="w-3 h-3" />
              Detailed Table
            </button>
          </div>

          {/* Action Filter */}
          <select
            value={actionFilter}
            onChange={(e) => setActionFilter(e.target.value as any)}
            className="bg-slate-950 border border-slate-800 text-xs font-mono text-slate-300 rounded-lg px-2.5 py-1.5 focus:outline-none"
          >
            <option value="ALL">All Events</option>
            <option value="CROSSED">Line Crossing Only</option>
            <option value="DISCOVERED">Detected Only</option>
          </select>

          {/* Search box */}
          <div className="relative">
            <Search className="w-3.5 h-3.5 absolute left-2.5 top-1/2 -translate-y-1/2 text-slate-500" />
            <input
              type="text"
              placeholder="Filter by ID / Class..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="bg-slate-950 border border-slate-800 text-xs font-mono text-slate-300 rounded-lg pl-8 pr-3 py-1.5 focus:outline-none focus:border-sky-500 w-44"
            />
          </div>

          {/* Export CSV */}
          <button
            onClick={handleExportCSV}
            disabled={logs.length === 0}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-emerald-400 bg-emerald-950/40 hover:bg-emerald-900/40 border border-emerald-800/60 disabled:opacity-50 transition-colors"
          >
            <FileSpreadsheet className="w-3.5 h-3.5" />
            Export CSV
          </button>

          {/* Clear */}
          <button
            onClick={onClearLogs}
            className="p-1.5 rounded-lg text-slate-400 hover:text-rose-400 bg-slate-950 border border-slate-800 hover:border-rose-900 transition-colors"
            title="Clear Event Log"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* VIEW MODE 1: Live Event Feed (Terminal Style) */}
      {viewMode === 'feed' ? (
        <div className="p-3 bg-slate-950 border border-slate-800/80 rounded-lg max-h-60 overflow-y-auto font-mono text-xs flex flex-col gap-1.5">
          {filteredLogs.length === 0 ? (
            <div className="py-8 text-center text-slate-500">
              No live events recorded yet. Waiting for tracking stream...
            </div>
          ) : (
            filteredLogs.slice(-40).map((log) => {
              const idFormatted = String(log.trackId).padStart(2, '0');
              const classFormatted = log.className.toUpperCase();
              const actionText = formatFeedAction(log.action);

              let actionColor = 'text-sky-400';
              if (log.action === 'CROSSED_IN') actionColor = 'text-emerald-400';
              if (log.action === 'CROSSED_OUT') actionColor = 'text-amber-400';

              return (
                <div
                  key={log.id}
                  className="flex items-center gap-2 hover:bg-slate-900/60 px-2 py-1 rounded transition-colors"
                >
                  <span className="text-slate-500">[{log.timestamp}]</span>
                  <span className="text-purple-400 font-bold">ID {idFormatted}</span>
                  <span className="text-slate-400">→</span>
                  <span className="text-slate-200 font-semibold">{classFormatted}</span>
                  <span className={`font-medium ${actionColor}`}>{actionText}</span>
                  <span className="text-slate-600 text-[10px] ml-auto">
                    {Math.round(log.confidence * 100)}%
                  </span>
                </div>
              );
            })
          )}
        </div>
      ) : (
        /* VIEW MODE 2: Structured Table View */
        <div className="overflow-x-auto max-h-60 overflow-y-auto border border-slate-800/80 rounded-lg bg-slate-950">
          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-slate-900/90 text-slate-400 border-b border-slate-800 sticky top-0">
              <tr>
                <th className="py-2 px-3">Time</th>
                <th className="py-2 px-3">Tracking ID</th>
                <th className="py-2 px-3">Class</th>
                <th className="py-2 px-3">Confidence</th>
                <th className="py-2 px-3">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/50 text-slate-300">
              {filteredLogs.length === 0 ? (
                <tr>
                  <td colSpan={5} className="py-8 text-center text-slate-500">
                    No detection events recorded yet.
                  </td>
                </tr>
              ) : (
                filteredLogs.map((log) => {
                  const isCrossIn = log.action === 'CROSSED_IN';
                  const isCrossOut = log.action === 'CROSSED_OUT';
                  const isDiscovered = log.action === 'DISCOVERED';

                  return (
                    <tr key={log.id} className="hover:bg-slate-900/50 transition-colors">
                      <td className="py-2 px-3 text-slate-400">{log.timestamp}</td>
                      <td className="py-2 px-3 font-bold text-purple-400">
                        #{String(log.trackId).padStart(2, '0')}
                      </td>
                      <td className="py-2 px-3 capitalize font-medium text-slate-200">{log.className}</td>
                      <td className="py-2 px-3 text-slate-400">{Math.round(log.confidence * 100)}%</td>
                      <td className="py-2 px-3">
                        {isCrossIn && (
                          <span className="inline-flex items-center gap-1 text-[11px] font-bold text-emerald-400 bg-emerald-950/60 border border-emerald-800/50 px-2 py-0.5 rounded">
                            <ArrowDownRight className="w-3 h-3" /> CROSSED IN
                          </span>
                        )}
                        {isCrossOut && (
                          <span className="inline-flex items-center gap-1 text-[11px] font-bold text-amber-400 bg-amber-950/60 border border-amber-800/50 px-2 py-0.5 rounded">
                            <ArrowUpRight className="w-3 h-3" /> CROSSED OUT
                          </span>
                        )}
                        {isDiscovered && (
                          <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-sky-400 bg-sky-950/60 border border-sky-800/50 px-2 py-0.5 rounded">
                            <Sparkles className="w-3 h-3" /> DETECTED
                          </span>
                        )}
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};
