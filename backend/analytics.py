"""
Tripwire Spatial Analytics & Event Manager (Python Implementation)
Calculates line crossing vector intersections, IN/OUT direction,
object distribution, and rolling telemetry event logs.
"""

import time
from typing import List, Dict, Any, Set
from .tracker import SingleTrack


class TripwireAnalytics:
    def __init__(self, tripwire_ratio: float = 0.58):
        self.tripwire_ratio = tripwire_ratio
        self.count_in = 0
        self.count_out = 0
        self.total_unique_tracked = 0
        self.discovered_ids: Set[int] = set()
        self.crossed_ids: Dict[int, str] = {}  # track_id -> 'IN' | 'OUT'
        self.recent_events: List[Dict[str, Any]] = []
        self.current_class_counts: Dict[str, int] = {}
        self.total_class_counts: Dict[str, int] = {}

    def reset(self):
        self.count_in = 0
        self.count_out = 0
        self.total_unique_tracked = 0
        self.discovered_ids.clear()
        self.crossed_ids.clear()
        self.recent_events.clear()
        self.current_class_counts.clear()

    def process_tracks(self, tracks: List[SingleTrack], frame_height: int) -> Dict[str, Any]:
        tripwire_y = frame_height * self.tripwire_ratio
        current_counts: Dict[str, int] = {}

        now_str = time.strftime("%H:%M:%S")

        for track in tracks:
            tid = track.track_id
            cname = track.class_name
            current_counts[cname] = current_counts.get(cname, 0) + 1

            # Discover new track
            if tid not in self.discovered_ids:
                self.discovered_ids.add(tid)
                self.total_unique_tracked += 1
                self.total_class_counts[cname] = self.total_class_counts.get(cname, 0) + 1
                self.recent_events.append({
                    "id": f"{tid}_{time.time()}",
                    "timestamp": now_str,
                    "trackId": tid,
                    "className": cname,
                    "confidence": round(track.score, 2),
                    "action": "DISCOVERED"
                })

            # Check line crossing
            if len(track.trail) >= 2:
                prev_y = track.trail[-2][1]
                curr_y = track.trail[-1][1]

                # Check if crossed the horizontal line
                if prev_y < tripwire_y and curr_y >= tripwire_y:
                    if self.crossed_ids.get(tid) != "IN":
                        self.count_in += 1
                        self.crossed_ids[tid] = "IN"
                        self.recent_events.append({
                            "id": f"{tid}_{time.time()}_in",
                            "timestamp": now_str,
                            "trackId": tid,
                            "className": cname,
                            "confidence": round(track.score, 2),
                            "action": "CROSSED_IN"
                        })
                elif prev_y > tripwire_y and curr_y <= tripwire_y:
                    if self.crossed_ids.get(tid) != "OUT":
                        self.count_out += 1
                        self.crossed_ids[tid] = "OUT"
                        self.recent_events.append({
                            "id": f"{tid}_{time.time()}_out",
                            "timestamp": now_str,
                            "trackId": tid,
                            "className": cname,
                            "confidence": round(track.score, 2),
                            "action": "CROSSED_OUT"
                        })

        self.current_class_counts = current_counts

        if len(self.recent_events) > 200:
            self.recent_events = self.recent_events[-200:]

        return self.get_summary()

    def get_summary(self) -> Dict[str, Any]:
        return {
            "activeCount": len(self.current_class_counts),
            "totalUnique": self.total_unique_tracked,
            "countIn": self.count_in,
            "countOut": self.count_out,
            "classDistribution": self.current_class_counts,
            "totalClassDistribution": self.total_class_counts,
            "eventsCount": len(self.recent_events)
        }
