"""
Physical & Speed Analytics Module for Football Analysis.

This module provides data structures and tracking engines for:
1. PlayerSpeedTracker: Computes smoothed instantaneous player speeds (km/h and m/s),
   accumulates real-world pitch distance traveled, filters tracking jitter,
   detects sprint bursts, and classifies movement into standard workrate zones.
2. PlayerPhysicalStats: Container for individual player physical performance metrics.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class WorkrateZone(str, Enum):
    """Standard physiological velocity zones for soccer movement analysis."""
    WALKING = "walking"          # 0.0 - 7.2 km/h (0.0 - 2.0 m/s)
    JOGGING = "jogging"          # 7.2 - 14.4 km/h (2.0 - 4.0 m/s)
    RUNNING = "running"          # 14.4 - 19.8 km/h (4.0 - 5.5 m/s)
    HIGH_SPEED = "high_speed"    # 19.8 - 25.2 km/h (5.5 - 7.0 m/s)
    SPRINTING = "sprinting"      # >= 25.2 km/h (>= 7.0 m/s)


# Velocity thresholds in km/h
ZONE_THRESHOLDS = {
    WorkrateZone.WALKING: (0.0, 7.2),
    WorkrateZone.JOGGING: (7.2, 14.4),
    WorkrateZone.RUNNING: (14.4, 19.8),
    WorkrateZone.HIGH_SPEED: (19.8, 25.2),
    WorkrateZone.SPRINTING: (25.2, 100.0),
}


@dataclass
class PlayerPhysicalStats:
    """
    Structured record of physical metrics for a single player.

    Attributes:
        player_id: Tracker ID of the player.
        team_id: Team ID of the player (0 or 1).
        current_speed_kmh: Instantaneous smoothed speed in km/h.
        max_speed_kmh: Peak sprint speed in km/h.
        avg_speed_kmh: Average speed across all tracked frames in km/h.
        total_distance_m: Total distance covered in meters.
        total_distance_km: Total distance covered in kilometers.
        sprint_count: Number of sustained sprint bursts (>= 25.2 km/h).
        distance_by_zone_m: Distance covered in each workrate zone in meters.
        time_by_zone_s: Time spent in each workrate zone in seconds.
        active_frames: Total number of frames the player was tracked.
    """
    player_id: int
    team_id: int
    current_speed_kmh: float = 0.0
    max_speed_kmh: float = 0.0
    avg_speed_kmh: float = 0.0
    total_distance_m: float = 0.0
    total_distance_km: float = 0.0
    sprint_count: int = 0
    distance_by_zone_m: Dict[str, float] = field(default_factory=lambda: {z.value: 0.0 for z in WorkrateZone})
    time_by_zone_s: Dict[str, float] = field(default_factory=lambda: {z.value: 0.0 for z in WorkrateZone})
    active_frames: int = 0


class PlayerSpeedTracker:
    """
    Tracks and computes physical athletic metrics (speed, distance, workrate)
    for players using smoothed 2D real-world pitch metric coordinates.

    Args:
        fps: Video frame rate (default 30.0 fps).
        window_size: Number of frames in moving window for temporal speed smoothing.
        deadband_speed_kmh: Speed threshold below which micro-movements are treated as stationary.
        max_human_speed_kmh: Maximum realistic human speed cap to reject tracker glitch jumps.
        sprint_threshold_kmh: Minimum speed to qualify for sprint zone (default 25.2 km/h).
        min_sprint_frames: Minimum consecutive frames above threshold to trigger 1 sprint event.
    """

    def __init__(
        self,
        fps: float = 30.0,
        window_size: int = 7,
        deadband_speed_kmh: float = 0.8,
        max_human_speed_kmh: float = 38.0,
        sprint_threshold_kmh: float = 25.2,
        min_sprint_frames: int = 5,
    ) -> None:
        self.fps = fps
        self.dt = 1.0 / max(fps, 1.0)
        self.window_size = max(5, window_size)
        self.deadband_speed_kmh = deadband_speed_kmh
        self.max_human_speed_kmh = max_human_speed_kmh
        self.sprint_threshold_kmh = sprint_threshold_kmh
        self.min_sprint_frames = min_sprint_frames

        # Player storage
        self._positions: Dict[int, Deque[Tuple[int, float, float]]] = {}
        self._smoothed_pos: Dict[int, Tuple[float, float]] = {}
        self._player_teams: Dict[int, int] = {}
        self._current_speeds: Dict[int, float] = {}
        self._max_speeds: Dict[int, float] = {}
        self._speed_histories: Dict[int, List[float]] = {}
        self._total_distances: Dict[int, float] = {}
        self._last_positions: Dict[int, Tuple[float, float]] = {}
        self._active_frames: Dict[int, int] = {}

        # Sprint state machine
        self._sprint_consecutive_frames: Dict[int, int] = {}
        self._sprint_in_progress: Dict[int, bool] = {}
        self._sprint_counts: Dict[int, int] = {}

        # Zone breakdowns
        self._zone_distances: Dict[int, Dict[str, float]] = {}
        self._zone_times: Dict[int, Dict[str, float]] = {}

    def _classify_zone(self, speed_kmh: float) -> str:
        """Classify a speed value into one of the 5 physiological workrate zones."""
        if speed_kmh < 7.2:
            return WorkrateZone.WALKING.value
        elif speed_kmh < 14.4:
            return WorkrateZone.JOGGING.value
        elif speed_kmh < 19.8:
            return WorkrateZone.RUNNING.value
        elif speed_kmh < 25.2:
            return WorkrateZone.HIGH_SPEED.value
        else:
            return WorkrateZone.SPRINTING.value

    def update(
        self,
        frame_id: int,
        player_xy_pitch: np.ndarray,
        player_tracker_ids: np.ndarray,
        player_team_ids: np.ndarray,
    ) -> None:
        """
        Process current frame player positions and update speeds and distances.

        Args:
            frame_id: Current sequential frame number.
            player_xy_pitch: Shape (N, 2) player positions in pitch meters.
            player_tracker_ids: Shape (N,) tracker IDs.
            player_team_ids: Shape (N,) team IDs.
        """
        if len(player_xy_pitch) == 0:
            return

        for pos, pid, tid in zip(player_xy_pitch, player_tracker_ids, player_team_ids):
            pid = int(pid)
            tid = int(tid)
            raw_x, raw_y = float(pos[0]), float(pos[1])

            # Apply Exponential Moving Average (EMA) to pitch coordinates to filter homography jitter
            if pid in self._smoothed_pos:
                prev_sx, prev_sy = self._smoothed_pos[pid]
                sx = 0.3 * raw_x + 0.7 * prev_sx
                sy = 0.3 * raw_y + 0.7 * prev_sy
            else:
                sx, sy = raw_x, raw_y
            self._smoothed_pos[pid] = (sx, sy)

            # Initialize structures if first seen
            if pid not in self._positions:
                self._positions[pid] = deque(maxlen=self.window_size)
                self._current_speeds[pid] = 0.0
                self._max_speeds[pid] = 0.0
                self._speed_histories[pid] = []
                self._total_distances[pid] = 0.0
                self._active_frames[pid] = 0
                self._sprint_consecutive_frames[pid] = 0
                self._sprint_in_progress[pid] = False
                self._sprint_counts[pid] = 0
                self._zone_distances[pid] = {z.value: 0.0 for z in WorkrateZone}
                self._zone_times[pid] = {z.value: 0.0 for z in WorkrateZone}

            self._player_teams[pid] = tid
            self._active_frames[pid] += 1
            self._positions[pid].append((frame_id, sx, sy))

            # Calculate smoothed speed using sliding window displacement
            pos_history = list(self._positions[pid])
            if len(pos_history) >= 2:
                oldest_f, oldest_x, oldest_y = pos_history[0]
                delta_frames = frame_id - oldest_f

                if delta_frames >= 2:
                    dt_span = delta_frames * self.dt
                    dx = sx - oldest_x
                    dy = sy - oldest_y
                    dist_span = float(np.sqrt(dx * dx + dy * dy))
                    calc_speed_kmh = (dist_span / dt_span) * 3.6

                    # Apply realistic speed ceiling (maximum 34 km/h for soccer sprint)
                    calc_speed_kmh = min(calc_speed_kmh, 34.0)

                    # Deadband filter for standing/micro-jitter
                    if calc_speed_kmh < self.deadband_speed_kmh:
                        calc_speed_kmh = 0.0

                    # EMA filter on velocity to eliminate sudden frame spikes
                    prev_speed = self._current_speeds.get(pid, 0.0)
                    speed_kmh = 0.25 * calc_speed_kmh + 0.75 * prev_speed
                else:
                    speed_kmh = self._current_speeds.get(pid, 0.0)
            else:
                speed_kmh = 0.0

            # Store current and max speed
            self._current_speeds[pid] = round(speed_kmh, 1)
            self._speed_histories[pid].append(speed_kmh)
            if speed_kmh > self._max_speeds[pid]:
                self._max_speeds[pid] = round(speed_kmh, 1)

            # Accumulate frame-to-frame distance
            if pid in self._last_positions:
                last_x, last_y = self._last_positions[pid]
                step_dx = sx - last_x
                step_dy = sy - last_y
                step_dist = float(np.sqrt(step_dx * step_dx + step_dy * step_dy))

                # Accumulate distance only if speed exceeds deadband and step is realistic (< 1.5m per frame)
                if speed_kmh >= self.deadband_speed_kmh and step_dist < 1.5:
                    self._total_distances[pid] += step_dist
                    zone = self._classify_zone(speed_kmh)
                    self._zone_distances[pid][zone] += step_dist

            self._last_positions[pid] = (sx, sy)

            # Accumulate time in zone
            zone = self._classify_zone(speed_kmh)
            self._zone_times[pid][zone] += self.dt

            # Sprint detection FSM with hysteresis
            if speed_kmh >= self.sprint_threshold_kmh:
                self._sprint_consecutive_frames[pid] += 1
                if self._sprint_consecutive_frames[pid] >= self.min_sprint_frames and not self._sprint_in_progress[pid]:
                    self._sprint_in_progress[pid] = True
                    self._sprint_counts[pid] += 1
            else:
                self._sprint_consecutive_frames[pid] = 0
                if speed_kmh < (self.sprint_threshold_kmh - 3.0):  # Hysteresis reset threshold
                    self._sprint_in_progress[pid] = False

    def get_player_speed_kmh(self, player_id: int) -> float:
        """Get current instantaneous speed in km/h for a player."""
        return self._current_speeds.get(player_id, 0.0)

    def is_player_sprinting(self, player_id: int) -> bool:
        """Check if a player is currently in a sprint burst."""
        return self.get_player_speed_kmh(player_id) >= self.sprint_threshold_kmh

    def get_player_stats(self, player_id: int) -> Optional[PlayerPhysicalStats]:
        """Get full aggregated physical statistics for a single player."""
        if player_id not in self._current_speeds:
            return None

        tid = self._player_teams.get(player_id, -1)
        curr_speed = self._current_speeds.get(player_id, 0.0)
        max_speed = self._max_speeds.get(player_id, 0.0)
        history = self._speed_histories.get(player_id, [])
        avg_speed = float(np.mean(history)) if history else 0.0

        total_dist_m = self._total_distances.get(player_id, 0.0)
        total_dist_km = total_dist_m / 1000.0

        return PlayerPhysicalStats(
            player_id=player_id,
            team_id=tid,
            current_speed_kmh=round(curr_speed, 1),
            max_speed_kmh=round(max_speed, 1),
            avg_speed_kmh=round(avg_speed, 1),
            total_distance_m=round(total_dist_m, 1),
            total_distance_km=round(total_dist_km, 3),
            sprint_count=self._sprint_counts.get(player_id, 0),
            distance_by_zone_m={k: round(v, 1) for k, v in self._zone_distances.get(player_id, {}).items()},
            time_by_zone_s={k: round(v, 2) for k, v in self._zone_times.get(player_id, {}).items()},
            active_frames=self._active_frames.get(player_id, 0),
        )

    def get_all_stats(self) -> Dict[int, PlayerPhysicalStats]:
        """Get physical statistics for all tracked players."""
        stats_map: Dict[int, PlayerPhysicalStats] = {}
        for pid in self._current_speeds.keys():
            stat = self.get_player_stats(pid)
            if stat is not None:
                stats_map[pid] = stat
        return stats_map

    def get_team_stats(self, team_id: int) -> List[PlayerPhysicalStats]:
        """Get physical statistics for all players in a specific team."""
        all_stats = self.get_all_stats()
        return [s for s in all_stats.values() if s.team_id == team_id]

    def get_summary_dataframe(self) -> pd.DataFrame:
        """
        Export full physical performance summary as a Pandas DataFrame.

        Returns:
            DataFrame with player ID, team, max speed, avg speed, distance, sprints.
        """
        rows = []
        for pid, s in self.get_all_stats().items():
            row = {
                "player_id": s.player_id,
                "team_id": s.team_id,
                "max_speed_kmh": s.max_speed_kmh,
                "avg_speed_kmh": s.avg_speed_kmh,
                "total_distance_m": s.total_distance_m,
                "total_distance_km": s.total_distance_km,
                "sprints": s.sprint_count,
                "walking_dist_m": s.distance_by_zone_m.get(WorkrateZone.WALKING.value, 0.0),
                "jogging_dist_m": s.distance_by_zone_m.get(WorkrateZone.JOGGING.value, 0.0),
                "running_dist_m": s.distance_by_zone_m.get(WorkrateZone.RUNNING.value, 0.0),
                "hsr_dist_m": s.distance_by_zone_m.get(WorkrateZone.HIGH_SPEED.value, 0.0),
                "sprint_dist_m": s.distance_by_zone_m.get(WorkrateZone.SPRINTING.value, 0.0),
            }
            rows.append(row)

        if not rows:
            return pd.DataFrame(columns=[
                "player_id", "team_id", "max_speed_kmh", "avg_speed_kmh",
                "total_distance_m", "total_distance_km", "sprints"
            ])

        df = pd.DataFrame(rows)
        return df.sort_values(by=["team_id", "total_distance_m"], ascending=[True, False])
