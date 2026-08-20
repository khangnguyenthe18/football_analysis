"""
Tactical Analytics Module for Football Analysis.

This module provides data structures and analytics engines for:
1. HeatmapTracker: Accumulates 2D player pitch trajectories and computes
   smoothed 2D density maps (KDE-like Gaussian grids) for teams or individual players.
2. PassingNetworkAnalyzer: Analyzes completed pass events from PassDetector,
   calculates player centroid positions, generates passing connectivity matrices,
   and measures tactical pass involvement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import cv2

from sports.common.pass_detector import PassEvent, EventType
from sports.configs.football import SoccerPitchConfiguration


@dataclass
class PassingNetworkData:
    """
    Structured container for tactical passing network analysis results.

    Attributes:
        team_id: Identifier of the analyzed team (0 or 1).
        player_positions: Mapping of player_id -> (x_m, y_m) average centroid.
        player_involvements: Mapping of player_id -> total successful passes made & received.
        pass_connections: List of tuples (passer_id, receiver_id, pass_count).
        total_completed_passes: Total completed passes for the team.
        top_combinations: List of top passing duos sorted by frequency.
    """
    team_id: int
    player_positions: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    player_involvements: Dict[int, int] = field(default_factory=dict)
    pass_connections: List[Tuple[int, int, int]] = field(default_factory=list)
    total_completed_passes: int = 0
    top_combinations: List[Tuple[int, int, int]] = field(default_factory=list)


class HeatmapTracker:
    """
    Tracks player positions over time in pitch-metric coordinates (meters)
    and computes normalized 2D Gaussian density heatmaps.

    Args:
        pitch_length_m: Pitch length in meters (default 120.0m).
        pitch_width_m: Pitch width in meters (default 70.0m).
    """

    def __init__(
        self,
        pitch_length_m: float = 120.0,
        pitch_width_m: float = 70.0,
    ) -> None:
        self.pitch_length = pitch_length_m
        self.pitch_width = pitch_width_m

        # Mapping: player_id -> List of (x_m, y_m)
        self._trajectories: Dict[int, List[Tuple[float, float]]] = {}
        # Mapping: player_id -> team_id
        self._player_teams: Dict[int, int] = {}

    def update(
        self,
        player_xy_pitch: np.ndarray,
        player_tracker_ids: np.ndarray,
        player_team_ids: np.ndarray,
    ) -> None:
        """
        Record player positions for the current frame.

        Args:
            player_xy_pitch: Shape (N, 2) player positions in meters.
            player_tracker_ids: Shape (N,) tracker IDs.
            player_team_ids: Shape (N,) team IDs.
        """
        if len(player_xy_pitch) == 0:
            return

        for pos, pid, tid in zip(player_xy_pitch, player_tracker_ids, player_team_ids):
            pid = int(pid)
            tid = int(tid)
            x, y = float(pos[0]), float(pos[1])

            # Filter coordinates within pitch bounds (with reasonable tolerance)
            if -5.0 <= x <= self.pitch_length + 5.0 and -5.0 <= y <= self.pitch_width + 5.0:
                if pid not in self._trajectories:
                    self._trajectories[pid] = []
                self._trajectories[pid].append((x, y))
                self._player_teams[pid] = tid

    def get_player_positions(self, player_id: int) -> np.ndarray:
        """Get all recorded (x, y) coordinates for a specific player."""
        pts = self._trajectories.get(player_id, [])
        return np.array(pts, dtype=np.float32) if pts else np.empty((0, 2), dtype=np.float32)

    def get_player_centroid(self, player_id: int) -> Optional[Tuple[float, float]]:
        """Calculate the average (x, y) position in meters for a player."""
        pts = self.get_player_positions(player_id)
        if len(pts) == 0:
            return None
        mean_xy = np.mean(pts, axis=0)
        return float(mean_xy[0]), float(mean_xy[1])

    def get_team_players(self, team_id: int) -> List[int]:
        """Return list of player IDs belonging to a given team."""
        return [pid for pid, tid in self._player_teams.items() if tid == team_id]

    def generate_density_grid(
        self,
        team_id: Optional[int] = None,
        player_id: Optional[int] = None,
        grid_width: int = 1200,
        grid_height: int = 700,
        gaussian_kernel_size: int = 41,
        gaussian_sigma: float = 15.0,
    ) -> np.ndarray:
        """
        Generate a smoothed 2D density matrix [0.0, 1.0] for a player or an entire team.

        Args:
            team_id: Optional team ID filter (0 or 1).
            player_id: Optional individual player ID filter.
            grid_width: Resolution of density grid along X-axis (pitch length).
            grid_height: Resolution of density grid along Y-axis (pitch width).
            gaussian_kernel_size: Size of the Gaussian smoothing kernel (must be odd).
            gaussian_sigma: Standard deviation for Gaussian kernel smoothing.

        Returns:
            np.ndarray: 2D float32 array with shape (grid_height, grid_width) normalized to [0, 1].
        """
        grid = np.zeros((grid_height, grid_width), dtype=np.float32)

        # Collect points
        pts_list: List[Tuple[float, float]] = []
        if player_id is not None:
            pts_list = self._trajectories.get(player_id, [])
        elif team_id is not None:
            for pid, tid in self._player_teams.items():
                if tid == team_id:
                    pts_list.extend(self._trajectories.get(pid, []))
        else:
            for pts in self._trajectories.values():
                pts_list.extend(pts)

        if not pts_list:
            return grid

        pts_arr = np.array(pts_list, dtype=np.float32)

        # Map meters to grid pixels
        scale_x = grid_width / self.pitch_length
        scale_y = grid_height / self.pitch_width

        gx = np.clip(np.round(pts_arr[:, 0] * scale_x).astype(np.int32), 0, grid_width - 1)
        gy = np.clip(np.round(pts_arr[:, 1] * scale_y).astype(np.int32), 0, grid_height - 1)

        # 2D Histogram binning
        np.add.at(grid, (gy, gx), 1.0)

        # Gaussian smoothing
        if gaussian_kernel_size % 2 == 0:
            gaussian_kernel_size += 1

        smoothed = cv2.GaussianBlur(
            grid,
            (gaussian_kernel_size, gaussian_kernel_size),
            sigmaX=gaussian_sigma,
            sigmaY=gaussian_sigma
        )

        max_val = np.max(smoothed)
        if max_val > 0:
            smoothed = smoothed / max_val

        return smoothed


class PassingNetworkAnalyzer:
    """
    Analyzes passing sequences and generates structured passing network data.

    Calculates:
    - Pass counts between pairs of players ($C_{ij}$).
    - Average position (centroid) of each player when passing/receiving.
    - Total pass involvement score per player.
    """

    def __init__(self) -> None:
        self._events: List[PassEvent] = []

    def update_events(self, events: List[PassEvent]) -> None:
        """Set or update the list of pass events from PassDetector."""
        self._events = list(events)

    def add_event(self, event: PassEvent) -> None:
        """Add a single pass event."""
        self._events.append(event)

    def compute_network(
        self,
        team_id: int,
        heatmap_tracker: Optional[HeatmapTracker] = None,
        min_passes: int = 1,
    ) -> PassingNetworkData:
        """
        Compute passing network statistics for a given team.

        Args:
            team_id: The team to analyze (0 or 1).
            heatmap_tracker: Optional HeatmapTracker to supply full-match average positions.
            min_passes: Minimum number of passes between a pair to be included in connections.

        Returns:
            PassingNetworkData container with positions, involvements, and pass links.
        """
        completed_passes = [
            e for e in self._events
            if e.event_type == EventType.COMPLETED.value and e.passer_team == team_id and e.receiver_team == team_id
        ]

        player_involvements: Dict[int, int] = {}
        pass_matrix: Dict[Tuple[int, int], int] = {}
        event_positions: Dict[int, List[Tuple[float, float]]] = {}

        for p in completed_passes:
            p_id = p.passer_id
            r_id = p.receiver_id

            if p_id == -1 or r_id == -1 or p_id == r_id:
                continue

            # Record involvements
            player_involvements[p_id] = player_involvements.get(p_id, 0) + 1
            player_involvements[r_id] = player_involvements.get(r_id, 0) + 1

            # Record connection pair
            pair = (min(p_id, r_id), max(p_id, r_id))
            pass_matrix[pair] = pass_matrix.get(pair, 0) + 1

            # Accumulate positions during pass events
            if p_id not in event_positions:
                event_positions[p_id] = []
            event_positions[p_id].append(p.start_pos_2d)

            if r_id not in event_positions:
                event_positions[r_id] = []
            event_positions[r_id].append(p.end_pos_2d)

        # Compute centroid player positions
        player_positions: Dict[int, Tuple[float, float]] = {}
        all_players = set(player_involvements.keys())

        for pid in all_players:
            # First try using heatmap tracker full match centroid
            if heatmap_tracker is not None:
                centroid = heatmap_tracker.get_player_centroid(pid)
                if centroid is not None:
                    player_positions[pid] = centroid
                    continue

            # Fallback to average position during pass events
            if pid in event_positions and event_positions[pid]:
                arr = np.array(event_positions[pid])
                mean_pos = np.mean(arr, axis=0)
                player_positions[pid] = (float(mean_pos[0]), float(mean_pos[1]))
            else:
                player_positions[pid] = (60.0, 35.0)

        # Filter connections by min_passes threshold
        filtered_connections: List[Tuple[int, int, int]] = []
        for (p1, p2), count in pass_matrix.items():
            if count >= min_passes:
                filtered_connections.append((p1, p2, count))

        # Sort combinations by pass frequency
        top_combos = sorted(filtered_connections, key=lambda x: x[2], reverse=True)

        return PassingNetworkData(
            team_id=team_id,
            player_positions=player_positions,
            player_involvements=player_involvements,
            pass_connections=filtered_connections,
            total_completed_passes=len(completed_passes),
            top_combinations=top_combos,
        )

