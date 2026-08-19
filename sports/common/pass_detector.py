"""
Pass Detection Module for Football Analysis Pipeline.

This module implements a Finite State Machine (FSM) to detect, classify, and
track passing events in broadcast soccer footage. It integrates with the
existing supervision-based pipeline and operates entirely in real-world metric
coordinates (meters) using homography transformations.

Classes:
    BallStateInterpolator: 2D Kalman Filter for ball position smoothing.
    PassDetector: FSM engine that classifies possession and passing events.
    PassAnnotator: Visualization overlay for pass arrows, ball trails, and HUD.

Typical usage in a supervision video loop:
    >>> detector = PassDetector(fps=30.0)
    >>> annotator = PassAnnotator(team_colors=(color_0, color_1))
    >>> for frame in frames:
    ...     event = detector.update(frame_id, player_xy, ids, teams, ball_xy)
    ...     frame = annotator.annotate(frame, detector, transformer)
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
import supervision as sv


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

class PassState(Enum):
    """FSM states for the pass detection engine."""
    IDLE = "IDLE"
    IN_POSSESSION = "IN_POSSESSION"
    PASS_IN_FLIGHT = "PASS_IN_FLIGHT"


class EventType(str, Enum):
    """Classification of a resolved passing event."""
    COMPLETED = "completed"
    INTERCEPTED = "intercepted"
    INCOMPLETE = "incomplete"
    CARRY = "carry"


@dataclass
class PassEvent:
    """
    Structured record of a single passing event.

    Attributes:
        pass_id: Unique sequential identifier for this event.
        frame_start: Frame number when possession was released.
        frame_end: Frame number when the pass was resolved.
        passer_id: Tracker ID of the player who initiated the pass.
        passer_team: Team ID (0 or 1) of the passer.
        receiver_id: Tracker ID of the receiving player (-1 if incomplete).
        receiver_team: Team ID of the receiver (-1 if incomplete).
        start_pos_2d: Pitch coordinates (meters) where the pass originated.
        end_pos_2d: Pitch coordinates (meters) where the pass ended.
        pass_distance_m: Euclidean distance of the pass in meters.
        pass_duration_s: Duration of the pass flight in seconds.
        event_type: Classification string from EventType enum.
    """
    pass_id: int
    frame_start: int
    frame_end: int
    passer_id: int
    passer_team: int
    receiver_id: int
    receiver_team: int
    start_pos_2d: Tuple[float, float]
    end_pos_2d: Tuple[float, float]
    pass_distance_m: float
    pass_duration_s: float
    event_type: str


# ---------------------------------------------------------------------------
# Ball State Interpolator (Lightweight 2D Kalman Filter)
# ---------------------------------------------------------------------------

class BallStateInterpolator:
    """
    Lightweight 2D Kalman Filter for ball position smoothing and gap-filling.

    Maintains a state vector [x, y, vx, vy] in pitch-meters. Handles up to
    ``max_missing_frames`` consecutive dropped detections by running
    prediction-only steps, providing continuous ball position and velocity
    estimates.

    Args:
        dt: Time step between frames in seconds (1/fps).
        process_noise: Process noise magnitude (higher = more responsive).
        measurement_noise: Measurement noise magnitude (higher = smoother).
        max_missing_frames: Maximum consecutive frames to interpolate over.
    """

    def __init__(
        self,
        dt: float = 1 / 30.0,
        process_noise: float = 0.5,
        measurement_noise: float = 1.0,
        max_missing_frames: int = 5,
    ) -> None:
        self.dt = dt
        self.max_missing_frames = max_missing_frames
        self._missing_count: int = 0
        self._initialized: bool = False

        # State vector: [x, y, vx, vy]
        self.x = np.zeros(4, dtype=np.float64)

        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=np.float64)

        # Measurement matrix (we observe x, y only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        # Covariance matrices
        self.P = np.eye(4, dtype=np.float64) * 100.0
        self.Q = np.eye(4, dtype=np.float64) * process_noise
        self.R = np.eye(2, dtype=np.float64) * measurement_noise

    def predict(self) -> np.ndarray:
        """
        Run one prediction step.

        Returns:
            Predicted position [x, y] in meters.
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].copy()

    def update(self, measurement: Optional[np.ndarray]) -> np.ndarray:
        """
        Run a full predict-update cycle.

        If measurement is None, runs prediction only and increments the
        missing counter. If the missing counter exceeds ``max_missing_frames``,
        returns the last known position without further prediction drift.

        Args:
            measurement: Ball position [x, y] in pitch-meters, or None if
                the ball was not detected this frame.

        Returns:
            Estimated position [x, y] in meters.
        """
        if measurement is not None and not self._initialized:
            self.x[:2] = measurement
            self.x[2:] = 0.0
            self._initialized = True
            self._missing_count = 0
            return self.x[:2].copy()

        if not self._initialized:
            return np.zeros(2, dtype=np.float64)

        # Prediction step
        predicted_pos = self.predict()

        if measurement is not None:
            # Kalman gain
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)

            # Innovation
            y = measurement - self.H @ self.x
            self.x = self.x + K @ y
            self.P = (np.eye(4) - K @ self.H) @ self.P

            self._missing_count = 0
            return self.x[:2].copy()
        else:
            self._missing_count += 1
            if self._missing_count <= self.max_missing_frames:
                return predicted_pos
            else:
                # Stop drifting — hold last position
                return self.x[:2].copy()

    def get_velocity(self) -> Tuple[np.ndarray, float]:
        """
        Get current estimated velocity.

        Returns:
            Tuple of (velocity_vector [vx, vy], speed_scalar) in m/s.
        """
        vel = self.x[2:4].copy()
        speed = float(np.linalg.norm(vel))
        return vel, speed

    @property
    def position(self) -> np.ndarray:
        """Current estimated position [x, y] in meters."""
        return self.x[:2].copy()

    @property
    def is_active(self) -> bool:
        """Whether the filter has been initialized and is not stale."""
        return self._initialized and self._missing_count <= self.max_missing_frames


# ---------------------------------------------------------------------------
# Pass Detector (FSM Engine)
# ---------------------------------------------------------------------------

class PassDetector:
    """
    Finite State Machine for detecting and classifying passing events.

    Processes frame-by-frame data in real-world pitch coordinates (meters)
    and maintains internal state to track ball possession, pass flight, and
    pass resolution.

    Args:
        control_radius_m: Maximum distance (meters) for a player to be
            considered in control of the ball.
        min_possession_frames: Minimum consecutive frames a player must
            hold the ball to establish possession.
        min_pass_speed_ms: Minimum ball speed (m/s) to trigger a pass.
        max_flight_frames: Maximum frames a pass can be in flight before
            being classified as incomplete.
        min_touch_frames: Minimum frames a player must touch the ball to
            register as a valid reception (filters deflections).
        fps: Video frame rate for time calculations.
    """

    def __init__(
        self,
        control_radius_m: float = 2.0,
        min_possession_frames: int = 3,
        min_pass_speed_ms: float = 5.0,
        max_flight_frames: int = 90,
        min_touch_frames: int = 2,
        fps: float = 30.0,
    ) -> None:
        # Thresholds
        self.control_radius = control_radius_m
        self.min_possession_frames = min_possession_frames
        self.min_pass_speed = min_pass_speed_ms
        self.max_flight_frames = max_flight_frames
        self.min_touch_frames = min_touch_frames
        self.fps = fps

        # Anti-spam cooldown (Fix #1)
        self._event_cooldown: int = 0
        self._EVENT_COOLDOWN_FRAMES: int = 8  # ~0.27s at 30fps

        # Drift counter for gradual ball loss (Fix #3)
        self._drift_count: int = 0
        self._MAX_DRIFT_FRAMES: int = 10  # ~0.33s at 30fps

        # FSM state
        self._state: PassState = PassState.IDLE
        self._possessor_id: int = -1
        self._possessor_team: int = -1
        self._possession_count: int = 0
        self._candidate_id: int = -1
        self._candidate_count: int = 0

        # Pass tracking
        self._pass_start_frame: int = -1
        self._pass_start_pos: np.ndarray = np.zeros(2)
        self._flight_frame_count: int = 0

        # Ball interpolator
        self._ball_filter = BallStateInterpolator(
            dt=1.0 / fps,
            process_noise=0.5,
            measurement_noise=1.0,
            max_missing_frames=5,
        )

        # Ball trail for visualization (pixel positions stored separately)
        self._ball_trail: Deque[np.ndarray] = deque(maxlen=30)

        # Event log
        self._events: List[PassEvent] = []
        self._next_pass_id: int = 1

        # Team stats
        self._team_stats: Dict[int, Dict[str, int]] = {
            0: {"completed": 0, "intercepted": 0, "incomplete": 0, "total": 0},
            1: {"completed": 0, "intercepted": 0, "incomplete": 0, "total": 0},
        }

        # Possession tracking (frame-by-frame)
        self._possession_frames: Dict[int, int] = {0: 0, 1: 0}
        self._total_tracked_frames: int = 0
        self._current_possessing_team: int = -1

        # Active pass info for annotator
        self._active_pass_start_pitch: Optional[np.ndarray] = None
        self._active_passer_id: int = -1
        self._active_passer_team: int = -1

    @property
    def state(self) -> PassState:
        """Current FSM state."""
        return self._state

    @property
    def ball_position(self) -> np.ndarray:
        """Current estimated ball position in pitch-meters."""
        return self._ball_filter.position

    @property
    def ball_trail(self) -> List[np.ndarray]:
        """Recent ball positions for trail rendering."""
        return list(self._ball_trail)

    def _find_nearest_player(
        self,
        ball_xy: np.ndarray,
        player_xy: np.ndarray,
        player_ids: np.ndarray,
        player_teams: np.ndarray,
    ) -> Tuple[int, int, float]:
        """
        Find the player nearest to the ball.

        Returns:
            Tuple of (tracker_id, team_id, distance_m). Returns (-1, -1, inf)
            if no players are present.
        """
        if len(player_xy) == 0:
            return -1, -1, float("inf")

        distances = np.linalg.norm(player_xy - ball_xy, axis=1)
        idx = int(np.argmin(distances))
        return int(player_ids[idx]), int(player_teams[idx]), float(distances[idx])

    def _log_event(
        self,
        frame_end: int,
        receiver_id: int,
        receiver_team: int,
        end_pos: np.ndarray,
        event_type: EventType,
    ) -> PassEvent:
        """Create and store a PassEvent, update team statistics."""
        duration = (frame_end - self._pass_start_frame) / self.fps
        distance = float(np.linalg.norm(end_pos - self._pass_start_pos))

        event = PassEvent(
            pass_id=self._next_pass_id,
            frame_start=self._pass_start_frame,
            frame_end=frame_end,
            passer_id=self._possessor_id,
            passer_team=self._possessor_team,
            receiver_id=receiver_id,
            receiver_team=receiver_team,
            start_pos_2d=(float(self._pass_start_pos[0]),
                          float(self._pass_start_pos[1])),
            end_pos_2d=(float(end_pos[0]), float(end_pos[1])),
            pass_distance_m=round(distance, 2),
            pass_duration_s=round(duration, 3),
            event_type=event_type.value,
        )

        self._events.append(event)
        self._next_pass_id += 1

        # Update team stats (only for actual passes, not carries)
        # Filter: ignore very short-distance events (< 2m) which are
        # contested ball situations, not real pass attempts.
        if event_type != EventType.CARRY and distance >= 2.0:
            passer_team = self._possessor_team
            if passer_team in self._team_stats:
                self._team_stats[passer_team]["total"] += 1
                if event_type == EventType.COMPLETED:
                    self._team_stats[passer_team]["completed"] += 1
                elif event_type == EventType.INTERCEPTED:
                    self._team_stats[passer_team]["intercepted"] += 1
                elif event_type == EventType.INCOMPLETE:
                    self._team_stats[passer_team]["incomplete"] += 1

        # Set cooldown after logging an event to prevent spam
        self._event_cooldown = self._EVENT_COOLDOWN_FRAMES

        return event

    def update(
        self,
        frame_id: int,
        player_xy_pitch: np.ndarray,
        player_tracker_ids: np.ndarray,
        player_team_ids: np.ndarray,
        ball_xy_pitch: Optional[np.ndarray],
    ) -> Optional[PassEvent]:
        """
        Process a single frame and advance the FSM.

        All coordinates must be in **pitch-meters** (divide centimeter
        pitch coordinates by 100 before calling).

        Args:
            frame_id: Sequential frame number.
            player_xy_pitch: Shape (N, 2) player positions in meters.
            player_tracker_ids: Shape (N,) tracker IDs.
            player_team_ids: Shape (N,) team IDs (0 or 1).
            ball_xy_pitch: Shape (2,) ball position in meters, or None if
                the ball was not detected in this frame.

        Returns:
            A PassEvent if a pass was resolved this frame, else None.
        """
        # --- Kalman filter update ---
        ball_pos = self._ball_filter.update(ball_xy_pitch)
        _, ball_speed = self._ball_filter.get_velocity()

        # Store trail position
        if self._ball_filter.is_active:
            self._ball_trail.append(ball_pos.copy())

        # Find nearest player to ball
        nearest_id, nearest_team, nearest_dist = self._find_nearest_player(
            ball_pos, player_xy_pitch, player_tracker_ids, player_team_ids
        )

        # --- Possession tracking (frame-by-frame) ---
        if nearest_dist <= self.control_radius and nearest_team in (0, 1):
            self._possession_frames[nearest_team] += 1
            self._total_tracked_frames += 1
            self._current_possessing_team = nearest_team
        elif self._state == PassState.PASS_IN_FLIGHT:
            # During pass flight, attribute possession to the passer's team
            if self._possessor_team in (0, 1):
                self._possession_frames[self._possessor_team] += 1
                self._total_tracked_frames += 1
                self._current_possessing_team = self._possessor_team
        else:
            # Loose ball / no player within control radius
            self._current_possessing_team = -1

        result: Optional[PassEvent] = None

        # Decrement cooldown timer
        if self._event_cooldown > 0:
            self._event_cooldown -= 1

        # ============================================================
        # FSM State Transitions
        # ============================================================

        if self._state == PassState.IDLE:
            # ---- IDLE: waiting for any player to establish possession ----
            if nearest_dist <= self.control_radius:
                if nearest_id == self._candidate_id:
                    self._candidate_count += 1
                else:
                    self._candidate_id = nearest_id
                    self._candidate_count = 1

                if self._candidate_count >= self.min_possession_frames:
                    self._state = PassState.IN_POSSESSION
                    self._possessor_id = nearest_id
                    self._possessor_team = nearest_team
                    self._possession_count = self._candidate_count
                    self._candidate_id = -1
                    self._candidate_count = 0
                    self._drift_count = 0
            else:
                self._candidate_id = -1
                self._candidate_count = 0

        elif self._state == PassState.IN_POSSESSION:
            # ---- IN_POSSESSION: a player has the ball ----
            if nearest_dist <= self.control_radius and nearest_id == self._possessor_id:
                # Still in possession — reset drift counter
                self._possession_count += 1
                self._drift_count = 0

            elif (ball_speed >= self.min_pass_speed
                  and nearest_dist > self.control_radius):
                # Ball is moving fast and left the possessor → pass initiated
                self._state = PassState.PASS_IN_FLIGHT
                self._pass_start_frame = frame_id
                self._pass_start_pos = ball_pos.copy()
                self._flight_frame_count = 1
                self._candidate_id = -1
                self._candidate_count = 0
                self._drift_count = 0

                # Store for annotator
                self._active_pass_start_pitch = self._pass_start_pos.copy()
                self._active_passer_id = self._possessor_id
                self._active_passer_team = self._possessor_team

            elif nearest_dist <= self.control_radius and nearest_id != self._possessor_id:
                # Different player grabbed the ball directly (no flight phase)
                # FIX #2: Silent handover — do NOT log event here.
                # Only real passes (ball goes through PASS_IN_FLIGHT) create events.
                # This prevents spam from detection jitter when players contest.
                self._possessor_id = nearest_id
                self._possessor_team = nearest_team
                self._possession_count = 1
                self._drift_count = 0
                self._active_pass_start_pitch = None

            else:
                # Ball drifted slightly but not fast enough
                # FIX #3: Track drift duration — if ball stays away too long,
                # transition to IDLE instead of holding IN_POSSESSION forever
                self._drift_count += 1
                if self._drift_count >= self._MAX_DRIFT_FRAMES:
                    # Ball has been away too long → lost possession
                    self._state = PassState.IDLE
                    self._possessor_id = -1
                    self._possessor_team = -1
                    self._possession_count = 0
                    self._drift_count = 0
                    self._active_pass_start_pitch = None

        elif self._state == PassState.PASS_IN_FLIGHT:
            # ---- PASS_IN_FLIGHT: ball is traveling between players ----
            self._flight_frame_count += 1

            if nearest_dist <= self.control_radius:
                # A player is close to the ball — check if it's a touch
                if nearest_id == self._candidate_id:
                    self._candidate_count += 1
                else:
                    self._candidate_id = nearest_id
                    self._candidate_count = 1

                if self._candidate_count >= self.min_touch_frames:
                    # Valid reception confirmed
                    if nearest_id == self._possessor_id:
                        # DRIBBLE FILTER: ball returned to original possessor
                        result = self._log_event(
                            frame_end=frame_id,
                            receiver_id=nearest_id,
                            receiver_team=nearest_team,
                            end_pos=ball_pos,
                            event_type=EventType.CARRY,
                        )
                    elif nearest_team == self._possessor_team:
                        # Completed pass to teammate
                        result = self._log_event(
                            frame_end=frame_id,
                            receiver_id=nearest_id,
                            receiver_team=nearest_team,
                            end_pos=ball_pos,
                            event_type=EventType.COMPLETED,
                        )
                    else:
                        # Intercepted by opponent
                        result = self._log_event(
                            frame_end=frame_id,
                            receiver_id=nearest_id,
                            receiver_team=nearest_team,
                            end_pos=ball_pos,
                            event_type=EventType.INTERCEPTED,
                        )

                    # Transfer possession to receiver
                    self._state = PassState.IN_POSSESSION
                    self._possessor_id = nearest_id
                    self._possessor_team = nearest_team
                    self._possession_count = self._candidate_count
                    self._candidate_id = -1
                    self._candidate_count = 0
                    self._flight_frame_count = 0
                    self._active_pass_start_pitch = None

            else:
                # Ball still in flight, no player nearby
                self._candidate_id = -1
                self._candidate_count = 0

                if self._flight_frame_count >= self.max_flight_frames:
                    # Timeout → incomplete pass
                    result = self._log_event(
                        frame_end=frame_id,
                        receiver_id=-1,
                        receiver_team=-1,
                        end_pos=ball_pos,
                        event_type=EventType.INCOMPLETE,
                    )
                    self._state = PassState.IDLE
                    self._possessor_id = -1
                    self._possessor_team = -1
                    self._possession_count = 0
                    self._flight_frame_count = 0
                    self._active_pass_start_pitch = None

        return result

    # ----- Data Export Methods -----

    def get_events_dataframe(self) -> pd.DataFrame:
        """
        Export all logged pass events as a Pandas DataFrame.

        Returns:
            DataFrame with one row per event, columns matching PassEvent fields.
        """
        if not self._events:
            return pd.DataFrame(columns=[
                "pass_id", "frame_start", "frame_end", "passer_id",
                "passer_team", "receiver_id", "receiver_team",
                "start_pos_2d", "end_pos_2d", "pass_distance_m",
                "pass_duration_s", "event_type",
            ])
        return pd.DataFrame([asdict(e) for e in self._events])

    def get_events_json(self, indent: int = 2) -> str:
        """
        Export all logged pass events as a JSON string.

        Args:
            indent: JSON indentation level.

        Returns:
            JSON-formatted string of all events.
        """
        return json.dumps([asdict(e) for e in self._events], indent=indent)

    def get_team_stats(self) -> Dict[int, Dict[str, int]]:
        """
        Get pass statistics per team.

        Returns:
            Dict keyed by team_id (0, 1), each containing counts of
            completed, intercepted, incomplete passes and total attempts.
        """
        return self._team_stats.copy()

    def get_pass_accuracy(self, team_id: int) -> float:
        """
        Calculate pass accuracy percentage for a given team.

        Args:
            team_id: Team identifier (0 or 1).

        Returns:
            Accuracy as a percentage (0.0–100.0). Returns 0.0 if no passes.
        """
        stats = self._team_stats.get(team_id, {"completed": 0, "total": 0})
        if stats["total"] == 0:
            return 0.0
        return round(100.0 * stats["completed"] / stats["total"], 1)

    def get_possession_pct(self, team_id: int) -> float:
        """
        Calculate ball possession percentage for a given team.

        Args:
            team_id: Team identifier (0 or 1).

        Returns:
            Possession as a percentage (0.0–100.0). Returns 50.0 if no data.
        """
        if self._total_tracked_frames == 0:
            return 50.0
        return round(
            100.0 * self._possession_frames.get(team_id, 0)
            / self._total_tracked_frames, 1
        )

    @property
    def current_possessing_team(self) -> int:
        """Team ID currently in possession (-1 if unknown / loose ball)."""
        return self._current_possessing_team

    def get_active_pass(self) -> Optional[Dict]:
        """
        Get info about the currently in-flight pass (for real-time rendering).

        Returns:
            Dict with passer info and start position, or None if no active pass.
        """
        if (self._state == PassState.PASS_IN_FLIGHT
                and self._active_pass_start_pitch is not None):
            return {
                "passer_id": self._active_passer_id,
                "passer_team": self._active_passer_team,
                "start_pos": self._active_pass_start_pitch.copy(),
                "current_ball_pos": self._ball_filter.position.copy(),
                "flight_frames": self._flight_frame_count,
            }
        return None


# ---------------------------------------------------------------------------
# Pass Annotator (Visualization)
# ---------------------------------------------------------------------------

class PassAnnotator:
    """
    Draws pass detection and possession overlays on video frames.

    Renders:
    1. Ball trail during flight phase
    2. Broadcast-style horizontal possession bars at top center
    3. Real-time active possession indicator at bottom left
    4. Live pass statistics HUD scoreboard

    Args:
        team_colors: Tuple of two sv.Color objects for Team 1 (index 0) and Team 2 (index 1).
        team_names: Tuple of display names for the teams, default ("Team 1", "Team 2").
        arrow_thickness: Thickness of pass arrow lines in pixels.
        trail_length: Number of recent ball positions to render as a trail.
        hud_position: Top-left corner (x, y) of the pass stats HUD in pixels.
        hud_font_scale: Font scale for HUD text.
    """

    def __init__(
        self,
        team_colors: Tuple[sv.Color, sv.Color] = (
            sv.Color.from_hex("#FF1493"),
            sv.Color.from_hex("#00BFFF"),
        ),
        team_names: Tuple[str, str] = ("Team 1", "Team 2"),
        arrow_thickness: int = 2,
        trail_length: int = 15,
        hud_position: Tuple[int, int] = (20, 20),
        hud_font_scale: float = 0.65,
    ) -> None:
        self.team_colors = team_colors
        self.team_names = team_names
        self.arrow_thickness = arrow_thickness
        self.trail_length = trail_length
        self.hud_position = hud_position
        self.hud_font_scale = hud_font_scale

    def _pitch_to_pixel(
        self,
        points_m: np.ndarray,
        view_transformer: object,
    ) -> Optional[np.ndarray]:
        """
        Convert pitch-meter coordinates back to pixel coordinates.

        Uses the inverse of the ViewTransformer's homography matrix.

        Args:
            points_m: Shape (N, 2) points in pitch-meters.
            view_transformer: A ViewTransformer instance (must have attribute `m`).

        Returns:
            Shape (N, 2) pixel coordinates, or None if inversion fails.
        """
        try:
            H_inv = np.linalg.inv(view_transformer.m)
            points_cm = points_m * 100.0
            points_reshaped = points_cm.reshape(-1, 1, 2).astype(np.float32)
            pixel_points = cv2.perspectiveTransform(points_reshaped, H_inv)
            return pixel_points.reshape(-1, 2).astype(np.int32)
        except (np.linalg.LinAlgError, AttributeError):
            return None

    def _draw_ball_trail(
        self,
        frame: np.ndarray,
        trail_pixels: np.ndarray,
    ) -> np.ndarray:
        """Draw a fading ball trail as colored dots."""
        n = len(trail_pixels)
        if n < 2:
            return frame

        for i in range(n):
            alpha = (i + 1) / n  # 0→1 (oldest→newest)
            radius = max(2, int(alpha * 5))
            color = (
                int(0 * (1 - alpha) + 0 * alpha),
                int(255 * alpha),
                int(255 * (1 - alpha)),
            )
            center = (int(trail_pixels[i][0]), int(trail_pixels[i][1]))
            cv2.circle(frame, center, radius, color, -1)

        return frame

    def _draw_hud(
        self,
        frame: np.ndarray,
        pass_detector: PassDetector,
    ) -> np.ndarray:
        """Draw broadcast-style possession bars (top center), active possession (bottom left), and pass stats."""
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        color_0 = self.team_colors[0].as_bgr()
        color_1 = self.team_colors[1].as_bgr()

        name_0 = self.team_names[0]
        name_1 = self.team_names[1]

        # ============================================================
        # 1. TOP CENTER: Possession Bars (Broadcast TV Style)
        # ============================================================
        poss_0 = pass_detector.get_possession_pct(0)
        poss_1 = pass_detector.get_possession_pct(1)

        bar_total_w = 280
        bar_h = 16
        bar_x = (w - bar_total_w) // 2
        bar_y_0 = 14
        bar_y_1 = bar_y_0 + bar_h + 6

        # Semi-transparent background for top HUD
        bg_x1 = bar_x - 110
        bg_x2 = bar_x + bar_total_w + 75
        bg_y1 = 6
        bg_y2 = bar_y_1 + bar_h + 8
        overlay_top = frame.copy()
        cv2.rectangle(overlay_top, (bg_x1, bg_y1), (bg_x2, bg_y2), (15, 15, 15), -1)
        cv2.addWeighted(overlay_top, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (80, 80, 80), 1)

        # Team 1 Bar
        cv2.putText(frame, name_0, (bar_x - 100, bar_y_0 + 13),
                    font, 0.5, color_0, 2)
        cv2.rectangle(frame, (bar_x, bar_y_0),
                      (bar_x + bar_total_w, bar_y_0 + bar_h),
                      (45, 45, 45), -1)
        fill_w_0 = int(bar_total_w * poss_0 / 100.0)
        if fill_w_0 > 0:
            cv2.rectangle(frame, (bar_x, bar_y_0),
                          (bar_x + fill_w_0, bar_y_0 + bar_h),
                          color_0, -1)
        cv2.rectangle(frame, (bar_x, bar_y_0),
                      (bar_x + bar_total_w, bar_y_0 + bar_h),
                      (120, 120, 120), 1)
        cv2.putText(frame, f"{poss_0:.1f}%",
                    (bar_x + bar_total_w + 10, bar_y_0 + 13),
                    font, 0.5, (255, 255, 255), 1)

        # Team 2 Bar
        cv2.putText(frame, name_1, (bar_x - 100, bar_y_1 + 13),
                    font, 0.5, color_1, 2)
        cv2.rectangle(frame, (bar_x, bar_y_1),
                      (bar_x + bar_total_w, bar_y_1 + bar_h),
                      (45, 45, 45), -1)
        fill_w_1 = int(bar_total_w * poss_1 / 100.0)
        if fill_w_1 > 0:
            cv2.rectangle(frame, (bar_x, bar_y_1),
                          (bar_x + fill_w_1, bar_y_1 + bar_h),
                          color_1, -1)
        cv2.rectangle(frame, (bar_x, bar_y_1),
                      (bar_x + bar_total_w, bar_y_1 + bar_h),
                      (120, 120, 120), 1)
        cv2.putText(frame, f"{poss_1:.1f}%",
                    (bar_x + bar_total_w + 10, bar_y_1 + 13),
                    font, 0.5, (255, 255, 255), 1)

        # ============================================================
        # 2. BOTTOM LEFT: Real-time Possession Indicator
        # ============================================================
        curr_team = pass_detector.current_possessing_team
        if curr_team == 0:
            poss_label = f"Possession: {name_0}"
            poss_color = color_0
        elif curr_team == 1:
            poss_label = f"Possession: {name_1}"
            poss_color = color_1
        else:
            poss_label = "Possession: ---"
            poss_color = (180, 180, 180)

        overlay_bot = frame.copy()
        cv2.rectangle(overlay_bot, (0, h - 38), (280, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay_bot, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, poss_label, (12, h - 14),
                    font, 0.6, poss_color, 2)

        # ============================================================
        # 3. TOP LEFT: Pass Statistics Box
        # ============================================================
        x, y = self.hud_position
        stats_0 = pass_detector.get_team_stats().get(0, {})
        stats_1 = pass_detector.get_team_stats().get(1, {})
        acc_0 = pass_detector.get_pass_accuracy(0)
        acc_1 = pass_detector.get_pass_accuracy(1)

        completed_0 = stats_0.get("completed", 0)
        total_0 = stats_0.get("total", 0)
        completed_1 = stats_1.get("completed", 0)
        total_1 = stats_1.get("total", 0)

        line_1 = f"{name_0}: {completed_0}/{total_0} passes ({acc_0:.0f}%)"
        line_2 = f"{name_1}: {completed_1}/{total_1} passes ({acc_1:.0f}%)"

        overlay_stat = frame.copy()
        cv2.rectangle(overlay_stat, (x, y), (x + 270, y + 54), (15, 15, 15), -1)
        cv2.addWeighted(overlay_stat, 0.65, frame, 0.35, 0, frame)
        cv2.rectangle(frame, (x, y), (x + 270, y + 54), (80, 80, 80), 1)

        cv2.putText(frame, line_1, (x + 10, y + 20), font, 0.45, color_0, 1)
        cv2.putText(frame, line_2, (x + 10, y + 42), font, 0.45, color_1, 1)

        return frame

    def annotate(
        self,
        frame: np.ndarray,
        pass_detector: PassDetector,
        view_transformer: object,
        frame_id: int = 0,
    ) -> np.ndarray:
        """
        Annotate a frame with pass detection and possession overlays.

        Args:
            frame: The video frame to annotate (modified in-place).
            pass_detector: The PassDetector instance with current state.
            view_transformer: A ViewTransformer for coordinate conversion.
            frame_id: Current frame number.

        Returns:
            The annotated frame.
        """
        annotated = frame.copy()

        # 1. Draw ball trail
        trail = pass_detector.ball_trail
        if trail and len(trail) >= 2:
            trail_arr = np.array(trail[-self.trail_length:])
            trail_pixels = self._pitch_to_pixel(trail_arr, view_transformer)
            if trail_pixels is not None:
                self._draw_ball_trail(annotated, trail_pixels)

        # 2. Draw HUD (Possession bars + Pass statistics)
        annotated = self._draw_hud(annotated, pass_detector)

        return annotated
