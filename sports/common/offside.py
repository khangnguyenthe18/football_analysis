"""
Offside Detection & VAR Module for Football Analysis.

This module implements Semi-Automated Offside Technology (SAOT) logic:
1. Detects exact offside line (second-to-last defender and ball position)
   at the moment a pass is released (frame_start).
2. Determines attack direction per team and checks IFAB Law 11 criteria
   (Halfway line rule, behind-the-ball rule, second-to-last defender).
3. Produces structured OffsideDecision records with precision metric margins.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from sports.common.pass_detector import PassEvent


class AttackDirection(str, Enum):
    """Direction of team attack across pitch length axis (0m to 120m)."""
    RIGHT = "RIGHT"  # Attacking from X=0m towards X=120m (opponent goal at X=120m)
    LEFT = "LEFT"    # Attacking from X=120m towards X=0m (opponent goal at X=0m)


@dataclass
class OffsideDecision:
    """
    Structured record of a single evaluated offside situation.

    Attributes:
        pass_id: Unique pass identifier from PassDetector.
        frame_id: Frame number at pass release (kick-off moment).
        passer_id: Player ID initiating the pass.
        receiver_id: Targeted receiving attacker player ID.
        attacking_team: Team ID on attack (0 or 1).
        defending_team: Defending team ID (0 or 1).
        is_offside: Boolean verdict (True if offside, False if onside).
        margin_meters: Offside distance margin in meters (positive = offside, negative = onside).
        margin_cm: Offside distance margin in centimeters.
        offside_line_x_m: Calculated X coordinate of defensive line in meters.
        attacker_x_m: Receiver attacker X coordinate in meters.
        second_last_def_id: Player ID of the second-to-last defender.
        second_last_def_x_m: X coordinate of the second-to-last defender in meters.
        ball_x_m: X coordinate of the ball at pass release in meters.
        display_until_frame: Frame number until which the VAR overlay remains visible.
    """
    pass_id: int
    frame_id: int
    passer_id: int
    receiver_id: int
    attacking_team: int
    defending_team: int
    is_offside: bool
    margin_meters: float
    margin_cm: float
    offside_line_x_m: float
    attacker_x_m: float
    second_last_def_id: int
    second_last_def_x_m: float
    ball_x_m: float
    display_until_frame: int = 0


class OffsideDetector:
    """
    Semi-Automated Offside Technology (SAOT) engine.

    Evaluates potential offside events on attacking passes based on real-world
    metric positions, defender depth, ball position, and attack direction.

    Args:
        pitch_length_m: Pitch length in meters (default 120.0m).
        pitch_width_m: Pitch width in meters (default 70.0m).
        display_duration_frames: Number of video frames to persist the VAR offside overlay (default 50 frames ~ 1.7s).
    """

    def __init__(
        self,
        pitch_length_m: float = 120.0,
        pitch_width_m: float = 70.0,
        display_duration_frames: int = 50,
    ) -> None:
        self.pitch_length = pitch_length_m
        self.pitch_width = pitch_width_m
        self.halfway_x = pitch_length_m / 2.0  # 60.0m
        self.display_duration_frames = display_duration_frames

        # Known or inferred attack directions: team_id -> AttackDirection
        self._team_directions: Dict[int, AttackDirection] = {
            0: AttackDirection.RIGHT,
            1: AttackDirection.LEFT,
        }

        self._decisions: List[OffsideDecision] = []
        self._active_decision: Optional[OffsideDecision] = None
        self._checked_pass_ids: set[int] = set()

    def set_team_attack_direction(self, team_id: int, direction: AttackDirection) -> None:
        """Explicitly configure attack direction for a team."""
        self._team_directions[team_id] = direction

    def infer_attack_directions(
        self,
        player_xy_pitch: np.ndarray,
        player_team_ids: np.ndarray,
    ) -> None:
        """
        Infer team attack directions from spatial centroid positions.
        The team occupying the lower X region (<60m) attacks RIGHT; the other attacks LEFT.
        """
        if len(player_xy_pitch) == 0:
            return

        t0_x = player_xy_pitch[player_team_ids == 0, 0]
        t1_x = player_xy_pitch[player_team_ids == 1, 0]

        if len(t0_x) > 0 and len(t1_x) > 0:
            mean_0 = float(np.mean(t0_x))
            mean_1 = float(np.mean(t1_x))
            if mean_0 < mean_1:
                self._team_directions[0] = AttackDirection.RIGHT
                self._team_directions[1] = AttackDirection.LEFT
            else:
                self._team_directions[0] = AttackDirection.LEFT
                self._team_directions[1] = AttackDirection.RIGHT

    def evaluate_pass(
        self,
        pass_event: PassEvent,
        player_xy_pitch: np.ndarray,
        player_tracker_ids: np.ndarray,
        player_team_ids: np.ndarray,
        ball_xy_pitch: Optional[np.ndarray] = None,
    ) -> Optional[OffsideDecision]:
        """
        Evaluate if a pass event contains an offside situation.

        Args:
            pass_event: The PassEvent instance.
            player_xy_pitch: Shape (N, 2) player coordinates in meters at pass release.
            player_tracker_ids: Shape (N,) player tracker IDs.
            player_team_ids: Shape (N,) player team IDs.
            ball_xy_pitch: Optional (2,) ball position in meters.

        Returns:
            OffsideDecision record if evaluation succeeded, else None.
        """
        if pass_event.pass_id in self._checked_pass_ids:
            return None

        att_team = pass_event.passer_team
        def_team = 1 - att_team if att_team in (0, 1) else 1
        att_dir = self._team_directions.get(att_team, AttackDirection.RIGHT)

        start_x = float(pass_event.start_pos_2d[0])
        end_x = float(pass_event.end_pos_2d[0])

        # 1. Attacking receiver position
        receiver_id = pass_event.receiver_id
        if receiver_id == -1:
            att_pos = np.array(pass_event.end_pos_2d, dtype=np.float32)
        else:
            rec_mask = (player_tracker_ids == receiver_id)
            if rec_mask.sum() > 0:
                att_pos = player_xy_pitch[rec_mask][0]
            else:
                att_pos = np.array(pass_event.end_pos_2d, dtype=np.float32)

        attacker_x = float(att_pos[0])

        # Ball position at release
        if ball_xy_pitch is not None:
            ball_x = float(ball_xy_pitch[0])
        else:
            ball_x = start_x

        # 2. Filter: Only evaluate offside for forward attacking passes into/near opponent half
        if att_dir == AttackDirection.RIGHT:
            is_forward = (end_x > start_x - 1.0)
            in_attacking_zone = (attacker_x > 50.0 or end_x > 50.0)
            if not (is_forward and in_attacking_zone):
                return None  # Routine pass in defensive zone, do not trigger offside line

            # Defending players on their defensive half (X > 45m)
            def_mask = (player_team_ids == def_team) & (player_xy_pitch[:, 0] > 45.0)
            if def_mask.sum() < 1:
                # Fallback to all defenders
                def_mask = (player_team_ids == def_team)
                if def_mask.sum() < 1:
                    return None

            def_xy = player_xy_pitch[def_mask]
            def_ids = player_tracker_ids[def_mask]

            # Defending goal at X = 120m (highest X)
            sorted_indices = np.argsort(-def_xy[:, 0])  # Deepest defenders first
            if len(sorted_indices) >= 2:
                sec_def_idx = sorted_indices[1]  # Second-to-last defender
            else:
                sec_def_idx = sorted_indices[0]  # Last defender

            sec_def_id = int(def_ids[sec_def_idx])
            sec_def_x = float(def_xy[sec_def_idx, 0])

            # Offside line is furthest forward between 2nd-last defender and ball
            offside_line_x = max(sec_def_x, ball_x)
            margin = attacker_x - offside_line_x

            # Offside rule criteria
            is_offside = bool((attacker_x > self.halfway_x) and (attacker_x > offside_line_x + 0.05) and (attacker_x > ball_x + 0.05))

        else:  # AttackDirection.LEFT
            is_forward = (end_x < start_x + 1.0)
            in_attacking_zone = (attacker_x < 70.0 or end_x < 70.0)
            if not (is_forward and in_attacking_zone):
                return None

            # Defending players on their defensive half (X < 75m)
            def_mask = (player_team_ids == def_team) & (player_xy_pitch[:, 0] < 75.0)
            if def_mask.sum() < 1:
                def_mask = (player_team_ids == def_team)
                if def_mask.sum() < 1:
                    return None

            def_xy = player_xy_pitch[def_mask]
            def_ids = player_tracker_ids[def_mask]

            # Defending goal at X = 0m (lowest X)
            sorted_indices = np.argsort(def_xy[:, 0])  # Deepest defenders first
            if len(sorted_indices) >= 2:
                sec_def_idx = sorted_indices[1]
            else:
                sec_def_idx = sorted_indices[0]

            sec_def_id = int(def_ids[sec_def_idx])
            sec_def_x = float(def_xy[sec_def_idx, 0])

            offside_line_x = min(sec_def_x, ball_x)
            margin = offside_line_x - attacker_x

            is_offside = bool((attacker_x < self.halfway_x) and (attacker_x < offside_line_x - 0.05) and (attacker_x < ball_x - 0.05))

        # Only activate HUD overlay if it is an actual OFFSIDE or a close decision (|margin| <= 5.0m)
        should_display = is_offside or (abs(margin) <= 5.0)

        decision = OffsideDecision(
            pass_id=pass_event.pass_id,
            frame_id=pass_event.frame_start,
            passer_id=pass_event.passer_id,
            receiver_id=receiver_id,
            attacking_team=att_team,
            defending_team=def_team,
            is_offside=is_offside,
            margin_meters=round(margin, 2),
            margin_cm=round(margin * 100.0, 1),
            offside_line_x_m=round(offside_line_x, 2),
            attacker_x_m=round(attacker_x, 2),
            second_last_def_id=sec_def_id,
            second_last_def_x_m=round(sec_def_x, 2),
            ball_x_m=round(ball_x, 2),
            display_until_frame=pass_event.frame_start + self.display_duration_frames if should_display else 0,
        )

        self._decisions.append(decision)
        self._checked_pass_ids.add(pass_event.pass_id)
        if should_display:
            self._active_decision = decision

        return decision

    def get_active_decision(self, current_frame_id: int) -> Optional[OffsideDecision]:
        """Return the active OffsideDecision if within the display frame window."""
        if self._active_decision is not None:
            if current_frame_id <= self._active_decision.display_until_frame:
                return self._active_decision
            else:
                self._active_decision = None
        return None

    def get_all_decisions(self) -> List[OffsideDecision]:
        """Return list of all evaluated offside decisions."""
        return list(self._decisions)

    def get_summary_dataframe(self) -> pd.DataFrame:
        """Export all offside evaluations as a Pandas DataFrame."""
        if not self._decisions:
            return pd.DataFrame(columns=[
                "pass_id", "frame_id", "passer_id", "receiver_id", "attacking_team",
                "is_offside", "margin_cm", "offside_line_x_m", "attacker_x_m"
            ])
        return pd.DataFrame([asdict(d) for d in self._decisions])
