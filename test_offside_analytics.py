"""
Unit and functional test for Offside Detection (SAOT) & VAR Visualizer Engine.
"""

import os
import sys
import numpy as np
import cv2
import supervision as sv

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sports.configs.football import SoccerPitchConfiguration
from sports.common.pass_detector import PassEvent, EventType
from sports.common.offside import OffsideDetector, AttackDirection
from sports.common.view import ViewTransformer
from sports.annotators.football import draw_offside_line_on_frame, draw_offside_pitch_snapshot


def test_offside_pipeline():
    print(">>> Testing Offside Detection & SAOT VAR Engine...")
    config = SoccerPitchConfiguration()
    detector = OffsideDetector(pitch_length_m=120.0, pitch_width_m=70.0, display_duration_frames=30)
    detector.set_team_attack_direction(0, AttackDirection.RIGHT)  # Team 0 attacks right (towards 120m)

    # 1. Test Scenario 1: Clear OFFSIDE Situation
    # Team 0 (Attacking): Passer #8 at (70m, 35m), Receiver #9 at (92m, 30m)
    # Team 1 (Defending): GK #1 at (115m, 35m), 2nd-last defender #4 at (88m, 30m), other defenders at (85m, 20m)
    pass_event_offside = PassEvent(
        pass_id=1,
        frame_start=100,
        frame_end=125,
        passer_id=8,
        passer_team=0,
        receiver_id=9,
        receiver_team=0,
        start_pos_2d=(70.0, 35.0),
        end_pos_2d=(92.0, 30.0),
        pass_distance_m=22.5,
        pass_duration_s=0.83,
        event_type=EventType.COMPLETED.value
    )

    player_xy_1 = np.array([
        [70.0, 35.0],  # #8 (Attacker Passer)
        [92.0, 30.0],  # #9 (Attacker Receiver)
        [115.0, 35.0], # #1 (Defending GK - deepest)
        [88.0, 30.0],  # #4 (2nd-last defender - offside line at 88m)
        [85.0, 20.0],  # #5 (Other defender)
    ], dtype=np.float32)

    p_ids_1 = np.array([8, 9, 1, 4, 5])
    t_ids_1 = np.array([0, 0, 1, 1, 1])

    decision_1 = detector.evaluate_pass(
        pass_event=pass_event_offside,
        player_xy_pitch=player_xy_1,
        player_tracker_ids=p_ids_1,
        player_team_ids=t_ids_1,
        ball_xy_pitch=np.array([70.0, 35.0]),
    )

    assert decision_1 is not None, "Offside evaluation returned None"
    assert decision_1.is_offside is True, f"Expected OFFSIDE, got is_offside={decision_1.is_offside}"
    assert decision_1.second_last_def_id == 4, f"Expected defender #4, got #{decision_1.second_last_def_id}"
    assert abs(decision_1.offside_line_x_m - 88.0) < 0.1, f"Offside line mismatch: {decision_1.offside_line_x_m}"
    assert abs(decision_1.margin_cm - 400.0) < 5.0, f"Expected margin +400cm, got {decision_1.margin_cm}cm"
    print(f"  [OK] Scenario 1 verified: OFFSIDE (+{decision_1.margin_cm:.1f} cm)")

    # 2. Test Scenario 2: Valid ONSIDE Situation
    # Receiver #10 is at (84m, 25m) behind the 2nd-last defender at 88m
    pass_event_onside = PassEvent(
        pass_id=2,
        frame_start=200,
        frame_end=220,
        passer_id=8,
        passer_team=0,
        receiver_id=10,
        receiver_team=0,
        start_pos_2d=(65.0, 35.0),
        end_pos_2d=(84.0, 25.0),
        pass_distance_m=21.5,
        pass_duration_s=0.67,
        event_type=EventType.COMPLETED.value
    )

    player_xy_2 = np.array([
        [65.0, 35.0],  # #8 (Attacker Passer)
        [84.0, 25.0],  # #10 (Attacker Receiver - onside at 84m)
        [115.0, 35.0], # #1 (Defending GK)
        [88.0, 30.0],  # #4 (2nd-last defender at 88m)
    ], dtype=np.float32)

    decision_2 = detector.evaluate_pass(
        pass_event=pass_event_onside,
        player_xy_pitch=player_xy_2,
        player_tracker_ids=np.array([8, 10, 1, 4]),
        player_team_ids=np.array([0, 0, 1, 1]),
        ball_xy_pitch=np.array([65.0, 35.0]),
    )

    assert decision_2 is not None, "Onside evaluation returned None"
    assert decision_2.is_offside is False, f"Expected ONSIDE, got is_offside={decision_2.is_offside}"
    assert decision_2.margin_cm < 0, f"Expected negative margin for onside, got {decision_2.margin_cm}cm"
    print(f"  [OK] Scenario 2 verified: ONSIDE ({decision_2.margin_cm:.1f} cm)")

    # 3. Test Active Decision Window
    assert detector.get_active_decision(210) is not None, "Active decision not found in display window"
    assert detector.get_active_decision(300) is None, "Active decision failed to expire"
    print("  [OK] Display frame window persistence verified.")

    # 4. Test 3D Perspective Projection & Frame Drawing
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    dummy_frame[:] = (35, 90, 45)  # Grass background

    # Create a realistic 4-point homography ViewTransformer
    src_pts = np.array([[200, 150], [1080, 150], [1200, 680], [80, 680]], dtype=np.float32)
    dst_pts = np.array([[0, 0], [12000, 0], [12000, 7000], [0, 7000]], dtype=np.float32)
    transformer = ViewTransformer(source=src_pts, target=dst_pts)

    var_frame = draw_offside_line_on_frame(
        frame=dummy_frame,
        decision=decision_1,
        view_transformer=transformer,
        config=config
    )
    assert var_frame.shape == (720, 1280, 3), "VAR frame shape mismatch"
    print("  [OK] draw_offside_line_on_frame 3D laser projection verified.")

    # 5. Test 2D Radar Snapshot
    snap_img = draw_offside_pitch_snapshot(
        config=config,
        decision=decision_1,
        player_xy_pitch=player_xy_1,
        player_ids=p_ids_1,
        player_teams=t_ids_1
    )
    assert snap_img.ndim == 3, "Snapshot image format invalid"
    print(f"  [OK] draw_offside_pitch_snapshot verified (Shape: {snap_img.shape}).")

    # 6. Save Test Outputs
    output_dir = os.path.join(os.path.dirname(__file__), 'test_output')
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, 'test_offside_var_frame.png'), var_frame)
    cv2.imwrite(os.path.join(output_dir, 'test_offside_pitch_snapshot.png'), snap_img)
    detector.get_summary_dataframe().to_csv(os.path.join(output_dir, 'test_offside_summary.csv'), index=False)
    print(f"  [OK] Test artifacts saved to '{output_dir}'")
    print(">>> ALL OFFSIDE (SAOT) & VAR TESTS PASSED!\n")


if __name__ == '__main__':
    test_offside_pipeline()
