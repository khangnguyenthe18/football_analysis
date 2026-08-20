"""
Unit and functional test for Physical & Speed Analytics Engine.
"""

import os
import sys
import numpy as np
import cv2
import supervision as sv

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sports.common.physical import PlayerSpeedTracker, WorkrateZone
from sports.annotators.football import draw_player_speed_badges, draw_physical_dashboard


def test_physical_pipeline():
    print(">>> Testing Physical & Speed Analytics Engine...")
    fps = 30.0
    tracker = PlayerSpeedTracker(fps=fps, window_size=7)

    # 1. Test Player Simulation:
    # Player 1: Standing stationary with slight jitter (+/- 0.05m) -> Speed should be 0 km/h (Deadband)
    # Player 2: Jogging at constant ~12 km/h (~3.33 m/s) -> 3.33 / 30 = 0.111m per frame
    # Player 3: Sprinting burst at ~28.8 km/h (~8.0 m/s) -> 8.0 / 30 = 0.267m per frame
    np.random.seed(42)

    p1_pos = [30.0, 30.0]
    p2_pos = [20.0, 20.0]
    p3_pos = [10.0, 50.0]

    for frame_id in range(60):  # 2.0 seconds at 30 fps
        # Player 1 micro-jitter
        p1_x = p1_pos[0] + np.random.uniform(-0.02, 0.02)
        p1_y = p1_pos[1] + np.random.uniform(-0.02, 0.02)

        # Player 2 jogging rightwards
        p2_pos[0] += 3.33 / fps
        p2_x, p2_y = p2_pos[0], p2_pos[1]

        # Player 3 sprinting rightwards
        p3_pos[0] += 8.0 / fps
        p3_x, p3_y = p3_pos[0], p3_pos[1]

        xy_pitch = np.array([[p1_x, p1_y], [p2_x, p2_y], [p3_x, p3_y]], dtype=np.float32)
        p_ids = np.array([1, 2, 3])
        t_ids = np.array([0, 0, 1])

        tracker.update(frame_id, xy_pitch, p_ids, t_ids)

    # 2. Validate Speeds & Stats
    stats_1 = tracker.get_player_stats(1)
    stats_2 = tracker.get_player_stats(2)
    stats_3 = tracker.get_player_stats(3)

    assert stats_1 is not None and stats_2 is not None and stats_3 is not None, "Stats retrieval failed"

    # Check Player 1 (Stationary)
    assert stats_1.current_speed_kmh < 1.0, f"Stationary player has unexpected speed: {stats_1.current_speed_kmh}"
    assert stats_1.total_distance_m < 0.5, f"Stationary player accumulated noise distance: {stats_1.total_distance_m}"
    print(f"  [OK] Stationary player deadband verified: Speed = {stats_1.current_speed_kmh} km/h, Dist = {stats_1.total_distance_m}m")

    # Check Player 2 (Jogging ~12 km/h)
    assert 8.0 <= stats_2.avg_speed_kmh <= 14.0, f"Jogging speed mismatch: {stats_2.avg_speed_kmh}"
    assert 4.0 <= stats_2.total_distance_m <= 8.0, f"Jogging distance mismatch: {stats_2.total_distance_m}"
    print(f"  [OK] Jogging player verified: Avg Speed = {stats_2.avg_speed_kmh} km/h, Dist = {stats_2.total_distance_m}m")

    # Check Player 3 (Sprinting ~28.8 km/h)
    assert stats_3.max_speed_kmh >= 25.2, f"Sprint peak speed not detected: {stats_3.max_speed_kmh}"
    assert stats_3.sprint_count >= 1, f"Sprint burst count failed to trigger: {stats_3.sprint_count}"
    assert tracker.is_player_sprinting(3), "is_player_sprinting returned False for sprinting player"
    print(f"  [OK] Sprinting player verified: Peak Speed = {stats_3.max_speed_kmh} km/h, Sprints = {stats_3.sprint_count}")

    # 3. Test DataFrame Export
    df = tracker.get_summary_dataframe()
    assert len(df) == 3, f"Expected 3 rows in DataFrame, got {len(df)}"
    print("  [OK] get_summary_dataframe verified.")

    # 4. Test draw_player_speed_badges
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    dummy_xyxy = np.array([
        [200, 300, 260, 420],
        [400, 300, 460, 420],
        [700, 300, 760, 420],
    ])
    detections = sv.Detections(
        xyxy=dummy_xyxy,
        tracker_id=np.array([1, 2, 3]),
        class_id=np.array([2, 2, 2])
    )
    badge_frame = draw_player_speed_badges(
        dummy_frame, detections, tracker,
        custom_color_lookup=np.array([0, 0, 1])
    )
    assert badge_frame.shape == (720, 1280, 3), "Badge frame dimension mismatch"
    print("  [OK] draw_player_speed_badges verified.")

    # 5. Test draw_physical_dashboard
    t0_stats = tracker.get_team_stats(0)
    t1_stats = tracker.get_team_stats(1)
    dash_img = draw_physical_dashboard(
        team_0_stats=t0_stats,
        team_1_stats=t1_stats,
        team_0_name="Manchester Red",
        team_1_name="Manchester Blue",
    )
    assert dash_img.shape == (700, 1200, 3), f"Dashboard shape mismatch: {dash_img.shape}"
    print("  [OK] draw_physical_dashboard verified.")

    # 6. Save test outputs
    output_dir = os.path.join(os.path.dirname(__file__), 'test_output')
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, 'test_speed_badges.png'), badge_frame)
    cv2.imwrite(os.path.join(output_dir, 'test_physical_dashboard.png'), dash_img)
    df.to_csv(os.path.join(output_dir, 'test_physical_stats.csv'), index=False)
    print(f"  [OK] Outputs saved to '{output_dir}'")
    print(">>> ALL PHYSICAL & SPEED ANALYTICS TESTS PASSED!\n")


if __name__ == '__main__':
    test_physical_pipeline()
