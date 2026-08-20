"""
Unit and functional test for Tactical Analytics (Heatmap & Passing Network).
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
from sports.common.tactical import HeatmapTracker, PassingNetworkAnalyzer
from sports.annotators.football import draw_pitch_heatmap, draw_passing_network, draw_pitch


def test_tactical_pipeline():
    print(">>> Testing Tactical Analytics Engine...")
    config = SoccerPitchConfiguration()

    # 1. Test HeatmapTracker
    tracker = HeatmapTracker(pitch_length_m=120.0, pitch_width_m=70.0)

    # Simulate 50 frames of player movement for Team 0 and Team 1
    np.random.seed(42)
    t0_players = [1, 2, 3, 4, 5]
    t1_players = [10, 11, 12, 13, 14]

    for frame_idx in range(50):
        # Team 0 clustered around (40, 35)
        xy_t0 = np.random.normal(loc=[40.0, 35.0], scale=[10.0, 8.0], size=(len(t0_players), 2))
        # Team 1 clustered around (80, 35)
        xy_t1 = np.random.normal(loc=[80.0, 35.0], scale=[12.0, 10.0], size=(len(t1_players), 2))

        all_xy = np.vstack([xy_t0, xy_t1])
        all_ids = np.array(t0_players + t1_players)
        all_teams = np.array([0] * len(t0_players) + [1] * len(t1_players))

        tracker.update(all_xy, all_ids, all_teams)

    # Test density generation
    density_0 = tracker.generate_density_grid(team_id=0)
    density_1 = tracker.generate_density_grid(team_id=1)

    assert density_0.shape == (700, 1200), f"Density shape mismatch: {density_0.shape}"
    assert 0.0 <= np.max(density_0) <= 1.0, f"Density values out of bounds: max={np.max(density_0)}"
    print("  [OK] HeatmapTracker density generation verified.")

    # 2. Test draw_pitch_heatmap
    hm_img_0 = draw_pitch_heatmap(config, density_0, title="Team 1 - Heatmap", colormap=cv2.COLORMAP_JET)
    assert hm_img_0.ndim == 3 and hm_img_0.shape[2] == 3, "Heatmap image format invalid"
    print(f"  [OK] draw_pitch_heatmap verified (Output shape: {hm_img_0.shape}).")

    # 3. Test PassingNetworkAnalyzer
    analyzer = PassingNetworkAnalyzer()

    # Create mock completed pass events for Team 0
    # Player 1 -> Player 2 (3 times)
    # Player 2 -> Player 3 (4 times)
    # Player 3 -> Player 1 (2 times)
    # Player 4 -> Player 5 (1 time)
    mock_events = [
        PassEvent(1, 10, 20, 1, 0, 2, 0, (30.0, 20.0), (45.0, 30.0), 18.0, 0.33, EventType.COMPLETED.value),
        PassEvent(2, 30, 40, 1, 0, 2, 0, (32.0, 22.0), (46.0, 28.0), 15.0, 0.33, EventType.COMPLETED.value),
        PassEvent(3, 50, 60, 1, 0, 2, 0, (31.0, 21.0), (44.0, 32.0), 17.0, 0.33, EventType.COMPLETED.value),
        PassEvent(4, 70, 80, 2, 0, 3, 0, (48.0, 30.0), (65.0, 45.0), 22.0, 0.33, EventType.COMPLETED.value),
        PassEvent(5, 90, 100, 2, 0, 3, 0, (47.0, 31.0), (66.0, 44.0), 21.0, 0.33, EventType.COMPLETED.value),
        PassEvent(6, 110, 120, 2, 0, 3, 0, (46.0, 29.0), (64.0, 43.0), 20.0, 0.33, EventType.COMPLETED.value),
        PassEvent(7, 130, 140, 2, 0, 3, 0, (48.0, 32.0), (67.0, 46.0), 23.0, 0.33, EventType.COMPLETED.value),
        PassEvent(8, 150, 160, 3, 0, 1, 0, (65.0, 45.0), (35.0, 25.0), 36.0, 0.33, EventType.COMPLETED.value),
        PassEvent(9, 170, 180, 3, 0, 1, 0, (64.0, 44.0), (34.0, 24.0), 35.0, 0.33, EventType.COMPLETED.value),
        PassEvent(10, 190, 200, 4, 0, 5, 0, (20.0, 50.0), (40.0, 55.0), 20.6, 0.33, EventType.COMPLETED.value),
    ]

    analyzer.update_events(mock_events)
    net_data = analyzer.compute_network(team_id=0, heatmap_tracker=tracker, min_passes=1)

    assert net_data.total_completed_passes == 10, f"Expected 10 passes, got {net_data.total_completed_passes}"
    assert len(net_data.top_combinations) >= 3, "Expected at least 3 combinations"
    assert net_data.top_combinations[0] == (2, 3, 4), f"Top combination mismatch: {net_data.top_combinations[0]}"
    print("  [OK] PassingNetworkAnalyzer metrics verified.")

    # 4. Test draw_passing_network
    net_img = draw_passing_network(
        config=config,
        player_positions=net_data.player_positions,
        pass_connections=net_data.pass_connections,
        player_involvements=net_data.player_involvements,
        team_color=sv.Color.from_hex('#FF1493'),
        team_name="Team 1",
        total_passes=net_data.total_completed_passes,
    )
    assert net_img.ndim == 3 and net_img.shape[2] == 3, "Passing network image format invalid"
    print(f"  [OK] draw_passing_network verified (Output shape: {net_img.shape}).")

    # 5. Save test outputs to scratch
    output_dir = os.path.join(os.path.dirname(__file__), 'test_output')
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, 'test_heatmap_team0.png'), hm_img_0)
    cv2.imwrite(os.path.join(output_dir, 'test_passing_network_team0.png'), net_img)
    print(f"  [OK] Sample images saved to '{output_dir}'")
    print(">>> ALL TACTICAL ANALYTICS TESTS PASSED!\n")


if __name__ == '__main__':
    test_tactical_pipeline()
