import argparse
from enum import Enum
from typing import Iterator, List

import os

# Detect headless environment (e.g., Google Colab, CI servers)
HEADLESS = (os.name != 'nt') and (os.environ.get('DISPLAY') is None)
if HEADLESS:
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

from sports.annotators.football import (
    draw_pitch,
    draw_points_on_pitch,
    draw_pitch_heatmap,
    draw_passing_network,
    draw_player_speed_badges,
    draw_physical_dashboard,
    draw_match_analysis_screen,
)
from sports.common.ball import BallTracker, BallAnnotator
from sports.common.pass_detector import PassDetector, PassAnnotator
from sports.common.physical import PlayerSpeedTracker
from sports.common.tactical import HeatmapTracker, PassingNetworkAnalyzer
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from sports.configs.football import SoccerPitchConfiguration

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-player-detection.pt')
PITCH_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-pitch-detection.pt')
BALL_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-ball-detection.pt')

BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

STRIDE = 60
CONFIG = SoccerPitchConfiguration()

COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']
VERTEX_LABEL_ANNOTATOR = sv.VertexLabelAnnotator(
    color=[sv.Color.from_hex(color) for color in CONFIG.colors],
    text_color=sv.Color.from_hex('#FFFFFF'),
    border_radius=5,
    text_thickness=1,
    text_scale=0.5,
    text_padding=5,
)
EDGE_ANNOTATOR = sv.EdgeAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    thickness=2,
    edges=CONFIG.edges,
)
TRIANGLE_ANNOTATOR = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    base=20,
    height=15,
)
BOX_ANNOTATOR = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
BOX_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
)
ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)


class Mode(Enum):
    """
    Enum class representing different modes of operation for Soccer AI video analysis.
    """
    PITCH_DETECTION = 'PITCH_DETECTION'
    PLAYER_DETECTION = 'PLAYER_DETECTION'
    BALL_DETECTION = 'BALL_DETECTION'
    PLAYER_TRACKING = 'PLAYER_TRACKING'
    TEAM_CLASSIFICATION = 'TEAM_CLASSIFICATION'
    RADAR = 'RADAR'
    PASS_DETECTION = 'PASS_DETECTION'


def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    """
    Extract crops from the frame based on detected bounding boxes.

    Args:
        frame (np.ndarray): The frame from which to extract crops.
        detections (sv.Detections): Detected objects with bounding boxes.

    Returns:
        List[np.ndarray]: List of cropped images.
    """
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def resolve_goalkeepers_team_id(
    players: sv.Detections,
    players_team_id: np.array,
    goalkeepers: sv.Detections
) -> np.ndarray:
    """
    Resolve the team IDs for detected goalkeepers based on the proximity to team
    centroids.

    Args:
        players (sv.Detections): Detections of all players.
        players_team_id (np.array): Array containing team IDs of detected players.
        goalkeepers (sv.Detections): Detections of goalkeepers.

    Returns:
        np.ndarray: Array containing team IDs for the detected goalkeepers.

    This function calculates the centroids of the two teams based on the positions of
    the players. Then, it assigns each goalkeeper to the nearest team's centroid by
    calculating the distance between each goalkeeper and the centroids of the two teams.
    """
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0_centroid = players_xy[players_team_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players_team_id == 1].mean(axis=0)
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
    return np.array(goalkeepers_team_id)


def render_radar(
    detections: sv.Detections,
    keypoints: sv.KeyPoints,
    color_lookup: np.ndarray
) -> np.ndarray:
    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    transformer = ViewTransformer(
        source=keypoints.xy[0][mask].astype(np.float32),
        target=np.array(CONFIG.vertices)[mask].astype(np.float32)
    )
    xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    transformed_xy = transformer.transform_points(points=xy)

    radar = draw_pitch(config=CONFIG)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 0],
        face_color=sv.Color.from_hex(COLORS[0]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 1],
        face_color=sv.Color.from_hex(COLORS[1]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 2],
        face_color=sv.Color.from_hex(COLORS[2]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 3],
        face_color=sv.Color.from_hex(COLORS[3]), radius=20, pitch=radar)
    return radar


def run_pitch_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run pitch detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)

        annotated_frame = frame.copy()
        annotated_frame = VERTEX_LABEL_ANNOTATOR.annotate(
            annotated_frame, keypoints, CONFIG.labels)
        yield annotated_frame


def run_player_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run player detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        annotated_frame = frame.copy()
        annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = BOX_LABEL_ANNOTATOR.annotate(annotated_frame, detections)
        yield annotated_frame


def run_ball_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run ball detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    ball_tracker = BallTracker(buffer_size=20)
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)

    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    slicer = sv.InferenceSlicer(
        callback=callback,
        overlap_filter=sv.OverlapFilter.NONE,
        slice_wh=(640, 640),
    )

    for frame in frame_generator:
        detections = slicer(frame).with_nms(threshold=0.1)
        detections = ball_tracker.update(detections)
        annotated_frame = frame.copy()
        annotated_frame = ball_annotator.annotate(annotated_frame, detections)
        yield annotated_frame


def run_player_tracking(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run player tracking on a video and yield annotated frames with tracked players.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels=labels)
        yield annotated_frame


def run_team_classification(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run team classification on a video and yield annotated frames with team colors.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE)

    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = team_classifier.predict(crops)

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers)

        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        detections = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
                players_team_id.tolist() +
                goalkeepers_team_id.tolist() +
                [REFEREE_CLASS_ID] * len(referees)
        )
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels, custom_color_lookup=color_lookup)
        yield annotated_frame


def run_radar(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE)

    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = team_classifier.predict(crops)

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers)

        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        detections = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
            players_team_id.tolist() +
            goalkeepers_team_id.tolist() +
            [REFEREE_CLASS_ID] * len(referees)
        )
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels,
            custom_color_lookup=color_lookup)

        h, w, _ = frame.shape
        radar = render_radar(detections, keypoints, color_lookup)
        radar = sv.resize_image(radar, (w // 2, h // 2))
        radar_h, radar_w, _ = radar.shape
        rect = sv.Rect(
            x=w // 2 - radar_w // 2,
            y=h - radar_h,
            width=radar_w,
            height=radar_h
        )
        annotated_frame = sv.draw_image(annotated_frame, radar, opacity=0.5, rect=rect)
        yield annotated_frame


def run_pass_detection(
    source_video_path: str,
    device: str,
    reports_dir: str = 'reports',
    show_speed: bool = True,
) -> Iterator[np.ndarray]:
    """
    Run pass detection on a video and yield annotated frames.

    Combines player detection, tracking, team classification, ball detection,
    and pitch homography to feed a PassDetector FSM. Overlays pass arrows,
    ball trails, live HUD scoreboard, and real-time player speed badges.

    At the end of the video, exports tactical analytics (2D Heatmaps, Passing Network)
    and athletic physical performance metrics (Speed, Distance, Physical Dashboard).

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').
        reports_dir (str): Directory where analytics PNGs and CSVs are saved.
        show_speed (bool): Whether to render real-time speed badges on players.

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    # --- Load models ---
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)

    # --- Phase 1: Collect crops for team classifier fitting ---
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE)

    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops for team classifier'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    # --- Phase 2: Get video FPS for timing calculations ---
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    fps = video_info.fps if video_info.fps else 30.0

    # --- Initialize detectors, tactical & physical trackers, and annotators ---
    pass_detector = PassDetector(fps=fps)
    heatmap_tracker = HeatmapTracker(
        pitch_length_m=CONFIG.length / 100.0,
        pitch_width_m=CONFIG.width / 100.0
    )
    speed_tracker = PlayerSpeedTracker(fps=fps)

    pass_annotator = PassAnnotator(
        team_colors=(
            sv.Color.from_hex(COLORS[0]),
            sv.Color.from_hex(COLORS[1]),
        ),
    )

    player_tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    ball_tracker = BallTracker(buffer_size=20)

    def ball_callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    ball_slicer = sv.InferenceSlicer(
        callback=ball_callback,
        overlap_filter=sv.OverlapFilter.NONE,
        slice_wh=(640, 640),
    )

    # --- Phase 3: Process frames ---
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    frame_id = 0
    current_transformer = None

    for frame in frame_generator:
        # --- Pitch keypoints → ViewTransformer ---
        pitch_result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(pitch_result)

        mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
        if mask.sum() >= 4:
            try:
                current_transformer = ViewTransformer(
                    source=keypoints.xy[0][mask].astype(np.float32),
                    target=np.array(CONFIG.vertices)[mask].astype(np.float32)
                )
            except ValueError:
                pass  # Keep previous transformer if homography fails

        # --- Player detection + tracking + team classification ---
        player_result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(player_result)
        detections = player_tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        player_crops = get_crops(frame, players)
        players_team_id = team_classifier.predict(player_crops)

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers)

        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        all_detections = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
            players_team_id.tolist() +
            goalkeepers_team_id.tolist() +
            [REFEREE_CLASS_ID] * len(referees)
        )

        # --- Ball detection ---
        ball_detections = ball_slicer(frame).with_nms(threshold=0.1)
        ball_detections = ball_tracker.update(ball_detections)

        # --- Feed PassDetector, HeatmapTracker & PlayerSpeedTracker ---
        if current_transformer is not None:
            # Transform player positions to pitch coordinates (cm → meters)
            all_field_players = sv.Detections.merge([players, goalkeepers])
            all_field_team_ids = np.array(
                players_team_id.tolist() + goalkeepers_team_id.tolist()
            )

            if len(all_field_players) > 0 and all_field_players.tracker_id is not None:
                player_xy_pixel = all_field_players.get_anchors_coordinates(
                    sv.Position.BOTTOM_CENTER)
                player_xy_pitch_cm = current_transformer.transform_points(player_xy_pixel)
                player_xy_pitch_m = player_xy_pitch_cm / 100.0  # cm → meters

                player_ids = all_field_players.tracker_id

                # Ball position
                ball_xy_pitch_m = None
                if len(ball_detections) > 0:
                    ball_xy_pixel = ball_detections.get_anchors_coordinates(
                        sv.Position.CENTER)
                    ball_xy_pitch_cm = current_transformer.transform_points(ball_xy_pixel)
                    ball_xy_pitch_m = ball_xy_pitch_cm[0] / 100.0  # cm → meters

                # Update pass detection FSM
                pass_detector.update(
                    frame_id=frame_id,
                    player_xy_pitch=player_xy_pitch_m,
                    player_tracker_ids=player_ids,
                    player_team_ids=all_field_team_ids,
                    ball_xy_pitch=ball_xy_pitch_m,
                )

                # Update trajectory tracker for heatmaps
                heatmap_tracker.update(
                    player_xy_pitch=player_xy_pitch_m,
                    player_tracker_ids=player_ids,
                    player_team_ids=all_field_team_ids,
                )

                # Update speed & physical tracker
                speed_tracker.update(
                    frame_id=frame_id,
                    player_xy_pitch=player_xy_pitch_m,
                    player_tracker_ids=player_ids,
                    player_team_ids=all_field_team_ids,
                )

        # --- Annotate frame ---
        annotated_frame = frame.copy()

        # Player ellipses & speed badges
        if len(all_detections) > 0:
            if show_speed and current_transformer is not None:
                # Live Speed & Distance Badges (Abdullah Tarek style: feet ring + solid ID box + 2 lines of metrics)
                annotated_frame = draw_player_speed_badges(
                    annotated_frame, all_detections, speed_tracker,
                    custom_color_lookup=color_lookup, color_palette=COLORS
                )
            else:
                annotated_frame = ELLIPSE_ANNOTATOR.annotate(
                    annotated_frame, all_detections, custom_color_lookup=color_lookup)
                if all_detections.tracker_id is not None:
                    labels = [str(tid) if (i < len(color_lookup) and color_lookup[i] in (0, 1)) else '' for i, tid in enumerate(all_detections.tracker_id)]
                    annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
                        annotated_frame, all_detections, labels,
                        custom_color_lookup=color_lookup)

        # Pass detection overlays (arrows, trail, HUD)
        if current_transformer is not None:
            annotated_frame = pass_annotator.annotate(
                annotated_frame, pass_detector, current_transformer, frame_id)

        frame_id += 1
        yield annotated_frame

    # --- Post-Processing: Generate Reports & Match Analysis Screen ---
    # Wrap everything in try-except to ensure the match analysis screen
    # is ALWAYS yielded at the end of the video, even if report generation fails.
    import traceback as _tb

    match_analysis_img = None
    vid_w = video_info.width if video_info.width else 1920
    vid_h = video_info.height if video_info.height else 1080

    try:
        # --- Print summary after video ends ---
        df = pass_detector.get_events_dataframe()
        if not df.empty:
            print("\n" + "=" * 70)
            print("PASS DETECTION & POSSESSION SUMMARY")
            print("=" * 70)
            print(df.to_string(index=False))
            print(f"\nTeam 1 pass accuracy: {pass_detector.get_pass_accuracy(0):.1f}% | Possession: {pass_detector.get_possession_pct(0):.1f}%")
            print(f"Team 2 pass accuracy: {pass_detector.get_pass_accuracy(1):.1f}% | Possession: {pass_detector.get_possession_pct(1):.1f}%")
            print("=" * 70)
        else:
            print("\nNo pass events detected.")

        # --- Generate and Export Tactical & Physical Analytics ---
        os.makedirs(reports_dir, exist_ok=True)

        # 1. 2D Positional Heatmaps
        density_0 = heatmap_tracker.generate_density_grid(team_id=0)
        density_1 = heatmap_tracker.generate_density_grid(team_id=1)

        img_hm_0 = draw_pitch_heatmap(
            config=CONFIG,
            density_grid=density_0,
            title="Team 1 - Positional Heatmap",
            colormap=cv2.COLORMAP_JET
        )
        img_hm_1 = draw_pitch_heatmap(
            config=CONFIG,
            density_grid=density_1,
            title="Team 2 - Positional Heatmap",
            colormap=cv2.COLORMAP_HOT
        )

        hm0_path = os.path.join(reports_dir, "team_1_heatmap.png")
        hm1_path = os.path.join(reports_dir, "team_2_heatmap.png")
        cv2.imwrite(hm0_path, img_hm_0)
        cv2.imwrite(hm1_path, img_hm_1)

        # 2. Passing Networks
        passing_analyzer = PassingNetworkAnalyzer()
        passing_analyzer.update_events(pass_detector.events)
        net_0 = passing_analyzer.compute_network(team_id=0, heatmap_tracker=heatmap_tracker, min_passes=1)
        net_1 = passing_analyzer.compute_network(team_id=1, heatmap_tracker=heatmap_tracker, min_passes=1)

        img_net_0 = draw_passing_network(
            config=CONFIG,
            player_positions=net_0.player_positions,
            pass_connections=net_0.pass_connections,
            player_involvements=net_0.player_involvements,
            team_color=sv.Color.from_hex(COLORS[0]),
            team_name="Team 1",
            total_passes=net_0.total_completed_passes
        )
        img_net_1 = draw_passing_network(
            config=CONFIG,
            player_positions=net_1.player_positions,
            pass_connections=net_1.pass_connections,
            player_involvements=net_1.player_involvements,
            team_color=sv.Color.from_hex(COLORS[1]),
            team_name="Team 2",
            total_passes=net_1.total_completed_passes
        )

        net0_path = os.path.join(reports_dir, "team_1_passing_network.png")
        net1_path = os.path.join(reports_dir, "team_2_passing_network.png")
        cv2.imwrite(net0_path, img_net_0)
        cv2.imwrite(net1_path, img_net_1)

        # 3. Physical Performance & Speed Analytics
        df_physical = speed_tracker.get_summary_dataframe()
        phys_csv_path = os.path.join(reports_dir, "physical_stats.csv")
        if not df_physical.empty:
            df_physical.to_csv(phys_csv_path, index=False)

        team_0_phys = speed_tracker.get_team_stats(0)
        team_1_phys = speed_tracker.get_team_stats(1)
        img_physical_dash = draw_physical_dashboard(
            team_0_stats=team_0_phys,
            team_1_stats=team_1_phys,
            team_0_color=sv.Color.from_hex(COLORS[0]),
            team_1_color=sv.Color.from_hex(COLORS[1]),
            team_0_name="Team 1",
            team_1_name="Team 2",
        )
        phys_dash_path = os.path.join(reports_dir, "physical_dashboard.png")
        cv2.imwrite(phys_dash_path, img_physical_dash)

        # 4. Generate All-in-One Match Analysis Screen
        t0_completed_passes = sum(1 for e in pass_detector.events if e.event_type == "completed" and e.passer_team == 0)
        t1_completed_passes = sum(1 for e in pass_detector.events if e.event_type == "completed" and e.passer_team == 1)
        t0_total_dist = sum(s.total_distance_m for s in team_0_phys)
        t1_total_dist = sum(s.total_distance_m for s in team_1_phys)
        t0_peak_spd = max([s.max_speed_kmh for s in team_0_phys] + [0.0])
        t1_peak_spd = max([s.max_speed_kmh for s in team_1_phys] + [0.0])

        t0_stats = {
            'completed_passes': t0_completed_passes,
            'pass_accuracy': pass_detector.get_pass_accuracy(0),
            'possession_pct': pass_detector.get_possession_pct(0),
            'total_distance_m': t0_total_dist,
            'peak_speed_kmh': t0_peak_spd,
        }
        t1_stats = {
            'completed_passes': t1_completed_passes,
            'pass_accuracy': pass_detector.get_pass_accuracy(1),
            'possession_pct': pass_detector.get_possession_pct(1),
            'total_distance_m': t1_total_dist,
            'peak_speed_kmh': t1_peak_spd,
        }

        match_analysis_img = draw_match_analysis_screen(
            img_hm_0=img_hm_0,
            img_hm_1=img_hm_1,
            img_net_0=img_net_0,
            img_net_1=img_net_1,
            team_0_stats=t0_stats,
            team_1_stats=t1_stats,
            team_0_color=sv.Color.from_hex(COLORS[0]),
            team_1_color=sv.Color.from_hex(COLORS[1]),
            team_0_name="Team 1",
            team_1_name="Team 2",
            width=vid_w,
            height=vid_h,
        )
        match_analysis_path = os.path.join(reports_dir, "match_analysis_summary.png")
        cv2.imwrite(match_analysis_path, match_analysis_img)

        print("\n" + "=" * 70)
        print("ANALYTICS & PHYSICAL PERFORMANCE REPORTS GENERATED")
        print("=" * 70)
        print(f"Reports saved to folder: '{os.path.abspath(reports_dir)}'")
        print(f"  [+] Match Analysis Summary:  {match_analysis_path}")
        print(f"  [+] Heatmap Team 1:          {hm0_path}")
        print(f"  [+] Heatmap Team 2:          {hm1_path}")
        print(f"  [+] Passing Network Team 1:  {net0_path}")
        print(f"  [+] Passing Network Team 2:  {net1_path}")
        print(f"  [+] Physical Stats CSV:      {phys_csv_path}")
        print(f"  [+] Physical Dashboard:      {phys_dash_path}")
        if net_0.top_combinations:
            top_duos_0 = ", ".join([f"#{p1} <-> #{p2} ({cnt} passes)" for p1, p2, cnt in net_0.top_combinations[:3]])
            print(f"  [>] Team 1 Top Combinations: {top_duos_0}")
        if net_1.top_combinations:
            top_duos_1 = ", ".join([f"#{p1} <-> #{p2} ({cnt} passes)" for p1, p2, cnt in net_1.top_combinations[:3]])
            print(f"  [>] Team 2 Top Combinations: {top_duos_1}")
        print("=" * 70 + "\n")

    except Exception as exc:
        print("\n" + "!" * 70)
        print(f"[ERROR] Report generation failed: {exc}")
        _tb.print_exc()
        print("!" * 70 + "\n")

    # 5. ALWAYS append Match Analysis Screen to video (even if reports failed)
    if match_analysis_img is None:
        # Fallback: create simple text-only summary screen
        print("[WARN] Creating fallback match analysis screen...")
        match_analysis_img = np.zeros((vid_h, vid_w, 3), dtype=np.uint8)
        match_analysis_img[:] = (18, 22, 28)
        cv2.putText(match_analysis_img, "MATCH ANALYSIS", (vid_w // 2 - 200, vid_h // 2 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(match_analysis_img, "Report generation encountered an error",
                    (vid_w // 2 - 280, vid_h // 2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    summary_frames_count = int(fps * 8.0)
    print(f"\n[+] Appending Match Analysis Summary Screen ({summary_frames_count} frames ~ {summary_frames_count/fps:.1f}s) to end of video...")
    for _ in range(summary_frames_count):
        yield match_analysis_img
    print(f"[+] Match Analysis Screen appended successfully!")


def main(
    source_video_path: str,
    target_video_path: str,
    device: str,
    mode: Mode,
    reports_dir: str = 'reports',
    show_speed: bool = True
) -> None:
    if mode == Mode.PITCH_DETECTION:
        frame_generator = run_pitch_detection(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.PLAYER_DETECTION:
        frame_generator = run_player_detection(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.BALL_DETECTION:
        frame_generator = run_ball_detection(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.PLAYER_TRACKING:
        frame_generator = run_player_tracking(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.TEAM_CLASSIFICATION:
        frame_generator = run_team_classification(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.RADAR:
        frame_generator = run_radar(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.PASS_DETECTION:
        frame_generator = run_pass_detection(
            source_video_path=source_video_path,
            device=device,
            reports_dir=reports_dir,
            show_speed=show_speed
        )
    else:
        raise NotImplementedError(f"Mode {mode} is not implemented.")

    video_info = sv.VideoInfo.from_video_path(source_video_path)
    if mode == Mode.PASS_DETECTION:
        # Extend total_frames in VideoInfo to account for 8s Match Analysis summary
        summary_frames = int(video_info.fps * 8.0)
        tot = (video_info.total_frames + summary_frames) if video_info.total_frames else None
        video_info = sv.VideoInfo(
            width=video_info.width,
            height=video_info.height,
            fps=video_info.fps,
            total_frames=tot
        )

    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame in frame_generator:
            sink.write_frame(frame)

            if not HEADLESS:
                cv2.imshow("frame", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        if not HEADLESS:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Soccer AI Video Analysis')
    parser.add_argument('--source_video_path', type=str, required=True)
    parser.add_argument('--target_video_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--mode', type=Mode, default=Mode.PLAYER_DETECTION)
    parser.add_argument('--reports_dir', type=str, default='reports')
    parser.add_argument('--no-speed', dest='show_speed', action='store_false', help='Disable live speed badges on players')
    parser.set_defaults(show_speed=True)
    args = parser.parse_args()
    main(
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        device=args.device,
        mode=args.mode,
        reports_dir=args.reports_dir,
        show_speed=args.show_speed
    )
