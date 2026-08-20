from typing import Optional, List, Dict, Tuple, Any

import cv2
import supervision as sv
import numpy as np

from sports.configs.football import SoccerPitchConfiguration


def draw_pitch(
    config: SoccerPitchConfiguration,
    background_color: sv.Color = sv.Color(34, 139, 34),
    line_color: sv.Color = sv.Color.WHITE,
    padding: int = 50,
    line_thickness: int = 4,
    point_radius: int = 8,
    scale: float = 0.1
) -> np.ndarray:
    """
    Draws a soccer pitch with specified dimensions, colors, and scale.

    Args:
        config (SoccerPitchConfiguration): Configuration object containing the
            dimensions and layout of the pitch.
        background_color (sv.Color, optional): Color of the pitch background.
            Defaults to sv.Color(34, 139, 34).
        line_color (sv.Color, optional): Color of the pitch lines.
            Defaults to sv.Color.WHITE.
        padding (int, optional): Padding around the pitch in pixels.
            Defaults to 50.
        line_thickness (int, optional): Thickness of the pitch lines in pixels.
            Defaults to 4.
        point_radius (int, optional): Radius of the penalty spot points in pixels.
            Defaults to 8.
        scale (float, optional): Scaling factor for the pitch dimensions.
            Defaults to 0.1.

    Returns:
        np.ndarray: Image of the soccer pitch.
    """
    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)
    scaled_circle_radius = int(config.centre_circle_radius * scale)
    scaled_penalty_spot_distance = int(config.penalty_spot_distance * scale)

    pitch_image = np.ones(
        (scaled_width + 2 * padding,
         scaled_length + 2 * padding, 3),
        dtype=np.uint8
    ) * np.array(background_color.as_bgr(), dtype=np.uint8)

    for start, end in config.edges:
        point1 = (int(config.vertices[start - 1][0] * scale) + padding,
                  int(config.vertices[start - 1][1] * scale) + padding)
        point2 = (int(config.vertices[end - 1][0] * scale) + padding,
                  int(config.vertices[end - 1][1] * scale) + padding)
        cv2.line(
            img=pitch_image,
            pt1=point1,
            pt2=point2,
            color=line_color.as_bgr(),
            thickness=line_thickness
        )

    centre_circle_center = (
        scaled_length // 2 + padding,
        scaled_width // 2 + padding
    )
    cv2.circle(
        img=pitch_image,
        center=centre_circle_center,
        radius=scaled_circle_radius,
        color=line_color.as_bgr(),
        thickness=line_thickness
    )

    penalty_spots = [
        (
            scaled_penalty_spot_distance + padding,
            scaled_width // 2 + padding
        ),
        (
            scaled_length - scaled_penalty_spot_distance + padding,
            scaled_width // 2 + padding
        )
    ]
    for spot in penalty_spots:
        cv2.circle(
            img=pitch_image,
            center=spot,
            radius=point_radius,
            color=line_color.as_bgr(),
            thickness=-1
        )

    return pitch_image


def draw_points_on_pitch(
    config: SoccerPitchConfiguration,
    xy: np.ndarray,
    face_color: sv.Color = sv.Color.RED,
    edge_color: sv.Color = sv.Color.BLACK,
    radius: int = 10,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Draws points on a soccer pitch.

    Args:
        config (SoccerPitchConfiguration): Configuration object containing the
            dimensions and layout of the pitch.
        xy (np.ndarray): Array of points to be drawn, with each point represented by
            its (x, y) coordinates.
        face_color (sv.Color, optional): Color of the point faces.
            Defaults to sv.Color.RED.
        edge_color (sv.Color, optional): Color of the point edges.
            Defaults to sv.Color.BLACK.
        radius (int, optional): Radius of the points in pixels.
            Defaults to 10.
        thickness (int, optional): Thickness of the point edges in pixels.
            Defaults to 2.
        padding (int, optional): Padding around the pitch in pixels.
            Defaults to 50.
        scale (float, optional): Scaling factor for the pitch dimensions.
            Defaults to 0.1.
        pitch (Optional[np.ndarray], optional): Existing pitch image to draw points on.
            If None, a new pitch will be created. Defaults to None.

    Returns:
        np.ndarray: Image of the soccer pitch with points drawn on it.
    """
    if pitch is None:
        pitch = draw_pitch(
            config=config,
            padding=padding,
            scale=scale
        )

    for point in xy:
        scaled_point = (
            int(point[0] * scale) + padding,
            int(point[1] * scale) + padding
        )
        cv2.circle(
            img=pitch,
            center=scaled_point,
            radius=radius,
            color=face_color.as_bgr(),
            thickness=-1
        )
        cv2.circle(
            img=pitch,
            center=scaled_point,
            radius=radius,
            color=edge_color.as_bgr(),
            thickness=thickness
        )

    return pitch


def draw_paths_on_pitch(
    config: SoccerPitchConfiguration,
    paths: List[np.ndarray],
    color: sv.Color = sv.Color.WHITE,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Draws paths on a soccer pitch.

    Args:
        config (SoccerPitchConfiguration): Configuration object containing the
            dimensions and layout of the pitch.
        paths (List[np.ndarray]): List of paths, where each path is an array of (x, y)
            coordinates.
        color (sv.Color, optional): Color of the paths.
            Defaults to sv.Color.WHITE.
        thickness (int, optional): Thickness of the paths in pixels.
            Defaults to 2.
        padding (int, optional): Padding around the pitch in pixels.
            Defaults to 50.
        scale (float, optional): Scaling factor for the pitch dimensions.
            Defaults to 0.1.
        pitch (Optional[np.ndarray], optional): Existing pitch image to draw paths on.
            If None, a new pitch will be created. Defaults to None.

    Returns:
        np.ndarray: Image of the soccer pitch with paths drawn on it.
    """
    if pitch is None:
        pitch = draw_pitch(
            config=config,
            padding=padding,
            scale=scale
        )

    for path in paths:
        scaled_path = [
            (
                int(point[0] * scale) + padding,
                int(point[1] * scale) + padding
            )
            for point in path if point.size > 0
        ]

        if len(scaled_path) < 2:
            continue

        for i in range(len(scaled_path) - 1):
            cv2.line(
                img=pitch,
                pt1=scaled_path[i],
                pt2=scaled_path[i + 1],
                color=color.as_bgr(),
                thickness=thickness
            )

        return pitch


def draw_pitch_voronoi_diagram(
    config: SoccerPitchConfiguration,
    team_1_xy: np.ndarray,
    team_2_xy: np.ndarray,
    team_1_color: sv.Color = sv.Color.RED,
    team_2_color: sv.Color = sv.Color.WHITE,
    opacity: float = 0.5,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Draws a Voronoi diagram on a soccer pitch representing the control areas of two
    teams.

    Args:
        config (SoccerPitchConfiguration): Configuration object containing the
            dimensions and layout of the pitch.
        team_1_xy (np.ndarray): Array of (x, y) coordinates representing the positions
            of players in team 1.
        team_2_xy (np.ndarray): Array of (x, y) coordinates representing the positions
            of players in team 2.
        team_1_color (sv.Color, optional): Color representing the control area of
            team 1. Defaults to sv.Color.RED.
        team_2_color (sv.Color, optional): Color representing the control area of
            team 2. Defaults to sv.Color.WHITE.
        opacity (float, optional): Opacity of the Voronoi diagram overlay.
            Defaults to 0.5.
        padding (int, optional): Padding around the pitch in pixels.
            Defaults to 50.
        scale (float, optional): Scaling factor for the pitch dimensions.
            Defaults to 0.1.
        pitch (Optional[np.ndarray], optional): Existing pitch image to draw the
            Voronoi diagram on. If None, a new pitch will be created. Defaults to None.

    Returns:
        np.ndarray: Image of the soccer pitch with the Voronoi diagram overlay.
    """
    if pitch is None:
        pitch = draw_pitch(
            config=config,
            padding=padding,
            scale=scale
        )

    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)

    voronoi = np.zeros_like(pitch, dtype=np.uint8)

    team_1_color_bgr = np.array(team_1_color.as_bgr(), dtype=np.uint8)
    team_2_color_bgr = np.array(team_2_color.as_bgr(), dtype=np.uint8)

    y_coordinates, x_coordinates = np.indices((
        scaled_width + 2 * padding,
        scaled_length + 2 * padding
    ))

    y_coordinates -= padding
    x_coordinates -= padding

    def calculate_distances(xy, x_coordinates, y_coordinates):
        return np.sqrt((xy[:, 0][:, None, None] * scale - x_coordinates) ** 2 +
                       (xy[:, 1][:, None, None] * scale - y_coordinates) ** 2)

    distances_team_1 = calculate_distances(team_1_xy, x_coordinates, y_coordinates)
    distances_team_2 = calculate_distances(team_2_xy, x_coordinates, y_coordinates)

    min_distances_team_1 = np.min(distances_team_1, axis=0)
    min_distances_team_2 = np.min(distances_team_2, axis=0)

    control_mask = min_distances_team_1 < min_distances_team_2

    voronoi[control_mask] = team_1_color_bgr
    voronoi[~control_mask] = team_2_color_bgr

    overlay = cv2.addWeighted(voronoi, opacity, pitch, 1 - opacity, 0)

    return overlay


def draw_pitch_heatmap(
    config: SoccerPitchConfiguration,
    density_grid: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
    opacity: float = 0.65,
    threshold: float = 0.08,
    padding: int = 50,
    scale: float = 0.1,
    title: Optional[str] = None,
    pitch: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Renders a smooth 2D density heatmap on top of a soccer pitch diagram.

    Args:
        config: Pitch configuration with metric dimensions.
        density_grid: 2D numpy array [0.0, 1.0] representing spatial density.
        colormap: OpenCV colormap (default cv2.COLORMAP_JET).
        opacity: Heatmap overlay blending factor (0.0 to 1.0).
        threshold: Minimum density value to display (below this, pitch grass is visible).
        padding: Padding around pitch in pixels.
        scale: Scale factor for pitch dimensions.
        title: Optional title string to display on top banner.
        pitch: Optional existing pitch image to draw on.

    Returns:
        np.ndarray: Pitch image with heatmap overlay.
    """
    if pitch is None:
        pitch = draw_pitch(config=config, padding=padding, scale=scale)

    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)

    if density_grid is None or density_grid.size == 0 or np.max(density_grid) == 0:
        return pitch

    # Resize density grid to match inner pitch dimensions
    grid_normalized = np.clip(density_grid, 0.0, 1.0)
    resized_grid = cv2.resize(
        grid_normalized,
        (scaled_length, scaled_width),
        interpolation=cv2.INTER_CUBIC
    )

    # Colorize using OpenCV colormap
    grid_u8 = (resized_grid * 255).astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(grid_u8, colormap)

    # Place colored heatmap inside padded pitch region
    full_overlay = pitch.copy()
    inner_roi = full_overlay[padding:padding + scaled_width, padding:padding + scaled_length]

    # Create smooth mask based on threshold
    mask = (resized_grid >= threshold).astype(np.float32)[:, :, None]

    # Alpha blend inner region
    blended_inner = (
        colored_heatmap * (opacity * mask) +
        inner_roi * (1.0 - opacity * mask)
    ).astype(np.uint8)

    full_overlay[padding:padding + scaled_width, padding:padding + scaled_length] = blended_inner

    # Redraw pitch lines on top for crisp field markings
    line_color = sv.Color.WHITE.as_bgr()
    for start, end in config.edges:
        pt1 = (int(config.vertices[start - 1][0] * scale) + padding,
               int(config.vertices[start - 1][1] * scale) + padding)
        pt2 = (int(config.vertices[end - 1][0] * scale) + padding,
               int(config.vertices[end - 1][1] * scale) + padding)
        cv2.line(full_overlay, pt1, pt2, line_color, 2, cv2.LINE_AA)

    # Centre circle
    centre_circle_center = (
        scaled_length // 2 + padding,
        scaled_width // 2 + padding
    )
    cv2.circle(
        full_overlay, centre_circle_center,
        int(config.centre_circle_radius * scale),
        line_color, 2, cv2.LINE_AA
    )

    # Penalty spots
    penalty_spots = [
        (int(config.penalty_spot_distance * scale) + padding, scaled_width // 2 + padding),
        (scaled_length - int(config.penalty_spot_distance * scale) + padding, scaled_width // 2 + padding)
    ]
    for spot in penalty_spots:
        cv2.circle(full_overlay, spot, 5, line_color, -1, cv2.LINE_AA)

    # Optional Title Banner
    if title:
        h, w = full_overlay.shape[:2]
        banner_h = 36
        banner = full_overlay.copy()
        cv2.rectangle(banner, (0, 0), (w, banner_h), (20, 20, 20), -1)
        cv2.addWeighted(banner, 0.75, full_overlay, 0.25, 0, full_overlay)
        cv2.putText(
            full_overlay, title, (20, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA
        )

    return full_overlay


def draw_passing_network(
    config: SoccerPitchConfiguration,
    player_positions: Dict[int, Tuple[float, float]],
    pass_connections: List[Tuple[int, int, int]],
    player_involvements: Dict[int, int],
    team_color: sv.Color = sv.Color.from_hex('#FF1493'),
    team_name: str = "Team",
    total_passes: int = 0,
    min_node_radius: int = 14,
    max_node_radius: int = 28,
    min_edge_thickness: int = 2,
    max_edge_thickness: int = 8,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Renders a tactical passing network diagram on the 2D soccer pitch.

    Args:
        config: Pitch configuration with metric dimensions.
        player_positions: Mapping of player_id -> (x_m, y_m) average position in meters.
        pass_connections: List of (player_a, player_b, pass_count) tuples.
        player_involvements: Mapping of player_id -> total pass count.
        team_color: Color representing the team nodes.
        team_name: Name of the team for the header banner.
        total_passes: Total completed passes count.
        min_node_radius: Minimum player circle radius.
        max_node_radius: Maximum player circle radius.
        min_edge_thickness: Minimum pass connection line thickness.
        max_edge_thickness: Maximum pass connection line thickness.
        padding: Padding around pitch in pixels.
        scale: Scale factor for pitch dimensions.
        pitch: Optional existing pitch image.

    Returns:
        np.ndarray: Annotated passing network diagram image.
    """
    if pitch is None:
        pitch = draw_pitch(
            config=config,
            background_color=sv.Color(28, 33, 39),  # Modern dark slate pitch
            line_color=sv.Color(120, 130, 140),
            padding=padding,
            scale=scale
        )

    img = pitch.copy()
    color_bgr = team_color.as_bgr()

    # Convert meter positions to pixel coordinates
    pixel_positions: Dict[int, Tuple[int, int]] = {}
    for pid, (xm, ym) in player_positions.items():
        px = int(xm * 100 * scale) + padding
        py = int(ym * 100 * scale) + padding
        pixel_positions[pid] = (px, py)

    # 1. Draw Pass Connections (Edges)
    if pass_connections:
        max_passes = max([c[2] for c in pass_connections] + [1])
        for p1, p2, count in pass_connections:
            if p1 not in pixel_positions or p2 not in pixel_positions:
                continue

            pt1 = pixel_positions[p1]
            pt2 = pixel_positions[p2]

            # Thickness proportional to pass count
            rel_strength = count / max_passes
            thickness = int(min_edge_thickness + rel_strength * (max_edge_thickness - min_edge_thickness))

            # Edge line with slight transparency
            edge_overlay = img.copy()
            cv2.line(edge_overlay, pt1, pt2, color_bgr, thickness, cv2.LINE_AA)
            cv2.addWeighted(edge_overlay, 0.75, img, 0.25, 0, img)

            # Draw small pass count badge midway if passes >= 2
            if count >= 2:
                mid_x = (pt1[0] + pt2[0]) // 2
                mid_y = (pt1[1] + pt2[1]) // 2
                cv2.circle(img, (mid_x, mid_y), 9, (20, 20, 20), -1, cv2.LINE_AA)
                cv2.circle(img, (mid_x, mid_y), 9, color_bgr, 1, cv2.LINE_AA)
                cv2.putText(
                    img, str(count), (mid_x - 4, mid_y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA
                )

    # 2. Draw Player Nodes
    max_inv = max(list(player_involvements.values()) + [1])
    for pid, pt in pixel_positions.items():
        inv = player_involvements.get(pid, 1)
        rel_inv = inv / max_inv
        radius = int(min_node_radius + rel_inv * (max_node_radius - min_node_radius))

        # Node shadow/glow
        cv2.circle(img, pt, radius + 3, (10, 10, 10), -1, cv2.LINE_AA)
        # Node body
        cv2.circle(img, pt, radius, color_bgr, -1, cv2.LINE_AA)
        # Node border
        cv2.circle(img, pt, radius, (255, 255, 255), 2, cv2.LINE_AA)

        # Player ID label inside node
        label = str(pid)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        text_x = pt[0] - text_size[0] // 2
        text_y = pt[1] + text_size[1] // 2
        cv2.putText(
            img, label, (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
        )

    # 3. Header Title & Stats Banner
    h, w = img.shape[:2]
    banner_h = 44
    banner = img.copy()
    cv2.rectangle(banner, (0, 0), (w, banner_h), (15, 18, 22), -1)
    cv2.addWeighted(banner, 0.85, img, 0.15, 0, img)
    cv2.rectangle(img, (0, 0), (w, banner_h), (60, 65, 75), 1)

    # Team color indicator bar
    cv2.rectangle(img, (12, 10), (18, banner_h - 10), color_bgr, -1)

    title_text = f"{team_name.upper()} - PASSING NETWORK"
    cv2.putText(
        img, title_text, (28, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA
    )

    stats_text = f"Completed Passes: {total_passes}  |  Active Players: {len(pixel_positions)}"
    stats_size = cv2.getTextSize(stats_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
    cv2.putText(
        img, stats_text, (w - stats_size[0] - 20, 27),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 200, 210), 1, cv2.LINE_AA
    )

    return img


def draw_player_speed_badges(
    frame: np.ndarray,
    detections: sv.Detections,
    speed_tracker: Any,
    custom_color_lookup: Optional[np.ndarray] = None,
    color_palette: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Renders Abdullah Tarek style player visual badges:
    - Ellipse / semi-circle ring at feet
    - Solid Team ID rectangle with crisp black bold player ID
    - Line 1: '{speed:.2f} km/h'
    - Line 2: '{distance:.2f} m'

    Args:
        frame: The video frame to annotate.
        detections: Player detections containing bounding boxes and tracker IDs.
        speed_tracker: PlayerSpeedTracker instance.
        custom_color_lookup: Optional array mapping detection index to team/color index.
        color_palette: Hex color list for teams.

    Returns:
        np.ndarray: Annotated video frame.
    """
    if len(detections) == 0 or detections.tracker_id is None:
        return frame

    annotated = frame.copy()

    # Màu cố định cho 2 đội: Đội 1 = Hồng (#FF1493), Đội 2 = Xanh (#00BFFF)
    palette = [
        sv.Color.from_hex(color_palette[0]).as_bgr() if color_palette and len(color_palette) > 0 else sv.Color.from_hex('#FF1493').as_bgr(),
        sv.Color.from_hex(color_palette[1]).as_bgr() if color_palette and len(color_palette) > 1 else sv.Color.from_hex('#00BFFF').as_bgr()
    ]

    bottom_anchors = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

    for i, (tid, anchor) in enumerate(zip(detections.tracker_id, bottom_anchors)):
        team_idx = int(custom_color_lookup[i]) if custom_color_lookup is not None and i < len(custom_color_lookup) else 0

        # 1. Bỏ qua hoàn toàn trọng tài (Trọng tài có team_idx không thuộc (0, 1) hoặc class_id == 3)
        if team_idx not in (0, 1):
            continue
        if detections.class_id is not None and detections.class_id[i] == 3:  # REFEREE_CLASS_ID = 3
            continue

        tid = int(tid)
        speed = speed_tracker.get_player_speed_kmh(tid)
        dist = speed_tracker.get_player_distance_m(tid)

        team_bgr = palette[team_idx % len(palette)]
        ax, ay = int(anchor[0]), int(anchor[1])

        # A. Vòng bán nguyệt / ellipse dưới chân cầu thủ
        ellipse_rx = 20
        ellipse_ry = 9
        cv2.ellipse(annotated, (ax, ay), (ellipse_rx, ellipse_ry), 0, -30, 210, team_bgr, 2, cv2.LINE_AA)

        # B. Khung ID hình chữ nhật viền đen, nền màu đội bóng
        id_str = str(tid)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale_id = 0.52
        thickness_id = 2
        (id_w, id_h), _ = cv2.getTextSize(id_str, font, font_scale_id, thickness_id)

        badge_w = max(34, id_w + 14)
        badge_h = 22
        bx1 = ax - badge_w // 2
        by1 = ay + 2
        bx2 = bx1 + badge_w
        by2 = by1 + badge_h

        # Khung chữ nhật màu đội
        cv2.rectangle(annotated, (bx1, by1), (bx2, by2), team_bgr, -1)
        # Viền đen 1px
        cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 0, 0), 1, cv2.LINE_AA)
        # Số áo / ID màu đen in đậm
        id_tx = ax - id_w // 2
        id_ty = by1 + (badge_h + id_h) // 2 - 2
        cv2.putText(annotated, id_str, (id_tx, id_ty), font, font_scale_id, (0, 0, 0), thickness_id, cv2.LINE_AA)

        # C. Dòng 1: Vận tốc (VD: 6.73 km/h) - Chữ màu đen
        speed_str = f"{speed:.2f} km/h"
        font_scale_stats = 0.42
        thickness_stats = 1
        (sw, sh), _ = cv2.getTextSize(speed_str, font, font_scale_stats, thickness_stats)
        cv2.putText(annotated, speed_str, (ax - sw // 2, by2 + 15), font, font_scale_stats, (0, 0, 0), thickness_stats, cv2.LINE_AA)

        # D. Dòng 2: Quãng đường (VD: 19.44 m) - Chữ màu đen
        dist_str = f"{dist:.2f} m"
        (dw, dh), _ = cv2.getTextSize(dist_str, font, font_scale_stats, thickness_stats)
        cv2.putText(annotated, dist_str, (ax - dw // 2, by2 + 29), font, font_scale_stats, (0, 0, 0), thickness_stats, cv2.LINE_AA)

    return annotated


def draw_physical_dashboard(
    team_0_stats: List[Any],
    team_1_stats: List[Any],
    team_0_color: sv.Color = sv.Color.from_hex('#FF1493'),
    team_1_color: sv.Color = sv.Color.from_hex('#00BFFF'),
    team_0_name: str = "Team 1",
    team_1_name: str = "Team 2",
    width: int = 1200,
    height: int = 700,
) -> np.ndarray:
    """
    Renders a comprehensive TV-broadcast style post-match athletic performance dashboard.

    Args:
        team_0_stats: List of PlayerPhysicalStats for Team 1.
        team_1_stats: List of PlayerPhysicalStats for Team 2.
        team_0_color: Primary color for Team 1.
        team_1_color: Primary color for Team 2.
        team_0_name: Name of Team 1.
        team_1_name: Name of Team 2.
        width: Output canvas width in pixels.
        height: Output canvas height in pixels.

    Returns:
        np.ndarray: Dashboard image.
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (22, 26, 32)  # Dark slate background

    c0 = team_0_color.as_bgr()
    c1 = team_1_color.as_bgr()

    # 1. Header Banner
    header_h = 60
    cv2.rectangle(canvas, (0, 0), (width, header_h), (14, 16, 20), -1)
    cv2.line(canvas, (0, header_h), (width, header_h), (50, 55, 65), 1)

    cv2.putText(canvas, "MATCH PHYSICAL & WORKRATE DASHBOARD", (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # 2. Team Overview Cards (Top Half)
    col_w = (width - 60) // 2
    y_card = header_h + 20
    card_h = 240

    def draw_team_card(x: int, name: str, color: Tuple[int, int, int], stats: List[Any]):
        # Card Background
        cv2.rectangle(canvas, (x, y_card), (x + col_w, y_card + card_h), (30, 35, 43), -1)
        cv2.rectangle(canvas, (x, y_card), (x + col_w, y_card + card_h), (60, 65, 75), 1)
        # Accent top bar
        cv2.rectangle(canvas, (x, y_card), (x + col_w, y_card + 4), color, -1)

        # Team Title
        cv2.putText(canvas, name.upper(), (x + 20, y_card + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        # Team aggregates
        total_dist_km = sum([s.total_distance_km for s in stats])
        total_sprints = sum([s.sprint_count for s in stats])
        peak_speed = max([s.max_speed_kmh for s in stats] + [0.0])

        metrics = [
            ("Total Distance Covered", f"{total_dist_km:.2f} km"),
            ("Peak Sprint Speed", f"{peak_speed:.1f} km/h"),
            ("Total Sprint Bursts", f"{total_sprints}"),
            ("Active Tracked Players", f"{len(stats)} players"),
        ]

        for i, (label, val) in enumerate(metrics):
            my = y_card + 75 + i * 38
            cv2.putText(canvas, label, (x + 20, my),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 180, 190), 1, cv2.LINE_AA)
            cv2.putText(canvas, val, (x + col_w - 140, my),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)

    draw_team_card(20, team_0_name, c0, team_0_stats)
    draw_team_card(40 + col_w, team_1_name, c1, team_1_stats)

    # 3. Leaderboards (Bottom Half): Top Speeds & Top Distances
    y_table = y_card + card_h + 20
    table_h = height - y_table - 20

    def draw_top_players_table(x: int, title: str, players_sorted: List[Any], metric_fn, unit: str, color: Tuple[int, int, int]):
        cv2.rectangle(canvas, (x, y_table), (x + col_w, y_table + table_h), (30, 35, 43), -1)
        cv2.rectangle(canvas, (x, y_table), (x + col_w, y_table + table_h), (60, 65, 75), 1)

        cv2.putText(canvas, title, (x + 20, y_table + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        for rank, p in enumerate(players_sorted[:5]):
            py = y_table + 68 + rank * 44
            rank_str = f"#{rank + 1}"
            cv2.putText(canvas, rank_str, (x + 20, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 150, 160), 1, cv2.LINE_AA)

            player_str = f"Player #{p.player_id}"
            cv2.putText(canvas, player_str, (x + 65, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 235, 240), 1, cv2.LINE_AA)

            val_str = f"{metric_fn(p):.1f} {unit}"
            cv2.putText(canvas, val_str, (x + col_w - 130, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    all_players = team_0_stats + team_1_stats
    top_speed_players = sorted(all_players, key=lambda s: s.max_speed_kmh, reverse=True)
    top_dist_players = sorted(all_players, key=lambda s: s.total_distance_m, reverse=True)

    draw_top_players_table(20, "TOP SPEED LEADERBOARD (KM/H)", top_speed_players, lambda s: s.max_speed_kmh, "km/h", (0, 215, 255))
    draw_top_players_table(40 + col_w, "TOP DISTANCE COVERED (METERS)", top_dist_players, lambda s: s.total_distance_m, "m", (100, 255, 120))

    return canvas


def draw_offside_line_on_frame(
    frame: np.ndarray,
    decision: Any,
    view_transformer: Any,
    config: SoccerPitchConfiguration,
) -> np.ndarray:
    """
    Renders 3D perspective transversal offside laser line on the video frame with VAR HUD graphic.

    Args:
        frame: The video frame to annotate.
        decision: OffsideDecision record.
        view_transformer: ViewTransformer instance with inverse_transform_points.
        config: SoccerPitchConfiguration.

    Returns:
        np.ndarray: Annotated frame with offside laser line and VAR banner.
    """
    if decision is None or view_transformer is None:
        return frame

    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Sample multiple points across the transversal line (Y from 0 to 7000 cm)
    line_x_cm = decision.offside_line_x_m * 100.0
    y_samples = np.linspace(0.0, float(config.width), num=25, dtype=np.float32)
    pitch_points = np.stack([np.full_like(y_samples, line_x_cm), y_samples], axis=-1)

    pixel_pts = view_transformer.inverse_transform_points(pitch_points)

    is_offside = decision.is_offside
    laser_color = (0, 0, 255) if is_offside else (0, 230, 100)  # Red for Offside, Green for Onside
    glow_color = (0, 0, 180) if is_offside else (0, 150, 50)

    # Filter on-screen valid projected points
    valid_mask = (pixel_pts[:, 0] >= -100) & (pixel_pts[:, 0] <= w + 100) & \
                 (pixel_pts[:, 1] >= -100) & (pixel_pts[:, 1] <= h + 100)
    valid_pts = pixel_pts[valid_mask].astype(np.int32)

    if len(valid_pts) >= 2:
        # Draw perspective laser line across the pitch
        cv2.polylines(annotated, [valid_pts], isClosed=False, color=glow_color, thickness=6, lineType=cv2.LINE_AA)
        cv2.polylines(annotated, [valid_pts], isClosed=False, color=laser_color, thickness=2, lineType=cv2.LINE_AA)

    # Broadcast VAR Check Graphic HUD (Top-Right)
    hud_w = 340
    hud_h = 74
    hx1 = w - hud_w - 20
    hy1 = 20
    hx2 = hx1 + hud_w
    hy2 = hy1 + hud_h

    overlay = annotated.copy()
    cv2.rectangle(overlay, (hx1, hy1), (hx2, hy2), (15, 18, 24), -1)
    cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)
    cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2), (70, 75, 85), 1)

    # Accent left status bar
    cv2.rectangle(annotated, (hx1, hy1), (hx1 + 6, hy2), laser_color, -1)

    # Title with VAR icon
    cv2.putText(annotated, "VAR CHECK - OFFSIDE TECHNOLOGY", (hx1 + 16, hy1 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 210, 220), 1, cv2.LINE_AA)

    # Verdict text
    if is_offside:
        verdict = f"OFFSIDE: +{decision.margin_cm:.1f} cm"
    else:
        verdict = f"ONSIDE: {decision.margin_cm:.1f} cm"

    cv2.putText(annotated, verdict, (hx1 + 16, hy1 + 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, laser_color, 2, cv2.LINE_AA)

    # Subtext: Passer & Receiver
    subtext = f"Passer #{decision.passer_id} -> Receiver #{decision.receiver_id}"
    cv2.putText(annotated, subtext, (hx1 + 16, hy1 + 64),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 170, 180), 1, cv2.LINE_AA)

    return annotated


def draw_offside_pitch_snapshot(
    config: SoccerPitchConfiguration,
    decision: Any,
    player_xy_pitch: np.ndarray,
    player_ids: np.ndarray,
    player_teams: np.ndarray,
    team_colors: Tuple[sv.Color, sv.Color] = (sv.Color.from_hex('#FF1493'), sv.Color.from_hex('#00BFFF')),
    padding: int = 50,
    scale: float = 0.1,
) -> np.ndarray:
    """
    Renders a 2D pitch metric snapshot showing player positions and offside line at kick-off moment.

    Args:
        config: SoccerPitchConfiguration.
        decision: OffsideDecision instance.
        player_xy_pitch: Shape (N, 2) player positions in meters.
        player_ids: Shape (N,) player IDs.
        player_teams: Shape (N,) team IDs.
        team_colors: Tuple of team colors.
        padding: Pitch padding in pixels.
        scale: Pitch scale factor.

    Returns:
        np.ndarray: Annotated 2D snapshot image.
    """
    pitch = draw_pitch(
        config=config,
        background_color=sv.Color(24, 28, 34),
        line_color=sv.Color(120, 130, 140),
        padding=padding,
        scale=scale
    )

    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)

    # 1. Draw Offside Line across 2D pitch
    offside_x_px = int(decision.offside_line_x_m * 100.0 * scale) + padding
    line_color = (0, 0, 255) if decision.is_offside else (0, 230, 100)

    cv2.line(
        pitch,
        (offside_x_px, padding),
        (offside_x_px, padding + scaled_width),
        line_color, 2, cv2.LINE_AA
    )

    # 2. Draw Players
    for pos, pid, tid in zip(player_xy_pitch, player_ids, player_teams):
        pid = int(pid)
        tid = int(tid)
        px = int(pos[0] * 100.0 * scale) + padding
        py = int(pos[1] * 100.0 * scale) + padding

        p_color = team_colors[tid % 2].as_bgr()

        # Highlight receiver or second-last defender
        if pid == decision.receiver_id:
            cv2.circle(pitch, (px, py), 16, line_color, 2, cv2.LINE_AA)
            cv2.circle(pitch, (px, py), 12, p_color, -1, cv2.LINE_AA)
        elif pid == decision.second_last_def_id:
            cv2.circle(pitch, (px, py), 16, (0, 215, 255), 2, cv2.LINE_AA)
            cv2.circle(pitch, (px, py), 12, p_color, -1, cv2.LINE_AA)
        else:
            cv2.circle(pitch, (px, py), 9, p_color, -1, cv2.LINE_AA)

        cv2.putText(pitch, str(pid), (px - 5, py + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    # 3. Banner Title
    h, w = pitch.shape[:2]
    banner_h = 44
    cv2.rectangle(pitch, (0, 0), (w, banner_h), (14, 16, 20), -1)
    cv2.line(pitch, (0, banner_h), (w, banner_h), (60, 65, 75), 1)

    verdict = "OFFSIDE" if decision.is_offside else "ONSIDE"
    title_str = f"SAOT VAR SNAPSHOT - PASS #{decision.pass_id} [{verdict}]"
    cv2.putText(pitch, title_str, (25, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    margin_str = f"Margin: {decision.margin_cm:+.1f} cm  |  Def Line: {decision.offside_line_x_m:.1f}m"
    cv2.putText(pitch, margin_str, (w - 360, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, line_color, 1, cv2.LINE_AA)

    return pitch


def draw_match_analysis_screen(
    img_hm_0: np.ndarray,
    img_hm_1: np.ndarray,
    img_net_0: np.ndarray,
    img_net_1: np.ndarray,
    team_0_stats: Dict[str, Any],
    team_1_stats: Dict[str, Any],
    team_0_color: sv.Color = sv.Color.from_hex('#FF1493'),
    team_1_color: sv.Color = sv.Color.from_hex('#00BFFF'),
    team_0_name: str = "Team 1",
    team_1_name: str = "Team 2",
    width: int = 1280,
    height: int = 720,
) -> np.ndarray:
    """
    Renders an all-in-one broadcast-quality Match Analysis summary screen containing:
    1. Header Bar: Match Analysis Title
    2. Team Comparative Performance Cards (Completed Passes, Accuracy, Possession, Peak Speed, Total Distance)
    3. 2x2 Tactical Grid:
       - Top-Left: Team 1 Positional Heatmap
       - Top-Right: Team 2 Positional Heatmap
       - Bottom-Left: Team 1 Passing Network
       - Bottom-Right: Team 2 Passing Network

    Args:
        img_hm_0: Team 1 Heatmap image.
        img_hm_1: Team 2 Heatmap image.
        img_net_0: Team 1 Passing Network image.
        img_net_1: Team 2 Passing Network image.
        team_0_stats: Dict with keys ['completed_passes', 'pass_accuracy', 'possession_pct', 'total_distance_m', 'peak_speed_kmh'].
        team_1_stats: Dict with keys ['completed_passes', 'pass_accuracy', 'possession_pct', 'total_distance_m', 'peak_speed_kmh'].
        team_0_color: Color for Team 1.
        team_1_color: Color for Team 2.
        team_0_name: Name for Team 1.
        team_1_name: Name for Team 2.
        width: Canvas width in pixels.
        height: Canvas height in pixels.

    Returns:
        np.ndarray: Match analysis canvas image.
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (18, 22, 28)  # Deep slate dark background

    c0 = team_0_color.as_bgr()
    c1 = team_1_color.as_bgr()

    # 1. Header Bar
    header_h = max(42, int(height * 0.065))
    cv2.rectangle(canvas, (0, 0), (width, header_h), (12, 14, 18), -1)
    cv2.line(canvas, (0, header_h), (width, header_h), (50, 55, 65), 1)

    cv2.putText(canvas, "MATCH ANALYSIS - FULL TIME SUMMARY", (24, int(header_h * 0.68)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "AI COMPUTER VISION ANALYTICS", (width - 280, int(header_h * 0.68)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 175, 190), 1, cv2.LINE_AA)

    # 2. Team Comparative Performance Cards (Top Section)
    stats_y = header_h + 10
    stats_h = max(110, int(height * 0.19))
    col_w = (width - 50) // 2

    def draw_team_summary_card(x: int, name: str, color: Tuple[int, int, int], stats: Dict[str, Any]):
        cv2.rectangle(canvas, (x, stats_y), (x + col_w, stats_y + stats_h), (26, 30, 38), -1)
        cv2.rectangle(canvas, (x, stats_y), (x + col_w, stats_y + stats_h), (60, 65, 75), 1)
        # Accent top bar
        cv2.rectangle(canvas, (x, stats_y), (x + col_w, stats_y + 3), color, -1)

        # Team Title
        cv2.putText(canvas, name.upper(), (x + 16, stats_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)

        completed_p = stats.get('completed_passes', 0)
        acc_p = stats.get('pass_accuracy', 0.0)
        poss_p = stats.get('possession_pct', 0.0)
        tot_dist = stats.get('total_distance_m', 0.0)
        peak_spd = stats.get('peak_speed_kmh', 0.0)

        # Left column metrics
        m1 = f"Completed Passes: {completed_p}"
        m2 = f"Possession Rate: {poss_p:.1f}%"
        cv2.putText(canvas, m1, (x + 16, stats_y + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (230, 235, 240), 1, cv2.LINE_AA)
        cv2.putText(canvas, m2, (x + 16, stats_y + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (230, 235, 240), 1, cv2.LINE_AA)

        # Right column metrics
        m3 = f"Total Distance: {tot_dist:.1f} m"
        m4 = f"Peak Sprint Speed: {peak_spd:.1f} km/h"
        rx = x + col_w // 2 + 10
        cv2.putText(canvas, m3, (rx, stats_y + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (230, 235, 240), 1, cv2.LINE_AA)
        cv2.putText(canvas, m4, (rx, stats_y + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 215, 255), 1, cv2.LINE_AA)

    draw_team_summary_card(18, team_0_name, c0, team_0_stats)
    draw_team_summary_card(32 + col_w, team_1_name, c1, team_1_stats)

    # 3. 2x2 Tactical Grid of 4 Pitches
    grid_y = stats_y + stats_h + 10
    grid_h = height - grid_y - 12
    grid_w = width - 36
    cell_w = (grid_w - 14) // 2
    cell_h = (grid_h - 10) // 2

    def place_subpanel(img: np.ndarray, row: int, col: int, label: str, tag_color: Tuple[int, int, int]):
        px = 18 + col * (cell_w + 14)
        py = grid_y + row * (cell_h + 10)

        # Resize image to cell
        if img is not None and img.size > 0:
            resized = cv2.resize(img, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
            canvas[py:py + cell_h, px:px + cell_w] = resized

        # Cell border
        cv2.rectangle(canvas, (px, py), (px + cell_w, py + cell_h), (55, 60, 72), 1)

        # Title Pill Tag
        tw = min(220, cell_w - 10)
        cv2.rectangle(canvas, (px + 6, py + 6), (px + 6 + tw, py + 26), (15, 18, 24), -1)
        cv2.rectangle(canvas, (px + 6, py + 6), (px + 6 + tw, py + 26), tag_color, 1)
        cv2.putText(canvas, label, (px + 12, py + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    # Place 4 panels
    place_subpanel(img_hm_0, 0, 0, f"{team_0_name} - Positional Heatmap", c0)
    place_subpanel(img_hm_1, 0, 1, f"{team_1_name} - Positional Heatmap", c1)
    place_subpanel(img_net_0, 1, 0, f"{team_0_name} - Passing Network", c0)
    place_subpanel(img_net_1, 1, 1, f"{team_1_name} - Passing Network", c1)

    return canvas


