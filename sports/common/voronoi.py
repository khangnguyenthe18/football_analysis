"""
Voronoi Tactical Minimap — Real-time 2D pitch overlay with Voronoi pitch control.

Renders a small bird's-eye-view minimap showing:
- 2D soccer pitch with field markings
- Player dots (team-colored circles with jersey numbers)
- Semi-transparent Voronoi regions showing spatial control per team
- Ball position marker

The minimap is composited onto the bottom-right corner of each video frame.
"""

from typing import Optional, Dict, Tuple, List
import cv2
import numpy as np

try:
    from scipy.spatial import Voronoi
except ImportError:
    Voronoi = None  # Graceful fallback if scipy not available

from sports.configs.football import SoccerPitchConfiguration


def _clip_polygon_to_rect(
    polygon: np.ndarray,
    x_min: float, y_min: float, x_max: float, y_max: float
) -> np.ndarray:
    """
    Clip a convex/concave polygon to a rectangle using Sutherland-Hodgman algorithm.

    Args:
        polygon: (M, 2) array of polygon vertices.
        x_min, y_min, x_max, y_max: Clipping rectangle bounds.

    Returns:
        Clipped polygon as (K, 2) array. May be empty.
    """
    def clip_edge(pts: List, edge_start, edge_end):
        """Clip polygon points against a single edge."""
        if len(pts) == 0:
            return []
        result = []
        for i in range(len(pts)):
            current = pts[i]
            previous = pts[i - 1]
            curr_inside = _is_inside(current, edge_start, edge_end)
            prev_inside = _is_inside(previous, edge_start, edge_end)

            if curr_inside:
                if not prev_inside:
                    result.append(_intersect(previous, current, edge_start, edge_end))
                result.append(current)
            elif prev_inside:
                result.append(_intersect(previous, current, edge_start, edge_end))
        return result

    pts = polygon.tolist()

    # Clip against each edge of the rectangle (counter-clockwise winding)
    edges = [
        ((x_min, y_max), (x_min, y_min)),  # left   (keep points with x >= x_min)
        ((x_max, y_min), (x_max, y_max)),  # right  (keep points with x <= x_max)
        ((x_min, y_min), (x_max, y_min)),  # top    (keep points with y >= y_min)
        ((x_max, y_max), (x_min, y_max)),  # bottom (keep points with y <= y_max)
    ]
    for e_start, e_end in edges:
        pts = clip_edge(pts, e_start, e_end)
        if len(pts) == 0:
            return np.array([], dtype=np.float32).reshape(0, 2)

    return np.array(pts, dtype=np.float32)


def _is_inside(point, edge_start, edge_end) -> bool:
    """Check if point is on the inside (left side) of the directed edge."""
    return ((edge_end[0] - edge_start[0]) * (point[1] - edge_start[1]) -
            (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0])) >= 0


def _intersect(p1, p2, edge_start, edge_end):
    """Compute intersection of line segment p1→p2 with edge line."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = edge_start
    x4, y4 = edge_end

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return p2  # Parallel lines, return current point

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return [x1 + t * (x2 - x1), y1 + t * (y2 - y1)]


class VoronoiMinimap:
    """
    Renders a tactical minimap with Voronoi pitch control overlay.

    The minimap shows a bird's-eye view of the pitch with:
    - Voronoi regions colored by team (semi-transparent)
    - Player dots with jersey numbers
    - Ball marker

    Args:
        config: Soccer pitch configuration (dimensions in cm).
        minimap_width: Width of the minimap in pixels.
        padding: Internal padding around pitch edges in pixels.
        alpha: Voronoi region transparency (0=invisible, 1=opaque).
        player_radius: Radius of player dots in pixels.
        ball_radius: Radius of ball marker in pixels.
    """

    def __init__(
        self,
        config: SoccerPitchConfiguration,
        minimap_width: int = 360,
        padding: int = 14,
        alpha: float = 0.75,
        player_radius: int = 8,
        ball_radius: int = 4,
        ema_alpha: float = 0.12,
        ghost_ttl: int = 8,
        frame_blend: float = 0.55,
        jump_threshold_cm: float = 1500.0,
    ):
        self.config = config
        self.padding = padding
        self.alpha = alpha
        self.player_radius = player_radius
        self.ball_radius = ball_radius

        # EMA smoothing for position stability
        self._ema_alpha = ema_alpha   # weight for new data (lower = smoother)
        self._smoothed: Dict[int, np.ndarray] = {}  # tracker_id → smoothed (x,y) cm
        self._smoothed_team: Dict[int, int] = {}     # tracker_id → team_id

        # Ghost persistence: keep players visible for ghost_ttl frames after disappearing
        self._ghost_ttl = ghost_ttl
        self._ghost_age: Dict[int, int] = {}  # tracker_id → frames since last seen

        # Temporal frame blending: blend previous minimap with current
        self._frame_blend = frame_blend  # weight for previous frame (0 = no blend)
        self._prev_minimap: Optional[np.ndarray] = None

        # Ball position smoothing
        self._smoothed_ball: Optional[np.ndarray] = None

        # Outlier rejection: max allowed position jump per frame (cm)
        self._jump_threshold = jump_threshold_cm

        # Pitch bounds for clamping (cm)
        self._pitch_x_max_cm = float(config.length)
        self._pitch_y_max_cm = float(config.width)

        # Compute scale: map pitch cm → minimap pixels
        usable_w = minimap_width - 2 * padding
        self.scale = usable_w / config.length
        self.minimap_w = minimap_width
        self.minimap_h = int(config.width * self.scale) + 2 * padding

        # Pitch boundary in minimap pixel coordinates
        self.pitch_x_min = padding
        self.pitch_y_min = padding
        self.pitch_x_max = padding + int(config.length * self.scale)
        self.pitch_y_max = padding + int(config.width * self.scale)

        # Pre-render the base pitch image (cached)
        self._base_pitch = self._draw_pitch_base()

    def _draw_pitch_base(self) -> np.ndarray:
        """Draw the static pitch lines on a dark background."""
        bg_color = (30, 30, 30)  # BGR dark/near-black
        line_color = (140, 140, 140)  # dim gray lines
        line_thick = 1

        img = np.ones((self.minimap_h, self.minimap_w, 3), dtype=np.uint8)
        img[:] = bg_color

        s = self.scale
        p = self.padding
        cfg = self.config

        # Draw pitch edges from config
        for start_idx, end_idx in cfg.edges:
            v1 = cfg.vertices[start_idx - 1]
            v2 = cfg.vertices[end_idx - 1]
            pt1 = (int(v1[0] * s) + p, int(v1[1] * s) + p)
            pt2 = (int(v2[0] * s) + p, int(v2[1] * s) + p)
            cv2.line(img, pt1, pt2, line_color, line_thick)

        # Centre circle
        center = (int(cfg.length / 2 * s) + p, int(cfg.width / 2 * s) + p)
        radius = int(cfg.centre_circle_radius * s)
        cv2.circle(img, center, radius, line_color, line_thick)

        # Centre spot
        cv2.circle(img, center, 2, line_color, -1)

        # Penalty spots
        ps_dist = cfg.penalty_spot_distance
        ps1 = (int(ps_dist * s) + p, int(cfg.width / 2 * s) + p)
        ps2 = (int((cfg.length - ps_dist) * s) + p, int(cfg.width / 2 * s) + p)
        cv2.circle(img, ps1, 2, line_color, -1)
        cv2.circle(img, ps2, 2, line_color, -1)

        return img

    def _pitch_cm_to_minimap_px(self, xy_cm: np.ndarray) -> np.ndarray:
        """Convert pitch coordinates (cm) to minimap pixel coordinates."""
        return (xy_cm * self.scale + self.padding).astype(np.float32)

    def _smooth_positions(
        self, xy_cm: np.ndarray, jersey_ids: np.ndarray,
        team_ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply EMA smoothing + ghost persistence to player positions.

        Returns smoothed positions including ghost players (recently
        disappeared players kept for ghost_ttl frames).

        Returns:
            (smoothed_xy, all_jersey_ids, all_team_ids) — may be larger than
            input if ghost players are added.
        """
        smoothed = np.copy(xy_cm)
        a = self._ema_alpha

        current_ids = set()
        for i in range(len(jersey_ids)):
            tid = int(jersey_ids[i])
            current_ids.add(tid)

            # Clamp position to pitch boundaries
            smoothed[i][0] = np.clip(smoothed[i][0], 0, self._pitch_x_max_cm)
            smoothed[i][1] = np.clip(smoothed[i][1], 0, self._pitch_y_max_cm)

            if tid in self._smoothed:
                # Outlier rejection: if position jumps too far, keep old position
                delta = np.linalg.norm(smoothed[i] - self._smoothed[tid])
                if delta > self._jump_threshold:
                    smoothed[i] = self._smoothed[tid]  # reject, keep old
                else:
                    smoothed[i] = a * smoothed[i] + (1.0 - a) * self._smoothed[tid]

            self._smoothed[tid] = smoothed[i].copy()
            self._smoothed_team[tid] = int(team_ids[i])
            self._ghost_age[tid] = 0  # reset age — player is visible

        # Collect ghost players (recently lost)
        ghost_xy = []
        ghost_jersey = []
        ghost_team = []
        expired = []
        for tid, pos in self._smoothed.items():
            if tid not in current_ids:
                age = self._ghost_age.get(tid, 0) + 1
                self._ghost_age[tid] = age
                if age <= self._ghost_ttl:
                    ghost_xy.append(pos)
                    ghost_jersey.append(tid)
                    ghost_team.append(self._smoothed_team.get(tid, 0))
                else:
                    expired.append(tid)

        # Prune fully expired ghosts
        for tid in expired:
            self._smoothed.pop(tid, None)
            self._smoothed_team.pop(tid, None)
            self._ghost_age.pop(tid, None)

        # Merge current + ghosts
        if ghost_xy:
            all_xy = np.vstack([smoothed, np.array(ghost_xy)])
            all_jersey = np.concatenate([jersey_ids, np.array(ghost_jersey)])
            all_team = np.concatenate([team_ids, np.array(ghost_team)])
        else:
            all_xy = smoothed
            all_jersey = jersey_ids
            all_team = team_ids

        return all_xy, all_jersey, all_team

    def _smooth_ball(self, ball_cm: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Apply EMA smoothing to ball position."""
        if ball_cm is None:
            return self._smoothed_ball  # use last known position
        if self._smoothed_ball is not None:
            smoothed = 0.3 * ball_cm + 0.7 * self._smoothed_ball
        else:
            smoothed = ball_cm.copy()
        self._smoothed_ball = smoothed
        return smoothed

    @staticmethod
    def _brighten_color(
        color: Tuple[int, int, int], factor: float = 1.4
    ) -> Tuple[int, int, int]:
        """Make a BGR color brighter for Voronoi region fill."""
        return tuple(min(255, int(c * factor)) for c in color)

    def render(
        self,
        player_xy_pitch_cm: np.ndarray,
        player_team_ids: np.ndarray,
        player_jersey_ids: np.ndarray,
        team_colors: Dict[int, Tuple[int, int, int]],
        ball_xy_pitch_cm: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Render the tactical minimap for the current frame.

        Args:
            player_xy_pitch_cm: (N, 2) player positions in pitch cm.
            player_team_ids: (N,) team IDs (0 or 1).
            player_jersey_ids: (N,) tracker/jersey IDs.
            team_colors: {0: BGR_tuple, 1: BGR_tuple}.
            ball_xy_pitch_cm: (2,) ball position in pitch cm, or None.

        Returns:
            Minimap image as np.ndarray (H x W x 3).
        """
        # Start from cached pitch base
        minimap = self._base_pitch.copy()

        # Apply EMA smoothing + ghost persistence for stability
        smoothed_cm, all_jersey_ids, all_team_ids = self._smooth_positions(
            player_xy_pitch_cm, player_jersey_ids, player_team_ids
        )

        # Smooth ball position
        smooth_ball = self._smooth_ball(ball_xy_pitch_cm)

        if len(smoothed_cm) < 2 or Voronoi is None:
            # Not enough players for Voronoi or scipy not available
            self._draw_players(minimap, smoothed_cm, all_team_ids,
                               all_jersey_ids, team_colors)
            if smooth_ball is not None:
                self._draw_ball(minimap, smooth_ball)
            return minimap

        # Convert smoothed positions to minimap pixel coords
        player_px = self._pitch_cm_to_minimap_px(smoothed_cm)

        # --- Voronoi computation ---
        # Add dummy far-away points to ensure all regions are bounded
        far = 50000.0
        dummy_points = np.array([
            [-far, -far], [-far, far], [far, -far], [far, far]
        ], dtype=np.float32)
        all_points = np.vstack([player_px, dummy_points])

        try:
            vor = Voronoi(all_points)
        except Exception:
            # Fallback: skip Voronoi if computation fails
            self._draw_players(minimap, smoothed_cm, all_team_ids,
                               all_jersey_ids, team_colors)
            if smooth_ball is not None:
                self._draw_ball(minimap, smooth_ball)
            return minimap

        # --- Draw Voronoi regions ---
        # Paint team-colored regions directly onto the pitch area
        n_players = len(player_px)

        for i in range(n_players):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]

            if -1 in region or len(region) == 0:
                continue

            polygon = np.array([vor.vertices[v] for v in region], dtype=np.float32)

            # Clip to pitch boundary
            clipped = _clip_polygon_to_rect(
                polygon,
                self.pitch_x_min, self.pitch_y_min,
                self.pitch_x_max, self.pitch_y_max
            )

            if len(clipped) < 3:
                continue

            team_id = int(all_team_ids[i])
            base_color = team_colors.get(team_id, (128, 128, 128))

            # Create a muted but visible version for the region fill
            # Blend team color with dark background for a rich look
            fill_color = tuple(
                int(c * 0.55 + 25) for c in base_color
            )

            pts = clipped.astype(np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(minimap, [pts], fill_color)

        # --- Draw Voronoi edges ---
        edge_color = (15, 15, 15)  # dark/black edges
        for ridge_idx, (p1_idx, p2_idx) in enumerate(vor.ridge_points):
            # Only draw edges between real players (not dummy points)
            if p1_idx >= n_players or p2_idx >= n_players:
                continue

            ridge_vertices = vor.ridge_vertices[ridge_idx]
            if -1 in ridge_vertices:
                continue

            v1 = vor.vertices[ridge_vertices[0]].astype(int)
            v2 = vor.vertices[ridge_vertices[1]].astype(int)

            # Clip to minimap bounds
            if (0 <= v1[0] < self.minimap_w and 0 <= v1[1] < self.minimap_h and
                    0 <= v2[0] < self.minimap_w and 0 <= v2[1] < self.minimap_h):
                cv2.line(minimap, tuple(v1), tuple(v2), edge_color, 2, cv2.LINE_AA)

        # --- Re-draw pitch lines on top of Voronoi (so lines stay crisp) ---
        pitch_lines = self._base_pitch.copy()
        # Create mask of pitch lines (gray lines on dark background)
        line_mask = cv2.inRange(pitch_lines, (130, 130, 130), (255, 255, 255))
        minimap[line_mask > 0] = (200, 200, 200)  # draw lines slightly brighter on top of voronoi

        # --- Draw players and ball ---
        self._draw_players(minimap, smoothed_cm, all_team_ids,
                           all_jersey_ids, team_colors)
        if smooth_ball is not None:
            self._draw_ball(minimap, smooth_ball)

        # --- Temporal frame blending for smooth transitions ---
        if self._prev_minimap is not None and self._frame_blend > 0:
            fb = self._frame_blend
            minimap = cv2.addWeighted(
                self._prev_minimap, fb, minimap, 1.0 - fb, 0
            )
        self._prev_minimap = minimap.copy()

        return minimap

    def _draw_players(
        self,
        img: np.ndarray,
        xy_cm: np.ndarray,
        team_ids: np.ndarray,
        jersey_ids: np.ndarray,
        team_colors: Dict[int, Tuple[int, int, int]],
    ):
        """Draw player circles with jersey numbers."""
        r = self.player_radius

        for i in range(len(xy_cm)):
            px = int(xy_cm[i][0] * self.scale + self.padding)
            py = int(xy_cm[i][1] * self.scale + self.padding)

            # Bounds check
            if not (0 <= px < self.minimap_w and 0 <= py < self.minimap_h):
                continue

            team_id = int(team_ids[i])
            color = team_colors.get(team_id, (128, 128, 128))

            # Filled circle
            cv2.circle(img, (px, py), r, color, -1, cv2.LINE_AA)
            # Black border
            cv2.circle(img, (px, py), r, (0, 0, 0), 1, cv2.LINE_AA)

            # Jersey number (white, centered)
            jersey = str(int(jersey_ids[i]))
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.32 if len(jersey) <= 2 else 0.26
            thickness = 1
            (tw, th), _ = cv2.getTextSize(jersey, font, font_scale, thickness)
            tx = px - tw // 2
            ty = py + th // 2
            cv2.putText(img, jersey, (tx, ty), font, font_scale,
                        (255, 255, 255), thickness, cv2.LINE_AA)

    def _draw_ball(self, img: np.ndarray, ball_cm: np.ndarray):
        """Draw ball marker on the minimap."""
        bx = int(ball_cm[0] * self.scale + self.padding)
        by = int(ball_cm[1] * self.scale + self.padding)

        if 0 <= bx < self.minimap_w and 0 <= by < self.minimap_h:
            cv2.circle(img, (bx, by), self.ball_radius, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (bx, by), self.ball_radius, (0, 0, 0), 1, cv2.LINE_AA)

    @staticmethod
    def composite_on_frame(
        frame: np.ndarray,
        minimap: np.ndarray,
        margin: int = 16,
    ) -> np.ndarray:
        """
        Overlay the minimap onto the bottom-right corner of the video frame.

        Args:
            frame: Video frame (H x W x 3).
            minimap: Minimap image (h x w x 3).
            margin: Pixel margin from frame edge.

        Returns:
            Frame with minimap composited.
        """
        fh, fw = frame.shape[:2]
        mh, mw = minimap.shape[:2]

        # Position: bottom-right corner
        x1 = fw - mw - margin
        y1 = fh - mh - margin
        x2 = x1 + mw
        y2 = y1 + mh

        # Bounds check
        if x1 < 0 or y1 < 0:
            return frame

        result = frame.copy()

        # Draw dark semi-transparent background (border effect)
        border = 3
        bx1 = max(0, x1 - border)
        by1 = max(0, y1 - border)
        bx2 = min(fw, x2 + border)
        by2 = min(fh, y2 + border)

        # Dark border
        cv2.rectangle(result, (bx1, by1), (bx2, by2), (20, 20, 20), -1)

        # Place minimap
        result[y1:y2, x1:x2] = minimap

        # Thin outer border
        cv2.rectangle(result, (x1 - 1, y1 - 1), (x2, y2), (80, 85, 95), 1)

        return result
