"""src/pinky_wall_parking_maneuver/pinky_wall_parking_maneuver/obstacles.py - 라이다 장애물 (핑키 퍼크 -> OBB) 과 충돌 검사

역할: 라이다 점에서 다른 핑키를 찾아 차체 크기의 장애물로 바꾸고, 내 로봇 자세가 장애물과 겹치는지 판정한다. ROS 미사용.
방법: 라이다 높이 (약 97 mm) 에서 다른 핑키는 라이다 퍼크 (반지름 25 mm) 의 원호로만 보인다 (시뮬 scan_shape_check: 점 2~6 개).
      퍼크 중심 = 차체 중심 (base_footprint x -17 mm). 그래서 점 묶음에 반지름 고정 원을 맞춰 중심을 구하고,
      방향을 알면 그 방향의 상자 (OBB), 모르면 상자가 어느 방향이든 들어가는 원 (반 대각선 80.7 mm + 여유) 으로 둔다.
      충돌: 상자-상자 = SAT (두 상자 축 4 개에 투영해 겹침), 상자-원 = 상자 안 최근접점 거리, 상자-점 = 상자 좌표로 범위 비교.
실행: 라이브러리 (경로 계획, 단위 시험에서 호출).
"""

# --- 표준 라이브러리 ---------------------------------------------------------------
from dataclasses import dataclass, field                                            # 파라미터 묶음
import math                                                                         # 삼각 함수

# --- 외부 라이브러리 ---------------------------------------------------------------
import numpy as np                                                                  # 점 배열


# ==================================================================================
# 파라미터
# ==================================================================================
@dataclass
class ObsParams:
    puck_r:    float = 0.025                                                        # 라이다 퍼크 반지름 (m, 시뮬 측정 25.0 mm)
    gap:       float = 0.02                                                         # 이웃 점 거리 한도 (m, 묶음 나누기)
    max_span:  float = 0.055                                                        # 묶음 폭 한도 (m, 퍼크 지름 + 여유)
    max_resid: float = 0.003                                                        # 원 맞춤 잔차 한도 (m, RMS)
    body_x:    tuple = (-0.076, 0.042)                                              # 핑키 외곽 x (base_footprint 기준, m)
    body_w:    float = 0.055                                                        # 핑키 외곽 반폭 (m)
    margin:    float = 0.01                                                         # 장애물 여유 (m)
    track_n:   int = 15                                                             # 퍼크 중심 누적 개수
    assoc:     float = 0.05                                                         # 같은 퍼크로 볼 거리 (m)


def body_half(p):
    """Return the half length, half width and center x offset of a Pinky body."""
    return (p.body_x[1] - p.body_x[0]) / 2, p.body_w, (p.body_x[0] + p.body_x[1]) / 2


# ==================================================================================
# 상자 (OBB)
# ==================================================================================
@dataclass
class Obb:
    cx: float                                                                       # 중심 x (m)
    cy: float                                                                       # 중심 y (m)
    yaw: float                                                                      # 방향 (rad)
    hl: float                                                                       # 반길이 (m)
    hw: float                                                                       # 반폭 (m)

    def axes(self):
        """Return the unit length and width axes."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)                               # 방향 성분
        return np.array([c, s]), np.array([-s, c])

    def corners(self):
        """Return the 4 corners as a 4x2 array."""
        u, v = self.axes()                                                          # 축
        c = np.array([self.cx, self.cy])                                            # 중심
        return np.array([c + a * self.hl * u + b * self.hw * v
                         for a, b in ((1, 1), (-1, 1), (-1, -1), (1, -1))])

    def local(self, pts):
        """Return points (Nx2) in the box frame."""
        u, v = self.axes()                                                          # 축
        d = np.atleast_2d(pts) - [self.cx, self.cy]                                 # 중심 기준
        return np.column_stack([d @ u, d @ v])


def robot_obb(pose, p, extra=0.0):
    """Return the box of a Pinky whose base_footprint is at pose (x, y, yaw)."""
    hl, hw, off = body_half(p)                                                      # 차체 치수
    x, y, yaw = pose                                                                # 자세
    return Obb(x + off * math.cos(yaw), y + off * math.sin(yaw), yaw, hl + extra, hw + extra)


def pinky_obstacle(center, yaw, p):
    """Return an Obb (yaw known) or a (center, radius) circle (yaw None) around a puck center."""
    hl, hw, _ = body_half(p)                                                        # 차체 치수
    if yaw is None:
        return (np.asarray(center, dtype=float), math.hypot(hl, hw) + p.margin)
    return Obb(center[0], center[1], yaw, hl + p.margin, hw + p.margin)


# ==================================================================================
# 충돌 검사
# ==================================================================================
def obb_overlap(a, b):
    """Return True if two boxes overlap or touch (SAT on the 4 box axes)."""
    ca, cb = a.corners(), b.corners()                                               # 꼭짓점
    for axis in (*a.axes(), *b.axes()):
        pa, pb = ca @ axis, cb @ axis                                               # 축 투영
        if pa.max() < pb.min() or pb.max() < pa.min():
            return False                                                            # 분리 축 있음
    return True


def obb_circle_overlap(a, center, r):
    """Return True if a box and a circle overlap or touch."""
    x, y = a.local(center)[0]                                                       # 상자 좌표 중심
    dx = max(abs(x) - a.hl, 0.0)                                                    # 상자 밖 x 거리
    dy = max(abs(y) - a.hw, 0.0)                                                    # 상자 밖 y 거리
    return math.hypot(dx, dy) <= r


def points_in_obb(a, pts):
    """Return True if any point lies inside the box."""
    if len(pts) == 0:
        return False
    q = a.local(pts)                                                                # 상자 좌표
    return bool(np.any((np.abs(q[:, 0]) <= a.hl) & (np.abs(q[:, 1]) <= a.hw)))


@dataclass
class Obstacles:
    boxes:   list = field(default_factory=list)                                     # Obb 목록
    circles: list = field(default_factory=list)                                     # (중심, 반지름) 목록
    points:  np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))           # 그 밖의 점 (벽, 상자)


def pose_collides(pose, obs, p):
    """Return True if a Pinky at pose (x, y, yaw) hits any obstacle (points get the margin)."""
    me = robot_obb(pose, p)                                                         # 내 차체
    if any(obb_overlap(me, b) for b in obs.boxes) or any(obb_circle_overlap(me, c, r) for c, r in obs.circles):
        return True
    return points_in_obb(robot_obb(pose, p, p.margin), obs.points)


# ==================================================================================
# 퍼크 찾기
# ==================================================================================
def cluster(pts, gap):
    """Split points (Nx2, scan order) where neighbours are farther than gap."""
    pts = np.atleast_2d(np.asarray(pts, dtype=float))                               # 점 배열
    if len(pts) == 0:
        return []
    cut = np.where(np.linalg.norm(np.diff(pts, axis=0), axis=1) > gap)[0] + 1       # 나눌 곳
    groups = np.split(pts, cut)                                                     # 묶음
    if len(groups) > 1 and np.linalg.norm(pts[0] - pts[-1]) <= gap:
        groups = [np.vstack([groups[-1], groups[0]])] + groups[1:-1]                # 한 바퀴 이음
    return groups


def fit_puck(pts, sensor, p):
    """Fit a fixed-radius circle; return (center, rms) or None if it is not a puck."""
    pts = np.asarray(pts, dtype=float)                                              # 묶음 점
    if len(pts) < 2:
        return None
    span = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))                        # 묶음 폭
    if span > p.max_span:
        return None
    m = pts.mean(axis=0)                                                            # 점 평균
    d = m - np.asarray(sensor, dtype=float)                                         # 센서 -> 점
    c = m + p.puck_r * d / np.linalg.norm(d)                                        # 초기값: 센서 반대쪽
    for _ in range(20):
        v = pts - c                                                                 # 중심 -> 점
        dist = np.linalg.norm(v, axis=1)                                            # 거리
        J = -v / dist[:, None]                                                      # 잔차 미분
        step = np.linalg.lstsq(J, -(dist - p.puck_r), rcond=None)[0]                # 가우스-뉴턴
        c = c + step
        if np.linalg.norm(step) < 1e-6:
            break
    rms = float(np.sqrt(np.mean((np.linalg.norm(pts - c, axis=1) - p.puck_r) ** 2)))
    if rms > p.max_resid or np.linalg.norm(c - sensor) < np.linalg.norm(d):
        return None                                                                 # 원 아님 / 센서 쪽 해
    return c, rms


def split_scan(pts, sensor, p):
    """Return (puck centers, other points) from scan points (Nx2, scan order)."""
    centers, rest = [], []                                                          # 결과
    for g in cluster(pts, p.gap):
        fit = fit_puck(g, sensor, p)                                                # 원 맞춤
        if fit is None:
            rest.append(g)
        else:
            centers.append(fit[0])
    return centers, (np.vstack(rest) if rest else np.zeros((0, 2)))


class PuckTrack:
    """Median of puck centers over recent scans (fixed frame, e.g. odom)."""

    def __init__(self, p):
        self.p, self.tracks = p, []                                                 # 파라미터, [중심 목록]

    def update(self, centers):
        """Add the puck centers of one scan."""
        for c in (np.asarray(c, dtype=float) for c in centers):
            dist = [np.linalg.norm(np.median(t, axis=0) - c) for t in self.tracks]  # 트랙까지 거리
            if dist and min(dist) <= self.p.assoc:
                best = self.tracks[int(np.argmin(dist))]                            # 가장 가까운 트랙
                best.append(c)
                del best[:-self.p.track_n]
            else:
                self.tracks.append([c])

    def pucks(self, min_hits=3):
        """Return the median centers of tracks seen at least min_hits times."""
        return [np.median(t, axis=0) for t in self.tracks if len(t) >= min_hits]
