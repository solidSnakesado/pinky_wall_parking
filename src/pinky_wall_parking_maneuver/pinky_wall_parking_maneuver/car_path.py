"""src/pinky_wall_parking_maneuver/pinky_wall_parking_maneuver/car_path.py - 차량식 경로 계획·추종

역할: 일반 차량처럼 (제자리 회전 없이, 최소 회전 반경 R, 전진·후진 전환 허용) 시작 자세에서 목표 자세까지 가는 경로를 만들고 따라간다. ROS 미사용.
방법: 계획 = Reeds-Shepp 최단 경로 (rsplan). 반경은 R x plan_margin (추종 여유), 목표 직전에 runway 만큼 들어가는 방향으로 직진.
      같은 방향 (전진 / 후진) 구간끼리 묶어 기어로 나눈다.
      추종 = 기어마다 pure pursuit. 앞보기 점까지 원호 곡률 k = 2 y / (x^2 + y^2) 를 |k| <= 1 / R 로 자르고 w = v k.
      그래서 늘 |w| <= |v| / R (차량 조건) 이고 v = 0 이면 w = 0. 기어 끝 접선 방향 남은 거리 <= goal_tol 이면 그 기어를 끝낸다.
실행: 라이브러리 (t_parking, 단위 시험에서 호출). rsplan 은 pip 패키지 (pip install rsplan).
"""

# --- 표준 라이브러리 ---------------------------------------------------------------
from dataclasses import dataclass                                                   # 파라미터 묶음
import math                                                                         # 삼각 함수

# --- 외부 라이브러리 ---------------------------------------------------------------
import numpy as np                                                                  # 경로 점 배열

try:
    from rsplan import planner                                                      # Reeds-Shepp 경로
except ImportError as exc:                                                          # rosdep 에 없는 pip 패키지
    raise ImportError("rsplan 이 필요합니다: pip install rsplan") from exc


# ==================================================================================
# 파라미터
# ==================================================================================
@dataclass
class CarParams:
    turn_radius: float = 0.15                                                       # 최소 회전 반경 R (m, 추종 곡률 한도)
    plan_margin: float = 1.2                                                        # 계획 반경 = R x 이 값 (추종 여유)
    runway:      float = 0.08                                                       # 목표 직전 직진 구간 (m, 끝 방향 맞춤, 들어가는 방향)
    step:        float = 0.005                                                      # 경로 점 간격 (m)
    min_gear:    float = 0.001                                                      # 이보다 짧은 기어는 버림 (m)
    lookahead:   float = 0.03                                                       # 앞보기 거리 (m)
    v_max:       float = 0.04                                                       # 속도 (m/s)
    v_slow:      float = 0.02                                                       # 기어 끝 근처 속도 (m/s)
    slow_dist:   float = 0.05                                                       # 이 거리 안에서 v_slow (m)
    goal_tol:    float = 0.003                                                      # 기어 끝 판정 (m)


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))                                     # -pi ~ pi


# ==================================================================================
# 계획: Reeds-Shepp -> 기어 목록
# ==================================================================================
def plan(start, goal, p, final_dir=-1):
    """start, goal = (x, y, yaw), final_dir = 목표에 들어가는 방향 (+1 전진, -1 후진).

    반환: [(점 배열 (N, 3), 방향 +1 / -1), ...] (기어 순서). 목표 직전 runway 는 final_dir 방향 직진.
    """
    gx, gy, gyaw = goal
    pre    = (gx - final_dir * p.runway * math.cos(gyaw), gy - final_dir * p.runway * math.sin(gyaw), gyaw)
    radius = p.turn_radius * p.plan_margin
    path   = planner.path(tuple(start), pre, radius, 0.0, p.step)
    steps  = np.arange(p.step, p.runway + p.step / 2, p.step)                       # runway 점
    wps    = [(w.x, w.y, w.yaw, int(w.driving_direction)) for w in path.waypoints()]
    wps   += [(pre[0] + final_dir * s * math.cos(gyaw), pre[1] + final_dir * s * math.sin(gyaw), gyaw, final_dir)
              for s in steps]
    gears  = []
    for x, y, yaw, d in wps:
        if not gears or gears[-1][1] != d:
            gears.append(([], d))
        gears[-1][0].append((x, y, yaw))
    out = []
    for pts, d in gears:
        pts = np.array(pts, dtype=float)
        if len(pts) >= 2 and np.hypot(*np.diff(pts[:, :2], axis=0).T).sum() >= p.min_gear:
            out.append((pts, d))
    return out


# ==================================================================================
# 추종: 기어 하나 (pure pursuit, 곡률 제한)
# ==================================================================================
class GearTracker:

    def __init__(self, pts, direction, p):
        self.pts, self.dir, self.p = pts, direction, p
        seg      = np.hypot(*np.diff(pts[:, :2], axis=0).T)
        self.s   = np.concatenate(([0.0], np.cumsum(seg)))                          # 누적 거리 (m)
        self.i   = 0                                                                # 지나온 가장 가까운 점

    def to_end(self, x, y):
        """기어 끝까지 남은 거리 (끝 접선 방향, 진행 방향 +)."""
        ex, ey, eyaw = self.pts[-1]
        return self.dir * ((ex - x) * math.cos(eyaw) + (ey - y) * math.sin(eyaw))

    def target(self):
        """앞보기 점. 기어 끝을 넘으면 끝 접선을 따라 연장한 점."""
        j = int(np.searchsorted(self.s, self.s[self.i] + self.p.lookahead))
        if j < len(self.pts):
            return self.pts[j, :2]
        ex, ey, eyaw = self.pts[-1]
        extra = self.s[self.i] + self.p.lookahead - self.s[-1]
        return np.array([ex + self.dir * extra * math.cos(eyaw), ey + self.dir * extra * math.sin(eyaw)])

    def step(self, pose):
        """로봇 pose = (x, y, yaw) 로 한 주기. 반환: (v, w, 끝남)."""
        x, y, yaw = pose
        win    = self.pts[self.i: self.i + 40, :2]                                  # 앞쪽만 찾음 (되돌아가지 않음)
        self.i += int(np.argmin(np.hypot(win[:, 0] - x, win[:, 1] - y)))
        rest   = self.to_end(x, y)
        if rest <= self.p.goal_tol:
            return 0.0, 0.0, True
        tx, ty = self.target()
        dx, dy = tx - x, ty - y
        lx     = math.cos(yaw) * dx + math.sin(yaw) * dy                            # 로봇 기준 앞보기 점
        ly     = -math.sin(yaw) * dx + math.cos(yaw) * dy
        kmax   = 1.0 / self.p.turn_radius
        k      = max(-kmax, min(kmax, 2.0 * ly / max(lx * lx + ly * ly, 1e-9)))     # 원호 곡률, 차량 조건으로 자름
        v      = self.dir * (self.p.v_slow if rest < self.p.slow_dist else self.p.v_max)
        return v, v * k, False


class PathFollower:
    """기어를 차례로 추종. 기어가 바뀔 때 한 주기 멈춤."""

    def __init__(self, gears, p):
        self.trackers = [GearTracker(pts, d, p) for pts, d in gears]
        self.k        = 0                                                           # 지금 기어 번호

    @property
    def done(self):
        return self.k >= len(self.trackers)

    def step(self, pose):
        if self.done:
            return 0.0, 0.0
        v, w, end = self.trackers[self.k].step(pose)
        if end:
            self.k += 1                                                             # 다음 기어 (방향 전환)
        return v, w
