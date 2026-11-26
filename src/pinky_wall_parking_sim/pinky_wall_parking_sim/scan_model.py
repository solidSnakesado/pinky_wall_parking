"""src/pinky_wall_parking_sim/pinky_wall_parking_sim/scan_model.py - 이상적인 시뮬 스캔에 실물 라이다 특성 덧입히기

역할: MuJoCo 브리지 스캔 (잡음·점 빠짐 없음) 을 실물 RPLIDAR C1 처럼 바꾼다 (시뮬레이터 센서 잡음 모델, Gazebo ray noise 와 같은 역할).
방법 (스캔 1장, 순서대로):
  1. 스치는 광선 점 빠짐: 이웃 두 점으로 면 방향을 구하고 광선과 면 사이 각이 grazing_deg 미만이면 점 없음 (정답 거리로 판정)
  2. 근거리 오차: 실물 보정표 REAL_NEAR (잰 거리 -> 빼는 양) 의 역. r_잰 = r_참 + bias(r_잰) 를 고정점 반복으로 풂
  3. 랜덤 거리 잡음: 정규분포 sigma (m)
  4. near_drop 미만: 점 없음 (실물은 값이 들쭉날쭉, 처리 쪽에서 버림)
  5. 장착 yaw: angle_min 을 -yaw_offset 만큼 옮김 (처리 쪽 laser_yaw_offset_deg 로 되돌리면 정답 방향)
  6. 로봇 몸 점 (선택): 라이다 좌표 점 목록을 가장 가까운 광선에 넣음
- 값 근거: 1, 2, 4, 5 = 실측 (2026-11-17 ~ 18 녹화), 3 = 추정 (미측정), 6 = 위치만 실측
- 점 없음 = inf (LaserScan 관례, 처리 쪽 isfinite 로 걸러짐)
실행: 라이브러리 (scan_realism_node, 단위 시험에서 호출)
"""

# --- 표준 라이브러리 ---------------------------------------------------------------
from dataclasses import dataclass                                                   # 기준값 묶음
from dataclasses import field                                                       # 목록 기본값
import math                                                                         # 각도 변환

# --- 외부 라이브러리 ---------------------------------------------------------------
import numpy as np                                                                  # 거리 배열


NEAR_R    = [0.065, 0.075, 0.085, 0.095, 0.105]                                     # 잰 거리 (m), REAL_NEAR 와 같은 점
NEAR_BIAS = [0.0095, 0.0085, 0.0034, 0.0034, 0.0]                                   # 실제보다 멀게 재는 양 (m)


@dataclass
class ScanModel:
    grazing_deg:    float = 18.0                                                    # 이보다 얕게 스치면 점 없음 (deg)
    neighbor_gap:   float = 0.02                                                    # 면 방향 구할 이웃 점 최대 간격 (m)
    near_r:         list  = field(default_factory=lambda: list(NEAR_R))             # 근거리 표 거리 (m)
    near_bias:      list  = field(default_factory=lambda: list(NEAR_BIAS))          # 근거리 표 오차 (m)
    sigma:          float = 0.002                                                   # 랜덤 잡음 (m), 추정값
    near_drop:      float = 0.060                                                   # 이보다 가까우면 점 없음 (m)
    yaw_offset_deg: float = -2.2                                                    # 장착 yaw 오차 (deg)
    self_points:    list  = field(default_factory=list)                             # [x0, y0, x1, y1, ...] 라이다 좌표 (m)


def grazing_mask(r, angles, model):
    """Return True where the ray hits its surface at less than grazing_deg (point is lost)."""
    ok   = np.isfinite(r)
    p    = np.column_stack([np.where(ok, r, 0.0) * np.cos(angles), np.where(ok, r, 0.0) * np.sin(angles)])
    lost = np.zeros(len(r), dtype=bool)
    for i in range(1, len(r) - 1):
        if not (ok[i - 1] and ok[i] and ok[i + 1]):                                 # 끝점·끊긴 곳은 판정 안 함
            continue
        a, b = p[i - 1] - p[i], p[i + 1] - p[i]
        if np.hypot(*a) > model.neighbor_gap or np.hypot(*b) > model.neighbor_gap:
            continue
        t = p[i + 1] - p[i - 1]                                                     # 면 방향
        d = p[i] / r[i]                                                             # 광선 방향
        s = abs(d[0] * t[1] - d[1] * t[0]) / max(np.hypot(*t), 1e-9)                # sin (광선과 면 사이 각)
        lost[i] = s < math.sin(math.radians(model.grazing_deg))
    return lost


def near_bias(r_true, model, iters=30):
    """Return measured range for true range, inverting the real correction table (bias given at measured range)."""
    r = np.array(r_true, dtype=float)
    if not model.near_r:
        return r
    xs, ys = np.asarray(model.near_r), np.asarray(model.near_bias)
    m = r.copy()
    for _ in range(iters):                                                          # 표 기울기 0.6 미만 -> 수렴
        b = np.where(m <= xs[-1], np.interp(m, xs, ys), 0.0)                        # 표 밖 (멀리) = 0
        m = np.where(np.isfinite(r), r + b, r)
    return m


def add_self_points(r, angle_min, inc, model):
    """Put the robot body points (laser frame) into the nearest rays when closer than the current range."""
    pts = np.asarray(model.self_points, dtype=float).reshape(-1, 2)
    for x, y in pts:
        i = int(round((math.atan2(y, x) - angle_min) / inc)) % len(r)
        r[i] = min(r[i], math.hypot(x, y))
    return r


def apply(ranges, angle_min, inc, model, rng):
    """Return (ranges, angle_min) of the realistic scan for an ideal scan (inf = no return)."""
    r      = np.array(ranges, dtype=float)
    angles = angle_min + inc * np.arange(len(r))
    lost   = grazing_mask(r, angles, model)                                         # 정답 거리로 판정
    m      = near_bias(r, model)
    if model.sigma > 0:
        m = m + rng.normal(0.0, model.sigma, len(m))
    m[lost | (m < model.near_drop)] = np.inf
    out_min = angle_min - math.radians(model.yaw_offset_deg)                        # 실물이 보고하는 각
    if model.self_points:                                                           # 몸 점은 보고 각 기준 실측
        m = add_self_points(m, out_min, inc, model)
    return m, out_min
