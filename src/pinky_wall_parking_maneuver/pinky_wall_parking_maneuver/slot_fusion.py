"""src/pinky_wall_parking_maneuver/pinky_wall_parking_maneuver/slot_fusion.py - 카메라 칸 선 + 라이다 앞뒤 차 -> 평행 주차 칸 하나

역할: 바닥 지도의 선 (line_map.lines, 지도 좌표) 과 라이다로 찾은 이웃 차 중심으로 빈 칸 사각형을 정한다. ROS 미사용.
방법: 차선 쪽 선 (로봇에 가장 가까운 나란한 긴 선) 의 방향 e, 칸 쪽 법선 n 으로 s (차선을 따라), t (칸 쪽) 좌표를 둔다.
      - 방향·차선 쪽 경계: 카메라 차선 쪽 선 (없으면 칸 없음)
      - 연석 쪽 경계: 카메라 연석 선 (간격 width +- tol_w), 없으면 표준 폭 width (칸 규격 = 사전 지식)
      - 앞뒤 경계: 1) 나눔 선 두 개 (간격 length +- tol_l)  2) 나눔 선 하나 + 표준 길이 (반대쪽은 차 끝과 겹치지 않아야)
                   3) 라이다 앞뒤 차 사이 빈 공간 (차체 끝 사이 >= length) 가운데 + 표준 길이
      - 칸 띠 (0 < t < 폭) 안의 차 중심이 칸 안에 있으면 "차 있음" -> 빼고, 차체가 칸 안으로 overlap 넘게 들어와도 뺌
      - 칸은 카메라 차선 쪽 선이 보인 구간 (+- tol_l) 안에만 둠 (선 밖의 빈 곳을 칸으로 잡지 않게)
      - 차선 쪽 선은 로봇 진행 방향과 lane_deg 안으로 나란한 선만 (나눔 선 옆을 지날 때 나눔 선을 고르지 않게)
      - 카메라 후보와 라이다 빈 공간 후보가 같은 칸이면 중심 차이 agree 안일 때만 카메라 (check cam+lidar),
        다르면 라이다 (lidar(cam mismatch)). 한쪽만 있으면 cam only / lidar only. 임시 규칙: 실제 차는 칸 가운데에
        서 있지 않을 수 있어 라이다만으로는 칸 경계가 틀릴 수 있음 -> 카메라 나눔 선 정확도를 높인 뒤 다시 정함
      - 후보 순서: check (cam+lidar > cam only > lidar) -> 카메라로 본 앞뒤 변 수 -> 로봇에서 가까운 것
      결과 FusedSlot: 중심, 방향 (로봇 진행 방향 쪽), 길이, 폭, side (+1 = 칸이 진행 방향 오른쪽), 변별 출처 + check
실행: 라이브러리 (runs/fusion_eval.py, 주차 노드 탐색 모드에서 호출 예정)
"""

# --- 표준 라이브러리 ---------------------------------------------------------------
from dataclasses import dataclass                                                   # 파라미터 묶음
import math                                                                         # 각도

# --- 외부 라이브러리 ---------------------------------------------------------------
import numpy as np                                                                  # 벡터


@dataclass
class FusionParams:
    width:  float = 0.120                                                           # 표준 칸 폭 (m)
    length: float = 0.230                                                           # 표준 칸 길이 (m)
    tol_w:  float = 0.025                                                           # 폭 허용 (m)
    tol_l:  float = 0.035                                                           # 길이 허용 (m)
    car_hl: float = 0.059                                                           # 이웃 차 반 길이 (m)
    min_lane: float = 0.15                                                          # 차선 쪽 선 최소 길이 (m)
    par_deg: float = 10.0                                                           # 나란함 / 수직 판정 (deg)
    overlap: float = 0.005                                                          # 차체가 칸에 들어와도 되는 깊이 (m)
    agree:  float = 0.010                                                           # 카메라 칸과 라이다 칸 중심 일치 허용 (m)
    lane_deg: float = 30.0                                                          # 차선 쪽 선 = 로봇 진행 방향과 이 각도 안


@dataclass
class FusedSlot:
    cx: float                                                                       # 칸 중심 x (지도, m)
    cy: float                                                                       # 칸 중심 y (지도, m)
    yaw: float                                                                      # 칸 길이 방향 (rad)
    length: float                                                                   # 길이 (m)
    width: float                                                                    # 폭 (m)
    side: int                                                                       # +1 = 진행 방향 오른쪽
    src: dict                                                                       # 변별 출처 lane/curb/near/far


def frame(lines, robot, fp):
    """Return (e, n, t0, lane s-range) of the lane-side line nearest to robot (x, y, yaw), or None."""
    best, lim = None, math.cos(math.radians(fp.lane_deg))
    for th, d, s0, s1 in lines:
        m = np.array([-math.sin(th), math.cos(th)])
        r = m @ robot[:2] - d                                                       # 로봇 -> 선 (부호 있는 거리)
        along = abs(math.cos(th - robot[2])) >= lim                                 # 진행 방향과 나란함
        if along and s1 - s0 >= fp.min_lane and (best is None or abs(r) < abs(best[2])):
            best = (th, d, r, s0, s1)
    if best is None:
        return None
    th, d, r, s0, s1 = best
    u = np.array([math.cos(th), math.sin(th)])
    flip = u @ [math.cos(robot[2]), math.sin(robot[2])] < 0                         # 로봇 진행 방향 쪽으로
    e = -u if flip else u
    n = -math.copysign(1.0, r) * np.array([-math.sin(th), math.cos(th)])            # 로봇 반대쪽 = 칸 쪽
    rng = (-s1, -s0) if flip else (s0, s1)
    return e, n, float(n @ (np.array([-math.sin(th), math.cos(th)]) * d)), rng


def classify(lines, e, n, t0, fp):
    """Return (t of parallel lines, (s, t-range) of perpendicular lines) in the slot frame."""
    par, per = [], []
    for th, d, s0, s1 in lines:
        u = np.array([math.cos(th), math.sin(th)])
        m = np.array([-math.sin(th), math.cos(th)]) * d
        a, b = m + u * s0, m + u * s1                                               # 끝점
        if abs(u @ n) < math.sin(math.radians(fp.par_deg)):
            par.append(float((a + b) / 2 @ n - t0))
        elif abs(u @ e) < math.sin(math.radians(fp.par_deg)):
            ta, tb = sorted((a @ n - t0, b @ n - t0))
            per.append((float((a + b) / 2 @ e), ta, tb))
    return par, per


def fuse(lines, cars, robot, fp=None):
    """Return FusedSlot candidates (best first) from map lines, neighbor car centers and robot pose (x, y, yaw)."""
    fp = fp or FusionParams()
    fr = frame(lines, robot, fp)
    if fr is None:
        return []
    e, n, t0, lane = fr
    par, per = classify(lines, e, n, t0, fp)
    curb = [t for t in par if abs(t - fp.width) < fp.tol_w]
    width, curb_src = (min(curb, key=lambda t: abs(t - fp.width)), "cam") if curb else (fp.width, "std")
    divs = sorted(s for s, ta, tb in per if tb > 0.0 and ta < width)                # 칸 띠에 걸친 나눔 선
    car_s = sorted(float(np.array(c) @ e) for c in cars if 0.0 < np.array(c) @ n - t0 < width)
    cands = []                                                                      # (s0, s1, 앞 출처, 뒤 출처)
    for a, b in zip(divs, divs[1:]):
        if abs(b - a - fp.length) < fp.tol_l:
            cands.append((a, b, "cam", "cam"))
    for s in divs:
        cands += [(s - fp.length, s, "std", "cam"), (s, s + fp.length, "cam", "std")]
    for a, b in zip(car_s, car_s[1:]):
        lo, hi = a + fp.car_hl, b - fp.car_hl                                       # 차체 끝 사이 빈 공간
        if hi - lo >= fp.length:
            mid = (lo + hi) / 2
            cands.append((mid - fp.length / 2, mid + fp.length / 2, "lidar", "lidar"))
    ok = [q for q in cands if lane[0] - fp.tol_l <= q[0] and q[1] <= lane[1] + fp.tol_l
          and not any(q[0] - fp.car_hl + fp.overlap < c < q[1] + fp.car_hl - fp.overlap for c in car_s)]
    cam = sorted((q for q in ok if q[2] != "lidar"), key=lambda q: -((q[2] == "cam") + (q[3] == "cam")))
    picked, used = [], set()                                                        # (후보, check)
    for q in (q for q in ok if q[2] == "lidar"):
        mid = (q[0] + q[1]) / 2
        same = [k for k, c in enumerate(cam) if abs((c[0] + c[1]) / 2 - mid) < fp.length / 2]
        used.update(same)
        agree = [cam[k] for k in same if abs((cam[k][0] + cam[k][1]) / 2 - mid) <= fp.agree]
        picked.append((agree[0], "cam+lidar") if agree else (q, "lidar(cam mismatch)" if same else "lidar only"))
    picked += [(c, "cam only") for k, c in enumerate(cam) if k not in used]
    rank = {"cam+lidar": 3, "cam only": 2, "lidar(cam mismatch)": 1, "lidar only": 1}
    s_robot = float(np.array(robot[:2]) @ e)
    picked.sort(key=lambda p: (rank[p[1]], (p[0][2] == "cam") + (p[0][3] == "cam"),
                               -abs((p[0][0] + p[0][1]) / 2 - s_robot)), reverse=True)
    side = 1 if e[0] * n[1] - e[1] * n[0] < 0 else -1                               # n 이 e 오른쪽이면 +1
    best = []                                                                       # 같은 칸은 가장 좋은 것만
    for (s0, s1, near, far), check in picked:
        q = e * (s0 + s1) / 2 + n * (t0 + width / 2)
        f = FusedSlot(float(q[0]), float(q[1]), math.atan2(e[1], e[0]), s1 - s0, width, side,
                      {"lane": "cam", "curb": curb_src, "near": near, "far": far, "check": check})
        if all(math.hypot(f.cx - g.cx, f.cy - g.cy) > fp.tol_l for g in best):
            best.append(f)
    return best
