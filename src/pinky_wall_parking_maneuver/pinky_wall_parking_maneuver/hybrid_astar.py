"""src/pinky_wall_parking_maneuver/pinky_wall_parking_maneuver/hybrid_astar.py - 장애물을 피하는 차량식 경로 계획 (Hybrid A*)

역할: 장애물 (obstacles.Obstacles) 과 작업 영역 안에서 시작 자세 -> 목표 자세 차량식 경로를 찾는다. ROS 미사용.
방법: Hybrid A* (Dolgov 2008). 상태 격자 x, y 20 mm x yaw 5 deg, 같은 칸은 비용이 낮은 노드만 남긴다.
      펼치기 = 호 길이 50 mm, 곡률 {-1, -0.5, 0, 0.5, 1} / R_plan, 전진 / 후진 (제자리 회전 없음).
      비용 = 거리 + 후진 가중 + 기어 전환 + 조향 변화.
      휴리스틱 = max(장애물 무시 Reeds-Shepp 길이 (rsplan), 장애물 반영 2D 거리 (grid_heuristic)). 2D 로 못 가는 노드는 버린다.
      RS 길이는 느려서 (약 0.2 ms) 넣을 때는 2D 거리만 쓰고, 꺼낼 때 RS 를 반영해 다시 넣는다 (지연 계산).
      노드 shot_every 개마다 목표까지 Reeds-Shepp 경로를 시도해 충돌이 없으면 끝 (분석적 확장).
      충돌 = 10 mm 마다 obstacles.pose_collides + 작업 영역 밖. 목표 직전 runway 는 final_dir 방향 직진 (car_path.plan 과 같음).
      결과는 car_path.plan 과 같은 기어 목록 [(점 (N, 3), 방향 +1 / -1), ...] 이라 PathFollower 로 따라간다.
실행: 라이브러리 (경로 계획 노드, 단위 시험에서 호출). rsplan 은 pip 패키지 (pip install rsplan).
"""

# --- 표준 라이브러리 ---------------------------------------------------------------
from dataclasses import dataclass                                                   # 파라미터 묶음
import heapq                                                                        # 우선순위 큐
import math                                                                         # 삼각 함수
import time                                                                         # 시간 제한

# --- 외부 라이브러리 ---------------------------------------------------------------
import numpy as np                                                                  # 점 배열

# --- 이 패키지 ---------------------------------------------------------------------
from pinky_wall_parking_maneuver.car_path import planner                            # rsplan Reeds-Shepp
from pinky_wall_parking_maneuver.car_path import wrap                               # 각도 정리
from pinky_wall_parking_maneuver.grid_heuristic import GridHeuristic                # 장애물 반영 2D 거리
from pinky_wall_parking_maneuver.obstacles import pose_collides                     # 자세 충돌


# ==================================================================================
# 파라미터
# ==================================================================================
@dataclass
class HaParams:
    xy_res:      float = 0.02                                                       # 격자 x, y (m)
    yaw_bins:    int = 72                                                           # 격자 yaw 칸 수 (5 deg)
    step:        float = 0.05                                                       # 펼치기 호 길이 (m)
    sample:      float = 0.01                                                       # 충돌 검사 / 경로 점 간격 (m)
    steer:       tuple = (-1.0, -0.5, 0.0, 0.5, 1.0)                                # 곡률 / (1 / R_plan)
    rev_cost:    float = 1.5                                                        # 후진 거리 가중
    switch_cost: float = 0.05                                                       # 기어 전환 비용 (m)
    steer_cost:  float = 0.005                                                      # 조향 변화 비용 (m)
    shot_every:  int = 10                                                           # 분석적 확장 주기 (노드)
    max_nodes:   int = 30000                                                        # 펼칠 노드 한도
    timeout:     float = 5.0                                                        # 시간 한도 (s)


@dataclass
class Node:
    pose:   tuple                                                                   # (x, y, yaw)
    g:      float                                                                   # 지나온 비용
    d:      int                                                                     # 방향 (+1, -1, 0 = 시작)
    k:      float                                                                   # 곡률 계수
    pts:    list                                                                    # 이 노드까지 점 [(x, y, yaw, d)]
    parent: object                                                                  # 부모 노드


def arc(pose, d, kappa, length, ds):
    """Return poses every ds along a constant-curvature path (direction d, curvature kappa)."""
    x, y, yaw = pose                                                                # 시작 자세
    out = []                                                                        # 결과
    n = max(1, int(round(length / ds)))                                             # 점 수
    for i in range(1, n + 1):
        s = d * length * i / n                                                      # 부호 있는 호 길이
        if abs(kappa) < 1e-9:
            out.append((x + s * math.cos(yaw), y + s * math.sin(yaw), yaw))
        else:
            th = yaw + kappa * s                                                    # 끝 방향
            out.append((x + (math.sin(th) - math.sin(yaw)) / kappa,
                        y - (math.cos(th) - math.cos(yaw)) / kappa, wrap(th)))
    return out


def gears_of(points, min_gear=0.001):
    """Group [(x, y, yaw, d)] into [(points (N, 3), d), ...] by direction."""
    gears = []                                                                      # 기어 목록
    for x, y, yaw, d in points:
        if not gears or gears[-1][1] != d:
            start = [gears[-1][0][-1]] if gears else []                             # 전환점 이어 붙임
            gears.append((start, d))
        gears[-1][0].append((x, y, yaw))
    out = []                                                                        # 결과
    for pts, d in gears:
        pts = np.array(pts, dtype=float)                                            # 점 배열
        if len(pts) >= 2 and np.hypot(*np.diff(pts[:, :2], axis=0).T).sum() >= min_gear:
            out.append((pts, d))
    return out


class HybridAStar:
    """Car-like planner over obstacles inside a rectangle (xmin, xmax, ymin, ymax)."""

    def __init__(self, obs, bounds, car, op, hp=None):
        self.obs, self.bounds, self.car, self.op = obs, bounds, car, op             # 장애물, 영역, 차량, 장애물 파라미터
        self.hp     = hp or HaParams()                                              # 탐색 파라미터
        self.radius = car.turn_radius * car.plan_margin                             # 계획 반경 R_plan
        self.expanded = 0                                                           # 펼친 노드 수 (진단)
        hl  = (op.body_x[1] - op.body_x[0]) / 2                                     # 차체 반길이
        off = (op.body_x[0] + op.body_x[1]) / 2                                     # 차체 중심 x
        self.off  = off                                                             # 차체 중심 x
        self.r_me = math.hypot(hl + op.margin, op.body_w + op.margin)               # 차체 감싸는 원
        near = [((b.cx, b.cy), math.hypot(b.hl, b.hw)) for b in obs.boxes]          # 상자 감싸는 원
        near += [(tuple(c), r) for c, r in obs.circles] + [(tuple(q), 0.0) for q in obs.points]
        self.near_c = np.array([c for c, _ in near]).reshape(-1, 2)                 # 감싸는 원 중심
        self.near_r = np.array([r for _, r in near]) + self.r_me                    # 겹칠 수 있는 거리

    def free(self, pose):
        """Return True if the robot at pose is inside the bounds and hits nothing."""
        x, y, yaw = pose                                                            # 자세
        c, s = math.cos(yaw), math.sin(yaw)                                         # 방향 성분
        bx, bw = self.op.body_x, self.op.body_w                                     # 차체 치수
        for lx, ly in ((bx[0], -bw), (bx[0], bw), (bx[1], -bw), (bx[1], bw)):
            px, py = x + c * lx - s * ly, y + s * lx + c * ly                       # 꼭짓점
            if not (self.bounds[0] <= px <= self.bounds[1] and self.bounds[2] <= py <= self.bounds[3]):
                return False
        cx, cy = x + c * self.off, y + s * self.off                                 # 차체 중심
        d = np.hypot(self.near_c[:, 0] - cx, self.near_c[:, 1] - cy)                # 감싸는 원 거리
        return not (np.any(d <= self.near_r) and pose_collides(pose, self.obs, self.op))

    def key(self, pose):
        """Return the grid cell of a pose."""
        b = int(round(wrap(pose[2]) / (2 * math.pi / self.hp.yaw_bins))) % self.hp.yaw_bins
        return (int(round(pose[0] / self.hp.xy_res)), int(round(pose[1] / self.hp.xy_res)), b)

    def rs(self, a, b):
        """Return the Reeds-Shepp path from a to b (rsplan)."""
        return planner.path(tuple(a), tuple(b), self.radius, 0.0, self.hp.sample)

    def center(self, pose):
        """Return the body center of a pose."""
        return pose[0] + self.off * math.cos(pose[2]), pose[1] + self.off * math.sin(pose[2])

    def h(self, pose, pre):
        """Return max(Reeds-Shepp length, obstacle-aware 2D distance) to pre (inf if unreachable)."""
        d2 = self.grid(self.center(pose))                                           # 2D 거리
        return d2 if math.isinf(d2) else max(d2, self.rs(pose, pre).total_length)

    def shot(self, node, pre):
        """Return Reeds-Shepp points [(x, y, yaw, d)] to pre if collision free, else None."""
        wps = [(w.x, w.y, w.yaw, int(w.driving_direction)) for w in self.rs(node.pose, pre).waypoints()]
        return wps if all(self.free(w[:3]) for w in wps[1:]) else None

    def plan(self, start, goal, final_dir=-1):
        """Return gears from start to goal (runway in final_dir), or None if not found."""
        hp, run = self.hp, self.car.runway                                          # 파라미터, runway 길이
        gx, gy, gyaw = goal                                                         # 목표
        pre = (gx - final_dir * run * math.cos(gyaw), gy - final_dir * run * math.sin(gyaw), gyaw)
        runway = [(*p, final_dir) for p in arc(pre, final_dir, 0.0, run, hp.sample)]
        if not self.free(start) or not all(self.free(p[:3]) for p in [pre] + [r[:3] for r in runway]):
            return None
        self.grid = GridHeuristic(self.obs, self.bounds, self.center(pre), self.op.body_w)
        if math.isinf(self.h(start, pre)):
            return None                                                             # 2D 로도 못 감
        root = Node(tuple(start), 0.0, 0, 0.0, [(*start, 0)], None)                 # 시작 노드
        heap, best, t0 = [(self.h(start, pre), 0, root, True)], {}, time.monotonic()
        tie, self.expanded = 1, 0                                                   # 같은 f 정렬용, 진단
        while heap and self.expanded < hp.max_nodes and time.monotonic() - t0 < hp.timeout:
            f, _, node, exact = heapq.heappop(heap)
            if not exact:                                                           # 지연 휴리스틱: 꺼낼 때 RS 반영
                heapq.heappush(heap, (node.g + max(f - node.g, self.rs(node.pose, pre).total_length), tie, node, True))
                tie += 1
                continue
            self.expanded += 1
            if self.expanded % hp.shot_every == 1:
                tail = self.shot(node, pre)                                         # 분석적 확장
                if tail is not None:
                    return self.build(node, tail[1:] + runway)
            for d in (1, -1):
                for k in hp.steer:
                    pts = arc(node.pose, d, k / self.radius, hp.step, hp.sample)    # 펼치기
                    if not all(self.free(p) for p in pts):
                        continue
                    g = node.g + hp.step * (hp.rev_cost if d < 0 else 1.0)          # 거리 비용
                    g += hp.switch_cost * (node.d != 0 and node.d != d) + hp.steer_cost * abs(k - node.k)
                    cell = self.key(pts[-1])                                        # 격자 칸
                    hv = self.grid(self.center(pts[-1]))                            # 2D 거리 (RS 는 꺼낼 때)
                    if math.isinf(hv) or (cell in best and best[cell] <= g):
                        continue
                    best[cell] = g
                    child = Node(pts[-1], g, d, k, [(*p, d) for p in pts], node)    # 자식 노드
                    heapq.heappush(heap, (g + hv, tie, child, False))
                    tie += 1
        return None

    def build(self, node, tail):
        """Return gears from the node chain plus the tail points."""
        chain = []                                                                  # 노드 점 (거꾸로)
        while node is not None:
            chain.append(node.pts)
            node = node.parent
        pts = [p for seg in reversed(chain) for p in seg] + list(tail)              # 전체 점
        first = next((p[3] for p in pts if p[3] != 0), 1)                           # 첫 방향
        pts = [(x, y, yaw, d if d != 0 else first) for x, y, yaw, d in pts]         # 시작점 방향 채움
        return gears_of(pts, self.car.min_gear)
