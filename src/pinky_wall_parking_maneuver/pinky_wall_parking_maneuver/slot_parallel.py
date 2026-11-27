"""src/pinky_wall_parking_maneuver/pinky_wall_parking_maneuver/slot_parallel.py - 평행 주차 칸 경로 (칸 밖 Hybrid A* + 칸 안 탐색)

역할: 앞뒤에 차가 선 평행 칸 (핑키 170 x 120 mm = 실제 6.0 x 2.0 m 칸을 중형 세단 비율로, 또는 230 mm 칸) 에
      차량식으로 들어가는 경로와, 들어간 뒤 마무리 정렬 경로를 만든다. ROS 미사용.
방법: 칸 안 = slot_search.InSlot (출차 역순, style multi = 다점 왕복 / video = 후진 S 자 + 짧은 전진).
      칸 밖 = 지금 자세 -> 칸 안 경로 시작 자세까지 hybrid_astar (이웃 상자 + 연석, 칸 안과 같은 여유).
      둘을 이어 car_path 기어 목록으로 낸다 (PathFollower 로 추종). 정렬 = InSlot.align (지금 자세 -> 허용 범위,
      추종 오차로 연석 여유 안쪽에 서도 빠져나오도록 정렬만 연석 여유 0).
      연석 = 선에서 curb_margin 안쪽까지 (칸 폭 120 mm - 차체 110 mm = 양쪽 5 mm 라 이웃 여유 8 mm 와 따로).
      칸 안 반경 = R x 1.2 (0.18 m): R 그대로면 추종이 더 꺾을 여유가 없어 끝 방향 오차 약 8 deg (이상 기구학 시험).
      keep_lines = 앞뒤 이웃 칸 사각형 (나눔 선 중심부터, 연석 선 ~ 차선 쪽 선) 도 장애물 (칸 선 넘지 않기).
      line_intrusion = 차체가 이웃 칸 안으로 들어간 깊이 (칸 띠로 자른 차체 다각형, 주행 감시용).
      neighbor_boxes = 앞뒤 이웃 칸 중심 assoc 안의 라이다 원통 -> 이웃 핑키 상자 (칸 방향).
실행: 라이브러리 (주차 노드, 단위 시험에서 호출).
"""

# --- 표준 라이브러리 ---------------------------------------------------------------
from dataclasses import dataclass                                                   # 파라미터 묶음
from dataclasses import replace                                                     # 파라미터 일부 바꾸기
import math                                                                         # 각도

# --- 이 패키지 ---------------------------------------------------------------------
from pinky_wall_parking_maneuver.hybrid_astar import gears_of                       # 기어 나누기
from pinky_wall_parking_maneuver.hybrid_astar import HybridAStar                    # 칸 밖 계획
from pinky_wall_parking_maneuver.obstacles import Obb                               # 상자
from pinky_wall_parking_maneuver.obstacles import Obstacles                         # 장애물 묶음
from pinky_wall_parking_maneuver.obstacles import robot_obb                         # 내 차체
from pinky_wall_parking_maneuver.slot_search import InSlot                          # 칸 안 탐색


@dataclass
class SlotParams:
    length:  float = 0.170                                                          # 칸 길이 (m, 차체 1.44 배)
    width:   float = 0.120                                                          # 칸 폭 (m, 차체 1.09 배)
    margin:  float = 0.008                                                          # 이웃 차 여유 (m, 시뮬 추종 오차 약 9 mm)
    curb_margin: float = 0.003                                                      # 연석 선 안쪽 여유 (m, 칸 폭 여유 5 mm 안)
    radius:  float = 0.18                                                           # 칸 안 계획 반경 (m, R 0.15 x 1.2)
    ds:      float = 0.001                                                          # 움직임 검사 간격 (m)
    reach:   float = 0.25                                                           # 한 번 움직임 최대 길이 (m)
    key_xy:  float = 0.002                                                          # 같은 상태 판정 x, y (m)
    key_deg: float = 1.0                                                            # 같은 상태 판정 yaw (deg)
    timeout: float = 40.0                                                           # 출차 탐색 시간 한도 (s)
    max_yaw: float = math.radians(45.0)                                             # 칸 방향과의 최대 각도 (rad)
    entry_gear: int = -1                                                            # 칸에 처음 들어가는 기어
    style:   str = "multi"                                                          # multi / video
    max_changes: int = 1                                                            # video: 칸 안 기어 전환 한도
    tol_yaw: float = math.radians(1.0)                                              # 정렬 허용 방향 (rad)
    tol_y:   float = 0.004                                                          # 정렬 허용 옆 (m)
    tol_x:   float = 0.015                                                          # 정렬 허용 길이 방향 (m)
    align_timeout: float = 5.0                                                      # 정렬 탐색 시간 한도 (s)
    align_min: float = 0.012                                                        # 정렬 한 번 움직임 최소 길이 (m)
    keep_lines: bool = False                                                        # True: 앞뒤 이웃 칸도 장애물 (칸 선 넘지 않기)


@dataclass
class Slot:
    cx: float                                                                       # 칸 중심 x (world, m)
    cy: float                                                                       # 칸 중심 y (world, m)
    yaw: float                                                                      # 칸 길이 방향 (world, rad)
    side: int = 1                                                                   # 차선 쪽 = 칸 왼쪽 (+1) / 오른쪽 (-1)

    def to_world(self, p):
        """Convert a slot-frame pose (x, y, yaw) to world."""
        x, y, a = p[0], self.side * p[1], self.side * p[2]                          # 오른쪽 칸은 y 뒤집기
        c, s = math.cos(self.yaw), math.sin(self.yaw)                               # 칸 방향
        return (self.cx + c * x - s * y, self.cy + s * x + c * y, math.atan2(math.sin(a + self.yaw),
                                                                             math.cos(a + self.yaw)))

    def to_slot(self, p):
        """Convert a world pose (x, y, yaw) to the slot frame."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)                               # 칸 방향
        dx, dy = p[0] - self.cx, p[1] - self.cy                                     # 칸 중심 기준
        a = math.atan2(math.sin(p[2] - self.yaw), math.cos(p[2] - self.yaw))        # 칸 기준 방향
        return (c * dx + s * dy, self.side * (-s * dx + c * dy), self.side * a)


def plan_slot_parallel(start, slot, neighbors, bounds, car, op, sp=None, hp=None):
    """Return gears from start into a parallel slot (outside: Hybrid A*, inside: reversed unpark), or None."""
    sp = sp or SlotParams()                                                         # 칸 파라미터
    inside = InSlot(slot, neighbors, op, sp).park()                                 # 칸 안 경로
    if inside is None:
        return None
    entry = inside[0][0][0]                                                         # 칸 안 경로 시작 자세
    curb = slot.to_world((0.0, -sp.width / 2 + sp.curb_margin - 0.5, 0.0))          # 연석 상자 (선 안쪽 여유)
    boxes = [Obb(b.cx, b.cy, b.yaw, b.hl + sp.margin, b.hw + sp.margin) for b in neighbors]
    boxes += line_boxes(slot, sp)                                                   # 칸 선 넘지 않기 (keep_lines)
    obs = Obstacles(boxes=boxes + [Obb(curb[0], curb[1], slot.yaw, 2.0, 0.5)])      # 칸 밖 장애물
    outside = HybridAStar(obs, bounds, car, op, hp).plan(start, entry, inside[0][1])
    if outside is None:
        return None
    pts = [(x, y, a, d) for g, d in outside for x, y, a in g]                       # 칸 밖 점
    pts += [(x, y, a, d) for g, d in inside for x, y, a in g[1:]]                   # 칸 안 점
    return gears_of(pts, car.min_gear)


def line_boxes(slot, sp):
    """Return the front / rear neighbor slot rectangles (world Obb, line centers) if keep_lines, else []."""
    if not sp.keep_lines:
        return []
    return [Obb(*slot.to_world((sx, 0.0, 0.0)), sp.length / 2, sp.width / 2) for sx in (-sp.length, sp.length)]


def line_intrusion(pose, slot, sp, op):
    """Return how far (m) the body at pose enters the front / rear neighbor slots (0 if it does not)."""
    pts = [slot.to_slot((x, y, 0))[:2] for x, y in robot_obb(pose, op).corners()]   # 차체 모서리
    for s in (1.0, -1.0):                                                           # 칸 띠 |y| <= 폭/2 로 자르기
        out = []
        for p, q in zip(pts, pts[1:] + pts[:1]):
            fp, fq = s * p[1] - sp.width / 2, s * q[1] - sp.width / 2               # <= 0 = 띠 안
            if fp <= 0.0:
                out.append(p)
            if fp * fq < 0.0:                                                       # 띠 경계를 지나는 변
                t = fp / (fp - fq)
                out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
        pts = out
        if not pts:
            return 0.0
    return max(0.0, max(abs(x) for x, _ in pts) - sp.length / 2)                    # 나눔 선 밖 깊이


def neighbor_boxes(pucks, slot, sp, op, assoc):
    """Return (boxes, notes): Pinky boxes (slot yaw) for pucks within assoc of the front / rear slot centers."""
    hl, hw = (op.body_x[1] - op.body_x[0]) / 2, op.body_w                           # 핑키 반치수
    boxes, notes = [], []
    for sx in (-sp.length, sp.length):
        ex, ey, ea = slot.to_world((sx, 0.0, 0.0))                                  # 이웃 칸 중심
        d = [math.hypot(c[0] - ex, c[1] - ey) for c in pucks]                       # 원통까지 거리
        k = min(range(len(d)), key=d.__getitem__) if d else None
        if k is not None and d[k] <= assoc:
            boxes.append(Obb(float(pucks[k][0]), float(pucks[k][1]), ea, hl, hw))
        notes.append(f"slot ({ex:.3f}, {ey:.3f}): " + (f"neighbor {d[k] * 1000:.1f} mm away"
                     if k is not None and d[k] <= assoc else "no neighbor"))
    return boxes, notes


def plan_align(pose, slot, neighbors, car, op, sp=None):
    """Return alignment gears from pose (inside the slot), [] if aligned, or None if not found."""
    sp = replace(sp or SlotParams(), curb_margin=0.0, style="multi")                # 정렬: 연석 선까지, 기어 번갈아
    moves = InSlot(slot, neighbors, op, sp).align(pose)                             # 정렬 움직임
    if not moves:
        return moves
    pts = [(x, y, a, d) for g, d in moves for x, y, a in g]                         # 점 (기어 포함)
    return gears_of(pts, car.min_gear)
