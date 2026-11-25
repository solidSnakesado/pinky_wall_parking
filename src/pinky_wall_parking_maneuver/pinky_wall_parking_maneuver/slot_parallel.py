"""src/pinky_wall_parking_maneuver/pinky_wall_parking_maneuver/slot_parallel.py - 좁은 평행 주차 칸 경로 (출차 역순 + Hybrid A*)

역할: 앞뒤에 차가 선 평행 주차 칸 (핑키 170 x 120 mm, 실제 6.0 x 2.0 m 칸 / 중형 세단 비율) 에 차량식으로 들어가는 경로를 만든다. ROS 미사용.
방법: 칸 안 = 출차 역순. 칸 안 주차 자세에서 차선 쪽으로 빠져나오는 움직임 조합을 너비 우선으로 찾는다.
      한 번 움직임 = 최대 조향 (좌 / 우) 또는 직진으로 충돌 직전까지 (또는 그 절반), 다음 움직임은 반대 기어 (실제 차량 다점 평행 주차).
      실제 차량 방식: 칸 방향과의 각도 <= 45 deg (운전 교본의 45 도 진입), 칸에 처음 들어가는 기어 = entry_gear
      (-1 = 후진 진입, 실제 차와 보이는 동작이 같음 / +1 = 전진 진입, 바퀴 축 쪽 끝이 먼저라 실제 차 후진과 기하가 같음).
      칸 진입 움직임은 차체 중심이 칸 안 (이웃 윗면 아래, 칸 길이 안) 인 자세에서 칸 밖까지 한 번에 이어진다. 칸 밖 상태는 탐색하지 않는다.
      차량식 움직임은 거꾸로 가도 같은 조건 (|w| <= |v| / R) 이라 찾은 출차를 순서·방향을 뒤집어 주차 경로로 쓴다.
      칸 밖 = 지금 자세 -> 출차 끝 자세까지 hybrid_astar. 둘을 이어 car_path 기어 목록으로 낸다 (PathFollower 로 추종).
      칸 좌표: 원점 = 칸 중심, x = 칸 길이 방향 (주차 차량 진행 방향), +y = 차선 쪽, -y 끝 = 연석 (넘으면 안 됨).
      칸 안 충돌 = 내 차체 OBB 대 이웃 OBB / 연석 상자 SAT (스칼라 계산, 1 mm 마다).
실행: 라이브러리 (주차 노드, 단위 시험에서 호출).
"""

# --- 표준 라이브러리 ---------------------------------------------------------------
from collections import deque                                                       # 너비 우선 큐
from dataclasses import dataclass                                                   # 파라미터 묶음
import math                                                                         # 삼각 함수
import time                                                                         # 시간 제한

# --- 이 패키지 ---------------------------------------------------------------------
from pinky_wall_parking_maneuver.hybrid_astar import gears_of                       # 기어 나누기
from pinky_wall_parking_maneuver.hybrid_astar import HybridAStar                    # 칸 밖 계획
from pinky_wall_parking_maneuver.obstacles import Obb                               # 상자
from pinky_wall_parking_maneuver.obstacles import Obstacles                         # 장애물 묶음


# ==================================================================================
# 파라미터
# ==================================================================================
@dataclass
class SlotParams:
    length:  float = 0.170                                                          # 칸 길이 (m, 차체 1.44 배)
    width:   float = 0.120                                                          # 칸 폭 (m, 차체 1.09 배)
    margin:  float = 0.005                                                          # 칸 안 여유 (m)
    radius:  float = 0.15                                                           # 칸 안 반경 (m, 최소 회전 반경)
    ds:      float = 0.001                                                          # 움직임 검사 간격 (m)
    reach:   float = 0.25                                                           # 한 번 움직임 최대 길이 (m)
    key_xy:  float = 0.002                                                          # 같은 상태 판정 x, y (m)
    key_deg: float = 1.0                                                            # 같은 상태 판정 yaw (deg)
    timeout: float = 40.0                                                           # 탐색 시간 한도 (s, 170 칸 후진 진입 약 23 s)
    max_yaw: float = math.radians(45.0)                                             # 칸 방향과의 최대 각도 (rad)
    entry_gear: int = -1                                                            # 칸에 처음 들어가는 기어


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


def sat_hit(r, b):
    """Return True if boxes r and b, each (cx, cy, cos, sin, hl, hw), overlap (scalar SAT)."""
    dx, dy = b[0] - r[0], b[1] - r[1]                                               # 중심 차
    for c, s in ((r[2], r[3]), (-r[3], r[2]), (b[2], b[3]), (-b[3], b[2])):
        ra = r[4] * abs(c * r[2] + s * r[3]) + r[5] * abs(-c * r[3] + s * r[2])     # r 투영 반길이
        rb = b[4] * abs(c * b[2] + s * b[3]) + b[5] * abs(-c * b[3] + s * b[2])     # b 투영 반길이
        if abs(dx * c + dy * s) > ra + rb:
            return False                                                            # 분리 축
    return True


class InSlot:
    """Unpark search inside one slot (slot frame); park = the reversed unpark."""

    def __init__(self, slot, neighbors, op, sp):
        self.slot, self.op, self.sp = slot, op, sp                                  # 칸, 차체 치수, 파라미터
        self.hl = (op.body_x[1] - op.body_x[0]) / 2                                 # 차체 반길이
        self.off = (op.body_x[0] + op.body_x[1]) / 2                                # 차체 중심 x (base 기준)
        m, boxes, top = sp.margin, [], sp.width / 2                                 # 여유, 장애물, 빠져나온 기준
        for b in neighbors:
            x, y, a = slot.to_slot((b.cx, b.cy, b.yaw))                             # 이웃 (칸 좌표)
            boxes.append((x, y, math.cos(a), math.sin(a), b.hl + m, b.hw + m))
            top = max(top, max(slot.to_slot((cx, cy, 0.0))[1] for cx, cy in b.corners()))
        boxes.append((0.0, -sp.width / 2 - m - 0.5, 1.0, 0.0, 2.0, 0.5))            # 연석 (칸 -y 끝)
        self.boxes, self.top = boxes, top + m                                       # 장애물, 이 위면 빠져나옴

    def body(self, p):
        """Return the body box (cx, cy, cos, sin, hl, hw) at slot pose p."""
        c, s = math.cos(p[2]), math.sin(p[2])                                       # 방향
        return (p[0] + self.off * c, p[1] + self.off * s, c, s, self.hl, self.op.body_w)

    def hit(self, p):
        """Return True if the body at slot pose p overlaps a neighbor or the curb."""
        if abs(p[2]) > self.sp.max_yaw:
            return True                                                             # 실제 차량 진입 각도 넘음
        r = self.body(p)                                                            # 내 차체
        return any(sat_hit(r, b) for b in self.boxes)

    def out(self, p):
        """Return True if the whole body is beyond the neighbors on the lane side."""
        r = self.body(p)                                                            # 내 차체
        return r[1] - r[4] * abs(r[3]) - r[5] * abs(r[2]) > self.top

    def deep(self, p):
        """Return True if the body center at p is inside the slot band (below the neighbors, within the length)."""
        r = self.body(p)                                                            # 내 차체
        return r[1] < self.top and abs(r[0]) < self.sp.length / 2

    def stroke(self, p, d, k):
        """Return slot poses every ds along one move (gear d, curvature k) until a hit."""
        out, (x, y, th) = [], p                                                     # 결과, 자세
        while len(out) * self.sp.ds < self.sp.reach:
            t2 = th + d * k * self.sp.ds                                            # 다음 방향
            if k == 0:
                nx, ny = x + d * self.sp.ds * math.cos(th), y + d * self.sp.ds * math.sin(th)
            else:
                nx, ny = x + (math.sin(t2) - math.sin(th)) / k, y - (math.cos(t2) - math.cos(th)) / k
            if self.hit((nx, ny, t2)):
                break
            x, y, th = nx, ny, t2
            out.append((x, y, th))
        return out

    def unpark(self):
        """Return unpark moves [(slot poses, gear), ...] from the parked pose, or None."""
        start = (-self.off, 0.0, 0.0)                                               # 차체 중심 = 칸 중심
        if self.hit(start):
            return None
        k, sp, seen = 1.0 / self.sp.radius, self.sp, set()                          # 곡률, 파라미터, 본 상태
        q, t0 = deque([(start, 0, None)]), time.monotonic()                         # (자세, 기어, 부모)
        while q and time.monotonic() - t0 < sp.timeout:
            node = q.popleft()
            for d in (1, -1):
                if d == node[1]:
                    continue
                for kk in (k, -k, 0.0):
                    pts = self.stroke(node[0], d, kk)                               # 한 번 움직임
                    hit_out = next((i for i, z in enumerate(pts) if self.out(z)), None)
                    if hit_out is not None:
                        if d == -sp.entry_gear and self.deep(node[0]):              # 칸 진입 = entry_gear 한 번에
                            return self.chain((pts[:hit_out + 1], d, node))
                        pts = pts[:hit_out]                                         # 칸 밖 상태는 탐색 안 함
                    for z in {pts[-1], pts[len(pts) // 2]} if pts else ():
                        key = (round(z[0] / sp.key_xy), round(z[1] / sp.key_xy), round(math.degrees(z[2]) / sp.key_deg))
                        if key not in seen:
                            seen.add(key)
                            q.append((z, d, (pts[:pts.index(z) + 1], d, node)))
        return None

    @staticmethod
    def chain(last):
        """Return moves from the root to last (each node = (poses, gear, parent node))."""
        moves = []                                                                  # 결과 (거꾸로)
        while last is not None:
            pts, d, node = last
            moves.append((pts, d))
            last = node[2]
        return moves[::-1]

    def park(self):
        """Return park moves in world [(world poses, gear), ...] (reversed unpark), or None."""
        moves = self.unpark()                                                       # 출차
        if moves is None:
            return None
        start = (-self.off, 0.0, 0.0)                                               # 주차 자세
        out, prev = [], start                                                       # 결과, 이전 끝점
        for pts, d in moves:
            out.append(([prev] + pts, d))
            prev = pts[-1]
        return [([self.slot.to_world(p) for p in pts[::-1]], -d) for pts, d in out[::-1]]


def plan_slot_parallel(start, slot, neighbors, bounds, car, op, sp=None, hp=None):
    """Return gears from start into a parallel slot (outside: Hybrid A*, inside: reversed unpark), or None."""
    sp = sp or SlotParams()                                                         # 칸 파라미터
    inside = InSlot(slot, neighbors, op, sp).park()                                 # 칸 안 경로
    if inside is None:
        return None
    entry = inside[0][0][0]                                                         # 칸 안 경로 시작 자세
    curb = slot.to_world((0.0, -sp.width / 2 - sp.margin - 0.5, 0.0))               # 연석 상자 중심
    boxes = [Obb(b.cx, b.cy, b.yaw, b.hl + sp.margin, b.hw + sp.margin) for b in neighbors]
    obs = Obstacles(boxes=boxes + [Obb(curb[0], curb[1], slot.yaw, 2.0, 0.5)])      # 칸 밖 장애물
    outside = HybridAStar(obs, bounds, car, op, hp).plan(start, entry, inside[0][1])
    if outside is None:
        return None
    pts = [(x, y, a, d) for g, d in outside for x, y, a in g]                       # 칸 밖 점
    pts += [(x, y, a, d) for g, d in inside for x, y, a in g[1:]]                   # 칸 안 점
    return gears_of(pts, car.min_gear)
