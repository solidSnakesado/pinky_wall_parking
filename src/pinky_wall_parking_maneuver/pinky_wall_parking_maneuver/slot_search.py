"""src/pinky_wall_parking_maneuver/pinky_wall_parking_maneuver/slot_search.py - 평행 칸 안 움직임 탐색 (출차 역순, 마무리 정렬)

역할: 앞뒤에 차가 선 평행 칸 안에서 차량식 움직임 조합을 찾는다. ROS 미사용. slot_parallel 에서 호출.
방법: 한 번 움직임 = 최대 조향 (좌 / 우) 또는 직진, 충돌 직전까지 또는 그 일부. 너비 우선 탐색 (움직임 수가 적은 순).
      출차: 주차 자세 -> 차선 쪽으로 빠져나오는 조합을 찾아 순서·방향을 뒤집어 주차 경로로 (거꾸로 가도 |w| <= |v| / R).
        칸에 들어가는 기어 = entry_gear, 그 기어 구간은 칸 안 (차체 중심이 이웃 윗면 아래, 칸 길이 안) 에서 칸 밖까지.
        style "multi" = 움직일 때마다 기어 전환 (다점 왕복, 좁은 칸).
        style "video" = 같은 기어 안에서 조향을 바꿀 수 있고 기어 전환 <= max_changes (실제 차 영상: 후진 S 자 + 짧은 전진).
      정렬: 지금 자세 -> 주차 허용 범위. 움직임 길이에 방향을 0 으로 만드는 길이 (|yaw| x R) 를 더함 (마무리 정렬 왕복).
      실제 차량 조건: 칸 방향과의 각도 <= max_yaw (45 deg, 운전 교본 진입 각도).
      칸 좌표: 원점 = 칸 중심, x = 칸 길이 방향, +y = 차선 쪽, -y 끝 = 연석 선 (curb_margin 안쪽까지만). 충돌 = 스칼라 SAT, ds 마다.
실행: 라이브러리 (칸 Slot, 파라미터 SlotParams 는 slot_parallel 에 있음).
"""

# --- 표준 라이브러리 ---------------------------------------------------------------
from collections import deque                                                       # 너비 우선 큐
import math                                                                         # 삼각 함수
import time                                                                         # 시간 제한


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
    """Moves inside one slot (slot frame): reversed unpark and final alignment."""

    def __init__(self, slot, neighbors, op, sp):
        self.slot, self.op, self.sp = slot, op, sp                                  # 칸, 차체 치수, 파라미터
        self.hl = (op.body_x[1] - op.body_x[0]) / 2                                 # 차체 반길이
        self.off = (op.body_x[0] + op.body_x[1]) / 2                                # 차체 중심 x (base 기준)
        self.goal = (-self.off, 0.0, 0.0)                                           # 주차 자세 (차체 중심 = 칸 중심)
        m, boxes, top = sp.margin, [], sp.width / 2                                 # 여유, 장애물, 빠져나온 기준
        for b in neighbors:
            x, y, a = slot.to_slot((b.cx, b.cy, b.yaw))                             # 이웃 (칸 좌표)
            boxes.append((x, y, math.cos(a), math.sin(a), b.hl + m, b.hw + m))
            top = max(top, max(slot.to_slot((cx, cy, 0.0))[1] for cx, cy in b.corners()))
        curb = -sp.width / 2 + sp.curb_margin - 0.5                                 # 연석 상자 중심 (선 안쪽 여유)
        boxes.append((0.0, curb, 1.0, 0.0, 2.0, 0.5))
        if sp.keep_lines:                                                           # 칸 선 넘지 않기: 앞뒤 이웃 칸 (선 중심)
            lm = sp.line_margin                                                     # 칸 선 여유 (칸 쪽으로 늘림)
            boxes += [(sx - math.copysign(lm / 2, sx), 0.0, 1.0, 0.0, sp.length / 2 + lm / 2, sp.width / 2)
                      for sx in (-sp.length, sp.length)]
        self.boxes, self.top = boxes, top + m                                       # 장애물, 이 위면 빠져나옴

    def body(self, p):
        """Return the body box (cx, cy, cos, sin, hl, hw) at slot pose p."""
        c, s = math.cos(p[2]), math.sin(p[2])                                       # 방향
        return (p[0] + self.off * c, p[1] + self.off * s, c, s, self.hl, self.op.body_w)

    def hit(self, p):
        """Return True if the pose breaks the angle limit or the body overlaps a neighbor or the curb."""
        return abs(p[2]) > self.sp.max_yaw or any(sat_hit(self.body(p), b) for b in self.boxes)

    def out(self, p):
        """Return True if the whole body is beyond the neighbors on the lane side."""
        r = self.body(p)
        return r[1] - r[4] * abs(r[3]) - r[5] * abs(r[2]) > self.top

    def deep(self, p):
        """Return True if the body center at p is inside the slot band (below the neighbors, within the length)."""
        r = self.body(p)                                                            # 내 차체
        return r[1] < self.top and abs(r[0]) < self.sp.length / 2

    def stroke(self, p, d, k):
        """Return slot poses every ds along one move (gear d, curvature k) until a hit."""
        out, (x, y, th), ds = [], p, self.sp.ds                                     # 결과, 자세, 간격
        while len(out) * ds < self.sp.reach:
            t2 = th + d * k * ds                                                    # 다음 방향
            if k == 0:
                nx, ny = x + d * ds * math.cos(th), y + d * ds * math.sin(th)
            else:
                nx, ny = x + (math.sin(t2) - math.sin(th)) / k, y - (math.cos(t2) - math.cos(th)) / k
            if self.hit((nx, ny, t2)):
                break
            x, y, th = nx, ny, t2
            out.append((x, y, th))
        return out

    def search(self, start, done, timeout, extra=None, min_n=1):
        """Breadth-first search from start until done(pts, d, k, node) gives an end index; moves or None."""
        k, sp, seen, video = 1.0 / self.sp.radius, self.sp, set(), self.sp.style == "video"
        q, t0 = deque([(start, 0, None, 0, self.deep(start))]), time.monotonic()    # (자세, 기어, 부모, 전환, 구간 시작 깊음)
        while q and time.monotonic() - t0 < timeout:
            node = q.popleft()
            for d in (1, -1):
                ch = node[3] + (node[1] != 0 and d != node[1])                      # 기어 전환 수
                if (not video and d == node[1]) or (video and ch > sp.max_changes):
                    continue
                deep0 = node[4] if d == node[1] else self.deep(node[0])             # 이 기어 구간 시작이 칸 안
                for kk in (k, -k, 0.0):
                    pts = self.stroke(node[0], d, kk)                               # 한 번 움직임
                    i = done(pts, d, kk, (node[0], d, node[2], ch, deep0))
                    if i is not None:
                        return self.chain((pts[:i + 1], d, node))
                    # 칸 밖으로 나간 점부터는 탐색하지 않음
                    pts = pts[:next((i for i, z in enumerate(pts) if self.out(z)), len(pts))]
                    more = (len(pts) // 4,) if video else ()                        # video: 짧은 움직임도
                    more += tuple(extra(pts, d, kk, node[0])) if extra else ()      # 정렬: 방향 0 길이
                    for n in sorted(n for n in {len(pts), len(pts) // 2, *more} if n >= min_n):
                        z = pts[n - 1]
                        key = (round(z[0] / sp.key_xy), round(z[1] / sp.key_xy), round(math.degrees(z[2]) / sp.key_deg),
                               *((d, ch) if video else ()))
                        if key not in seen:
                            seen.add(key)
                            q.append((z, d, (pts[:n], d, node), ch, deep0))
        return None

    @staticmethod
    def chain(last):
        """Return moves from the root to last (each node = (poses, gear, parent node, ...))."""
        moves = []                                                                  # 결과 (거꾸로)
        while last is not None:
            pts, d, node = last
            moves.append((pts, d))
            last = node[2]
        return moves[::-1]

    def park(self):
        """Return park moves in world [(world poses, gear), ...] (reversed unpark), or None."""
        def done(pts, d, k, node):
            i = next((i for i, z in enumerate(pts) if self.out(z)), None)           # 처음 빠져나온 점
            return i if i is not None and d == -self.sp.entry_gear and node[4] else None
        if self.hit(self.goal):
            return None
        moves = self.search(self.goal, done, self.sp.timeout)                       # 출차
        if moves is None:
            return None
        out, prev = [], self.goal                                                   # 결과, 이전 끝점
        for pts, d in moves:
            out.append(([prev] + pts, d))
            prev = pts[-1]
        return [([self.slot.to_world(p) for p in pts[::-1]], -d) for pts, d in out[::-1]]

    def aligned(self, p):
        """Return True if slot pose p is within the parking tolerance."""
        g, sp = self.goal, self.sp                                                  # 주차 자세, 파라미터
        return abs(p[2]) <= sp.tol_yaw and abs(p[1]) <= sp.tol_y and abs(p[0] - g[0]) <= sp.tol_x

    def align(self, world_pose):
        """Return moves in world from world_pose to the parking tolerance, [] if already there, or None."""
        start = self.slot.to_slot(world_pose)                                       # 칸 좌표 자세
        if self.aligned(start):
            return []

        m = max(1, round(self.sp.align_min / self.sp.ds))                           # 최소 움직임 점 수

        def lengths(pts, d, k, p):
            n0 = round(abs(p[2]) * self.sp.radius / self.sp.ds)                     # 방향 0 이 되는 점 수
            ns = [n0] if k != 0 and d * k * p[2] < 0 else []                        # 방향 0 길이
            return [n for n in ns + [m, m * 3 // 2, 2 * m] if m <= n <= len(pts)]   # + 최소 길이 배수

        def done(pts, d, k, node):
            ns = lengths(pts, d, k, node[0]) + [len(pts), len(pts) // 2]            # 확인할 끝 점
            return next((n - 1 for n in ns if n >= m and self.aligned(pts[n - 1])), None)
        moves = self.search(start, done, self.sp.align_timeout, lengths, m)         # 정렬
        if moves is None:
            return None
        out, prev = [], start                                                       # 결과, 이전 끝점
        for pts, d in moves:
            out.append(([self.slot.to_world(p) for p in [prev] + pts], d))
            prev = pts[-1]
        return out
