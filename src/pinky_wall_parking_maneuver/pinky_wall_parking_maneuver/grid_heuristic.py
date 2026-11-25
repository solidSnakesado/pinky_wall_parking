"""src/pinky_wall_parking_maneuver/pinky_wall_parking_maneuver/grid_heuristic.py - 장애물을 반영한 2D 거리 (Hybrid A* 휴리스틱)

역할: 목표까지 차체 중심이 갈 수 있는 최단 거리 (방향 무시) 를 격자에서 미리 구한다. ROS 미사용.
방법: Hybrid A* (Dolgov 2008) 의 두 번째 휴리스틱. 격자 20 mm, 8 방향 다익스트라.
      막힌 칸 = 칸 중심에서 clear - 0.71 res 안에 장애물이 있는 칸 (clear = 차체 반폭, 0.71 res = 칸 안 최대 오차).
      차체는 중심 반지름 반폭 원을 늘 덮으므로 막힌 칸에 차체 중심이 오면 반드시 충돌한다 -> 갈 수 있는 자세를 막지 않는다.
      못 가는 칸 = inf.
실행: 라이브러리 (hybrid_astar 에서 호출).
"""

# --- 표준 라이브러리 ---------------------------------------------------------------
import heapq                                                                        # 우선순위 큐
import math                                                                         # 제곱근

# --- 외부 라이브러리 ---------------------------------------------------------------
import numpy as np                                                                  # 격자 배열


class GridHeuristic:
    """Obstacle-aware 2D distance to a goal point on a grid over bounds."""

    def __init__(self, obs, bounds, goal_xy, clear, res=0.02):
        self.x0, self.y0, self.res = bounds[0], bounds[2], res                      # 격자 원점, 칸 크기
        self.nx = int(math.ceil((bounds[1] - bounds[0]) / res)) + 1                 # x 칸 수
        self.ny = int(math.ceil((bounds[3] - bounds[2]) / res)) + 1                 # y 칸 수
        gx, gy = np.meshgrid(self.x0 + res * np.arange(self.nx), self.y0 + res * np.arange(self.ny), indexing="ij")
        blocked = np.zeros((self.nx, self.ny), dtype=bool)                          # 막힌 칸
        clear  -= res * math.sqrt(0.5)                                              # 칸 안 최대 오차만큼 줄임
        for b in obs.boxes:
            u, v = b.axes()                                                         # 상자 축
            lx, ly = (gx - b.cx) * u[0] + (gy - b.cy) * u[1], (gx - b.cx) * v[0] + (gy - b.cy) * v[1]
            blocked |= np.hypot(np.maximum(np.abs(lx) - b.hl, 0), np.maximum(np.abs(ly) - b.hw, 0)) <= clear
        for c, r in obs.circles:
            blocked |= np.hypot(gx - c[0], gy - c[1]) <= r + clear
        for q in obs.points:
            blocked |= np.hypot(gx - q[0], gy - q[1]) <= clear
        self.dist = self.dijkstra(blocked, self.cell(goal_xy))                      # 목표까지 거리

    def cell(self, xy):
        """Return the grid index of a point (clamped)."""
        i = int(round((xy[0] - self.x0) / self.res))                                # x 칸
        j = int(round((xy[1] - self.y0) / self.res))                                # y 칸
        return min(max(i, 0), self.nx - 1), min(max(j, 0), self.ny - 1)

    def dijkstra(self, blocked, start):
        """Return 8-connected distances (m) from start over free cells."""
        dist = np.full((self.nx, self.ny), np.inf)                                  # 거리
        dist[start] = 0.0
        heap = [(0.0, start)]                                                       # 큐
        steps = [(di, dj, self.res * math.hypot(di, dj)) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj]
        while heap:
            d, (i, j) = heapq.heappop(heap)
            if d > dist[i, j]:
                continue
            for di, dj, c in steps:
                a, b = i + di, j + dj                                               # 이웃 칸
                if 0 <= a < self.nx and 0 <= b < self.ny and not blocked[a, b] and d + c < dist[a, b]:
                    dist[a, b] = d + c
                    heapq.heappush(heap, (d + c, (a, b)))
        return dist

    def __call__(self, xy):
        """Return the distance (m) from a point to the goal (inf if unreachable)."""
        return float(self.dist[self.cell(xy)])
