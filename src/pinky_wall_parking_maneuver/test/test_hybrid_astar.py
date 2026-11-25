"""src/pinky_wall_parking_maneuver/test/test_hybrid_astar.py - hybrid_astar 단위 시험

역할: 빈 공간 = Reeds-Shepp, 벽 돌아가기, 곡률 한도 (제자리 회전 없음), 두 핑키 사이 평행 주차 자세 (정방향 / 반대 방향 출발), 막힌 목표 (None) 를 확인한다.
방법: 평행 주차 = scene_parallel_two 배치 (주차 핑키 base 0.30 / 0.75 m, 차체 중심 -17 mm, 방향 앎 = 상자), 출발 (-0.30, 0.25, 0).
실행: cd ~/pinky_wall_parking && python3 -m pytest -q src/pinky_wall_parking_maneuver/test/test_hybrid_astar.py
"""

# --- 표준 라이브러리 ---------------------------------------------------------------
import math                                                                         # 각도 계산
import time                                                                         # 계획 시간

# --- 외부 라이브러리 ---------------------------------------------------------------
import numpy as np                                                                  # 점 배열

# --- 이 패키지 ---------------------------------------------------------------------
from pinky_wall_parking_maneuver.car_path import CarParams                          # 차량 파라미터
from pinky_wall_parking_maneuver.car_path import planner                            # Reeds-Shepp
from pinky_wall_parking_maneuver.hybrid_astar import HaParams                       # 탐색 파라미터
from pinky_wall_parking_maneuver.hybrid_astar import HybridAStar                    # 계획기
from pinky_wall_parking_maneuver.obstacles import ObsParams                         # 장애물 파라미터
from pinky_wall_parking_maneuver.obstacles import Obstacles                         # 장애물 묶음
from pinky_wall_parking_maneuver.obstacles import pinky_obstacle                    # 퍼크 -> 상자

CAR, OP = CarParams(), ObsParams()                                                  # 기본 파라미터
R_PLAN  = CAR.turn_radius * CAR.plan_margin                                         # 계획 반경


def length(gears):
    """Return the total path length of gears."""
    return sum(np.hypot(*np.diff(p[:, :2], axis=0).T).sum() for p, _ in gears)


def all_free(ha, gears):
    """Return True if every path point is collision free."""
    return all(ha.free(tuple(q)) for p, _ in gears for q in p)


def end_error(gears, goal):
    """Return (position mm, yaw deg) error of the last point."""
    x, y, yaw = gears[-1][0][-1]                                                    # 마지막 점
    e_yaw = math.atan2(math.sin(yaw - goal[2]), math.cos(yaw - goal[2]))            # 방향 오차
    return math.hypot(x - goal[0], y - goal[1]) * 1000, abs(math.degrees(e_yaw))


def test_open_field_matches_reeds_shepp():
    ha = HybridAStar(Obstacles(), (-2.0, 2.0, -2.0, 2.0), CAR, OP)                  # 빈 공간
    for goal, fd in (((0.6, 0.3, math.pi / 2), 1), ((0.5, -0.4, 0.0), -1), ((-0.4, 0.2, math.pi), 1)):
        gears = ha.plan((0.0, 0.0, 0.0), goal, fd)
        pre = (goal[0] - fd * CAR.runway * math.cos(goal[2]), goal[1] - fd * CAR.runway * math.sin(goal[2]), goal[2])
        rs = planner.path((0.0, 0.0, 0.0), pre, R_PLAN, 0.0, 0.01).total_length + CAR.runway
        assert length(gears) <= rs * 1.05
        assert end_error(gears, goal)[0] < 2.0


def test_wall_detour():
    wall = np.column_stack([np.full(31, 0.4), np.linspace(-0.15, 0.15, 31)])        # 벽 점 (x 0.4)
    ha = HybridAStar(Obstacles(points=wall), (-0.3, 1.2, -0.6, 0.6), CAR, OP)       # 벽 있는 영역
    gears = ha.plan((0.0, 0.0, 0.0), (0.8, 0.0, 0.0), 1)
    assert gears is not None and all_free(ha, gears)
    assert max(abs(q[1]) for p, _ in gears for q in p) > 0.15                       # 벽을 돌아감
    assert end_error(gears, (0.8, 0.0, 0.0))[0] < 2.0


def test_curvature_limit():
    wall = np.column_stack([np.full(31, 0.4), np.linspace(-0.15, 0.15, 31)])        # 벽 점
    ha = HybridAStar(Obstacles(points=wall), (-0.3, 1.2, -0.6, 0.6), CAR, OP)
    for p, _ in ha.plan((0.0, 0.0, 0.0), (0.8, 0.0, 0.0), 1):
        ds = np.hypot(*np.diff(p[:, :2], axis=0).T)                                 # 점 간격
        dyaw = np.abs(np.arctan2(np.sin(np.diff(p[:, 2])), np.cos(np.diff(p[:, 2]))))
        ok = ds > 1e-4                                                              # 겹친 점 제외
        assert np.all(dyaw[ok] <= ds[ok] / R_PLAN * 1.05 + 1e-6)                    # |k| <= 1 / R_plan
        assert np.all(dyaw[~ok] < 1e-6)                                             # 제자리 회전 없음


def test_parallel_between_two_pinkies():
    obs = Obstacles(boxes=[pinky_obstacle((0.283, 0.0), 0.0, OP),
                           pinky_obstacle((0.733, 0.0), 0.0, OP)])                  # 주차 핑키 2 대
    ha = HybridAStar(obs, (-0.6, 1.3, -0.10, 0.6), CAR, OP)                         # 아래쪽 -0.10 = 연석
    goal = (0.525, 0.0, 0.0)                                                        # 두 핑키 사이 (차체 중심 0.508)
    t0 = time.monotonic()
    gears = ha.plan((-0.30, 0.25, 0.0), goal, -1)
    assert gears is not None and time.monotonic() - t0 < 5.0
    assert all_free(ha, gears)
    e_xy, e_yaw = end_error(gears, goal)
    assert e_xy < 2.0 and e_yaw < 0.5
    assert gears[-1][1] == -1                                                       # 마지막은 후진 runway
    t0 = time.monotonic()
    back = ha.plan((-0.30, 0.25, math.pi), goal, -1)                                # 반대 방향 출발
    assert back is not None and time.monotonic() - t0 < 5.0 and all_free(ha, back)
    assert end_error(back, goal)[0] < 2.0 and len(back) >= 2                        # 방향 전환 필요


def test_enclosed_goal_returns_none():
    s = np.linspace(-0.2, 0.2, 41)                                                  # 상자 한 변
    ring = np.vstack([np.column_stack([0.5 + s, np.full(41, -0.2)]), np.column_stack([0.5 + s, np.full(41, 0.2)]),
                      np.column_stack([np.full(41, 0.3), s]), np.column_stack([np.full(41, 0.7), s])])
    ha = HybridAStar(Obstacles(points=ring), (-0.3, 0.9, -0.4, 0.4), CAR, OP, HaParams(max_nodes=4000))
    assert ha.plan((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), 1) is None
