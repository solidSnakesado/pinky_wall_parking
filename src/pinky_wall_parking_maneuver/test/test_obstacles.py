"""src/pinky_wall_parking_maneuver/test/test_obstacles.py - obstacles 단위 시험

역할: 퍼크 원 맞춤 (시뮬 실측 점, 합성 원호), 벽 거르기, 묶음 나누기, SAT / 원 / 점 충돌, 방향 모름 (원) 과 방향 앎 (상자) 차이를 확인한다.
방법: 시뮬 실측 = scan_shape_check (scene_parallel_two, 로봇 출발점) 의 parked_0 점 4 개, parked_1 점 2 개 (핑키 기준 mm).
실행: cd ~/pinky_wall_parking && python3 -m pytest -q src/pinky_wall_parking_maneuver/test/test_obstacles.py
"""

# --- 표준 라이브러리 ---------------------------------------------------------------
import math                                                                         # 각도 계산

# --- 외부 라이브러리 ---------------------------------------------------------------
import numpy as np                                                                  # 점 배열

# --- 이 패키지 ---------------------------------------------------------------------
from pinky_wall_parking_maneuver.obstacles import cluster                           # 묶음 나누기
from pinky_wall_parking_maneuver.obstacles import fit_puck                          # 퍼크 원 맞춤
from pinky_wall_parking_maneuver.obstacles import Obb                               # 상자
from pinky_wall_parking_maneuver.obstacles import obb_circle_overlap                # 상자-원
from pinky_wall_parking_maneuver.obstacles import obb_overlap                       # 상자-상자
from pinky_wall_parking_maneuver.obstacles import ObsParams                         # 파라미터
from pinky_wall_parking_maneuver.obstacles import Obstacles                         # 장애물 묶음
from pinky_wall_parking_maneuver.obstacles import pinky_obstacle                    # 퍼크 -> 장애물
from pinky_wall_parking_maneuver.obstacles import points_in_obb                     # 상자-점
from pinky_wall_parking_maneuver.obstacles import pose_collides                     # 자세 충돌
from pinky_wall_parking_maneuver.obstacles import PuckTrack                         # 중심 누적
from pinky_wall_parking_maneuver.obstacles import split_scan                        # 퍼크 / 나머지

P      = ObsParams()                                                                # 기본 파라미터
SENSOR = np.array([-0.017, 0.0])                                                    # 로봇 라이다 (world, 출발점)
SIM    = {(0.60, -0.25): [(-41.2, -6.4), (-41.3, 5.6), (-36.7, 15.5), (-26.3, 23.3)],
          (1.05, -0.25): [(-41.4, -5.7), (-38.6, 12.6)]}                            # 시뮬 실측 (핑키 기준 mm)


def arc(center, n, noise=0.0, seed=0):
    """Return n puck points facing the sensor, 1 deg rays apart, with noise (m)."""
    rng = np.random.default_rng(seed)                                               # 잡음
    c = np.asarray(center)                                                          # 중심
    base = math.atan2(*(SENSOR - c)[::-1])                                          # 센서 쪽 방향
    a = base + np.radians(np.linspace(-12.0 * (n - 1) / 2, 12.0 * (n - 1) / 2, n))  # 점 각도
    pts = c + P.puck_r * np.column_stack([np.cos(a), np.sin(a)])                    # 원 위 점
    return pts + rng.normal(0.0, noise, pts.shape)


def test_fit_sim_points():
    for (px, py), local in SIM.items():
        pts = np.array([[px + x / 1000, py + y / 1000] for x, y in local])          # world 점
        c, rms = fit_puck(pts, SENSOR, P)                                           # 원 맞춤
        assert np.linalg.norm(c - [px - 0.017, py]) < 0.001                         # 중심 = 차체 중심
        assert rms < 0.001


def test_fit_synthetic_arcs():
    for n in (3, 4, 6):
        for seed in range(20):
            c, _ = fit_puck(arc((0.8, 0.3), n, 0.0005, seed), SENSOR, P)            # 잡음 0.5 mm
            assert np.linalg.norm(c - [0.8, 0.3]) < 0.005


def test_two_points_need_tracking():
    t = PuckTrack(P)                                                                # 누적기
    for seed in range(P.track_n):
        t.update([fit_puck(arc((0.8, 0.3), 2, 0.0005, seed), SENSOR, P)[0]])        # 점 2 개 (먼 핑키)
    assert np.linalg.norm(t.pucks()[0] - [0.8, 0.3]) < 0.003


def test_wall_is_not_puck():
    line = np.column_stack([np.full(5, 0.5), np.linspace(0.0, 0.05, 5)])            # 50 mm 직선
    assert fit_puck(line, SENSOR, P) is None
    long = np.column_stack([np.full(20, 0.5), np.linspace(0.0, 0.3, 20)])           # 긴 벽
    assert fit_puck(long, SENSOR, P) is None


def test_cluster_and_split():
    wall = np.column_stack([np.full(20, 0.5), np.linspace(-0.6, -0.3, 20)])         # 벽
    pts = np.vstack([wall, arc((0.6, 0.2), 4)])                                     # 벽 + 퍼크
    assert len(cluster(pts, P.gap)) == 2
    centers, rest = split_scan(pts, SENSOR, P)                                      # 퍼크 / 나머지
    assert len(centers) == 1 and len(rest) == 20
    assert np.linalg.norm(centers[0] - [0.6, 0.2]) < 0.001


def test_sat_boxes():
    a = Obb(0.0, 0.0, 0.0, 0.05, 0.05)                                              # 기준 상자
    edge = 0.05 + 0.05 * math.sqrt(2)                                               # 45 도 꼭짓점 접촉 거리
    assert obb_overlap(a, Obb(edge - 0.002, 0.0, math.pi / 4, 0.05, 0.05))
    assert not obb_overlap(a, Obb(edge + 0.002, 0.0, math.pi / 4, 0.05, 0.05))
    assert obb_overlap(a, Obb(0.099, 0.0, 0.0, 0.05, 0.05))
    assert not obb_overlap(a, Obb(0.101, 0.0, 0.0, 0.05, 0.05))
    assert not obb_overlap(a, Obb(0.09, 0.09, 0.3, 0.02, 0.02))


def test_circle_and_points():
    a = Obb(0.0, 0.0, 0.5, 0.05, 0.03)                                              # 회전 상자
    u = np.array([math.cos(0.5), math.sin(0.5)])                                    # 길이 축
    assert obb_circle_overlap(a, u * 0.069, 0.02)
    assert not obb_circle_overlap(a, u * 0.071, 0.02)
    assert points_in_obb(a, [u * 0.049])
    assert not points_in_obb(a, [u * 0.051, [0.0, 0.2]])
    assert not points_in_obb(a, np.zeros((0, 2)))


def test_unknown_yaw_is_safer():
    hl, hw = 0.059, 0.055                                                           # 핑키 반치수
    pose = (0.017, hw + 0.075, 0.0)                                                 # 옆 75 mm (차체 중심 0)
    box = Obstacles(boxes=[pinky_obstacle((0.0, 0.0), 0.0, P)])                     # 방향 앎
    ring = Obstacles(circles=[pinky_obstacle((0.0, 0.0), None, P)])                 # 방향 모름
    assert not pose_collides(pose, box, P)
    assert pose_collides(pose, ring, P)
    assert abs(ring.circles[0][1] - (math.hypot(hl, hw) + P.margin)) < 1e-9
    assert pose_collides((0.017, hw + 0.06, 0.0), box, P)                           # 여유 10 mm 안


def test_points_collide_with_margin():
    obs = Obstacles(points=np.array([[0.042 + 0.008, 0.0]]))                        # 앞 끝 8 mm 앞
    assert pose_collides((0.0, 0.0, 0.0), obs, P)
    obs = Obstacles(points=np.array([[0.042 + 0.012, 0.0]]))                        # 앞 끝 12 mm 앞
    assert not pose_collides((0.0, 0.0, 0.0), obs, P)


def test_track_median():
    t = PuckTrack(P)                                                                # 누적기
    rng = np.random.default_rng(1)                                                  # 잡음
    for _ in range(10):
        t.update([np.array([0.6, -0.25]) + rng.normal(0, 0.002, 2),
                  np.array([1.05, -0.25]) + rng.normal(0, 0.002, 2)])
    t.update([np.array([0.3, 0.3])])                                                # 한 번 본 것
    got = sorted(t.pucks(), key=lambda c: c[0])                                     # 중앙값
    assert len(got) == 2
    assert np.linalg.norm(got[0] - [0.6, -0.25]) < 0.002
    assert np.linalg.norm(got[1] - [1.05, -0.25]) < 0.002
