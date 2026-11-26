"""src/pinky_wall_parking_maneuver/test/test_ros_helpers.py - ros_helpers.scan_points 단위 시험

역할: 근거리 보정 (REAL_NEAR) 이 perception line_extraction 과 같게 들어가는지, 시뮬 실물 특성 (scan_model) 을 되돌리는지 확인한다.
방법: LaserScan 대신 같은 필드를 가진 SimpleNamespace 를 씀.
실행: cd ~/pinky_wall_parking/src/pinky_wall_parking_maneuver && python3 -m pytest -q test/test_ros_helpers.py
"""

# --- 표준 라이브러리 ---------------------------------------------------------------
import math                                                                         # 각도
from types import SimpleNamespace                                                   # 가짜 LaserScan

# --- 외부 라이브러리 ---------------------------------------------------------------
import numpy as np                                                                  # 비교
from pinky_wall_parking_perception.line_extraction import REAL_NEAR                 # 실물 근거리 보정표
from pinky_wall_parking_perception.line_extraction import scan_to_points            # 기준 구현
from pinky_wall_parking_sim.scan_model import apply                                 # 시뮬 실물 특성
from pinky_wall_parking_sim.scan_model import ScanModel                             # 특성 값

# --- 이 패키지 ---------------------------------------------------------------------
from pinky_wall_parking_maneuver.ros_helpers import compose                         # 자세 곱
from pinky_wall_parking_maneuver.ros_helpers import scan_points                     # 시험 대상


def scan(ranges, angle_min=0.0, inc=math.radians(1.0)):
    return SimpleNamespace(ranges=list(ranges), angle_min=angle_min, angle_increment=inc,
                           range_min=0.02, range_max=12.0)


def test_near_off_is_unchanged():
    r = [0.055, 0.07, 0.2, float("inf")]
    pts = scan_points(scan(r), (0.0, 0.0, 0.0))
    assert len(pts) == 3 and math.isclose(pts[0][0], 0.055)


def test_near_matches_line_extraction():
    r = np.array([0.055, 0.065, 0.072, 0.088, 0.10, 0.3])
    msg = scan(r, inc=0.0)                                                          # 모두 x 축 방향
    ours = scan_points(msg, (0.0, 0.0, 0.0), REAL_NEAR)
    ref = scan_to_points(r, 0.0, 0.0, 0.02, 12.0, near=REAL_NEAR)
    ref = ref[np.isfinite(ref[:, 0])]
    assert np.allclose(ours, ref)


def test_undoes_sim_real_c1():
    d = 0.30                                                                        # 앞 벽 x = 0.30
    a = np.radians(np.arange(-40, 41))
    m = ScanModel(sigma=0.0)                                                        # 잡음만 끔
    r, amin = apply(d / np.cos(a), float(a[0]), math.radians(1.0), m, np.random.default_rng(0))
    yaw = math.radians(m.yaw_offset_deg)                                            # 노드: tf yaw + offset
    pts = scan_points(scan(r, amin), compose((0.0, 0.0, 0.0), (0.0, 0.0, yaw)), REAL_NEAR)
    assert np.allclose(pts[:, 0], d, atol=1e-9)                                     # 보정하면 정답 벽
