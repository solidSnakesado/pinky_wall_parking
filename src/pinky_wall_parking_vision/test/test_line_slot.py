"""src/pinky_wall_parking_vision/test/test_line_slot.py - line_slot 단위 시험

역할: 카메라 모델 (픽셀 <-> 바닥) 과 칸 선 검출을 합성 영상으로 확인한다.
방법: 230 x 120 mm 칸 3 개의 파란 선 (폭 5 mm) 을 같은 카메라 모델로 3 x 3 초표본 그림 -> 흰 판 영상을 만든다.
      - 바닥 점 -> 픽셀 -> 바닥 점 왕복이 같음, 바퀴 축 앞 0.15 m 바닥이 화면 아래쪽에 보임
      - 차선을 따라 똑바로 (칸 옆 200 mm): 차선 쪽 선과 연석 선을 거리 3 mm, 각도 1 deg 안으로 찾음
      - 칸 쪽으로 15 deg 돌아봄: 칸 사각형 (중심 8 mm, 방향 1.5 deg, 길이·폭 10 mm 안) 을 찾음
실행: cd ~/pinky_wall_parking && python3 -m pytest -q src/pinky_wall_parking_vision/test/test_line_slot.py
"""

# --- 표준 라이브러리 ---------------------------------------------------------------
import math                                                                         # 각도

# --- 외부 라이브러리 ---------------------------------------------------------------
import numpy as np                                                                  # 배열

# --- 이 패키지 ---------------------------------------------------------------------
from pinky_wall_parking_vision.line_slot import Camera                              # 카메라 모델
from pinky_wall_parking_vision.line_slot import detect                              # 시험 대상

CAM   = Camera()                                                                    # 시뮬 기본 카메라
SS    = 3                                                                           # 초표본 배수
LINES = [((0.255, -0.26), (0.945, -0.26)), ((0.255, -0.14), (0.945, -0.14))] + \
        [((x, -0.26), (x, -0.14)) for x in (0.255, 0.485, 0.715, 0.945)]            # 장면 선 (world, m)
BLUE, WHITE = np.array([26.0, 77.0, 204.0]), 235.0                                  # 선, 판 색


def render(pose, width=0.005):
    """Return a synthetic RGB image of the slot lines seen from robot pose (x, y, yaw) in world."""
    v, u = np.mgrid[0:480 * SS, 0:640 * SS]
    g, ok = CAM.ground((u.ravel() + 0.5) / SS - 0.5, (v.ravel() + 0.5) / SS - 0.5)  # 부분 픽셀 -> 바닥
    c, s = math.cos(pose[2]), math.sin(pose[2])
    wx, wy = pose[0] + c * g[:, 0] - s * g[:, 1], pose[1] + s * g[:, 0] + c * g[:, 1]
    hit = np.zeros(len(g), bool)                                                    # 선 위 부분 픽셀
    for (ax, ay), (bx, by) in LINES:
        n = math.hypot(bx - ax, by - ay)
        ex, ey = (bx - ax) / n, (by - ay) / n
        t, d = (wx - ax) * ex + (wy - ay) * ey, -(wx - ax) * ey + (wy - ay) * ex
        hit |= (t >= -width / 2) & (t <= n + width / 2) & (np.abs(d) <= width / 2)
    rgb = np.where((hit & ok)[:, None], BLUE, WHITE).reshape(480, SS, 640, SS, 3).mean(axis=(1, 3))
    return rgb.astype(np.uint8)


def to_world(pose, x, y):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return pose[0] + c * x - s * y, pose[1] + s * x + c * y


def test_camera_round_trip():
    pts = np.array([[0.15, 0.0], [0.3, -0.1], [0.6, 0.2]])
    px, ok = CAM.pixel(pts)
    back, ok2 = CAM.ground(px[:, 0], px[:, 1])
    assert ok.all() and ok2.all() and np.allclose(back, pts, atol=1e-9)
    assert 380 < CAM.pixel(np.array([[0.15, 0.0]]))[0][0, 1] < 480                  # 0.15 m 앞 = 화면 아래쪽


def test_straight_finds_lane_and_curb():
    pose = (0.1, 0.0, 0.0)                                                          # 차선을 따라, 칸 옆 200 mm
    _, lines, _ = detect(render(pose))
    found = [(math.degrees(th), d if th < math.pi / 2 else -d) for th, d, _, _ in lines]
    for want in (-0.14, -0.26):
        best = min(found, key=lambda f: abs(f[1] - want))                           # 가장 가까운 선
        assert abs(best[1] - want) < 0.003 and min(best[0], 180 - best[0]) < 1.0


def test_turned_finds_slot():
    pose = (0.15, 0.0, math.radians(-15))                                           # 칸 쪽으로 돌아봄
    slots, _, _ = detect(render(pose))
    best = min(slots, key=lambda r: math.hypot(*np.subtract(to_world(pose, r.cx, r.cy), (0.6, -0.2))))
    x, y = to_world(pose, best.cx, best.cy)
    assert math.hypot(x - 0.6, y + 0.2) < 0.008                                     # 칸 중심
    assert abs(math.degrees(best.yaw + pose[2])) < 1.5                              # 칸 방향 (world 0)
    assert abs(best.length - 0.23) < 0.01 and abs(best.width - 0.12) < 0.01
