"""src/pinky_wall_parking_vision/test/test_line_slot.py - line_slot 단위 시험

역할: 카메라 모델 (픽셀 <-> 바닥) 과 칸 선 검출을 합성 영상으로 확인한다.
방법: 230 x 120 mm 칸 3 개의 파란 선 (폭 5 mm) 을 같은 카메라 모델로 3 x 3 초표본 그림 -> 흰 판 영상을 만든다.
      - 바닥 점 -> 픽셀 -> 바닥 점 왕복이 같음, 바퀴 축 앞 0.15 m 바닥이 화면 아래쪽에 보임
      - 차선을 따라 똑바로 (칸 옆 200 mm): 차선 쪽 선과 연석 선을 거리 3 mm, 각도 1 deg 안으로 찾음
      - 칸 쪽으로 15 deg 돌아봄: 칸 사각형 (중심 8 mm, 방향 1.5 deg, 길이·폭 10 mm 안) 을 찾음
      - 세워 둔 차의 파란 바퀴 (차선 쪽 옆면, 높이 56 mm) 를 그리면 비스듬한 가짜 선이 생기고,
        그 차 상자 (boxes) 를 주면 가짜 선이 사라지고 차선 쪽 선은 그대로 찾음
      - 세워 둔 차 옆을 지날 때 상자를 줘도 차 옆 5 mm 의 차선 쪽 선은 지워지지 않음 (바닥면 여유 0)
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
WHEEL = (0.34, 0.40, -0.145, 0.056)                                                 # 바퀴 x0, x1, 옆면 y, 높이 (world)
CAR   = (0.370, -0.200, 0.0, 0.059, 0.055)                                          # 그 차 상자 (world, 차체 중심)


def render(pose, width=0.005, wheel=False):
    """Return a synthetic RGB image of the slot lines (and a blue wheel, yaw 0 poses) seen from robot pose."""
    v, u = np.mgrid[0:480 * SS, 0:640 * SS]
    uu, vv = (u.ravel() + 0.5) / SS - 0.5, (v.ravel() + 0.5) / SS - 0.5
    g, ok = CAM.ground(uu, vv)                                                      # 부분 픽셀 -> 바닥
    c, s = math.cos(pose[2]), math.sin(pose[2])
    wx, wy = pose[0] + c * g[:, 0] - s * g[:, 1], pose[1] + s * g[:, 0] + c * g[:, 1]
    hit = np.zeros(len(g), bool)                                                    # 선 위 부분 픽셀
    for (ax, ay), (bx, by) in LINES:
        n = math.hypot(bx - ax, by - ay)
        ex, ey = (bx - ax) / n, (by - ay) / n
        t, d = (wx - ax) * ex + (wy - ay) * ey, -(wx - ax) * ey + (wy - ay) * ex
        hit |= (t >= -width / 2) & (t <= n + width / 2) & (np.abs(d) <= width / 2)
    hit &= ok
    if wheel:
        rb = CAM.rot @ np.stack([(uu + 0.5 - CAM.c[0]) / CAM.f, (vv + 0.5 - CAM.c[1]) / CAM.f, np.ones(len(uu))])
        t = (WHEEL[2] - pose[1] - CAM.pos[1]) / np.where(rb[1] < -1e-9, rb[1], -1e-9)
        x, z = pose[0] + CAM.pos[0] + t * rb[0], CAM.pos[2] + t * rb[2]
        tg = np.where(rb[2] < -1e-9, -CAM.pos[2] / np.where(rb[2] < -1e-9, rb[2], -1.0), np.inf)
        hit |= (t > 0) & (t < tg) & (x >= WHEEL[0]) & (x <= WHEEL[1]) & (z >= 0) & (z <= WHEEL[3])
    rgb = np.where(hit[:, None], BLUE, WHITE).reshape(480, SS, 640, SS, 3).mean(axis=(1, 3))
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


def test_neighbor_box_removes_wheel_streaks():
    pose = (0.0, 0.0, 0.0)                                                          # 출발 위치 (첫 사진과 같음)
    rgb = render(pose, wheel=True)
    box = (CAR[0] - pose[0], CAR[1] - pose[1], 0.0, CAR[3], CAR[4])                 # 로봇 기준 차 상자

    def oblique(lines):
        return [ln for ln in lines if min(abs(math.degrees(ln[0]) % 90), 90 - math.degrees(ln[0]) % 90) > 5]
    assert oblique(detect(rgb)[1])                                                  # 상자 없음: 가짜 선
    _, lines, _ = detect(rgb, boxes=[box])
    assert not oblique(lines)                                                       # 상자 있음: 없음
    assert min(abs((d if th < math.pi / 2 else -d) + 0.14) for th, d, _, _ in lines) < 0.003


def test_box_keeps_lane_beside_cars():
    for x in (0.35, 0.45):                                                          # 앞 차 옆, 빈 칸 옆
        boxes = [(cx - x, -0.2, 0.0, 0.059, 0.055) for cx in (0.370, 0.830)]        # 앞뒤 차 (로봇 기준)
        _, lines, _ = detect(render((x, 0.0, 0.0)), boxes=boxes)
        assert min(abs((d if th < math.pi / 2 else -d) + 0.14) for th, d, _, _ in lines) < 0.003
