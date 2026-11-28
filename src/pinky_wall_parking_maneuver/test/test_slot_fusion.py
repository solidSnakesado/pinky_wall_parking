"""src/pinky_wall_parking_maneuver/test/test_slot_fusion.py - slot_fusion 단위 시험

역할: 장면 (230 x 120 칸 3 개, 가운데 빈 칸, 앞뒤 칸에 차) 의 선과 차 중심을 돌리고 옮겨 넣어 빈 칸을 찾는지 확인.
방법: line_map.lines 와 같은 (theta, d, s0, s1) 선을 직접 만든다 (장면 좌표 -> 지도: 20 deg 회전 + (0.3, -0.1) 이동).
      - 나눔 선 두 개: 빈 칸 중심 1 mm, 방향 0.1 deg, 앞뒤 출처 cam
      - 먼 쪽 나눔 선만: near = std (표준 길이), far = cam
      - 나눔 선 없음, 연석 선 없음: near / far = lidar, curb = std
      - 빈 칸에 차가 있으면 그 칸은 후보가 아님. 차선 쪽 선이 없는 곳 (첫 나눔 선 앞) 은 칸이 아님
      - 차선 쪽 선이 없으면 후보 없음
      - 로봇 옆을 지나는 긴 수직 선이 있어도 차선 쪽 선을 고름 (진행 방향과 나란한 선만)
      - 먼 쪽 나눔 선이 라이다 칸과 5 mm 다르면 카메라 (cam+lidar), 20 mm 다르면 라이다 (lidar(cam mismatch))
      - 차선 쪽 선이 6 mm 떨어진 긴 조각 + 로봇 쪽 짧은 조각으로 나뉘어도 한 선으로 묶어 빈 칸을 찾음
      - diag 이유: ok / no lane line / no raw cands / outside lane / car overlap
실행: cd ~/pinky_wall_parking && python3 -m pytest -q src/pinky_wall_parking_maneuver/test/test_slot_fusion.py
"""

# --- 표준 라이브러리 ---------------------------------------------------------------
import math                                                                         # 각도

# --- 이 패키지 ---------------------------------------------------------------------
from pinky_wall_parking_maneuver.slot_fusion import fuse                            # 시험 대상

ROT, OFF = math.radians(20), (0.3, -0.1)                                            # 장면 -> 지도
EMPTY, CARS = (0.600, -0.200), [(0.370, -0.200), (0.830, -0.200)]                   # 빈 칸 중심, 차 중심 (장면)


def to_map(p):
    c, s = math.cos(ROT), math.sin(ROT)
    return (OFF[0] + c * p[0] - s * p[1], OFF[1] + s * p[0] + c * p[1])


def seg(a, b):
    """Return the (theta, d, s0, s1) map line of scene segment a-b."""
    a, b = to_map(a), to_map(b)
    th = math.atan2(b[1] - a[1], b[0] - a[0]) % math.pi
    e, n = (math.cos(th), math.sin(th)), (-math.sin(th), math.cos(th))
    sa, sb = sorted((a[0] * e[0] + a[1] * e[1], b[0] * e[0] + b[1] * e[1]))
    return th, a[0] * n[0] + a[1] * n[1], sa, sb


LANE, CURB = seg((0.255, -0.14), (0.945, -0.14)), seg((0.255, -0.26), (0.945, -0.26))


def div(x):
    return seg((x, -0.26), (x, -0.14))


def run(lines, cars=CARS, x=0.3, diag=None):
    return fuse(lines, [to_map(c) for c in cars], to_map((x, 0.0)) + (ROT,), diag=diag)  # 로봇: 차선 위, 칸 쪽 진행


def err(f):
    x, y = to_map(EMPTY)
    return math.hypot(f.cx - x, f.cy - y), abs(math.degrees(math.remainder(f.yaw - ROT, 2 * math.pi)))


def test_two_dividers():
    f = run([LANE, CURB, div(0.485), div(0.715)])[0]
    d, a = err(f)
    assert d < 0.001 and a < 0.1 and f.side == 1
    assert abs(f.length - 0.23) < 1e-6 and abs(f.width - 0.12) < 1e-6
    assert (f.src["near"], f.src["far"], f.src["curb"]) == ("cam", "cam", "cam")


def test_far_divider_only_uses_standard_length():
    f = run([LANE, CURB, div(0.715)])[0]
    assert err(f)[0] < 0.001 and (f.src["near"], f.src["far"]) == ("std", "cam")


def test_no_divider_uses_lidar_gap():
    f = run([LANE])[0]
    assert err(f)[0] < 0.001 and (f.src["near"], f.src["far"], f.src["curb"]) == ("lidar", "lidar", "std")


def test_occupied_and_unlined_slots_rejected():
    full = run([LANE, CURB] + [div(x) for x in (0.255, 0.485, 0.715, 0.945)], CARS + [EMPTY])
    assert all(err(f)[0] > 0.1 for f in full)                                       # 차가 있는 빈 칸은 아님
    front = run([LANE, CURB, div(0.255)], [])                                       # 첫 나눔 선 앞은 선 없음
    assert front and all(math.cos(ROT) * (f.cx - OFF[0]) + math.sin(ROT) * (f.cy - OFF[1]) > 0.25 for f in front)
    assert run([div(0.485), div(0.715)]) == []                                      # 차선 쪽 선 없음


def test_lane_follows_heading():
    long_div = seg((0.715, -0.30), (0.715, 0.0))                                    # 로봇 옆을 지나는 긴 수직 선
    f = run([LANE, CURB, long_div], x=0.715)[0]
    assert err(f)[0] < 0.001


def test_camera_lidar_agreement():
    near = run([LANE, CURB, div(0.720)])[0]                                         # 5 mm 차이
    assert near.src["check"] == "cam+lidar" and abs(err(near)[0] - 0.005) < 1e-6
    far = run([LANE, CURB, div(0.735)])[0]                                          # 20 mm 차이
    assert far.src["check"] == "lidar(cam mismatch)" and err(far)[0] < 0.001


def test_split_lane_pieces_are_grouped():
    long_lane = seg((0.255, -0.143), (0.945, -0.143))                               # 지도에서 나뉜 차선 쪽 선
    short_piece = seg((0.590, -0.137), (0.748, -0.137))                             # 로봇 쪽 6 mm, 짧음 (0.158 m)
    f = run([long_lane, short_piece, CURB, div(0.715)])[0]
    assert err(f)[0] < 0.003 and f.src["check"] == "cam+lidar"


def test_diag_reasons():
    d = {}
    assert run([LANE, CURB, div(0.485), div(0.715)], diag=d) and d["reason"] == "ok"
    assert run([div(0.485), div(0.715)], diag=d) == [] and d == {"reason": "no lane line", "along": 0, "longest": 0.0}
    assert run([LANE], [], diag=d) == [] and d == {"reason": "no raw cands", "divs": 0, "cars": 0}
    short_lane = seg((0.255, -0.14), (0.450, -0.14))                                # 보인 구간이 칸 앞에서 끝남
    assert run([short_lane, div(0.715)], [], diag=d) == []
    assert d == {"reason": "outside lane", "raw": 2, "in_lane": 0}
    assert run([LANE, CURB, div(0.485), div(0.715)], CARS + [EMPTY], diag=d) == []  # 칸마다 차
    assert d["reason"] == "car overlap" and d["in_lane"] > 0
