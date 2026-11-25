"""src/pinky_wall_parking_maneuver/test/test_maneuver.py - car_path, t_parking 단위 시험

역할: 차량식 경로 추종 정확도, 차량 조건 (|w| <= |v| / R, 제자리 회전 없음), T자 상태 기계 (칸 고정, 중단) 를 확인한다.
방법: 단륜차 모델 (x' = v cos, y' = v sin, yaw' = w) 을 dt 0.05 s 로 적분해 닫힌 고리로 돌린다.
      칸 인식은 정답 입구 + 잡음 (x, y 1~2 mm, yaw 1.5 deg) 을 칸 앞 +-250 mm 에서만 10 Hz 로 준다 (시뮬 녹화 t_pass_lat500 기준).
실행: cd ~/pinky_wall_parking && python3 -m pytest -q src/pinky_wall_parking_maneuver/test/test_maneuver.py
"""

# --- 표준 라이브러리 ---------------------------------------------------------------
import math                                                                         # 각도 계산
import random                                                                       # 인식 잡음

# --- 외부 라이브러리 ---------------------------------------------------------------
import pytest                                                                       # 시험 틀

# --- 이 패키지 ---------------------------------------------------------------------
from pinky_wall_parking_maneuver.car_path import CarParams                          # 차량 파라미터
from pinky_wall_parking_maneuver.car_path import PathFollower                       # 경로 추종
from pinky_wall_parking_maneuver.car_path import plan                               # 경로 계획
from pinky_wall_parking_maneuver.car_path import wrap                               # 각도 정리
from pinky_wall_parking_maneuver.t_parking import along                             # 진행 방향 거리
from pinky_wall_parking_maneuver.t_parking import Phase                             # 상태 이름
from pinky_wall_parking_maneuver.t_parking import stage_pose                        # 정렬 자세
from pinky_wall_parking_maneuver.t_parking import TParking                          # T자 상태 기계


# ==================================================================================
# 시험 설정
# ==================================================================================
CP      = CarParams()
DT      = 0.05                                                                      # 적분 주기 (s)
SLOT    = (0.5, -0.45, math.pi / 2)                                                 # 칸 입구 (바깥 = +y), 녹화 t_pass_lat500
GOAL    = stage_pose(SLOT, 0.29)                                                    # 정렬 자세
POS_TOL = 5.0                                                                       # 도착 위치 허용 (mm)
YAW_TOL = 1.5                                                                       # 도착 방향 허용 (deg)


def run(step, pose, steps=4000):
    """닫힌 고리. step(pose, i) -> (v, w). 반환: 마지막 자세, |w| R / |v| 최댓값, 제자리 회전 여부."""
    x, y, th = pose
    worst, spin = 0.0, False
    for i in range(steps):
        out = step((x, y, th), i)
        if out is None:
            break
        v, w = out
        if abs(v) > 1e-9:
            worst = max(worst, abs(w) * CP.turn_radius / abs(v))
        elif abs(w) > 1e-9:
            spin = True
        x, y, th = x + v * math.cos(th) * DT, y + v * math.sin(th) * DT, th + w * DT
    return (x, y, th), worst, spin


def err(pose, goal):
    return (1000 * math.dist(pose[:2], goal[:2]), abs(math.degrees(wrap(pose[2] - goal[2]))))


# ==================================================================================
# 차량식 경로 추종
# ==================================================================================
@pytest.mark.parametrize("start", [(0.60, 0.0, 0.0), (0.50, 0.0, 0.0), (0.30, 0.0, 0.0),
                                   (0.60, 0.02, 0.1), (0.75, 0.0, 0.0)])
def test_follow_to_stage(start):
    f = PathFollower(plan(start, GOAL, CP), CP)
    pose, worst, spin = run(lambda p, i: None if f.done else f.step(p), start)
    pos, yaw = err(pose, GOAL)
    assert f.done and pos < POS_TOL and yaw < YAW_TOL, (pos, yaw)
    assert worst <= 1.0 + 1e-6 and not spin                                         # 차량 조건


def test_plan_gears():
    gears = plan((0.65, 0.0, 0.0), GOAL, CP)
    assert gears[-1][1] == -1                                                       # 마지막은 후진 (뒤가 칸으로)
    assert all(d in (1, -1) for _, d in gears)


# ==================================================================================
# T자 상태 기계
# ==================================================================================
def detector(seed, reach=0.25):
    rnd = random.Random(seed)

    def det(pose, i):
        if i % 2 or abs(along(pose, SLOT)) > reach:                                 # 10 Hz, 칸 앞 +- reach 에서만
            return None
        return (SLOT[0] + rnd.gauss(0, 0.002), SLOT[1] + rnd.gauss(0, 0.001),
                SLOT[2] + math.radians(rnd.gauss(0, 1.5)))
    return det


@pytest.mark.parametrize("start", [(0.0, 0.0, 0.0), (0.0, 0.02, 0.05), (0.2, -0.01, -0.05)])
def test_t_parking(start):
    t, det = TParking(), detector(1)
    pose, worst, spin = run(lambda p, i: None if t.phase in (Phase.DONE, Phase.ABORT) else t.step(p, det(p, i)), start)
    pos, yaw = err(pose, GOAL)
    assert t.phase == Phase.DONE, t.reason
    assert pos < POS_TOL and yaw < YAW_TOL, (pos, yaw)
    assert math.dist(t.slot[:2], SLOT[:2]) < 0.003                                  # 칸 고정 오차 3 mm 안
    assert worst <= 1.0 + 1e-6 and not spin


def test_t_parking_no_slot():
    t = TParking()
    run(lambda p, i: None if t.phase == Phase.ABORT else t.step(p, None), (0.0, 0.0, 0.0))
    assert t.phase == Phase.ABORT and t.reason == "slot not found"
