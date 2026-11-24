"""src/pinky_wall_parking_front/test/test_front_control.py - front_control 단위 시험

역할: 오차 계산 부호, 제어 법칙 부호, 상태 전환 (정지·비상 정지) 을 확인한다.
방법: - 6단계 시뮬 값 (옆 +20 mm 출발: center_x 289, vp_x 319.3, 모서리 간격 198) 으로 e = -19.9 mm 확인
      - 칸 축 좌표에서 핀홀 카메라로 특징을 만들고 단륜차 모델로 움직여 닫힌 고리 주차를 확인
        (옆 +-20 mm, 방향 +-10 deg 출발 -> 칸 축에 맞고 뒤 벽 30 mm 근처에서 멈춤)
      - 초음파 정지, 초음파 값 튐, 특징 끊김 비상 정지
실행: cd ~/pinky_wall_parking && python3 -m pytest -q src/pinky_wall_parking_front/test/test_front_control.py
"""

# --- 표준 라이브러리 ---------------------------------------------------------------
import math                                                                         # 각도 계산

# --- 외부 라이브러리 ---------------------------------------------------------------
import pytest                                                                       # 시험 틀

# --- 이 패키지 ---------------------------------------------------------------------
from pinky_wall_parking_front.front_control import ControlParams                    # 파라미터
from pinky_wall_parking_front.front_control import FrontParking                     # 상태 기계
from pinky_wall_parking_front.front_control import MedianFilter                     # 중앙값 필터
from pinky_wall_parking_front.front_control import Phase                            # 상태 이름
from pinky_wall_parking_front.front_control import slot_errors                      # 오차 계산
from pinky_wall_parking_front.front_control import steer                            # 제어 법칙


# ==================================================================================
# 시험 설정
# ==================================================================================
P        = ControlParams(z_scale=1.0, z_offset_mm=0.0)                               # 이상적인 핀홀 카메라 (거리 보정 없음)
BACK_MM  = 420.0                                                                    # 출발 위치 렌즈 -> 뒤 벽 (mm)
SEE_MM   = 140.0                                                                    # 이보다 가까우면 특징 안 보임 (시뮬 결과)
DT       = 1.0 / 15.0                                                               # 카메라 주기 (s)
Y_TOL    = 5.0                                                                      # 도착 옆 오차 허용 (mm)
YAW_TOL  = 2.0                                                                      # 도착 방향 오차 허용 (deg)
GAP_TOL  = 5.0                                                                      # 정지 간격 허용 (mm)


def camera(x, y, th):
    """칸 축 좌표 (뒤 벽 x = BACK_MM, 칸 가운데 y = 0, 왼쪽 +) 의 로봇에서 본 특징."""
    if BACK_MM - x < SEE_MM:
        return None
    us = []
    for yc in (P.slot_mm / 2, -P.slot_mm / 2):                                      # 왼쪽, 오른쪽 모서리
        d = (BACK_MM - x) * math.cos(th) + (yc - y) * math.sin(th)                  # 깊이
        s = -(BACK_MM - x) * math.sin(th) + (yc - y) * math.cos(th)                 # 왼쪽 + 거리
        us.append(P.cx_px - P.f_px * s / d)
    vp_x = P.cx_px + P.f_px * math.tan(th)
    return vp_x, (us[0] + us[1]) / 2, us[1] - us[0]


def drive(y0, th0, steps=1500):
    """닫힌 고리 주행. 반환: (상태 기계, 마지막 x, y, th)."""
    fp, x, y, th = FrontParking(P), 0.0, y0, th0
    for i in range(steps):
        v, w = fp.step(i * DT, camera(x, y, th), (x / 1000.0, y / 1000.0, th))
        if fp.phase in (Phase.STOP, Phase.ABORT):
            break
        x  += 1000.0 * v * math.cos(th) * DT
        y  += 1000.0 * v * math.sin(th) * DT
        th += w * DT
    return fp, x, y, th


# ==================================================================================
# 오차 계산·제어 법칙 부호
# ==================================================================================
def test_errors_sim_values():
    psi, e, z = slot_errors(319.3, 289.0, 198.0, ControlParams())                   # 시뮬 보정값
    assert e == pytest.approx(-19.9, abs=0.1)                                       # 로봇이 칸 축 오른쪽
    assert z == pytest.approx(419.5, abs=1.0)                                       # 녹화 실제 거리 (front_sim_s0)


def test_steer_sign():
    assert steer(math.radians(5), 0.0, P) < 0                                       # 왼쪽을 향함 -> 오른쪽으로 돎
    assert steer(0.0, -10.0, P) > 0                                                 # 축 오른쪽 -> 왼쪽으로 돎
    assert abs(steer(1.0, -100.0, P)) <= P.w_max


def test_median_filter():
    f = MedianFilter(5)
    assert [f(v) for v in (10, 200, 12, 11, 13)][-1] == 12                          # 튀는 값 무시


# ==================================================================================
# 닫힌 고리 주차: 칸 축에 맞추고 뒤 벽 30 mm 근처에서 멈춤
# ==================================================================================
@pytest.mark.parametrize("y0, deg0", [(20.0, 0.0), (-20.0, 0.0), (0.0, 10.0), (0.0, -10.0), (15.0, -5.0)])
def test_closed_loop(y0, deg0):
    fp, x, y, th = drive(y0, math.radians(deg0))
    assert fp.phase == Phase.STOP and fp.reason == "odom distance", fp.reason
    assert abs(y) < Y_TOL, y
    assert abs(math.degrees(th)) < YAW_TOL, math.degrees(th)
    assert abs(BACK_MM - x - P.stop_mm) < GAP_TOL, x


# ==================================================================================
# 초음파 정지·값 튐, 특징 끊김
# ==================================================================================
def test_ultrasonic_stop():
    fp = FrontParking(P)
    assert fp.step(0.0, camera(0, 0, 0), (0, 0, 0), us_m=0.40)[0] > 0
    assert fp.step(0.1, camera(0, 0, 0), (0, 0, 0), us_m=-0.0088) == (0.0, 0.0)
    assert fp.phase == Phase.STOP


def test_ultrasonic_jump():
    fp = FrontParking(P)
    fp.step(0.0, camera(0, 0, 0), (0, 0, 0), us_m=0.97)
    assert fp.phase == Phase.ABORT


def test_lost_far():
    fp = FrontParking(P)
    fp.step(0.0, camera(0, 0, 0), (0, 0, 0))
    fp.step(0.3, None, (0, 0, 0))
    assert fp.phase == Phase.APPROACH                                               # 짧은 끊김은 계속
    fp.step(0.6, None, (0, 0, 0))
    assert fp.phase == Phase.ABORT and fp.reason == "features lost"
