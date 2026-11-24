"""src/pinky_wall_parking_front/pinky_wall_parking_front/front_control.py - 전방 주차 제어 계산

역할: 칸 특징 (소실점, 칸 가운데, 모서리 간격) 으로 방향·옆 오차를 구하고, 상태에 맞는 속도 명령을 만든다. ROS 미사용.
방법: 방향 오차 psi = atan((vp_x - cx) / f), 옆 오차 e = 칸 폭 * (center_x - vp_x) / 모서리 간격, 거리 Z = f * 칸 폭 / 모서리 간격.
      접근 중에는 w = -k_psi * psi - k_e * e. 특징이 사라지면 칸 축 방향을 odom yaw 로 유지하며 직진하고,
      초음파 또는 odom 이동량으로 멈춘다.
실행: 라이브러리 (front_parking_node, test_front_control 에서 호출)
"""

# --- 표준 라이브러리 ---------------------------------------------------------------
from collections import deque                                                       # 최근 값 보관
from dataclasses import dataclass                                                   # 파라미터 묶음
from enum import Enum                                                               # 상태 이름
import math                                                                         # 삼각 함수
import statistics                                                                   # 중앙값


# ==================================================================================
# 파라미터 (이득은 추정값, 시뮬에서 조정)
# ==================================================================================
@dataclass
class ControlParams:
    f_px:        float = 637.0                                                      # 초점 거리 (px)
    cx_px:       float = 320.0                                                      # 영상 가운데 x (px)
    slot_mm:     float = 130.0                                                      # 칸 폭 (mm)
    k_psi:       float = 1.0                                                        # 방향 오차 이득 (1/s)
    k_e:         float = 0.0125                                                     # 옆 오차 이득 (rad/s/mm)
    k_yaw:       float = 1.0                                                        # 눈 감고 직진 yaw 이득 (1/s)
    w_max:       float = 0.3                                                        # 회전 속도 한도 (rad/s)
    v_approach:  float = 0.02                                                       # 전진 속도 (m/s)
    median_n:    int   = 5                                                          # 중앙값 필터 장 수
    lost_sec:    float = 0.5                                                        # 접근 중 특징 끊김 한도 (s)
    blind_mm:    float = 160.0                                                      # 이 거리 안에서 끊기면 눈 감고 직진 (mm)
    stop_mm:     float = 30.0                                                       # 앞 끝 - 뒤 벽 정지 간격 (mm)
    us_stop_m:   float = 0.03                                                       # 초음파 정지 거리 (m, 실측 전 추정)
    us_jump_m:   float = 0.9                                                        # 이 값 이상이면 초음파 값 튐 (m)


class Phase(Enum):
    APPROACH = "approach"                                                           # 카메라로 맞추며 접근
    BLIND    = "blind"                                                              # 특징 없음, 칸 축 yaw 유지 직진
    STOP     = "stop"                                                               # 정상 정지
    ABORT    = "abort"                                                              # 비상 정지


# ==================================================================================
# 계산 함수
# ==================================================================================
def slot_errors(vp_x, center_x, gap_px, p):
    psi = math.atan((vp_x - p.cx_px) / p.f_px)                                      # + = 로봇이 칸 축보다 왼쪽을 향함
    e   = p.slot_mm * (center_x - vp_x) / gap_px                                    # - = 로봇이 칸 축 오른쪽 (mm)
    z   = p.f_px * p.slot_mm / gap_px                                               # 뒤 벽까지 거리 (mm)
    return psi, e, z


def clip(x, limit):
    return max(-limit, min(limit, x))


def steer(psi, e, p):
    return clip(-p.k_psi * psi - p.k_e * e, p.w_max)                                # 반시계 = +


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))                                     # -pi ~ pi


class MedianFilter:

    def __init__(self, n):
        self.buf = deque(maxlen=n)

    def __call__(self, x):
        self.buf.append(x)
        return statistics.median(self.buf)


# ==================================================================================
# 상태 기계
# ==================================================================================
class FrontParking:

    def __init__(self, p):
        self.p       = p
        self.phase   = Phase.APPROACH
        self.reason  = ""                                                           # 정지 이유 (로그용)
        self.f_vp    = MedianFilter(p.median_n)
        self.f_cx    = MedianFilter(p.median_n)
        self.f_gap   = MedianFilter(p.median_n)
        self.t0      = None                                                         # 첫 호출 시각
        self.seen    = None                                                         # 마지막으로 본 (t, x, y, yaw, psi, z)
        self.cmd     = (0.0, 0.0)                                                   # 마지막 명령 (v, w)
        self.yaw_ref = 0.0                                                          # 칸 축 yaw (rad)
        self.left_mm = 0.0                                                          # 눈 감고 갈 거리 (mm)

    def finish(self, phase, reason):
        self.phase, self.reason = phase, reason
        self.cmd = (0.0, 0.0)
        return self.cmd

    def step(self, t, feat, odom, us_m=None):
        """한 주기 계산. feat = (vp_x, center_x, gap_px) 또는 None, odom = (x, y, yaw), us_m = 초음파 (m) 또는 None."""
        if self.phase in (Phase.STOP, Phase.ABORT):
            return self.cmd
        if self.t0 is None:
            self.t0 = t
        if us_m is not None and us_m >= self.p.us_jump_m:
            return self.finish(Phase.ABORT, "ultrasonic jump")
        if us_m is not None and us_m <= self.p.us_stop_m:
            return self.finish(Phase.STOP, "ultrasonic")
        if self.phase == Phase.APPROACH:
            return self.approach(t, feat, odom)
        return self.blind(odom)

    def approach(self, t, feat, odom):
        x, y, yaw = odom
        if feat is not None:
            vp_x, cx, gap = self.f_vp(feat[0]), self.f_cx(feat[1]), self.f_gap(feat[2])
            psi, e, z = slot_errors(vp_x, cx, gap, self.p)
            self.seen = (t, x, y, yaw, psi, z)
            self.cmd  = (self.p.v_approach, steer(psi, e, self.p))
            return self.cmd
        if self.seen is not None and self.seen[5] <= self.p.blind_mm:
            _, _, _, yaw0, psi0, z0 = self.seen
            self.phase   = Phase.BLIND
            self.yaw_ref = wrap(yaw0 - psi0)                                        # 칸 축 방향
            self.left_mm = z0 - self.p.stop_mm
            return self.blind(odom)
        last = self.t0 if self.seen is None else self.seen[0]
        if t - last > self.p.lost_sec:
            return self.finish(Phase.ABORT, "features lost")
        return self.cmd                                                             # 짧은 끊김은 직전 명령 유지

    def blind(self, odom):
        x, y, yaw = odom
        x0, y0  = self.seen[1], self.seen[2]
        ux, uy  = math.cos(self.yaw_ref), math.sin(self.yaw_ref)                    # 칸 축 단위 벡터
        moved   = 1000.0 * ((x - x0) * ux + (y - y0) * uy)                          # 칸 축 방향 이동 (mm)
        if moved >= self.left_mm:
            return self.finish(Phase.STOP, "odom distance")
        w = clip(-self.p.k_yaw * wrap(yaw - self.yaw_ref), self.p.w_max)
        self.cmd = (self.p.v_approach, w)
        return self.cmd
