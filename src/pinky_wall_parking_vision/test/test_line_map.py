"""src/pinky_wall_parking_vision/test/test_line_map.py - line_map 단위 시험

역할: 차선을 따라 50 mm 마다 찍은 합성 영상 (test_line_slot.render) 을 쌓아 칸을 찾고, odom 오차를 자세 보정이 줄이는지 확인.
방법: 로봇이 x 0 ~ 0.55 m 를 똑바로 가며 12 장 (칸 옆 200 mm). odom 은 그대로 또는 방향 쏠림 3 deg/m 를 넣어 만든다.
      - 오차 없음: 차선 쪽 선·연석 선 3 mm, 빈 칸 (0.600, -0.200) 중심 5 mm·방향 0.5 deg·길이·폭 10 mm 안
      - 쏠림 +-3 deg/m: 보정 끔 (기본) 이면 마지막 자세 방향 오차 1.6 deg 넘음, 보정 켜면 그보다 작음
실행: cd ~/pinky_wall_parking && python3 -m pytest -q src/pinky_wall_parking_vision/test/test_line_map.py
"""

# --- 표준 라이브러리 ---------------------------------------------------------------
from functools import lru_cache                                                     # 같은 위치 영상 재사용
import math                                                                         # 각도

# --- 외부 라이브러리 ---------------------------------------------------------------
import numpy as np                                                                  # 배열
from test_line_slot import render                                                   # 합성 영상 (같은 폴더)

# --- 이 패키지 ---------------------------------------------------------------------
from pinky_wall_parking_vision.line_map import LineMap                              # 시험 대상
from pinky_wall_parking_vision.line_map import MapParams                            # 보정 켜고 끄기

XS = np.arange(0.0, 0.56, 0.05)                                                     # 찍는 위치 (m)


@lru_cache(maxsize=None)
def shot(x):
    return render((x, 0.0, 0.0))


def build(drift_deg_per_m=0.0, correct=False):
    """Return (LineMap, last map pose) built along XS with odometry yaw drift."""
    lm, th, ox, oy, prev = LineMap(mp=MapParams(correct=correct)), 0.0, 0.0, 0.0, 0.0
    for x in XS:
        ds, prev = x - prev, x
        th += math.radians(drift_deg_per_m) * ds                                    # odom 방향 쏠림
        ox, oy = ox + ds * math.cos(th), oy + ds * math.sin(th)
        pose = lm.add(shot(float(x)), (ox, oy, th))
    return lm, pose


def empty_slot(lm):
    return min(lm.slots(), key=lambda r: math.hypot(r.cx - 0.6, r.cy + 0.2))


def test_no_drift_finds_lines_and_slot():
    lm, _ = build()
    par = [(d if th < math.pi / 2 else -d) for th, d, _, _ in lm.lines() if min(th, math.pi - th) < 0.17]
    for want in (-0.14, -0.26):
        assert min(abs(d - want) for d in par) < 0.003
    r = empty_slot(lm)
    assert math.hypot(r.cx - 0.6, r.cy + 0.2) < 0.005 and abs(math.degrees(r.yaw)) < 0.5
    assert abs(r.length - 0.23) < 0.01 and abs(r.width - 0.12) < 0.01


def test_correction_reduces_heading_drift():
    for drift in (3.0, -3.0):
        _, raw = build(drift)                                                       # 보정 끔 (기본)
        _, fix = build(drift, correct=True)
        assert abs(math.degrees(raw[2])) > 1.6                                      # 0.55 m x 3 deg/m
        assert abs(math.degrees(fix[2])) < abs(math.degrees(raw[2]))
