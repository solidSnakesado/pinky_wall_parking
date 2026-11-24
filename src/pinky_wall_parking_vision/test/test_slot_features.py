"""src/pinky_wall_parking_vision/test/test_slot_features.py - slot_features 단위 시험

역할: 실물 2호기 영상 5장과 1호기 영상 3장으로 특징 검출 결과를 확인한다.
방법: 칸 축 위에 놓고 찍은 영상이므로 다음을 확인한다.
      - 바닥이 보이는 4장: 뒤 벽, 좌우 모서리, 소실점을 모두 찾음
      - 가까워질수록 뒤 벽 경계 y 가 커짐 (거리 단서)
      - 칸 가운데와 소실점이 화면 가운데 근처 (배치 오차 범위)
      - 바닥이 안 보이는 1장 (뒤 벽 100 mm): 아무것도 찾지 않음
      - 1호기 (카메라 옆 기울기 약 -1.4 deg): 기울기 보정 후 특징을 모두 찾고 경계가 수평에 가까움
      - 옆 벽 선이 앞 끝면 아래 평평한 구간에 끌려가지 않음 (기울기 크기)
실행: cd ~/pinky_wall_parking && python3 -m pytest -q src/pinky_wall_parking_vision/test/test_slot_features.py
"""

# --- 표준 라이브러리 ---------------------------------------------------------------
import math                                                                         # 기울기 각도
from pathlib import Path                                                            # 시험 영상 경로

# --- 외부 라이브러리 ---------------------------------------------------------------
import cv2                                                                          # 영상 읽기
import pytest                                                                       # 시험 틀

# --- 이 패키지 ---------------------------------------------------------------------
from pinky_wall_parking_vision.slot_features import extract_features                # 시험 대상
from pinky_wall_parking_vision.slot_features import FeatureParams                   # 1호기 기울기 보정


# ==================================================================================
# 시험 데이터
# ==================================================================================
DATA    = Path(__file__).parent / "data"                                            # 시험 영상 폴더
VISIBLE = ["r2_f230", "r2_f100", "r2_f000", "r2_b140"]                              # 먼 곳 -> 가까운 곳
CENTER_TOL = 25.0                                                                   # 화면 가운데 허용 (px)
R1      = ["r1_f230", "r1_f100", "r1_f000"]                                         # 1호기, 먼 곳 -> 가까운 곳
R1_ROLL = -1.4                                                                      # 1호기 정렬 상태 뒤 벽 기울기 (deg)
LEVEL_TOL = 1.0                                                                     # 보정 후 뒤 벽 기울기 허용 (deg)
SIDE_MIN  = 0.6                                                                     # 옆 벽 선 기울기 크기 하한 (실제 약 0.7~1.0)


def load(name):
    image = cv2.imread(str(DATA / f"{name}.jpg"))
    assert image is not None, f"missing test image {name}.jpg"
    return image


# ==================================================================================
# 바닥이 보이는 영상: 특징을 모두 찾음
# ==================================================================================
@pytest.mark.parametrize("name", VISIBLE)
def test_features_found(name):
    f = extract_features(load(name))
    assert f.back_found                                                             # 뒤 벽 경계
    assert f.left_found and f.right_found                                           # 두 모서리
    assert f.left_x < f.width / 2 < f.right_x                                       # 모서리가 가운데 양쪽
    assert f.vp_found                                                               # 소실점


# ==================================================================================
# 칸 가운데·소실점이 화면 가운데 근처
# ==================================================================================
@pytest.mark.parametrize("name", VISIBLE)
def test_centered(name):
    f = extract_features(load(name))
    assert abs(f.center_x - f.width / 2) < CENTER_TOL
    assert abs(f.vp_x - f.width / 2) < CENTER_TOL


# ==================================================================================
# 가까워질수록 뒤 벽 경계가 아래로, 모서리 간격이 넓어짐
# ==================================================================================
def test_distance_order():
    fs = [extract_features(load(n)) for n in VISIBLE]
    ys = [f.back_y for f in fs]
    ws = [f.right_x - f.left_x for f in fs]
    assert ys == sorted(ys), ys                                                     # 뒤 벽 경계 y 증가
    assert ws == sorted(ws), ws                                                     # 모서리 간격 증가


# ==================================================================================
# 바닥이 안 보이는 영상: 아무것도 찾지 않음
# ==================================================================================
def test_too_close():
    f = extract_features(load("r2_b100"))
    assert not f.back_found
    assert not (f.left_found or f.right_found or f.vp_found)


# ==================================================================================
# 1호기: 옆 기울기 보정 후 특징을 모두 찾고, 거리 순서가 맞음
# ==================================================================================
@pytest.mark.parametrize("name", R1)
def test_r1_level(name):
    f = extract_features(load(name), FeatureParams(roll_deg=R1_ROLL))
    assert f.back_found and f.left_found and f.right_found and f.vp_found
    assert abs(math.degrees(math.atan(f.back_slope))) < LEVEL_TOL


def test_r1_distance_order():
    fs = [extract_features(load(n), FeatureParams(roll_deg=R1_ROLL)) for n in R1]
    ys = [f.back_y for f in fs]
    ws = [f.right_x - f.left_x for f in fs]
    assert ys == sorted(ys), ys                                                     # 뒤 벽 경계 y 증가
    assert ws == sorted(ws), ws                                                     # 모서리 간격 증가


# ==================================================================================
# 옆 벽 선: 양쪽 모두 찾고, 앞 끝면 아래 평평한 구간에 끌려가지 않음
# ==================================================================================
@pytest.mark.parametrize("name, roll", [(n, 0.0) for n in VISIBLE] + [(n, R1_ROLL) for n in R1])
def test_side_slope(name, roll):
    f = extract_features(load(name), FeatureParams(roll_deg=roll))
    assert set(f.side_lines) == {"L", "R"}
    for m, _ in f.side_lines.values():
        assert abs(m) > SIDE_MIN, (name, m)
