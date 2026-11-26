"""src/pinky_wall_parking_sim/test/test_scan_model.py - scan_model 단위 시험

역할: 실물 라이다 특성 (근거리 오차, 6 cm 미만, 스치는 광선 점 빠짐, 잡음, 장착 yaw, 몸 점) 이 하나씩 맞게 들어가는지 확인한다.
방법: 정면 벽 (x = d) 을 1° 간격 광선으로 잰 이상적인 스캔을 만들어 넣음 (ROS 불필요).
실행: cd ~/pinky_wall_parking/src/pinky_wall_parking_sim && python3 -m pytest -q test/test_scan_model.py
"""

# --- 표준 라이브러리 ---------------------------------------------------------------
import math                                                                         # 각도

# --- 외부 라이브러리 ---------------------------------------------------------------
import numpy as np                                                                  # 비교

# --- 이 패키지 ---------------------------------------------------------------------
from pinky_wall_parking_sim.scan_model import apply                                 # 특성 적용
from pinky_wall_parking_sim.scan_model import near_bias                             # 근거리 오차
from pinky_wall_parking_sim.scan_model import ScanModel                             # 특성 값

INC = math.radians(1.0)                                                             # 시뮬 광선 간격


def wall_scan(d, half_deg=80):
    """Ideal scan of a wall at x = d, rays from -half_deg to +half_deg."""
    a = np.radians(np.arange(-half_deg, half_deg + 1))
    return d / np.cos(a), float(a[0])


def quiet(**kw):
    """Model with only the named effects on."""
    base = dict(grazing_deg=0.0, near_r=[], near_bias=[], sigma=0.0, near_drop=0.0, yaw_offset_deg=0.0)
    return ScanModel(**{**base, **kw})


def test_near_bias_inverts_real_table():
    m = ScanModel()
    true = np.array([0.060, 0.070, 0.080, 0.090, 0.120])
    meas = near_bias(true, m)
    corr = np.interp(meas, m.near_r, m.near_bias)                                   # 처리 쪽 보정 (REAL_NEAR)
    assert np.allclose(meas - corr, true, atol=2e-4)                                # 보정하면 정답으로 돌아옴
    assert meas[0] > true[0] + 0.009 and meas[-1] == true[-1]                       # 가까우면 멀게, 멀면 그대로


def test_near_drop():
    r, a0 = wall_scan(0.055)
    out, _ = apply(r, a0, INC, quiet(near_drop=0.060), np.random.default_rng(0))
    assert not np.isfinite(out[80])                                                 # 정면 55 mm -> 점 없음
    assert np.isfinite(out[0])                                                      # 옆 광선은 멀어서 남음


def test_grazing_drops_shallow_rays_only():
    r, a0 = wall_scan(0.30)
    out, _ = apply(r, a0, INC, quiet(grazing_deg=18.0, neighbor_gap=0.2), np.random.default_rng(0))
    ang = np.degrees(a0 + INC * np.arange(len(r)))
    lost = ~np.isfinite(out)
    assert not lost[np.abs(ang) < 70].any()                                         # 벽과 20° 이상 = 남음
    assert lost[(np.abs(ang) > 73) & (np.abs(ang) < 80)].all()                      # 벽과 18° 미만 = 빠짐


def test_noise_sigma_and_seed():
    r, a0 = wall_scan(0.50, half_deg=40)
    r = np.tile(r, 20)                                                              # 점 많이
    a, _ = apply(r, a0, INC, quiet(sigma=0.002), np.random.default_rng(1))
    b, _ = apply(r, a0, INC, quiet(sigma=0.002), np.random.default_rng(1))
    assert np.array_equal(a, b)                                                     # 같은 seed = 같은 잡음
    assert abs(np.std(a - r) - 0.002) < 0.0003


def test_yaw_offset_shifts_reported_angle():
    r, a0 = wall_scan(0.30)
    out, amin = apply(r, a0, INC, quiet(yaw_offset_deg=-2.2), np.random.default_rng(0))
    assert math.isclose(math.degrees(amin - a0), 2.2, abs_tol=1e-9)                 # 처리 쪽 -2.2° 로 되돌림
    assert np.array_equal(out, r)


def test_self_points_and_inf_kept():
    r = np.full(360, np.inf)
    out, amin = apply(r, -math.pi, INC, quiet(self_points=[0.024, -0.029]), np.random.default_rng(0))
    i = int(np.argmin(out))
    assert math.isclose(out[i], math.hypot(0.024, -0.029))
    assert abs(math.degrees(amin + INC * i - math.atan2(-0.029, 0.024))) <= 0.5     # 보고 각 기준 위치
    assert np.isfinite(out).sum() == 1
