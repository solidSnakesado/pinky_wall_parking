"""src/pinky_wall_parking_vision/pinky_wall_parking_vision/line_slot.py - 바닥 칸 선 -> 평행 주차 칸 (앞 카메라 한 장)

역할: 앞 카메라 영상 한 장에서 바닥의 파란 칸 선을 찾아 로봇 기준 (base_footprint, m) 칸 사각형 후보를 낸다. ROS 미사용.
방법: (주차선·차선 검출의 표준 구성: 색 분리 -> 바닥 역투영 (IPM) -> 직선 검출 -> 기하 조건)
      1. HSV 로 채도 높은 파란 픽셀만 남김 (흰 판, 회색 바닥, 검은 차체, 연한 파랑 바퀴 제외)
      2. 픽셀 광선을 바닥 (z = 0) 과 만나는 점으로 (Camera: 내부 값 + URDF 장착 위치, 실물은 보정 값으로 교체)
      3. 마스크를 바닥 격자 (res m/칸) 로 역투영 (warpPerspective) -> HoughLinesP -> 같은 직선 (각도, 거리) 끼리 합침
         far 0.70 m: 그보다 멀면 픽셀 1 개가 약 1 cm 깊이라 선이 뭉개짐. 나눔 선 (시선과 수직) 은 가까울 때만 보임
      4. 가장 긴 선 방향과 나란한 선 = 차선 쪽 / 연석 선 후보, 수직 선 = 나눔 선 후보
         나란한 두 선 간격 ~ width, 이웃한 나눔 선 간격 ~ length 인 사각형 = 칸 (본 변 표시)
실행: 라이브러리 (runs/line_slot_eval.py, test_line_slot 에서 호출)
"""

# --- 표준 라이브러리 ---------------------------------------------------------------
from dataclasses import dataclass                                                   # 파라미터 묶음
import math                                                                         # 각도

# --- 외부 라이브러리 ---------------------------------------------------------------
import cv2                                                                          # 색 변환, 직선 검출
import numpy as np                                                                  # 배열


@dataclass
class CamParams:
    width:  int = 640                                                               # 영상 폭 (px)
    height: int = 480                                                               # 영상 높이 (px)
    hfov:   float = 0.934                                                           # 수평 화각 (rad, 53.5 deg)
    tilt_deg: float = 8.0                                                           # 아래로 기울기 (deg)
    mount:  tuple = (0.020, 0.0, 0.0775)                                            # base_footprint -> 마운트 (m)
    link:   tuple = (0.015, 0.0, -0.0121)                                           # 마운트 -> 카메라 (마운트 좌표)


@dataclass
class LineParams:
    hue:     tuple = (95, 130)                                                      # 파랑 (OpenCV 0~180)
    sat_min: int = 70                                                               # 채도 최소 (먼 선은 흰 판과 섞여 옅어짐)
    val_min: int = 40                                                               # 밝기 최소
    near:    float = 0.10                                                           # 바닥 점 앞 최소 (m)
    far:     float = 0.70                                                           # 바닥 점 앞 최대 (m, 픽셀 1 개가 약 1 cm 깊이)
    side:    float = 0.60                                                           # 바닥 점 옆 최대 (m)
    res:     float = 0.002                                                          # 위에서 본 격자 (m/칸)
    min_len: float = 0.03                                                           # 선분 최소 길이 (m)
    merge_deg: float = 4.0                                                          # 같은 직선 각도 차 (deg)
    merge_d: float = 0.008                                                          # 같은 직선 거리 차 (m)
    width:   float = 0.120                                                          # 칸 폭 (m)
    length:  float = 0.230                                                          # 칸 길이 (m)
    tol_w:   float = 0.025                                                          # 폭 허용 (m)
    tol_l:   float = 0.035                                                          # 길이 허용 (m)


@dataclass
class SlotRect:
    cx: float                                                                       # 칸 중심 x (base, m)
    cy: float                                                                       # 칸 중심 y (base, m)
    yaw: float                                                                      # 칸 길이 방향 (rad)
    length: float                                                                   # 나눔 선 간격 (m)
    width: float                                                                    # 두 긴 선 간격 (m, 하나면 기본값)
    edges: str                                                                      # 본 변: L 차선 쪽, C 연석, D 나눔 선 2


class Camera:
    """Pinhole camera on the Pinky (optical: x right, y down, z forward) for pixel <-> ground."""

    def __init__(self, p=None):
        p = p or CamParams()
        self.f = (p.width / 2) / math.tan(p.hfov / 2)                               # 초점 거리 (px, 정사각 픽셀)
        self.c = (p.width / 2, p.height / 2)                                        # 주점 (camera_info 와 같음)
        t = math.radians(p.tilt_deg)
        ry = np.array([[math.cos(t), 0, math.sin(t)], [0, 1, 0], [-math.sin(t), 0, math.cos(t)]])
        self.pos = np.array(p.mount) + ry @ np.array(p.link)                        # 카메라 위치 (base)
        self.rot = ry @ np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)  # base <- optical

    def ground(self, u, v):
        """Return ground points (N, 2) and a validity mask for pixel centers (u, v) (index coords)."""
        rays = np.stack([(np.asarray(u) + 0.5 - self.c[0]) / self.f,
                         (np.asarray(v) + 0.5 - self.c[1]) / self.f, np.ones(np.shape(u))])
        rb = self.rot @ rays                                                        # 광선 (base)
        ok = rb[2] < -1e-6                                                          # 아래로 향하는 광선
        s = np.where(ok, -self.pos[2] / np.where(ok, rb[2], -1.0), 0.0)             # 바닥까지 배수
        return (self.pos[:2, None] + rb[:2] * s).T, ok

    def pixel(self, pts):
        """Return pixel coords (N, 2, index) and an in-front mask for ground points (N, 2)."""
        pts = np.asarray(pts, dtype=float)
        pb = np.column_stack([pts, np.zeros(len(pts))]) - self.pos                  # 카메라 -> 점 (base)
        po = pb @ self.rot                                                          # optical 좌표
        ok = po[:, 2] > 1e-6
        z = np.where(ok, po[:, 2], 1.0)
        return np.column_stack([self.f * po[:, 0] / z + self.c[0] - 0.5, self.f * po[:, 1] / z + self.c[1] - 0.5]), ok


def line_mask(rgb, lp):
    """Return a uint8 mask of saturated blue pixels in an RGB image."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return cv2.inRange(hsv, (lp.hue[0], lp.sat_min, lp.val_min), (lp.hue[1], 255, 255))


def top_view(mask, cam, lp):
    """Return the mask warped onto the ground grid (rows = y from -side, cols = x from 0, res m/cell)."""
    nx, ny = int(lp.far / lp.res) + 1, int(2 * lp.side / lp.res) + 1                # 격자 크기
    corners = np.array([[lp.near, -lp.side], [lp.near, lp.side], [lp.far, lp.side], [lp.far, -lp.side]])
    px, _ = cam.pixel(corners)                                                      # 바닥 네 점 -> 픽셀
    cells = np.column_stack([corners[:, 0], corners[:, 1] + lp.side]) / lp.res      # 바닥 네 점 -> 격자
    h = cv2.getPerspectiveTransform(px.astype(np.float32), cells.astype(np.float32))
    return cv2.warpPerspective(mask, h, (nx, ny), flags=cv2.INTER_NEAREST)          # 바닥 평면 역투영 (IPM)


def fit_lines(top, lp):
    """Return merged ground lines [(theta, d, s0, s1)] (direction theta in [0, pi), n.p = d, extent s0..s1)."""
    segs = cv2.HoughLinesP(top, 1, math.radians(0.5), 30, minLineLength=int(lp.min_len / lp.res), maxLineGap=6)
    if segs is None:
        return []
    raw = []                                                                        # (theta, d, s0, s1, 길이)
    for x0, y0, x1, y1 in segs.reshape(-1, 4) * lp.res:
        a, b = np.array([x0, y0 - lp.side]), np.array([x1, y1 - lp.side])           # 끝점 (base)
        th = math.atan2(b[1] - a[1], b[0] - a[0]) % math.pi                         # 방향 [0, pi)
        n, e = np.array([-math.sin(th), math.cos(th)]), np.array([math.cos(th), math.sin(th)])
        s = sorted((a @ e, b @ e))
        raw.append([th, (a @ n + b @ n) / 2, s[0], s[1], s[1] - s[0]])
    lines = []                                                                      # 합친 선
    for th, d, s0, s1, w in sorted(raw, key=lambda r: -r[4]):
        for ln in lines:
            dth = (th - ln[0] + math.pi / 2) % math.pi - math.pi / 2                # 각도 차 (-pi/2, pi/2]
            flip = abs(th - ln[0]) > math.pi / 2                                    # 0 / 180 deg 근처 뒤집힘
            dd, ss = (-d, (-s1, -s0)) if flip else (d, (s0, s1))
            if abs(dth) < math.radians(lp.merge_deg) and abs(dd - ln[1]) < lp.merge_d:
                tot = ln[4] + w
                ln[1] = (ln[1] * ln[4] + dd * w) / tot                              # 길이 가중 평균 거리
                ln[2], ln[3], ln[4] = min(ln[2], ss[0]), max(ln[3], ss[1]), tot
                break
        else:
            lines.append([th, d, s0, s1, w])
    return [tuple(ln[:4]) for ln in lines]


def find_slots(lines, lp):
    """Return SlotRect candidates (base frame) from merged ground lines."""
    if not lines:
        return []
    th0 = max(lines, key=lambda ln: ln[3] - ln[2])[0]                               # 긴 선 방향
    par, per = [], []                                                               # 나란한 선 (거리), 수직 선 (위치)
    for th, d, s0, s1 in lines:
        dth = (th - th0 + math.pi / 2) % math.pi - math.pi / 2
        if abs(dth) < math.radians(10):
            par.append(d if abs(th - th0) < math.pi / 2 else -d)
        elif abs(abs(dth) - math.pi / 2) < math.radians(10):
            n = np.array([-math.sin(th), math.cos(th)])                             # 수직 선 법선
            mid = n * d + np.array([math.cos(th), math.sin(th)]) * (s0 + s1) / 2    # 수직 선 가운데
            per.append(mid @ np.array([math.cos(th0), math.sin(th0)]))
    par, per = sorted(par, key=abs), sorted(per)                                    # 가까운 선부터, 앞뒤 순
    pair = next(((a, b) for i, a in enumerate(par) for b in par[i + 1:]
                 if abs(abs(a - b) - lp.width) < lp.tol_w), None)
    if pair:
        mid_d, width, edges = (pair[0] + pair[1]) / 2, abs(pair[1] - pair[0]), "LC"
    elif par:
        mid_d, width, edges = par[0] + math.copysign(lp.width / 2, par[0]), lp.width, "L"
    else:
        return []
    e, n = np.array([math.cos(th0), math.sin(th0)]), np.array([-math.sin(th0), math.cos(th0)])
    out = []
    for s0, s1 in zip(per, per[1:]):
        if abs(s1 - s0 - lp.length) < lp.tol_l:
            c = e * (s0 + s1) / 2 + n * mid_d                                       # 칸 중심
            yaw = th0 if math.cos(th0) >= 0 else th0 - math.pi                      # 로봇 앞 방향 쪽
            out.append(SlotRect(float(c[0]), float(c[1]), yaw, s1 - s0, width, edges + "D"))
    return out


def detect(rgb, cam=None, lp=None):
    """Return (SlotRect list, merged lines, top-view mask) for one RGB image."""
    cam, lp = cam or Camera(), lp or LineParams()
    top = top_view(line_mask(rgb, lp), cam, lp)
    lines = fit_lines(top, lp)
    return find_slots(lines, lp), lines, top
