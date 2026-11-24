"""src/pinky_wall_parking_vision/pinky_wall_parking_vision/slot_features.py - 영상 속 벽 칸 특징 찾기

역할: 전방 카메라 영상 한 장에서 벽 칸의 특징 (뒤 벽 아래 경계, 뒤쪽 좌우 모서리,
      옆 벽 바닥 경계선의 소실점) 을 픽셀 단위로 찾는다. ROS 를 쓰지 않는다.
방법: 흰 바닥에는 색이 없으므로 열마다 가장 아래 색 픽셀을 찾아 벽과 바닥의 경계를 만든다.
      화면 가운데의 평평한 구간 = 뒤 벽, 그 양 끝 = 뒤쪽 모서리,
      바깥으로 내려가는 구간 = 옆 벽 바닥 경계선, 두 선이 만나는 점 = 소실점.
      카메라 옆 기울기 (roll_deg) 는 영상을 돌려 보정하고, 뒤 벽은 직선을 다시 맞추며 넓힌다.
실행: 라이브러리 (slot_vision_node, test_slot_features 에서 호출)
"""

# --- 표준 라이브러리 ---------------------------------------------------------------
from dataclasses import dataclass, field                                            # 파라미터·결과 묶음

# --- 외부 라이브러리 ---------------------------------------------------------------
import cv2                                                                          # 색 변환, 모폴로지
import numpy as np                                                                  # 배열 계산


# ==================================================================================
# 파라미터
# ==================================================================================
@dataclass
class FeatureParams:
    sat_min:      int   = 90                                                        # 색 픽셀 채도 하한
    val_min:      int   = 60                                                        # 색 픽셀 밝기 하한
    close_width:  int   = 31                                                        # 줄눈 메우는 가로 길이 (px)
    center_frac:  float = 0.1                                                       # 뒤 벽 기준 구간 (화면 폭 비율, 반쪽)
    flat_tol:     float = 4.0                                                       # 뒤 벽 평평함 허용 (px)
    gap_max:      int   = 24                                                        # 뒤 벽 구간 안 끊김 허용 (px)
    min_back:     int   = 40                                                        # 뒤 벽 최소 길이 (px)
    side_min:     int   = 15                                                        # 옆 벽 경계선 최소 점 수
    side_jump:    float = 15.0                                                      # 옆 벽 경계선 한 칸 최대 변화 (px)
    edge_margin:  int   = 3                                                         # 화면 가장자리 여유 (px)
    back_iter:    int   = 3                                                         # 뒤 벽 직선 다시 맞춤 횟수
    roll_deg:     float = 0.0                                                       # 정렬 상태 뒤 벽 기울기 (로봇별, deg)


# ==================================================================================
# 결과
# ==================================================================================
@dataclass
class SlotFeatures:
    width:       int
    height:      int
    back_found:  bool  = False                                                      # 뒤 벽 경계 찾음
    back_y:      float = 0.0                                                        # 칸 가운데에서 뒤 벽 경계 y
    back_slope:  float = 0.0                                                        # 뒤 벽 경계 기울기 (dy/dx)
    left_found:  bool  = False                                                      # 왼쪽 모서리 보임
    right_found: bool  = False                                                      # 오른쪽 모서리 보임
    left_x:      float = 0.0                                                        # 왼쪽 모서리 x
    right_x:     float = 0.0                                                        # 오른쪽 모서리 x
    center_x:    float = 0.0                                                        # 두 모서리 가운데 x
    vp_found:    bool  = False                                                      # 소실점 구함
    vp_x:        float = 0.0                                                        # 소실점 x
    vp_y:        float = 0.0                                                        # 소실점 y (확인용)
    envelope:    np.ndarray = field(default=None, repr=False)                       # 열별 경계 y (-1 = 없음)
    side_lines:  dict  = field(default_factory=dict, repr=False)                    # "L"/"R" -> (기울기, 절편)


# ==================================================================================
# 열별 가장 아래 색 픽셀
# ==================================================================================
def color_envelope(image, p):
    hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)                                   # 색 공간 변환
    mask = (hsv[..., 1] > p.sat_min) & (hsv[..., 2] > p.val_min)                    # 색 있는 픽셀
    mask = mask.astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))        # 잡음 제거
    kern = np.ones((1, p.close_width), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)                            # 세로 줄눈 메우기
    rows = np.arange(mask.shape[0])[:, None]                                        # 행 번호
    low  = (mask * rows).max(axis=0)                                                # 열별 가장 아래 색 행
    return np.where(mask.any(axis=0), low, -1)                                      # 색 없는 열 = -1


# ==================================================================================
# 뒤 벽 구간 (화면 가운데에서 좌우로 넓힘)
# ==================================================================================
def find_back(low, bottom, p):
    w    = len(low)
    cx   = w // 2
    half = max(int(w * p.center_frac), 1)
    win  = low[cx - half: cx + half]
    win  = win[(win >= 0) & (win < bottom)]                                         # 바닥 위에 있는 값만
    if len(win) < half:
        return None                                                                 # 가운데에 뒤 벽 없음
    a, b = 0.0, float(np.median(win))                                               # 처음은 수평선
    cols = np.arange(w)
    for _ in range(p.back_iter):
        ok = (low >= 0) & (np.abs(low - (a * cols + b)) <= p.flat_tol)              # 직선에 가까운 열
        xl = cx
        while xl > 0 and ok[max(xl - p.gap_max, 0): xl].any():
            xl -= 1                                                                 # 왼쪽으로 넓힘
        xr = cx
        while xr < w - 1 and ok[xr + 1: xr + 1 + p.gap_max].any():
            xr += 1                                                                 # 오른쪽으로 넓힘
        xs = cols[xl: xr + 1]
        xs = xs[ok[xs]]
        if len(xs) < p.min_back:
            return None                                                             # 뒤 벽이 너무 짧음
        a, b = np.polyfit(xs, low[xs], 1)                                           # 직선 다시 맞춤
    return a, b, int(xs[0]), int(xs[-1])


# ==================================================================================
# 옆 벽 바닥 경계선 (모서리에서 바깥으로 내려가는 점들)
# ==================================================================================
def find_side(low, bottom, start_x, start_y, step, p):
    pts  = []
    prev = start_y
    x    = start_x + step * p.edge_margin
    while 0 <= x < len(low):
        y = low[x]
        if y < 0 or y >= bottom or y < prev - 2 or y - prev > p.side_jump:
            break                                                                   # 경계선이 끊김
        pts.append((x, y))
        prev = y
        x   += step
    if len(pts) < p.side_min:
        return None
    pts = np.array(pts, dtype=float)
    return tuple(np.polyfit(pts[:, 0], pts[:, 1], 1))                               # (기울기, 절편)


# ==================================================================================
# 카메라 옆 기울기 보정 (화면 가운데 기준 회전, 특징 좌표는 회전한 영상 기준)
# ==================================================================================
def level_image(image, p):
    if not p.roll_deg:
        return image
    h, w = image.shape[:2]
    rot  = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), p.roll_deg, 1.0)             # 양수 = 반시계
    return cv2.warpAffine(image, rot, (w, h), borderValue=(255, 255, 255))          # 빈 곳 = 흰색 (색 없음)


# ==================================================================================
# 영상 한 장 -> 특징
# ==================================================================================
def extract_features(image, p=None):
    p      = p or FeatureParams()
    image  = level_image(image, p)
    h, w   = image.shape[:2]
    res    = SlotFeatures(width=w, height=h)
    low    = color_envelope(image, p)
    res.envelope = low
    bottom = h - p.edge_margin                                                      # 이 아래 = 바닥 안 보임
    back   = find_back(low, bottom, p)
    if back is None:
        return res
    a, b, xl, xr  = back
    res.back_found = True
    res.back_slope = float(a)
    res.back_y     = float(a * (xl + xr) / 2 + b)
    lines = {}
    for name, x0, step in (("L", xl, -1), ("R", xr, 1)):
        line = find_side(low, bottom, x0, a * x0 + b, step, p)
        if line is not None:
            lines[name] = line
    res.side_lines  = lines
    res.left_found  = xl > p.edge_margin and "L" in lines                           # 모서리가 화면 안
    res.right_found = xr < w - 1 - p.edge_margin and "R" in lines
    res.left_x      = float(xl)
    res.right_x     = float(xr)
    res.center_x    = (xl + xr) / 2.0
    if "L" in lines and "R" in lines:
        (m1, c1), (m2, c2) = lines["L"], lines["R"]
        if abs(m1 - m2) > 1e-3:
            res.vp_found = True
            res.vp_x     = float((c2 - c1) / (m1 - m2))                             # 두 선의 교점
            res.vp_y     = float(m1 * res.vp_x + c1)
    return res


# ==================================================================================
# 확인용 그림
# ==================================================================================
def draw_features(image, f):
    out = image.copy()
    w   = f.width
    cv2.line(out, (w // 2, 0), (w // 2, f.height - 1), (200, 200, 200), 1)          # 화면 가운데
    if f.envelope is not None:
        for x, y in enumerate(f.envelope):
            if y >= 0:
                cv2.circle(out, (x, int(y)), 1, (255, 0, 255), -1)                  # 경계 점
    if not f.back_found:
        return out
    ys = [f.back_y + f.back_slope * (x - f.center_x) for x in (f.left_x, f.right_x)]
    pl = (int(f.left_x), int(ys[0]))
    pr = (int(f.right_x), int(ys[1]))
    cv2.line(out, pl, pr, (0, 255, 0), 2)                                           # 뒤 벽 경계
    for pt, seen in ((pl, f.left_found), (pr, f.right_found)):
        cv2.circle(out, pt, 6, (0, 255, 0) if seen else (0, 0, 255), 2)             # 모서리 (빨강 = 안 보임)
    for name, (m, c) in f.side_lines.items():
        x0, x1 = (0, pl[0]) if name == "L" else (pr[0], w - 1)
        cv2.line(out, (x0, int(m * x0 + c)), (x1, int(m * x1 + c)), (255, 128, 0), 2)
    if f.vp_found:
        cv2.circle(out, (int(f.vp_x), int(f.vp_y)), 6, (0, 255, 255), 2)            # 소실점
    return out
