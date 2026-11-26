"""src/pinky_wall_parking_vision/pinky_wall_parking_vision/line_map.py - 여러 장의 칸 선을 바닥 지도에 쌓기 + 차선 쪽 선으로 자세 보정

역할: 차선을 따라 가며 찍은 영상들을 지도 좌표 (= 첫 장의 odom) 격자에 쌓아, 한 장으로는 안 보이는 변 (앞 나눔 선,
      연석 선 조각) 을 이어 칸을 찾는다. ROS 미사용.
방법: (주차 칸 지도에 흔한 증거 격자 + 지도 맞춤 자세 보정)
      1. 사진마다 line_slot 과 같은 선 마스크 (파란 선 - 이웃 차 상자) 와 보인 영역 (시야, near~far, 차에 안 가려진 바닥)
         을 위에서 본 격자로 만든다
      2. 자세 보정 (correct, 기본 끔): 사진의 차선 쪽 선 (Hough 로 고른 뒤 띠 안 점 최소제곱) 을 지도 기준 선
         (처음 ref_frames 장 평균) 과 비교해 방향·옆 위치 차이의 gain 배만큼 보정 (corr: odom -> 지도) 을 갱신.
         합성 시험: 한 장의 선 방향이 0.5~1 deg 흔들려, odom 이 맞을 때는 오히려 약 0.5 deg 오차를 만들고
         쏠림 3 deg/m 일 때는 마지막 방향 오차를 1.65 -> 0.3~0.8 deg 로 줄임 -> 실물 odom 측정 뒤 다시 판단.
         길이 방향 (차선을 따라) 은 이 선으로 보정되지 않음 (C 단계: 라이다 앞뒤 차로 보완)
      3. 보정된 자세로 두 격자를 지도 격자에 회전·이동 (warpAffine) 해서 센다 (선으로 보인 횟수, 보인 횟수)
      4. 보인 횟수 >= min_seen, 선 비율 >= min_ratio 인 칸 = 선 -> line_slot.fit_lines / find_slots (지도 좌표)
      오도메트리 오차는 움직인 거리만큼 쌓이므로 칸 근처 짧은 구간만 쌓고 탐색마다 새 지도로 시작한다
실행: 라이브러리 (runs/line_map_eval.py, test_line_map 에서 호출)
"""

# --- 표준 라이브러리 ---------------------------------------------------------------
from dataclasses import dataclass                                                   # 파라미터 묶음
import math                                                                         # 각도

# --- 외부 라이브러리 ---------------------------------------------------------------
import cv2                                                                          # 격자 변환
import numpy as np                                                                  # 배열

# --- 이 패키지 ---------------------------------------------------------------------
from pinky_wall_parking_vision.line_slot import Camera                              # 카메라 모델
from pinky_wall_parking_vision.line_slot import find_slots                          # 칸 조건
from pinky_wall_parking_vision.line_slot import fit_lines                           # 직선 검출
from pinky_wall_parking_vision.line_slot import line_mask                           # 파란 픽셀
from pinky_wall_parking_vision.line_slot import LineParams                          # 검출 파라미터
from pinky_wall_parking_vision.line_slot import occluder_mask                       # 이웃 차 영역
from pinky_wall_parking_vision.line_slot import top_view                            # 바닥 역투영


@dataclass
class MapParams:
    bounds: tuple = (-0.2, 1.6, -0.6, 0.6)                                          # 지도 x0, x1, y0, y1 (m)
    min_seen: int = 2                                                               # 칸을 본 최소 횟수
    min_ratio: float = 0.5                                                          # 선으로 보인 비율 최소
    gain: float = 0.5                                                               # 보정 갱신 비율
    max_deg: float = 3.0                                                            # 한 번 보정 방향 한도 (deg)
    max_d: float = 0.015                                                            # 한 번 보정 옆 위치 한도 (m)
    ref_frames: int = 2                                                             # 기준 선을 정하는 장 수
    correct: bool = False                                                           # 자세 보정 사용 (기본 끔, 아래 참고)
    band: float = 0.006                                                             # 최소제곱 맞춤 띠 반폭 (m)


def compose(a, b):
    """Return pose a o b (x, y, yaw)."""
    c, s = math.cos(a[2]), math.sin(a[2])
    return (a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], a[2] + b[2])


def to_map(line, pose):
    """Return a robot-frame line (theta, d) in the map frame given the robot pose."""
    th = line[0] + pose[2]
    return th, line[1] + (-math.sin(th) * pose[0] + math.cos(th) * pose[1])


def align(line, ref):
    """Return (theta, d) flipped by pi if needed so that theta is within pi/2 of ref theta."""
    th, d = line
    if abs(math.remainder(th - ref[0], 2 * math.pi)) > math.pi / 2:
        th, d = th + math.pi, -d
    return ref[0] + math.remainder(th - ref[0], 2 * math.pi), d                     # ref 근처 각도로 (평균용)


class LineMap:
    """Evidence grid of slot lines in the map frame, built from images and odometry poses."""

    def __init__(self, cam=None, lp=None, mp=None):
        self.cam, self.lp, self.mp = cam or Camera(), lp or LineParams(), mp or MapParams()
        x0, x1, y0, y1 = self.mp.bounds
        self.size = (int(round((x1 - x0) / self.lp.res)) + 1, int(round((y1 - y0) / self.lp.res)) + 1)
        self.hit = np.zeros(self.size[::-1], np.uint16)                             # 선으로 보인 횟수
        self.seen = np.zeros(self.size[::-1], np.uint16)                            # 보인 횟수
        self.corr = (0.0, 0.0, 0.0)                                                 # 지도 <- odom 보정
        self.ref, self.refs = None, []                                              # 지도 기준 차선 쪽 선

    def lane(self, top, lines, pose):
        """Return the lane-side line (map frame) seen in this image, or None (Hough guess -> least squares)."""
        cands = [align(to_map(ln[:2], pose), self.ref or (to_map(ln[:2], pose)[0], 0)) + (ln[3] - ln[2],)
                 for ln in lines]
        if self.ref:
            cands = [c for c in cands if abs(math.remainder(c[0] - self.ref[0], 2 * math.pi)) < math.radians(10)
                     and abs(c[1] - self.ref[1]) < 0.06]
        if not cands:
            return None
        th, d = max(cands, key=lambda c: c[2])[:2]
        r, q = np.nonzero(top)                                                      # 선 칸 (행 = y, 열 = x)
        c, s = math.cos(pose[2]), math.sin(pose[2])
        px, py = q * self.lp.res, r * self.lp.res - self.lp.side                    # 로봇 기준
        pts = np.column_stack([pose[0] + c * px - s * py, pose[1] + s * px + c * py])
        near = pts[np.abs(pts @ [-math.sin(th), math.cos(th)] - d) < self.mp.band]  # 그 선 띠 안 점
        if len(near) < 20:
            return th, d
        m = near.mean(axis=0)
        e = np.linalg.eigh(np.cov((near - m).T))[1][:, 1]                           # 주성분 방향
        th2 = math.atan2(e[1], e[0])
        return align((th2, -math.sin(th2) * m[0] + math.cos(th2) * m[1]), (th, d))

    def correct(self, obs, odom):
        """Update the odom -> map correction from a lane-line observation made at pose corr o odom."""
        pose = compose(self.corr, odom)
        dth = math.remainder(self.ref[0] - obs[0], 2 * math.pi)
        a = self.mp.gain * max(-math.radians(self.mp.max_deg), min(math.radians(self.mp.max_deg), dth))
        c, s = math.cos(a), math.sin(a)
        n = np.array([-math.sin(obs[0] + a), math.cos(obs[0] + a)])                 # 돌린 뒤 선 법선
        q = np.array([-math.sin(obs[0]), math.cos(obs[0])]) * obs[1] - pose[:2]     # 로봇 -> 선 위 한 점
        d_new = n @ (np.array([[c, -s], [s, c]]) @ q + pose[:2])                    # 로봇 둘레로 돌린 뒤 거리
        dd = self.mp.gain * max(-self.mp.max_d, min(self.mp.max_d, self.ref[1] - d_new))
        t = np.array(pose[:2]) - np.array([[c, -s], [s, c]]) @ pose[:2] + dd * n
        self.corr = compose((float(t[0]), float(t[1]), a), self.corr)

    def add(self, rgb, odom, boxes=None):
        """Add one image taken at odom pose (x, y, yaw); return the corrected map pose."""
        lp, cam = self.lp, self.cam
        occ = occluder_mask(boxes or [], cam, lp)
        mask = line_mask(rgb, lp)
        mask[occ > 0] = 0
        top, vis = top_view(mask, cam, lp), top_view(255 - occ, cam, lp)            # 선, 보인 영역
        lines = fit_lines(top, lp)
        obs = self.lane(top, lines, compose(self.corr, odom))
        if obs is not None and self.ref is None:
            self.refs.append(obs)
            if len(self.refs) >= self.mp.ref_frames:
                self.ref = tuple(np.mean([align(r, self.refs[0]) for r in self.refs], axis=0))
        elif obs is not None and self.mp.correct:
            self.correct(obs, odom)
        pose = compose(self.corr, odom)
        c, s = math.cos(pose[2]), math.sin(pose[2])
        t = (np.array([[c, -s], [s, c]]) @ [0.0, -lp.side] + pose[:2] - np.array(self.mp.bounds[::2])) / lp.res
        m = np.array([[c, -s, t[0]], [s, c, t[1]]], np.float32)                     # 로봇 격자 -> 지도 격자
        self.hit += cv2.warpAffine(top, m, self.size, flags=cv2.INTER_NEAREST) > 0
        self.seen += cv2.warpAffine(vis, m, self.size, flags=cv2.INTER_NEAREST) > 0
        return pose

    def evidence(self):
        """Return the uint8 mask of map cells judged as line."""
        ok = (self.seen >= self.mp.min_seen) & (self.hit >= self.mp.min_ratio * np.maximum(self.seen, 1))
        return ok.astype(np.uint8) * 255

    def lines(self):
        """Return merged map-frame lines as (theta, d, s0, s1)."""
        off = np.array([self.mp.bounds[0], self.mp.bounds[2] + self.lp.side])       # 격자 원점 차이
        out = []
        for th, d, s0, s1 in fit_lines(self.evidence(), self.lp):
            n, e = np.array([-math.sin(th), math.cos(th)]), np.array([math.cos(th), math.sin(th)])
            out.append((th, d + n @ off, s0 + e @ off, s1 + e @ off))
        return out

    def slots(self):
        """Return SlotRect candidates in the map frame."""
        return find_slots(self.lines(), self.lp)
