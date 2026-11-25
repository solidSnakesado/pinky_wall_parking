"""src/pinky_wall_parking_maneuver/pinky_wall_parking_maneuver/t_parking.py - 차선에서 칸 앞 정렬 (후면 T자 / 전면)

역할: 차선을 따라 전진하며 옆의 칸을 라이다로 찾고, 지나친 뒤 차량식 경로 (car_path) 로 칸 앞 정렬 자세까지 간다. ROS 미사용.
방법: SEARCH - 출발 방향 (차선) 을 유지하며 전진. 칸 입구 인식값 (odom, 바깥 방향 yaw) 중 로봇이 칸 앞 +-lock_window 안일 때 것만 모음.
      LOCK   - 로봇이 칸 중심을 lock_after 지나면 모은 값의 중앙값으로 칸 고정, 정렬 자세 = 입구에서 바깥으로 stage_dist,
               yaw = 바깥 방향 (후면: 뒤가 칸을 향함, 후진으로 도착) 또는 안쪽 방향 (front: 앞이 칸을 향함, 전진으로 도착).
               현재 자세에서 Reeds-Shepp 경로 계획.
      TRACK  - 기어별 추종. 끝나면 DONE (노드가 후면은 Docking, 전면은 전방 주차로 넘김). 칸을 못 찾고 search_max 를 가면 ABORT.
실행: 라이브러리 (t_parking_node, 단위 시험에서 호출)
"""

# --- 표준 라이브러리 ---------------------------------------------------------------
from dataclasses import dataclass                                                   # 파라미터 묶음
from enum import Enum                                                               # 상태 이름
import math                                                                         # 삼각 함수

# --- 외부 라이브러리 ---------------------------------------------------------------
import numpy as np                                                                  # 중앙값

# --- 이 패키지 ---------------------------------------------------------------------
from pinky_wall_parking_maneuver.car_path import CarParams                          # 차량 파라미터
from pinky_wall_parking_maneuver.car_path import PathFollower                       # 경로 추종
from pinky_wall_parking_maneuver.car_path import plan                               # 경로 계획
from pinky_wall_parking_maneuver.car_path import wrap                               # 각도 정리


# ==================================================================================
# 파라미터
# ==================================================================================
@dataclass
class TParams:
    stage_dist:  float = 0.29                                                       # 정렬 자세: 입구 -> 로봇 중심 (m)
    lock_window: float = 0.10                                                       # 칸 앞 +- 이 거리 안 인식값만 씀 (m)
    lock_after:  float = 0.10                                                       # 칸 중심을 이만큼 지나면 고정 (m)
    min_count:   int   = 5                                                          # 고정에 필요한 인식 수
    search_max:  float = 1.50                                                       # 칸을 못 찾고 이만큼 가면 중단 (m)
    k_lane:      float = 2.0                                                        # 차선 방향 유지 이득 (1/s)
    front:       bool  = False                                                      # True: 전면 (앞이 칸, 전진으로 도착)


class Phase(Enum):
    SEARCH = "search"                                                               # 차선 따라 전진하며 칸 찾기
    TRACK  = "track"                                                                # 정렬 자세까지 경로 추종
    DONE   = "done"                                                                 # 정렬 자세 도착
    ABORT  = "abort"                                                                # 중단


def along(pose, point):
    """로봇 진행 방향으로 본 점까지 거리 (앞 +)."""
    x, y, yaw = pose
    return math.cos(yaw) * (point[0] - x) + math.sin(yaw) * (point[1] - y)


def lock_slot(dets):
    """인식값 (x, y, yaw) 들의 중앙값. yaw 는 첫 값 기준으로 펴서 중앙값."""
    d   = np.array(dets, dtype=float)
    ref = d[0, 2]
    yaw = ref + np.median([wrap(a - ref) for a in d[:, 2]])
    return float(np.median(d[:, 0])), float(np.median(d[:, 1])), wrap(float(yaw))


def stage_pose(slot, dist, front=False):
    """칸 입구 (바깥 방향 yaw) 에서 바깥으로 dist 간 자세. 후면: 앞 = 바깥, 전면: 앞 = 칸."""
    x, y, yaw = slot
    return x + dist * math.cos(yaw), y + dist * math.sin(yaw), wrap(yaw + math.pi) if front else yaw


# ==================================================================================
# 상태 기계
# ==================================================================================
class TParking:

    def __init__(self, cp=None, tp=None):
        self.cp     = cp or CarParams()
        self.tp     = tp or TParams()
        self.phase  = Phase.SEARCH
        self.reason = ""                                                            # 끝난 이유 (로그용)
        self.start  = None                                                          # 출발 자세 (차선 방향)
        self.dets   = []                                                            # 칸 앞에서 모은 인식값
        self.slot   = None                                                          # 고정한 칸 입구
        self.goal   = None                                                          # 정렬 자세
        self.follow = None                                                          # 경로 추종

    def step(self, pose, det=None):
        """로봇 pose = (x, y, yaw), det = 칸 입구 (x, y, 바깥 yaw) 또는 None (모두 odom). 반환: (v, w)."""
        if self.start is None:
            self.start = pose
        if self.phase == Phase.SEARCH:
            return self.search(pose, det)
        if self.phase == Phase.TRACK:
            v, w = self.follow.step(pose)
            if self.follow.done:
                self.phase, self.reason = Phase.DONE, "at staging pose"
            return v, w
        return 0.0, 0.0

    def search(self, pose, det):
        if det is not None and abs(along(pose, det)) <= self.tp.lock_window:
            self.dets.append(det)
        if len(self.dets) >= self.tp.min_count and along(pose, lock_slot(self.dets)) <= -self.tp.lock_after:
            self.slot   = lock_slot(self.dets)
            self.goal   = stage_pose(self.slot, self.tp.stage_dist, self.tp.front)
            final       = 1 if self.tp.front else -1                                # 도착할 때 진행 방향
            self.follow = PathFollower(plan(pose, self.goal, self.cp, final_dir=final), self.cp)
            self.phase  = Phase.TRACK
            return self.follow.step(pose)
        if math.dist(pose[:2], self.start[:2]) > self.tp.search_max:
            self.phase, self.reason = Phase.ABORT, "slot not found"
            return 0.0, 0.0
        v = self.cp.v_max
        w = -self.tp.k_lane * wrap(pose[2] - self.start[2])
        w = max(-v / self.cp.turn_radius, min(v / self.cp.turn_radius, w))          # 차량 조건
        return v, w
