"""src/pinky_wall_parking_maneuver/test/test_slot_parallel.py - slot_parallel 단위 시험

역할: 170 x 120 mm 평행 칸 (이웃 = 칸 가운데) 에 실제 차량 방식 (45 deg 이하, 최대 조향 전진·후진) 으로 들어가는지,
      230 mm 칸에 영상 방식 (후진 S 자 + 짧은 전진, 기어 2 개) 으로 들어가는지, 마무리 정렬이 허용 범위로 돌아오는지 확인한다.
방법: 전진 진입 / 후진 진입 둘 다, 이웃 상자 + 연석과 충돌 없음 (obstacles.pose_collides 로 따로 검사), 곡률 한도,
      칸 방향 각도 <= 45 deg, 진입 기어, 145 mm 칸 (이웃이 라인에 붙음) = None, 칸 회전 / 오른쪽 칸에도 같은 결과,
      PathFollower 로 단륜차 모델 (dt 0.05 s) 을 닫힌 고리로 돌려 끝 오차 / 접촉 없음.
실행: cd ~/pinky_wall_parking && python3 -m pytest -q src/pinky_wall_parking_maneuver/test/test_slot_parallel.py
"""

# --- 표준 라이브러리 ---------------------------------------------------------------
import math                                                                         # 각도 계산

# --- 외부 라이브러리 ---------------------------------------------------------------
import numpy as np                                                                  # 점 배열

# --- 이 패키지 ---------------------------------------------------------------------
from pinky_wall_parking_maneuver.car_path import CarParams                          # 차량 파라미터
from pinky_wall_parking_maneuver.car_path import PathFollower                       # 경로 추종
from pinky_wall_parking_maneuver.obstacles import Obb                               # 상자
from pinky_wall_parking_maneuver.obstacles import ObsParams                         # 장애물 파라미터
from pinky_wall_parking_maneuver.obstacles import Obstacles                         # 장애물 묶음
from pinky_wall_parking_maneuver.obstacles import pose_collides                     # 자세 충돌
from pinky_wall_parking_maneuver.obstacles import robot_obb                         # 내 차체
from pinky_wall_parking_maneuver.slot_parallel import InSlot                        # 칸 안 경로
from pinky_wall_parking_maneuver.slot_parallel import plan_align                    # 마무리 정렬
from pinky_wall_parking_maneuver.slot_parallel import plan_slot_parallel            # 전체 경로
from pinky_wall_parking_maneuver.slot_parallel import Slot                          # 칸
from pinky_wall_parking_maneuver.slot_parallel import SlotParams                    # 칸 파라미터

CAR, OP = CarParams(), ObsParams()                                                  # 기본 파라미터
HL, HW  = 0.059, 0.055                                                              # 핑키 반길이, 반폭


def scene(slot, length, sp):
    """Return (neighbors centered in their slots, Obstacles with margin and curb) for a slot."""
    nbs = []                                                                        # 이웃 (world)
    for sx in (-length, length):
        x, y, a = slot.to_world((sx, 0.0, 0.0))                                     # 이웃 칸 중심
        nbs.append(Obb(x, y, a, HL, HW))
    cx, cy, _ = slot.to_world((0.0, -sp.width / 2 + sp.curb_margin - 0.5, 0.0))     # 연석 상자 (선 안쪽 여유)
    boxes = [Obb(b.cx, b.cy, b.yaw, HL + sp.margin, HW + sp.margin) for b in nbs]
    return nbs, Obstacles(boxes=boxes + [Obb(cx, cy, slot.yaw, 2.0, 0.5)])


def curb_gap(slot, sp, pose):
    """Return the distance (m) from the body's lowest corner to the curb line (slot frame, + = inside)."""
    return min(slot.to_slot((x, y, 0.0))[1] for x, y in robot_obb(pose, OP).corners()) + sp.width / 2


def check_inside(moves, slot, obs, sp):
    """Assert collision free, |k| <= 1/R, |slot yaw| <= max_yaw, alternating gears, parked at the end."""
    op = ObsParams(margin=0.0)                                                      # 여유는 상자에 포함
    for (pts, d), nxt in zip(moves, moves[1:] + [None]):
        p = np.array(pts)                                                           # 자세 배열
        assert not any(pose_collides(tuple(q), obs, op) for q in pts)
        assert all(abs(slot.to_slot(q)[2]) <= sp.max_yaw + 1e-6 for q in pts)
        assert all(curb_gap(slot, sp, q) >= sp.curb_margin - 1e-6 for q in pts)     # 연석 선 안쪽 (따로 검사)
        ds = np.hypot(*np.diff(p[:, :2], axis=0).T)                                 # 점 간격
        dyaw = np.abs(np.arctan2(np.sin(np.diff(p[:, 2])), np.cos(np.diff(p[:, 2]))))
        assert np.all(dyaw <= ds / sp.radius * 1.01 + 1e-6)                         # 제자리 회전 없음
        assert nxt is None or nxt[1] == -d or sp.style == "video"                   # multi: 기어 번갈아
    x, y, a = slot.to_slot(moves[-1][0][-1])                                        # 마지막 자세 (칸 좌표)
    assert abs(x + OP.body_x[0] / 2 + OP.body_x[1] / 2) < 1e-6 and abs(y) < 1e-6 and abs(a) < 1e-6


def test_forward_entry_170():
    sp, slot = SlotParams(entry_gear=1), Slot(0.0, 0.0, 0.0, 1)                     # 전진 진입
    nbs, obs = scene(slot, 0.170, sp)
    moves = InSlot(slot, nbs, OP, sp).park()
    assert moves is not None and len(moves) <= 18 and moves[0][1] == 1
    check_inside(moves, slot, obs, sp)


def test_reverse_entry_170():
    sp, slot = SlotParams(entry_gear=-1), Slot(0.0, 0.0, 0.0, 1)                    # 후진 진입 (실제 차와 같은 동작)
    nbs, obs = scene(slot, 0.170, sp)
    moves = InSlot(slot, nbs, OP, sp).park()
    assert moves is not None and moves[0][1] == -1
    assert slot.to_slot(moves[0][0][0])[1] > slot.to_slot(moves[0][0][-1])[1]       # 차선 -> 칸 안으로 후진
    check_inside(moves, slot, obs, sp)


def test_neighbors_at_lines_145_is_none():
    sp, slot = SlotParams(length=0.145, entry_gear=1, timeout=10.0), Slot(0.0, 0.0, 0.0, 1)
    lo, hi = -0.0725 - HL, 0.0725 + HL                                              # 라인에 붙은 이웃 중심
    nbs = [Obb(lo, 0.0, 0.0, HL, HW), Obb(hi, 0.0, 0.0, HL, HW)]                    # 이웃
    assert InSlot(slot, nbs, OP, sp).park() is None


def test_rotated_and_right_side_slot_same_moves():
    sp = SlotParams(length=0.200, entry_gear=1)                                     # 빠른 경우 (200 칸)
    counts = []                                                                     # 움직임 수
    for slot in (Slot(0.0, 0.0, 0.0, 1), Slot(0.4, -0.2, math.radians(30), 1), Slot(0.4, -0.2, math.radians(30), -1)):
        nbs, obs = scene(slot, 0.200, sp)
        moves = InSlot(slot, nbs, OP, sp).park()
        check_inside(moves, slot, obs, sp)
        counts.append(len(moves))
    assert counts[0] == counts[1] == counts[2]


def test_full_plan_from_lane():
    sp, slot = SlotParams(entry_gear=1), Slot(0.0, 0.0, 0.0, 1)                     # 전진 진입, 170 칸
    nbs, obs = scene(slot, 0.170, sp)
    gears = plan_slot_parallel((-0.45, 0.30, 0.0), slot, nbs, (-0.8, 0.8, -0.07, 0.6), CAR, OP, sp)
    assert gears is not None
    x, y, a = gears[-1][0][-1]                                                      # 마지막 점
    assert math.hypot(x + (OP.body_x[0] + OP.body_x[1]) / 2, y) < 0.002 and abs(a) < math.radians(0.5)
    assert not any(pose_collides(tuple(q), obs, ObsParams(margin=0.0)) for p, _ in gears for q in p)
    assert all(curb_gap(slot, sp, q) >= sp.curb_margin - 1e-6 for p, _ in gears for q in p)


def test_video_style_230_two_gears():
    for eg in (-1, 1):
        sp, slot = SlotParams(length=0.230, style="video", entry_gear=eg), Slot(0.0, 0.0, 0.0, 1)
        nbs, obs = scene(slot, 0.230, sp)
        moves = InSlot(slot, nbs, OP, sp).park()
        assert moves is not None and moves[0][1] == eg
        assert 1 + sum(a[1] != b[1] for a, b in zip(moves, moves[1:])) <= 2         # 기어 2 개 이하
        check_inside(moves, slot, obs, sp)


def test_video_style_170_is_none():
    sp, slot = SlotParams(style="video", entry_gear=-1, timeout=10.0), Slot(0.0, 0.0, 0.0, 1)
    assert InSlot(slot, scene(slot, 0.170, sp)[0], OP, sp).park() is None


def test_align_returns_to_tolerance():
    sp, slot = SlotParams(), Slot(0.3, -0.1, math.radians(20), 1)                   # 회전한 칸
    nbs, obs = scene(slot, 0.170, sp)
    for err in ((0.003, 0.002, -8.0), (-0.010, 0.004, 5.0), (0.0, -0.003, -3.0)):
        pose = slot.to_world((0.017 + err[0], err[1], math.radians(err[2])))        # 칸 안 어긋난 자세
        gears = plan_align(pose, slot, nbs, CAR, OP, sp)
        x, y, a = slot.to_slot(gears[-1][0][-1])                                    # 끝 (칸 좌표)
        assert abs(a) <= sp.tol_yaw + 1e-9 and abs(y) <= sp.tol_y and abs(x - 0.017) <= sp.tol_x
        near = Obstacles(boxes=obs.boxes[:2])                                       # 이웃만 (연석은 선으로 검사)
        assert not any(pose_collides(tuple(q), near, ObsParams(margin=0.0)) for p, _ in gears for q in p)
        low = min(0.0, curb_gap(slot, sp, pose)) - 1e-6                             # 정렬은 연석 선까지
        assert all(curb_gap(slot, sp, q) >= low for p, _ in gears for q in p)
    assert plan_align(slot.to_world((0.017, 0.0, 0.0)), slot, nbs, CAR, OP, sp) == []


def test_follow_ends_in_slot():
    sp, slot = SlotParams(entry_gear=1), Slot(0.0, 0.0, 0.0, 1)                     # 전진 진입 (빠른 경우)
    nbs, _ = scene(slot, 0.170, sp)
    gears = plan_slot_parallel((-0.6, 0.2, 0.0), slot, nbs, (-0.9, 0.9, -0.07, 0.5), CAR, OP, sp)
    follow, (x, y, th) = PathFollower(gears, CAR), (-0.6, 0.2, 0.0)                 # 추종기, 단륜차 자세
    bodies = Obstacles(boxes=nbs)                                                   # 이웃 차체 (여유 없음)
    for _ in range(8000):
        if follow.done:
            break
        v, w = follow.step((x, y, th))
        x, y, th = x + v * 0.05 * math.cos(th), y + v * 0.05 * math.sin(th), th + w * 0.05
        assert abs(w) <= abs(v) / CAR.turn_radius + 1e-9                            # 제자리 회전 없음
        assert not pose_collides((x, y, th), bodies, ObsParams(margin=0.0))         # 실제 접촉 없음
    assert follow.done
    fix = plan_align((x, y, th), slot, nbs, CAR, OP, sp)                            # 마무리 정렬
    for pts, _ in fix or []:
        x, y, th = pts[-1]                                                          # 정렬 끝 (이상 추종)
    assert math.hypot(x + (OP.body_x[0] + OP.body_x[1]) / 2, y) < 0.016 and abs(math.degrees(th)) <= 1.0
