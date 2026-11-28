"""src/pinky_wall_parking_maneuver/test/test_side_overshoot.py - 칸 폭 방향 선 넘침 시험

역할: slot_parallel.side_overshoot 가 차체가 칸의 차선 쪽 / 연석 쪽 선을 넘은 깊이를 맞게 내는지 확인.
      칸 가운데 = 0, 차선 쪽으로 40 mm = 35 mm (40 + 반폭 55 - 칸 반폭 60), 연석 쪽으로 10 mm = 5 mm, 앞뒤로 밀림 = 0.
실행: cd ~/pinky_wall_parking/src/pinky_wall_parking_maneuver && python3 -m pytest -q test/test_side_overshoot.py
"""

# --- 이 패키지 ---------------------------------------------------------------------
from pinky_wall_parking_maneuver.obstacles import ObsParams                         # 차체 치수
from pinky_wall_parking_maneuver.slot_parallel import side_overshoot                # 시험 대상
from pinky_wall_parking_maneuver.slot_parallel import Slot                          # 칸
from pinky_wall_parking_maneuver.slot_parallel import SlotParams                    # 칸 파라미터

OP, SP = ObsParams(), SlotParams()
OFF = (OP.body_x[0] + OP.body_x[1]) / 2                                             # 차체 중심 x (base 기준)


def over(slot, dx, dy):
    return side_overshoot(slot.to_world((dx - OFF, dy, 0.0)), slot, SP, OP)         # 차체 중심을 (dx, dy) 에


def test_side_overshoot_depth():
    for slot in (Slot(0.6, -0.2, 0.3, 1), Slot(0.0, 0.0, -1.0, -1)):                # 돌린 칸, 오른쪽 칸
        assert over(slot, 0.0, 0.0) == 0.0
        assert abs(over(slot, 0.0, 0.040) - 0.035) < 1e-9                           # 차선 쪽
        assert abs(over(slot, 0.0, -0.010) - 0.005) < 1e-9                          # 연석 쪽
        assert over(slot, 0.030, 0.0) == 0.0                                        # 길이 방향은 line_intrusion 몫
