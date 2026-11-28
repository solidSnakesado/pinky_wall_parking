"""src/pinky_wall_parking_maneuver/test/test_search_budget.py - 칸 안 탐색 노드 한도 시험

역할: 칸 안 탐색 (출차를 거꾸로, InSlot.park) 이 시간이 아니라 노드 수 (max_nodes) 로 멈추는지 확인한다.
      노드 수로 멈추면 PC 성능과 관계없이 같은 결과가 나온다.
      기어 수 상한 (max_gears): 170 mm 칸 + keep_lines + 선 여유 4 mm (칸 안 85 번) 은 못 들어가는 칸으로 판정.
실행: cd ~/pinky_wall_parking/src/pinky_wall_parking_maneuver && python3 -m pytest -q test/test_search_budget.py
"""

# --- 이 패키지 ---------------------------------------------------------------------
from pinky_wall_parking_maneuver.car_path import CarParams                          # 차량 파라미터
from pinky_wall_parking_maneuver.obstacles import Obb                               # 이웃 차 상자
from pinky_wall_parking_maneuver.obstacles import ObsParams                         # 차체 치수
from pinky_wall_parking_maneuver.slot_parallel import plan_slot_parallel            # 전체 경로
from pinky_wall_parking_maneuver.slot_parallel import Slot                          # 칸
from pinky_wall_parking_maneuver.slot_parallel import SlotParams                    # 칸 파라미터
from pinky_wall_parking_maneuver.slot_search import InSlot                          # 칸 안 탐색

OP = ObsParams()
HL = (OP.body_x[1] - OP.body_x[0]) / 2                                              # 차체 반길이


def park(sp):
    slot = Slot(0.0, 0.0, 0.0, 1)
    nbs = [Obb(*slot.to_world((sx, 0.0, 0.0)), HL, OP.body_w) for sx in (-sp.length, sp.length)]
    ins = InSlot(slot, nbs, OP, sp)
    return ins.park(), ins.nodes


def test_node_limit_stops_search():
    moves, nodes = park(SlotParams(length=0.230, entry_gear=1, keep_lines=True, max_nodes=1000))
    assert moves is None and nodes == 1000                                          # 6594 개가 필요한 장면을 1000 개에서 멈춤


def test_same_nodes_as_reference():
    moves, nodes = park(SlotParams(length=0.230, entry_gear=1, keep_lines=True))
    assert moves is not None and nodes == 6594 and len(moves) == 11                 # runs/plan_nodes.py 기준값 (PC 무관)


def test_too_many_gears_is_rejected():
    sp, slot, why = SlotParams(length=0.170, keep_lines=True, line_margin=0.004), Slot(0.0, 0.0, 0.0, 1), {}
    nbs = [Obb(*slot.to_world((sx, 0.0, 0.0)), HL, OP.body_w) for sx in (-sp.length, sp.length)]
    bounds = (-0.8, 0.8, -0.07, 0.6)                                                # 판 범위 (칸 기준)
    assert plan_slot_parallel((-0.4, 0.3, 0.0), slot, nbs, bounds, CarParams(), OP, sp, why=why) is None
    assert why["reason"] == "too many gears (in-slot 85)"                           # 칸 밖 계획 전에 판정
