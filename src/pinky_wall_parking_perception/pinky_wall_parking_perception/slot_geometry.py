"""
src/pinky_wall_parking_perception/pinky_wall_parking_perception/slot_geometry.py - 직선 조각에서 벽 주차 칸 찾기 (칸 모양 맞추기)

- ROS 없이 numpy 만 사용 (line_extraction 과 같은 방식으로 노드와 분리, 따로 시험 가능)
- 옆 벽 쌍 : 거의 나란하고 (angle_tol), 사이 거리가 칸 폭 범위 (width_min ~ width_max),
             길이 방향으로 min_overlap 이상 겹치는 직선 2개 (옆 벽 안쪽 면)
- 뒤 벽    : 옆 벽과 거의 수직, 두 옆 벽 사이를 가로막는 직선 중 가장 가까운 것
- 입구 중심 = 뒤 벽 앞면에서 칸 깊이 (depth) 만큼 입구 쪽, 두 옆 벽 가운데
- 후보가 여럿이면 옆 벽 길이 합이 가장 긴 것

사용:
  slot = find_slot(segments, 0.190, 0.115, 0.200, math.radians(10), 0.04)
  center, outward, width, (side_a, side_b), back = slot     # 없으면 None
"""

# ==================== 표준 라이브러리 ======================
import math                                                                         # 각도 계산

# ==================== 외부 라이브러리 ======================
import numpy as np                                                                  # 직선 계산

# =========================================================
# 칸 모양 맞추기 (ROS 없이 계산만)
# =========================================================
def find_slot(segments, depth, width_min, width_max, angle_tol, min_overlap):
    """직선 조각 목록 -> (입구 중심 xy, 바깥 방향 단위벡터, 폭, 옆 벽 2개, 뒤 벽) 또는 None."""
    best, best_score = None, 0.0
    for i, a in enumerate(segments):
        for b in segments[i + 1:]:
            if abs(math.remainder(a.angle - b.angle, math.pi)) > angle_tol:         # 나란하지 않음
                continue
            axis    = a.direction                                                   # 칸 축 (부호는 뒤 벽으로 정함)
            lateral = np.array([-axis[1], axis[0]])                                 # 칸 폭 방향
            offset  = float(np.dot(b.center - a.center, lateral))                   # 두 직선 사이 거리 (부호 있음)
            width   = abs(offset)
            if not width_min <= width <= width_max:
                continue
            a_range = sorted([np.dot(a.start, axis), np.dot(a.end, axis)])          # 길이 방향 범위
            b_range = sorted([np.dot(b.start, axis), np.dot(b.end, axis)])
            overlap = min(a_range[1], b_range[1]) - max(a_range[0], b_range[0])
            if overlap < min_overlap:
                continue
            back = _find_back(segments, a, b, axis, lateral, offset, angle_tol, a_range, b_range)
            if back is None:
                continue
            back_seg, inward = back                                                 # inward: 입구 -> 뒤 벽 방향
            score = a.length + b.length
            if score <= best_score:
                continue
            mid_lat = np.dot(a.center, lateral) + offset / 2                        # 두 옆 벽 가운데 (폭 방향)
            face    = float(np.dot(back_seg.center, inward))                        # 뒤 벽 앞면 (칸 축 방향)
            center  = mid_lat * lateral + (face - depth) * inward                   # 입구 중심
            best, best_score = (center, -inward, width, (a, b), back_seg), score
    return best

def _find_back(segments, a, b, axis, lateral, offset, angle_tol, a_range, b_range):
    """두 옆 벽 사이를 가로막는 수직 직선 중 가장 가까운 것 -> (뒤 벽, 안쪽 방향) 또는 None."""
    lat_a   = float(np.dot(a.center, lateral))
    lat_lo, lat_hi = sorted([lat_a, lat_a + offset])                                # 칸 안쪽 폭 범위
    found   = None
    for c in segments:
        if c is a or c is b:
            continue
        if abs(abs(math.remainder(c.angle - a.angle, math.pi)) - math.pi / 2) > angle_tol:
            continue                                                                # 수직 아님
        c_lat   = sorted([np.dot(c.start, lateral), np.dot(c.end, lateral)])
        cover   = min(c_lat[1], lat_hi) - max(c_lat[0], lat_lo)                     # 칸 폭을 덮는 길이
        if cover < 0.5 * (lat_hi - lat_lo):
            continue
        c_ax    = float(np.dot(c.center, axis))
        far_end = max(a_range[1], b_range[1])                                       # 옆 벽 끝 (축 +쪽)
        near_end = min(a_range[0], b_range[0])                                      # 옆 벽 끝 (축 -쪽)
        if c_ax >= far_end - 0.02:                                                  # 뒤 벽이 축 +쪽
            inward, dist = axis, c_ax - far_end
        elif c_ax <= near_end + 0.02:                                               # 뒤 벽이 축 -쪽
            inward, dist = -axis, near_end - c_ax
        else:
            continue
        if found is None or dist < found[2]:
            found = (c, inward, dist)
    return None if found is None else found[:2]