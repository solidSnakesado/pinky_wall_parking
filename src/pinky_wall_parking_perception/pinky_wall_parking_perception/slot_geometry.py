"""
src/pinky_wall_parking_perception/pinky_wall_parking_perception/slot_geometry.py - 직선 조각에서 벽 주차 칸 찾기 (칸 모양 맞추기)

- ROS 없이 numpy 만 사용 (line_extraction 과 같은 방식으로 노드와 분리, 따로 시험 가능)
- 옆 벽 쌍 : 거의 나란하고 (angle_tol), 사이 거리가 칸 폭 범위 (width_min ~ width_max),
             길이 방향으로 min_overlap 이상 겹치는 직선 2개 (옆 벽 안쪽 면)
- 뒤 벽    : 옆 벽과 거의 수직, 두 옆 벽 사이를 가로막는 직선 중 가장 가까운 것
- 입구 중심 = 뒤 벽 앞면에서 칸 깊이 (depth) 만큼 입구 쪽, 두 옆 벽 가운데
- 후보가 여럿이면 옆 벽 길이 합이 가장 긴 것, 방향은 뒤 벽 법선 (실물에서 옆 벽보다 곧게 보임)
- 입구 모양 방식 (find_slot_mouth, 칸 밖용): 실물에서는 옆 벽 안쪽 면이 비스듬해 점이 안 돌아오므로
    뒤 벽 (방향 기준) + 그 앞 depth 에 좌우 하나씩 있는 옆 벽 앞 끝면 (짧은 직선, 각도 25° 까지 허용) 으로 찾음
- 뒤 벽 기준 방식 (find_slot_back, 입구 통과 구간용): 앞 끝면이 라이다 10 cm 안이면 기울고 잘려 보이므로
    멀리 있어 정확한 뒤 벽 + 그 양 끝의 옆 벽 안쪽 면·앞 끝면 (하나 이상) 으로 찾음. side 자리는 뒤 벽
- find_slot_any: 옆 벽 방식 -> 입구 모양 방식 -> 뒤 벽 기준 방식 순서

사용:
  slot = find_slot_any(segments, 0.190, 0.115, 0.200, math.radians(10), 0.04)
  center, outward, width, (side_a, side_b), back = slot     # 없으면 None (입구 모양: side = 앞 끝면)
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
            normal  = np.array([-back_seg.direction[1], back_seg.direction[0]])     # 뒤 벽 법선 (가장 곧은 면)
            normal  = normal if np.dot(normal, inward) > 0 else -normal
            best, best_score = (center, -normal, width, (a, b), back_seg), score
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

# =========================================================
# 입구 모양 방식: 뒤 벽 기준 + 앞 끝면 2개 (칸 밖용)
# =========================================================
def find_slot_mouth(segments, depth, width_min, width_max, angle_tol,
                    face_len=(0.02, 0.09), face_tol=math.radians(25), depth_tol=0.03, back_min=0.06):
    """뒤 벽 앞 depth 에 앞 끝면이 좌우 하나씩 -> find_slot 과 같은 형식. 방향은 뒤 벽 법선."""
    faces = [f for f in segments if face_len[0] <= f.length <= face_len[1]]
    best, best_err = None, depth_tol
    for c in segments:                                                              # c: 뒤 벽 후보
        if not back_min <= c.length <= width_max + 0.02:
            continue
        along   = c.direction                                                       # 칸 폭 방향
        normal  = np.array([-along[1], along[0]])                                   # 칸 축 방향
        left, right = [], []
        for f in faces:                                                             # 앞 끝면: 각도는 느슨하게
            if f is c or abs(math.remainder(f.angle - c.angle, math.pi)) > face_tol:
                continue
            s = float(np.dot(f.center - c.center, normal))                          # 뒤 벽 -> 앞 끝면 (부호 있음)
            if abs(abs(s) - depth) > depth_tol:
                continue
            e = sorted([np.dot(f.start - c.center, along), np.dot(f.end - c.center, along)])
            if e[0] > 0:
                right.append((f, s, e[0]))                                          # 안쪽 끝 = 뒤 벽 가운데 쪽
            elif e[1] < 0:
                left.append((f, s, e[1]))
        for fl, sl, il in left:
            for fr, sr, ir in right:
                gap, ms = ir - il, (sl + sr) / 2                                    # 칸 폭, 입구 선 위치
                err = abs(abs(ms) - depth)
                if sl * sr <= 0 or not width_min - 0.02 <= gap <= width_max or err >= best_err:
                    continue
                outward = normal if ms > 0 else -normal                             # 칸 안 -> 입구
                center  = c.center + ms * normal + (il + ir) / 2 * along            # 앞 끝면 안쪽 끝의 가운데
                best, best_err = (center, outward, gap, (fl, fr), c), err
    return best

# =========================================================
# 뒤 벽 기준 방식: 뒤 벽 + 옆 벽 안쪽 면 / 앞 끝면 하나 이상 (입구 통과 구간용)
# =========================================================
def find_slot_back(segments, depth, width_min, width_max, angle_tol,
                   back_min=0.09, back_range=0.45, side_tol=math.radians(20), face_tol=math.radians(25),
                   depth_tol=0.03, end_tol=0.045):
    """뒤 벽 (라이다 back_range 안) 양 끝의 옆 벽 안쪽 면·앞 끝면으로 폭을 정함 -> find_slot 과 같은 형식."""
    best, best_key = None, None
    for c in segments:                                                              # c: 뒤 벽 후보
        if not back_min <= c.length <= width_max + 0.02 or np.linalg.norm(c.center) > back_range:
            continue                                                                # 길이·거리 밖 (가까운 칸만)
        along = c.direction                                                         # 칸 폭 방향
        ends  = sorted([float(np.dot(c.start - c.center, along)), float(np.dot(c.end - c.center, along))])
        for sign in (1.0, -1.0):                                                    # 칸 밖 방향 두 가지 모두 시험
            outward = sign * np.array([-along[1], along[0]])
            if float(np.dot(-c.center, outward)) < depth - 0.06:                    # 로봇이 입구 근처·밖일 때만
                continue                                                            # (칸 안에서 옆 벽을 뒤 벽으로 오인 방지)
            lo, hi, support = ends[0], ends[1], 0
            for f in segments:
                if f is c or f.length < 0.02:
                    continue
                diff = abs(math.remainder(f.angle - c.angle, math.pi))
                ax   = sorted([float(np.dot(f.start - c.center, outward)), float(np.dot(f.end - c.center, outward))])
                lat  = sorted([float(np.dot(f.start - c.center, along)), float(np.dot(f.end - c.center, along))])
                if abs(diff - math.pi / 2) <= side_tol and ax[0] > -0.02 and ax[1] < depth + 0.02:
                    edge = (lat[0] + lat[1]) / 2                                    # 옆 벽 안쪽 면: 면 위치
                elif diff <= face_tol and abs((ax[0] + ax[1]) / 2 - depth) <= depth_tol and f.length <= 0.09:
                    edge = lat[0] if lat[0] > 0 else lat[1]                         # 앞 끝면: 안쪽 끝
                else:
                    continue
                if edge < 0 and abs(edge - ends[0]) <= end_tol:
                    lo, support = min(lo, edge), support + 1
                elif edge > 0 and abs(edge - ends[1]) <= end_tol:
                    hi, support = max(hi, edge), support + 1
            key = (support, -float(np.linalg.norm(c.center)))                       # 근거 많은 것, 가까운 것 우선
            if support == 0 or not width_min <= hi - lo <= width_max or (best_key and key <= best_key):
                continue
            center = c.center + (lo + hi) / 2 * along + depth * outward
            best, best_key = (center, outward, hi - lo, (c, c), c), key
    return best

def find_slot_any(segments, depth, width_min, width_max, angle_tol, min_overlap):
    """옆 벽 방식 (칸 안·정면) -> 입구 모양 방식 (칸 밖) -> 뒤 벽 기준 방식 (입구 통과 구간) 순서."""
    return (find_slot(segments, depth, width_min, width_max, angle_tol, min_overlap)
            or find_slot_mouth(segments, depth, width_min, width_max, angle_tol)
            or find_slot_back(segments, depth, width_min, width_max, angle_tol))
