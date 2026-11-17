"""
src/pinky_wall_parking_perception/pinky_wall_parking_perception/line_extraction.py - 2D 라이다 점에서 직선 조각 추출

- ROS 없이 numpy 만 사용 (노드와 분리해 따로 시험 가능)
- 방법 (2D 라이다 직선 추출의 표준 구성, laser_line_extraction 패키지와 같은 순서)
    1. 거리값 -> 점 (x, y), 범위 밖/무한대 값은 끊김으로 처리
    2. 이웃한 점 사이가 max_gap 보다 멀면 덩어리를 나눔
    3. Split: 덩어리 양 끝을 잇는 직선에서 가장 먼 점이 split_dist 보다 멀면 그 점에서 둘로 나눔 (반복)
    4. 직선 맞춤: 조각마다 전체 최소제곱 (PCA) -> 중심, 방향, 끝점, 오차
    5. Merge: 이웃한 조각이 거의 같은 직선이면 합침
    6. 점 수/길이가 너무 작은 조각은 버림
- 기준값 (m, rad) 은 LineParams 로 바꿀 수 있음 (칸 벽 190 mm, 거리 0.05 ~ 1 m 기준 기본값)

사용:
  points = scan_to_points(ranges, angle_min, angle_increment, range_min, range_max)
  segments = extract_lines(points, LineParams())
"""

# ==================== 표준 라이브러리 ======================
from dataclasses import dataclass                                                   # 기준값/결과 묶음
import math                                                                         # 각도 계산

# ==================== 외부 라이브러리 ======================
import numpy as np                                                                  # 점 계산

# =========================================================
# 기준값
# =========================================================
@dataclass
class LineParams:
    max_gap:        float = 0.04                                                    # 이웃 점 최대 간격 (m), 넘으면 덩어리 나눔
    split_dist:     float = 0.015                                                   # 직선에서 이 거리보다 먼 점이 있으면 나눔 (m)
    merge_angle:    float = math.radians(5.0)                                       # 합칠 때 두 직선 방향 차 최대 (rad)
    merge_dist:     float = 0.015                                                   # 합칠 때 두 직선 사이 거리 최대 (m)
    min_points:     int   = 4                                                       # 조각 최소 점 수
    min_length:     float = 0.05                                                    # 조각 최소 길이 (m)

# =========================================================
# 직선 조각 (결과)
# =========================================================
@dataclass
class Segment:
    start:      np.ndarray                                                          # 시작 끝점 (x, y), 직선 위로 투영
    end:        np.ndarray                                                          # 끝 끝점 (x, y), 직선 위로 투영
    center:     np.ndarray                                                          # 점들의 중심 (x, y)
    direction:  np.ndarray                                                          # 단위 방향 (start -> end)
    normal:     np.ndarray                                                          # 단위 법선 (방향을 왼쪽으로 90도)
    length:     float                                                               # 끝점 사이 길이 (m)
    rms:        float                                                               # 점 -> 직선 거리 제곱평균 (m)
    count:      int                                                                 # 점 수

    @property
    def angle(self):
        """방향 각도 (rad, -pi ~ pi)."""
        return math.atan2(self.direction[1], self.direction[0])

# =========================================================
# 거리값 -> 점
# =========================================================
def scan_to_points(ranges, angle_min, angle_increment, range_min, range_max):
    """LaserScan 값 -> (N, 2) 배열. 쓸 수 없는 값은 NaN 으로 남겨 끊김으로 씀."""
    ranges  = np.asarray(ranges, dtype=float)
    angles  = angle_min + angle_increment * np.arange(len(ranges))                  # 광선마다 각도
    valid   = np.isfinite(ranges) & (ranges >= range_min) & (ranges <= range_max)
    points  = np.full((len(ranges), 2), np.nan)
    points[valid, 0] = ranges[valid] * np.cos(angles[valid])
    points[valid, 1] = ranges[valid] * np.sin(angles[valid])
    return points

# =========================================================
# 직선 맞춤 (전체 최소제곱, PCA)
# =========================================================
def fit_segment(points):
    """(N, 2) 점 -> Segment. 방향은 첫 점 -> 마지막 점 쪽으로 맞춤."""
    center      = points.mean(axis=0)
    _, _, vt    = np.linalg.svd(points - center, full_matrices=False)               # 가장 큰 분산 방향 = 직선 방향
    direction   = vt[0]
    if np.dot(points[-1] - points[0], direction) < 0:                               # 점 순서와 같은 방향
        direction = -direction
    normal      = np.array([-direction[1], direction[0]])
    along       = (points - center) @ direction                                     # 직선 방향 좌표
    across      = (points - center) @ normal                                        # 직선과의 거리
    start       = center + along.min() * direction
    end         = center + along.max() * direction
    return Segment(start=start, end=end, center=center, direction=direction, normal=normal,
                   length=float(along.max() - along.min()),
                   rms=float(np.sqrt(np.mean(across ** 2))), count=len(points))

# =========================================================
# 덩어리 나누기 (끊김·간격)
# =========================================================
def split_clusters(points, max_gap):
    """NaN 이나 max_gap 보다 먼 간격에서 끊어 연속된 점 덩어리 목록 반환."""
    clusters, current = [], []
    previous = None
    for point in points:
        if np.isnan(point[0]):                                                      # 측정 없음 -> 끊김
            previous = None
            if current:
                clusters.append(np.array(current))
                current = []
            continue
        if previous is not None and np.linalg.norm(point - previous) > max_gap:
            clusters.append(np.array(current))
            current = []
        current.append(point)
        previous = point
    if current:
        clusters.append(np.array(current))
    first, last = points[0], points[-1]                                             # 스캔 시작/끝은 이어진 방향
    if (len(clusters) > 1 and not np.isnan(first[0]) and not np.isnan(last[0])
            and np.linalg.norm(first - last) <= max_gap):
        clusters[0] = np.vstack([clusters.pop(), clusters[0]])                      # 마지막 + 처음 합침
    return clusters

# =========================================================
# Split (양 끝 직선에서 가장 먼 점으로 나누기, 반복)
# =========================================================
def split_points(points, split_dist, min_points):
    """점 덩어리 -> 직선에 가까운 점 묶음 목록."""
    pieces, stack = [], [points]
    while stack:
        part = stack.pop()
        if len(part) < max(3, min_points):                                          # 나눌 수 없을 만큼 작음
            pieces.append(part)
            continue
        chord   = part[-1] - part[0]
        norm    = np.linalg.norm(chord)
        if norm < 1e-9:
            pieces.append(part)
            continue
        normal  = np.array([-chord[1], chord[0]]) / norm
        dist    = np.abs((part - part[0]) @ normal)                                 # 양 끝 직선과의 거리
        index   = int(np.argmax(dist[1:-1])) + 1                                    # 양 끝 제외
        if dist[index] > split_dist:
            stack.append(part[index:])                                              # 나눈 점은 양쪽에 포함
            stack.append(part[:index + 1])
        else:
            pieces.append(part)
    return pieces

# =========================================================
# Merge (이웃한 같은 직선 합치기)
# =========================================================
def merge_pieces(pieces, params):
    """순서대로 놓인 점 묶음 중 이웃이 같은 직선이면 합침."""
    merged = []
    for part in pieces:
        if merged:
            last    = fit_segment(merged[-1])
            this    = fit_segment(part)
            angle   = abs(math.remainder(last.angle - this.angle, math.pi))         # 방향 차 (0 ~ pi/2)
            gap     = abs(np.dot(this.center - last.center, last.normal))           # 직선 사이 거리
            close   = np.linalg.norm(part[0] - merged[-1][-1]) <= params.max_gap
            if close and angle < params.merge_angle and gap < params.merge_dist:
                merged[-1] = np.vstack([merged[-1], part])
                continue
        merged.append(part)
    return merged

# =========================================================
# 전체: 점 -> 직선 조각 목록
# =========================================================
def extract_lines(points, params=None):
    """(N, 2) 점 (스캔 순서, NaN = 끊김) -> Segment 목록 (스캔 순서)."""
    params      = params or LineParams()
    segments    = []
    for cluster in split_clusters(points, params.max_gap):
        pieces  = split_points(cluster, params.split_dist, params.min_points)
        pieces  = sorted(pieces, key=lambda p: _first_index(cluster, p))            # 스캔 순서로 정렬
        for part in merge_pieces(pieces, params):
            if len(part) < params.min_points:
                continue
            segment = fit_segment(part)
            if segment.length >= params.min_length:
                segments.append(segment)
    return segments

# =========================================================
# 덩어리 안에서 묶음의 첫 점 위치 (정렬용)
# =========================================================
def _first_index(cluster, part):
    """part 의 첫 점이 cluster 몇 번째 점인지."""
    return int(np.argmin(np.linalg.norm(cluster - part[0], axis=1)))