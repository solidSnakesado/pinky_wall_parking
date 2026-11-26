"""
src/pinky_wall_parking_perception/pinky_wall_parking_perception/bag_eval.py - 녹화한 실물 /scan 으로 칸 인식 분석

- ros2 bag 녹화 파일 (로봇 정지 상태) 을 rosbag2_py 로 직접 읽음 (ROS 노드·재생 없이)
- line_extraction, slot_geometry 를 그대로 불러 씀 -> 노드와 같은 계산
- 녹화 파일마다 출력
    라이다  : 스캔당 유효 점 수, 빠진 점 수, 같은 광선 거리값의 흔들림 (표준편차 중앙값, 1 m 이내)
    직선    : 스캔당 직선 수, 칸 옆 벽/뒤 벽 직선의 rms
    칸 인식 : 찾은 횟수 / 스캔 수, 아래 값의 평균 ± 표준편차
              뒤 끝 거리 = 로봇 뒤 끝 (base_link x -0.076) -> 입구 선, 칸 밖이면 +, 칸 안이면 - (mm)
              각도 = 로봇이 칸 축과 틀어진 각도, 위에서 볼 때 시계 반대 방향 + (deg)
              좌우 = 로봇 중심이 칸 가운데 선에서 벗어난 거리 (mm), 폭 (mm)
    실측    : 파일 이름의 d<mm> (뒤 끝 거리), in<mm> (칸 안), a<deg> (각도) 를 함께 출력
- 라이다 -> base_link 변환은 실물 tf 로 확인한 값 (x -0.017 m, yaw 180 deg) 을 기본값으로 씀
- 실물 근거리 보정 (REAL_NEAR) 기본 켬, --no-near 로 끔. 칸 찾기는 find_slot_any (옆 벽 / 입구 모양)

실행:
  ros2 run pinky_wall_parking_perception bag_eval ~/pinky_wall_parking/runs/bags/p*
  기준값 바꾸기: --split-dist 0.03 --merge-dist 0.03 --min-points 4 --max-gap 0.04
  시뮬 실물 라이다 녹화: --topic /scan_real (실물 녹화와 같은 --laser-yaw 177.8 로 비교)
"""

# ==================== 표준 라이브러리 ======================
import argparse                                                                     # 명령줄 인자
import math                                                                         # 각도 계산
from pathlib import Path                                                            # 녹화 폴더 이름
import re                                                                           # 이름에서 실측값

# ==================== 외부 라이브러리 ======================
import numpy as np                                                                  # 통계

# ==================== ROS 2 ================================
from rclpy.serialization import deserialize_message                                 # 메시지 복원
import rosbag2_py                                                                   # 녹화 파일 읽기
from rosidl_runtime_py.utilities import get_message                                 # 메시지 형식

# ==================== 이 패키지 ============================
from pinky_wall_parking_perception.line_extraction import (                         # 직선 추출
    LineParams, REAL_NEAR, extract_lines, scan_to_points)
from pinky_wall_parking_perception.slot_geometry import find_slot_any               # 칸 모양 맞추기

ROBOT_REAR = -0.076                                                                 # base_link 기준 로봇 뒤 끝 x (m)

# =========================================================
# 녹화 파일 -> LaserScan 목록
# =========================================================
def read_scans(path, topic="/scan"):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(path), storage_id=""),
                rosbag2_py.ConverterOptions("cdr", "cdr"))                          # 저장 형식 자동 판별
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    msg_type, scans = get_message(types[topic]), []
    while reader.has_next():
        _, data, _ = reader.read_next()
        scans.append(deserialize_message(data, msg_type))
    return scans

# =========================================================
# 칸 결과 -> 로봇 기준 값 (뒤 끝 거리, 각도, 좌우)
# =========================================================
def robot_values(slot):
    center, outward, width, sides, back = slot
    normal  = np.array([-outward[1], outward[0]])                                   # 칸 폭 방향
    rear    = np.array([ROBOT_REAR, 0.0])
    dist    = float(np.dot(rear - center, outward))                                 # 칸 밖 +, 칸 안 -
    angle   = -math.degrees(math.atan2(outward[1], outward[0]))                     # 로봇이 틀어진 각도
    lateral = float(np.dot(-center, normal))                                        # 로봇 중심 -> 칸 가운데 선
    rms     = [s.rms for s in (*sides, back)]
    return dist * 1000, angle, lateral * 1000, width * 1000, rms

# =========================================================
# 녹화 파일 하나 분석
# =========================================================
def analyze(path, args, params):
    scans = read_scans(path, args.topic)
    c, s  = math.cos(math.radians(args.laser_yaw)), math.sin(math.radians(args.laser_yaw))
    ranges = np.array([np.asarray(m.ranges, dtype=float) for m in scans])
    valid  = np.isfinite(ranges) & (ranges >= scans[0].range_min) & (ranges <= scans[0].range_max)
    stable = valid.all(axis=0)                                                      # 모든 스캔에서 측정된 광선
    stable[stable] = np.median(ranges[:, stable], axis=0) < 1.0                     # 그중 1 m 이내
    noise  = np.std(ranges[:, stable], axis=0) * 1000 if stable.any() else np.array([np.nan])

    counts, found, rms_all = [], [], []
    for m in scans:
        p = scan_to_points(m.ranges, m.angle_min, m.angle_increment, m.range_min, m.range_max,
                           near=None if args.no_near else REAL_NEAR)
        p = np.column_stack((c * p[:, 0] - s * p[:, 1] + args.laser_x,
                             s * p[:, 0] + c * p[:, 1]))                            # base_link 좌표
        segments = extract_lines(p, params)
        counts.append(len(segments))
        slot = find_slot_any(segments, args.depth, args.width_min, args.width_max,
                             math.radians(args.angle_tol), args.min_overlap)
        if slot is not None:
            *values, rms = robot_values(slot)
            found.append(values)
            rms_all += rms

    name    = Path(path).name
    truth   = {k: int(v) for k, v in re.findall(r"(d|in|a)(\d+)", name)}
    t_dist  = truth.get("d", -truth.get("in", 0)) if ("d" in truth or "in" in truth) else None
    print(f"\n== {name}  (실측: 뒤 끝 거리 {t_dist} mm, 각도 {truth.get('a')} deg)")
    print(f"  라이다  : 스캔 {len(scans)}, 광선 {ranges.shape[1]}, 유효 점 {valid.sum(1).mean():.0f}, "
          f"빠진 점 {(~valid).sum(1).mean():.0f}, 거리 흔들림 {np.median(noise):.1f} mm "
          f"(1 m 이내 광선 {stable.sum()}개 중앙값)")
    print(f"  직선    : 스캔당 {np.mean(counts):.1f}개, 칸 벽 rms "
          f"{np.mean(rms_all) * 1000 if rms_all else float('nan'):.1f} mm")
    print(f"  칸 인식 : {len(found)}/{len(scans)}")
    if found:
        f = np.array(found)
        for i, label in enumerate(["뒤 끝 거리 (mm)", "각도 (deg)", "좌우 (mm)", "폭 (mm)"]):
            print(f"    {label:14s} {f[:, i].mean():8.1f} ± {f[:, i].std():5.1f}")

# =========================================================
# 실행
# =========================================================
def main():
    d = LineParams()
    parser = argparse.ArgumentParser(description="Analyze recorded /scan bags for wall slot detection")
    parser.add_argument("bags", nargs="+", help="ros2 bag folders")
    parser.add_argument("--split-dist", type=float, default=d.split_dist)
    parser.add_argument("--merge-dist", type=float, default=d.merge_dist)
    parser.add_argument("--max-gap", type=float, default=d.max_gap)
    parser.add_argument("--min-points", type=int, default=d.min_points)
    parser.add_argument("--min-length", type=float, default=d.min_length)
    parser.add_argument("--depth", type=float, default=0.190)
    parser.add_argument("--width-min", type=float, default=0.115)
    parser.add_argument("--width-max", type=float, default=0.200)
    parser.add_argument("--angle-tol", type=float, default=10.0, help="deg")
    parser.add_argument("--min-overlap", type=float, default=0.04)
    parser.add_argument("--laser-x", type=float, default=-0.017, help="laser x in base_link (m)")
    parser.add_argument("--laser-yaw", type=float, default=180.0, help="laser yaw in base_link (deg)")
    parser.add_argument("--no-near", action="store_true", help="disable real near-range correction")
    parser.add_argument("--topic", default="/scan", help="scan topic in the bag (sim real_c1: /scan_real)")
    args = parser.parse_args()
    params = LineParams(max_gap=args.max_gap, split_dist=args.split_dist, merge_dist=args.merge_dist,
                        min_points=args.min_points, min_length=args.min_length)
    print(f"기준값: split {args.split_dist}, merge {args.merge_dist}, gap {args.max_gap}, "
          f"min_points {args.min_points}, min_length {args.min_length}, near {not args.no_near}")
    for path in sorted(args.bags):
        analyze(path, args, params)

if __name__ == "__main__":
    main()