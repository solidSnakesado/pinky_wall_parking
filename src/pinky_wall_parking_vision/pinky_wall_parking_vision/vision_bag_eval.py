"""src/pinky_wall_parking_vision/pinky_wall_parking_vision/vision_bag_eval.py - 녹화 영상으로 칸 특징 검출 평가

역할: 칸으로 다가가며 녹화한 ros2 bag 에서 거리별 검출률과 흔들림을 낸다 (ROS 노드·재생 없이).
방법: rosbag2_py 로 압축 영상과 /odom 을 읽고, 영상마다 slot_features.extract_features 를 그대로 부른다.
      거리 = 출발 때 앞 끝 ↔ 뒤 벽 거리 - odom 이동량 (출발 방향으로 투영, 영상 시각에 맞춰 보간).
      프레임별 CSV 와 거리 구간별 요약 CSV 를 출력 폴더에 저장하고, 요약은 화면에도 출력한다.
실행: ros2 run pinky_wall_parking_vision vision_bag_eval ~/pinky_wall_parking/runs/bags/vis_sim_straight
      옵션: --start-back 420 (출발 때 앞 끝 ↔ 뒤 벽 mm), --roll -1.4, --bin 20, --out ~/pinky_wall_parking/runs
"""

# --- 표준 라이브러리 ---------------------------------------------------------------
import argparse                                                                     # 명령줄 인자
import csv                                                                          # 결과 저장
import math                                                                         # 각도 계산
from pathlib import Path                                                            # 경로

# --- 외부 라이브러리 ---------------------------------------------------------------
import cv2                                                                          # JPEG 풀기
import numpy as np                                                                  # 보간·통계

# --- ROS 2 -------------------------------------------------------------------------
from rclpy.serialization import deserialize_message                                 # 메시지 복원
import rosbag2_py                                                                   # 녹화 파일 읽기
from rosidl_runtime_py.utilities import get_message                                 # 메시지 형식

# --- 이 패키지 ---------------------------------------------------------------------
from pinky_wall_parking_vision.slot_features import extract_features                # 특징 찾기
from pinky_wall_parking_vision.slot_features import FeatureParams                   # 검출 파라미터

FRAME_FIELDS = ["t", "back_mm", "back_found", "left_found", "right_found", "vp_found",
                "back_y", "slope_deg", "left_x", "right_x", "center_x", "vp_x", "vp_y"]


# ==================================================================================
# 녹화 파일 -> 영상 (시각, JPEG), odom (시각, x, y, yaw)
# ==================================================================================
def read_bag(path, image_topic, odom_topic):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(path), storage_id=""),
                rosbag2_py.ConverterOptions("cdr", "cdr"))                          # 저장 형식 자동 판별
    types  = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types()}
    reader.set_filter(rosbag2_py.StorageFilter(topics=[image_topic, odom_topic]))
    images, odom = [], []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        msg   = deserialize_message(data, types[topic])
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9              # 메시지 시각 (시뮬 = /clock)
        if topic == image_topic:
            images.append((stamp, bytes(msg.data)))
        else:
            p, q = msg.pose.pose.position, msg.pose.pose.orientation
            yaw  = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            odom.append((stamp, p.x, p.y, yaw))
    return images, np.array(odom).reshape(-1, 4)


# ==================================================================================
# odom -> 출발 방향 이동량 (mm)
# ==================================================================================
def travel_mm(odom):
    t, x, y, yaw = odom.T
    ahead = (x - x[0]) * math.cos(yaw[0]) + (y - y[0]) * math.sin(yaw[0])           # 출발 방향 성분
    return t, ahead * 1000.0


# ==================================================================================
# 영상마다 특징 찾기 -> 프레임별 결과
# ==================================================================================
def evaluate(images, odom, start_back, params):
    t, s = travel_mm(odom)
    rows = []
    for stamp, data in images:
        f = extract_features(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), params)
        rows.append({
            "t": round(stamp, 3), "back_mm": round(start_back - float(np.interp(stamp, t, s)), 1),
            "back_found": int(f.back_found), "left_found": int(f.left_found),
            "right_found": int(f.right_found), "vp_found": int(f.vp_found),
            "back_y": round(f.back_y, 1), "slope_deg": round(math.degrees(math.atan(f.back_slope)), 2),
            "left_x": f.left_x, "right_x": f.right_x, "center_x": f.center_x,
            "vp_x": round(f.vp_x, 1), "vp_y": round(f.vp_y, 1)})
    return rows


# ==================================================================================
# 거리 구간별 요약 (검출률 %, 평균, 표준편차)
# ==================================================================================
def mean_std(values):
    return (round(float(np.mean(values)), 1), round(float(np.std(values)), 1)) if values else ("", "")


def summarize(rows, bin_mm):
    bins = {}
    for r in rows:
        bins.setdefault(int(r["back_mm"] // bin_mm) * bin_mm, []).append(r)
    out = []
    for lo in sorted(bins, reverse=True):                                           # 먼 곳 -> 가까운 곳
        rs   = bins[lo]
        both = [r for r in rs if r["left_found"] and r["right_found"]]
        vps  = [r for r in rs if r["vp_found"]]
        back = [r for r in rs if r["back_found"]]
        row  = {"back_mm": f"{lo:.0f}-{lo + bin_mm:.0f}", "n": len(rs),
                "back_pct": round(100.0 * len(back) / len(rs)),
                "corner_pct": round(100.0 * len(both) / len(rs)),
                "vp_pct": round(100.0 * len(vps) / len(rs))}
        for key, src in (("back_y", back), ("center_x", both), ("vp_x", vps)):
            row[key], row[key + "_sd"] = mean_std([r[key] for r in src])
        row["width"], _ = mean_std([r["right_x"] - r["left_x"] for r in both])
        out.append(row)
    return out


def write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ==================================================================================
# 실행
# ==================================================================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate slot features on recorded camera bags")
    parser.add_argument("bags", nargs="+", help="ros2 bag folders")
    parser.add_argument("--start-back", type=float, default=420.0, help="front end to back wall at start (mm)")
    parser.add_argument("--roll", type=float, default=0.0, help="camera roll correction (deg)")
    parser.add_argument("--bin", type=float, default=20.0, help="distance bin (mm)")
    parser.add_argument("--out", default=str(Path.home() / "pinky_wall_parking" / "runs"))
    parser.add_argument("--image-topic", default="/camera/image_raw/compressed")
    parser.add_argument("--odom-topic", default="/odom")
    args   = parser.parse_args()
    params = FeatureParams(roll_deg=args.roll)
    for bag in args.bags:
        name         = Path(bag).name
        images, odom = read_bag(bag, args.image_topic, args.odom_topic)
        if not images or len(odom) < 2:
            print(f"{name}: images {len(images)}, odom {len(odom)} -> skip")
            continue
        rows    = evaluate(images, odom, args.start_back, params)
        summary = summarize(rows, args.bin)
        write_csv(Path(args.out) / f"{name}_frames.csv", rows)
        write_csv(Path(args.out) / f"{name}_summary.csv", summary)
        print(f"== {name}: {len(rows)} images, roll {args.roll} deg -> {args.out}/{name}_*.csv")
        print("back_mm    n  back% corner%  vp%  back_y(sd)    center_x(sd)   vp_x(sd)      width")
        for s in summary:
            print(f"{s['back_mm']:>8} {s['n']:4} {s['back_pct']:5} {s['corner_pct']:6} {s['vp_pct']:5}  "
                  f"{s['back_y']!s:>6}({s['back_y_sd']!s:>4})  {s['center_x']!s:>6}({s['center_x_sd']!s:>4})  "
                  f"{s['vp_x']!s:>6}({s['vp_x_sd']!s:>4})  {s['width']!s:>5}")


if __name__ == "__main__":
    main()
