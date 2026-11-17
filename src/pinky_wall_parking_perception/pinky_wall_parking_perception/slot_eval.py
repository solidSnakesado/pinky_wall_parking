"""
src/pinky_wall_parking_perception/pinky_wall_parking_perception/slot_eval.py - 라이다 칸 인식 평가 (시뮬 정답과 비교)

- 구독  : /scan (인식 시도 횟수), /slot/pose, /slot/width (인식 결과), /ground_truth (시뮬 정답 로봇 자세, world)
- 정답  : slot_world.slot_opening(폭, 틈) 의 월드 입구 위치·방향 (+y 쪽) 을 정답 로봇 자세 기준 (base_link) 으로 바꿈
- 스캔마다 기록 (CSV): 로봇 자세, 입구까지 거리, 칸 축과 틀어진 각도, 인식 여부, 위치·방향·폭 오차
    각도 0 = 로봇 앞쪽이 칸 바깥 방향 (뒤쪽 라이다가 칸 입구를 정면으로 봄)
- summary_period 초마다 거리·각도 구간별 인식 성공률, 평균·최대 오차를 터미널에 출력
- 저장: <out_dir>/slot_eval_<폭>mm_<시각>.csv

실행 (slot_detector 와 함께, 로봇은 키보드로 칸 앞에서 움직임):
  ros2 run pinky_wall_parking_perception slot_eval --ros-args -p use_sim_time:=true -p slot_width:=130.0
"""

# ==================== 표준 라이브러리 ======================
from collections import deque                                                       # 정답 자세 최근 기록
import csv                                                                          # 결과 파일
from datetime import datetime                                                       # 파일 이름 시각
import math                                                                         # 각도 계산
from pathlib import Path                                                            # 저장 폴더

# ==================== ROS 2 ================================
from geometry_msgs.msg import PoseStamped                                           # 인식 결과, 정답 자세
import rclpy                                                                        # ROS 2 파이썬
from rclpy.node import Node                                                         # 노드 기본 클래스
from rclpy.qos import qos_profile_sensor_data                                       # best-effort QoS
from sensor_msgs.msg import LaserScan                                               # 인식 시도 기준
from std_msgs.msg import Float32                                                    # 칸 폭

# ==================== 이 프로젝트 ==========================
from pinky_wall_parking_sim.slot_world import slot_opening                          # 정답 입구 위치

# 요약 구간: 칸 축과 틀어진 각도 (deg), 입구까지 거리 (m)
ANGLE_BINS  = [0.0, 5.0, 10.0, 20.0, 30.0, 180.0]
DIST_BINS   = [0.0, 0.15, 0.30, 0.60]
COLUMNS     = ["t", "robot_x", "robot_y", "robot_yaw_deg", "dist_m", "angle_deg", "detected",
               "err_x_mm", "err_y_mm", "err_pos_mm", "err_yaw_deg", "width_mm"]

# =========================================================
# 자세 도우미
# =========================================================
def stamp_sec(stamp):
    """builtin_interfaces/Time -> 초."""
    return stamp.sec + stamp.nanosec * 1e-9

def pose_xy_yaw(pose):
    """geometry_msgs/Pose -> (x, y, yaw)."""
    q = pose.orientation
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    return pose.position.x, pose.position.y, yaw

def wrap_deg(angle):
    """rad -> -180 ~ 180 deg."""
    return math.degrees(math.remainder(angle, 2 * math.pi))

# =========================================================
# 평가 노드
# =========================================================
class SlotEval(Node):
    def __init__(self):
        super().__init__("slot_eval")
        width_mm    = self.declare_parameter("slot_width", 130.0).value             # 시뮬 칸 폭 (mm)
        gap_mm      = self.declare_parameter("side_gap", 50.0).value                # 방 옆 벽 틈 (mm)
        out_dir     = self.declare_parameter("out_dir", "~/pinky_wall_parking/runs").value
        period      = self.declare_parameter("summary_period", 5.0).value           # 요약 출력 주기 (s)

        self.width_true = width_mm / 1000.0
        self.opening    = slot_opening(self.width_true, gap_mm / 1000.0)            # 월드 입구 중심 (m)
        self.out_yaw    = math.pi / 2                                               # 월드 칸 바깥 방향 (+y)
        self.truths     = deque(maxlen=100)                                         # (t, x, y, yaw), 50 Hz
        self.pending    = None                                                      # 처리 중인 스캔 기록
        self.results    = {}                                                        # 스캔 시각 -> 인식 결과
        self.last_t     = None                                                      # 마지막 인식 결과 시각
        self.rows       = []                                                        # 요약용 전체 기록

        path = Path(out_dir).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        path = path / f"slot_eval_{width_mm:g}mm_{datetime.now():%Y%m%d_%H%M%S}.csv"
        self.file   = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(COLUMNS)

        self.create_subscription(PoseStamped, "ground_truth", self.on_truth, 10)
        self.create_subscription(LaserScan, "scan", self.on_scan, qos_profile_sensor_data)
        self.create_subscription(PoseStamped, "slot/pose", self.on_pose, 10)
        self.create_subscription(Float32, "slot/width", self.on_width, 10)
        self.create_timer(period, self.print_summary)
        self.get_logger().info(f"slot_eval: width {width_mm:g} mm, opening "
                               f"({self.opening[0]:.3f}, {self.opening[1]:.3f}) m -> {path}")

    # ---- 정답 자세 저장 ----------------------------------------
    def on_truth(self, msg):
        self.truths.append((stamp_sec(msg.header.stamp), *pose_xy_yaw(msg.pose)))

    # ---- 스캔 = 인식 시도 1번 ----------------------------------
    def on_scan(self, msg):
        self.finish()                                                               # 이전 스캔 기록 마감
        if not self.truths:
            return
        t       = stamp_sec(msg.header.stamp)
        _, x, y, yaw = min(self.truths, key=lambda item: abs(item[0] - t))          # 스캔 시각과 가장 가까운 정답
        dx, dy  = self.opening[0] - x, self.opening[1] - y
        c, s    = math.cos(yaw), math.sin(yaw)
        self.pending = {
            "t": t, "robot_x": x, "robot_y": y, "robot_yaw_deg": math.degrees(yaw),
            "true_xy": (c * dx + s * dy, -s * dx + c * dy),                         # base_link 기준 정답 입구
            "true_yaw": self.out_yaw - yaw,
            "dist_m": math.hypot(dx, dy), "angle_deg": wrap_deg(yaw - self.out_yaw),
            "detected": 0}

    # ---- 인식 결과 (스캔 시각으로 저장, 도착 순서와 무관) ------
    def on_pose(self, msg):
        t = stamp_sec(msg.header.stamp)
        self.results[t] = list(pose_xy_yaw(msg.pose)) + [math.nan]                  # x, y, yaw, 폭
        self.last_t = t
        for old in [k for k in self.results if k < t - 2.0]:                        # 2초 지난 것 정리
            del self.results[old]

    def on_width(self, msg):
        if self.last_t in self.results:                                             # 바로 앞 pose 의 폭
            self.results[self.last_t][3] = msg.data * 1000

    # ---- 기록 마감 (CSV 한 줄) ---------------------------------
    def finish(self):
        rec, self.pending = self.pending, None
        if rec is None:
            return
        result = self.results.pop(rec["t"], None)
        if result is not None:
            mx, my, myaw, width = result
            tx, ty = rec["true_xy"]
            rec.update(detected=1, err_x_mm=(mx - tx) * 1000, err_y_mm=(my - ty) * 1000,
                       err_pos_mm=math.hypot(mx - tx, my - ty) * 1000,
                       err_yaw_deg=wrap_deg(myaw - rec["true_yaw"]), width_mm=width)
        self.writer.writerow([f"{rec[k]:.4f}" if isinstance(rec.get(k), float) else rec.get(k, "")
                              for k in COLUMNS])
        self.rows.append(rec)

    # ---- 구간별 요약 -------------------------------------------
    def print_summary(self):
        self.file.flush()
        if not self.rows:
            return
        lines = [f"scans {len(self.rows)} | angle(deg) dist(m): detected/scans, "
                 "pos err mean/max (mm), yaw err mean/max (deg), width err mean (mm)"]
        for a0, a1 in zip(ANGLE_BINS, ANGLE_BINS[1:]):
            for d0, d1 in zip(DIST_BINS, DIST_BINS[1:]):
                group = [r for r in self.rows
                         if a0 <= abs(r["angle_deg"]) < a1 and d0 <= r["dist_m"] < d1]
                if not group:
                    continue
                hits = [r for r in group if r["detected"]]
                text = f"  {a0:4.0f}~{a1:<4.0f} {d0:.2f}~{d1:.2f}: {len(hits):4d}/{len(group):<4d}"
                if hits:
                    pos     = [r["err_pos_mm"] for r in hits]
                    yaw     = [abs(r["err_yaw_deg"]) for r in hits]
                    width   = [abs(r["width_mm"] - self.width_true * 1000) for r in hits]
                    text   += (f"  pos {sum(pos) / len(pos):5.1f}/{max(pos):5.1f}"
                               f"  yaw {sum(yaw) / len(yaw):4.2f}/{max(yaw):4.2f}"
                               f"  width {sum(width) / len(width):4.1f}")
                lines.append(text)
        self.get_logger().info("\n".join(lines))

    def destroy_node(self):
        self.finish()
        self.print_summary()
        self.file.close()
        super().destroy_node()

# =========================================================
# 실행
# =========================================================
def main(args=None):
    rclpy.init(args=args)
    node = SlotEval()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == "__main__":
    main()