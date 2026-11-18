"""
src/pinky_wall_parking_perception/pinky_wall_parking_perception/slot_detector.py - 라이다로 벽 주차 칸 입구 위치·방향 찾기 (ROS 2 노드)

- 구독  : /scan (sensor_msgs/LaserScan, best-effort = 핑키 발행 QoS)
- 발행  : /slot/pose    (geometry_msgs/PoseStamped, base_link 기준 칸 입구 중심, 방향 = 칸 안 -> 입구)
          /slot/width   (std_msgs/Float32, 측정한 칸 안쪽 폭 m)
          /slot/markers (visualization_msgs/MarkerArray, rviz2 표시: 직선 조각 + 찾은 칸)
- 방법
    1. tf2 로 라이다 좌표 (rplidar_link, 뒤로 돌려 장착) -> base_link 변환 (고정 변환, 처음 한 번)
    2. line_extraction 으로 직선 조각 추출 (Split-and-Merge + 최소제곱)
    3. slot_geometry 로 칸 모양 맞추기 (나란한 옆 벽 2개 + 수직 뒤 벽, 없으면 옆 벽 앞 끝면 2개 + 뒤 벽)
- 실물은 near_correction:=true (RPLIDAR C1 근거리 보정, 6 cm 미만 버림), 시뮬은 false (기본)
- 실물은 laser_yaw_offset_deg:=-2.2 (라이다 장착 각도 보정, tf yaw 에 더함), 시뮬은 0 (기본)
- 결과는 base_link 기준 (측정 그대로): 4단계 제어기가 바로 쓰고, 평가도 로봇 기준으로 비교

실행:
실물: ros2 run pinky_wall_parking_perception slot_detector --ros-args -p near_correction:=true \
          -p laser_yaw_offset_deg:=-2.2
  실물: ros2 run pinky_wall_parking_perception slot_detector --ros-args -p near_correction:=true
"""

# ==================== 표준 라이브러리 ======================
import math                                                                         # 각도 계산

# ==================== 외부 라이브러리 ======================
import numpy as np                                                                  # 점 좌표 변환

# ==================== ROS 2 ================================
from geometry_msgs.msg import Point, PoseStamped                                    # 결과 자세, 표시 점
import rclpy                                                                        # ROS 2 파이썬
from rclpy.node import Node                                                         # 노드 기본 클래스
from rclpy.qos import qos_profile_sensor_data                                       # best-effort QoS
from rclpy.time import Time                                                         # tf 조회 시각
from sensor_msgs.msg import LaserScan                                               # 라이다
from std_msgs.msg import Float32                                                    # 칸 폭
from tf2_ros import Buffer, TransformException, TransformListener                   # 좌표 변환
from visualization_msgs.msg import Marker, MarkerArray                              # rviz2 표시

# ==================== 이 패키지 ============================
from pinky_wall_parking_perception.line_extraction import (                         # 직선 추출
    LineParams, REAL_NEAR, extract_lines, scan_to_points)
from pinky_wall_parking_perception.slot_geometry import find_slot_any               # 칸 모양 맞추기

# =========================================================
# ROS 2 노드
# =========================================================
class SlotDetector(Node):
    def __init__(self):
        super().__init__("slot_detector")
        # 파라미터 (m, deg)
        self.base_frame     = self.declare_parameter("base_frame", "base_link").value
        self.slot_depth     = self.declare_parameter("slot_depth", 0.190).value     # 칸 깊이 (옆 벽 길이)
        self.width_min      = self.declare_parameter("width_min", 0.115).value      # 로봇 폭 110 mm + 여유
        self.width_max      = self.declare_parameter("width_max", 0.200).value      # 벽 두께 포함 폭은 제외
        self.angle_tol      = math.radians(self.declare_parameter("angle_tol_deg", 10.0).value)
        self.min_overlap    = self.declare_parameter("min_overlap", 0.04).value     # 옆 벽 겹침 최소
        near                = self.declare_parameter("near_correction", False).value
        self.near           = REAL_NEAR if near else None                           # 실물 근거리 보정
        yaw_offset_deg      = self.declare_parameter("laser_yaw_offset_deg", 0.0).value
        self.yaw_offset     = math.radians(yaw_offset_deg)                          # 실물 라이다 장착 각도 보정
        defaults            = LineParams()
        self.line_params    = LineParams(
            max_gap     = self.declare_parameter("max_gap", defaults.max_gap).value,
            split_dist  = self.declare_parameter("split_dist", defaults.split_dist).value,
            merge_angle = math.radians(self.declare_parameter("merge_angle_deg", 5.0).value),
            merge_dist  = self.declare_parameter("merge_dist", defaults.merge_dist).value,
            min_points  = self.declare_parameter("min_points", defaults.min_points).value,
            min_length  = self.declare_parameter("min_length", defaults.min_length).value)

        self.tf_buffer      = Buffer()
        self.tf_listener    = TransformListener(self.tf_buffer, self)
        self.laser_to_base  = None                                                  # (cos, sin, x, y), 처음 한 번 조회

        self.pose_pub       = self.create_publisher(PoseStamped, "slot/pose", 10)
        self.width_pub      = self.create_publisher(Float32, "slot/width", 10)
        self.marker_pub     = self.create_publisher(MarkerArray, "slot/markers", 10)
        self.create_subscription(LaserScan, "scan", self.on_scan, qos_profile_sensor_data)
        self.get_logger().info(f"slot_detector ready: depth {self.slot_depth} m, "
                               f"width {self.width_min}~{self.width_max} m, near_correction {near}, "
                               f"laser_yaw_offset {yaw_offset_deg} deg")

    # ---- 라이다 -> base_link 변환 (고정) ----------------------
    def lookup_laser(self, frame):
        try:
            t = self.tf_buffer.lookup_transform(self.base_frame, frame, Time()).transform
        except TransformException as error:
            self.get_logger().warn(f"tf {self.base_frame} <- {frame}: {error}",
                                   throttle_duration_sec=2.0)
            return None
        q   = t.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        yaw += self.yaw_offset                                                      # tf yaw + 장착 보정
        return (math.cos(yaw), math.sin(yaw), t.translation.x, t.translation.y)

    # ---- 스캔 처리 --------------------------------------------
    def on_scan(self, msg):
        if self.laser_to_base is None:
            self.laser_to_base = self.lookup_laser(msg.header.frame_id)
            if self.laser_to_base is None:
                return
        c, s, tx, ty = self.laser_to_base
        points  = scan_to_points(msg.ranges, msg.angle_min, msg.angle_increment,
                                 msg.range_min, msg.range_max, near=self.near)      # 라이다 좌표
        points  = np.column_stack((c * points[:, 0] - s * points[:, 1] + tx,
                                   s * points[:, 0] + c * points[:, 1] + ty))       # base_link 좌표 (NaN 유지)
        segments = extract_lines(points, self.line_params)
        slot    = find_slot_any(segments, self.slot_depth, self.width_min, self.width_max,
                                self.angle_tol, self.min_overlap)
        self.publish_markers(msg.header.stamp, segments, slot)
        if slot is None:
            return
        center, outward, width, _, _ = slot
        pose = PoseStamped()
        pose.header.stamp, pose.header.frame_id = msg.header.stamp, self.base_frame
        pose.pose.position.x, pose.pose.position.y = float(center[0]), float(center[1])
        yaw = math.atan2(outward[1], outward[0])                                    # 칸 안 -> 입구 방향
        pose.pose.orientation.z, pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        self.pose_pub.publish(pose)
        self.width_pub.publish(Float32(data=float(width)))

    # ---- rviz2 표시 --------------------------------------------
    def publish_markers(self, stamp, segments, slot):
        lines = Marker(type=Marker.LINE_LIST, action=Marker.ADD, ns="segments", id=0)
        lines.header.stamp, lines.header.frame_id = stamp, self.base_frame
        lines.scale.x = 0.005
        # 파랑: 모든 직선
        lines.color.r, lines.color.g, lines.color.b, lines.color.a = 0.2, 0.6, 1.0, 1.0
        for seg in segments:
            lines.points += [Point(x=float(seg.start[0]), y=float(seg.start[1])),
                             Point(x=float(seg.end[0]), y=float(seg.end[1]))]
        arrow = Marker(type=Marker.ARROW, ns="slot", id=1)
        arrow.header = lines.header
        if slot is None:
            arrow.action = Marker.DELETE
        else:
            center, outward, _, _, _ = slot
            arrow.action = Marker.ADD
            arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.01, 0.02, 0.02
            arrow.color.r, arrow.color.g, arrow.color.a = 1.0, 0.3, 1.0             # 주황: 입구 방향
            tip = center + 0.1 * outward
            arrow.points = [Point(x=float(center[0]), y=float(center[1])),
                            Point(x=float(tip[0]), y=float(tip[1]))]
        self.marker_pub.publish(MarkerArray(markers=[lines, arrow]))

# =========================================================
# 실행
# =========================================================
def main(args=None):
    rclpy.init(args=args)
    node = SlotDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == "__main__":
    main()
