"""src/pinky_wall_parking_sim/pinky_wall_parking_sim/scan_realism_node.py - 시뮬 스캔에 실물 라이다 특성 적용 (ROS 2 노드)

역할: 브리지의 이상적인 스캔을 받아 scan_model 로 실물 RPLIDAR C1 특성 (근거리 오차, 입사각별 점 빠짐, 잡음, 장착 yaw) 을 덧입혀 다시 발행한다.
- 구독 : scan_in (sensor_msgs/LaserScan, 기본 /scan)
- 발행 : scan_out (sensor_msgs/LaserScan, 기본 /scan_real), 헤더·시간·frame 그대로
- 파라미터: ScanModel 필드 이름 그대로 (config/lidar_real_c1.yaml), seed (-1 = 매번 다름)
- 처리 쪽은 scan:=/scan_real 로 remap, 실물과 같게 laser_yaw_offset_deg:=-2.2, near_correction:=true
실행: ros2 run pinky_wall_parking_sim scan_realism_node --ros-args --params-file config/lidar_real_c1.yaml
      (보통은 scene_sim.launch.py lidar_model:=real_c1)
"""

# --- 표준 라이브러리 ---------------------------------------------------------------
from dataclasses import fields                                                      # 파라미터 목록

# --- ROS 2, 외부 라이브러리 -------------------------------------------------------
import numpy as np                                                                  # 난수
import rclpy                                                                        # ROS 2 파이썬
from rclpy.node import Node                                                         # 노드 기본 클래스
from rclpy.qos import qos_profile_sensor_data                                       # 센서 QoS
from sensor_msgs.msg import LaserScan                                               # 라이다

# --- 이 패키지 ---------------------------------------------------------------------
from pinky_wall_parking_sim.scan_model import apply                                 # 특성 적용
from pinky_wall_parking_sim.scan_model import ScanModel                             # 특성 값


class ScanRealismNode(Node):

    def __init__(self):
        super().__init__("scan_realism")
        base = ScanModel()
        vals = {}
        for f in fields(ScanModel):                                                 # 기본값 = ScanModel
            default = getattr(base, f.name)
            if isinstance(default, list):                                           # 빈 목록은 형식 지정 필요
                default = [float(v) for v in default] or [0.0]
            vals[f.name] = self.declare_parameter(f.name, default).value
        if vals["self_points"] == [0.0]:                                            # [0.0] = 몸 점 없음
            vals["self_points"] = []
        self.model = ScanModel(**vals)
        seed       = self.declare_parameter("seed", -1).value
        self.rng   = np.random.default_rng(None if seed < 0 else seed)
        self.pub   = self.create_publisher(LaserScan, "scan_out", qos_profile_sensor_data)
        self.create_subscription(LaserScan, "scan_in", self.on_scan, qos_profile_sensor_data)
        m = self.model
        self.get_logger().info(f"scan_realism: return prob {m.return_prob} at {m.return_deg} deg, "
                               f"sigma {m.sigma * 1000:.1f} mm, "
                               f"near_drop {m.near_drop * 1000:.0f} mm, yaw {m.yaw_offset_deg} deg, "
                               f"near table {len(m.near_r)} pts, self pts {len(m.self_points) // 2}")

    def on_scan(self, msg):
        ranges, angle_min = apply(msg.ranges, msg.angle_min, msg.angle_increment, self.model, self.rng)
        out = LaserScan()
        out.header          = msg.header
        out.angle_min       = float(angle_min)
        out.angle_max       = float(angle_min + msg.angle_increment * (len(ranges) - 1))
        out.angle_increment = msg.angle_increment
        out.time_increment  = msg.time_increment
        out.scan_time       = msg.scan_time
        out.range_min       = msg.range_min
        out.range_max       = msg.range_max
        out.ranges          = [float(v) for v in ranges]
        out.intensities     = list(msg.intensities)
        self.pub.publish(out)


def main():
    rclpy.init()
    node = ScanRealismNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
