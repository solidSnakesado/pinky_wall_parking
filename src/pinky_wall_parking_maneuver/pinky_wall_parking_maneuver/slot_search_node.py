"""src/pinky_wall_parking_maneuver/pinky_wall_parking_maneuver/slot_search_node.py - 칸 위치 모르는 상태에서 평행 칸 찾기 (ROS 2 노드)

역할: 차선을 따라 저속 직진하며 카메라 칸 선 지도 (line_map) 와 라이다 앞뒤 차 (원통) 를 쌓고, slot_fusion 으로 빈 칸을
      확정하면 정지해서 slot_parallel/goal 로 주차 노드 (slot_parallel_node) 에 넘긴다.
- 구독 : 영상 (image_topic, Image 또는 compressed = true 면 CompressedImage, reliable depth 1), odom, scan, tf
- 발행 : cmd_vel, slot_parallel/goal (PoseStamped, odom, 칸 중심·방향), slot_search/state (String). 서비스: slot_search/start
방법: CRUISE: 출발 방향 유지 (w = -k_yaw x 방향 오차) 로 cruise_v 직진, 영상은 image_period 마다 한 장
        - 라이다: 스캔 -> odom 점 -> 원통 원 맞춤 -> PuckTrack (odom 누적) = 이웃 차 중심
        - 카메라: LineMap.add (영상, odom, 이웃 차 상자 = 로봇 기준, 로봇 진행 방향과 나란하다고 봄) -> 지도 선
        - fuse (지도 선, 이웃 차 중심, odom) -> 첫 후보가 need 출처 (기본 cam+lidar) 이고
          최근 confirm_k 개 후보가 모두 평균에서 confirm_tol 안이면 확정 (밀려가는 창, 장마다 흔들림을 평균으로 줄임)
      STOP: 정지, settle 초 뒤 그 confirm_k 개의 중심·방향 평균을 goal 로 한 번 발행 -> DONE. max_dist 를 가도 못 찾으면 FAIL (정지)
      칸 side 는 주차 노드 slot_side 와 같아야 함 (다르면 경고). 시뮬 = image_topic camera/image_raw, compressed false
      slot_length = 표준 칸 길이 (사전 지식, 주차 노드 length 와 같게)
실행: ros2 run pinky_wall_parking_maneuver slot_search_node --ros-args -p use_sim_time:=true
      ros2 service call /slot_search/start std_srvs/srv/Trigger
"""

# --- 표준 라이브러리 ---------------------------------------------------------------
import math                                                                         # 각도

# --- ROS 2, 외부 라이브러리 -------------------------------------------------------
from cv_bridge import CvBridge                                                      # 영상 변환
from cv_bridge import CvBridgeError                                                 # 영상 형식 오류
from geometry_msgs.msg import PoseStamped                                           # 칸 goal
from geometry_msgs.msg import Twist                                                 # 속도 명령
from nav_msgs.msg import Odometry                                                   # odom
import numpy as np                                                                  # 평균
from pinky_wall_parking_perception.line_extraction import REAL_NEAR                 # 실물 근거리 보정표
from pinky_wall_parking_vision.line_map import LineMap                              # 칸 선 지도
import rclpy                                                                        # ROS 2 파이썬
from rclpy.node import Node                                                         # 노드 기본 클래스
from rclpy.qos import qos_profile_sensor_data                                       # 센서 QoS
from rclpy.qos import QoSProfile                                                    # 영상 QoS
from rclpy.qos import ReliabilityPolicy                                             # 영상 QoS
from rclpy.time import Time                                                         # tf 최신
from sensor_msgs.msg import CompressedImage                                         # 압축 영상
from sensor_msgs.msg import Image                                                   # 원본 영상
from sensor_msgs.msg import LaserScan                                               # 라이다
from std_msgs.msg import String                                                     # 상태
from std_srvs.srv import Trigger                                                    # 시작
from tf2_ros import Buffer                                                          # tf 버퍼
from tf2_ros import TransformListener                                               # tf 수신

# --- 이 패키지 ---------------------------------------------------------------------
from pinky_wall_parking_maneuver.obstacles import body_half                         # 차체 치수
from pinky_wall_parking_maneuver.obstacles import ObsParams                         # 원통·차체 파라미터
from pinky_wall_parking_maneuver.obstacles import PuckTrack                         # 원통 누적
from pinky_wall_parking_maneuver.obstacles import split_scan                        # 원통 찾기
from pinky_wall_parking_maneuver.ros_helpers import compose                         # 자세 곱
from pinky_wall_parking_maneuver.ros_helpers import scan_points                     # 스캔 -> 점
from pinky_wall_parking_maneuver.ros_helpers import twist                           # 속도 메시지
from pinky_wall_parking_maneuver.slot_fusion import fuse                            # 칸 융합
from pinky_wall_parking_maneuver.slot_fusion import FusionParams                    # 칸 규격
from pinky_wall_parking_maneuver.t_parking_node import yaw_of                       # 쿼터니언 -> yaw


def local(pose, pt):
    """Point pt (odom) in the robot frame at pose."""
    dx, dy, c, s = pt[0] - pose[0], pt[1] - pose[1], math.cos(pose[2]), math.sin(pose[2])
    return c * dx + s * dy, -s * dx + c * dy


class SlotSearchNode(Node):

    def __init__(self):
        super().__init__("slot_search")
        self.p = p = {n: self.declare_parameter(n, v).value for n, v in (
            ("image_topic", "camera/image_raw"), ("compressed", False), ("image_period", 0.2),
            ("cruise_v", 0.03), ("k_yaw", 1.5), ("max_dist", 0.6), ("need", "cam+lidar"),
            ("confirm_k", 6), ("confirm_tol", 0.005), ("slot_side", 1), ("settle", 0.5),
            ("laser_yaw_offset_deg", 0.0), ("near_correction", False), ("slot_length", 0.230))}
        self.op, self.fp, self.bridge = ObsParams(), FusionParams(length=p["slot_length"]), CvBridge()
        self.phase, self.odom, self.lidar, self.start = "IDLE", None, None, None    # 상태, odom, 라이다 tf, 출발
        self.map, self.track, self.hits, self.img_t = None, None, [], -1e9          # 지도, 원통, 연속 후보, 영상 시각
        self.dbg_t = -1e9                                                           # 진단 로그 시각
        self.buf = Buffer()
        self.tfl = TransformListener(self.buf, self)
        self.cmd_pub   = self.create_publisher(Twist, "cmd_vel", 10)
        self.goal_pub  = self.create_publisher(PoseStamped, "slot_parallel/goal", 10)
        self.state_pub = self.create_publisher(String, "slot_search/state", 10)
        self.create_subscription(Odometry, "odom", self.on_odom, 10)
        self.create_subscription(LaserScan, "scan", self.on_scan, qos_profile_sensor_data)
        img_type = CompressedImage if p["compressed"] else Image
        img_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)       # 큰 영상은 best-effort 면 거의 다 잃음
        self.create_subscription(img_type, p["image_topic"], self.on_image, img_qos)
        self.create_service(Trigger, "slot_search/start", self.on_start)
        self.create_timer(0.05, self.on_timer)                                      # 20 Hz
        self.get_logger().info(f"slot_search ready: image {p['image_topic']} (compressed {p['compressed']}), "
                               f"cruise {p['cruise_v']} m/s, max {p['max_dist']} m, need {p['need']}, "
                               f"confirm {p['confirm_k']} x {p['confirm_tol'] * 1000:.0f} mm")

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def set_phase(self, phase):
        self.phase = phase
        self.state_pub.publish(String(data=phase))
        self.get_logger().info(f"slot search {phase}")

    def on_odom(self, msg):
        q = msg.pose.pose
        self.odom = (q.position.x, q.position.y, yaw_of(q.orientation))

    def on_start(self, request, response):
        if self.odom is None or self.phase == "CRUISE":
            response.success, response.message = False, f"busy or no odom ({self.phase})"
            return response
        self.start, self.map, self.track, self.hits = self.odom, LineMap(), PuckTrack(self.op), []
        self.set_phase("CRUISE")
        response.success, response.message = True, "started"
        return response

    def on_scan(self, msg):
        if self.odom is None or self.phase != "CRUISE":
            return
        if self.lidar is None and self.buf.can_transform("base_footprint", msg.header.frame_id, Time()):
            t = self.buf.lookup_transform("base_footprint", msg.header.frame_id, Time()).transform
            self.lidar = (t.translation.x, t.translation.y,                         # base <- 라이다 + 장착 보정
                          yaw_of(t.rotation) + math.radians(self.p["laser_yaw_offset_deg"]))
        if self.lidar is None:
            return
        sensor = compose(self.odom, self.lidar)
        pts = scan_points(msg, sensor, REAL_NEAR if self.p["near_correction"] else None)
        self.track.update(split_scan(pts, sensor[:2], self.op)[0])

    def on_image(self, msg):
        if self.phase != "CRUISE" or self.odom is None or self.now() - self.img_t < self.p["image_period"]:
            return
        self.img_t, odom = self.now(), self.odom
        try:
            rgb = (self.bridge.compressed_imgmsg_to_cv2(msg, "rgb8") if self.p["compressed"]
                   else self.bridge.imgmsg_to_cv2(msg, "rgb8"))
        except CvBridgeError as exc:                                                # 한 장 오류로 멈추지 않게
            return self.get_logger().warn(f"image: {exc}")
        hl, hw, _ = body_half(self.op)
        cars = self.track.pucks()                                                   # 이웃 차 중심 (odom)
        boxes = [(*local(odom, c), 0.0, hl, hw) for c in cars]                      # 로봇 기준 상자 (방향 = 진행 방향)
        self.map.add(rgb, odom, boxes)
        lines, diag = self.map.lines(), {}
        self.check(fuse(lines, cars, odom, self.fp, diag=diag), len(lines), cars, diag)

    def check(self, cands, n_lines, cars, diag):
        best = cands[0] if cands else None
        if best is None or best.src["check"] != self.p["need"]:
            if self.now() - self.dbg_t >= 1.0:                                      # 진단: 확정에 못 쓰는 경우 (1 초 1 번)
                self.dbg_t = self.now()
                what = (f"best ({best.cx:.3f}, {best.cy:.3f}) {best.src['check']}, {len(cands)} cands" if best
                        else "no candidate (" + " ".join(f"{k} {v}" for k, v in diag.items()) + ")")
                pucks = ", ".join(f"({c[0]:.3f}, {c[1]:.3f})" for c in cars)
                self.get_logger().info(f"skip: {what} | lines {n_lines} | cars [{pucks}]")
            return
        self.hits = (self.hits + [best])[-self.p["confirm_k"]:]                     # 최근 confirm_k 개
        mx, my = np.mean([h.cx for h in self.hits]), np.mean([h.cy for h in self.hits])
        spread = max(math.hypot(h.cx - mx, h.cy - my) for h in self.hits)           # 평균에서 가장 먼 후보
        self.get_logger().info(f"candidate ({best.cx:.3f}, {best.cy:.3f}) {best.src} "
                               f"[{len(self.hits)}, spread {spread * 1000:.1f} mm]")
        if len(self.hits) >= self.p["confirm_k"] and spread <= self.p["confirm_tol"] and self.phase == "CRUISE":
            self.stop_t = self.now()
            self.set_phase("STOP")

    def on_timer(self):
        if self.phase == "CRUISE":
            if math.hypot(self.odom[0] - self.start[0], self.odom[1] - self.start[1]) > self.p["max_dist"]:
                self.cmd_pub.publish(twist(0.0, 0.0))
                return self.set_phase("FAIL")
            err = math.remainder(self.odom[2] - self.start[2], 2 * math.pi)         # 출발 방향 유지
            self.cmd_pub.publish(twist(self.p["cruise_v"], -self.p["k_yaw"] * err))
        elif self.phase == "STOP":
            self.cmd_pub.publish(twist(0.0, 0.0))
            if self.now() - self.stop_t >= self.p["settle"]:                        # 멈춘 뒤 넘김 (주차 노드 관측)
                self.handoff()

    def handoff(self):
        h = self.hits
        cx, cy = float(np.mean([s.cx for s in h])), float(np.mean([s.cy for s in h]))
        yaw = math.atan2(np.mean([math.sin(s.yaw) for s in h]), np.mean([math.cos(s.yaw) for s in h]))
        side = h[-1].side
        if side != self.p["slot_side"]:
            self.get_logger().warn(f"slot side {side} != slot_side {self.p['slot_side']} (parking node)")
        msg = PoseStamped()
        msg.header.frame_id, msg.header.stamp = "odom", self.get_clock().now().to_msg()
        msg.pose.position.x, msg.pose.position.y = cx, cy
        msg.pose.orientation.z, msg.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        self.goal_pub.publish(msg)
        self.get_logger().info(f"goal slot ({cx:.3f}, {cy:.3f}, {math.degrees(yaw):.1f} deg, side {side}), "
                               f"length {h[-1].length:.3f}, width {h[-1].width:.3f}, {h[-1].src}")
        self.set_phase("DONE")


def main(args=None):
    rclpy.init(args=args)
    node = SlotSearchNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
