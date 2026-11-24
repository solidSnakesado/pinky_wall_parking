"""src/pinky_wall_parking_front/pinky_wall_parking_front/front_parking_node.py - 전방 주차 (ROS 2 노드)

역할: 카메라 칸 특징과 odom (실물은 초음파도) 으로 front_control 상태 기계를 돌려 속도 명령을 낸다.
- 구독 : slot_vision/features (pinky_wall_parking_interfaces/SlotFeatures)
         odom                 (nav_msgs/Odometry)
         us_sensor/range      (sensor_msgs/Range, use_ultrasonic 이 true 일 때만)
- 발행 : cmd_vel              (geometry_msgs/Twist)
         front_parking/state  (std_msgs/String, 상태 이름 또는 정지 이유)
- 서비스: front_parking/start (std_srvs/Trigger) 를 받아야 움직이기 시작한다.
방법: 특징 메시지마다 step 을 부른다 (모서리 둘 + 소실점이 모두 있을 때만 특징으로 넘김).
      특징 메시지가 lost_sec 넘게 안 오면 타이머가 비상 정지한다. 끝나면 0 속도를 여러 번 보낸다.
실행: ros2 run pinky_wall_parking_front front_parking_node --ros-args -p use_ultrasonic:=false
      ros2 service call /front_parking/start std_srvs/srv/Trigger
"""

# --- 표준 라이브러리 ---------------------------------------------------------------
from dataclasses import fields                                                      # ControlParams 항목 목록
import math                                                                         # yaw 계산

# --- ROS 2 -------------------------------------------------------------------------
from geometry_msgs.msg import Twist                                                 # 속도 명령
from nav_msgs.msg import Odometry                                                   # odom
from pinky_wall_parking_interfaces.msg import SlotFeatures                          # 칸 특징
import rclpy                                                                        # ROS 2 파이썬
from rclpy.node import Node                                                         # 노드 기본 클래스
from rclpy.signals import SignalHandlerOptions                                      # Ctrl+C 직접 처리
from sensor_msgs.msg import Range                                                   # 초음파
from std_msgs.msg import String                                                     # 상태
from std_srvs.srv import Trigger                                                    # 시작 요청

# --- 이 패키지 ---------------------------------------------------------------------
from pinky_wall_parking_front.front_control import ControlParams                    # 제어 파라미터
from pinky_wall_parking_front.front_control import FrontParking                     # 상태 기계
from pinky_wall_parking_front.front_control import Phase                            # 상태 이름


# ==================================================================================
# 메시지 -> 라이브러리 입력
# ==================================================================================
def to_feat(msg):
    if not (msg.left_found and msg.right_found and msg.vp_found):
        return None                                                                 # 칸 일부가 안 보임
    return msg.vp.x, msg.center_x, msg.right_x - msg.left_x


def to_pose(msg):
    q   = msg.pose.pose.orientation
    yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    return msg.pose.pose.position.x, msg.pose.pose.position.y, yaw


# ==================================================================================
# ROS 2 노드
# ==================================================================================
class FrontParkingNode(Node):

    def __init__(self):
        super().__init__("front_parking")
        defaults       = ControlParams()
        values         = {}
        for item in fields(ControlParams):                                          # 파라미터 이름 = 항목 이름
            values[item.name] = self.declare_parameter(item.name, getattr(defaults, item.name)).value
        self.params    = ControlParams(**values)
        self.use_us    = self.declare_parameter("use_ultrasonic", False).value
        self.us_stale  = self.declare_parameter("us_stale_sec", 0.5).value          # 이보다 오래된 초음파는 안 씀
        self.fp        = None                                                       # 상태 기계 (start 전에는 None)
        self.odom      = None                                                       # 최신 (x, y, yaw)
        self.us        = None                                                       # 최신 (시각, 거리 m)
        self.last_feat = 0.0                                                        # 마지막 특징 메시지 시각

        self.cmd_pub   = self.create_publisher(Twist, "cmd_vel", 10)
        self.state_pub = self.create_publisher(String, "front_parking/state", 10)
        self.create_subscription(SlotFeatures, "slot_vision/features", self.on_features, 10)
        self.create_subscription(Odometry, "odom", self.on_odom, 10)
        if self.use_us:
            self.create_subscription(Range, "us_sensor/range", self.on_range, 10)
        self.create_service(Trigger, "front_parking/start", self.on_start)
        self.create_timer(0.05, self.on_timer)                                      # 특징 끊김 감시 (20 Hz)
        self.get_logger().info(f"front_parking ready: use_ultrasonic {self.use_us}")

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------------------------
    # 입력
    # ------------------------------------------------------------------------------
    def on_odom(self, msg):
        self.odom = to_pose(msg)

    def on_range(self, msg):
        self.us = (self.now(), msg.range)

    def on_start(self, request, response):
        if self.odom is None:
            response.success, response.message = False, "no odom"
            return response
        self.fp        = FrontParking(self.params)
        self.last_feat = self.now()
        response.success, response.message = True, "started"
        self.get_logger().info("front parking started")
        return response

    # ------------------------------------------------------------------------------
    # 한 주기
    # ------------------------------------------------------------------------------
    def on_features(self, msg):
        t = self.now()
        self.last_feat = t
        if self.fp is None or self.fp.phase in (Phase.STOP, Phase.ABORT):
            return
        us_m = None
        if self.use_us and self.us is not None and t - self.us[0] <= self.us_stale:
            us_m = self.us[1]
        v, w = self.fp.step(t, to_feat(msg), self.odom, us_m)
        self.publish(v, w)

    def on_timer(self):
        if self.fp is None or self.fp.phase in (Phase.STOP, Phase.ABORT):
            return
        if self.now() - self.last_feat > self.params.lost_sec:
            self.fp.finish(Phase.ABORT, "no feature messages")                      # 카메라 또는 vision 멈춤
            self.publish(0.0, 0.0)

    def publish(self, v, w):
        cmd = Twist()
        cmd.linear.x, cmd.angular.z = v, w
        done = self.fp.phase in (Phase.STOP, Phase.ABORT)
        for _ in range(10 if done else 1):                                          # 끝나면 0 속도를 여러 번
            self.cmd_pub.publish(cmd)
        self.state_pub.publish(String(data=self.fp.reason or self.fp.phase.value))
        if done:
            self.get_logger().info(f"front parking {self.fp.phase.value}: {self.fp.reason}")

    def halt(self):
        for _ in range(10):
            self.cmd_pub.publish(Twist())


# ==================================================================================
# 실행
# ==================================================================================
def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)           # Ctrl+C 뒤에도 발행 가능
    node = FrontParkingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.halt()                                                                 # Ctrl+C 에도 정지 명령
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
