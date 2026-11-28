"""src/pinky_wall_parking_maneuver/pinky_wall_parking_maneuver/t_parking_node.py - T자 후면 주차 (ROS 2 노드)

역할: 차선을 따라 전진하며 라이다로 옆 칸을 찾고, 차량식 경로로 칸 앞 정렬 자세까지 간 뒤 진입을 넘긴다
      (파라미터 front: false = 후면, Docking Server / true = 전면, 전방 주차 노드).
- 구독 : odom       (nav_msgs/Odometry)
         slot/pose  (geometry_msgs/PoseStamped, base_link, x = 칸 밖 방향. slot_detector)
- 발행 : cmd_vel           (geometry_msgs/Twist)
         t_parking/state   (std_msgs/String, 상태 이름 또는 끝난 이유)
         t_parking/path    (nav_msgs/Path, odom, 계획 경로 표시용)
- 서비스: t_parking/start (std_srvs/Trigger) 로 시작. 정렬 후 넘기기는 handoff.py (파라미터 dock = 넘길지 여부)
방법: 칸 인식 (base_link) 을 그 순간의 odom 자세로 odom 에 옮겨 TParking 에 넘긴다. 20 Hz 로 step.
      odom 이 odom_timeout 넘게 안 오면 중단. 끝나면 0 속도를 여러 번 보낸다 (Ctrl+C 포함).
실행: ros2 launch pinky_wall_parking_maneuver t_parking_sim.launch.py
      ros2 service call /t_parking/start std_srvs/srv/Trigger
"""

# --- 표준 라이브러리 ---------------------------------------------------------------
from dataclasses import fields                                                      # 파라미터 항목 목록
import math                                                                         # yaw 계산

# --- ROS 2 -------------------------------------------------------------------------
from geometry_msgs.msg import PoseStamped                                           # 칸 입구 / 경로 점
from geometry_msgs.msg import Twist                                                 # 속도 명령
from nav_msgs.msg import Odometry                                                   # odom
from nav_msgs.msg import Path                                                       # 계획 경로 표시
import rclpy                                                                        # ROS 2 파이썬
from rclpy.node import Node                                                         # 노드 기본 클래스
from rclpy.signals import SignalHandlerOptions                                      # Ctrl+C 직접 처리
from std_msgs.msg import String                                                     # 상태
from std_srvs.srv import Trigger                                                    # 시작 요청

# --- 이 패키지 ---------------------------------------------------------------------
from pinky_wall_parking_maneuver.car_path import CarParams                          # 차량 파라미터
from pinky_wall_parking_maneuver.handoff import Handoff                             # 진입 제어로 넘기기
from pinky_wall_parking_maneuver.ros_helpers import spin                            # Ctrl+C 처리 spin
from pinky_wall_parking_maneuver.t_parking import Phase                             # 상태 이름
from pinky_wall_parking_maneuver.t_parking import TParams                           # T자 파라미터
from pinky_wall_parking_maneuver.t_parking import TParking                          # T자 상태 기계


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def to_odom(robot, local):
    """로봇 odom 자세 robot 에서 본 local (x, y, yaw) 를 odom 으로."""
    rx, ry, ryaw = robot
    lx, ly, lyaw = local
    c, s = math.cos(ryaw), math.sin(ryaw)
    return rx + c * lx - s * ly, ry + s * lx + c * ly, math.atan2(math.sin(ryaw + lyaw), math.cos(ryaw + lyaw))


def declare(node, cls):
    """ROS 파라미터를 dataclass 항목과 같은 이름으로 선언하고, 값을 채운 객체를 돌려줌."""
    base = cls()
    return cls(**{f.name: node.declare_parameter(f.name, getattr(base, f.name)).value for f in fields(cls)})


# ==================================================================================
# ROS 2 노드
# ==================================================================================
class TParkingNode(Node):

    def __init__(self):
        super().__init__("t_parking")
        self.cp        = declare(self, CarParams)
        self.tp        = declare(self, TParams)
        self.use_dock  = self.declare_parameter("dock", True).value                 # 정렬 후 진입 제어로 넘김
        self.dock_type = self.declare_parameter("dock_type", "slot_dock").value
        self.timeout   = self.declare_parameter("odom_timeout", 0.5).value          # odom 끊김 한도 (s)
        self.fsm       = None                                                       # 상태 기계 (start 전에는 None)
        self.odom      = None                                                       # 최신 (x, y, yaw)
        self.odom_t    = 0.0                                                        # 최신 odom 받은 시각
        self.det       = None                                                       # 새 칸 인식 (odom), 한 번 쓰면 None
        self.last      = None                                                       # 직전 상태 (바뀔 때만 로그)

        self.cmd_pub   = self.create_publisher(Twist, "cmd_vel", 10)
        self.state_pub = self.create_publisher(String, "t_parking/state", 10)
        self.path_pub  = self.create_publisher(Path, "t_parking/path", 10)
        self.handoff   = Handoff(self, self.tp.front, self.dock_type, self.on_handoff)
        self.create_subscription(Odometry, "odom", self.on_odom, 10)
        self.create_subscription(PoseStamped, "slot/pose", self.on_slot, 10)
        self.create_service(Trigger, "t_parking/start", self.on_start)
        self.create_timer(0.05, self.on_timer)                                      # 20 Hz
        self.get_logger().info(f"t_parking ready: R {self.cp.turn_radius} m, stage {self.tp.stage_dist} m, "
                               f"front {self.tp.front}, handoff {self.use_dock}")

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_odom(self, msg):
        p = msg.pose.pose
        self.odom, self.odom_t = (p.position.x, p.position.y, yaw_of(p.orientation)), self.now()

    def on_slot(self, msg):
        if self.odom is None:
            return
        self.det = to_odom(self.odom, (msg.pose.position.x, msg.pose.position.y, yaw_of(msg.pose.orientation)))

    def on_start(self, request, response):
        if self.odom is None:
            response.success, response.message = False, "no odom"
            return response
        self.fsm, self.last = TParking(self.cp, self.tp), None
        response.success, response.message = True, "started"
        self.get_logger().info("t parking started")
        return response

    def on_timer(self):
        if self.fsm is None or self.fsm.phase in (Phase.DONE, Phase.ABORT):
            return
        if self.now() - self.odom_t > self.timeout:
            self.fsm.phase, self.fsm.reason = Phase.ABORT, "odom timeout"
            self.finish()
            return
        det, self.det = self.det, None
        v, w = self.fsm.step(self.odom, det)
        cmd = Twist()
        cmd.linear.x, cmd.angular.z = v, w
        self.cmd_pub.publish(cmd)
        if self.fsm.phase != self.last:
            self.changed()
        if self.fsm.phase in (Phase.DONE, Phase.ABORT):
            self.finish()

    def changed(self):
        self.last = self.fsm.phase
        self.state_pub.publish(String(data=self.fsm.reason or self.fsm.phase.value))
        self.get_logger().info(f"t parking {self.fsm.phase.value} {self.fsm.reason}")
        if self.fsm.phase == Phase.TRACK:
            sx, sy, syaw = self.fsm.slot
            self.get_logger().info(f"slot locked ({len(self.fsm.dets)} dets): x {sx:.3f} y {sy:.3f} "
                                   f"yaw {math.degrees(syaw):.1f} deg, gears {len(self.fsm.follow.trackers)}")
            self.publish_path()

    def publish_path(self):
        path = Path()
        path.header.frame_id, path.header.stamp = "odom", self.get_clock().now().to_msg()
        for tracker in self.fsm.follow.trackers:
            for x, y, yaw in tracker.pts:
                pose = PoseStamped()
                pose.header = path.header
                pose.pose.position.x, pose.pose.position.y = float(x), float(y)
                pose.pose.orientation.z, pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
                path.poses.append(pose)
        self.path_pub.publish(path)

    def finish(self):
        self.halt()
        self.state_pub.publish(String(data=self.fsm.reason or self.fsm.phase.value))
        if self.fsm.phase == Phase.DONE and self.use_dock:
            self.handoff.start()

    def on_handoff(self, state):
        self.state_pub.publish(String(data=state))
        self.get_logger().info(state)

    def halt(self):
        for _ in range(10):
            self.cmd_pub.publish(Twist())


# ==================================================================================
# 실행
# ==================================================================================
def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)           # Ctrl+C 뒤에도 발행 가능
    node = TParkingNode()
    try:
        spin(node)                                                                  # Ctrl+C 를 0.1 s 안에 받음
    finally:
        node.halt()                                                                 # Ctrl+C 에도 정지 명령
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
