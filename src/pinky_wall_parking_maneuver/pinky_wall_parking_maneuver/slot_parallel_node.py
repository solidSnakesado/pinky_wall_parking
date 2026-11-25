"""src/pinky_wall_parking_maneuver/pinky_wall_parking_maneuver/slot_parallel_node.py - 라인 평행 칸 주차 (ROS 2 노드)

역할: 정지 상태에서 라이다로 앞뒤 칸의 핑키를 찾고, 좁은 평행 칸 경로 (slot_parallel) 를 만든 뒤 차량식으로 따라가 주차한다.
- 구독 : odom (nav_msgs/Odometry), scan (sensor_msgs/LaserScan), tf (base_footprint <- 라이다)
- 발행 : cmd_vel, slot_parallel/state (String), slot_parallel/path (Path, odom). 서비스: slot_parallel/start (Trigger)
방법: OBSERVE (observe_time 동안 정지, 스캔 -> odom 점 -> 퍼크 원 맞춤 -> 중앙값 누적)
      -> 이웃 칸 중심 assoc 안의 퍼크 = 이웃 핑키 (칸 방향 OBB, 방향 앎) -> PLAN (별도 스레드, 로봇 정지)
      -> DRIVE (PathFollower 20 Hz) -> 허용 범위 밖이면 ALIGN (칸 안 정렬 왕복, 최대 max_align 번) -> DONE.
      칸 위치는 파라미터 (odom = 출발 자세 기준, 라인 인식 전 단계). 진입 방식 = style (multi / video), entry_gear.
      안전 정지: 주행 중 스캔 점이 내 차체 + stop_margin 안에 들어오면 STOP. odom 이 odom_timeout 넘게 끊겨도 STOP.
실행: ros2 launch pinky_wall_parking_maneuver slot_parallel_sim.launch.py 뒤 /slot_parallel/start (Trigger) 호출
"""

# --- 표준 라이브러리 ---------------------------------------------------------------
import math                                                                         # 삼각 함수
import threading                                                                    # 계획 스레드

# --- ROS 2, 외부 라이브러리 -------------------------------------------------------
from geometry_msgs.msg import Twist                                                 # 속도 명령
from nav_msgs.msg import Odometry                                                   # odom
from nav_msgs.msg import Path                                                       # 경로 표시
import numpy as np                                                                  # 점 배열
import rclpy                                                                        # ROS 2 파이썬
from rclpy.node import Node                                                         # 노드 기본 클래스
from rclpy.qos import qos_profile_sensor_data                                       # 센서 QoS
from rclpy.signals import SignalHandlerOptions                                      # Ctrl+C 직접 처리
from rclpy.time import Time                                                         # tf 최신
from sensor_msgs.msg import LaserScan                                               # 라이다
from std_msgs.msg import String                                                     # 상태
from std_srvs.srv import Trigger                                                    # 시작 요청
from tf2_ros import Buffer                                                          # tf 버퍼
from tf2_ros import TransformListener                                               # tf 수신

# --- 이 패키지 ---------------------------------------------------------------------
from pinky_wall_parking_maneuver.car_path import CarParams                          # 차량 파라미터
from pinky_wall_parking_maneuver.car_path import PathFollower                       # 경로 추종
from pinky_wall_parking_maneuver.obstacles import Obb                               # 상자
from pinky_wall_parking_maneuver.obstacles import ObsParams                         # 장애물 파라미터
from pinky_wall_parking_maneuver.obstacles import points_in_obb                     # 안전 정지 검사
from pinky_wall_parking_maneuver.obstacles import PuckTrack                         # 퍼크 누적
from pinky_wall_parking_maneuver.obstacles import robot_obb                         # 내 차체
from pinky_wall_parking_maneuver.obstacles import split_scan                        # 퍼크 / 나머지
from pinky_wall_parking_maneuver.ros_helpers import compose                         # 자세 곱
from pinky_wall_parking_maneuver.ros_helpers import path_msg                        # 경로 메시지
from pinky_wall_parking_maneuver.ros_helpers import scan_points                     # 스캔 점
from pinky_wall_parking_maneuver.slot_parallel import plan_align                    # 마무리 정렬
from pinky_wall_parking_maneuver.slot_parallel import plan_slot_parallel            # 칸 경로
from pinky_wall_parking_maneuver.slot_parallel import Slot                          # 칸
from pinky_wall_parking_maneuver.slot_parallel import SlotParams                    # 칸 파라미터
from pinky_wall_parking_maneuver.t_parking_node import declare                      # dataclass 파라미터
from pinky_wall_parking_maneuver.t_parking_node import yaw_of                       # 쿼터니언 -> yaw


class SlotParallelNode(Node):

    def __init__(self):
        super().__init__("slot_parallel")
        self.cp, self.sp, self.op = declare(self, CarParams), declare(self, SlotParams), ObsParams()
        self.p = p = {n: self.declare_parameter(n, v).value for n, v in (
            ("slot_x", 0.600), ("slot_y", -0.200), ("slot_yaw_deg", 0.0), ("slot_side", 1),
            ("bounds", [-0.40, 1.40, -0.32, 0.50]), ("observe_time", 2.0), ("assoc", 0.08),
            ("stop_margin", 0.003), ("odom_timeout", 0.5), ("max_align", 3))}
        self.slot   = Slot(p["slot_x"], p["slot_y"], math.radians(p["slot_yaw_deg"]), int(p["slot_side"]))
        self.phase  = "IDLE"                                                        # 상태
        self.odom, self.odom_t, self.pts, self.lidar = None, 0.0, None, None        # 최신 odom, 스캔 점, 라이다 tf
        self.track, self.t0, self.gears, self.follow = None, 0.0, None, None        # 퍼크, 관측 시작, 경로, 추종
        self.nbs, self.aligns = [], 0                                               # 이웃 상자, 정렬 횟수
        self.buf = Buffer()                                                         # tf
        self.tfl = TransformListener(self.buf, self)
        self.cmd_pub   = self.create_publisher(Twist, "cmd_vel", 10)
        self.state_pub = self.create_publisher(String, "slot_parallel/state", 10)
        self.path_pub  = self.create_publisher(Path, "slot_parallel/path", 10)
        self.create_subscription(Odometry, "odom", self.on_odom, 10)
        self.create_subscription(LaserScan, "scan", self.on_scan, qos_profile_sensor_data)
        self.create_service(Trigger, "slot_parallel/start", self.on_start)
        self.create_timer(0.05, self.on_timer)                                      # 20 Hz
        self.get_logger().info(f"slot_parallel ready: slot ({self.slot.cx:.3f}, {self.slot.cy:.3f}, "
                               f"{p['slot_yaw_deg']:.1f} deg, side {self.slot.side}), length {self.sp.length:.3f}, "
                               f"style {self.sp.style}, entry_gear {self.sp.entry_gear}")

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_odom(self, msg):
        q = msg.pose.pose
        self.odom, self.odom_t = (q.position.x, q.position.y, yaw_of(q.orientation)), self.now()

    def on_scan(self, msg):
        if self.odom is None:
            return
        if self.lidar is None and self.buf.can_transform("base_footprint", msg.header.frame_id, Time()):
            t = self.buf.lookup_transform("base_footprint", msg.header.frame_id, Time()).transform
            self.lidar = (t.translation.x, t.translation.y, yaw_of(t.rotation))     # base <- 라이다
        if self.lidar is None:
            return
        sensor = compose(self.odom, self.lidar)                                     # 라이다 odom 자세
        self.pts = scan_points(msg, sensor)                                         # 스캔 점 (odom)
        if self.phase == "OBSERVE":
            self.track.update(split_scan(self.pts, sensor[:2], self.op)[0])

    def on_start(self, request, response):
        if self.odom is None or self.phase in ("OBSERVE", "PLAN", "DRIVE", "ALIGN"):
            response.success, response.message = False, f"busy or no odom ({self.phase})"
            return response
        self.track, self.t0, self.aligns = PuckTrack(self.op), self.now(), 0
        self.set_phase("OBSERVE")
        response.success, response.message = True, "started"
        return response

    def neighbors(self):
        """Return neighbor boxes (slot yaw) for pucks near the front / rear slot centers."""
        hl, hw = (self.op.body_x[1] - self.op.body_x[0]) / 2, self.op.body_w        # 핑키 반치수
        pucks, out = self.track.pucks(), []                                         # 누적 퍼크, 결과
        for sx in (-self.sp.length, self.sp.length):
            ex, ey, ea = self.slot.to_world((sx, 0.0, 0.0))                         # 이웃 칸 중심
            d = [math.hypot(c[0] - ex, c[1] - ey) for c in pucks]                   # 퍼크까지 거리
            if d and min(d) <= self.p["assoc"]:
                c = pucks[int(np.argmin(d))]
                out.append(Obb(float(c[0]), float(c[1]), ea, hl, hw))
            self.get_logger().info(f"slot ({ex:.3f}, {ey:.3f}): " + (f"neighbor {min(d) * 1000:.1f} mm away"
                                   if d and min(d) <= self.p["assoc"] else "no neighbor"))
        return out

    def plan(self, start, nbs):
        t, self.nbs = self.now(), nbs                                               # 계획 시작, 이웃
        self.gears = plan_slot_parallel(start, self.slot, nbs, tuple(self.p["bounds"]), self.cp, self.op, self.sp)
        self.get_logger().info(f"plan {'ok' if self.gears else 'failed'} in {self.now() - t:.1f} s, "
                               f"gears {len(self.gears) if self.gears else 0}")
        self.set_phase("READY" if self.gears else "FAIL")

    def on_timer(self):
        if self.phase in ("OBSERVE", "PLAN", "DRIVE", "ALIGN") and self.now() - self.odom_t > self.p["odom_timeout"]:
            return self.stop("STOP odom timeout")
        if self.phase == "OBSERVE" and self.now() - self.t0 >= self.p["observe_time"]:
            self.set_phase("PLAN")
            threading.Thread(target=self.plan, args=(self.odom, self.neighbors()), daemon=True).start()
        elif self.phase == "PLAN":
            self.cmd_pub.publish(Twist())                                           # 계획 중 정지
        elif self.phase == "READY":
            self.follow = PathFollower(self.gears, self.cp)
            self.publish_path()
            self.set_phase("DRIVE")
        elif self.phase in ("DRIVE", "ALIGN"):
            me = robot_obb(self.odom, self.op, self.p["stop_margin"])               # 내 차체 + 여유
            if self.pts is not None and points_in_obb(me, self.pts):
                return self.stop("STOP scan point inside the body")
            v, w = self.follow.step(self.odom)
            cmd = Twist()
            cmd.linear.x, cmd.angular.z = v, w
            self.cmd_pub.publish(cmd)
            if self.follow.done:
                self.finish_drive()

    def finish_drive(self):
        x, y, a = self.slot.to_slot(self.odom)                                      # 칸 좌표 base 자세
        x += (self.op.body_x[0] + self.op.body_x[1]) / 2                            # 차체 중심 (근사)
        self.get_logger().info(f"{self.phase} end (odom): x {x * 1000:+.1f} y {y * 1000:+.1f} mm, "
                               f"yaw {math.degrees(a):+.2f} deg")
        fix = plan_align(self.odom, self.slot, self.nbs, self.cp, self.op, self.sp)
        if not fix or self.aligns >= self.p["max_align"]:
            return self.stop("DONE" if fix == [] else f"DONE ({'align limit' if fix else 'align not found'})")
        self.aligns, self.gears = self.aligns + 1, fix
        self.follow = PathFollower(fix, self.cp)
        self.publish_path()
        self.set_phase("ALIGN")

    def set_phase(self, phase):
        self.phase = phase
        self.state_pub.publish(String(data=phase))
        self.get_logger().info(f"slot parallel {phase}")

    def stop(self, reason):
        self.halt(), self.set_phase(reason)

    def publish_path(self):
        self.path_pub.publish(path_msg(self.gears, "odom", self.get_clock().now().to_msg()))

    def halt(self):
        for _ in range(10):
            self.cmd_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)           # Ctrl+C 뒤에도 발행 가능
    node = SlotParallelNode()
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
