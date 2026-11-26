"""src/pinky_wall_parking_maneuver/pinky_wall_parking_maneuver/slot_parallel_eval.py - 시뮬 평행 주차 정답 평가 (ROS 2 노드)

역할: slot_parallel_node 의 odom 기준 결과를 MuJoCo 정답 로봇 위치로 다시 계산해 비교한다 (시뮬 전용, 주행에는 관여 안 함).
- 구독 : ground_truth (geometry_msgs/PoseStamped, world = 로봇 출발 = odom 원점), odom (nav_msgs/Odometry),
         slot_parallel/state (String), tf world -> parked_<i> (세워 둔 핑키 정답)
- 계산 (DRIVE / ALIGN 동안 20 Hz): 칸 선 침범 깊이 (정답 / odom, slot_parallel.line_intrusion),
         이웃 핑키 차체와의 최소 거리 (정답, 겹치면 0), odom 과 정답의 위치·방향 차이 최대
- 출력 : DONE / STOP / FAIL 때 요약 한 줄 (최종 칸 좌표 차체 중심 x, y, yaw 를 정답과 odom 으로)
- 파라미터: slot_x, slot_y, slot_yaw_deg, slot_side (slot_parallel_node 와 같은 값, odom 기준) + SlotParams
실행: ros2 launch pinky_wall_parking_maneuver slot_parallel_sim.launch.py eval:=true ...
"""

# --- 표준 라이브러리 ---------------------------------------------------------------
import math                                                                         # 각도

# --- ROS 2, 외부 라이브러리 -------------------------------------------------------
from geometry_msgs.msg import PoseStamped                                           # 정답 자세
from nav_msgs.msg import Odometry                                                   # odom
import numpy as np                                                                  # 거리 계산
import rclpy                                                                        # ROS 2 파이썬
from rclpy.node import Node                                                         # 노드 기본 클래스
from rclpy.time import Time                                                         # tf 최신
from std_msgs.msg import String                                                     # 상태
from tf2_ros import Buffer                                                          # tf 버퍼
from tf2_ros import TransformListener                                               # tf 수신

# --- 이 패키지 ---------------------------------------------------------------------
from pinky_wall_parking_maneuver.obstacles import obb_overlap                       # 겹침 검사
from pinky_wall_parking_maneuver.obstacles import ObsParams                         # 차체 치수
from pinky_wall_parking_maneuver.obstacles import robot_obb                         # 차체 상자
from pinky_wall_parking_maneuver.slot_parallel import line_intrusion                # 칸 선 침범 깊이
from pinky_wall_parking_maneuver.slot_parallel import Slot                          # 칸
from pinky_wall_parking_maneuver.slot_parallel import SlotParams                    # 칸 파라미터
from pinky_wall_parking_maneuver.t_parking_node import declare                      # dataclass 파라미터
from pinky_wall_parking_maneuver.t_parking_node import yaw_of                       # 쿼터니언 -> yaw

ACTIVE = ("DRIVE", "ALIGN")                                                         # 평가하는 상태
END    = ("DONE", "STOP", "FAIL")                                                   # 요약을 내는 상태 (앞부분)


def seg_point(p, a, b):
    """Distance from point p to segment ab."""
    ab, ap = b - a, p - a
    t = min(1.0, max(0.0, float(ap @ ab) / max(float(ab @ ab), 1e-12)))
    return float(np.hypot(*(ap - t * ab)))


def obb_distance(a, b):
    """Minimum distance between two boxes (0 if they overlap or touch)."""
    if obb_overlap(a, b):
        return 0.0
    ca, cb = a.corners(), b.corners()
    d = [seg_point(p, q[i], q[(i + 1) % 4]) for p_set, q in ((ca, cb), (cb, ca)) for p in p_set for i in range(4)]
    return min(d)


class SlotParallelEval(Node):

    def __init__(self):
        super().__init__("slot_parallel_eval")
        self.sp, self.op = declare(self, SlotParams), ObsParams()
        p = {n: self.declare_parameter(n, v).value for n, v in (
            ("slot_x", 0.600), ("slot_y", -0.200), ("slot_yaw_deg", 0.0), ("slot_side", 1), ("max_parked", 10))}
        self.slot = Slot(p["slot_x"], p["slot_y"], math.radians(p["slot_yaw_deg"]), int(p["slot_side"]))
        self.max_parked = int(p["max_parked"])
        self.gt, self.odom, self.phase, self.parked = None, None, "IDLE", None
        self.reset()
        self.buf = Buffer()
        self.tfl = TransformListener(self.buf, self)
        self.create_subscription(PoseStamped, "ground_truth", self.on_gt, 10)
        self.create_subscription(Odometry, "odom", self.on_odom, 10)
        self.create_subscription(String, "slot_parallel/state", self.on_state, 10)
        self.create_timer(0.05, self.on_timer)                                      # 20 Hz
        self.get_logger().info(f"slot_parallel_eval ready: slot ({self.slot.cx:.3f}, {self.slot.cy:.3f}), "
                               f"length {self.sp.length:.3f}, width {self.sp.width:.3f}")

    def reset(self):
        self.intr_gt = self.intr_odom = 0.0                                         # 침범 최대 (m)
        self.clear = math.inf                                                       # 이웃 최소 거리 (m)
        self.err_xy = self.err_yaw = 0.0                                            # odom 오차 최대
        self.samples = 0

    @staticmethod
    def pose(msg_pose):
        q = msg_pose
        return (q.position.x, q.position.y, yaw_of(q.orientation))

    def on_gt(self, msg):
        self.gt = self.pose(msg.pose)

    def on_odom(self, msg):
        self.odom = self.pose(msg.pose.pose)

    def neighbors(self):
        """Parked Pinky boxes from tf world -> parked_<i> (looked up once)."""
        if self.parked is None:
            found = []
            for i in range(self.max_parked):
                if not self.buf.can_transform("world", f"parked_{i}", Time()):
                    break
                t = self.buf.lookup_transform("world", f"parked_{i}", Time()).transform
                found.append(robot_obb((t.translation.x, t.translation.y, yaw_of(t.rotation)), self.op))
            if found:
                self.parked = found
                self.get_logger().info(f"parked pinky: {len(found)}")
        return self.parked or []

    def on_state(self, msg):
        prev, self.phase = self.phase, msg.data
        if self.phase == "OBSERVE":
            self.reset()
        if self.phase.startswith(END) and prev in ACTIVE + ("READY", "PLAN"):
            self.summary()

    def on_timer(self):
        nbs = self.neighbors()
        if self.phase not in ACTIVE or self.gt is None or self.odom is None:
            return
        self.samples += 1
        self.intr_gt   = max(self.intr_gt, line_intrusion(self.gt, self.slot, self.sp, self.op))
        self.intr_odom = max(self.intr_odom, line_intrusion(self.odom, self.slot, self.sp, self.op))
        me = robot_obb(self.gt, self.op)
        if nbs:
            self.clear = min(self.clear, min(obb_distance(me, b) for b in nbs))
        self.err_xy  = max(self.err_xy, math.hypot(self.gt[0] - self.odom[0], self.gt[1] - self.odom[1]))
        dyaw = (self.gt[2] - self.odom[2] + math.pi) % (2 * math.pi) - math.pi
        self.err_yaw = max(self.err_yaw, abs(dyaw))

    def in_slot(self, pose):
        x, y, a = self.slot.to_slot(pose)
        return x + (self.op.body_x[0] + self.op.body_x[1]) / 2, y, a                # 차체 중심 (근사)

    def summary(self):
        if self.gt is None or self.odom is None:
            return self.get_logger().warn("eval: no ground truth / odom")
        g, o = self.in_slot(self.gt), self.in_slot(self.odom)
        clear = "n/a" if math.isinf(self.clear) else f"{self.clear * 1000:.1f} mm"
        self.get_logger().info(
            f"EVAL {self.phase}: end truth x {g[0] * 1000:+.1f} y {g[1] * 1000:+.1f} mm "
            f"yaw {math.degrees(g[2]):+.2f} deg"
            f" | odom x {o[0] * 1000:+.1f} y {o[1] * 1000:+.1f} mm yaw {math.degrees(o[2]):+.2f} deg"
            f" | intrusion truth {self.intr_gt * 1000:.1f} / odom {self.intr_odom * 1000:.1f} mm"
            f" | neighbor min gap {clear} | odom err max {self.err_xy * 1000:.1f} mm "
            f"{math.degrees(self.err_yaw):.2f} deg ({self.samples} samples)")


def main(args=None):
    rclpy.init(args=args)
    node = SlotParallelEval()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
