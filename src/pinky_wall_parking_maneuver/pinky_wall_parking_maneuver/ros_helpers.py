"""src/pinky_wall_parking_maneuver/pinky_wall_parking_maneuver/ros_helpers.py - 노드 공용 작은 도구 (자세 곱, 스캔 점, 경로 메시지)

역할: 주차 노드들이 함께 쓰는 변환. 자세 a o b, LaserScan -> odom 점, 기어 목록 -> nav_msgs/Path, 속도 명령,
      Ctrl+C 를 놓치지 않는 spin (시뮬 시간이 멈춰도 period 마다 깨어남, 무시된 SIGINT 복구).
실행: 라이브러리 (slot_parallel_node 에서 호출).
"""

# --- 표준 라이브러리 ---------------------------------------------------------------
import math                                                                         # 삼각 함수
import signal                                                                       # SIGINT 처리기

# --- ROS 2, 외부 라이브러리 -------------------------------------------------------
from geometry_msgs.msg import PoseStamped                                           # 경로 점
from geometry_msgs.msg import Twist                                                 # 속도 명령
from nav_msgs.msg import Path                                                       # 경로
import numpy as np                                                                  # 점 배열
import rclpy                                                                        # spin


def compose(a, b):
    """Return pose a o b, each (x, y, yaw)."""
    c, s = math.cos(a[2]), math.sin(a[2])                                           # a 방향
    return a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], a[2] + b[2]


def scan_points(msg, sensor, near=None):
    """Return valid scan points (Nx2) in the frame where the lidar is at sensor (x, y, yaw).

    near: real lidar near-range table (cut, [(range, bias), ...]) as line_extraction.REAL_NEAR, None = off.
    """
    r = np.asarray(msg.ranges, dtype=float)                                         # 거리
    a = sensor[2] + msg.angle_min + msg.angle_increment * np.arange(len(r))         # 방향 각
    ok = np.isfinite(r) & (r >= msg.range_min) & (r <= msg.range_max)               # 유효 점
    if near is not None:                                                            # line_extraction 과 같은 보정
        cut, table = near
        ok &= r >= cut                                                              # 너무 가까운 값 버림
        r_tab, d_tab = np.array(table).T
        r = r - np.interp(r, r_tab, d_tab, left=d_tab[0], right=0.0)
    return np.column_stack([sensor[0] + r[ok] * np.cos(a[ok]), sensor[1] + r[ok] * np.sin(a[ok])])


def spin(node, period=0.1):
    """Spin until Ctrl+C or shutdown, waking every period s (wall clock) so Ctrl+C is seen even if sim time stops.

    rclpy.spin waits inside C until a callback is ready; with sim-time timers and a stopped /clock nothing
    becomes ready and KeyboardInterrupt (SignalHandlerOptions.NO) is never raised.
    A process started with & from a non-interactive shell inherits SIGINT ignored, and Python then installs
    no KeyboardInterrupt handler, so it is installed here explicitly.
    """
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=period)
    except KeyboardInterrupt:
        pass


def twist(v, w):
    """Return a geometry_msgs/Twist with linear x v and angular z w."""
    msg = Twist()
    msg.linear.x, msg.angular.z = float(v), float(w)
    return msg


def path_msg(gears, frame_id, stamp):
    """Return a nav_msgs/Path of all gear points."""
    path = Path()
    path.header.frame_id, path.header.stamp = frame_id, stamp
    for x, y, yaw in (q for pts, _ in gears for q in pts):
        pose = PoseStamped(header=path.header)
        pose.pose.position.x, pose.pose.position.y = float(x), float(y)
        pose.pose.orientation.z, pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        path.poses.append(pose)
    return path
