"""src/pinky_wall_parking_maneuver/pinky_wall_parking_maneuver/ros_helpers.py - 노드 공용 작은 도구 (자세 곱, 스캔 점, 경로 메시지)

역할: 주차 노드들이 함께 쓰는 변환. 자세 a o b, LaserScan -> odom 점, 기어 목록 -> nav_msgs/Path, 속도 명령.
실행: 라이브러리 (slot_parallel_node 에서 호출).
"""

# --- 표준 라이브러리 ---------------------------------------------------------------
import math                                                                         # 삼각 함수

# --- ROS 2, 외부 라이브러리 -------------------------------------------------------
from geometry_msgs.msg import PoseStamped                                           # 경로 점
from geometry_msgs.msg import Twist                                                 # 속도 명령
from nav_msgs.msg import Path                                                       # 경로
import numpy as np                                                                  # 점 배열


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
