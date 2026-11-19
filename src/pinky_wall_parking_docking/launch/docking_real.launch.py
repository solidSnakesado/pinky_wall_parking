"""
src/pinky_wall_parking_docking/launch/docking_real.launch.py - 실물 벽 칸 후면 진입 (Nav2 Docking Server) 실행 (PC)

- 핑키에서는 bringup 만 실행 (ros2 launch pinky_bringup bringup_robot.launch.xml)
- PC 실행 구성 (ROS_DOMAIN_ID=42, ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET)
    1. slot_detector: 실물 근거리 보정 + 라이다 장착 각도 보정 -> /slot/pose
    2. docking_server: detected_dock_pose <- /slot/pose, 속도 -> /cmd_vel_nav
    3. velocity_smoother: /cmd_vel_nav -> /cmd_vel (0.2 s 명령 끊기면 정지, 안전 단)
    4. lifecycle_manager: docking_server, velocity_smoother 자동 활성화
- 시뮬 시간 쓰지 않음 (use_sim_time false)
- 목표 깊이 target_depth: 입구 -> base_link 목표 (m). 0.102 = 뒤 벽 틈 12 mm (기본, 시뮬과 같음)
- 재시도 없음 (max_retries 0): 칸 인식이 끊기면 실패로 끝내고 그 자리에 정지

실행:
  ros2 launch pinky_wall_parking_docking docking_real.launch.py
  주차 요청은 시뮬과 같음 (ros2 action send_goal /dock_robot ...)
"""

# ==================== 표준 라이브러리 ======================
import os                                                                           # 경로 결합

# ==================== ROS 2 launch =========================
from ament_index_python.packages import get_package_share_directory                 # 패키지 설치 경로
from launch import LaunchDescription                                                # 실행 구성
from launch.actions import DeclareLaunchArgument                                    # 인자
from launch.substitutions import LaunchConfiguration                                # 인자 값 읽기
from launch_ros.actions import Node                                                 # 노드 실행
from launch_ros.parameter_descriptions import ParameterValue                        # 인자 -> 실수 파라미터

# =========================================================
# launch 구성
# =========================================================
def generate_launch_description():
    share       = get_package_share_directory("pinky_wall_parking_docking")
    docking     = os.path.join(share, "config", "docking_server.yaml")
    smoother    = os.path.join(share, "config", "velocity_smoother.yaml")
    real_time   = {"use_sim_time": False}
    depth       = ParameterValue(LaunchConfiguration("target_depth"), value_type=float)
    yaw_offset  = ParameterValue(LaunchConfiguration("laser_yaw_offset_deg"), value_type=float)

    return LaunchDescription([
        DeclareLaunchArgument("target_depth", default_value="0.102",
                              description="Entrance -> base_link goal depth in m (0.102 = rear gap 12 mm)"),
        DeclareLaunchArgument("laser_yaw_offset_deg", default_value="-2.2",
                              description="Real lidar mounting yaw correction in deg"),
        Node(package="pinky_wall_parking_perception", executable="slot_detector", output="screen",
             parameters=[real_time, {"near_correction": True, "laser_yaw_offset_deg": yaw_offset}]),
        Node(package="opennav_docking", executable="opennav_docking", name="docking_server",
             output="screen",
             parameters=[docking, real_time, {"slot_dock.external_detection_translation_x": depth,
                                              "max_retries": 0}],
             remappings=[("detected_dock_pose", "/slot/pose"), ("cmd_vel", "cmd_vel_nav")]),
        Node(package="nav2_velocity_smoother", executable="velocity_smoother", name="velocity_smoother",
             output="screen", parameters=[smoother, real_time],
             remappings=[("cmd_vel", "cmd_vel_nav"), ("cmd_vel_smoothed", "cmd_vel")]),
        Node(package="nav2_lifecycle_manager", executable="lifecycle_manager",
             name="lifecycle_manager_docking", output="screen",
             parameters=[real_time, {"autostart": True,
                                     "node_names": ["velocity_smoother", "docking_server"]}]),
    ])