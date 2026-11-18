"""
src/pinky_wall_parking_docking/launch/docking_sim.launch.py - 시뮬 벽 칸 후면 진입 (Nav2 Docking Server) 실행

- 실행 구성
    1. 시뮬 (pinky_wall_parking_sim/slot_sim.launch.py): 칸 폭, 로봇 시작 위치 인자 전달
    2. slot_detector (라이다 칸 인식) -> /slot/pose
    3. docking_server (opennav_docking): detected_dock_pose <- /slot/pose, 속도 -> /cmd_vel
    4. lifecycle_manager: docking_server 자동 활성화
- 모두 use_sim_time (시뮬 /clock)
- 주차 요청은 터미널에서 action 으로 (주차 관리 노드는 다음 단계)

실행:
  ros2 launch pinky_wall_parking_docking docking_sim.launch.py slot_width:=131 start_dist:=302
  주차 요청:
  ros2 action send_goal /dock_robot nav2_msgs/action/DockRobot \\
    "{use_dock_id: false, dock_type: slot_dock, navigate_to_staging_pose: false,
      dock_pose: {header: {frame_id: odom}}}"
"""

# ==================== 표준 라이브러리 ======================
import os                                                                           # 경로 결합

# ==================== ROS 2 launch =========================
from ament_index_python.packages import get_package_share_directory                 # 패키지 설치 경로
from launch import LaunchDescription                                                # 실행 구성
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription          # 인자, 다른 launch 포함
from launch.launch_description_sources import PythonLaunchDescriptionSource         # .launch.py 읽기
from launch.substitutions import LaunchConfiguration                                # 인자 값 읽기
from launch_ros.actions import Node                                                 # 노드 실행

# 시뮬로 넘기는 인자 (이름, 기본값)
SIM_ARGS = [("slot_width", "131"), ("start_dist", "302"), ("start_lat", "0"),
            ("start_yaw", "0"), ("use_gpu", "true"), ("viewer", "true"), ("camera", "false")]

# =========================================================
# launch 구성
# =========================================================
def generate_launch_description():
    sim_launch  = os.path.join(get_package_share_directory("pinky_wall_parking_sim"),
                               "launch", "slot_sim.launch.py")
    params      = os.path.join(get_package_share_directory("pinky_wall_parking_docking"),
                               "config", "docking_server.yaml")
    sim_time    = {"use_sim_time": True}

    return LaunchDescription([
        *[DeclareLaunchArgument(name, default_value=value) for name, value in SIM_ARGS],
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(sim_launch),
            launch_arguments={name: LaunchConfiguration(name) for name, _ in SIM_ARGS}.items()),
        Node(package="pinky_wall_parking_perception", executable="slot_detector",
             parameters=[sim_time], output="screen"),
        Node(package="opennav_docking", executable="opennav_docking", name="docking_server",
             parameters=[params, sim_time], output="screen",
             remappings=[("detected_dock_pose", "/slot/pose")]),
        Node(package="nav2_lifecycle_manager", executable="lifecycle_manager",
             name="lifecycle_manager_docking", output="screen",
             parameters=[sim_time, {"autostart": True, "node_names": ["docking_server"]}]),
    ])