"""src/pinky_wall_parking_maneuver/launch/t_parking_sim.launch.py - 시뮬 T자 후면 주차 실행

실행 구성:
  1. 시뮬 (pinky_wall_parking_sim/slot_sim.launch.py): 칸을 방 벽에서 side_gap 떼고, 칸 옆 차선에서 출발
  2. slot_detector (라이다 칸 인식) -> /slot/pose
  3. docking_server (opennav_docking, pinky_wall_parking_docking/config/docking_server.yaml)
     차량 조건을 위해 controller.v_angular_max = v_linear_min / R (0.02 / 0.15 = 0.13 rad/s) 로 덮어씀
  4. lifecycle_manager: docking_server 자동 활성화
  5. t_parking_node: 차선 주행 -> 칸 고정 -> 정렬 자세 (stage_dist 0.30 = docking staging) -> DockRobot 요청
- 모두 use_sim_time (시뮬 /clock)
실행: ros2 launch pinky_wall_parking_maneuver t_parking_sim.launch.py
      ros2 service call /t_parking/start std_srvs/srv/Trigger
"""

# --- 표준 라이브러리 ---------------------------------------------------------------
import os                                                                           # 경로 결합

# --- ROS 2 launch ------------------------------------------------------------------
from ament_index_python.packages import get_package_share_directory                 # 패키지 설치 경로
from launch import LaunchDescription                                                # 실행 구성
from launch.actions import DeclareLaunchArgument                                    # 인자
from launch.actions import IncludeLaunchDescription                                 # 다른 launch 포함
from launch.launch_description_sources import PythonLaunchDescriptionSource         # .launch.py 읽기
from launch.substitutions import LaunchConfiguration                                # 인자 값 읽기
from launch_ros.actions import Node                                                 # 노드 실행

# 시뮬로 넘기는 인자 (이름, 기본값). 출발 = 칸 입구 밖 450 mm 차선, 칸 축에서 500 mm 옆, 칸 쪽을 향함
SIM_ARGS = [("slot_width", "130"), ("side_gap", "400"), ("start_dist", "450"), ("start_lat", "500"),
            ("start_yaw", "-90"), ("use_gpu", "true"), ("viewer", "true"), ("camera", "false")]
SIM_TIME = {"use_sim_time": True}


# ==================================================================================
# launch 구성
# ==================================================================================
def generate_launch_description():
    sim_launch = os.path.join(get_package_share_directory("pinky_wall_parking_sim"),
                              "launch", "slot_sim.launch.py")
    dock_yaml  = os.path.join(get_package_share_directory("pinky_wall_parking_docking"),
                              "config", "docking_server.yaml")
    return LaunchDescription([
        *[DeclareLaunchArgument(name, default_value=value) for name, value in SIM_ARGS],
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(sim_launch),
            launch_arguments={name: LaunchConfiguration(name) for name, _ in SIM_ARGS}.items()),
        Node(package="pinky_wall_parking_perception", executable="slot_detector",
             parameters=[SIM_TIME], output="screen"),
        Node(package="opennav_docking", executable="opennav_docking", name="docking_server",
             parameters=[dock_yaml, SIM_TIME, {"controller.v_angular_max": 0.13}], output="screen",
             remappings=[("detected_dock_pose", "/slot/pose")]),
        Node(package="nav2_lifecycle_manager", executable="lifecycle_manager",
             name="lifecycle_manager_docking", output="screen",
             parameters=[SIM_TIME, {"autostart": True, "node_names": ["docking_server"]}]),
        Node(package="pinky_wall_parking_maneuver", executable="t_parking_node", name="t_parking",
             parameters=[SIM_TIME, {"stage_dist": 0.30}], output="screen"),
    ])
