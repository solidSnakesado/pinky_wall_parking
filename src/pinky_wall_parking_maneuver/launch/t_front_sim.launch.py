"""src/pinky_wall_parking_maneuver/launch/t_front_sim.launch.py - 시뮬 차선 -> 벽 칸 전면 주차 실행

실행 구성:
  1. 시뮬 (pinky_wall_parking_sim/slot_sim.launch.py): 카메라 켬, 칸을 방 벽에서 side_gap 떼고 칸 옆 차선에서 출발
  2. slot_detector (라이다 칸 인식) -> /slot/pose (차선에서 칸 찾기)
  3. slot_vision (카메라 칸 특징, pinky_wall_parking_vision/slot_vision.launch.py)
  4. front_parking (카메라 + odom 전면 진입, pinky_wall_parking_front/front_parking.launch.py, 시뮬은 초음파 없음)
  5. t_parking_node (front true): 차선 주행 -> 칸 고정 -> 전면 정렬 자세 (입구 밖 0.272 m = f230) -> front_parking/start
- 모두 use_sim_time (시뮬 /clock). 차선은 R + 0.272 = 0.42 m 이상 (기본 0.45 m)
실행: ros2 launch pinky_wall_parking_maneuver t_front_sim.launch.py
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
            ("start_yaw", "-90"), ("use_gpu", "true"), ("viewer", "true"), ("camera", "true")]
SIM_TIME = {"use_sim_time": True}


def include(package, name, args):
    path = os.path.join(get_package_share_directory(package), "launch", name)
    return IncludeLaunchDescription(PythonLaunchDescriptionSource(path), launch_arguments=args.items())


# ==================================================================================
# launch 구성
# ==================================================================================
def generate_launch_description():
    return LaunchDescription([
        *[DeclareLaunchArgument(name, default_value=value) for name, value in SIM_ARGS],
        include("pinky_wall_parking_sim", "slot_sim.launch.py",
                {name: LaunchConfiguration(name) for name, _ in SIM_ARGS}),
        Node(package="pinky_wall_parking_perception", executable="slot_detector",
             parameters=[SIM_TIME], output="screen"),
        include("pinky_wall_parking_vision", "slot_vision.launch.py",
                {"use_sim_time": "true", "publish_debug": "false"}),
        include("pinky_wall_parking_front", "front_parking.launch.py",
                {"use_sim_time": "true", "use_ultrasonic": "false"}),
        Node(package="pinky_wall_parking_maneuver", executable="t_parking_node", name="t_parking",
             parameters=[SIM_TIME, {"front": True, "stage_dist": 0.272}], output="screen"),
    ])
