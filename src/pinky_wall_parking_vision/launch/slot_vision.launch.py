"""src/pinky_wall_parking_vision/launch/slot_vision.launch.py - 카메라 벽 칸 특징 노드 실행

역할: slot_vision_node 를 파라미터 파일 (config/slot_vision.yaml) 과 로봇별 인자로 실행한다.
인자: roll_deg       카메라 옆 기울기 보정 (deg). 1호기 -1.4, 2호기·시뮬 0.0 (기본)
      publish_debug  확인용 영상 발행 (기본 true, 구독자가 있을 때만 그림)
      use_sim_time   시뮬 시간 사용 (기본 false, 시뮬은 true)
실행: ros2 launch pinky_wall_parking_vision slot_vision.launch.py use_sim_time:=true
      ros2 launch pinky_wall_parking_vision slot_vision.launch.py roll_deg:=-1.4
"""

# --- 표준 라이브러리 ---------------------------------------------------------------
import os                                                                           # 경로 결합

# --- ROS 2 launch ------------------------------------------------------------------
from ament_index_python.packages import get_package_share_directory                 # 패키지 설치 경로
from launch import LaunchDescription                                                # 실행 구성
from launch.actions import DeclareLaunchArgument                                    # 인자
from launch.substitutions import LaunchConfiguration                                # 인자 값 읽기
from launch_ros.actions import Node                                                 # 노드 실행
from launch_ros.parameter_descriptions import ParameterValue                        # 인자 문자열 -> 형식

# 인자 (이름, 기본값, 형식)
ARGS = [("roll_deg", "0.0", float), ("publish_debug", "true", bool), ("use_sim_time", "false", bool)]


# ==================================================================================
# launch 구성
# ==================================================================================
def generate_launch_description():
    params = os.path.join(get_package_share_directory("pinky_wall_parking_vision"),
                          "config", "slot_vision.yaml")
    values = {name: ParameterValue(LaunchConfiguration(name), value_type=kind) for name, _, kind in ARGS}
    return LaunchDescription([
        *[DeclareLaunchArgument(name, default_value=value) for name, value, _ in ARGS],
        Node(package="pinky_wall_parking_vision", executable="slot_vision_node", name="slot_vision",
             parameters=[params, values], output="screen"),
    ])
