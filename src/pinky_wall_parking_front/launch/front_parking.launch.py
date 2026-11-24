"""src/pinky_wall_parking_front/launch/front_parking.launch.py - 전방 주차 노드 실행

역할: front_parking_node 를 파라미터 파일 (config/front_parking.yaml) 과 시뮬·실물 인자로 실행한다.
      칸 특징은 slot_vision.launch.py 를 따로 실행해 받는다.
인자: use_ultrasonic  초음파 정지 사용 (기본 false = 시뮬, 실물은 true)
      use_sim_time    시뮬 시간 사용 (기본 false, 시뮬은 true)
      z_scale, z_offset_mm  거리 보정 (기본 = 시뮬 맞춤값 0.940, 24.5. 1호기 0.925, 42.5)
      v_approach, k_psi, k_e  속도·이득 (기본 0.02, 1.0, 0.0125. 거리 기준 같은 수렴: 0.04 -> 2.0, 0.025, 0.06 -> 3.0, 0.0375)
실행: ros2 launch pinky_wall_parking_front front_parking.launch.py use_sim_time:=true
      ros2 launch pinky_wall_parking_front front_parking.launch.py use_ultrasonic:=true z_scale:=0.925 z_offset_mm:=42.5
      주차 시작: ros2 service call /front_parking/start std_srvs/srv/Trigger
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
ARGS = [("use_ultrasonic", "false", bool), ("use_sim_time", "false", bool),
        ("z_scale", "0.940", float), ("z_offset_mm", "24.5", float),
        ("v_approach", "0.02", float), ("k_psi", "1.0", float), ("k_e", "0.0125", float)]


# ==================================================================================
# launch 구성
# ==================================================================================
def generate_launch_description():
    params = os.path.join(get_package_share_directory("pinky_wall_parking_front"),
                          "config", "front_parking.yaml")
    values = {name: ParameterValue(LaunchConfiguration(name), value_type=kind) for name, _, kind in ARGS}
    return LaunchDescription([
        *[DeclareLaunchArgument(name, default_value=value) for name, value, _ in ARGS],
        Node(package="pinky_wall_parking_front", executable="front_parking_node", name="front_parking",
             parameters=[params, values], output="screen"),
    ])
