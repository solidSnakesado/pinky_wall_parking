"""src/pinky_wall_parking_maneuver/launch/slot_parallel_sim.launch.py - 시뮬 라인 평행 칸 주차 실행

실행 구성:
  1. 시뮬 (pinky_wall_parking_sim/scene_sim.launch.py): scene (기본 scene_line_parallel.yaml = 170 mm 칸,
     scene_line_parallel_230.yaml = 230 mm 칸. 칸 3 개, 앞뒤 칸에 핑키)
  2. slot_parallel_node: 정지 관측 (라이다 퍼크 -> 이웃 OBB) -> 칸 경로 계획 -> 차량식 추종 -> 마무리 정렬
- 모두 use_sim_time (시뮬 /clock). 칸 위치 = 장면의 가운데 칸 (출발 = odom 원점 기준 (slot_x, slot_y), 0 deg, 차선 = 칸 왼쪽)
인자: scene (config 파일 이름), slot_length, style (multi / video), entry_gear (-1 후진 / +1 전진 진입), slot_x, slot_y,
      use_gpu, viewer
실행: ros2 launch pinky_wall_parking_maneuver slot_parallel_sim.launch.py entry_gear:=-1
      ros2 launch pinky_wall_parking_maneuver slot_parallel_sim.launch.py scene:=scene_line_parallel_230.yaml \
          slot_length:=0.230 style:=video
      ros2 service call /slot_parallel/start std_srvs/srv/Trigger
"""

# --- 표준 라이브러리 ---------------------------------------------------------------
import os                                                                           # 경로 결합

# --- ROS 2 launch ------------------------------------------------------------------
from ament_index_python.packages import get_package_share_directory                 # 패키지 설치 경로
from launch import LaunchDescription                                                # 실행 구성
from launch.actions import DeclareLaunchArgument                                    # 인자
from launch.actions import IncludeLaunchDescription                                 # 다른 launch 포함
from launch.actions import OpaqueFunction                                           # 인자 값으로 구성
from launch.launch_description_sources import PythonLaunchDescriptionSource         # .launch.py 읽기
from launch.substitutions import LaunchConfiguration                                # 인자 값 읽기
from launch_ros.actions import Node                                                 # 노드 실행

SIM_ARGS  = [("use_gpu", "true"), ("viewer", "true")]                               # 시뮬로 넘기는 인자
NODE_ARGS = [("scene", "scene_line_parallel.yaml"), ("slot_length", "0.170"), ("style", "multi"),
             ("entry_gear", "-1"), ("slot_x", "0.600"), ("slot_y", "-0.200")]       # 노드 / 장면 인자


def setup(context):
    arg = {n: LaunchConfiguration(n).perform(context) for n, _ in NODE_ARGS}        # 인자 값
    sim_share = get_package_share_directory("pinky_wall_parking_sim")
    sim_args  = {"scene": os.path.join(sim_share, "config", arg["scene"]), "camera": "false",
                 **{n: LaunchConfiguration(n) for n, _ in SIM_ARGS}}
    params    = {"use_sim_time": True, "slot_x": float(arg["slot_x"]), "slot_y": float(arg["slot_y"]),
                 "slot_yaw_deg": 0.0, "slot_side": 1, "length": float(arg["slot_length"]),
                 "style": arg["style"], "entry_gear": int(arg["entry_gear"])}
    return [IncludeLaunchDescription(PythonLaunchDescriptionSource(
                os.path.join(sim_share, "launch", "scene_sim.launch.py")), launch_arguments=sim_args.items()),
            Node(package="pinky_wall_parking_maneuver", executable="slot_parallel_node", name="slot_parallel",
                 output="screen", parameters=[params])]


def generate_launch_description():
    return LaunchDescription([
        *[DeclareLaunchArgument(name, default_value=value) for name, value in SIM_ARGS + NODE_ARGS],
        OpaqueFunction(function=setup),
    ])
