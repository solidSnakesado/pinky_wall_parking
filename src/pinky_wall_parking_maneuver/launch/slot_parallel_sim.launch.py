"""src/pinky_wall_parking_maneuver/launch/slot_parallel_sim.launch.py - 시뮬 라인 평행 칸 주차 실행

실행 구성:
  1. 시뮬 (pinky_wall_parking_sim/scene_sim.launch.py): scene (기본 scene_line_parallel.yaml = 170 mm 칸,
     scene_line_parallel_230.yaml = 230 mm 칸. 칸 3 개, 앞뒤 칸에 핑키)
  2. slot_parallel_node: 정지 관측 (라이다 퍼크 -> 이웃 OBB) -> 칸 경로 계획 -> 차량식 추종 -> 마무리 정렬
- 모두 use_sim_time (시뮬 /clock). 칸 위치 = 장면의 가운데 칸 (장면 좌표 (slot_x, slot_y), 0 deg, 차선 = 칸 왼쪽)
- start_y: 로봇 출발 y (장면 좌표, 비우면 0 = 장면 YAML 의 start). 노드에는 출발 (odom 원점) 기준으로 바꿔
  slot_y - start_y, bounds y - start_y 를 넘김 (두 라인 장면 모두 YAML start = 원점)
인자: scene (config 파일 이름), slot_length, style (multi / video), entry_gear (-1 후진 / +1 전진 진입), slot_x, slot_y,
      start_y, keep_lines (true = 칸 선 넘지 않기), use_gpu, viewer, camera, physics_config,
      lidar_model (ideal = /scan 그대로 / real_c1 = 실물 특성 /scan_real + 노드 laser yaw -2.2°, 근거리 보정),
      tol_x (정렬 허용 길이 방향 m, 비우면 SlotParams 기본)
실행: ros2 launch pinky_wall_parking_maneuver slot_parallel_sim.launch.py entry_gear:=-1
      ros2 launch pinky_wall_parking_maneuver slot_parallel_sim.launch.py scene:=scene_line_parallel_230.yaml \
          slot_length:=0.230 style:=video
      ros2 launch pinky_wall_parking_maneuver slot_parallel_sim.launch.py scene:=scene_line_parallel_230.yaml \
          slot_length:=0.230 style:=video entry_gear:=1 start_y:=-0.040
      ros2 launch pinky_wall_parking_maneuver slot_parallel_sim.launch.py scene:=scene_line_parallel_230.yaml \
          slot_length:=0.230 style:=multi entry_gear:=1 keep_lines:=true
      ros2 launch pinky_wall_parking_maneuver slot_parallel_sim.launch.py scene:=scene_line_parallel_230.yaml \
          slot_length:=0.230 style:=multi entry_gear:=1 keep_lines:=true lidar_model:=real_c1
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

SIM_ARGS  = [("use_gpu", "true"), ("viewer", "true"), ("camera", "false"),
             ("physics_config", ""), ("lidar_model", "ideal")]                      # 시뮬로 넘기는 인자
LIDARS    = {"ideal": ("/scan", 0.0, False),                                        # 시뮬 모델 -> (스캔, yaw, 근거리)
             "real_c1": ("/scan_real", -2.2, True)}                                 # 짝을 launch 에서 고정
NODE_ARGS = [("scene", "scene_line_parallel.yaml"), ("slot_length", "0.170"), ("style", "multi"),
             ("entry_gear", "-1"), ("slot_x", "0.600"), ("slot_y", "-0.200"),
             ("start_y", ""), ("keep_lines", "false"), ("tol_x", "")]               # 노드 / 장면 인자


def setup(context):
    arg = {n: LaunchConfiguration(n).perform(context) for n, _ in NODE_ARGS}        # 인자 값
    sim_share = get_package_share_directory("pinky_wall_parking_sim")
    lidar     = LaunchConfiguration("lidar_model").perform(context)
    if lidar not in LIDARS:                                                         # 오타를 launch 시작 때 알림
        raise ValueError(f"lidar_model must be one of {sorted(LIDARS)}: {lidar}")
    scan, yaw_off, near = LIDARS[lidar]
    dy        = float(arg["start_y"] or 0.0)                                        # 출발 y (장면 좌표)
    sim_args  = {"scene": os.path.join(sim_share, "config", arg["scene"]), "start_y": arg["start_y"],
                 **{n: LaunchConfiguration(n) for n, _ in SIM_ARGS}}
    params    = {"use_sim_time": True, "slot_x": float(arg["slot_x"]), "slot_y": float(arg["slot_y"]) - dy,
                 "slot_yaw_deg": 0.0, "slot_side": 1, "length": float(arg["slot_length"]),
                 "style": arg["style"], "entry_gear": int(arg["entry_gear"]),
                 "keep_lines": arg["keep_lines"].lower() == "true",
                 "laser_yaw_offset_deg": yaw_off, "near_correction": near,
                 "bounds": [-0.40, 1.40, -0.32 - dy, 0.50 - dy],                    # 판 범위 -> 출발 기준
                 **({"tol_x": float(arg["tol_x"])} if arg["tol_x"] else {})}        # 비우면 SlotParams 기본
    return [IncludeLaunchDescription(PythonLaunchDescriptionSource(
                os.path.join(sim_share, "launch", "scene_sim.launch.py")), launch_arguments=sim_args.items()),
            Node(package="pinky_wall_parking_maneuver", executable="slot_parallel_node", name="slot_parallel",
                 output="screen", parameters=[params], remappings=[("scan", scan)])]


def generate_launch_description():
    return LaunchDescription([
        *[DeclareLaunchArgument(name, default_value=value) for name, value in SIM_ARGS + NODE_ARGS],
        OpaqueFunction(function=setup),
    ])
