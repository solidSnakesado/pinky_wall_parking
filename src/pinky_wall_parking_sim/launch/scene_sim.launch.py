"""src/pinky_wall_parking_sim/launch/scene_sim.launch.py - 장면 YAML 로 핑키 MuJoCo 시뮬 실행

실행 구성:
  1. 장면 YAML (scene 인자, 기본 config/scene_parallel_two.yaml) -> SDF (~/.cache/pinky_wall_parking/scene_<이름>.world)
     세워 둔 핑키는 pinky_description URDF (pinky_mujoco 가 펼침) 의 visual / collision mesh 그대로
  2. pinky_mujoco launch_sim.launch.py (world = 위 SDF)
     physics_config: MuJoCo 설정 YAML. 비우면 포크 기본값, 이름만 주면 이 패키지 config/ 에서 찾음
     (실물 카메라 640x480, 화각 53.5° = mujoco_real_camera.yaml)
     start_y: 로봇 출발 y (m, 장면 좌표). 비우면 장면 YAML 의 start 그대로 (칸 옆 거리 바꿔 시험할 때)
  3. 세워 둔 핑키 정답 위치 static tf: world -> parked_<i> (평가용)
실행: ros2 launch pinky_wall_parking_sim scene_sim.launch.py
      ros2 launch pinky_wall_parking_sim scene_sim.launch.py scene:=/path/to/scene.yaml camera:=false
      ros2 launch pinky_wall_parking_sim scene_sim.launch.py camera:=true physics_config:=mujoco_real_camera.yaml
      ros2 launch pinky_wall_parking_sim scene_sim.launch.py scene:=/path/to/scene.yaml start_y:=-0.040
"""

# --- 표준 라이브러리 ---------------------------------------------------------------
import math                                                                         # 각도 변환
import os                                                                           # 경로 결합
from pathlib import Path                                                            # 월드 저장 폴더

# --- ROS 2 launch ------------------------------------------------------------------
from ament_index_python.packages import get_package_share_directory                 # 패키지 설치 경로
from launch import LaunchDescription                                                # 실행 구성
from launch.actions import DeclareLaunchArgument                                    # 인자
from launch.actions import IncludeLaunchDescription                                 # 다른 launch 포함
from launch.actions import OpaqueFunction                                           # 실행 시 계산
from launch.launch_description_sources import PythonLaunchDescriptionSource         # .launch.py 읽기
from launch.substitutions import LaunchConfiguration                                # 인자 값 읽기
from launch_ros.actions import Node                                                 # static tf 발행
from pinky_mujoco.description import expand_urdf                                    # 핑키 xacro 펼치기

# --- 이 패키지 ---------------------------------------------------------------------
from pinky_wall_parking_sim.scene_world import load_scene                           # 장면 YAML
from pinky_wall_parking_sim.scene_world import scene_poses                          # 세워 둔 핑키 월드 자세
from pinky_wall_parking_sim.scene_world import write_scene_world                    # SDF 저장

WORLD_DIR = Path.home() / ".cache" / "pinky_wall_parking"                           # 월드 저장 폴더
SCENE     = os.path.join(get_package_share_directory("pinky_wall_parking_sim"), "config", "scene_parallel_two.yaml")
CONFIG    = os.path.join(get_package_share_directory("pinky_wall_parking_sim"), "config")
ARGS      = [("scene", SCENE), ("use_gpu", "true"), ("viewer", "true"), ("camera", "false"),
             ("physics_config", ""), ("start_y", "")]


def setup(context):
    scene = load_scene(LaunchConfiguration("scene").perform(context))
    start_y = LaunchConfiguration("start_y").perform(context)                       # 비우면 YAML 그대로
    if start_y:
        scene.setdefault("start", {})["y"] = float(start_y)
    name  = scene.get("name", "scene")
    world = write_scene_world(WORLD_DIR / f"scene_{name}.world", scene, expand_urdf() if scene.get("parked") else None)
    _, parked = scene_poses(scene)
    print(f"[pinky_wall_parking_sim] scene {name}: boxes {len(scene.get('boxes', []))}, parked "
          + ", ".join(f"({x:.3f}, {y:.3f}, {math.degrees(yaw):.1f} deg)" for x, y, yaw in parked)
          + f", start y {scene.get('start', {}).get('y', 0.0):.3f}, world={world}")
    pinky = os.path.join(get_package_share_directory("pinky_mujoco"), "launch", "launch_sim.launch.py")
    args  = {"world": str(world), **{k: LaunchConfiguration(k) for k in ("use_gpu", "viewer", "camera")}}
    cfg   = LaunchConfiguration("physics_config").perform(context)                  # 비우면 포크 기본 설정
    if cfg and not os.path.isabs(cfg):                                              # 이름만 주면 이 패키지 config/
        cfg = os.path.join(CONFIG, cfg)
    if cfg and not os.path.isfile(cfg):                                             # 오타를 launch 시작 때 알림
        raise FileNotFoundError(f"physics_config not found: {cfg}")
    args["physics_config"] = cfg
    tfs   = [Node(package="tf2_ros", executable="static_transform_publisher", name=f"parked_{i}_tf",
                  arguments=["--x", f"{x:.4f}", "--y", f"{y:.4f}", "--yaw", f"{yaw:.6f}",
                             "--frame-id", "world", "--child-frame-id", f"parked_{i}"])
             for i, (x, y, yaw) in enumerate(parked)]
    return [IncludeLaunchDescription(PythonLaunchDescriptionSource(pinky), launch_arguments=args.items()), *tfs]


# ==================================================================================
# launch 구성
# ==================================================================================
def generate_launch_description():
    return LaunchDescription([
        *[DeclareLaunchArgument(name, default_value=value) for name, value in ARGS],
        OpaqueFunction(function=setup),
    ])
