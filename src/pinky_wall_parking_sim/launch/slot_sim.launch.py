"""
src/pinky_wall_parking_sim/launch/slot_sim.launch.py - 종이벽돌 벽 주차 칸 월드로 핑키 MuJoCo 시뮬 실행

- 실행 순서
    1. 인자 (칸 폭, 방 옆 벽 틈, 시작 위치) 로 SDF 월드 생성 (slot_world.write_slot_world)
       -> ~/.cache/pinky_wall_parking/slot_<폭>mm.world 에 저장
    2. 핑키 프로 시뮬 launch (pinky_mujoco/launch_sim.launch.py) 를 world:=<저장 경로> 로 포함
    3. 칸 입구 정답 자세를 static tf (world -> slot_entrance, x = 칸 밖 방향) 로 발행 (평가용)
- 핑키 프로 launch 를 불러다 사용 (핑키 launch 의 URDF 저장 방식과 같은 구조)
- 카메라는 실물 값: physics_config (기본 config/mujoco_real_camera.yaml, 640x480, 화각 53.5°), cam_tilt_deg (기본 8)
  -> 포크 wall-parking 의 launch_sim 에 추가한 physics_config 인자로 전달
- 원본 영상을 압축 영상 (camera/image_raw/compressed) 으로도 발행 (실물과 같은 토픽, image_transport republish)
- use_gpu 기본값 true (이 PC 확인 결과: 소프트웨어 렌더링은 카메라 13.5 Hz 로 제한)

실행:
  ros2 launch pinky_wall_parking_sim slot_sim.launch.py
  칸 폭 지정: slot_width:=130   (mm, 0 초과 190 이하)
  칸 왼쪽 벽 <-> 방 옆 벽 틈: side_gap:=50   (mm)
  창 없이: viewer:=false,  카메라 끄기: camera:=false
  시작 위치 (칸 입구 기준): start_dist:=250 start_lat:=0 start_yaw:=0   (mm, mm, deg)
    start_dist 비우면 (기본) 원래 배치. 음수 = 로봇 base 가 입구선보다 칸 안쪽, start_yaw 0 = 로봇 뒷면이 칸을 향함
  실물 칸 모양: mouth_width:=135 (입구 쪽 폭 mm, 비우면 slot_width = 나란함),
    side_end:=0 (방 옆 벽이 입구선 밖으로 나온 길이 mm, 비우면 판 끝까지)
    c1~c5 녹화 당시: slot_width:=130 mouth_width:=135 side_gap:=13 side_end:=0
    카메라 설정: physics_config:=<yaml> (빈 값 = 핑키 기본 mujoco.yaml), cam_tilt_deg:=8 (아래로 숙임, deg)
  라이다: lidar_model:=ideal (기본, /scan 그대로) / real_c1 (scan_realism_node: /scan -> 실물 특성 -> /scan_real,
    값 = config/lidar_real_c1.yaml). 처리 노드는 scan:=/scan_real, laser_yaw_offset_deg:=-2.2, near_correction:=true
"""

# ==================== 표준 라이브러리 ======================
import math                                                                         # 각도 변환
import os                                                                           # 경로 결합
from pathlib import Path                                                            # 월드 저장 폴더

# ==================== ROS 2 launch =========================
from ament_index_python.packages import get_package_share_directory                 # 패키지 설치 경로
from launch import LaunchDescription                                                # 실행 구성
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription          # 인자, 다른 launch 포함
from launch.actions import OpaqueFunction                                           # 실행 시 계산
from launch.launch_description_sources import PythonLaunchDescriptionSource         # .launch.py 읽기
from launch.substitutions import LaunchConfiguration                                # 인자 값 읽기
from launch_ros.actions import Node                                                 # static tf 발행

# ==================== 이 패키지 ============================
from pinky_wall_parking_sim.slot_world import start_layout, write_slot_world        # SDF 월드 생성

# 월드 저장 폴더
WORLD_DIR = Path.home() / ".cache" / "pinky_wall_parking"
# 실물 카메라 값 MuJoCo 설정 (포크 mujoco.yaml 복사본)
REAL_CAMERA_CONFIG = os.path.join(get_package_share_directory("pinky_wall_parking_sim"),
                                  "config", "mujoco_real_camera.yaml")
# 라이다 모델 -> 특성 파일 (빈 값 = 스캔 그대로)
LIDARS = {"ideal": "", "real_c1": "lidar_real_c1.yaml"}

# =========================================================
# 실행 시 계산: 월드 생성 -> 핑키 시뮬 포함
# =========================================================
def setup(context):
    def arg(name):
        text = LaunchConfiguration(name).perform(context).strip()
        return float(text) if text else None                                        # 빈 값 = None
    width_mm, gap_mm = arg("slot_width"), arg("side_gap")                           # 칸 폭, 방 옆 벽 틈 (mm)
    dist_mm, lat_mm, yaw_deg = arg("start_dist"), arg("start_lat"), arg("start_yaw")
    mouth_mm, end_mm = arg("mouth_width"), arg("side_end")                          # 실물 칸 모양 (mm)
    start = None if dist_mm is None else (dist_mm / 1000.0, lat_mm / 1000.0, math.radians(yaw_deg))
    mouth = None if mouth_mm is None else mouth_mm / 1000.0
    end   = None if end_mm is None else end_mm / 1000.0
    world_path  = WORLD_DIR / f"slot_{width_mm:g}mm.world"
    write_slot_world(world_path, width_mm / 1000.0, gap_mm / 1000.0, start, mouth, end)  # 범위 밖이면 오류
    _, entrance = start_layout(width_mm / 1000.0, gap_mm / 1000.0, start)           # 칸 입구 정답 (월드)
    print(f"[pinky_wall_parking_sim] slot width={width_mm:g} mm, mouth={mouth_mm} mm, side gap={gap_mm:g} mm, "
          f"side end={end_mm} mm, "
          f"start={start}, entrance=({entrance[0]:.3f}, {entrance[1]:.3f}, "
          f"{math.degrees(entrance[2]):.1f} deg), world={world_path}")

    pinky_launch = os.path.join(get_package_share_directory("pinky_mujoco"),
                                "launch", "launch_sim.launch.py")
    lidar = LaunchConfiguration("lidar_model").perform(context)
    if lidar not in LIDARS:                                                         # 오타를 launch 시작 때 알림
        raise ValueError(f"lidar_model must be one of {sorted(LIDARS)}: {lidar}")
    config = os.path.join(get_package_share_directory("pinky_wall_parking_sim"), "config", LIDARS[lidar])
    realism = [Node(package="pinky_wall_parking_sim", executable="scan_realism_node", name="scan_realism",
                    output="screen", parameters=[config, {"use_sim_time": True}],
                    remappings=[("scan_in", "/scan"), ("scan_out", "/scan_real")])] if LIDARS[lidar] else []
    return realism + [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(pinky_launch),
            launch_arguments={
                "world":   str(world_path),                                         # world_name 보다 우선
                "use_gpu": LaunchConfiguration("use_gpu").perform(context),
                "viewer":  LaunchConfiguration("viewer").perform(context),
                "camera":  LaunchConfiguration("camera").perform(context),
                "physics_config": LaunchConfiguration("physics_config").perform(context),
                "cam_tilt_deg":   LaunchConfiguration("cam_tilt_deg").perform(context),
            }.items(),
        ),
        Node(package="tf2_ros", executable="static_transform_publisher",
             name="slot_entrance_tf",
             arguments=["--x", f"{entrance[0]:.4f}", "--y", f"{entrance[1]:.4f}",
                        "--yaw", f"{entrance[2]:.6f}",
                        "--frame-id", "world", "--child-frame-id", "slot_entrance"]),
        Node(package="image_transport", executable="republish", name="camera_compress",
             parameters=[{"in_transport": "raw", "out_transport": "compressed",
                          "use_sim_time": True}],
             remappings=[("in", "camera/image_raw"),
                         ("out/compressed", "camera/image_raw/compressed")]),
    ]

# =========================================================
# launch 구성
# =========================================================
def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("slot_width", default_value="150",
                              description="Slot inner width in mm (0 < w <= 190)"),
        DeclareLaunchArgument("side_gap", default_value="50",
                              description="Gap between slot left wall and side room wall in mm"),
        DeclareLaunchArgument("start_dist", default_value="",
                              description="Robot base distance out of the slot entrance in mm (empty: original)"),
        DeclareLaunchArgument("start_lat", default_value="0",
                              description="Robot lateral offset from slot axis in mm"),
        DeclareLaunchArgument("start_yaw", default_value="0",
                              description="Robot yaw in deg (0: rear faces the slot)"),
        DeclareLaunchArgument("use_gpu", default_value="true",
                              description="true: hardware OpenGL. false: Mesa software rendering"),
        DeclareLaunchArgument("viewer", default_value="true",
                              description="Open the MuJoCo viewer window"),
        DeclareLaunchArgument("camera", default_value="true",
                              description="Publish /camera/image_raw"),
        DeclareLaunchArgument("physics_config", default_value=REAL_CAMERA_CONFIG,
                              description="MuJoCo settings YAML (empty: pinky default mujoco.yaml)"),
        DeclareLaunchArgument("mouth_width", default_value="",
                              description="Slot width at the entrance in mm (empty: slot_width, parallel walls)"),
        DeclareLaunchArgument("side_end", default_value="",
                              description="Room side wall end past the entrance line in mm (empty: board end)"),
        DeclareLaunchArgument("cam_tilt_deg", default_value="8",
                              description="Camera tilt down in deg (real Pinky estimate before calibration)"),
        DeclareLaunchArgument("lidar_model", default_value="ideal",
                              description="ideal: /scan as is. real_c1: also /scan_real with real RPLIDAR C1 traits"),
        OpaqueFunction(function=setup),
    ])