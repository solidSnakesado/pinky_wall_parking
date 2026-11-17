"""
src/pinky_wall_parking_sim/launch/slot_sim.launch.py - 종이벽돌 벽 주차 칸 월드로 핑키 MuJoCo 시뮬 실행

- 실행 순서
    1. 인자 (칸 폭, 방 옆 벽 틈) 로 SDF 월드 생성 (slot_world.write_slot_world)
       -> ~/.cache/pinky_wall_parking/slot_<폭>mm.world 에 저장
    2. 핑키 프로 시뮬 launch (pinky_mujoco/launch_sim.launch.py) 를 world:=<저장 경로> 로 포함
- 핑키 프로 파일은 수정하지 않고 launch 를 불러다 사용 (핑키 launch 의 URDF 저장 방식과 같은 구조)
- use_gpu 기본값 true (이 PC 확인 결과: 소프트웨어 렌더링은 카메라 13.5 Hz 로 제한)

실행:
  ros2 launch pinky_wall_parking_sim slot_sim.launch.py
  칸 폭 지정: slot_width:=130   (mm, 0 초과 190 이하)
  칸 왼쪽 벽 <-> 방 옆 벽 틈: side_gap:=50   (mm)
  창 없이: viewer:=false,  카메라 끄기: camera:=false
"""

# ==================== 표준 라이브러리 ======================
import os                                                                           # 경로 결합
from pathlib import Path                                                            # 월드 저장 폴더

# ==================== ROS 2 launch =========================
from ament_index_python.packages import get_package_share_directory                 # 패키지 설치 경로
from launch import LaunchDescription                                                # 실행 구성
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription          # 인자, 다른 launch 포함
from launch.actions import OpaqueFunction                                           # 실행 시 계산
from launch.launch_description_sources import PythonLaunchDescriptionSource         # .launch.py 읽기
from launch.substitutions import LaunchConfiguration                                # 인자 값 읽기

# ==================== 이 패키지 ============================
from pinky_wall_parking_sim.slot_world import slot_opening, write_slot_world        # SDF 월드 생성

# 월드 저장 폴더
WORLD_DIR = Path.home() / ".cache" / "pinky_wall_parking"

# =========================================================
# 실행 시 계산: 월드 생성 -> 핑키 시뮬 포함
# =========================================================
def setup(context):
    width_mm    = float(LaunchConfiguration("slot_width").perform(context))         # 칸 폭 (mm)
    gap_mm      = float(LaunchConfiguration("side_gap").perform(context))           # 방 옆 벽 틈 (mm)
    world_path  = WORLD_DIR / f"slot_{width_mm:g}mm.world"
    write_slot_world(world_path, width_mm / 1000.0, gap_mm / 1000.0)                # 범위 밖이면 오류
    slot_x, slot_y = slot_opening(width_mm / 1000.0, gap_mm / 1000.0)               # 입구 중심 (m)
    print(f"[pinky_wall_parking_sim] slot width={width_mm:g} mm, side gap={gap_mm:g} mm, "
          f"opening=({slot_x:.3f}, {slot_y:.3f}) m, world={world_path}")

    pinky_launch = os.path.join(get_package_share_directory("pinky_mujoco"),
                                "launch", "launch_sim.launch.py")
    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(pinky_launch),
            launch_arguments={
                "world":   str(world_path),                                         # world_name 보다 우선
                "use_gpu": LaunchConfiguration("use_gpu").perform(context),
                "viewer":  LaunchConfiguration("viewer").perform(context),
                "camera":  LaunchConfiguration("camera").perform(context),
            }.items(),
        ),
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
        DeclareLaunchArgument("use_gpu", default_value="true",
                              description="true: hardware OpenGL. false: Mesa software rendering"),
        DeclareLaunchArgument("viewer", default_value="true",
                              description="Open the MuJoCo viewer window"),
        DeclareLaunchArgument("camera", default_value="true",
                              description="Publish /camera/image_raw"),
        OpaqueFunction(function=setup),
    ])