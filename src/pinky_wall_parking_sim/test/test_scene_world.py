"""src/pinky_wall_parking_sim/test/test_scene_world.py - scene_world 단위 시험

역할: 장면 YAML -> SDF 변환 (상자 수, 바닥 라인, 세워 둔 핑키 링크·변환, 출발 자세 이동) 을 확인한다.
방법: 핑키 URDF 와 같은 구조 (base_footprint -> base_link 고정, 바퀴 회전 관절, 위쪽 라이다) 의 작은 URDF 를 씀 (xacro·ROS 불필요).
실행: cd ~/pinky_wall_parking/src/pinky_wall_parking_sim && python3 -m pytest -q test/test_scene_world.py
"""

# --- 표준 라이브러리 ---------------------------------------------------------------
import xml.etree.ElementTree as ET                                                  # SDF 읽기

# --- 외부 라이브러리 ---------------------------------------------------------------
import numpy as np                                                                  # 비교
import pytest                                                                       # 시험 틀

# --- 이 패키지 ---------------------------------------------------------------------
from pinky_wall_parking_sim.scene_world import build_scene_sdf                      # 장면 -> SDF
from pinky_wall_parking_sim.scene_world import pinky_links                          # URDF 링크 변환
from pinky_wall_parking_sim.scene_world import scene_poses                          # 출발 기준 이동


URDF = """<robot name="mini">
  <link name="base_footprint"/>
  <link name="base_link"><collision><geometry><box size="0.09 0.08 0.086"/></geometry></collision></link>
  <link name="left_wheel"><collision><origin xyz="0 0.008 0" rpy="1.5708 0 0"/>
    <geometry><cylinder radius="0.028" length="0.0145"/></geometry></collision></link>
  <link name="rplidar_mount"><visual><geometry><mesh filename="package://p/m/lidar.dae"/></geometry></visual>
    <collision><geometry><cylinder radius="0.025" length="0.042"/></geometry></collision></link>
  <joint name="j0" type="fixed"><parent link="base_footprint"/><child link="base_link"/>
    <origin xyz="0 0 0.028"/></joint>
  <joint name="j1" type="continuous"><parent link="base_link"/><child link="left_wheel"/>
    <origin xyz="0 0.04055 0"/></joint>
  <joint name="j2" type="fixed"><parent link="base_link"/><child link="rplidar_mount"/>
    <origin xyz="-0.017 0 0.05"/></joint>
</robot>"""
SCENE = {"name": "t", "board": {"x": [0, 1], "y": [0, 1]},
         "boxes": [{"center": [0.5, -0.3, 0.1], "size": [1.0, 0.05, 0.2]}],
         "parked": [{"x": 0.30, "y": 0.0}, {"x": 0.75, "y": 0.0, "yaw_deg": 90}],
         "start": {"x": -0.30, "y": 0.25, "yaw_deg": 0}}


def test_pinky_links():
    links = {name: m for name, m, _ in pinky_links(URDF)}
    assert set(links) == {"base_footprint", "base_link", "left_wheel", "rplidar_mount"}
    assert np.allclose(links["rplidar_mount"][:3, 3], [-0.017, 0, 0.078])           # 관절 연결 (0.028 + 0.05)
    assert np.allclose(links["left_wheel"][:3, 3], [0, 0.04055, 0.028])


def test_scene_poses():
    shift, parked = scene_poses(SCENE)
    assert parked[0] == pytest.approx((0.60, -0.25, 0.0))                           # 로봇 출발 = 원점
    assert parked[1][:2] == pytest.approx((1.05, -0.25))


def test_build_scene_sdf():
    sdf    = ET.fromstring(build_scene_sdf(SCENE, URDF))
    models = {m.get("name"): m for m in sdf.find("world").findall("model")}
    assert set(models) == {"scene", "parked_0", "parked_1"}
    assert len(models["scene"].findall("link")) == 2                                # 흰 판 + 상자 1
    links = models["parked_0"].findall("link")
    assert sorted(link.get("name") for link in links) == ["base_link", "left_wheel", "rplidar_mount"]
    uris = [u.text for u in models["parked_0"].iter("uri")]
    assert uris == ["package://p/m/lidar.dae"]                                      # mesh 경로 그대로
    assert models["parked_1"].findtext("pose").split()[:2] == ["1.0500", "-0.2500"]


def test_lines_are_visual_only():
    scene = {"lines": [{"p0": [0.0, 0.0], "p1": [0.3, 0.0]}, {"p0": [0.0, 0.0], "p1": [0.1, 0.1], "width": 0.01}]}
    model = ET.fromstring(build_scene_sdf(scene)).find("world").find("model")
    lines = [link for link in model.findall("link") if link.get("name").startswith("line_")]
    assert len(lines) == 2
    assert all(link.find("collision") is None for link in lines)                    # 충돌·라이다 없음
    assert lines[0].findtext("visual/geometry/box/size") == "0.3050 0.0050 0.0002"  # 길이 + 폭 (끝 덮음)
    assert float(lines[0].findtext("pose").split()[2]) + 0.0001 <= 0.0004           # 선 윗면 (카메라 역투영)
    pose = [float(v) for v in lines[1].findtext("pose").split()]
    assert pose[:2] == pytest.approx([0.05, 0.05]) and pose[5] == pytest.approx(0.785398, abs=1e-5)


def test_no_parked_needs_no_urdf():
    sdf = ET.fromstring(build_scene_sdf({"boxes": []}))
    assert [m.get("name") for m in sdf.find("world").findall("model")] == ["scene"]
