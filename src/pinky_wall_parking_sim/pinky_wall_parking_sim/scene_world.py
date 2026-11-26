"""src/pinky_wall_parking_sim/pinky_wall_parking_sim/scene_world.py - 장면 YAML -> SDF 월드 (세워 둔 핑키 포함)

역할: 주차 시나리오 장면 (흰 판, 상자 벽, 세워 둔 핑키, 로봇 출발 자세) 을 YAML 로 적으면 핑키 MuJoCo 월드 로더가 읽는 SDF 를 만든다.
방법: - 상자: slot_world.add_box 그대로 (visual + collision).
      - 바닥 라인 (주차 칸 선): 두께 0.2 mm visual 상자, 윗면 0.4 mm (collision 없음 -> 충돌·라이다에 안 잡힘, 카메라·화면에만 보임).
        흰 판도 visual 만, 윗면 0.2 mm. 로봇은 z = 0 바닥을 달리므로 선이 높으면 카메라 역투영 거리가 길어짐
        (예전 윗면 2.2 mm -> 0.6 m 앞 약 +20 mm). 실물은 로봇·선 모두 판 위 (테이프 두께만 차이)
      - 세워 둔 핑키: 핑키 URDF (펼친 xacro) 의 링크별 visual / collision 을 고정 관절 (각도 0) 위치에 놓은 정지 model.
        mesh 는 package:// 경로 그대로 (월드 로더가 visual mesh 변환, collision mesh 는 라이다·충돌에 씀).
      - 로봇은 원점에 고정 (핑키 설정) 이므로 장면 전체를 start 의 역자세로 옮김.
장면 YAML: board {x: [x0, x1], y: [y0, y1]}, boxes [{name, center [x, y, z], size [x, y, z], color}],
           lines [{p0 [x, y], p1 [x, y], width, color}], parked [{x, y, yaw_deg}], start {x, y, yaw_deg}
           (m, deg, 장면 좌표)
실행: 라이브러리 (scene_sim.launch.py, 단위 시험에서 호출)
"""

# --- 표준 라이브러리 ---------------------------------------------------------------
import math                                                                         # 각도 변환
from pathlib import Path                                                            # 파일 경로
import xml.etree.ElementTree as ET                                                  # SDF, URDF (XML)

# --- 외부 라이브러리 ---------------------------------------------------------------
import numpy as np                                                                  # 변환 행렬
from scipy.spatial.transform import Rotation                                        # rpy <-> 행렬
import yaml                                                                         # 장면 파일

# --- 이 패키지 ---------------------------------------------------------------------
from pinky_wall_parking_sim.slot_world import add_box                               # 상자 link
from pinky_wall_parking_sim.slot_world import COLORS                                # 색
from pinky_wall_parking_sim.slot_world import compose                               # 2D 자세 곱
from pinky_wall_parking_sim.slot_world import inverse                               # 2D 자세 역


def matrix(xyz, rpy):
    m = np.eye(4)
    m[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()                         # URDF rpy = 고정축 x, y, z
    m[:3, 3] = xyz
    return m


def origin_of(element):
    o = element.find("origin") if element is not None else None
    xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
    rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
    return matrix(xyz, rpy)


def sdf_pose(m):
    rpy = Rotation.from_matrix(m[:3, :3]).as_euler("xyz")
    return " ".join(f"{v:.6f}" for v in (*m[:3, 3], *rpy))


# ==================================================================================
# 핑키 URDF -> 링크별 (이름, base_footprint 기준 변환, visual / collision 목록)
# ==================================================================================
def pinky_links(urdf_xml, root="base_footprint"):
    """모든 관절을 각도 0 으로 두고 root 기준 링크 변환을 구함. 반환: [(이름, 4x4, link 요소)]."""
    robot    = ET.fromstring(urdf_xml)
    links    = {link.get("name"): link for link in robot.findall("link")}
    children = {}
    for joint in robot.findall("joint"):
        children.setdefault(joint.find("parent").get("link"), []).append(
            (joint.find("child").get("link"), origin_of(joint)))
    out, stack = [], [(root, np.eye(4))]
    while stack:
        name, m = stack.pop()
        out.append((name, m, links[name]))
        stack += [(child, m @ o) for child, o in children.get(name, [])]
    return out


def add_geometry(parent, tag, element):
    """URDF visual / collision -> SDF visual / collision (mesh, box, cylinder, sphere)."""
    geom = element.find("geometry")[0]
    node = ET.SubElement(parent, tag, name=f"{tag}_{len(parent.findall(tag))}")
    ET.SubElement(node, "pose").text = sdf_pose(origin_of(element))
    shape = ET.SubElement(ET.SubElement(node, "geometry"), geom.tag)
    if geom.tag == "mesh":
        ET.SubElement(shape, "uri").text = geom.get("filename")
        ET.SubElement(shape, "scale").text = geom.get("scale", "1 1 1")
    elif geom.tag == "box":
        ET.SubElement(shape, "size").text = geom.get("size")
    elif geom.tag == "cylinder":
        ET.SubElement(shape, "radius").text = geom.get("radius")
        ET.SubElement(shape, "length").text = geom.get("length")
    elif geom.tag == "sphere":
        ET.SubElement(shape, "radius").text = geom.get("radius")


def add_line(model, name, p0, p1, width, color):
    """바닥 라인 = p0 -> p1 방향 visual 상자 (두께 0.2 mm, 판 윗면 0.2 mm 위에 얹힘). collision 없음."""
    (x0, y0), (x1, y1) = p0, p1                                                     # 양 끝
    link = ET.SubElement(model, "link", name=name)
    yaw  = math.atan2(y1 - y0, x1 - x0)                                             # 라인 방향
    ET.SubElement(link, "pose").text = f"{(x0 + x1) / 2:.4f} {(y0 + y1) / 2:.4f} 0.0003 0 0 {yaw:.6f}"
    element = ET.SubElement(link, "visual", name="visual")
    box = ET.SubElement(ET.SubElement(element, "geometry"), "box")
    ET.SubElement(box, "size").text = f"{math.hypot(x1 - x0, y1 - y0) + width:.4f} {width:.4f} 0.0002"
    material = ET.SubElement(element, "material")
    ET.SubElement(material, "diffuse").text = COLORS[color]
    ET.SubElement(material, "ambient").text = COLORS[color]


def add_parked(world, name, pose, links):
    """세워 둔 핑키 하나 = 정지 model. pose = (x, y, yaw) 월드."""
    model = ET.SubElement(world, "model", name=name)
    ET.SubElement(model, "static").text = "true"
    ET.SubElement(model, "pose").text = f"{pose[0]:.4f} {pose[1]:.4f} 0 0 0 {pose[2]:.6f}"
    for link_name, m, link in links:
        if link.find("visual") is None and link.find("collision") is None:
            continue
        node = ET.SubElement(model, "link", name=link_name)
        ET.SubElement(node, "pose").text = sdf_pose(m)
        for tag in ("collision", "visual"):
            for element in link.findall(tag):
                add_geometry(node, tag, element)


# ==================================================================================
# 장면 -> SDF
# ==================================================================================
def load_scene(path):
    return yaml.safe_load(Path(path).expanduser().read_text())


def scene_poses(scene):
    """장면 -> (장면 -> 월드 변환 자세, 세워 둔 핑키 월드 자세 목록). 로봇 출발 = 월드 원점."""
    s     = scene.get("start", {})
    start = (s.get("x", 0.0), s.get("y", 0.0), math.radians(s.get("yaw_deg", 0.0)))
    shift = inverse(start)                                                          # 장면 -> 로봇 출발 기준
    parked = [compose(shift, (p["x"], p["y"], math.radians(p.get("yaw_deg", 0.0)))) for p in scene.get("parked", [])]
    return shift, parked


def build_scene_sdf(scene, urdf_xml=None):
    """장면 dict (+ 핑키 URDF 문자열, 세워 둔 핑키가 있을 때) -> SDF 문자열."""
    shift, parked = scene_poses(scene)
    sdf   = ET.Element("sdf", version="1.6")
    world = ET.SubElement(sdf, "world", name=scene.get("name", "scene"))
    model = ET.SubElement(world, "model", name="scene")
    ET.SubElement(model, "static").text = "true"
    ET.SubElement(model, "pose").text = f"{shift[0]:.4f} {shift[1]:.4f} 0 0 0 {shift[2]:.6f}"
    if "board" in scene:
        (x0, x1), (y0, y1) = scene["board"]["x"], scene["board"]["y"]
        add_box(model, "board", ((x0 + x1) / 2, (y0 + y1) / 2, 0.0001), (x1 - x0, y1 - y0, 0.0002),
                "white", collision=False)                                           # 바닥 색만
    for i, box in enumerate(scene.get("boxes", [])):
        add_box(model, box.get("name", f"box_{i}"), tuple(box["center"]), tuple(box["size"]),
                box.get("color", "room"))
    for i, line in enumerate(scene.get("lines", [])):
        add_line(model, line.get("name", f"line_{i}"), line["p0"], line["p1"], line.get("width", 0.005),
                 line.get("color", "blue"))
    if parked:
        links = pinky_links(urdf_xml)
        for i, pose in enumerate(parked):
            add_parked(world, f"parked_{i}", pose, links)
    ET.indent(sdf, space="  ")
    return '<?xml version="1.0"?>\n' + ET.tostring(sdf, encoding="unicode") + "\n"


def write_scene_world(path, scene, urdf_xml=None):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_scene_sdf(scene, urdf_xml))
    return path
