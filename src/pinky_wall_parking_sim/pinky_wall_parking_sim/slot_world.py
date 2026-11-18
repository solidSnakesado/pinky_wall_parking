"""
src/pinky_wall_parking_sim/pinky_wall_parking_sim/slot_world.py - 종이벽돌 벽 주차 칸 SDF 월드 생성

- 핑키 프로 월드 로더 (pinky_mujoco/world.py) 가 읽는 SDF 형식으로 상자를 만듦
- 실물 배치 (흰 판 1장 + 방 모서리) 를 그대로 재현
    흰 판      : 900 x 600 mm, visual 만 (바닥 색만 바꿈, 충돌·라이다 영향 없음)
    방 벽 2개  : 판 위쪽 끝 (뒤 벽), 판 왼쪽 끝 (옆 벽)
    칸 벽 3개  : 벽돌 (190 x 95 x 50 mm) 을 긴 옆면 (190 x 50) 이 바닥에 닿게 세워 2단
                 -> 벽 하나 190 x 50 x 190 mm (길이 x 두께 x 높이)
- 칸 뒤 벽은 방 뒤 벽에 붙임, 칸 왼쪽 벽은 방 옆 벽에서 side_gap 만큼 띄움
- 칸 깊이 190 mm, 칸 안쪽 폭은 인자 (최대 190 mm = 칸 뒤 벽 길이)
- 벽돌·방 벽은 link 하나에 visual (색, 카메라에 보임) + collision (라이다·충돌, 화면에서는 숨김)
- 시작 위치: 로봇은 원점 고정 (핑키 설정), 월드 전체를 model pose 로 옮겨 칸 입구 기준 로봇 자세를 맞춤
    start = (dist, lat, yaw): 입구 중심 -> 바깥쪽 dist, 옆 lat, yaw 0 = 로봇 뒷면이 칸을 향함

좌표 (월드, m, 로봇 시작 = 원점 +x 방향, 사진 기준 왼쪽 = +x, 아래 = +y):
  흰 판 x -0.10 ~ +0.80, y -0.40 ~ +0.20
  방 뒤 벽 y = -0.40 (판 위쪽 끝), 방 옆 벽 x = +0.80 (판 왼쪽 끝)
  칸 입구 y = -0.16 (+y 쪽을 봄), 칸 안쪽 = x: slot_x +- width/2, y: -0.35 ~ -0.16

실행 (파일 확인용):
  python3 slot_world.py --width 130 --out /tmp/slot_130.world
  python3 slot_world.py --width 131 --start 0.25 0.0 0 --out /tmp/slot_start.world
"""

# ==================== 표준 라이브러리 ======================
import argparse                                                                     # 명령줄 인자
import math                                                                         # 시작 위치 회전
from pathlib import Path                                                            # 파일 경로
import xml.etree.ElementTree as ET                                                  # SDF (XML) 작성

# ==================== 벽돌·칸 치수 (m) ======================
BRICK_LENGTH    = 0.190                                                             # 벽돌 가로 (벽 길이 방향)
BRICK_HEIGHT    = 0.095                                                             # 벽돌 세로 (세워서 높이)
BRICK_THICK     = 0.050                                                             # 벽돌 두께 (벽 두께)
LAYERS          = 2                                                                 # 쌓은 단 수 -> 벽 높이 190 mm
SLOT_DEPTH      = BRICK_LENGTH                                                      # 칸 깊이 = 옆 벽 길이
MAX_WIDTH       = BRICK_LENGTH                                                      # 칸 폭 최대 = 뒤 벽 길이

# ==================== 흰 판·방 벽 배치 (m) ==================
BOARD_X         = (-0.10, 0.80)                                                     # 흰 판 x 범위 (900 mm)
BOARD_Y         = (-0.40, 0.20)                                                     # 흰 판 y 범위 (600 mm)
BOARD_THICK     = 0.001                                                             # 흰 판 두께 (visual, 바닥 위)
ROOM_BACK_Y     = BOARD_Y[0]                                                        # 방 뒤 벽 앞면 y
ROOM_SIDE_X     = BOARD_X[1]                                                        # 방 옆 벽 앞면 x
ROOM_THICK      = 0.050                                                             # 방 벽 두께
ROOM_HEIGHT     = 0.500                                                             # 방 벽 높이
SIDE_GAP        = 0.050                                                             # 칸 왼쪽 벽 <-> 방 옆 벽 기본 틈

# 색 (RGBA)
COLORS = {
    "red":    "0.85 0.12 0.12 1",
    "yellow": "0.95 0.76 0.05 1",
    "green":  "0.10 0.50 0.22 1",
    "blue":   "0.10 0.30 0.80 1",
    "purple": "0.50 0.20 0.65 1",
    "white":  "0.95 0.95 0.95 1",                                                   # 흰 판
    "room":   "0.80 0.62 0.35 1",                                                   # 방 벽 (갈색 흡음판)
}
WALL_COLORS = {                                                                     # 사진 기준 (아래 단, 위 단)
    "left":  ["red", "red"],                                                        # 칸 왼쪽 벽 (+x 쪽)
    "right": ["yellow", "green"],                                                   # 칸 오른쪽 벽 (-x 쪽)
    "back":  ["red", "yellow"],                                                     # 칸 뒤 벽
}

# =========================================================
# 칸 입구 중심 계산
# =========================================================
def slot_opening(width, side_gap=SIDE_GAP):
    """칸 폭 width -> 입구 중심 (x, y). 칸은 방 모서리에 붙여 놓음."""
    if not 0.0 < width <= MAX_WIDTH:
        raise ValueError(f"slot width must be in (0, {MAX_WIDTH * 1000:.0f}] mm: {width * 1000:.1f} mm")
    slot_x  = ROOM_SIDE_X - side_gap - BRICK_THICK - width / 2                      # 방 옆 벽 -> 틈 -> 왼쪽 벽 -> 칸
    slot_y  = ROOM_BACK_Y + BRICK_THICK + SLOT_DEPTH                                # 방 뒤 벽 -> 칸 뒤 벽 -> 깊이
    return slot_x, slot_y

# =========================================================
# 칸 벽 3개의 중심·크기 계산
# =========================================================
def wall_layout(width, slot_x, slot_y):
    """벽 이름 -> (중심 x, 중심 y, 크기 x, 크기 y). 입구는 +y 쪽, 칸 안쪽은 -y 방향."""
    side_y  = slot_y - SLOT_DEPTH / 2                                               # 옆 벽 중심 y
    back_y  = slot_y - SLOT_DEPTH - BRICK_THICK / 2                                 # 뒤 벽 중심 y (옆 벽 뒤)
    offset  = width / 2 + BRICK_THICK / 2                                           # 칸 중심 -> 옆 벽 중심
    return {
        "left":  (slot_x + offset, side_y, BRICK_THICK, BRICK_LENGTH),              # 길이 방향 = y
        "right": (slot_x - offset, side_y, BRICK_THICK, BRICK_LENGTH),
        "back":  (slot_x, back_y, BRICK_LENGTH, BRICK_THICK),                       # 길이 방향 = x
    }

# =========================================================
# 시작 위치 -> 월드 model pose, 칸 입구 정답 자세
# =========================================================
def compose(a, b):
    """2D 자세 a * b. 자세 = (x, y, yaw)."""
    c, s = math.cos(a[2]), math.sin(a[2])
    return (a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], a[2] + b[2])

def inverse(a):
    """2D 자세 a 의 역."""
    c, s = math.cos(a[2]), math.sin(a[2])
    return (-c * a[0] - s * a[1], s * a[0] - c * a[1], -a[2])

def start_layout(width, side_gap=SIDE_GAP, start=None):
    """start (dist, lat, yaw) m, rad -> (model pose, 칸 입구 자세) 월드 기준. None 이면 원래 배치."""
    entrance = (*slot_opening(width, side_gap), math.pi / 2)                        # 원래 배치: 바깥 = +y
    if start is None:
        return (0.0, 0.0, 0.0), entrance
    entrance_new = inverse(start)                                                   # start = 칸 입구 기준 로봇, 로봇 = 원점
    return compose(entrance_new, inverse(entrance)), entrance_new

# =========================================================
# 상자 하나 = SDF link (collision 은 선택)
# =========================================================
def add_box(model, name, center, size, color, collision=True):
    """model 아래에 상자 link 추가. center, size = (x, y, z) m."""
    link = ET.SubElement(model, "link", name=name)
    ET.SubElement(link, "pose").text = " ".join(f"{v:.4f}" for v in center) + " 0 0 0"
    size_text = " ".join(f"{v:.4f}" for v in size)
    tags = ("collision", "visual") if collision else ("visual",)                    # 라이다·충돌 / 화면·카메라
    for tag in tags:
        element = ET.SubElement(link, tag, name=tag)
        box     = ET.SubElement(ET.SubElement(element, "geometry"), "box")
        ET.SubElement(box, "size").text = size_text                                 # 전체 크기 (x y z)
        if tag == "visual":
            material = ET.SubElement(element, "material")
            ET.SubElement(material, "diffuse").text = COLORS[color]
            ET.SubElement(material, "ambient").text = COLORS[color]

# =========================================================
# 흰 판·방 벽 추가
# =========================================================
def add_room(model):
    """흰 판 (visual 만) + 방 뒤 벽 + 방 옆 벽 (모서리까지 덮음)."""
    board_center = ((BOARD_X[0] + BOARD_X[1]) / 2, (BOARD_Y[0] + BOARD_Y[1]) / 2, BOARD_THICK / 2)
    board_size   = (BOARD_X[1] - BOARD_X[0], BOARD_Y[1] - BOARD_Y[0], BOARD_THICK)
    add_box(model, "board", board_center, board_size, "white", collision=False)     # 바닥 색만

    back_x0, back_x1 = BOARD_X[0], ROOM_SIDE_X + ROOM_THICK                         # 모서리까지
    add_box(model, "room_back",
            ((back_x0 + back_x1) / 2, ROOM_BACK_Y - ROOM_THICK / 2, ROOM_HEIGHT / 2),
            (back_x1 - back_x0, ROOM_THICK, ROOM_HEIGHT), "room")
    side_y0, side_y1 = ROOM_BACK_Y - ROOM_THICK, BOARD_Y[1]                         # 모서리부터
    add_box(model, "room_side",
            (ROOM_SIDE_X + ROOM_THICK / 2, (side_y0 + side_y1) / 2, ROOM_HEIGHT / 2),
            (ROOM_THICK, side_y1 - side_y0, ROOM_HEIGHT), "room")

# =========================================================
# SDF 문자열 만들기
# =========================================================
def build_slot_sdf(width, side_gap=SIDE_GAP, start=None):
    """칸 폭 width (m), 방 옆 벽과의 틈 side_gap (m), 시작 위치 start -> SDF 문자열."""
    slot_x, slot_y = slot_opening(width, side_gap)
    pose, _ = start_layout(width, side_gap, start)
    sdf     = ET.Element("sdf", version="1.6")
    world   = ET.SubElement(sdf, "world", name="wall_slot")
    model   = ET.SubElement(world, "model", name="slot")
    ET.SubElement(model, "static").text = "true"                                    # 움직이지 않는 벽
    ET.SubElement(model, "pose").text = f"{pose[0]:.4f} {pose[1]:.4f} 0 0 0 {pose[2]:.6f}"
    add_room(model)
    for wall, (x, y, sx, sy) in wall_layout(width, slot_x, slot_y).items():
        for layer in range(LAYERS):                                                 # 아래 단부터
            z = BRICK_HEIGHT * (layer + 0.5)                                        # 벽돌 중심 높이
            add_box(model, f"{wall}_{layer}", (x, y, z), (sx, sy, BRICK_HEIGHT),
                    WALL_COLORS[wall][layer])
    ET.indent(sdf, space="  ")                                                      # 사람이 읽기 쉽게
    return '<?xml version="1.0"?>\n' + ET.tostring(sdf, encoding="unicode") + "\n"

# =========================================================
# SDF 파일로 저장
# =========================================================
def write_slot_world(path, width, side_gap=SIDE_GAP, start=None):
    """SDF 를 path 에 저장하고 Path 반환."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_slot_sdf(width, side_gap, start))
    return path

# =========================================================
# 명령줄 실행 (파일 확인용)
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Write a wall slot SDF world")
    parser.add_argument("--width", type=float, default=150.0, help="slot inner width (mm)")
    parser.add_argument("--gap", type=float, default=SIDE_GAP * 1000, help="slot <-> side room wall (mm)")
    parser.add_argument("--start", type=float, nargs=3, help="dist lat (m) yaw (deg)")
    parser.add_argument("--out", default="/tmp/slot.world", help="output SDF path")
    args = parser.parse_args()
    width, gap = args.width / 1000.0, args.gap / 1000.0
    start = None if args.start is None else (args.start[0], args.start[1], math.radians(args.start[2]))
    path = write_slot_world(args.out, width, gap, start)
    print(f"wrote {path}, entrance pose {start_layout(width, gap, start)[1]}")

if __name__ == "__main__":
    main()