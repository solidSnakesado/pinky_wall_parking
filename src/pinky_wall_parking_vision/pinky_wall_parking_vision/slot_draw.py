"""src/pinky_wall_parking_vision/pinky_wall_parking_vision/slot_draw.py - 칸 특징 확인용 그림

역할: slot_features 결과 (SlotFeatures) 를 영상 위에 그린다 (slot_vision_node 의 확인용 영상).
그림: 회색 = 화면 가운데, 분홍 = 열별 경계점, 초록 = 뒤 벽 경계·모서리 (빨강 원 = 화면 밖),
      파랑 = 옆 벽 바닥 경계선, 노랑 원 = 소실점.
실행: 라이브러리 (slot_vision_node 에서 호출)
"""

# --- 외부 라이브러리 ---------------------------------------------------------------
import cv2                                                                          # 선·원 그리기


# ==================================================================================
# 확인용 그림
# ==================================================================================
def draw_features(image, f):
    out = image.copy()
    w   = f.width
    cv2.line(out, (w // 2, 0), (w // 2, f.height - 1), (200, 200, 200), 1)          # 화면 가운데
    if f.envelope is not None:
        for x, y in enumerate(f.envelope):
            if y >= 0:
                cv2.circle(out, (x, int(y)), 1, (255, 0, 255), -1)                  # 경계 점
    if not f.back_found:
        return out
    ys = [f.back_y + f.back_slope * (x - f.center_x) for x in (f.left_x, f.right_x)]
    pl = (int(f.left_x), int(ys[0]))
    pr = (int(f.right_x), int(ys[1]))
    cv2.line(out, pl, pr, (0, 255, 0), 2)                                           # 뒤 벽 경계
    for pt, seen in ((pl, f.left_found), (pr, f.right_found)):
        cv2.circle(out, pt, 6, (0, 255, 0) if seen else (0, 0, 255), 2)             # 모서리 (빨강 = 안 보임)
    for name, (m, c) in f.side_lines.items():
        x0, x1 = (0, pl[0]) if name == "L" else (pr[0], w - 1)
        cv2.line(out, (x0, int(m * x0 + c)), (x1, int(m * x1 + c)), (255, 128, 0), 2)
    if f.vp_found:
        cv2.circle(out, (int(f.vp_x), int(f.vp_y)), 6, (0, 255, 255), 2)            # 소실점
    return out
