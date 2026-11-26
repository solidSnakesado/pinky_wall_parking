"""src/pinky_wall_parking_vision/pinky_wall_parking_vision/slot_vision_node.py - 카메라 영상 -> 벽 칸 특징 (ROS 2 노드)

역할: 전방 카메라 압축 영상에서 벽 칸 특징 (뒤 벽 경계, 뒤쪽 모서리, 소실점) 을 찾아 발행한다.
- 구독 : camera/image_raw/compressed (sensor_msgs/CompressedImage, best-effort, 최신 1장만)
- 발행 : slot_vision/features         (pinky_wall_parking_interfaces/SlotFeatures, header = 입력 영상 header)
         slot_vision/debug/compressed (sensor_msgs/CompressedImage, 확인용 그림, publish_debug 가 true 일 때만)
방법: cv_bridge 로 JPEG 를 풀고 slot_features.extract_features 로 특징을 찾는다 (ROS 없는 라이브러리).
      카메라 옆 기울기는 로봇별 파라미터 roll_deg 로 보정한다 (1호기 약 -1.4, 2호기·시뮬 0).
실행: ros2 run pinky_wall_parking_vision slot_vision_node --ros-args -p roll_deg:=-1.4
      (파라미터 파일은 --params-file src/pinky_wall_parking_vision/config/slot_vision.yaml)
"""

# --- 표준 라이브러리 ---------------------------------------------------------------
from dataclasses import fields                                                      # FeatureParams 항목 목록

# --- ROS 2 -------------------------------------------------------------------------
from cv_bridge import CvBridge, CvBridgeError                                       # 압축 영상 <-> OpenCV
from pinky_wall_parking_interfaces.msg import SlotFeatures as SlotFeaturesMsg       # 특징 메시지
import rclpy                                                                        # ROS 2 파이썬
from rclpy.node import Node                                                         # 노드 기본 클래스
from rclpy.qos import qos_profile_sensor_data, QoSProfile                           # best-effort QoS
from sensor_msgs.msg import CompressedImage                                         # 압축 영상

# --- 이 패키지 ---------------------------------------------------------------------
from pinky_wall_parking_vision.slot_draw import draw_features                       # 확인용 그림
from pinky_wall_parking_vision.slot_features import extract_features                # 특징 찾기
from pinky_wall_parking_vision.slot_features import FeatureParams                   # 검출 파라미터
from pinky_wall_parking_vision.slot_features import level_image                     # 옆 기울기 보정


# ==================================================================================
# 라이브러리 결과 -> 메시지
# ==================================================================================
def to_msg(f, header):
    msg              = SlotFeaturesMsg()
    msg.header       = header                                                       # 입력 영상 시각 그대로
    msg.image_width  = f.width
    msg.image_height = f.height
    msg.back_found   = f.back_found
    msg.back_y       = f.back_y
    msg.back_slope   = f.back_slope
    msg.left_found   = f.left_found
    msg.right_found  = f.right_found
    msg.left_x       = f.left_x
    msg.right_x      = f.right_x
    msg.center_x     = f.center_x
    msg.vp_found     = f.vp_found
    msg.vp.x         = f.vp_x
    msg.vp.y         = f.vp_y
    return msg


# ==================================================================================
# ROS 2 노드
# ==================================================================================
class SlotVisionNode(Node):

    def __init__(self):
        super().__init__("slot_vision")
        defaults           = FeatureParams()
        values             = {}
        for item in fields(FeatureParams):                                          # 파라미터 이름 = 항목 이름
            values[item.name] = self.declare_parameter(item.name, getattr(defaults, item.name)).value
        self.params        = FeatureParams(**values)
        self.publish_debug = self.declare_parameter("publish_debug", True).value
        self.bridge        = CvBridge()
        self.frames        = 0                                                      # 받은 영상 수 (로그용)

        qos                = QoSProfile(depth=1)                                    # 처리가 밀려도 최신 1장만
        qos.reliability    = qos_profile_sensor_data.reliability                    # best-effort
        qos.durability     = qos_profile_sensor_data.durability
        self.feat_pub      = self.create_publisher(SlotFeaturesMsg, "slot_vision/features", 10)
        self.debug_pub     = self.create_publisher(CompressedImage, "slot_vision/debug/compressed", 1)
        self.create_subscription(CompressedImage, "camera/image_raw/compressed", self.on_image, qos)
        self.get_logger().info(f"slot_vision ready: roll_deg {self.params.roll_deg}, "
                               f"publish_debug {self.publish_debug}")

    # ------------------------------------------------------------------------------
    # 영상 한 장 처리
    # ------------------------------------------------------------------------------
    def on_image(self, msg):
        try:
            image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            self.get_logger().warn(f"decode failed: {e}", throttle_duration_sec=5.0)
            return
        f = extract_features(image, self.params)
        self.feat_pub.publish(to_msg(f, msg.header))
        self.frames += 1
        if self.frames == 1:
            self.get_logger().info(f"first image {f.width}x{f.height}, back_found {f.back_found}")
        if self.publish_debug and self.debug_pub.get_subscription_count() > 0:
            out        = draw_features(level_image(image, self.params), f)          # 특징 좌표 = 보정한 영상 기준
            dbg        = self.bridge.cv2_to_compressed_imgmsg(out, dst_format="jpg")
            dbg.header = msg.header
            self.debug_pub.publish(dbg)


# ==================================================================================
# 실행
# ==================================================================================
def main(args=None):
    rclpy.init(args=args)
    node = SlotVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
