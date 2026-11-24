"""src/pinky_wall_parking_front/tools/cmd_vel_watchdog.py - 핑키 명령 시간 초과 정지 (핑키에서 직접 실행)

역할: 핑키 구동부에는 명령 시간 초과가 없어 마지막 cmd_vel 을 계속 따른다. PC 쪽 노드가 죽거나 Wi-Fi 가 끊기면
      로봇이 멈추지 않으므로, 핑키 안에서 cmd_vel 을 지켜보다가 끊기면 0 속도를 보낸다.
방법: cmd_vel 을 구독한다. 마지막 명령이 0 이 아닌데 timeout_sec 동안 새 명령이 없으면 0 속도를 몇 번 보낸다.
      0 을 보낸 뒤에는 다시 0 이 아닌 명령이 올 때까지 아무것도 보내지 않는다 (손 조작과 부딪히지 않음).
      시간은 time.monotonic 을 쓴다 (핑키 시계가 맞지 않거나 바뀌어도 영향 없음).
      ros2_control diff_drive_controller 의 cmd_vel_timeout (기본 0.5 s) 과 같은 동작.
실행: 빌드·설치 없이 파일 하나로 실행 (rclpy, geometry_msgs 는 핑키 ROS 2 에 있음)
      python3 ~/cmd_vel_watchdog.py --ros-args -p timeout_sec:=0.5
"""

# --- 표준 라이브러리 ---------------------------------------------------------------
import time                                                                         # 끊김 시간 (monotonic)

# --- ROS 2 -------------------------------------------------------------------------
from geometry_msgs.msg import Twist                                                 # 속도 명령
import rclpy                                                                        # ROS 2 파이썬
from rclpy.node import Node                                                         # 노드 기본 클래스


def is_zero(msg):
    return msg.linear.x == 0.0 and msg.linear.y == 0.0 and msg.angular.z == 0.0


class CmdVelWatchdog(Node):

    def __init__(self):
        super().__init__("cmd_vel_watchdog")
        self.timeout = self.declare_parameter("timeout_sec", 0.5).value             # 명령 끊김 한도 (s)
        self.repeat  = self.declare_parameter("stop_repeat", 5).value               # 0 속도 보내는 횟수
        self.last    = 0.0                                                          # 마지막 0 아닌 명령 시각
        self.active  = False                                                        # 0 아닌 명령을 따르는 중
        self.pub     = self.create_publisher(Twist, "cmd_vel", 10)
        self.create_subscription(Twist, "cmd_vel", self.on_cmd, 10)
        self.create_timer(0.05, self.on_timer)                                      # 20 Hz 검사
        self.get_logger().info(f"cmd_vel_watchdog ready: timeout {self.timeout} s")

    def on_cmd(self, msg):
        if is_zero(msg):
            self.active = False                                                     # 0 명령 (자기 것 포함) = 이미 멈춤
            return
        self.last, self.active = time.monotonic(), True

    def on_timer(self):
        if not self.active or time.monotonic() - self.last <= self.timeout:
            return
        self.active = False
        for _ in range(self.repeat):
            self.pub.publish(Twist())
        self.get_logger().warn(f"cmd_vel stale > {self.timeout} s: sent zero velocity")


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelWatchdog()
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
