"""src/pinky_wall_parking_maneuver/pinky_wall_parking_maneuver/handoff.py - 정렬 자세 도착 후 진입 제어로 넘기기

역할: 후면 = Nav2 Docking Server (nav2_msgs/action/DockRobot, dock_robot) 에 칸 안 진입 요청.
      전면 = 전방 주차 노드 (std_srvs/Trigger, front_parking/start) 시작 요청 (카메라 + 초음파).
방법: 모두 비동기 요청. 결과를 문자열로 report(state) 에 돌려준다.
실행: 라이브러리 (t_parking_node 에서 호출)
"""

# --- ROS 2 -------------------------------------------------------------------------
from nav2_msgs.action import DockRobot                                              # 후면 진입
from rclpy.action import ActionClient                                               # 액션 요청
from std_srvs.srv import Trigger                                                    # 전면 시작 요청


class Handoff:

    def __init__(self, node, front, dock_type, report):
        self.node, self.front, self.report = node, front, report
        self.dock_type = dock_type                                                  # Docking 플러그인 이름
        self.dock      = ActionClient(node, DockRobot, "dock_robot")
        self.front_srv = node.create_client(Trigger, "front_parking/start")

    def start(self):
        if self.front:
            self.start_front()
        else:
            self.start_dock()

    # --- 전면: 전방 주차 노드 -------------------------------------------------------
    def start_front(self):
        if not self.front_srv.wait_for_service(timeout_sec=2.0):
            self.report("front_parking/start not available")
            return
        self.node.get_logger().info("front parking start requested")
        self.front_srv.call_async(Trigger.Request()).add_done_callback(self.on_front)

    def on_front(self, future):
        res = future.result()
        self.report("front parking started" if res.success else f"front parking start failed ({res.message})")

    # --- 후면: Docking Server --------------------------------------------------------
    def start_dock(self):
        if not self.dock.wait_for_server(timeout_sec=2.0):
            self.report("dock_robot action server not available")
            return
        goal = DockRobot.Goal()
        goal.use_dock_id, goal.dock_type = False, self.dock_type
        goal.navigate_to_staging_pose  = False                                      # 이미 정렬 자세
        goal.dock_pose.header.frame_id = "odom"
        self.node.get_logger().info("dock request sent")
        self.dock.send_goal_async(goal).add_done_callback(self.on_dock_accepted)

    def on_dock_accepted(self, future):
        handle = future.result()
        if not handle.accepted:
            self.report("dock request rejected")
            return
        handle.get_result_async().add_done_callback(self.on_dock_result)

    def on_dock_result(self, future):
        result = future.result().result
        state  = "docked" if result.success else f"dock failed (error {result.error_code})"
        self.report(f"{state}, retries {result.num_retries}")
