// src/pinky_wall_parking_dock_plugin/src/slot_dock.cpp - 벽 주차 칸 dock 플러그인 구현
// - 원본 SimpleNonChargingDock 의 감지·필터/odom 변환·정렬 위치 계산은 그대로 사용
// - 바꾼 것: 목표 자세 고정 (getRefinedPose), 목표 선 통과 정지 판정 (isDocked), 고정 해제 (getStagingPose)

#include "pinky_wall_parking_dock_plugin/slot_dock.hpp"

#include <cmath>

namespace pinky_wall_parking_dock_plugin
{

// ---- 설정: 원본 설정 + 고정 거리, 정지 허용값 ----------------
void SlotDock::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
  const std::string & name, std::shared_ptr<tf2_ros::Buffer> tf)
{
  SimpleNonChargingDock::configure(parent, name, tf);
  const std::string lock_param = name + ".lock_remaining_distance";
  const std::string tol_param = name + ".docking_tolerance";
  if (!node_->has_parameter(lock_param)) {
    node_->declare_parameter(lock_param, lock_remaining_distance_);
  }
  if (!node_->has_parameter(tol_param)) {
    node_->declare_parameter(tol_param, docking_tolerance_);
  }
  node_->get_parameter(lock_param, lock_remaining_distance_);
  node_->get_parameter(tol_param, docking_tolerance_);
  RCLCPP_INFO(
    node_->get_logger(), "SlotDock %s: lock at %.3f m, stop within %.3f m (along slot axis)",
    name.c_str(), lock_remaining_distance_, docking_tolerance_);
}

// ---- 정렬 위치: 원본 계산, 도킹 시작·재시도 때 고정 해제 ----
geometry_msgs::msg::PoseStamped SlotDock::getStagingPose(
  const geometry_msgs::msg::Pose & pose, const std::string & frame)
{
  locked_ = false;
  return SimpleNonChargingDock::getStagingPose(pose, frame);
}

// ---- 목표 자세: 감지 구간은 원본, 고정 후에는 고정값 --------
bool SlotDock::getRefinedPose(geometry_msgs::msg::PoseStamped & pose, std::string id)
{
  if (locked_) {
    dock_pose_pub_->publish(dock_pose_);
    pose = dock_pose_;
    return true;
  }
  if (!SimpleNonChargingDock::getRefinedPose(pose, id)) {
    return false;                                 // 감지 끊김 -> 실패, 서버가 재시도
  }
  double remaining = 0.0;
  if (remainingDistance(remaining) && remaining <= lock_remaining_distance_) {
    locked_ = true;
    RCLCPP_INFO(node_->get_logger(), "SlotDock: dock pose locked, remaining %.3f m", remaining);
  }
  return true;
}

// ---- 정지 판정: 칸 축 방향 남은 거리만 봄 --------------------
bool SlotDock::isDocked()
{
  double remaining = 0.0;
  return remainingDistance(remaining) && remaining <= docking_tolerance_;
}

// ---- 남은 거리: 로봇 (base) 을 목표 자세 축에 투영 ------------
bool SlotDock::remainingDistance(double & remaining)
{
  if (dock_pose_.header.frame_id.empty()) {
    return false;                                 // 목표 자세 아직 없음
  }
  geometry_msgs::msg::PoseStamped base_pose;
  base_pose.header.stamp = rclcpp::Time(0);
  base_pose.header.frame_id = base_frame_id_;
  base_pose.pose.orientation.w = 1.0;
  try {
    tf2_buffer_->transform(base_pose, base_pose, dock_pose_.header.frame_id);
  } catch (const tf2::TransformException &) {
    return false;
  }
  const double yaw = tf2::getYaw(dock_pose_.pose.orientation);        // 목표 자세 x = 칸 안쪽
  const double dx = base_pose.pose.position.x - dock_pose_.pose.position.x;
  const double dy = base_pose.pose.position.y - dock_pose_.pose.position.y;
  remaining = -(dx * std::cos(yaw) + dy * std::sin(yaw));             // 목표 선 앞 = 양수
  return true;
}

}  // namespace pinky_wall_parking_dock_plugin

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(pinky_wall_parking_dock_plugin::SlotDock, opennav_docking_core::ChargingDock)