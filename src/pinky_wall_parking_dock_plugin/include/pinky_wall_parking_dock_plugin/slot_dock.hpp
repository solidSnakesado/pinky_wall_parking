// src/pinky_wall_parking_dock_plugin/include/pinky_wall_parking_dock_plugin/slot_dock.hpp
// - 벽 주차 칸 dock 플러그인 (Nav2 opennav_docking::SimpleNonChargingDock 상속)
// - 감지 구간: 남은 거리 > lock_remaining_distance 이면 라이다 감지로 목표 자세 갱신 (원본 처리)
// - 고정 구간: 남은 거리 <= lock_remaining_distance 가 되면 목표 자세 고정 (근거리 라이다 오차 제외)
// - 정지 판정: 칸 축 방향 남은 거리 <= docking_tolerance (좌우는 보지 않음, 마지막 구간은 odom)
// - 남은 거리 = 목표 선까지 칸 안쪽 방향 거리 (양수 = 아직 목표 선 앞)

#ifndef PINKY_WALL_PARKING_DOCK_PLUGIN__SLOT_DOCK_HPP_
#define PINKY_WALL_PARKING_DOCK_PLUGIN__SLOT_DOCK_HPP_

#include <memory>
#include <string>

#include "opennav_docking/simple_non_charging_dock.hpp"

namespace pinky_wall_parking_dock_plugin
{

class SlotDock : public opennav_docking::SimpleNonChargingDock
{
public:
  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
    const std::string & name, std::shared_ptr<tf2_ros::Buffer> tf) override;

  geometry_msgs::msg::PoseStamped getStagingPose(
    const geometry_msgs::msg::Pose & pose, const std::string & frame) override;

  bool getRefinedPose(geometry_msgs::msg::PoseStamped & pose, std::string id) override;

  bool isDocked() override;

protected:
  bool remainingDistance(double & remaining);   // 목표 선까지 남은 거리 (m), tf 실패 시 false

  double lock_remaining_distance_{0.035};       // 이 거리 안이면 목표 자세 고정 (m)
  double docking_tolerance_{0.002};             // 이 거리 안이면 정지 (m)
  bool locked_{false};                          // 목표 자세 고정 여부
};

}  // namespace pinky_wall_parking_dock_plugin

#endif  // PINKY_WALL_PARKING_DOCK_PLUGIN__SLOT_DOCK_HPP_