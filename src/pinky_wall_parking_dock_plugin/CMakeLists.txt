cmake_minimum_required(VERSION 3.8)
project(pinky_wall_parking_dock_plugin)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(pluginlib REQUIRED)
find_package(opennav_docking REQUIRED)
find_package(opennav_docking_core REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# Nav2 Jazzy 는 simple_non_charging_dock 라이브러리를 설치하지만 CMake 로 내보내지 않음
# -> opennav_docking 설치 경로 (share/opennav_docking/cmake -> lib) 에서 직접 찾음
find_library(SIMPLE_NON_CHARGING_DOCK_LIB simple_non_charging_dock
  HINTS ${opennav_docking_DIR}/../../../lib REQUIRED)

add_library(slot_dock SHARED src/slot_dock.cpp)
target_include_directories(slot_dock PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(slot_dock
  rclcpp rclcpp_lifecycle pluginlib opennav_docking opennav_docking_core
  geometry_msgs tf2 tf2_ros tf2_geometry_msgs)
target_link_libraries(slot_dock ${SIMPLE_NON_CHARGING_DOCK_LIB})

pluginlib_export_plugin_description_file(opennav_docking_core slot_dock_plugin.xml)

install(TARGETS slot_dock
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
