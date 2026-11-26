cmake_minimum_required(VERSION 3.16)
project(excavator_depth_to_cloud)

set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(depth_to_cloud src/depth_to_cloud.cpp)
target_link_libraries(depth_to_cloud rclcpp::rclcpp ${sensor_msgs_TARGETS} tf2_ros::tf2_ros tf2_eigen::tf2_eigen Eigen3::Eigen)
install(TARGETS depth_to_cloud DESTINATION lib/${PROJECT_NAME})

ament_package()
