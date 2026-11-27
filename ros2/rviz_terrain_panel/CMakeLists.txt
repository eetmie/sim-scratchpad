cmake_minimum_required(VERSION 3.16)
project(excavator_rviz)

find_package(ament_cmake REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rviz_common REQUIRED)
find_package(std_srvs REQUIRED)

add_library(terrain_panel SHARED terrain_panel.cpp)
target_compile_features(terrain_panel PRIVATE cxx_std_17)
target_link_libraries(terrain_panel PRIVATE pluginlib::pluginlib rclcpp::rclcpp
  rviz_common::rviz_common std_srvs::std_srvs__rosidl_typesupport_cpp)
pluginlib_export_plugin_description_file(rviz_common plugin.xml)
install(TARGETS terrain_panel LIBRARY DESTINATION lib)
ament_package()
