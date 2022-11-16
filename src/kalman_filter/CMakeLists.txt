cmake_minimum_required(VERSION 3.0.2)
project(kalman_filter)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
)
find_package(Eigen3 REQUIRED)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES kalman_filter
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

add_library(kalman_filter src/KalmanFilter.cpp)

add_executable(zjstest src/test.cpp)
target_link_libraries(zjstest ${catkin_LIBRARIES} kalman_filter)