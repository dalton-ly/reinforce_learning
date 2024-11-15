cmake_minimum_required(VERSION 3.5.0)
project(reinforce_learning VERSION 0.1.0 LANGUAGES C CXX)
# set(CMAKE_BUILD_TYPE Release)
find_package(Eigen3 3.3 REQUIRED)
include_directories("include")
add_executable(reinforce_learning main.cpp other.cpp)
target_link_libraries(reinforce_learning Eigen3::Eigen)


