cmake_minimum_required(VERSION 3.5.0)
project(reinforce_learning VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_BUILD_TYPE Release)

find_package(Eigen3 3.3 REQUIRED)

include_directories("include")


# 获取源文件列表
file(GLOB_RECURSE SRC_DIR_LIST "src/*.cpp")

# 添加可执行文件
add_executable(reinforce_learning main.cpp ${SRC_DIR_LIST})

# 链接 Eigen3 和 Python3 库
target_link_libraries(reinforce_learning Eigen3::Eigen)
