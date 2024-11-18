cmake_minimum_required(VERSION 3.5.0)
project(reinforce_learning VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_BUILD_TYPE Release)

# 查找 Eigen3 和 Python3 开发文件
find_package(Eigen3 3.3 REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Development)

# 输出 Python 的相关路径
message(STATUS "Python3 include dirs: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3 libraries: ${Python3_LIBRARIES}")
message(STATUS "Python3 executable: ${Python3_EXECUTABLE}")

# 包含 Eigen3 和其他头文件
include_directories("include")
include_directories("/home/ly/miniconda3/lib/python3.12/site-packages/numpy/_core/include")

# 获取源文件列表
file(GLOB_RECURSE SRC_DIR_LIST "src/*.cpp")

# 添加可执行文件
add_executable(reinforce_learning main.cpp ${SRC_DIR_LIST})

# 链接 Eigen3 和 Python3 库
target_link_libraries(reinforce_learning Eigen3::Eigen Python3::Python)
