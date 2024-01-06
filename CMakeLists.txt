cmake_minimum_required(VERSION 3.10)
project(MonoVO)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Add executable
add_executable(MonoVO mono_vo.cpp)

# Link libraries
target_link_libraries(MonoVO ${OpenCV_LIBS})