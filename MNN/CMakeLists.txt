cmake_minimum_required(VERSION 3.10)
project(Ultra-face-mnn)
set(OpenCV_DIR "/Users/naviocean/build_opencv")

#opencv
find_package(OpenCV REQUIRED)

#3.set environment variable
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenCV REQUIRED)
include_directories(
        mnn/include
        src
)

link_directories(mnn/lib)

add_executable(Ultra-face-mnn src/main.cpp src/UltraFace.cpp src/net.cpp src/Vectorization.cpp)
target_link_libraries(Ultra-face-mnn MNN ${OpenCV_LIBS})