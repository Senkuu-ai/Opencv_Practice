cmake_minimum_required(VERSION 3.21)
project(OpenCVDemo)

set(CMAKE_CXX_STANDARD 11)

set(OpenCV_DIR "D:\\OpenCV\\opencv\\mingw_build\\install")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(OpenCVDemo Channels.cpp
        Grayscale_2.cpp
        Thresholding_3.cpp
        hist.cpp
        hist.cpp
        Gamma.cpp
)
target_link_libraries(OpenCVDemo ${OpenCV_LIBS})
