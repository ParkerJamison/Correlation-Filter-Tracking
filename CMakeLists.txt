cmake_minimum_required(VERSION 3.16)
project(CorrelationFilter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_library(cft_core
    CFT_Track.cpp
    TrackID.cpp
)

target_include_directories(cft_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(cft_core
    PUBLIC
        ${OpenCV_LIBS}
)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE cft_core)