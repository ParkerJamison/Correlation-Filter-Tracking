cmake_minimum_required(VERSION 3.16)
project(CorrelationFilter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(main
    main.cpp
    CFT_Track.cpp
    TrackID.cpp)

target_link_libraries(main PRIVATE ${OpenCV_LIBS})
target_link_directories(main PRIVATE ${OpenCV_INCLUDE_DIRS})