cmake_minimum_required(VERSION 3.10)
project(MyProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-Wall -Wextra)

find_package(Eigen3 REQUIRED)

add_executable(main.o src/main.cpp)
add_executable(test.o tests/main.cpp)

target_link_libraries(main.o PRIVATE Eigen3::Eigen)
target_link_libraries(test.o PRIVATE Eigen3::Eigen)

target_include_directories(main.o PRIVATE ${CMAKE_SOURCE_DIR}/lib)
target_include_directories(test.o PRIVATE ${CMAKE_SOURCE_DIR}/lib)
