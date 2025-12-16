cmake_minimum_required(VERSION 3.10)
project(MyProject)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-Wall -Wextra)

find_package(Eigen3 REQUIRED)

add_executable(test tests/main.cpp)
target_link_libraries(test PRIVATE Eigen3::Eigen)
target_include_directories(test PRIVATE ${CMAKE_SOURCE_DIR}/lib)
