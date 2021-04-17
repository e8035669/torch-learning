cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(resnet)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

add_compile_options(-g3)

find_package(Torch REQUIRED)

add_executable(resnet main.cpp)
target_link_libraries(resnet "${TORCH_LIBRARIES}")
set_property(TARGET resnet PROPERTY CXX_STANDARD 14)
