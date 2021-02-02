cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(dcgan)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

add_compile_options(-g3)

find_package(Torch REQUIRED)

add_executable(dcgan dcgan.cpp)
target_link_libraries(dcgan "${TORCH_LIBRARIES}")
set_property(TARGET dcgan PROPERTY CXX_STANDARD 14)
