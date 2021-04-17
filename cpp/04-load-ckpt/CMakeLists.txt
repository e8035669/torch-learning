cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(load-ckpt)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

add_compile_options(-g3)

find_package(Torch REQUIRED)

add_executable(load-ckpt main.cpp)
target_link_libraries(load-ckpt "${TORCH_LIBRARIES}")
set_property(TARGET load-ckpt PROPERTY CXX_STANDARD 14)
