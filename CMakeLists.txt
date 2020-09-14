cmake_minimum_required(VERSION 3.18)
project(matrix)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


include_directories(include)
add_library(matrix src/matrix.cpp include/matrix.hpp include/nn.hpp)
set_property(TARGET matrix PROPERTY CXX_STANDARD 20)

add_executable(app bin/app.cpp)
set_property(TARGET app PROPERTY CXX_STANDARD 20)
target_link_libraries(app gsl.a gslcblas.a)
