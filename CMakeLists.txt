cmake_minimum_required(VERSION 3.16)

project(test-nn)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# add_compile_options(-Wall -Wextra -Wuninitialized -fsanitize=address -fno-omit-frame-pointer -pedantic -g)
# add_link_options(-fsanitize=address)
add_compile_options(-Wall -Wextra -Wuninitialized -pedantic -g -O3)

add_executable(test-nn src/main.cpp src/layer.cpp src/model.cpp src/trainer.cpp
    src/math.cpp src/functions.cpp)
target_link_directories(test-nn PUBLIC ~/Programming/usr/lib/)
target_include_directories(test-nn PUBLIC ~/Programming/usr/include/)
target_link_libraries(test-nn openblas)
