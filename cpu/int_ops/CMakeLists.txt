cmake_minimum_required(VERSION 3.0)
project(int_ops)

set(CMAKE_PREFIX_PATH onednn_package)

find_library(ONEDNN dnnl)

add_executable(IntOps src/main/matmul.cc)
target_include_directories(IntOps PRIVATE onednn_package/include)
target_link_libraries(IntOps PRIVATE ${ONEDNN})