cmake_minimum_required(VERSION 3.0)
project(int_ops)

find_package(OpenMP REQUIRED)

ADD_LIBRARY(ONEDNN STATIC IMPORTED)
SET_TARGET_PROPERTIES(ONEDNN PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/onednn_package/lib/libdnnl.a)

add_executable(test
  src/test/test_matmul.cc
  src/main/matmul.h
  src/main/matmul.cc
)
target_include_directories(test PRIVATE src/main)
target_include_directories(test PRIVATE onednn_package/include)
target_link_libraries(test PRIVATE ONEDNN OpenMP::OpenMP_CXX)

add_library(IntOps SHARED
  src/main/matmul.h
  src/main/matmul.cc
)
target_include_directories(IntOps PRIVATE onednn_package/include)
set_target_properties(IntOps PROPERTIES 
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
  PUBLIC_HEADER src/main/matmul.h
)
target_link_libraries(IntOps PRIVATE ONEDNN OpenMP::OpenMP_CXX)