project(Hellocuda LANGUAGES CXX CUDA)
cmake_minimum_required(VERSION 3.21)

find_package(CUDA REQUIRED)
message(STATUS)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80)
endif(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)



include_directories(CMAKE_INCLUDE_CURRENT_DIR
    ${CMAKE_INCLUDE_CURRENT_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

file(GLOB MATRIX_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

add_library(matrix SHARED ${MATRIX_FILES})

file(GLOB CUDA_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)

add_library(kernel SHARED ${CUDA_FILES})
target_link_libraries(kernel matrix)

add_executable(main main.cpp)

target_link_libraries(main kernel matrix)

