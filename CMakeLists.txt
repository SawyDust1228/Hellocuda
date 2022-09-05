project(Hellocuda LANGUAGES CXX CUDA)
cmake_minimum_required(VERSION 3.21)


include_directories(${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include /usr/local/cuda/include )

find_package(CUDA REQUIRED)

find_package(Python REQUIRED)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80)
    message(STATUS "CMAKE_CUDA_ARCHITECTURES : ${CMAKE_CUDA_ARCHITECTURES}")
endif(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)

if(${Python_FOUND})   
  message(STATUS "The Python Includes : ${Python_LIBRARIES}")
  execute_process(COMMAND python -c
  "import torch; print(torch.__path__[0]);"
  OUTPUT_VARIABLE TORCH_OUTPUT OUTPUT_STRIP_TRAILING_WHITESPACE)
  message(STATUS "${TORCH_OUTPUT}")
  find_package(Torch REQUIRED HINTS ${TORCH_OUTPUT} NO_DEFAULT_PATH)
endif()


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

target_link_libraries(main kernel matrix ${TORCH_LIBRARIES} ${Python_LIBRARIES})

