# Optional CMake build (Linux and Windows). The plain Makefile and
# scripts/build_windows.bat do the same thing with fewer moving parts.
cmake_minimum_required(VERSION 3.18)
project(hcminer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)   # whatever card is in this machine
  else()
    set(CMAKE_CUDA_ARCHITECTURES 89 120)   # Ada and Blackwell
  endif()
endif()

add_executable(hcminer-gpu src/cuda/miner.cu)
target_compile_options(hcminer-gpu PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -Xptxas=-O3>)
set_target_properties(hcminer-gpu PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/src/cuda")
