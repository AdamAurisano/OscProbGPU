cmake_minimum_required(VERSION 3.18)
project(OscProbGPU VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  set(_opg_cuda_default ON)
else()
  set(_opg_cuda_default OFF)
endif()

option(OPG_ENABLE_CUDA "Build the CUDA backend" ${_opg_cuda_default})
option(OPG_ENABLE_TESTS "Build the test suite" ON)
option(OPG_ENABLE_PYTHON "Build the Python bindings" OFF)

if(OPG_ENABLE_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Default: V100 (sm_70). With old drivers (e.g. 450 + CUDA 11.8) there is
    # no PTX JIT, so real SASS for the target architecture must be shipped.
    set(CMAKE_CUDA_ARCHITECTURES 70)
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  find_package(CUDAToolkit REQUIRED)
endif()

find_package(OpenMP COMPONENTS CXX)

# ---------------------------------------------------------------------------
# Header-only physics/math core
# ---------------------------------------------------------------------------
add_library(opg_core INTERFACE)
add_library(OscProbGPU::core ALIAS opg_core)
target_include_directories(
  opg_core INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                     $<INSTALL_INTERFACE:include>)
target_compile_features(opg_core INTERFACE cxx_std_17)
target_compile_definitions(
  opg_core INTERFACE OPG_DATA_DIR="${PROJECT_SOURCE_DIR}/data")

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(OPG_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
