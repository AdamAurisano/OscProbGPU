add_library(opg_doctest INTERFACE)
target_include_directories(opg_doctest
                           INTERFACE ${PROJECT_SOURCE_DIR}/third_party/doctest)

# CPU tests: always built, no CUDA toolkit needed.
set(OPG_OSCPROB_DIR "${PROJECT_SOURCE_DIR}/../OscProb"
    CACHE PATH "Original OscProb sources (optional, for bitwise tests)")

add_executable(opg_tests_cpu test_main.cpp test_core.cpp test_numerics.cpp
                             test_models_cpu.cpp test_averaging.cpp)
target_link_libraries(opg_tests_cpu PRIVATE oscprobgpu opg_doctest)

# Compile the original Kopp eigensolvers (LGPL) into the test, if available,
# to check that our port is bit-identical.
set(_kopp ${OPG_OSCPROB_DIR}/MatrixDecomp)
if(EXISTS ${_kopp}/zheevh3.cxx)
  message(STATUS "Using original MatrixDecomp from ${_kopp} in tests")
  target_sources(opg_tests_cpu PRIVATE ${_kopp}/zheevh3.cxx ${_kopp}/zheevc3.cxx
                                       ${_kopp}/zheevq3.cxx ${_kopp}/zhetrd3.cxx)
  target_include_directories(opg_tests_cpu PRIVATE ${_kopp})
  target_compile_definitions(opg_tests_cpu PRIVATE OPG_HAVE_KOPP_ORIG)
endif()
# Eigen (optional) to cross-check the matrix exponential against OscProb's
set(OPG_EIGEN_INCLUDE "${PROJECT_SOURCE_DIR}/build-ref/oscprob-install/include/eigen3"
    CACHE PATH "Eigen3 include directory (optional, for tests)")
if(EXISTS ${OPG_EIGEN_INCLUDE}/unsupported/Eigen/MatrixFunctions)
  message(STATUS "Using Eigen from ${OPG_EIGEN_INCLUDE} in tests")
  target_include_directories(opg_tests_cpu SYSTEM PRIVATE ${OPG_EIGEN_INCLUDE})
  target_compile_definitions(opg_tests_cpu PRIVATE OPG_HAVE_EIGEN)
endif()
target_compile_definitions(
  opg_tests_cpu PRIVATE OPG_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
add_test(NAME cpu COMMAND opg_tests_cpu)
set_tests_properties(cpu PROPERTIES LABELS "cpu")

# GPU tests
if(OPG_ENABLE_CUDA)
  add_executable(opg_tests_gpu test_main.cpp gpu/test_cuda_smoke.cu
                               gpu/test_models_gpu.cpp gpu/test_averaging_gpu.cpp)
  target_link_libraries(opg_tests_gpu PRIVATE oscprobgpu opg_doctest
                                              CUDA::cudart)
  target_compile_definitions(
    opg_tests_gpu PRIVATE OPG_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
  add_test(NAME gpu COMMAND opg_tests_gpu)
  set_tests_properties(gpu PROPERTIES LABELS "gpu")
endif()
