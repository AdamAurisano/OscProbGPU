add_library(opg_doctest INTERFACE)
target_include_directories(opg_doctest
                           INTERFACE ${PROJECT_SOURCE_DIR}/third_party/doctest)

# CPU tests: always built, no CUDA toolkit needed.
add_executable(opg_tests_cpu test_main.cpp test_core.cpp)
target_link_libraries(opg_tests_cpu PRIVATE opg_core opg_doctest)
target_compile_definitions(
  opg_tests_cpu PRIVATE OPG_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
add_test(NAME cpu COMMAND opg_tests_cpu)
set_tests_properties(cpu PROPERTIES LABELS "cpu")

# GPU tests
if(OPG_ENABLE_CUDA)
  add_executable(opg_tests_gpu test_main.cpp gpu/test_cuda_smoke.cu)
  target_link_libraries(opg_tests_gpu PRIVATE opg_core opg_doctest
                                              CUDA::cudart)
  target_compile_definitions(
    opg_tests_gpu PRIVATE OPG_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
  add_test(NAME gpu COMMAND opg_tests_gpu)
  set_tests_properties(gpu PROPERTIES LABELS "gpu")
endif()
