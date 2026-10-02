find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/nanobind
                 ${CMAKE_CURRENT_BINARY_DIR}/nanobind)

nanobind_add_module(_oscprobgpu NB_STATIC src/module.cpp)
target_link_libraries(_oscprobgpu PRIVATE oscprobgpu)

# Lay out an importable package in the build tree: build/python/oscprobgpu
set(_pkg ${CMAKE_CURRENT_BINARY_DIR}/oscprobgpu)
set_target_properties(_oscprobgpu PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${_pkg})
configure_file(oscprobgpu/__init__.py ${_pkg}/__init__.py COPYONLY)

if(SKBUILD)
  install(TARGETS _oscprobgpu LIBRARY DESTINATION oscprobgpu)
  install(FILES oscprobgpu/__init__.py DESTINATION oscprobgpu)
endif()

if(OPG_ENABLE_TESTS)
  add_test(NAME python
           COMMAND ${Python_EXECUTABLE} -m pytest -q
                   ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  set_tests_properties(
    python PROPERTIES ENVIRONMENT
    "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}:$ENV{PYTHONPATH};OPG_TEST_DATA_DIR=${PROJECT_SOURCE_DIR}/tests/data"
    LABELS "python")
endif()
