get_filename_component(AZUKI_PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../.. ABSOLUTE)

if(NOT TARGET azuki_lib)
  message(FATAL_ERROR
    "azuki_puffer_env must be built via the top-level azuki-tcg CMake project.\n"
    "Run `cmake -S ${AZUKI_PROJECT_ROOT} -B ${AZUKI_PROJECT_ROOT}/build` and build the azuki_puffer_env target from there.")
endif()

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import numpy, sys; sys.stdout.write(numpy.get_include())"
  RESULT_VARIABLE PYTHON_NUMPY_RESULT
  OUTPUT_VARIABLE PYTHON_NUMPY_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT PYTHON_NUMPY_RESULT EQUAL 0 OR PYTHON_NUMPY_INCLUDE_DIR STREQUAL "")
  message(FATAL_ERROR
    "Failed to locate NumPy headers for ${Python3_EXECUTABLE}.\n"
    "Install NumPy in that interpreter (e.g. `${Python3_EXECUTABLE} -m pip install numpy`).")
endif()

set(AZUKI_PUFFER_ENV_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/binding.c
)

add_library(azuki_puffer_env MODULE ${AZUKI_PUFFER_ENV_SOURCES})
set_target_properties(azuki_puffer_env PROPERTIES
  OUTPUT_NAME binding
  PREFIX ""
)

target_include_directories(azuki_puffer_env PRIVATE
  ${AZUKI_PROJECT_ROOT}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${Python3_INCLUDE_DIRS}
  ${PYTHON_NUMPY_INCLUDE_DIR}
)

target_link_libraries(azuki_puffer_env PRIVATE azuki_lib Python3::Python)
