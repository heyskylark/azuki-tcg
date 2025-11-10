get_filename_component(AZUKI_PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../.. ABSOLUTE)

if(NOT TARGET azuki_lib)
  message(FATAL_ERROR
    "azuki_puffer_env must be built via the top-level azuki-tcg CMake project.\n"
    "Run `cmake -S ${AZUKI_PROJECT_ROOT} -B ${AZUKI_PROJECT_ROOT}/build` and build the azuki_puffer_env target from there.")
endif()

find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

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
)

target_link_libraries(azuki_puffer_env PRIVATE
  azuki_lib
  Python3::Python
  Python3::NumPy
)
