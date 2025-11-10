cmake_minimum_required(VERSION 3.20)
project(azuki_puffer_env C)

get_filename_component(AZUKI_PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../.. ABSOLUTE)

if(NOT TARGET azuki_lib)
  message(FATAL_ERROR
    "azuki_puffer_env must be built via the top-level azuki-tcg CMake project.\n"
    "Run `cmake -S ${AZUKI_PROJECT_ROOT} -B ${AZUKI_PROJECT_ROOT}/build` and build the azuki_puffer_env target from there.")
endif()

set(AZUKI_PUFFER_ENV_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/binding.c
)

add_executable(azuki_puffer_env ${AZUKI_PUFFER_ENV_SOURCES})

target_include_directories(azuki_puffer_env PRIVATE
  ${AZUKI_PROJECT_ROOT}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(azuki_puffer_env PRIVATE azuki_lib)
