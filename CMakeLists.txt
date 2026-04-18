cmake_minimum_required(VERSION 3.20)
project(azuki-tcg C)

option(BUILD_PYTHON_BINDINGS "Build Python bindings for PufferLib (requires Python3 + NumPy)" ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CURSES_NEED_NCURSES TRUE)
find_package(Curses REQUIRED)

find_package(flecs CONFIG QUIET)

if(NOT flecs_FOUND)
  include(FetchContent)
  FetchContent_Declare(
  flecs_src
  GIT_REPOSITORY https://github.com/SanderMertens/flecs.git
    GIT_TAG v4.1.1
  )
  FetchContent_MakeAvailable(flecs_src)

  if(TARGET flecs AND NOT TARGET flecs::flecs)
    add_library(flecs::flecs ALIAS flecs)
  endif()
endif()

file(GLOB_RECURSE AZUKI_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
)
# Keep copied v2 snapshots out of the default azuki_lib build until we wire a
# dedicated v2 C target. This avoids duplicate symbol collisions.
list(FILTER AZUKI_SOURCES EXCLUDE REGEX ".*/src/.*/v2/.*\\.c$")

set(AZUKI_MAIN ${CMAKE_CURRENT_SOURCE_DIR}/src/main.c)
set(AZUKI_LIB_SOURCES ${AZUKI_SOURCES})
list(REMOVE_ITEM AZUKI_LIB_SOURCES ${AZUKI_MAIN})

if(TARGET flecs::flecs)
  set(AZUKI_FLECS_TARGET flecs::flecs)
else()
  set(AZUKI_FLECS_TARGET flecs)
endif()

add_library(azuki_obj OBJECT ${AZUKI_LIB_SOURCES})

target_include_directories(azuki_obj PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CURSES_INCLUDE_DIRS}
)

target_link_libraries(azuki_obj PUBLIC ${AZUKI_FLECS_TARGET})
target_link_libraries(azuki_obj PUBLIC ${CURSES_LIBRARIES})

add_library(azuki_lib STATIC $<TARGET_OBJECTS:azuki_obj>)
target_include_directories(azuki_lib PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CURSES_INCLUDE_DIRS}
)
target_link_libraries(azuki_lib PUBLIC ${AZUKI_FLECS_TARGET})
target_link_libraries(azuki_lib PUBLIC ${CURSES_LIBRARIES})

add_executable(azuki ${AZUKI_MAIN} $<TARGET_OBJECTS:azuki_obj>)
target_include_directories(azuki PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CURSES_INCLUDE_DIRS}
)
target_link_libraries(azuki PRIVATE ${AZUKI_FLECS_TARGET})
target_link_libraries(azuki PRIVATE ${CURSES_LIBRARIES})

enable_testing()

add_executable(world_tests
  tests/test_world.c
  $<TARGET_OBJECTS:azuki_obj>
)
target_include_directories(world_tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CURSES_INCLUDE_DIRS}
)
target_link_libraries(world_tests PRIVATE ${AZUKI_FLECS_TARGET})
target_link_libraries(world_tests PRIVATE ${CURSES_LIBRARIES})
add_test(NAME world_tests COMMAND world_tests)
add_test(NAME stt04_017_regression
  COMMAND world_tests --run-stt04-017-regression)

if(BUILD_PYTHON_BINDINGS)
  add_subdirectory(python/src)
endif()
