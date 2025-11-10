cmake_minimum_required(VERSION 3.20)
project(azuki-tcg C)

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

set(AZUKI_MAIN ${CMAKE_CURRENT_SOURCE_DIR}/src/main.c)
set(AZUKI_LIB_SOURCES ${AZUKI_SOURCES})
list(REMOVE_ITEM AZUKI_LIB_SOURCES ${AZUKI_MAIN})

add_library(azuki_lib ${AZUKI_LIB_SOURCES})

target_include_directories(azuki_lib PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CURSES_INCLUDE_DIRS}
)

# Link against the imported target if available, otherwise the alias we set up
if(TARGET flecs::flecs)
  target_link_libraries(azuki_lib PUBLIC flecs::flecs)
else()
  # Fallback if package config exported a different canonical target name
  # Adjust if your environment provides a different target.
  target_link_libraries(azuki_lib PUBLIC flecs)
endif()

target_link_libraries(azuki_lib PUBLIC ${CURSES_LIBRARIES})

add_executable(azuki ${AZUKI_MAIN})
target_link_libraries(azuki PRIVATE azuki_lib)

enable_testing()

add_executable(world_tests
  tests/test_world.c
)
target_link_libraries(world_tests PRIVATE azuki_lib)
add_test(NAME world_tests COMMAND world_tests)

add_subdirectory(python/src)
