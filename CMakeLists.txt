cmake_minimum_required(VERSION 3.20)
project(azuki-tcg C)

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

add_executable(azuki
  src/main.c
  src/components.c
)

# Link against the imported target if available, otherwise the alias we set up
if(TARGET flecs::flecs)
  target_link_libraries(azuki PRIVATE flecs::flecs)
else()
  # Fallback if package config exported a different canonical target name
  # Adjust if your environment provides a different target.
  target_link_libraries(azuki PRIVATE flecs)
endif()

target_include_directories(azuki PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
