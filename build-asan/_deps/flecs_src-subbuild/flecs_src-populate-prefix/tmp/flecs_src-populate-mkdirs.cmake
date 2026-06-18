# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/skylark/git/azuki-tcg/build-asan/_deps/flecs_src-src")
  file(MAKE_DIRECTORY "/home/skylark/git/azuki-tcg/build-asan/_deps/flecs_src-src")
endif()
file(MAKE_DIRECTORY
  "/home/skylark/git/azuki-tcg/build-asan/_deps/flecs_src-build"
  "/home/skylark/git/azuki-tcg/build-asan/_deps/flecs_src-subbuild/flecs_src-populate-prefix"
  "/home/skylark/git/azuki-tcg/build-asan/_deps/flecs_src-subbuild/flecs_src-populate-prefix/tmp"
  "/home/skylark/git/azuki-tcg/build-asan/_deps/flecs_src-subbuild/flecs_src-populate-prefix/src/flecs_src-populate-stamp"
  "/home/skylark/git/azuki-tcg/build-asan/_deps/flecs_src-subbuild/flecs_src-populate-prefix/src"
  "/home/skylark/git/azuki-tcg/build-asan/_deps/flecs_src-subbuild/flecs_src-populate-prefix/src/flecs_src-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/skylark/git/azuki-tcg/build-asan/_deps/flecs_src-subbuild/flecs_src-populate-prefix/src/flecs_src-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/skylark/git/azuki-tcg/build-asan/_deps/flecs_src-subbuild/flecs_src-populate-prefix/src/flecs_src-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
