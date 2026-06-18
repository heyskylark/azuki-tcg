# CMake generated Testfile for 
# Source directory: /home/skylark/git/azuki-tcg
# Build directory: /home/skylark/git/azuki-tcg/build-asan
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[world_tests]=] "/home/skylark/git/azuki-tcg/build-asan/world_tests")
set_tests_properties([=[world_tests]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/skylark/git/azuki-tcg/CMakeLists.txt;86;add_test;/home/skylark/git/azuki-tcg/CMakeLists.txt;0;")
add_test([=[stt04_017_regression]=] "/home/skylark/git/azuki-tcg/build-asan/world_tests" "--run-stt04-017-regression")
set_tests_properties([=[stt04_017_regression]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/skylark/git/azuki-tcg/CMakeLists.txt;87;add_test;/home/skylark/git/azuki-tcg/CMakeLists.txt;0;")
subdirs("_deps/flecs_src-build")
subdirs("python/src")
