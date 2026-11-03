# kt2check: the Android ggml engine against the Python package's outputs. It
# builds for the host (CPU, or Metal on a Mac) or for Android with the NDK's
# toolchain file, to run over adb. See README.md.

cmake_minimum_required(VERSION 3.19)
project(kt2check C CXX)

add_subdirectory(../../android/src/main/cpp engine)

add_executable(kt2check check.cpp)
target_link_libraries(kt2check PRIVATE kittentts2_engine)
if(ANDROID)
    # NDK 27 and older would not run on devices and emulator images with 16 KB pages.
    target_link_options(kt2check PRIVATE "-Wl,-z,max-page-size=16384")
endif()
