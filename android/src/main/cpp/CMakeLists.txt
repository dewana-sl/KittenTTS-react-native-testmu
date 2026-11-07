# Kitten TTS 2's ggml engine (kittentts2/engine.h): the static library
# kittentts2_engine, and on Android libkittentts2.so with the JNI bridge for
# KittenTTS2GGMLModule.java. ggml is built in, with its Vulkan backend for the GPU.
#
# ggml, Vulkan-Headers and SPIRV-Headers come from the pinned downloads below,
# or from local folders: KITTENTTS2_GGML_SOURCE_DIR (a llama.cpp/ggml checkout),
# KITTENTTS2_VULKAN_HEADERS_DIR and KITTENTTS2_SPIRV_HEADERS_DIR (their include folders).

cmake_minimum_required(VERSION 3.19)
project(kittentts2 C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(KITTENTTS2_GGML_SOURCE_DIR "" CACHE PATH "A ggml checkout to build instead of the pinned release")
set(KITTENTTS2_VULKAN_HEADERS_DIR "" CACHE PATH "Vulkan-Headers' include folder")
set(KITTENTTS2_SPIRV_HEADERS_DIR "" CACHE PATH "SPIRV-Headers' include folder")
if(ANDROID)
    option(KITTENTTS2_VULKAN "Build ggml's Vulkan backend" ON)
else()
    option(KITTENTTS2_VULKAN "Build ggml's Vulkan backend" OFF)
endif()

include(FetchContent)

if(KITTENTTS2_VULKAN)
    if(NOT KITTENTTS2_VULKAN_HEADERS_DIR)
        FetchContent_Declare(kittentts2_vulkan_headers
            URL https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/vulkan-sdk-1.4.363.0.tar.gz
            URL_HASH SHA256=4a078be12bef21cfebc09d878b77a63cff9d68f899254a0b00d0e37ef73e7f7e)
        FetchContent_GetProperties(kittentts2_vulkan_headers)
        if(NOT kittentts2_vulkan_headers_POPULATED)
            FetchContent_Populate(kittentts2_vulkan_headers)
        endif()
        set(KITTENTTS2_VULKAN_HEADERS_DIR "${kittentts2_vulkan_headers_SOURCE_DIR}/include")
    endif()
    if(NOT KITTENTTS2_SPIRV_HEADERS_DIR)
        FetchContent_Declare(kittentts2_spirv_headers
            URL https://github.com/KhronosGroup/SPIRV-Headers/archive/refs/tags/vulkan-sdk-1.4.363.0.tar.gz
            URL_HASH SHA256=5b3a0d099f5c7649e29c0938476c462e71d66ad89a0bde42fa1ba3eb476956c9)
        FetchContent_GetProperties(kittentts2_spirv_headers)
        if(NOT kittentts2_spirv_headers_POPULATED)
            FetchContent_Populate(kittentts2_spirv_headers)
        endif()
        set(KITTENTTS2_SPIRV_HEADERS_DIR "${kittentts2_spirv_headers_SOURCE_DIR}/include")
    endif()

    # ggml-vulkan finds vulkan.hpp and spirv.hpp through Vulkan's include folder,
    # and asks for an SPIRV-Headers package it does not link.
    set(include_dir "${CMAKE_BINARY_DIR}/kittentts2-vulkan-include")
    file(COPY "${KITTENTTS2_VULKAN_HEADERS_DIR}/" "${KITTENTTS2_SPIRV_HEADERS_DIR}/" DESTINATION "${include_dir}")
    file(WRITE "${CMAKE_BINARY_DIR}/kittentts2-spirv/SPIRV-HeadersConfig.cmake" "")
    set(SPIRV-Headers_DIR "${CMAKE_BINARY_DIR}/kittentts2-spirv" CACHE PATH "" FORCE)
    set(Vulkan_INCLUDE_DIR "${include_dir}" CACHE PATH "" FORCE)

    if(ANDROID)
        # The NDK has glslc. Link against API 28's libvulkan, which has Vulkan 1.1's
        # entry points; on older Android the library does not load and the SDK uses ONNX.
        if(ANDROID_ABI STREQUAL "arm64-v8a")
            set(triple aarch64-linux-android)
        else()
            set(triple x86_64-linux-android)
        endif()
        set(Vulkan_LIBRARY "${CMAKE_SYSROOT}/usr/lib/${triple}/28/libvulkan.so" CACHE FILEPATH "" FORCE)
        set(Vulkan_GLSLC_EXECUTABLE "${ANDROID_NDK}/shader-tools/${ANDROID_HOST_TAG}/glslc" CACHE FILEPATH "" FORCE)

        # ggml-vulkan builds its shader generator for the build machine as a nested
        # CMake project. Gradle gives only this project its Ninja, and Ninja is not
        # usually on PATH, so the nested project's toolchain file says where it is.
        # Only PATH: CMake's own search paths lead to the NDK's clang, which targets Android.
        find_program(KITTENTTS2_HOST_CC NAMES gcc clang cl
                     NO_CMAKE_PATH NO_CMAKE_ENVIRONMENT_PATH NO_CMAKE_SYSTEM_PATH NO_CMAKE_FIND_ROOT_PATH)
        find_program(KITTENTTS2_HOST_CXX NAMES g++ clang++ cl
                     NO_CMAKE_PATH NO_CMAKE_ENVIRONMENT_PATH NO_CMAKE_SYSTEM_PATH NO_CMAKE_FIND_ROOT_PATH)
        if(NOT KITTENTTS2_HOST_CC OR NOT KITTENTTS2_HOST_CXX)
            message(FATAL_ERROR "The ggml engine needs a C++ compiler for this machine, to build its shader generator.")
        endif()
        file(WRITE "${CMAKE_BINARY_DIR}/kittentts2-host-toolchain.cmake"
"set(CMAKE_C_COMPILER \"${KITTENTTS2_HOST_CC}\")
set(CMAKE_CXX_COMPILER \"${KITTENTTS2_HOST_CXX}\")
set(CMAKE_MAKE_PROGRAM \"${CMAKE_MAKE_PROGRAM}\" CACHE FILEPATH \"\")
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_C_FLAGS -O2)
set(CMAKE_CXX_FLAGS -O2)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE NEVER)
")
        set(GGML_VULKAN_SHADERS_GEN_TOOLCHAIN "${CMAKE_BINARY_DIR}/kittentts2-host-toolchain.cmake" CACHE FILEPATH "" FORCE)
    endif()
endif()

# An app's debug build would otherwise build ggml unoptimised, which is too slow to use.
foreach(lang C CXX)
    string(REPLACE "-O0" "" CMAKE_${lang}_FLAGS_DEBUG "${CMAKE_${lang}_FLAGS_DEBUG}")
    set(CMAKE_${lang}_FLAGS_DEBUG "${CMAKE_${lang}_FLAGS_DEBUG} -O2")
endforeach()

set(GGML_VULKAN ${KITTENTTS2_VULKAN} CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
if(ANDROID AND ANDROID_ABI STREQUAL "arm64-v8a")
    # Phones since 2018 (Cortex-A55/A75 and later); jni.cpp checks the CPU before the SDK uses it.
    set(GGML_CPU_ARM_ARCH "armv8.2-a+dotprod+fp16" CACHE STRING "" FORCE)
endif()

if(KITTENTTS2_GGML_SOURCE_DIR)
    add_subdirectory("${KITTENTTS2_GGML_SOURCE_DIR}" ggml EXCLUDE_FROM_ALL)
else()
    # The ggml in KittenML's llama.cpp fork, which the engine is tested against.
    FetchContent_Declare(kittentts2_ggml
        URL https://github.com/KittenML/kitten-tts-2-cpp/archive/2b9912e72245f5597c664d2be44a6e7ba0a3f293.tar.gz
        URL_HASH SHA256=b889358e5ef674ea799b43076af77bdd1ac597247b5c0ab76925df012159164c)
    FetchContent_GetProperties(kittentts2_ggml)
    if(NOT kittentts2_ggml_POPULATED)
        FetchContent_Populate(kittentts2_ggml)
        add_subdirectory("${kittentts2_ggml_SOURCE_DIR}/ggml" "${kittentts2_ggml_BINARY_DIR}" EXCLUDE_FROM_ALL)
    endif()
endif()

add_library(kittentts2_engine STATIC
    kittentts2/audio_decoder.cpp
    kittentts2/engine.cpp
    kittentts2/json.cpp
    kittentts2/language_model.cpp
    kittentts2/runtime.cpp
    kittentts2/safetensors.cpp
    kittentts2/sampler.cpp)
target_include_directories(kittentts2_engine PUBLIC kittentts2)
target_compile_features(kittentts2_engine PUBLIC cxx_std_17)
target_link_libraries(kittentts2_engine PUBLIC ggml)
if(KITTENTTS2_VULKAN)
    # runtime.cpp asks Vulkan which version the device has when ggml-vulkan finds no GPU.
    target_compile_definitions(kittentts2_engine PRIVATE KITTENTTS2_VULKAN)
    target_include_directories(kittentts2_engine PRIVATE "${Vulkan_INCLUDE_DIR}")
    target_link_libraries(kittentts2_engine PRIVATE "${Vulkan_LIBRARY}")
endif()

if(ANDROID)
    add_library(kittentts2 SHARED kittentts2/jni.cpp)
    target_link_libraries(kittentts2 PRIVATE kittentts2_engine log)
    # Only the JNI entry points are exported.
    set_target_properties(kittentts2 PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
    target_link_options(kittentts2 PRIVATE "-Wl,--exclude-libs,ALL" "-Wl,-z,max-page-size=16384")
endif()
