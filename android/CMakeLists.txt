# Used when the app turns on Kitten TTS 2's ggml engine (kittentts.ggml=true in
# android/gradle.properties). The Android Gradle plugin builds one CMake project
# per module, so this one holds onnxruntime-react-native's JSI library, when the
# SDK builds it (as it does without the engine), and the engine (src/main/cpp).

cmake_minimum_required(VERSION 3.19)
project(KittenTTSReactNative C CXX)

if(KITTENTTS_ORT_ANDROID_DIR)
    # Its CMakeLists reads the extracted AARs from ${CMAKE_SOURCE_DIR}/build, where build.gradle puts them.
    add_subdirectory("${KITTENTTS_ORT_ANDROID_DIR}" onnxruntime)
endif()

# ggml's CPU and Vulkan builds cover 64-bit phones and emulators; 32-bit apps use ONNX Runtime.
if(ANDROID_ABI STREQUAL "arm64-v8a" OR ANDROID_ABI STREQUAL "x86_64")
    add_subdirectory(src/main/cpp kittentts2)
endif()
