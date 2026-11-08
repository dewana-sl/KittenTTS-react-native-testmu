# kt2check: the Android ggml engine against the Python package

`check.cpp` runs the engine in `android/src/main/cpp/kittentts2` outside an app:
it compares both models with reference outputs of the Python package and times
them. It builds for the host (CPU) or for Android, to run on a phone or emulator
over `adb`. Not shipped in the npm package.

## Build

For the host, CPU only:

```bash
cmake -S tools/kitten-tts-2-ggml -B build/kt2check -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF
ninja -C build/kt2check kt2check
```

For Android, with Vulkan (the SDK's CMake and Ninja are under `$ANDROID_HOME/cmake`):

```bash
cmake -S tools/kitten-tts-2-ggml -B build/kt2check-android -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28
ninja -C build/kt2check-android kt2check
```

CMake downloads ggml and the Vulkan and SPIR-V headers, pinned in
`android/src/main/cpp/CMakeLists.txt`. `-DKITTENTTS2_GGML_SOURCE_DIR`,
`-DKITTENTTS2_VULKAN_HEADERS_DIR` and `-DKITTENTTS2_SPIRV_HEADERS_DIR` use local
folders instead.

## Run

```bash
kt2check device
kt2check lm reference.json lm-ternary.safetensors
kt2check decoder decoder-reference.safetensors decoder-mlx.safetensors [out.wav]
kt2check probe decoder-reference.safetensors intermediates.safetensors decoder-mlx.safetensors
kt2check lm-compare reference.json lm-ternary.safetensors
```

- `lm` compares the logits of a prefill and three steps with the Python package's
  MLX backend, then times 300 tokens. `reference.json` holds the prompt, speaker,
  expected logits and config.
- `decoder` compares the mel with torch's for the same flow noise, the waveform
  for the same phases and source noise, then times `decode()`.
  `decoder-reference.safetensors` holds those tensors.
- `probe` compares the flow's stages with torch's, to find where a backend differs.
- `lm-compare` runs the prefill on the GPU and the CPU at several prompt lengths
  (`KT2_LENGTHS=64,200`), per layer (`KT2_LM_LAYERS=64`), and prints the largest
  input to each down projection on the CPU (`KT2_LM_SWIGLU=297`).

`KITTENTTS2_DEVICE=cpu` keeps everything on the CPU, `KITTENTTS2_DEVICE=gpu` uses an
Adreno or Mali GPU, which the engine otherwise leaves alone (`knownGpuProblem()` in
`runtime.cpp`), and `KITTENTTS2_DEBUG=1` lists the nodes that fall back to the CPU,
with the graph's splits and buffer sizes.

## What a GPU needs from the graphs

Three things in ggml's Vulkan backend shaped how the graphs are written. Keep them
in mind when changing them, and rerun these checks on the GPU:

- **Matrix products sum in float16 by default** on GPUs that have it. `matmul()`
  in `runtime.h` asks for float32; without it the multi-row prefill goes wrong
  from about 64 tokens, while single-token steps (a different kernel) look fine.
- **Matrix products hold their inputs in float16.** The language model's last
  layers feed the down projection values up to about 240,000, past float16's
  65504, so it is scaled by 1/32 around the product. Everything else stays well
  inside float16.
- **Some shaders write past their output and rely on robust buffer access** to
  drop it: `ggml_norm`, `ggml_soft_max*` and `ggml_sum_rows` run a grid rounded
  up to 512 rows without checking the row. Vulkan requires drivers to support
  this, but the Android emulator's GPU (gfxstream over MoltenVK) does not drop
  the writes, which corrupts whatever comes next in memory. The decoder avoids
  those ops: `layerNorm()` is a matmul for the mean plus `ggml_rms_norm`, and
  attention is `ggml_flash_attn_ext`.
