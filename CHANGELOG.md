# Changelog

## Unreleased

- Added Kitten TTS 2 on the device:
  `KittenTTS.create({ model: 'KittenML/kitten-tts-2' })` downloads about 2 GB of
  ONNX files on first use and returns a `KittenTTS2OnDevice` with the Python
  package's calls: the 115 built-in voices, expression tags, presets, sampling
  controls, `advanced` options, `normalizeText()` and chunked streaming. The text
  normalizer, tokenizer, prompt and sampler are ports of the Python package's and
  produce the same token ids. Runs on iOS and Android through ONNX Runtime on the
  CPU, with `precision: 'int8'` (the default, about 2.5 times faster) or
  `'float32'`; an iPhone 17 takes about 1.8 seconds per second of speech.
  `tools/kitten-tts-2-onnx` builds the files from the model repository. Voice
  cloning on the device and browsers are not supported yet.
- Added an MLX backend for Kitten TTS 2 on iOS 17+: the language model and
  s3gen's audio decoder on the GPU, as the Python package's `backend="mlx"`,
  through mlx-swift in the SDK's new iOS pod. It is opt-in (`{ "mlx": true }` in
  the Expo plugin, or `KITTENTTS_MLX=1 pod install`); `backend: 'auto'` uses it
  when it is built in. An iPhone 17 then generates faster than it speaks (a
  real-time factor of 0.55 to 0.79, against 1.8 on the CPU) and downloads
  1.25 GB instead of 2.02 GB.
- Added `decoder: 'student_w4'` for Kitten TTS 2 on MLX: the Python package's
  distilled one-step decoder, 175 MB less to download and about 0.3 GiB less peak
  memory on an iPhone 17, with lower voice similarity (0.684 against 0.752). The
  overall speed hardly changes, because the language model takes about 80% of the
  time.
- Fixed the Expo plugin failing iOS builds for apps with no `assets/kittentts`
  folder: it added a reference to the missing folder to the Xcode project.
- Added a ggml backend for Kitten TTS 2 on Android 9+: the language model and
  s3gen's audio decoder on the GPU through Vulkan, in C++ on the same weights
  files as MLX, so it also downloads 1.25 GB. It is opt-in (`{ "ggml": true }` in
  the Expo plugin, or `kittentts.ggml=true` in `android/gradle.properties`);
  `backend: 'auto'` uses it when it finds a GPU it can use, and otherwise ONNX
  Runtime. `tts.device` says `'gpu'` or `'cpu'`. In the Android emulator on an M5
  Pro Mac it generates faster than it speaks (a real-time factor of about 0.8,
  against about 4 on the CPU); it has not been measured on a phone yet.
- Added Kitten TTS 2 through the Kitten API:
  `KittenTTS.create({ model: 'kitten-tts-2', apiKey })` returns a `KittenTTS2`
  with the same `generate()`, `generateStreaming()`, `speak()` and `play()`,
  plus voice cloning from a `reference` recording, presets, decoding controls and
  `availableVoices`, like the Python package. Failed requests throw
  `KittenAPIError`. Existing on-device code is unchanged.
- Fixed native bundling: the `browser` map added with web support also applied
  on iOS and Android, so Metro bundled the web files and failed with
  "Unable to resolve module node:fs/promises".
- Added Swift-parity word timing metadata via `KittenTTSResult.wordTimings`.
- Added `KittenTTS.generateStreaming()` for sentence-by-sentence generation.
- Added `tts.play(result)` so apps can inspect timings before playback.
- Added React Native Web support through browser-specific ONNX Runtime Web,
  Cache API asset storage, CE phonemizer, and audio playback implementations.

## 0.8.0

- Initial developer-preview React Native SDK for KittenTTS.
- Supports iOS and Android with ONNX Runtime, Expo development builds, and bare React Native.
- Includes runtime model downloads, local cache reuse, built-in voices, WAV output, and example apps.
