# AGENTS.md

Guidance for coding agents working on `@kittentts/react-native`.

## Project Overview

This package is a React Native SDK for on-device KittenTTS speech synthesis.
The public API downloads/caches model assets, prepares a JavaScript phonemizer,
runs ONNX Runtime on-device, and optionally plays generated WAV audio.

Main flow:

1. `KittenTTS.create()` resolves config, prepares the phonemizer, downloads any
   missing model files, loads `voices.npz`, and creates the ONNX engine.
2. `tts.generate()` preprocesses text, phonemizes it, tokenizes IPA symbols,
   runs ONNX inference, and returns `KittenTTSResult`.
3. `tts.speak()` calls `generate()` and sends the result to an `AudioPlayer`.

## Important Paths

- `src/index.ts`: public exports.
- `src/KittenTTS.ts`: main SDK class and lifecycle.
- `src/KittenTTSConfig.ts`: user config and defaults.
- `src/*.web.ts`: React Native Web entrypoints and platform-specific browser implementations.
  Each one needs an entry in `package.json` under `browser` (native file to web
  file) **and** under `react-native` (native file to itself). Metro reads
  `browser` on iOS and Android too, so without the second entry the web file is
  bundled into native apps and bundling fails. A test enforces the pairing.
- `src/KittenTTSError.ts`: SDK error codes and helpers, including `KittenAPIError`
  for the Kitten API.
- `src/KittenTTS2.ts` / `src/KittenTTS2.web.ts`: Kitten TTS 2 through the hosted
  Kitten API. `KittenTTS.create()` returns one for `model: 'kitten-tts-2'`. They
  only wire in the platform's `AudioOutput` and file reading; the logic is in
  `src/api/KittenTTS2Base.ts` (requests, cloning uploads, voice listing) and
  `src/api/KittenAPI.ts` (fetch, retries, errors), which run in Node so
  `tests/kitten-tts-2.test.js` can test them with a fake `fetch`. Voices, models
  and API error codes are strings because the API adds them over time: do not
  validate them client-side.
- `src/KittenTTS2OnDevice.ts` / `.web.ts`: Kitten TTS 2 on the device.
  `KittenTTS.create()` returns one for a repository such as
  `'KittenML/kitten-tts-2'`. The native file downloads the files
  (`src/kittentts2/BundleDownloader.ts`, sizes and SHA-256 from `manifest.json`)
  and opens onnxruntime-react-native; the web file throws. Everything else is in
  `src/kittentts2/` and runs in Node: `KittenTTS2OnDeviceBase.ts` (the Python
  package's `generate`/`_synthesize`), `LanguageModel.ts` (one graph run per
  token; the KV cache stays in JS and goes in as offset-0 views, because the RN
  binding reads a tensor from the start of its buffer), `AudioDecoder.ts`
  (s3gen flow + HiFT), `sampling.ts` (`logits.py` in the same order),
  `text.ts` (`text.py`), `normalizer/` (kitten-text-processing's English FST
  normalizer, decoding order and float32 rounding kept as they are),
  `Tokenizer.ts` (Qwen2 byte-level BPE) and `bundle.ts` (manifest and
  voices.bin). Keep the ports line-for-line with the Python package: text and
  token ids must match it, which `tests/live/kitten-tts-2-on-device.live.test.js`
  checks against `tests/fixtures/kitten-tts-2-text.json`.
- `ios/` and `KittenTTSReactNative.podspec`: the opt-in MLX backend for Kitten
  TTS 2's language model. `KittenTTS2LanguageModel.swift` ports the Python
  package's `mlx/model.py` and `loader.py`, `KittenTTS2Sampler.swift` ports
  `sampling.ts`, `KittenTTS2AudioDecoder.swift` ports s3gen's flow and HiFT
  vocoder (weights from `tools/kitten-tts-2-onnx/decoder_mlx.py`, float16 at
  rest, evaluated layer by layer to bound memory), `KittenTTS2MLXEngine.swift`
  runs both on its own queue, and `KittenTTS2MLXModule.m` exposes it to JS as
  `KittenTTS2MLX` (used by `src/kittentts2/MLXLanguageModel.ts`). The podspec has no sources
  unless `KITTENTTS_MLX=1` or `"kittentts.mlx": "true"` in
  `ios/Podfile.properties.json` (the Expo plugin's `mlx` option), because MLX
  needs iOS 17 and pulls in mlx-swift through React Native's `spm_dependency`.
  MLX does not run in the iOS simulator: test it on a device. Building it needs
  Xcode's Metal Toolchain (`xcodebuild -downloadComponent MetalToolchain`).
- `tools/kitten-tts-2-onnx/`: the Python exporter for those files (`export.py`)
  and a parity check against the Python package (`check.py`). Not shipped in the
  npm package.
- `src/KittenModel.ts`: model names, download URLs, sizes, speed priors.
- `src/KittenVoice.ts`: voice enum and display helpers.
- `src/loader/ModelDownloader.ts`: model cache and Hugging Face downloads.
- `src/loader/NPZLoader.ts`: `voices.npz` ZIP/NPY parsing.
- `src/engine/TTSEngine.ts`: text-to-token-to-ONNX inference.
- `src/phonemizer/CEPhonemizer.ts`: JS/Emscripten phonemizer adapter.
- `src/audio/AudioOutput.ts`: optional playback helpers.
- `src/storage/AssetStorage.ts`: web/Node asset cache abstraction used by the web platform files.
- `vendor/cephonemizer/`: vendored C++ phonemizer source.
- `scripts/build-cephonemizer.js`: builds generated phonemizer runtime.
- `scripts/patch-onnxruntime-react-native.js`: postinstall ONNX Runtime compatibility patches.
- `examples/BareRNExample/`: bare React Native example.
- `examples/ExpoExample/`: Expo development-build example.
- `examples/ExpoWordTimingsExample/`: Expo SDK 55 word timing example.

## Build And Validation

Use these from the repository root:

```bash
npm run typecheck
npm run build
npm test
```

`npm run test:live` runs `tests/live/` against the real Kitten API. It is
skipped unless `KITTENML_API_KEY` is set, and it makes billed requests. The
on-device Kitten TTS 2 live tests need `KITTEN_TTS_2_ONNX` pointing at an
exported folder and `onnxruntime-node` installed (`npm install --no-save
onnxruntime-node`); they are skipped otherwise. Never
commit a key, and keep credentials in runtime-private (`#`) fields so a logged
or serialised instance cannot leak them.

`npm run build` writes compiled output to `lib/` and copies generated
phonemizer files. Do not hand-edit `lib/` unless the user explicitly asks;
prefer editing `src/` and rebuilding.

`npm run build:phonemizer` requires Emscripten and regenerates files under
`src/phonemizer/generated/`.

## Dependency Notes

- Runtime dependencies are `onnxruntime-react-native`,
  `@dr.pogodin/react-native-fs` via the `react-native-fs` alias, and `pako`.
- Playback is optional. Users must pass either `createExpoAudioPlayer()` with
  `expo-audio`, `createRNSoundPlayer()` with `react-native-sound`, or their own
  `AudioPlayer`.
- Expo Go is not supported because ONNX Runtime and filesystem access require
  native modules. Use an Expo development build or prebuilt native project.

## Model Cache Behavior

Model files are cached under:

```text
<DocumentDirectory>/KittenTTS/<model>/
```

unless `storageDirectory` is provided. `downloadModelIfNeeded()` should only
download missing files and should not treat partial failed downloads as valid
cache entries. Keep cache behavior conservative because first-run downloads are
large.

The default `CEPhonemizer` also caches dictionary files under:

```text
<DocumentDirectory>/KittenTTS/CEPhonemizer/
```

## Error Handling

Public SDK failures should be surfaced as `KittenTTSError` with a stable
`KittenTTSErrorCode` whenever practical. Preserve underlying errors in `cause`
when wrapping them. App developers should be able to catch:

```typescript
if (isKittenTTSError(error)) {
  console.log(error.code, error.message);
}
```

Avoid leaking raw native, filesystem, parser, or playback errors from public
entry points unless there is no reasonable SDK boundary around that code.

## Editing Rules

- Keep public API changes small and documented in `README.md`.
- Prefer adding focused tests or at least running `npm run typecheck` for SDK
  changes.
- Keep generated phonemizer output untouched unless rebuilding the phonemizer.
- Preserve the package's TypeScript-first API and avoid adding app-side setup
  that breaks autolinking.
- Do not remove the ONNX Runtime postinstall patch without validating iOS dev
  reload behavior.
