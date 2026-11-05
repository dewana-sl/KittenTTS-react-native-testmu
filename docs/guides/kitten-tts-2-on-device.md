# Kitten TTS 2 On The Device

Kitten TTS 2 is KittenML's 1.7B-parameter speech model, with expression tags,
decoding presets and 115 built-in voices. The SDK can run it inside your app,
the way `KittenTTS("KittenML/kitten-tts-2")` runs it in the Python package: same
voices, same prompt, same text normalization and the same sampling pipeline.
Nothing leaves the device once the model is downloaded.

It is a large model for a phone: a 1.25 to 2 GB download, and 2 to 2.7 GB of
memory while it is loaded. On an iPhone 17 it generates speech faster than it
plays when it runs on the GPU through [MLX](#ios-mlx), and about 1.8 times
slower than it plays on the CPU, which is what apps without a native engine
use. On Android it can run on the GPU through [ggml and Vulkan](#android-ggml),
tested so far in the emulator. Play long text with `generateStreaming()`. For
voice cloning, or when the download is too big, use
[Kitten TTS 2 through the Kitten API](kitten-api.md) instead.

## Quick Start

```tsx
import * as ExpoAudio from 'expo-audio';
import { KittenTTS, createExpoAudioPlayer } from '@kittentts/react-native';

const tts = await KittenTTS.create(
  { model: 'KittenML/kitten-tts-2', player: createExpoAudioPlayer(ExpoAudio) },
  (progress, info) => console.log(`${Math.round(progress * 100)}%`, info?.file),
);

console.log(tts.availableVoices); // ['AirportAnnouncer', 'Alfred', ..., 'Bruno', ...]

await tts.speak('Hello there, this runs on the phone.', 'Bruno');

const result = await tts.generate('[joyful] We won the grant <laugh> I can (((hardly))) believe it!', {
  voice: 'Luna',
  preset: 'expressive',
});
console.log(result.duration, result.wavBase64());
```

`KittenTTS.create()` returns a `KittenTTS2OnDevice` for a Kitten TTS 2
repository. The first call downloads the files and later calls load them from
the device, offline too. Loading takes a few seconds.

## Where The Files Come From

The SDK reads `manifest.json` from the repository's `onnx/` folder on Hugging
Face (`https://huggingface.co/KittenML/kitten-tts-2/resolve/main/onnx`) and
downloads the files it lists:

| File | Size | What it is |
| --- | --- | --- |
| `lm.onnx.data` | 1.43 GB | The speech language model's weights for ONNX Runtime, with its speaker head |
| `lm-int8.onnx`, `lm.onnx` | 0.5 MB | Its graph, for each `precision` |
| `lm-ternary.safetensors` | 954 MB | The same model as MLX and ggml read it: the published packed checkpoint |
| `flow.onnx` | 482 MB | Codec tokens to mel spectrogram (s3gen's flow) |
| `hift.onnx` | 83 MB | Mel spectrogram to 24 kHz audio (s3gen's HiFT vocoder) |
| `decoder-mlx.safetensors` | 271 MB | The flow's and vocoder's weights as MLX and ggml read them, in float16 |
| `voices.bin` | 19 MB | The built-in voices, precomputed |
| `decoder-student_w4-mlx.safetensors`, `voices-student_w4.bin` | 115 MB | The one-step decoder and its voices, downloaded only when you ask for it |
| tokenizer and grammars | 5 MB | Qwen's tokenizer, kitten-text-processing's English grammars |

The files are built from the model repository by
[`tools/kitten-tts-2-onnx`](../../tools/kitten-tts-2-onnx/README.md). To host
them yourself, upload its output folder and pass that URL:

```tsx
const tts = await KittenTTS.create({
  model: 'KittenML/kitten-tts-2',
  modelBaseURL: 'https://cdn.example.com/kitten-tts-2-onnx',
});
```

A device downloads one copy of each model, for the backend it uses: 1.25 GB in
all with MLX or ggml, 2.02 GB with ONNX Runtime. Files already on the device,
at the size and SHA-256 the manifest lists, are not downloaded again, and an
interrupted download resumes at the next file. `forceRedownload: true` deletes
them first. They live under
`<storageDirectory>/KittenML--kitten-tts-2/`, by default in the app's documents
folder.

## Voices

`tts.availableVoices` lists the names, as the Python package's
`available_voices` does. `tts.voices` adds each voice's language:

```tsx
const spanish = tts.voices.filter(voice => voice.language === 'es');
await tts.generate('Hola, ¿cómo estás?', spanish[0].id);
```

Without a voice, calls use `defaultVoice` from `create()`, or the model's own
default, `Bruno`.

## Generate

```tsx
await tts.generate(text, 'Bruno');
await tts.generate(text, {
  voice: 'Bruno',
  preset: 'stable',          // or 'expressive'
  temperature: 0.7,          // override one preset value
  maxNewTokens: 1000,        // per chunk; 25 tokens is about a second
  normalize: true,           // "$12.50" -> "twelve dollars fifty cents"
  advanced: { seed: 1 },     // the Python package's advanced= options
  signal: controller.signal, // cancel between tokens
});
```

| Option | Python | Default |
| --- | --- | --- |
| `voice` | `voice` | the model's default voice |
| `preset` | `preset` | `'stable'` |
| `temperature`, `topK`, `topP`, `minP` | `temperature`, `top_k`, `top_p`, `min_p` | the preset's |
| `maxNewTokens` | `max_new_tokens` | `1000` |
| `normalize` | `normalize` | `true` |
| `useReferencePrompt` | `use_reference_prompt` | the model's (`true`) |
| `advanced.repetitionPenalty` | `repetition_penalty` | `1.1` |
| `advanced.repetitionWindow` | `repetition_window` | the model's (`50`) |
| `advanced.tokenRunPenalty`, `advanced.tokenRunGrace` | `token_run_penalty`, `token_run_grace` | `1.3`, `10` |
| `advanced.chunkChars`, `advanced.chunkMinChars` | `chunk_chars`, `chunk_min_chars` | the model's (`380`, `130`) |
| `advanced.chunkGapSeconds` | `chunk_gap_s` | `0.16` |
| `advanced.useEmotion` | `use_emotion` | on when the text has a tag |
| `advanced.seed` | `seed` | random |

Text supports the same expression controls: a leading `[emotion]` tag (`mundane`,
`nervous`, `tender`, `angry`, `excited`, `stern`, `sad`, `contemplative`,
`surprised`, `joyful`), inline `<event>` tags (`pause`, `sigh`, `gasp`, `laugh`,
`giggle`, `sob`, `scoff`, `growl`, `um`, `gulp`) and `(((emphasis)))` spans.
`tts.normalizeText(text)` shows what the model will hear.

A seed reproduces a generation on this runtime. As between the Python package's
own backends, it does not reproduce one made in Python.

## Stream

Long text is generated in sentence chunks of about 380 characters.
`generateStreaming()` yields each chunk as soon as it is decoded, so playback
can start after the first one:

```tsx
for await (const chunk of tts.generateStreaming(longText, 'Eleanor')) {
  await tts.play(chunk);
}
```

`generate()` waits for every chunk and joins them with short pauses, trimming
each chunk's edge silence, as the Python package does.

## Run It On The GPU

Both platforms have a native engine that runs the language model and the audio
decoder on the GPU, built into the app only when you ask. The calls are the
same; `backend` decides, like the Python package's `backend=`:

```tsx
const tts = await KittenTTS.create({ model: 'KittenML/kitten-tts-2' });      // 'auto': the native engine, on the GPU
await KittenTTS.create({ model: 'KittenML/kitten-tts-2', backend: 'onnx' }); // always ONNX Runtime, on the CPU
await KittenTTS.create({ model: 'KittenML/kitten-tts-2', backend: 'mlx' });  // iOS; throws where MLX is missing
await KittenTTS.create({ model: 'KittenML/kitten-tts-2', backend: 'ggml' }); // Android; throws where ggml is missing
console.log(tts.backend, tts.device); // 'mlx' 'gpu', 'ggml' 'gpu', 'ggml' 'cpu' or 'onnx' 'cpu'
```

`create()` runs one short generation before it returns, so the GPU kernels
compile there and not in your first call.

### iOS: MLX

Kitten TTS 2 runs on the GPU through
[MLX](https://github.com/ml-explore/mlx-swift), as the Python package's
`backend="mlx"` runs its language model on a Mac:

- The language model: the same ternary weights unpacked into MLX's 4-bit format
  without rounding, bfloat16 activations, and the token loop and the sampler in
  Swift, so nothing crosses to JavaScript per token.
- The audio decoder: s3gen's flow and HiFT vocoder ported to MLX, in float32.
  Against the Python package's decoder with the same noise, the mel is within
  0.026 and the waveform within 0.004.

MLX needs iOS 17 or later and a real device; the iOS simulator lacks the GPU
features it uses.

It raises the app's minimum iOS version, so it is opt-in. With Expo, turn it on
in the config plugin and rebuild:

```json
{
  "expo": {
    "plugins": [["@kittentts/react-native", { "mlx": true }]]
  }
}
```

The plugin sets `"kittentts.mlx": "true"` and an iOS deployment target of 17.0
in `ios/Podfile.properties.json`. In a bare React Native app, raise the
Podfile's `platform :ios` to 17.0 and run `KITTENTTS_MLX=1 pod install`.

### Android: ggml

Android has no MLX. Kitten TTS 2 runs on the GPU through
[ggml](https://github.com/ggml-org/ggml)'s Vulkan backend, with both models
written for it in C++ (`android/src/main/cpp/kittentts2`) from the same weights
files as MLX, so the download is the same 1.25 GB:

- The language model: the ternary weights unpacked into ggml's 4-bit Q4_0 format
  without rounding, float32 activations, and the token loop and the sampler in
  C++. The prefill and every step match the Python package's MLX backend on the
  same top tokens.
- The audio decoder: s3gen's flow and HiFT vocoder. Against the Python package's
  decoder with the same noise, the mel is within 0.014 and the waveform within
  0.002.

It adds about 25 MB of native code per ABI and needs Vulkan's shaders compiled at
build time, so it is opt-in. With Expo, turn it on in the config plugin and rebuild:

```json
{
  "expo": {
    "plugins": [["@kittentts/react-native", { "ggml": true }]]
  }
}
```

In a bare React Native app, add `kittentts.ggml=true` to `android/gradle.properties`.
The first build downloads ggml (37 MB) and the Vulkan headers from GitHub, and
takes a few minutes longer. It needs a C++ compiler for your own machine, for the
step that generates the shaders.

It runs on Android 9 or later, on arm64 phones from about 2018 on (the CPU needs
dot-product and fp16 instructions) and on x86_64 emulators. 32-bit builds of the
app leave it out. On a phone it cannot use, or whose GPU ggml rejects, `'auto'`
falls back to ONNX Runtime, and `backend: 'ggml'` runs on the CPU if the phone
has no usable GPU; it was no faster than ONNX Runtime there.

The Android emulator's GPU works, and is a good place to develop, but its speed
says little about phones: it is your computer's GPU behind a translation layer.
It has not yet been measured on a phone.

## Speed, Precision And Memory

Without a native engine, the language model and the decoder run through ONNX
Runtime on the CPU, one generation at a time. The language model's weights are
the published ternary ones, exactly, packed at 4 bits as with MLX. `precision`
decides how its activations are multiplied:

- `'int8'` (the default) quantizes them to int8 inside each matrix product.
  About 2.5 times faster.
- `'float32'` keeps them in float32, which matches the Python package's float32
  output to within 2e-4 in the logits.

```tsx
await KittenTTS.create({ model: 'KittenML/kitten-tts-2', precision: 'float32' });
```

Both read the same weights file, so switching costs no second download.

Measured on an iPhone 17, in a Release build, on the same 8 sentences. The
real-time factor is seconds of generating per second of speech, so below 1 is
faster than the audio plays:

| iPhone 17 | Real-time factor | Words right (8 sentences) | Speaker similarity | Memory |
| --- | --- | --- | --- | --- |
| `backend: 'mlx'` | 0.55 to 0.79 | all | 0.752 | 2.1 to 2.7 GB |
| `backend: 'onnx'`, `'int8'`, 4 threads | 1.80 | all | 0.761 | 1.8 GB |
| `backend: 'onnx'`, `'float32'` | 5.3 (one sentence) | | | |

The MLX range is three runs of the same 8 sentences; the phone was slower the
warmer it got.

On an M5 Pro Mac with 4 threads:

| | Real-time factor | Words right (8 sentences) | Speaker similarity |
| --- | --- | --- | --- |
| `'int8'` | 0.69 | all | 0.766 |
| `'float32'` | 1.83 | all | 0.754 |
| The Python package, torch on the CPU | 2.95 | all | 0.763 |
| The Python package, MLX on the GPU | 0.27 | | |

Speaker similarity is the cosine between each clip's speaker embedding and the
voice's reference, as the Python package measures its backends; the Python
package itself scores 0.763 on these sentences. On the iPhone, loading the
model from the device took 5.8 s with MLX (warm-up included) and 7.4 s without.
Streaming a 29-second text delivered its first 18 seconds of audio after 13 s
with MLX and after 31 s without.

`ortNumThreads` sets the thread count. The default, 4, was the fastest on the
iPhone 17: 2 threads gave a real-time factor of 2.2, and 6 gave 3.0, because the
extra threads land on the slower efficiency cores.

The memory column is the app's footprint while generating on the iPhone 17,
which has 8 GB of RAM. With MLX the system sent the app low-memory warnings
during long chunks, so add the
`com.apple.developer.kernel.increased-memory-limit` entitlement to keep it from
being stopped. 6 GB phones may not have room.

In the Android emulator, on the same 8 sentences, in a Release build on an M5 Pro
Mac (an API 37 arm64 image with 6 cores and 8 GB of RAM):

| Android emulator | Real-time factor | Words right (8 sentences) | Speaker similarity |
| --- | --- | --- | --- |
| `backend: 'ggml'`, on the GPU | 0.77 to 0.82 | all | 0.752 |
| `backend: 'onnx'`, `'int8'`, 4 threads | 3.8 to 4.1 | all | 0.767 |

Loading took about 11 s with ggml, warm-up included, and streaming a 29-second
text delivered its first 18 seconds of audio after 11 to 13 s. ggml keeps about 1.3 GB of
weights and caches in GPU memory (0.82 GB for the language model, 0.27 GB for the
decoder, up to 0.18 GB for the attention cache) plus what a generation is
working on.

## The One-Step Decoder

`decoder: 'student_w4'` runs the Python package's optional distilled decoder on
MLX: one flow step in place of two, in a narrower network with 4-bit weights and a
shorter voice prompt (4 seconds in place of 10). The vocoder is the same. Other
backends throw, for now.

```tsx
const tts = await KittenTTS.create({ model: 'KittenML/kitten-tts-2', decoder: 'student_w4' });
console.log(tts.decoder, tts.availableDecoders); // 'student_w4' ['default', 'student_w4']
```

It is a choice for size, not for speed or quality. Measured on an iPhone 17, in a
Release build, on the same 8 sentences:

| iPhone 17, MLX | Decoder alone, real-time factor | Speaker similarity (lowest voice) | Download | Memory, usual and peak |
| --- | --- | --- | --- | --- |
| `'default'` | 0.11 to 0.13 | 0.752 (0.667) | 1.25 GB | 1.8 GiB, 2.2 GiB |
| `'student_w4'` | 0.09 | 0.684 (0.459) | 1.08 GB | 1.7 GiB, 1.9 GiB |

The words stay right, but it holds a voice less well, and some voices much less
well; listen to the voices you use. The language model takes about 80% of the time
of a generation, so a faster decoder hardly changes the overall speed: the two
were within the phone's warming-up (0.54 to 0.74 for the default, 0.65 to 0.72 for
the student). With the same noise, the mel is within 0.20 of the Python package's,
about as far as torch's own output moves when its input changes by 1e-7: the 8-bit
activations amplify rounding.

## Not On The Device Yet

- Voice cloning (`reference`): it needs the speech tokenizer, two speaker
  encoders and a transcription model on the device. Clone through the Kitten API.
- Browsers: `create()` throws there; use the Kitten API.
- The `student_w4` decoder on Android and on ONNX Runtime.

## Compared With The Python Package

| Python (`kittenml`) | React Native |
| --- | --- |
| `KittenTTS("KittenML/kitten-tts-2")` | `await KittenTTS.create({ model: 'KittenML/kitten-tts-2' })` |
| `m.generate(text, voice="Bruno")` | `await tts.generate(text, 'Bruno')` |
| `m.generate(text, preset="expressive", advanced={"seed": 1})` | `await tts.generate(text, { preset: 'expressive', advanced: { seed: 1 } })` |
| `for chunk in m.generate_stream(text)` | `for await (const chunk of tts.generateStreaming(text))` |
| `KittenTTS(repo, decoder="student_w4")` | `await KittenTTS.create({ model: repo, decoder: 'student_w4' })`, MLX only |
| `m.available_voices` | `tts.availableVoices` |
| `m.available_decoders` | `tts.availableDecoders` |
| `m.normalize_text(text)` | `tts.normalizeText(text)` |
| `m.sample_rate` | `tts.sampleRate` |
| `backend="mlx"`, `"auto"` | `backend: 'mlx'`, `'auto'` (iOS, with the MLX module) |
| `backend="torch"` (the CPU) | `backend: 'onnx'`, `precision: 'int8'` or `'float32'` |
| `m.backend` | `tts.backend` |
| returns a NumPy array | returns a `KittenTTSResult` (`samples`, `wavData()`, ...) |
