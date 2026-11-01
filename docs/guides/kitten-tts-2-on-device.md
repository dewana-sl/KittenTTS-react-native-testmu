# Kitten TTS 2 On The Device

Kitten TTS 2 is KittenML's 1.7B-parameter speech model, with expression tags,
decoding presets and 115 built-in voices. The SDK can run it inside your app,
the way `KittenTTS("KittenML/kitten-tts-2")` runs it in the Python package: same
voices, same prompt, same text normalization and the same sampling pipeline.
Nothing leaves the device once the model is downloaded.

It is a large model for a phone: a 1.5 to 2 GB download, and 2 to 2.5 GB of
memory while it is loaded. On an iPhone 17 it generates speech about as fast as
it plays when the language model runs on the GPU through
[MLX](#run-the-language-model-on-the-gpu-mlx), and about 1.8 times slower on the
CPU, which is what Android and apps without MLX use. Play long text with
`generateStreaming()`. For voice cloning, or when the download is too big, use
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
| `lm-ternary.safetensors` | 954 MB | The same model as MLX reads it: the published packed checkpoint |
| `flow.onnx` | 482 MB | Codec tokens to mel spectrogram (s3gen's flow) |
| `hift.onnx` | 83 MB | Mel spectrogram to 24 kHz audio (s3gen's HiFT vocoder) |
| `voices.bin` | 19 MB | The built-in voices, precomputed |
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

A device downloads one copy of the language model, for the backend it uses:
1.54 GB in all with MLX, 2.02 GB with ONNX Runtime. Files already on the device,
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

## Run The Language Model On The GPU (MLX)

On iOS the language model can run on the GPU through
[MLX](https://github.com/ml-explore/mlx-swift), as the Python package's
`backend="mlx"` runs it on a Mac: the same ternary weights unpacked into MLX's
4-bit format without rounding, bfloat16 activations, and the token loop and the
sampler in Swift. The audio decoder stays on ONNX Runtime. MLX needs iOS 17 or
later and a real device; the iOS simulator lacks the GPU features it uses.

It is built into the app only when you ask, since it raises the minimum iOS
version. With Expo, turn it on in the config plugin and rebuild:

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

`backend` then decides, like the Python package's `backend=`:

```tsx
await KittenTTS.create({ model: 'KittenML/kitten-tts-2' });                     // 'auto': MLX when it is built in
await KittenTTS.create({ model: 'KittenML/kitten-tts-2', backend: 'onnx' });    // always the CPU
await KittenTTS.create({ model: 'KittenML/kitten-tts-2', backend: 'mlx' });     // throws where MLX is missing
console.log(tts.backend); // 'mlx' or 'onnx'
```

## Speed, Precision And Memory

Without MLX, the language model and the decoder run through ONNX Runtime on the
CPU, one generation at a time. The language model's weights are the published
ternary ones, exactly, packed at 4 bits as with MLX. `precision` decides how its
activations are multiplied:

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
| `backend: 'mlx'` | 1.00 | all | 0.762 | 2.35 GB |
| `backend: 'onnx'`, `'int8'`, 4 threads | 1.80 | all | 0.761 | 1.8 GB |
| `backend: 'onnx'`, `'float32'` | 5.3 (one sentence) | | | |

On an M5 Pro Mac with 4 threads:

| | Real-time factor | Words right (8 sentences) | Speaker similarity |
| --- | --- | --- | --- |
| `'int8'` | 0.69 | all | 0.766 |
| `'float32'` | 1.83 | all | 0.754 |
| The Python package, torch on the CPU | 2.95 | all | 0.763 |
| The Python package, MLX on the GPU | 0.27 | | |

Speaker similarity is the cosine between each clip's speaker embedding and the
voice's reference, as the Python package measures its backends. On the iPhone,
loading the model from the device took 8.3 s with MLX and 7.4 s without.
Streaming a 29-second text delivered its first 18 seconds of audio after 20 s
with MLX and after 31 s without. With MLX, most of the remaining time is the
audio decoder on the CPU.

`ortNumThreads` sets the thread count. The default, 4, was the fastest on the
iPhone 17: 2 threads gave a real-time factor of 2.2, and 6 gave 3.0, because the
extra threads land on the slower efficiency cores.

The memory column is the app's footprint while generating on the iPhone 17,
which has 8 GB of RAM (2.6 GB at the peak seen with MLX). On iOS, add the
`com.apple.developer.kernel.increased-memory-limit` entitlement so the system
does not stop the app. 6 GB phones may not have room.

## Not On The Device Yet

- Voice cloning (`reference`): it needs the speech tokenizer, two speaker
  encoders and a transcription model on the device. Clone through the Kitten API.
- Browsers: `create()` throws there; use the Kitten API.
- The smaller `student_w4` decoder.
- The audio decoder on the GPU, and the GPU on Android: ONNX Runtime's mobile
  builds run them on the CPU.

## Compared With The Python Package

| Python (`kittenml`) | React Native |
| --- | --- |
| `KittenTTS("KittenML/kitten-tts-2")` | `await KittenTTS.create({ model: 'KittenML/kitten-tts-2' })` |
| `m.generate(text, voice="Bruno")` | `await tts.generate(text, 'Bruno')` |
| `m.generate(text, preset="expressive", advanced={"seed": 1})` | `await tts.generate(text, { preset: 'expressive', advanced: { seed: 1 } })` |
| `for chunk in m.generate_stream(text)` | `for await (const chunk of tts.generateStreaming(text))` |
| `m.available_voices` | `tts.availableVoices` |
| `m.normalize_text(text)` | `tts.normalizeText(text)` |
| `m.sample_rate` | `tts.sampleRate` |
| `backend="mlx"`, `"auto"` | `backend: 'mlx'`, `'auto'` (iOS, with the MLX module) |
| `backend="torch"` (the CPU) | `backend: 'onnx'`, `precision: 'int8'` or `'float32'` |
| `m.backend` | `tts.backend` |
| returns a NumPy array | returns a `KittenTTSResult` (`samples`, `wavData()`, ...) |
