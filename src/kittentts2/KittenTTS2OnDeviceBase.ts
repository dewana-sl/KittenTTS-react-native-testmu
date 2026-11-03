import type { AbortSignalLike } from '../api/KittenAPI';
import type { KittenTTS2Platform, KittenTTS2Voice } from '../api/KittenTTS2Base';
import { KittenTTSError, errorMessage, isKittenTTSError } from '../KittenTTSError';
import { KittenTTSResult } from '../KittenTTSResult';
import { AudioDecoder, type AudioDecoderLike } from './AudioDecoder';
import pako from 'pako';
import { type BundleManifest, type KittenTTS2Precision, type VoiceData, VoiceFile } from './bundle';
import { LanguageModel, type LanguageModelTokens, type TokenGenerator, cancelled } from './LanguageModel';
import { Normalizer } from './normalizer/Normalizer';
import type { OrtModule, OrtSessionOptions } from './ort';
import { Random, type SamplingSettings } from './sampling';
import { CHUNK_GAP_S, hasExpressionTag, joinChunks, normalizeText, splitForSynthesis } from './text';
import { QwenTokenizer } from './Tokenizer';

/** The Hugging Face repository the Python package loads Kitten TTS 2 from. */
export type KittenTTS2Repository = `${string}/kitten-tts-2${string}`;

function languageModelTokens(manifest: BundleManifest): LanguageModelTokens {
  return {
    speechEnd: manifest.token_map.speech_end_id,
    stop: manifest.token_map.stop_id,
    audioStart: manifest.token_map.audio_id_base,
    audioEnd: manifest.token_map.audio_id_base + manifest.token_map.num_audio_tokens,
  };
}

/** Whether `KittenTTS.create()` options select Kitten TTS 2 on the device. */
export function isKittenTTS2Repository(model: unknown): model is KittenTTS2Repository {
  return typeof model === 'string' && /^[\w.-]+\/kitten-tts-2[\w.-]*$/.test(model);
}

export function checkRepository(model: unknown): KittenTTS2Repository {
  if (!isKittenTTS2Repository(model)) {
    throw KittenTTSError.invalidArgument("model must be a Kitten TTS 2 repository such as 'KittenML/kitten-tts-2'.");
  }
  return model;
}

/** The repository's `onnx/` folder on Hugging Face, where the SDK looks by default. */
export function defaultBaseURL(repository: string): string {
  return `https://huggingface.co/${repository}/resolve/main/onnx`;
}

// The codec runs at 25 tokens a second. Within a long text each chunk's budget
// is sized from its length at the measured 20 characters a second, with slack,
// so one runaway chunk cannot use the whole allowance.
const TOKENS_PER_SECOND = 25;
const CHARS_PER_SECOND = 20;
const CHUNK_BUDGET_SLACK = 1.8;
const MIN_CHUNK_BUDGET = 200;
const DEFAULT_MAX_NEW_TOKENS = 1000;

/** The rarely needed knobs, as the Python package's `advanced=` dict holds them. */
export interface KittenTTS2AdvancedOptions {
  /** Penalty on tokens already seen. Defaults to 1.1. */
  repetitionPenalty?: number;
  /** Limit that penalty to the last N generated tokens. Defaults to the model's 50. */
  repetitionWindow?: number;
  /** Escalating penalty once one codec token repeats past `tokenRunGrace`. Defaults to 1.3. */
  tokenRunPenalty?: number;
  /** Defaults to 10. */
  tokenRunGrace?: number;
  /** Chunk length for long text, in characters. Defaults to the model's 380. */
  chunkChars?: number;
  /** Chunks shorter than this are merged into a neighbour. Defaults to the model's 130. */
  chunkMinChars?: number;
  /** Pause between joined chunks, in seconds. Defaults to 0.16. */
  chunkGapSeconds?: number;
  /** Emotion conditioning; by default on whenever the text carries an expression tag. */
  useEmotion?: boolean;
  /** Seed the sampler for a reproducible generation (on this runtime). */
  seed?: number;
}

/** Per-call options for the on-device `generate()`, `generateStreaming()` and `speak()`. */
export interface KittenTTS2OnDeviceGenerateOptions {
  /** A name from `availableVoices`, such as `'Bruno'`. Defaults to the model's default voice. */
  voice?: string;
  /** Decoding preset: `'stable'` (the default) or `'expressive'`. */
  preset?: 'stable' | 'expressive' | (string & {});
  /** Codec tokens per chunk at most; 25 tokens is about a second. Defaults to 1000. */
  maxNewTokens?: number;
  /** Read numbers, dates and currencies as words. Defaults to true. */
  normalize?: boolean;
  /** Include the voice's reference clip in the prompt. Defaults to the model's setting. */
  useReferencePrompt?: boolean;
  /** Override the preset's temperature. */
  temperature?: number;
  /** Override the preset's top-k (0 turns it off). */
  topK?: number;
  /** Override the preset's top-p. */
  topP?: number;
  /** Override the preset's min-p. */
  minP?: number;
  advanced?: KittenTTS2AdvancedOptions;
  /** Stops generation between tokens. */
  signal?: AbortSignalLike;
}

/** Everything a platform's `create()` loads before the instance exists. */
export interface KittenTTS2OnDeviceState {
  repository: string;
  manifest: BundleManifest;
  backend: 'mlx' | 'ggml' | 'onnx';
  device: 'gpu' | 'cpu';
  precision: KittenTTS2Precision | 'bfloat16';
  languageModel: TokenGenerator;
  decoder: AudioDecoderLike;
  tokenizer: QwenTokenizer;
  normalizer: Normalizer | null;
  voiceFile: VoiceFile;
  defaultVoice?: string;
  audioOutput: KittenTTS2Platform['audioOutput'];
}

/** A graph as the platform's ONNX Runtime opens it, with its session options. */
export interface KittenTTS2Graph {
  model: string | Uint8Array;
  options: OrtSessionOptions;
}

export interface KittenTTS2Graphs {
  lm: KittenTTS2Graph;
  flow: KittenTTS2Graph;
  hift: KittenTTS2Graph;
}

/** The models a backend other than ONNX Runtime loads. */
export interface KittenTTS2Models {
  languageModel: TokenGenerator;
  decoder: AudioDecoderLike;
}

/** How a platform reads the downloaded files, by their path in the manifest. */
export interface KittenTTS2BundleReader {
  readText(path: string): Promise<string>;
  readBytes(path: string, offset?: number, length?: number): Promise<Uint8Array>;
  /** What ONNX Runtime opens a graph from: a local path, a URL, or its bytes. */
  graph(path: string): Promise<string | Uint8Array>;
  /** Extra session options for one graph, such as where its external data is. */
  graphOptions?(path: string): OrtSessionOptions | undefined;
}

export interface KittenTTS2OpenOptions {
  repository: string;
  manifest: BundleManifest;
  reader: KittenTTS2BundleReader;
  ort: OrtModule;
  sessionOptions: OrtSessionOptions;
  audioOutput: KittenTTS2Platform['audioOutput'];
  defaultVoice?: string;
  precision?: KittenTTS2Precision;
  /** Loads the language model and the decoder in a native engine instead of ONNX Runtime. */
  native?: KittenTTS2NativeEngine;
}

/** A native engine as `open()` takes it: MLX (iOS) or ggml (Android). */
export interface KittenTTS2NativeEngine {
  backend: 'mlx' | 'ggml';
  device: 'gpu' | 'cpu';
  /** How its language model multiplies: `'bfloat16'` on MLX, `'float32'` on ggml's GPU, `'int8'` on ggml's CPU. */
  precision: KittenTTS2Precision | 'bfloat16';
  load(tokens: LanguageModelTokens): Promise<KittenTTS2Models>;
}

interface Call {
  text: string;
  voice: string;
  options: KittenTTS2OnDeviceGenerateOptions;
}

/**
 * Kitten TTS 2 on the device, with the Python package's calls: built-in voices,
 * expression tags, decoding presets and the same sampling pipeline. Both models
 * run through ONNX Runtime on the CPU, or in the platform's native engine: MLX
 * on iOS, ggml on Android, on the GPU.
 */
export class KittenTTS2OnDeviceBase {
  /** The model repository this instance was created from, such as `'KittenML/kitten-tts-2'`. */
  readonly model: string;
  /** Sample rate of every result: 24,000 Hz. */
  readonly sampleRate: number;
  /** The built-in voices. */
  readonly voices: readonly KittenTTS2Voice[];
  /** What runs the models: `'mlx'` (iOS), `'ggml'` (Android) or `'onnx'` (ONNX Runtime). */
  readonly backend: 'mlx' | 'ggml' | 'onnx';
  /** Where they run: `'gpu'` or `'cpu'`. ggml runs on the CPU on phones without a GPU it can use. */
  readonly device: 'gpu' | 'cpu';
  /**
   * How the language model multiplies its activations: `'int8'` or `'float32'`
   * on ONNX Runtime, `'bfloat16'` on MLX, `'float32'` on ggml's GPU and `'int8'` on its CPU.
   */
  readonly precision: KittenTTS2Precision | 'bfloat16';

  private readonly state: KittenTTS2OnDeviceState;
  private readonly voiceCache = new Map<string, Promise<VoiceData>>();
  private lock: Promise<void> = Promise.resolve();
  private disposed = false;

  protected constructor(state: KittenTTS2OnDeviceState) {
    this.state = state;
    this.model = state.repository;
    this.sampleRate = state.manifest.sample_rate;
    this.backend = state.backend;
    this.device = state.device;
    this.precision = state.precision;
    this.voices = Object.freeze(state.voiceFile.entries.map(entry => Object.freeze({
      id: entry.name,
      name: entry.name,
      language: entry.language,
      type: 'public',
      experimental: false,
    })));
  }

  /** Names accepted by `voice`. */
  get availableVoices(): string[] {
    return this.voices.map(voice => voice.id);
  }

  /** Names accepted by `preset`. */
  get presets(): string[] {
    return Object.keys(this.state.manifest.decode_presets);
  }

  /** Load the tokenizer, grammars, voices and graphs. Shared by each platform's `create()`. */
  protected static async open(options: KittenTTS2OpenOptions): Promise<KittenTTS2OnDeviceState> {
    const { manifest, reader } = options;
    let tokenizer: QwenTokenizer;
    let normalizer: Normalizer;
    let voiceFile: VoiceFile;
    try {
      tokenizer = QwenTokenizer.fromFiles(
        await reader.readText(manifest.tokenizer.vocab),
        await reader.readText(manifest.tokenizer.merges),
        await reader.readText(manifest.tokenizer.added_tokens),
      );
      const grammar = async (path: string) => pako.ungzip(await reader.readBytes(path));
      const en = manifest.normalizer.en;
      normalizer = new Normalizer({
        tagger: await grammar(en.tagger),
        verbalizer: await grammar(en.verbalizer),
        post: en.post ? await grammar(en.post) : undefined,
      });
      voiceFile = await VoiceFile.open((offset, length) => reader.readBytes(manifest.voices, offset, length));
    } catch (error) {
      if (isKittenTTSError(error)) throw error;
      throw KittenTTSError.invalidModelData(`Could not read the Kitten TTS 2 files: ${errorMessage(error)}`, error);
    }
    const defaultVoice = options.defaultVoice?.trim();
    if (defaultVoice && !voiceFile.entries.some(entry => entry.name === defaultVoice)) {
      throw KittenTTSError.invalidArgument(`defaultVoice '${defaultVoice}' is not one of the model's voices.`);
    }
    const precision = options.native ? options.native.precision : options.precision ?? manifest.lm.default_precision;
    if (!options.native && !manifest.lm.graphs[precision as KittenTTS2Precision]) {
      throw KittenTTSError.invalidArgument(`precision must be one of: ${Object.keys(manifest.lm.graphs).join(', ')}.`);
    }
    const graph = async (path: string) => ({
      model: await reader.graph(path),
      options: { ...options.sessionOptions, ...reader.graphOptions?.(path) },
    });
    const models = options.native
      ? await options.native.load(languageModelTokens(manifest))
      : await KittenTTS2OnDeviceBase.loadModels(options.ort, {
        lm: await graph(manifest.lm.graphs[precision as KittenTTS2Precision]),
        flow: await graph(manifest.decoder.flow),
        hift: await graph(manifest.decoder.hift),
      }, manifest);
    return {
      repository: options.repository,
      manifest,
      backend: options.native ? options.native.backend : 'onnx',
      device: options.native ? options.native.device : 'cpu',
      precision,
      tokenizer,
      normalizer,
      voiceFile,
      defaultVoice,
      audioOutput: options.audioOutput,
      ...models,
    };
  }

  /** Load the three ONNX graphs. */
  protected static async loadModels(
    ort: OrtModule,
    graphs: KittenTTS2Graphs,
    manifest: BundleManifest,
  ): Promise<KittenTTS2Models> {
    let languageModel: TokenGenerator | undefined;
    try {
      languageModel = await LanguageModel.load(ort, graphs.lm.model, {
        layers: manifest.lm.layers,
        kvHeads: manifest.lm.kv_heads,
        headDim: manifest.lm.head_dim,
        headStart: manifest.lm.head_start,
        speakerDim: manifest.lm.speaker_dim,
      }, languageModelTokens(manifest), graphs.lm.options);
      const decoder = await AudioDecoder.load(ort, graphs.flow, graphs.hift, {
        cfmSteps: manifest.decoder.cfm_steps,
        silenceToken: manifest.decoder.silence_token,
        tokenMelRatio: manifest.decoder.token_mel_ratio,
        samplesPerFrame: manifest.decoder.samples_per_frame,
        mels: manifest.decoder.mels,
      });
      return { languageModel, decoder };
    } catch (error) {
      // The language model alone holds 1.4 GB; do not leave it loaded.
      await languageModel?.release().catch(() => {});
      if (isKittenTTSError(error)) throw error;
      throw KittenTTSError.inferenceFailed(`Could not load Kitten TTS 2: ${errorMessage(error)}`, error);
    }
  }

  /**
   * One short generation, thrown away. MLX compiles its GPU kernels on first
   * use, which took several seconds on an iPhone 17; this moves that into create().
   */
  protected async warmUp(): Promise<void> {
    await this.generate('Hello.', { maxNewTokens: 12 }).catch(() => {});
  }

  /** The spoken form the model hears for `text`, as the Python package's `normalize_text()`. */
  normalizeText(text: string): string {
    return normalizeText(text, this.normalizerBackend());
  }

  /**
   * Synthesise speech and return it once complete. Long text is generated in
   * sentence chunks and joined with short pauses.
   *
   * @example
   * ```typescript
   * await tts.generate('Hello there.', 'Bruno');
   * await tts.generate('[joyful] We won the grant <laugh>!', { voice: 'Luna', preset: 'expressive' });
   * ```
   */
  generate(text: string, voice?: string): Promise<KittenTTSResult<string>>;
  generate(text: string, options?: KittenTTS2OnDeviceGenerateOptions): Promise<KittenTTSResult<string>>;
  async generate(
    text: string,
    voiceOrOptions?: string | KittenTTS2OnDeviceGenerateOptions,
  ): Promise<KittenTTSResult<string>> {
    const call = this.prepare(text, voiceOrOptions);
    const release = await this.acquire();
    try {
      const wavs: Float32Array[] = [];
      for await (const wav of this.synthesize(call)) wavs.push(wav);
      const gap = call.options.advanced?.chunkGapSeconds ?? CHUNK_GAP_S;
      const joined = joinChunks(wavs, this.sampleRate, gap);
      if (!joined) throw KittenTTSError.emptyOutput();
      return this.result(joined, call);
    } finally {
      release();
    }
  }

  /**
   * Synthesise speech chunk by chunk, yielding each chunk's audio as soon as it
   * is decoded. Text short enough for one chunk yields once.
   */
  generateStreaming(text: string, voice?: string): AsyncGenerator<KittenTTSResult<string>, void, void>;
  generateStreaming(
    text: string,
    options?: KittenTTS2OnDeviceGenerateOptions,
  ): AsyncGenerator<KittenTTSResult<string>, void, void>;
  async *generateStreaming(
    text: string,
    voiceOrOptions?: string | KittenTTS2OnDeviceGenerateOptions,
  ): AsyncGenerator<KittenTTSResult<string>, void, void> {
    const call = this.prepare(text, voiceOrOptions);
    const release = await this.acquire();
    try {
      let yielded = false;
      for await (const wav of this.synthesize(call)) {
        yielded = true;
        yield this.result(wav, call);
      }
      if (!yielded) throw KittenTTSError.emptyOutput();
    } finally {
      release();
    }
  }

  /** Synthesise speech and play it through the configured player. */
  speak(text: string, voice?: string): Promise<KittenTTSResult<string>>;
  speak(text: string, options?: KittenTTS2OnDeviceGenerateOptions): Promise<KittenTTSResult<string>>;
  async speak(
    text: string,
    voiceOrOptions?: string | KittenTTS2OnDeviceGenerateOptions,
  ): Promise<KittenTTSResult<string>> {
    const result = await this.generate(text, voiceOrOptions as KittenTTS2OnDeviceGenerateOptions);
    await this.play(result);
    return result;
  }

  /** Play a previously generated result. */
  async play(result: KittenTTSResult<string>, options: { onPlaybackStart?: () => void } = {}): Promise<void> {
    if (this.disposed) throw KittenTTSError.engineNotReady();
    await this.state.audioOutput.play(result.samples, result.sampleRate, options);
  }

  /** Stop any currently active audio playback. */
  async stopSpeaking(): Promise<void> {
    await this.state.audioOutput.stop();
  }

  /** Stop generation, free the models and stop playback. The instance cannot be used afterwards. */
  async dispose(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;
    await this.state.audioOutput.stop().catch(() => {});
    const release = await this.acquire();
    try {
      await this.state.languageModel.release();
      await this.state.decoder.release();
    } finally {
      release();
    }
  }

  private prepare(text: string, voiceOrOptions: string | KittenTTS2OnDeviceGenerateOptions | undefined): Call {
    if (this.disposed) throw KittenTTSError.engineNotReady();
    if (typeof text !== 'string') throw KittenTTSError.invalidArgument('text must be a string.');
    if (!text.trim()) throw KittenTTSError.emptyInput();
    const options: KittenTTS2OnDeviceGenerateOptions =
      typeof voiceOrOptions === 'string' || voiceOrOptions === undefined
        ? { voice: voiceOrOptions }
        : voiceOrOptions;
    if ((options as { reference?: unknown }).reference !== undefined) {
      throw KittenTTSError.invalidArgument(
        'Voice cloning is not available on the device yet; clone through the Kitten API, ' +
        "with KittenTTS.create({ model: 'kitten-tts-2', apiKey }).",
      );
    }
    const voice = options.voice?.trim() || this.state.defaultVoice || this.state.manifest.default_voice;
    if (!voice) throw KittenTTSError.invalidArgument('No voice given and the model declares no default voice.');
    if (!this.availableVoices.includes(voice)) {
      throw KittenTTSError.invalidArgument(`Voice '${voice}' is not available. Choose from availableVoices.`);
    }
    const preset = options.preset ?? this.state.manifest.default_preset;
    if (!this.state.manifest.decode_presets[preset]) {
      throw KittenTTSError.invalidArgument(`Unknown preset '${preset}'. Choose from: ${this.presets.join(', ')}.`);
    }
    return { text, voice, options };
  }

  /** `_synthesize` of the Python package: one waveform per text chunk. */
  private async *synthesize(call: Call): AsyncGenerator<Float32Array, void, void> {
    const { manifest, tokenizer, languageModel, decoder } = this.state;
    const { options } = call;
    const advanced = options.advanced ?? {};
    const spoken = normalizeText(call.text, this.normalizerBackend(), options.normalize ?? true);
    if (!spoken.trim()) throw KittenTTSError.emptyInput();
    const chunks = splitForSynthesis(
      spoken,
      advanced.chunkChars ?? manifest.generation.chunk_chars,
      advanced.chunkMinChars ?? manifest.generation.chunk_min_chars,
    );
    const settings = this.settings(options);
    const random = new Random(advanced.seed);
    const useEmotion = advanced.useEmotion ?? hasExpressionTag(call.text);
    const control = manifest.generation.emotion_control;
    const emotionIds = control && useEmotion ? tokenizer.encode(control) : [];
    const useReference = options.useReferencePrompt ?? manifest.generation.use_reference_prompt ?? true;
    const maxNewTokens = options.maxNewTokens ?? DEFAULT_MAX_NEW_TOKENS;
    // Checked between tokens: the caller's signal, or dispose().
    const isStopped = () => Boolean(options.signal?.aborted) || this.disposed;
    const stop = { get aborted() { return isStopped(); } };
    const tokens = manifest.token_map;

    try {
      const voice = await this.voiceData(call.voice);
      for (const chunk of chunks) {
        if (stop.aborted) throw cancelled();
        const budget = chunks.length === 1
          ? maxNewTokens
          : Math.max(MIN_CHUNK_BUDGET, Math.min(maxNewTokens,
            Math.floor((chunk.length / CHARS_PER_SECOND) * TOKENS_PER_SECOND * CHUNK_BUDGET_SLACK)));
        const prompt = [
          ...(useReference ? voice.prefix : [tokens.start_id]),
          ...emotionIds,
          tokens.text_start_id,
          ...tokenizer.encode(chunk),
          ...(tokens.final_seg_id ? [tokens.final_seg_id] : []),
          tokens.speech_start_id,
        ];
        const codes = await languageModel.generate(prompt, voice.speaker, settings, budget, random, stop);
        if (stop.aborted) throw cancelled();
        const wav = await decoder.decode(codes, voice.conditioning, random);
        if (wav) yield wav;
      }
    } catch (error) {
      if (isKittenTTSError(error)) throw error;
      throw KittenTTSError.inferenceFailed(errorMessage(error), error);
    }
  }

  private settings(options: KittenTTS2OnDeviceGenerateOptions): SamplingSettings {
    const preset = this.state.manifest.decode_presets[options.preset ?? this.state.manifest.default_preset];
    const advanced = options.advanced ?? {};
    return {
      temperature: options.temperature ?? preset.temperature,
      topK: options.topK ?? preset.top_k,
      topP: options.topP ?? preset.top_p,
      minP: options.minP ?? preset.min_p ?? 0,
      repetitionPenalty: advanced.repetitionPenalty ?? 1.1,
      repetitionWindow: advanced.repetitionWindow ?? this.state.manifest.generation.repetition_window ?? null,
      tokenRunPenalty: advanced.tokenRunPenalty ?? 1.3,
      tokenRunGrace: advanced.tokenRunGrace ?? 10,
    };
  }

  private voiceData(name: string): Promise<VoiceData> {
    let data = this.voiceCache.get(name);
    if (!data) {
      const entry = this.state.voiceFile.entries.find(item => item.name === name)!;
      data = this.state.voiceFile.load(entry);
      data.catch(() => this.voiceCache.delete(name));
      this.voiceCache.set(name, data);
    }
    return data;
  }

  private normalizerBackend(): ((text: string) => string) | null {
    const normalizer = this.state.normalizer;
    return normalizer ? (text: string) => normalizer.normalizeText(text) : null;
  }

  private result(samples: Float32Array, call: Call): KittenTTSResult<string> {
    return new KittenTTSResult<string>(samples, this.sampleRate, call.voice, 1, call.text, []);
  }

  /** One generation at a time: the language model's cache is shared. */
  private async acquire(): Promise<() => void> {
    let release!: () => void;
    const next = new Promise<void>(resolve => { release = resolve; });
    const previous = this.lock;
    this.lock = previous.then(() => next);
    await previous;
    return release;
  }
}

