import { NativeModules } from 'react-native';
import { base64ToUint8Array, uint8ArrayToBase64 } from '../audio/Base64';
import { KittenTTSError, errorMessage } from '../KittenTTSError';
import { type AudioDecoderLike, type DecoderConditioning, applyTrimFade, decoderTokens } from './AudioDecoder';
import { type LanguageModelTokens, type TokenGenerator, cancelled } from './LanguageModel';
import type { Random, SamplingSettings } from './sampling';

/**
 * The SDK's native engines for Kitten TTS 2: MLX on iOS (ios/KittenTTS2MLXEngine.swift)
 * and ggml on Android (android/src/main/cpp/kittentts2). Both run the language
 * model's token loop and sampler and the audio decoder natively, with the same calls.
 */
export type NativeBackend = 'mlx' | 'ggml';

interface NativeEngineModule {
  isAvailable(): Promise<boolean>;
  /** ggml only: "Vulkan0: <GPU name>", or "CPU" when there is no GPU it can use. */
  device?(): Promise<string>;
  load(path: string, config: Record<string, number>): Promise<void>;
  generate(
    prompt: number[],
    speaker: number[],
    settings: Record<string, number | null>,
    maxNewTokens: number,
    seed: number,
  ): Promise<number[]>;
  loadDecoder(path: string): Promise<void>;
  decode(tokens: number[], promptToken: number[], promptFeat: string, embedding: number[], seed: number): Promise<string>;
  cancel(): void;
  unload(): Promise<void>;
}

const MODULE_NAMES: Record<NativeBackend, string> = { mlx: 'KittenTTS2MLX', ggml: 'KittenTTS2GGML' };

const native = (backend: NativeBackend): NativeEngineModule | undefined =>
  (NativeModules as Record<string, NativeEngineModule | undefined>)[MODULE_NAMES[backend]];

function required(backend: NativeBackend): NativeEngineModule {
  const module = native(backend);
  if (!module) throw KittenTTSError.invalidArgument(`This app was built without the SDK's ${backend} engine.`);
  return module;
}

/** Whether this app was built with the engine and the device can run it. */
export async function isNativeAvailable(backend: NativeBackend): Promise<boolean> {
  const module = native(backend);
  if (!module) return false;
  try {
    return await module.isAvailable();
  } catch {
    return false;
  }
}

/** Where the engine runs: MLX always on the GPU, ggml on the GPU when it found one it can use. */
export async function nativeDevice(backend: NativeBackend): Promise<'gpu' | 'cpu'> {
  const module = native(backend);
  if (!module?.device) return 'gpu';
  try {
    return (await module.device()).startsWith('CPU') ? 'cpu' : 'gpu';
  } catch {
    return 'cpu';
  }
}

/**
 * The language model in a native engine, a port of the Python package's MLX
 * backend. The whole token loop and the sampler run natively; this only passes
 * the prompt in and the tokens out.
 */
export class NativeLanguageModel implements TokenGenerator {
  private constructor(private readonly module: NativeEngineModule, private readonly tokens: LanguageModelTokens) {}

  static async load(
    backend: NativeBackend,
    path: string,
    config: Record<string, number>,
    tokens: LanguageModelTokens,
  ): Promise<NativeLanguageModel> {
    const module = required(backend);
    try {
      await module.load(path, config);
    } catch (error) {
      throw KittenTTSError.inferenceFailed(`Could not load the ${backend} language model: ${errorMessage(error)}`, error);
    }
    return new NativeLanguageModel(module, tokens);
  }

  async generate(
    prompt: readonly number[],
    speaker: Float32Array,
    settings: SamplingSettings,
    maxNewTokens: number,
    random: Random,
    signal?: { readonly aborted: boolean },
  ): Promise<number[]> {
    if (signal?.aborted) throw cancelled();
    // The native sampler has its own generator; each chunk gets a seed from this one.
    const seed = Math.floor(random.next() * 0x100000000) >>> 0;
    const watch = setInterval(() => {
      if (signal?.aborted) this.module.cancel();
    }, 50);
    let ids: number[];
    try {
      ids = await this.module.generate([...prompt], Array.from(speaker), {
        ...settings,
        speechEnd: this.tokens.speechEnd,
        stop: this.tokens.stop,
        audioStart: this.tokens.audioStart,
        audioEnd: this.tokens.audioEnd,
      }, maxNewTokens, seed);
    } catch (error) {
      if (signal?.aborted || (error as { code?: string })?.code === 'CANCELLED') throw cancelled();
      throw KittenTTSError.inferenceFailed(errorMessage(error), error);
    } finally {
      clearInterval(watch);
    }
    const { audioStart, audioEnd } = this.tokens;
    return ids.filter(id => id >= audioStart && id < audioEnd).map(id => id - audioStart);
  }

  async release(): Promise<void> {
    await this.module.unload();
  }
}

/**
 * The audio decoder in a native engine: s3gen's flow and HiFT vocoder. Samples
 * come back as base64 float32 bytes.
 */
export class NativeAudioDecoder implements AudioDecoderLike {
  private constructor(private readonly module: NativeEngineModule, private readonly silenceToken: number) {}

  static async load(backend: NativeBackend, path: string, silenceToken: number): Promise<NativeAudioDecoder> {
    const module = required(backend);
    try {
      await module.loadDecoder(path);
    } catch (error) {
      throw KittenTTSError.inferenceFailed(`Could not load the ${backend} audio decoder: ${errorMessage(error)}`, error);
    }
    return new NativeAudioDecoder(module, silenceToken);
  }

  async decode(codes: readonly number[], conditioning: DecoderConditioning, random: Random): Promise<Float32Array | null> {
    const tokens = decoderTokens(codes, this.silenceToken);
    if (!tokens) return null;
    const feat = conditioning.promptFeat;
    const seed = Math.floor(random.next() * 0x100000000) >>> 0;
    let base64: string;
    try {
      base64 = await this.module.decode(
        tokens,
        Array.from(conditioning.promptToken),
        uint8ArrayToBase64(new Uint8Array(feat.buffer, feat.byteOffset, feat.byteLength)),
        Array.from(conditioning.embedding),
        seed,
      );
    } catch (error) {
      throw KittenTTSError.inferenceFailed(errorMessage(error), error);
    }
    const bytes = base64ToUint8Array(base64);
    return applyTrimFade(new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4));
  }

  async release(): Promise<void> {
    // The language model's release unloads both; the engine is shared.
  }
}
