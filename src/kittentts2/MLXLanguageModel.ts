import { NativeModules } from 'react-native';
import { base64ToUint8Array, uint8ArrayToBase64 } from '../audio/Base64';
import { KittenTTSError, errorMessage } from '../KittenTTSError';
import { type AudioDecoderLike, type DecoderConditioning, applyTrimFade, decoderTokens } from './AudioDecoder';
import { type LanguageModelTokens, type TokenGenerator, cancelled } from './LanguageModel';
import type { Random, SamplingSettings } from './sampling';

interface NativeMLX {
  isAvailable(): Promise<boolean>;
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

const native = (): NativeMLX | undefined => NativeModules.KittenTTS2MLX as NativeMLX | undefined;

/** Whether this app was built with the SDK's MLX module and runs on a GPU MLX supports. */
export async function isMLXAvailable(): Promise<boolean> {
  const module = native();
  if (!module) return false;
  try {
    return await module.isAvailable();
  } catch {
    return false;
  }
}

/**
 * The language model on the GPU through MLX (ios/KittenTTS2MLXEngine.swift), a
 * port of the Python package's MLX backend. The whole token loop and the
 * sampler run natively; this only passes the prompt in and the tokens out.
 */
export class MLXLanguageModel implements TokenGenerator {
  private constructor(private readonly module: NativeMLX, private readonly tokens: LanguageModelTokens) {}

  static async load(path: string, config: Record<string, number>, tokens: LanguageModelTokens): Promise<MLXLanguageModel> {
    const module = native();
    if (!module) throw KittenTTSError.invalidArgument('This app was built without the SDK\'s MLX module.');
    try {
      await module.load(path, config);
    } catch (error) {
      throw KittenTTSError.inferenceFailed(`Could not load the MLX language model: ${errorMessage(error)}`, error);
    }
    return new MLXLanguageModel(module, tokens);
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
 * The audio decoder on the GPU through MLX (ios/KittenTTS2AudioDecoder.swift):
 * s3gen's flow and HiFT vocoder. Samples come back as base64 float32 bytes.
 */
export class MLXAudioDecoder implements AudioDecoderLike {
  private constructor(private readonly module: NativeMLX, private readonly silenceToken: number) {}

  static async load(path: string, silenceToken: number): Promise<MLXAudioDecoder> {
    const module = native();
    if (!module) throw KittenTTSError.invalidArgument('This app was built without the SDK\'s MLX module.');
    try {
      await module.loadDecoder(path);
    } catch (error) {
      throw KittenTTSError.inferenceFailed(`Could not load the MLX audio decoder: ${errorMessage(error)}`, error);
    }
    return new MLXAudioDecoder(module, silenceToken);
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
