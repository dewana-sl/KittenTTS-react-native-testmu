import { KittenTTSError } from '../KittenTTSError';
import type { DecoderConditioning } from './AudioDecoder';
import { decodeUTF8 } from './utf8';

export const BUNDLE_FORMAT = 'kittentts-react-native/kitten-tts-2-onnx';

/**
 * How the language model multiplies: `'int8'` quantizes its activations inside
 * each matrix product (about 2.5 times faster on a CPU), `'float32'` keeps them
 * in float32. The weights are the published ones either way.
 */
export type KittenTTS2Precision = 'int8' | 'float32';

/**
 * Where the language model runs: `'onnx'`, ONNX Runtime on the CPU (iOS and
 * Android), or `'mlx'`, MLX on the GPU (iOS 17 and later, when the app builds
 * the SDK's MLX module). `'auto'` picks MLX when it is available.
 */
export type KittenTTS2Backend = 'auto' | 'mlx' | 'onnx';

export interface DecodePreset {
  temperature: number;
  top_p: number;
  top_k: number;
  min_p?: number;
}

/** manifest.json, written by tools/kitten-tts-2-onnx/export.py. */
export interface BundleManifest {
  format: string;
  version: number;
  source?: { repo: string; revision: string } | null;
  sample_rate: number;
  default_voice?: string;
  default_preset: string;
  decode_presets: Record<string, DecodePreset>;
  generation: {
    use_reference_prompt?: boolean;
    repetition_window?: number | null;
    emotion_control?: string | null;
    chunk_chars?: number;
    chunk_min_chars?: number;
  };
  token_map: {
    audio_id_base: number;
    num_audio_tokens: number;
    speech_start_id: number;
    speech_end_id: number;
    text_start_id: number;
    text_end_id: number;
    start_id: number;
    stop_id: number;
    final_seg_id?: number;
  };
  lm: {
    /** The language model graph per precision; both read lm.onnx.data. */
    graphs: Record<KittenTTS2Precision, string>;
    default_precision: KittenTTS2Precision;
    speaker_dim: number;
    head_start: number;
    layers: number;
    kv_heads: number;
    head_dim: number;
  };
  decoder: {
    name: string;
    flow: string;
    hift: string;
    cfm_steps: number;
    silence_token: number;
    token_mel_ratio: number;
    samples_per_frame: number;
    mels: number;
  };
  /** The packed checkpoint and architecture the MLX backend reads. */
  mlx?: {
    weights: string;
    /** s3gen's flow and vocoder weights, for the MLX decoder. */
    decoder: string;
    config: Record<string, number>;
  };
  tokenizer: { vocab: string; merges: string; added_tokens: string };
  normalizer: { en: { tagger: string; verbalizer: string; post?: string } };
  voices: string;
  /** Every file, with its size and hash; `backend` marks the files only one backend reads. */
  files: Array<{ path: string; size: number; sha256?: string; backend?: 'mlx' | 'onnx' }>;
}

export function checkManifest(value: unknown, source: string): BundleManifest {
  const manifest = value as Partial<BundleManifest> | null;
  if (!manifest || manifest.format !== BUNDLE_FORMAT || !Array.isArray(manifest.files)) {
    throw KittenTTSError.invalidModelData(`${source} is not a Kitten TTS 2 ONNX manifest.`);
  }
  if (manifest.version !== 1) {
    throw KittenTTSError.invalidModelData(
      `${source} has version ${manifest.version}; this SDK reads version 1. Update @kittentts/react-native.`,
    );
  }
  return manifest as BundleManifest;
}

/** A built-in voice, as voices.bin lists it. */
export interface VoiceEntry {
  name: string;
  id: string;
  language: string;
  transcript: string;
  prefix: [number, number];
  speaker: [number, number];
  prompt_token: [number, number];
  prompt_feat: [number, number];
  embedding: [number, number];
}

/** What generation needs from a built-in voice. */
export interface VoiceData {
  /** The language model's reference prefix: [START], transcript and codec tokens. */
  prefix: number[];
  /** The 512-dim speaker embedding the model's speaker head projects. */
  speaker: Float32Array;
  conditioning: DecoderConditioning;
}

const MAGIC = 'KTVOICE1';

/** Reads voices.bin through `read(offset, length)`, one voice at a time. */
export class VoiceFile {
  private constructor(
    readonly entries: VoiceEntry[],
    private readonly dataStart: number,
    private readonly read: (offset: number, length: number) => Promise<Uint8Array>,
  ) {}

  static async open(read: (offset: number, length: number) => Promise<Uint8Array>): Promise<VoiceFile> {
    const head = await read(0, 12);
    if (String.fromCharCode(...head.subarray(0, 8)) !== MAGIC) {
      throw KittenTTSError.invalidModelData('voices.bin is not a Kitten TTS 2 voice file.');
    }
    const headerLength = new DataView(head.buffer, head.byteOffset, 12).getUint32(8, true);
    const header = JSON.parse(decodeUTF8(await read(12, headerLength))) as { version: number; voices: VoiceEntry[] };
    if (header.version !== 1) {
      throw KittenTTSError.invalidModelData(`voices.bin has version ${header.version}; this SDK reads version 1.`);
    }
    return new VoiceFile(header.voices, 12 + headerLength, read);
  }

  async load(entry: VoiceEntry): Promise<VoiceData> {
    const spans = [entry.prefix, entry.speaker, entry.prompt_token, entry.prompt_feat, entry.embedding];
    const start = Math.min(...spans.map(([offset]) => offset));
    const end = Math.max(...spans.map(([offset, length]) => offset + length * 4));
    const bytes = await this.read(this.dataStart + start, end - start);
    // Copies, so every array starts its own buffer: ONNX Runtime's React Native
    // binding reads a tensor from the start of its buffer.
    const slice = ([offset, length]: [number, number]) =>
      bytes.buffer.slice(bytes.byteOffset + offset - start, bytes.byteOffset + offset - start + length * 4);
    return {
      prefix: Array.from(new Int32Array(slice(entry.prefix))),
      speaker: new Float32Array(slice(entry.speaker)),
      conditioning: {
        promptToken: new Int32Array(slice(entry.prompt_token)),
        promptFeat: new Float32Array(slice(entry.prompt_feat)),
        embedding: new Float32Array(slice(entry.embedding)),
      },
    };
  }
}
