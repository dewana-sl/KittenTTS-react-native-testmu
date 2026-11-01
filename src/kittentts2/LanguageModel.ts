import { KittenTTSError, KittenTTSErrorCode } from '../KittenTTSError';
import { type OrtModule, type OrtSession, type OrtSessionOptions, type OrtTensor, floats } from './ort';
import { type Random, type SamplerIds, type SamplingSettings, sampleToken } from './sampling';

export interface LanguageModelInfo {
  layers: number;
  kvHeads: number;
  headDim: number;
  headStart: number;
  speakerDim: number;
}

export interface LanguageModelTokens {
  speechEnd: number;
  stop: number;
  audioStart: number;
  audioEnd: number;
}

// The cache grows in blocks: reallocating per token would copy all of it every step.
const CACHE_BLOCK = 256;

/**
 * Kitten TTS 2's speech language model, one ONNX graph run per token.
 *
 * The graph returns only the new tokens' keys and values; the cache lives here
 * and goes back in as views over its own buffers. The React Native binding
 * passes a typed array by the start of its buffer, so every view starts at
 * offset 0, and row 0 of each buffer is the graph's unused row.
 */
export class LanguageModel {
  private keys: Float32Array[] = [];
  private values: Float32Array[] = [];
  private capacity = 0;
  private readonly rowSize: number;
  private readonly names: string[];
  private readonly ids: SamplerIds;

  private constructor(
    private readonly ort: OrtModule,
    private readonly session: OrtSession,
    private readonly info: LanguageModelInfo,
    private readonly tokens: LanguageModelTokens,
  ) {
    this.rowSize = info.kvHeads * info.headDim;
    this.names = ['logits'];
    for (let i = 0; i < info.layers; i += 1) this.names.push(`new_key_${i}`, `new_value_${i}`);
    this.ids = {
      headStart: info.headStart,
      audioStart: tokens.audioStart,
      audioEnd: tokens.audioEnd,
      exempt: [tokens.speechEnd, tokens.stop],
    };
  }

  static async load(
    ort: OrtModule,
    model: string | Uint8Array,
    info: LanguageModelInfo,
    tokens: LanguageModelTokens,
    options: OrtSessionOptions,
  ): Promise<LanguageModel> {
    return new LanguageModel(ort, await ort.InferenceSession.create(model, options), info, tokens);
  }

  /**
   * Run the model over `prompt` and sample until `<speech_end>`, `[STOP]` or
   * `maxNewTokens`. Returns the codec ids it wrote, 0-based.
   */
  async generate(
    prompt: readonly number[],
    speaker: Float32Array,
    settings: SamplingSettings,
    maxNewTokens: number,
    random: Random,
    signal?: { readonly aborted: boolean },
  ): Promise<number[]> {
    // The binding reads from the start of the buffer, so the embedding gets one of its own.
    const speakerData = Float32Array.from(speaker);
    const speakerTensor = new this.ort.Tensor('float32', speakerData, [1, this.info.speakerDim]);
    const history = [...prompt];
    const generated: number[] = [];
    try {
      if (signal?.aborted) throw cancelled();
      let logits = await this.step(prompt, speakerTensor, 1);
      let rows = 1 + prompt.length;
      while (generated.length < maxNewTokens) {
        const token = sampleToken(logits, history, prompt.length, settings, this.ids, random);
        if (token === this.tokens.speechEnd || token === this.tokens.stop) break;
        history.push(token);
        generated.push(token);
        if (generated.length === maxNewTokens) break;
        if (signal?.aborted) throw cancelled();
        logits = await this.step([token], speakerTensor, rows);
        rows += 1;
      }
    } finally {
      // Long prompts can leave hundreds of megabytes behind; nothing reuses them.
      this.releaseCache();
    }
    const { audioStart, audioEnd } = this.tokens;
    return generated.filter(id => id >= audioStart && id < audioEnd).map(id => id - audioStart);
  }

  async release(): Promise<void> {
    this.releaseCache();
    await this.session.release?.();
  }

  private async step(input: readonly number[], speaker: OrtTensor, rows: number): Promise<Float32Array> {
    this.ensureCapacity(rows + input.length);
    const { Tensor } = this.ort;
    const { kvHeads, headDim } = this.info;
    const feeds: Record<string, OrtTensor> = {
      input_ids: new Tensor('int64', BigInt64Array.from(input, id => BigInt(id)), [1, input.length]),
      speaker_embedding: speaker,
    };
    const length = rows * this.rowSize;
    for (let i = 0; i < this.info.layers; i += 1) {
      feeds[`past_key_${i}`] = new Tensor('float32', this.keys[i].subarray(0, length), [1, rows, kvHeads, headDim]);
      feeds[`past_value_${i}`] = new Tensor('float32', this.values[i].subarray(0, length), [1, rows, kvHeads, headDim]);
    }
    const outputs = await this.session.run(feeds, this.names);
    for (let i = 0; i < this.info.layers; i += 1) {
      this.keys[i].set(floats(outputs[`new_key_${i}`], 'keys'), length);
      this.values[i].set(floats(outputs[`new_value_${i}`], 'values'), length);
    }
    return floats(outputs.logits, 'logits');
  }

  private ensureCapacity(rows: number): void {
    if (rows <= this.capacity) return;
    const capacity = Math.ceil(rows / CACHE_BLOCK) * CACHE_BLOCK;
    const grow = (old: Float32Array | undefined) => {
      const next = new Float32Array(capacity * this.rowSize);
      if (old) next.set(old);
      return next;
    };
    this.keys = Array.from({ length: this.info.layers }, (_, i) => grow(this.keys[i]));
    this.values = Array.from({ length: this.info.layers }, (_, i) => grow(this.values[i]));
    this.capacity = capacity;
  }

  private releaseCache(): void {
    this.keys = [];
    this.values = [];
    this.capacity = 0;
  }
}

export function cancelled(): KittenTTSError {
  return new KittenTTSError(KittenTTSErrorCode.Cancelled, 'The request was cancelled.');
}
