import { type OrtModule, type OrtSession, type OrtSessionOptions, floats } from './ort';
import type { Random } from './sampling';

export interface AudioDecoderInfo {
  cfmSteps: number;
  silenceToken: number;
  tokenMelRatio: number;
  samplesPerFrame: number;
  mels: number;
}

/** What the decoder needs from a voice's reference clip, precomputed per voice. */
export interface DecoderConditioning {
  promptToken: Int32Array;
  /** Mel frames, `promptToken.length * 2` rows of 80. */
  promptFeat: Float32Array;
  /** The reference's 192-dim x-vector. */
  embedding: Float32Array;
}

/** What generation needs from an audio decoder backend: ONNX Runtime, or MLX on iOS. */
export interface AudioDecoderLike {
  /** Returns `null` when no valid codec token is left, as the Python package does. */
  decode(codes: readonly number[], conditioning: DecoderConditioning, random: Random): Promise<Float32Array | null>;
  release(): Promise<void>;
}

const CODEC_TOKENS = 6561;
// s3gen drops its last few tokens as lookahead; silence pads them so the final syllable stays.
const LOOKAHEAD_PAD = 3;
const SAMPLE_RATE = 24_000;

/** The decoder's input: the valid codec ids, then silence for the lookahead. `null` when none is valid. */
export function decoderTokens(codes: readonly number[], silenceToken: number): number[] | null {
  const valid = codes.filter(code => code >= 0 && code < CODEC_TOKENS);
  if (valid.length === 0) return null;
  return [...valid, ...new Array<number>(LOOKAHEAD_PAD).fill(silenceToken)];
}

/**
 * s3gen silences the first 20 ms and fades in over the next 20 ms, against
 * spillover from the reference clip. Applied in place.
 */
export function applyTrimFade(wav: Float32Array): Float32Array {
  const trim = SAMPLE_RATE / 50;
  const end = Math.min(2 * trim, wav.length);
  for (let i = 0; i < end; i += 1) {
    wav[i] *= i < trim ? 0 : (Math.cos(Math.PI - (Math.PI * (i - trim)) / (trim - 1)) + 1) / 2;
  }
  return wav;
}

/**
 * Codec tokens to a 24 kHz waveform: the s3gen flow (tokens to mel, in two
 * meanflow steps) and the HiFT vocoder (mel to audio), as `S3Codec.decode` runs them.
 */
export class AudioDecoder implements AudioDecoderLike {
  private constructor(
    private readonly ort: OrtModule,
    private readonly flow: OrtSession,
    private readonly hift: OrtSession,
    private readonly info: AudioDecoderInfo,
  ) {}

  static async load(
    ort: OrtModule,
    flow: { model: string | Uint8Array; options: OrtSessionOptions },
    hift: { model: string | Uint8Array; options: OrtSessionOptions },
    info: AudioDecoderInfo,
  ): Promise<AudioDecoder> {
    const flowSession = await ort.InferenceSession.create(flow.model, flow.options);
    const hiftSession = await ort.InferenceSession.create(hift.model, hift.options);
    return new AudioDecoder(ort, flowSession, hiftSession, info);
  }

  async decode(codes: readonly number[], conditioning: DecoderConditioning, random: Random): Promise<Float32Array | null> {
    const tokens = decoderTokens(codes, this.info.silenceToken);
    if (!tokens) return null;
    const { Tensor } = this.ort;
    const { mels, tokenMelRatio } = this.info;
    const prompt = conditioning.promptToken.length;
    const frames = conditioning.promptFeat.length / mels;
    const noise = random.fillNormal(new Float32Array(mels * tokenMelRatio * (prompt + tokens.length)));
    const flowOut = await this.flow.run({
      speech_tokens: new Tensor('int64', BigInt64Array.from(tokens, id => BigInt(id)), [1, tokens.length]),
      prompt_token: new Tensor('int64', BigInt64Array.from(conditioning.promptToken, id => BigInt(id)), [1, prompt]),
      prompt_feat: new Tensor('float32', conditioning.promptFeat, [1, frames, mels]),
      embedding: new Tensor('float32', conditioning.embedding, [1, conditioning.embedding.length]),
      noise: new Tensor('float32', noise, [1, mels, noise.length / mels]),
    }, ['mel']);
    const mel = floats(flowOut.mel, 'mel');
    const hiftOut = await this.hift.run({
      mel: new Tensor('float32', mel, [1, mels, mel.length / mels]),
    }, ['wav']);
    return applyTrimFade(Float32Array.from(floats(hiftOut.wav, 'wav')));
  }

  async release(): Promise<void> {
    await this.flow.release?.();
    await this.hift.release?.();
  }
}
