/**
 * The parts of ONNX Runtime's JavaScript API the Kitten TTS 2 runtime uses,
 * so the same code runs on onnxruntime-react-native, onnxruntime-web and, in
 * tests, onnxruntime-node.
 */

export type OrtTensorData = Float32Array | BigInt64Array | Int32Array | Uint8Array;

export interface OrtTensor {
  readonly data: unknown;
  readonly dims: readonly number[];
}

export interface OrtSession {
  run(feeds: Record<string, OrtTensor>, fetches?: readonly string[]): Promise<Record<string, OrtTensor>>;
  release?(): Promise<void>;
}

export interface OrtSessionOptions {
  intraOpNumThreads?: number;
  graphOptimizationLevel?: 'disabled' | 'basic' | 'extended' | 'all';
  executionProviders?: readonly unknown[];
  [option: string]: unknown;
}

export interface OrtModule {
  InferenceSession: {
    create(model: string | Uint8Array, options?: OrtSessionOptions): Promise<OrtSession>;
  };
  Tensor: new (type: 'float32' | 'int64' | 'int32', data: OrtTensorData, dims: readonly number[]) => OrtTensor;
}

/** Float32 data of an output tensor. */
export function floats(tensor: OrtTensor | undefined, name: string): Float32Array {
  if (!tensor || !(tensor.data instanceof Float32Array)) {
    throw new Error(`ONNX Runtime returned no float32 ${name}.`);
  }
  return tensor.data;
}
