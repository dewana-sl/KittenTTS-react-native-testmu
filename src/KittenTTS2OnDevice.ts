import * as RNFS from 'react-native-fs';
import { AudioOutput, type AudioPlayer } from './audio/AudioOutput';
import type { ProgressHandler } from './loader/ModelDownloader';
import { downloadBundle, fileReader } from './kittentts2/BundleDownloader';
import {
  KittenTTS2OnDeviceBase,
  type KittenTTS2Repository,
  checkRepository,
  defaultBaseURL,
} from './kittentts2/KittenTTS2OnDeviceBase';
import type { BundleManifest, KittenTTS2Backend, KittenTTS2Precision } from './kittentts2/bundle';
import {
  type NativeBackend,
  NativeAudioDecoder,
  NativeLanguageModel,
  isNativeAvailable,
  nativeDevice,
} from './kittentts2/NativeEngine';
import type { OrtModule } from './kittentts2/ort';
import { KittenTTSError } from './KittenTTSError';

/** Options for `KittenTTS.create({ model: 'KittenML/kitten-tts-2', ... })`. */
export interface KittenTTS2OnDeviceCreateOptions {
  /** The model repository, as the Python package takes it: `'KittenML/kitten-tts-2'`. */
  model: KittenTTS2Repository;
  /** Voice used when a call names none. Defaults to the model's default voice. */
  defaultVoice?: string;
  /**
   * Where the ONNX files are served from: a folder holding manifest.json and the
   * files it lists. Defaults to the repository's `onnx/` folder on Hugging Face.
   */
  modelBaseURL?: string;
  /** Root folder for downloaded models. Files live under `<storageDirectory>/<owner>--<name>/`. */
  storageDirectory?: string;
  /** Delete the downloaded files and download them again. */
  forceRedownload?: boolean;
  /** Download attempts per file. Defaults to 4. */
  downloadRetries?: number;
  /** ONNX Runtime threads. Defaults to 4. */
  ortNumThreads?: number;
  /**
   * Where the models run, as the Python package's `backend=`: `'mlx'` on the GPU
   * (iOS 17 and later, in an app built with the SDK's MLX module), `'ggml'` on
   * the GPU through OpenCL on Adreno 7xx and newer or Vulkan elsewhere (Android,
   * in an app built with the SDK's ggml engine; on the CPU when the phone has no
   * GPU it is faster on, such as Mali, Xclipse and older Adreno GPUs), `'onnx'`
   * on the CPU, or
   * `'auto'` (the default): the native engine when it runs on the GPU, else ONNX.
   */
  backend?: KittenTTS2Backend;
  /**
   * Which decoder turns codec tokens into a mel spectrogram, as the Python
   * package's `decoder=`: `'default'`, s3gen's flow in two steps, or
   * `'student_w4'`, its distilled student in one step, with a smaller download
   * and lower quality. The student runs on MLX only so far.
   */
  decoder?: string;
  /**
   * How the language model multiplies on ONNX Runtime: `'int8'` activations,
   * about 2.5 times faster, or `'float32'`. Defaults to the model folder's choice, `'int8'`.
   */
  precision?: KittenTTS2Precision;
  /**
   * Audio player for `speak()` and `play()`. Use {@link createExpoAudioPlayer}
   * or {@link createRNSoundPlayer}, or provide your own implementation.
   */
  player?: AudioPlayer;
}

/**
 * Kitten TTS 2 on the device: about 2 GB of model files, downloaded on first
 * use, and nothing sent anywhere afterwards. It has the Python package's calls:
 * built-in voices, expression tags and the decoding presets.
 *
 * @example
 * ```typescript
 * const tts = await KittenTTS.create({ model: 'KittenML/kitten-tts-2' }, progress => {
 *   console.log(`${Math.round(progress * 100)}%`);
 * });
 * const result = await tts.generate('[joyful] It runs on the phone <laugh>!', 'Bruno');
 * ```
 */
export class KittenTTS2OnDevice extends KittenTTS2OnDeviceBase {
  /** Download the model if needed and load it. */
  static async create(
    options: KittenTTS2OnDeviceCreateOptions,
    onProgress?: ProgressHandler,
  ): Promise<KittenTTS2OnDevice> {
    const repository = checkRepository(options?.model);
    const engine = await chooseBackend(options.backend ?? 'auto');
    const decoder = options.decoder ?? 'default';
    if (decoder !== 'default' && engine.backend !== 'mlx') {
      throw KittenTTSError.invalidArgument(`decoder '${decoder}' runs on MLX only so far; this device uses ${engine.backend}.`);
    }
    const root = options.storageDirectory ?? `${RNFS.DocumentDirectoryPath}/KittenTTS`;
    const directory = `${stripFileScheme(root)}/${repository.replace('/', '--')}`;
    const manifest = await downloadBundle({
      baseURL: (options.modelBaseURL ?? defaultBaseURL(repository)).replace(/\/+$/, ''),
      directory,
      force: options.forceRedownload ?? false,
      retries: Math.max(1, Math.floor(options.downloadRetries ?? 4)),
      backend: engine.backend === 'onnx' ? 'onnx' : 'native',
      decoder,
    }, onProgress);
    const ort = (await import('onnxruntime-react-native')) as unknown as OrtModule;
    const state = await KittenTTS2OnDeviceBase.open({
      repository,
      manifest,
      reader: fileReader(directory),
      ort,
      sessionOptions: {
        graphOptimizationLevel: 'all',
        intraOpNumThreads: Math.max(1, options.ortNumThreads ?? 4),
      },
      audioOutput: new AudioOutput(options.player),
      defaultVoice: options.defaultVoice,
      precision: options.precision,
      decoder,
      native: engine.backend === 'onnx' ? undefined : {
        backend: engine.backend,
        device: engine.device,
        precision: engine.backend === 'mlx' ? 'bfloat16' : engine.device === 'gpu' ? 'float32' : 'int8',
        load: tokens => loadNative(engine.backend as NativeBackend, directory, manifest, decoder, tokens),
      },
    });
    const tts = new KittenTTS2OnDevice(state);
    // The first call compiles the GPU kernels; do it now rather than in the app's first request.
    if (engine.backend !== 'onnx') await tts.warmUp();
    return tts;
  }
}

interface Engine {
  backend: 'mlx' | 'ggml' | 'onnx';
  device: 'gpu' | 'cpu';
}

async function chooseBackend(backend: KittenTTS2Backend): Promise<Engine> {
  if (backend === 'onnx') return { backend: 'onnx', device: 'cpu' };
  if (backend !== 'auto' && backend !== 'mlx' && backend !== 'ggml') {
    throw KittenTTSError.invalidArgument("backend must be 'auto', 'mlx', 'ggml' or 'onnx'.");
  }
  if (backend !== 'ggml' && await isNativeAvailable('mlx')) return { backend: 'mlx', device: 'gpu' };
  if (backend !== 'mlx' && await isNativeAvailable('ggml')) {
    const device = await nativeDevice('ggml');
    if (device === 'gpu' || backend === 'ggml') return { backend: 'ggml', device };
  }
  if (backend === 'mlx') {
    throw KittenTTSError.invalidArgument(
      "backend 'mlx' needs an iPhone or iPad with iOS 17 or later (not the simulator) and an app built " +
      'with the SDK\'s MLX module: the Expo plugin\'s mlx option, or KITTENTTS_MLX=1 pod install.',
    );
  }
  if (backend === 'ggml') {
    throw KittenTTSError.invalidArgument(
      "backend 'ggml' needs Android 9 or later on arm64 (2018 phones and newer) or x86_64, and an app built " +
      'with the SDK\'s ggml engine: the Expo plugin\'s ggml option, or kittentts.ggml=true in android/gradle.properties.',
    );
  }
  return { backend: 'onnx', device: 'cpu' };
}

/** Both models in the native engine: the language model, then the audio decoder. */
async function loadNative(
  backend: NativeBackend,
  directory: string,
  manifest: BundleManifest,
  decoderName: string,
  tokens: Parameters<typeof NativeLanguageModel.load>[3],
): Promise<{ languageModel: NativeLanguageModel; decoder: NativeAudioDecoder }> {
  const files = manifest.native;
  if (!files) {
    throw KittenTTSError.invalidModelData(`This model folder has no ${backend} weights; pass backend: 'onnx'.`);
  }
  const languageModel = await NativeLanguageModel.load(backend, `${directory}/${files.weights}`, {
    ...files.config,
    head_start: manifest.lm.head_start,
  }, tokens);
  const weights = manifest.decoders?.[decoderName]?.weights ?? files.decoder;
  const decoder = await NativeAudioDecoder.load(backend, `${directory}/${weights}`, manifest.decoder.silence_token);
  return { languageModel, decoder };
}

function stripFileScheme(path: string): string {
  return path.startsWith('file://') ? path.slice('file://'.length) : path;
}
