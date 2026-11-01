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
import { MLXLanguageModel, isMLXAvailable } from './kittentts2/MLXLanguageModel';
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
   * Where the language model runs, as the Python package's `backend=`: `'mlx'`
   * on the GPU (iOS 17 and later, in an app built with the SDK's MLX module),
   * `'onnx'` on the CPU, or `'auto'` (the default), MLX when it is available.
   */
  backend?: KittenTTS2Backend;
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
    const backend = await chooseBackend(options.backend ?? 'auto');
    const root = options.storageDirectory ?? `${RNFS.DocumentDirectoryPath}/KittenTTS`;
    const directory = `${stripFileScheme(root)}/${repository.replace('/', '--')}`;
    const manifest = await downloadBundle({
      baseURL: (options.modelBaseURL ?? defaultBaseURL(repository)).replace(/\/+$/, ''),
      directory,
      force: options.forceRedownload ?? false,
      retries: Math.max(1, Math.floor(options.downloadRetries ?? 4)),
      backend,
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
      mlx: backend === 'mlx' ? tokens => loadMLX(directory, manifest, tokens) : undefined,
    });
    return new KittenTTS2OnDevice(state);
  }
}

async function chooseBackend(backend: KittenTTS2Backend): Promise<'mlx' | 'onnx'> {
  if (backend === 'onnx') return 'onnx';
  if (backend !== 'auto' && backend !== 'mlx') {
    throw KittenTTSError.invalidArgument("backend must be 'auto', 'mlx' or 'onnx'.");
  }
  if (await isMLXAvailable()) return 'mlx';
  if (backend === 'mlx') {
    throw KittenTTSError.invalidArgument(
      "backend 'mlx' needs an iPhone or iPad with iOS 17 or later (not the simulator) and an app built " +
      'with the SDK\'s MLX module: the Expo plugin\'s mlx option, or KITTENTTS_MLX=1 pod install.',
    );
  }
  return 'onnx';
}

function loadMLX(
  directory: string,
  manifest: BundleManifest,
  tokens: Parameters<typeof MLXLanguageModel.load>[2],
): Promise<MLXLanguageModel> {
  if (!manifest.mlx) {
    throw KittenTTSError.invalidModelData('This model folder has no MLX weights; pass backend: \'onnx\'.');
  }
  return MLXLanguageModel.load(`${directory}/${manifest.mlx.weights}`, {
    ...manifest.mlx.config,
    head_start: manifest.lm.head_start,
  }, tokens);
}

function stripFileScheme(path: string): string {
  return path.startsWith('file://') ? path.slice('file://'.length) : path;
}
