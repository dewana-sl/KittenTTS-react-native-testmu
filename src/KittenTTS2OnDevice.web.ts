import type { AudioPlayer } from './audio/AudioOutput.web';
import type { ProgressHandler } from './loader/ModelDownloader.web';
import {
  KittenTTS2OnDeviceBase,
  type KittenTTS2Repository,
} from './kittentts2/KittenTTS2OnDeviceBase';
import type { KittenTTS2Precision } from './kittentts2/bundle';
import { KittenTTSError } from './KittenTTSError';

/** Options for `KittenTTS.create({ model: 'KittenML/kitten-tts-2', ... })`. */
export interface KittenTTS2OnDeviceCreateOptions {
  model: KittenTTS2Repository;
  defaultVoice?: string;
  modelBaseURL?: string;
  storageDirectory?: string;
  forceRedownload?: boolean;
  downloadRetries?: number;
  ortNumThreads?: number;
  precision?: KittenTTS2Precision;
  player?: AudioPlayer;
}

/**
 * Kitten TTS 2 on the device runs on iOS and Android. Its language model alone
 * is 1.4 GB, more than a browser tab can reasonably load into WebAssembly, so in
 * a browser use Kitten TTS 2 through the Kitten API instead.
 */
export class KittenTTS2OnDevice extends KittenTTS2OnDeviceBase {
  static async create(
    _options: KittenTTS2OnDeviceCreateOptions,
    _onProgress?: ProgressHandler,
  ): Promise<KittenTTS2OnDevice> {
    throw KittenTTSError.invalidArgument(
      'Kitten TTS 2 runs on the device on iOS and Android only. In a browser, use it through the ' +
      "Kitten API: KittenTTS.create({ model: 'kitten-tts-2', baseUrl }).",
    );
  }
}
