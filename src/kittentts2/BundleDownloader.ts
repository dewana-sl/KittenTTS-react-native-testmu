import * as RNFS from 'react-native-fs';
import { base64ToUint8Array } from '../audio/Base64';
import { KittenTTSError, errorMessage, isKittenTTSError } from '../KittenTTSError';
import type { DownloadProgressInfo, ProgressHandler } from '../loader/ModelDownloader';
import { type BundleManifest, checkDecoder, checkManifest, filesFor } from './bundle';
import type { KittenTTS2BundleReader } from './KittenTTS2OnDeviceBase';

const MANIFEST = 'manifest.json';
// What is on disk: each file's SHA-256 as the manifest gave it when it was downloaded.
const RECORD = 'downloaded.json';
const RETRY_DELAY_MS = 750;
const CONNECTION_TIMEOUT_MS = 30_000;
const READ_TIMEOUT_MS = 30_000;
const BACKGROUND_TIMEOUT_MS = 10 * 60_000;
const MANIFEST_TIMEOUT_MS = 15_000;

export interface BundleDownloadOptions {
  /** Where manifest.json and the files it lists are served from. */
  baseURL: string;
  /** Local folder for this model's files. */
  directory: string;
  force: boolean;
  retries: number;
  /** Which files to download: ONNX Runtime's, or the native engines'. Files marked for the other are skipped. */
  backend: 'native' | 'onnx';
  /** Which decoder's files to download. Files marked for another are skipped. */
  decoder: string;
}

/**
 * Download every file the manifest lists that is not already on disk, at its
 * listed size and hash, then keep the manifest beside them. Offline, a complete
 * earlier download is used as it is.
 */
export async function downloadBundle(
  options: BundleDownloadOptions,
  onProgress?: ProgressHandler,
): Promise<BundleManifest> {
  const { directory } = options;
  if (options.force) await RNFS.unlink(directory).catch(() => {});
  onProgress?.(0, { stage: 'checking-cache', asset: 'kitten-tts-2', cached: false });

  const manifest = await fetchManifest(options);
  const record = await readRecord(directory);
  checkDecoder(manifest, options.decoder);
  const files = filesFor(manifest, options.backend, options.decoder);
  const missing: Array<{ path: string; size: number; sha256?: string }> = [];
  for (const file of files) {
    const stat = await RNFS.stat(`${directory}/${file.path}`).catch(() => null);
    const changed = file.sha256 !== undefined && record[file.path] !== file.sha256;
    if (!stat || Number(stat.size) !== file.size || changed) missing.push(file);
  }
  if (missing.length === 0) {
    await RNFS.writeFile(`${directory}/${MANIFEST}`, JSON.stringify(manifest), 'utf8');
    onProgress?.(1, { stage: 'cached', asset: 'kitten-tts-2', cached: true });
    return manifest;
  }

  const total = files.reduce((sum, file) => sum + file.size, 0);
  let done = total - missing.reduce((sum, file) => sum + file.size, 0);
  for (const file of missing) {
    const target = `${directory}/${file.path}`;
    await RNFS.mkdir(target.slice(0, target.lastIndexOf('/')));
    await downloadWithRetries(`${options.baseURL}/${file.path}`, target, file, options.retries, (written, info) => {
      onProgress?.(Math.min(1, (done + written) / total), { ...info, asset: 'kitten-tts-2', file: file.path });
    });
    done += file.size;
    // Recorded file by file, so an interrupted download resumes where it stopped.
    if (file.sha256 !== undefined) {
      record[file.path] = file.sha256;
      await RNFS.writeFile(`${directory}/${RECORD}`, JSON.stringify(record), 'utf8');
    }
  }
  await RNFS.writeFile(`${directory}/${MANIFEST}`, JSON.stringify(manifest), 'utf8');
  onProgress?.(1, { stage: 'complete', asset: 'kitten-tts-2', cached: false });
  return manifest;
}

async function readRecord(directory: string): Promise<Record<string, string>> {
  try {
    const record = JSON.parse(await RNFS.readFile(`${directory}/${RECORD}`, 'utf8'));
    return record && typeof record === 'object' ? record as Record<string, string> : {};
  } catch {
    return {};
  }
}

async function fetchManifest(options: BundleDownloadOptions): Promise<BundleManifest> {
  const url = `${options.baseURL}/${MANIFEST}`;
  const cachedPath = `${options.directory}/${MANIFEST}`;
  let fetchError: unknown;
  try {
    const response = await fetchWithTimeout(url, MANIFEST_TIMEOUT_MS);
    if (response.ok) {
      const manifest = checkManifest(await response.json(), url);
      await RNFS.mkdir(options.directory);
      return manifest;
    }
    fetchError = new Error(`HTTP ${response.status}`);
  } catch (error) {
    if (isKittenTTSError(error)) throw error;
    fetchError = error;
  }
  if (await RNFS.exists(cachedPath)) {
    return checkManifest(JSON.parse(await RNFS.readFile(cachedPath, 'utf8')), cachedPath);
  }
  throw KittenTTSError.downloadFailed(
    `Could not fetch the Kitten TTS 2 manifest from ${url}: ${errorMessage(fetchError)}. ` +
    'Pass modelBaseURL if the ONNX files are hosted somewhere else.',
    fetchError,
  );
}

async function fetchWithTimeout(url: string, timeoutMs: number): Promise<Response> {
  const controller = typeof AbortController === 'function' ? new AbortController() : undefined;
  const timer = setTimeout(() => controller?.abort(), timeoutMs);
  try {
    return await fetch(url, { signal: controller?.signal });
  } finally {
    clearTimeout(timer);
  }
}

async function downloadWithRetries(
  url: string,
  target: string,
  file: { path: string; size: number },
  retries: number,
  onProgress: (written: number, info: DownloadProgressInfo) => void,
): Promise<void> {
  let lastError: unknown;
  for (let attempt = 1; attempt <= retries; attempt += 1) {
    const temp = `${target}.download`;
    try {
      await RNFS.unlink(temp).catch(() => {});
      onProgress(0, { stage: 'downloading', attempt, totalAttempts: retries });
      const response = await RNFS.downloadFile({
        fromUrl: url,
        toFile: temp,
        progressDivider: 1,
        connectionTimeout: CONNECTION_TIMEOUT_MS,
        readTimeout: READ_TIMEOUT_MS,
        backgroundTimeout: BACKGROUND_TIMEOUT_MS,
        progress: event => onProgress(event.bytesWritten, {
          stage: 'downloading',
          attempt,
          totalAttempts: retries,
          bytesWritten: event.bytesWritten,
          contentLength: event.contentLength,
        }),
      }).promise;
      if (response.statusCode !== 200) throw new Error(`HTTP ${response.statusCode} downloading ${url}`);
      const stat = await RNFS.stat(temp);
      if (Number(stat.size) !== file.size) {
        throw new Error(`${file.path} is ${stat.size} bytes; the manifest says ${file.size}.`);
      }
      await RNFS.unlink(target).catch(() => {});
      await RNFS.moveFile(temp, target);
      return;
    } catch (error) {
      lastError = error;
      await RNFS.unlink(temp).catch(() => {});
      if (attempt < retries) {
        onProgress(0, { stage: 'retrying', attempt: attempt + 1, totalAttempts: retries, message: errorMessage(error) });
        await new Promise(resolve => setTimeout(resolve, RETRY_DELAY_MS * attempt));
      }
    }
  }
  throw KittenTTSError.downloadFailed(`Failed after ${retries} attempts: ${errorMessage(lastError)}`, lastError);
}

/** Reads a downloaded model folder with react-native-fs. */
export function fileReader(directory: string): KittenTTS2BundleReader {
  return {
    readText: path => RNFS.readFile(`${directory}/${path}`, 'utf8'),
    readBytes: async (path, offset, length) => {
      const file = `${directory}/${path}`;
      if (length === undefined) return base64ToUint8Array(await RNFS.readFile(file, 'base64')).subarray(offset ?? 0);
      return base64ToUint8Array(await RNFS.read(file, length, offset ?? 0, 'base64'));
    },
    graph: async path => `${directory}/${path}`,
  };
}
