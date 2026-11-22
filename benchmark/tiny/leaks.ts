// Memory over repeated use, to catch leaks: the app's resident memory (RSS) and the
// JavaScript heap after each step. A leak grows with every cycle; memory that only
// waits for the garbage collector goes up and down.
//   - create, speak two languages, dispose: 8 times with Nano, 4 times with Mini and
//     Japanese (its 65 MB dictionary and an 83 MB model per cycle);
//   - one Nano instance: 60 generate() calls, then 5 long streams.
import { Platform } from 'react-native';
import * as RNFS from 'react-native-fs';
import DeviceInfo from 'react-native-device-info';
import { KittenTTS } from '@kittentts/react-native';

export interface MemorySample {
  phase: string;
  step: number;
  rssMB: number;
  jsHeapMB: number | null;
}

/** RSS from /proc on Android (the whole process, native included); task_info on iOS. */
async function memory(): Promise<{ rssMB: number; jsHeapMB: number | null }> {
  let rss = NaN;
  try {
    if (Platform.OS === 'android') {
      const status = await RNFS.readFile('/proc/self/status', 'utf8');
      rss = Number(/VmRSS:\s+(\d+)\s+kB/.exec(status)?.[1] ?? NaN) / 1024;
    } else {
      rss = (await DeviceInfo.getUsedMemory()) / 2 ** 20;
    }
  } catch {
    // Reported as null; the JavaScript heap is still measured.
  }
  const stats = (globalThis as any).HermesInternal?.getInstrumentedStats?.();
  return { rssMB: +rss.toFixed(1), jsHeapMB: stats ? +(stats.js_heapSize / 2 ** 20).toFixed(1) : null };
}

/** Least-squares growth per step, leaving out the first step (it warms caches up). */
function slope(ys: number[]): number {
  const pts = ys.slice(1);
  if (pts.length < 2) return 0;
  const n = pts.length;
  const mx = (n - 1) / 2;
  const my = pts.reduce((a, b) => a + b, 0) / n;
  let num = 0;
  let den = 0;
  pts.forEach((y, x) => { num += (x - mx) * (y - my); den += (x - mx) ** 2; });
  return +(num / den).toFixed(2);
}

export async function leakTest(
  texts: { en: string; de: string; ja: string; long: string },
  show: (status: string) => void,
): Promise<{ samples: MemorySample[]; summary: Record<string, unknown> }> {
  const samples: MemorySample[] = [];
  const sample = async (phase: string, step: number) => {
    samples.push({ phase, step, ...(await memory()) });
    show(`memory: ${phase} ${step}`);
  };
  await sample('start', 0);

  for (let c = 1; c <= 8; c++) {
    const tts = await KittenTTS.create({ model: 'kitten-tts-2-nano', weights: 'int8', languages: ['en', 'de'] });
    await tts.generate(texts.en);
    await tts.generate(texts.de, { language: 'de' });
    await tts.dispose();
    await sample('nano-cycle', c);
  }
  for (let c = 1; c <= 4; c++) {
    const tts = await KittenTTS.create({ model: 'kitten-tts-2-mini', weights: 'int8', languages: ['ja'] });
    await tts.generate(texts.ja, { language: 'ja' });
    await tts.dispose();
    await sample('mini-ja-cycle', c);
  }
  const tts = await KittenTTS.create({ model: 'kitten-tts-2-nano', weights: 'int8', languages: ['en'] });
  for (let i = 1; i <= 60; i++) {
    await tts.generate(texts.en);
    if (i % 10 === 0) await sample('nano-generate', i);
  }
  for (let i = 1; i <= 5; i++) {
    for await (const _ of tts.generateStreaming(texts.long)) { /* drain */ }
    await sample('nano-stream', i);
  }
  await tts.dispose();
  await sample('end', 0);

  const rss = (phase: string) => samples.filter(s => s.phase === phase).map(s => s.rssMB);
  const summary: Record<string, unknown> = { startMB: samples[0].rssMB, endMB: samples[samples.length - 1].rssMB };
  for (const phase of ['nano-cycle', 'mini-ja-cycle', 'nano-generate', 'nano-stream']) {
    const ys = rss(phase);
    summary[phase] = { firstMB: ys[0], lastMB: ys[ys.length - 1], maxMB: Math.max(...ys), growthPerStepMB: slope(ys) };
  }
  return { samples, summary };
}
