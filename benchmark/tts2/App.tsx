import { useRef, useState } from 'react';
import { NativeModules, Platform, Pressable, ScrollView, Text, TextInput } from 'react-native';
import { activateKeepAwakeAsync } from 'expo-keep-awake';
import * as RNFS from 'react-native-fs';
import { KittenTTS } from '@kittentts/react-native';

// Kitten TTS 2 on TestMu's real devices. The workflow serves the model files from
// the GitHub runner at a temporary address, which Appium types in before it taps
// Start. The app then runs every check and shows its report as JSON, which Appium
// reads off the screen.
const STORAGE = `${RNFS.DocumentDirectoryPath}/KittenTTS`;
const FILES = `${STORAGE}/KittenML--kitten-tts-2`;
const IOS = Platform.OS === 'ios';
const BACKEND = IOS ? 'mlx' : 'ggml';
// On iOS, s3gen's flow in two steps, the same flow held to one step, and the Python
// package's one-step student. The ggml engine on Android has only the first.
const DECODERS = IOS ? ['default', 'stock_1step', 'student_w4'] : ['default'];
const WEIGHTS: Record<string, string> = {
  default: 'decoder-mlx.safetensors',
  stock_1step: 'decoder-stock_1step-mlx.safetensors',
  student_w4: 'decoder-student_w4-mlx.safetensors',
};

// The same sentences, voices and seeds as the Mac and iPhone 17 comparisons.
const JOBS: Array<[string, { voice: string; preset?: string; advanced: { seed: number } }]> = [
  ['The quick brown fox jumps over the lazy dog near the river bank.', { voice: 'Bruno', advanced: { seed: 11 } }],
  ['Please remember to bring your umbrella, because it will rain this afternoon.', { voice: 'Luna', advanced: { seed: 12 } }],
  ['Our quarterly revenue grew by fifteen percent, beating every forecast we made.', { voice: 'Eleanor', advanced: { seed: 13 } }],
  ["I can't believe you finished the whole marathon in under four hours!", { voice: 'Jasper', preset: 'expressive', advanced: { seed: 14 } }],
  ['The museum opens at nine in the morning and closes at six in the evening.', { voice: 'Iris', advanced: { seed: 15 } }],
  ['She whispered the secret softly, so that nobody else in the room could hear.', { voice: 'Maeve', advanced: { seed: 16 } }],
  ['Turn left at the second traffic light, then continue straight for two miles.', { voice: 'Frank', advanced: { seed: 17 } }],
  ['Every morning he walked the dog, read the newspaper, and made a cup of tea.', { voice: 'Victor', advanced: { seed: 18 } }],
];
// About 19 seconds of speech, one chunk: the decoder-only check's long input.
const LONG = 'The first part of this text is long enough that the model splits it into chunks, '
  + 'so that each chunk stays inside the range the model handles well and can be played on its own. '
  + 'Here is a second sentence that keeps the first chunk going for a while longer, with a few more words.';
const STREAMED = `${LONG} And then the second chunk begins here, with its own sentences, which the app can start `
  + 'playing as soon as it arrives, while the next one is still being generated on the device.';
const ROUNDS = 5;
// When this JavaScript started: with the native process id below, shows whether a restart was a new process.
const JS_STARTED = new Date().toISOString();

type Step = { name: string; ok: boolean; seconds: number; detail?: unknown; error?: string };

const round = (value: number, digits = 2) => Number(value.toFixed(digits));
const median = (values: number[]) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
const message = (error: any) => `${error?.code ?? ''} ${error?.message ?? error}`.trim();

/**
 * Whether the audio is speech-like rather than silence, noise or NaN. Speech at
 * 24 kHz crosses zero on well under a quarter of its samples; noise on about half.
 */
function audioCheck(samples: Float32Array) {
  let sum = 0;
  let peak = 0;
  let finite = true;
  let crossings = 0;
  for (let i = 0; i < samples.length; i += 1) {
    const s = samples[i];
    if (!Number.isFinite(s)) finite = false;
    sum += s * s;
    peak = Math.max(peak, Math.abs(s));
    if (i > 0 && (s >= 0) !== (samples[i - 1] >= 0)) crossings += 1;
  }
  const rms = Math.sqrt(sum / Math.max(1, samples.length));
  const zcr = crossings / Math.max(1, samples.length - 1);
  return { rms: round(rms, 4), peak: round(peak, 3), zcr: round(zcr, 3), ok: finite && rms > 0.005 && peak < 1.5 && zcr < 0.25 };
}

/**
 * The whole benchmark, or with `gpuCheck` only a load and one sentence: the
 * runner restarts the app with different ggml-vulkan settings and runs that.
 * `backend` 'auto' checks what the SDK picks on its own.
 */
async function run(base: string, publish: (report: object) => void, gpuCheck = false, backend = BACKEND) {
  const constants = Platform.constants as Record<string, unknown>;
  const report = {
    app: 'kitten-tts-2-benchmark',
    platform: Platform.OS,
    os: String(Platform.Version),
    model: String(constants.Model ?? constants.interfaceIdiom ?? ''),
    backend: '' as string,
    device: '' as string,
    /** What the native engine says it runs on, and why it is not the GPU. */
    gpu: '' as string,
    /** Android: "<process id> <settings it started with>" (scripts/gpu-env.py), and when the JavaScript started. */
    process: { native: '', js: JS_STARTED },
    status: 'starting',
    done: false,
    steps: [] as Step[],
  };
  if (!IOS) report.process.native = await RNFS.readFile(`${RNFS.DocumentDirectoryPath}/ggml_env.txt`, 'utf8').catch(() => '');
  let lastPublish = 0;
  const show = (force = false) => {
    if (!force && Date.now() - lastPublish < 1000) return;
    lastPublish = Date.now();
    publish({ ...report, steps: [...report.steps] });
  };
  const step = async <T,>(name: string, fn: () => Promise<T>): Promise<T | undefined> => {
    report.status = name;
    show(true);
    const started = Date.now();
    try {
      const detail = await fn();
      report.steps.push({ name, ok: true, seconds: round((Date.now() - started) / 1000), detail });
      return detail;
    } catch (error) {
      report.steps.push({ name, ok: false, seconds: round((Date.now() - started) / 1000), error: message(error) });
      return undefined;
    } finally {
      show(true);
    }
  };

  const downloaded = new Set<string>();
  const open = async (decoder: string) => {
    const started = Date.now();
    let downloadEnded = 0;
    let downloading = false;
    const tts: any = await KittenTTS.create({
      model: 'KittenML/kitten-tts-2', modelBaseURL: base, storageDirectory: STORAGE, backend, decoder,
    } as any, (fraction: number, info: any) => {
      if (info?.stage === 'downloading') downloading = true;
      if (info?.stage === 'complete' || info?.stage === 'cached') {
        downloadEnded = Date.now();
        // Shown at once, so a crash while the models load is not reported as one in the download.
        report.status = `${decoder}: loading the models`;
        show(true);
        return;
      }
      report.status = `${decoder}: ${info?.stage ?? ''} ${Math.round(fraction * 100)}% ${info?.file ?? ''}`;
      show();
    });
    report.backend = tts.backend;
    report.device = tts.device;
    const ggml = (NativeModules as any).KittenTTS2GGML;
    if (!IOS && ggml?.device) report.gpu = await ggml.device().catch((error: unknown) => message(error));
    const files = (tts.state.manifest.files as Array<{ path: string; size: number; backend?: string; decoder?: string }>)
      .filter(f => (!f.backend || f.backend === (tts.backend === 'onnx' ? 'onnx' : 'native')) && (!f.decoder || f.decoder === decoder))
      .filter(f => !downloaded.has(f.path));
    files.forEach(f => downloaded.add(f.path));
    const megabytes = files.reduce((sum, f) => sum + f.size, 0) / 1e6;
    const downloadSeconds = downloading ? (downloadEnded - started) / 1000 : 0;
    return {
      tts,
      detail: {
        decoder: tts.decoder,
        backend: tts.backend,
        device: tts.device,
        downloadMB: round(megabytes, 0),
        downloadSeconds: round(downloadSeconds, 1),
        downloadMBps: downloadSeconds > 0 ? round(megabytes / downloadSeconds, 1) : null,
        loadSeconds: round((Date.now() - (downloadEnded || started)) / 1000, 1),
      },
    };
  };

  // The audio decoder's share of each generation.
  let decodeMs = 0;
  const timeDecoder = (tts: any) => {
    const inner = tts.state.decoder;
    const decode = inner.decode.bind(inner);
    inner.decode = async (...args: unknown[]) => {
      const started = Date.now();
      const out = await decode(...args);
      decodeMs += Date.now() - started;
      return out;
    };
    return decode;
  };
  const sentences = async (tts: any, jobs = JOBS) => {
    let audio = 0;
    let took = 0;
    const each: number[] = [];
    const checks: Array<ReturnType<typeof audioCheck>> = [];
    decodeMs = 0;
    for (const [text, options] of jobs) {
      const started = Date.now();
      const result = await tts.generate(text, options);
      const seconds = (Date.now() - started) / 1000;
      audio += result.duration;
      took += seconds;
      each.push(round(seconds / result.duration));
      checks.push(audioCheck(result.samples));
    }
    return {
      audioSeconds: round(audio),
      rtf: round(took / audio),
      decoderRtf: round(decodeMs / 1000 / audio, 3),
      each,
      audioOk: checks.every(c => c.ok),
      rms: checks.map(c => c.rms),
      zcr: checks.map(c => c.zcr),
    };
  };

  const codes: Record<string, number[]> = {};
  const conditioning: Record<string, unknown> = {};
  let decodeRaw: ((...args: any[]) => Promise<Float32Array | null>) | undefined;
  let tts: any;

  if (gpuCheck) {
    const opened = await step('load default', async () => {
      const result = await open('default');
      tts = result.tts;
      return result.detail;
    });
    if (opened) {
      timeDecoder(tts);
      await step('one sentence', () => sentences(tts, JOBS.slice(0, 1)));
    }
    await tts?.dispose().catch(() => {});
    // KITTENTTS2_CHECK=$FILES/ggml_check.txt: the engine's node-by-node check of the GPU against the CPU.
    const checkFile = `${RNFS.DocumentDirectoryPath}/ggml_check.txt`;
    if (!IOS && report.process.native.includes('KITTENTTS2_CHECK') && (await RNFS.exists(checkFile).catch(() => false))) {
      const read = (path: string) => RNFS.readFile(path, 'utf8').catch((error: unknown) => message(error));
      (report as any).check = { findings: (await read(checkFile)).slice(0, 30000), ops: (await read(`${checkFile}.ops`)).slice(0, 8000) };
    }
    report.status = 'done';
    report.done = true;
    show(true);
    return;
  }

  for (const decoder of DECODERS) {
    const opened = await step(`load ${decoder}`, async () => {
      const result = await open(decoder);
      tts = result.tts;
      return result.detail;
    });
    if (!opened) break;
    decodeRaw = timeDecoder(tts);
    const voice = tts.state.voiceFile.entries.find((entry: any) => entry.name === 'Bruno');
    conditioning[decoder] = (await tts.state.voiceFile.load(voice)).conditioning;
    if (decoder === 'default') {
      await step('codec tokens for the decoder check', async () => {
        const inner = tts.state.decoder;
        const timed = inner.decode;
        inner.decode = async (c: number[], ...rest: unknown[]) => { codes.current = [...c]; return timed(c, ...rest); };
        await tts.generate(JOBS[0][0], JOBS[0][1]);
        codes.short = codes.current;
        await tts.generate(LONG, { voice: 'Bruno', advanced: { seed: 3 } });
        codes.long = codes.current;
        inner.decode = timed;
        return { short: codes.short.length, long: codes.long.length };
      });
    }
    await step(`8 sentences ${decoder}`, () => sentences(tts));
    if (decoder === 'default') {
      await step('streaming, first chunk', async () => {
        const started = Date.now();
        const chunks: Array<{ seconds: number; atSeconds: number }> = [];
        for await (const chunk of tts.generateStreaming(STREAMED, { voice: 'Eleanor', advanced: { seed: 3 } })) {
          chunks.push({ seconds: round(chunk.duration), atSeconds: round((Date.now() - started) / 1000) });
        }
        return { chunks };
      });
      await step('cancel', async () => {
        const controller = new AbortController();
        setTimeout(() => controller.abort(), 700);
        const started = Date.now();
        try {
          await tts.generate('This sentence is cancelled before it is finished speaking, on purpose, after a moment.', { signal: controller.signal });
          return { cancelled: false };
        } catch (error: any) {
          if (error?.code !== 'CANCELLED') throw error;
          return { cancelled: true, afterMs: Date.now() - started };
        }
      });
    }
    if (decoder !== DECODERS[DECODERS.length - 1]) {
      await tts.dispose();
      tts = undefined;
    }
  }

  // Only the decoder, on the same codec tokens, the decoders taking turns.
  if (tts && decodeRaw && codes.short && codes.long) {
    const decode = decodeRaw;
    await step('decoder only', async () => {
      // The native decoders take a seed from next(); the ONNX one takes its noise from fillNormal().
      let state = 3;
      const uniform = () => ((state = (Math.imul(state, 1664525) + 1013904223) >>> 0) + 1) / 0x100000001;
      const random = {
        next: () => 0.25,
        fillNormal: (out: Float32Array) => {
          for (let i = 0; i < out.length; i += 1) out[i] = Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform());
          return out;
        },
      };
      const timings: Record<string, { short: number[]; long: number[] }> = {};
      let seconds = { short: 0, long: 0 };
      for (let r = 0; r < ROUNDS; r += 1) {
        for (const decoder of DECODERS) {
          if (!conditioning[decoder]) continue;
          if (IOS) await NativeModules.KittenTTS2MLX.loadDecoder(`${FILES}/${WEIGHTS[decoder]}`);
          for (const length of ['short', 'long'] as const) {
            const started = Date.now();
            const samples = await decode(codes[length], conditioning[decoder], random);
            ((timings[decoder] ??= { short: [], long: [] })[length]).push(Date.now() - started);
            seconds[length] = round((samples?.length ?? 0) / 24000);
          }
        }
        report.status = `decoder only, round ${r + 1}`;
        show(true);
      }
      return {
        seconds,
        median: Object.fromEntries(Object.entries(timings).map(([decoder, t]) => [decoder, {
          shortMs: median(t.short.slice(1)),
          longMs: median(t.long.slice(1)),
          shortRtf: round(median(t.short.slice(1)) / 1000 / seconds.short, 3),
          longRtf: round(median(t.long.slice(1)) / 1000 / seconds.long, 3),
        }])),
        timings,
      };
    });
  }
  await tts?.dispose().catch(() => {});
  report.status = 'done';
  report.done = true;
  show(true);
}

// Android finds elements by content description, iOS by test ID.
const id = (name: string) => (Platform.OS === 'android' ? { testID: name, accessibilityLabel: name } : { testID: name });

export default function App() {
  const [text, setText] = useState('{"status":"waiting for the model files address"}');
  const [base, setBase] = useState('');
  const started = useRef(false);
  const start = () => {
    if (started.current || !base.trim()) return;
    started.current = true;
    // "<address>#gpu-check" runs only the short GPU check; "<address>#auto" lets the SDK pick the engine;
    // "x#set-env#NAME=value,..." saves settings for the next start of the app (scripts/gpu-env.py).
    const [address, mode, env = ''] = base.trim().split('#');
    if (mode === 'set-env') {
      RNFS.writeFile(`${RNFS.DocumentDirectoryPath}/ggml_env_next.txt`, env, 'utf8')
        .then(() => setText(JSON.stringify({ status: 'settings saved', done: true, env })))
        .catch(error => setText(JSON.stringify({ done: true, error: message(error) })));
      return;
    }
    activateKeepAwakeAsync().catch(() => {}).finally(() => {
      run(address.replace(/\/+$/, ''), report => setText(JSON.stringify(report)), mode === 'gpu-check', mode === 'auto' ? 'auto' : BACKEND)
        .catch(error => setText(JSON.stringify({ done: true, error: message(error) })));
    });
  };
  return (
    <ScrollView contentContainerStyle={{ padding: 12, paddingTop: 56 }} keyboardShouldPersistTaps="handled">
      <Text style={{ fontSize: 16, fontWeight: '600', marginBottom: 8 }}>Kitten TTS 2 benchmark</Text>
      <TextInput
        {...id('tts2-base')}
        value={base}
        onChangeText={setBase}
        placeholder="Model files address"
        autoCapitalize="none"
        autoCorrect={false}
        style={{ borderWidth: 1, borderColor: '#999', padding: 8, fontSize: 12 }}
      />
      <Pressable {...id('tts2-start')} onPress={start} style={{ padding: 10, marginVertical: 8, backgroundColor: '#ddd' }}>
        <Text>Start</Text>
      </Pressable>
      <Text {...id('tts2-report')} style={{ fontSize: 9 }}>{text}</Text>
    </ScrollView>
  );
}
