// Kitten TTS 2 Nano / Pico / Mini benchmark for real devices (TestMu). It starts on
// launch, downloads the models and text data from Hugging Face like any app would, and
// shows its report as JSON (accessibility label "tiny-report") so the runner can read it
// while it runs: a crash still leaves the steps that finished.
import { useEffect, useState } from 'react';
import { Platform, ScrollView, StyleSheet, Text } from 'react-native';
import * as Device from 'expo-device';
import { activateKeepAwakeAsync } from 'expo-keep-awake';
import { KittenTTS } from '@kittentts/react-native';

const TEXTS: Record<string, string[]> = {
  en: [
    'Hello there. Dr. Smith read 3 books on 2026-10-13 at 3:45pm, costing $24.5B.',
    'Kevin met Søren at the café on Friday, and they talked for an hour.',
    'The quick brown fox jumps over the lazy dog, twice!',
  ],
  de: ['Guten Morgen, wie geht es dir heute?', 'Ich habe 25 Äpfel für 3,50 Euro gekauft.'],
  fr: ["Bonjour, je m'appelle Marie et j'habite à Paris.", "Il fait beau aujourd'hui, n'est-ce pas ?"],
  es: ['Hola, ¿cómo estás? Hoy hace buen tiempo.'],
  ru: ['Привет, как дела? Сегодня хорошая погода.'],
  hi: ['नमस्ते, आप कैसे हैं? आज मौसम अच्छा है।'],
  it: ['Buongiorno, come stai oggi?'],
  pt: ['Olá, tudo bem? Hoje está um dia bonito.'],
};

// Long enough for several chunks (350 phonemes each), so the first chunk shows how soon
// audio can start.
const PARAGRAPH = 'The little cat sat by the window and watched the rain fall over the quiet town. '
  + 'Every evening the baker closed his shop at six, swept the floor, and walked home along the river. '
  + 'On Sundays the market filled the square with apples, bread, flowers and old books, and children '
  + 'ran between the stalls while their parents argued happily about prices.';
const LONG = [PARAGRAPH, PARAGRAPH, PARAGRAPH].join(' ');

type Model = 'kitten-tts-2-nano' | 'kitten-tts-2-pico' | 'kitten-tts-2-mini';
const RUNS: Array<{ model: Model; weights: 'int8' | 'int4'; langs: string[]; stream?: boolean }> = [
  { model: 'kitten-tts-2-nano', weights: 'int8', langs: ['en', 'de', 'fr', 'es', 'ru', 'hi', 'it', 'pt'], stream: true },
  { model: 'kitten-tts-2-pico', weights: 'int4', langs: ['en', 'de'], stream: true },
  { model: 'kitten-tts-2-mini', weights: 'int8', langs: ['en', 'fr'], stream: true },
];

const median = (xs: number[]) => {
  const s = [...xs].sort((a, b) => a - b);
  return s.length ? (s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2) : 0;
};

export default function App() {
  const [report, setReport] = useState<any>({ status: 'starting', done: false });
  useEffect(() => {
    activateKeepAwakeAsync().catch(() => {});
    const r: any = {
      status: 'starting',
      done: false,
      device: {
        model: Device.modelName, brand: Device.brand, os: `${Platform.OS} ${Device.osVersion}`,
        year: Device.deviceYearClass, memoryGB: Device.totalMemory ? +(Device.totalMemory / 2 ** 30).toFixed(1) : null,
        cpu: Device.supportedCpuArchitectures,
      },
      runs: [],
    };
    const show = (status: string) => {
      r.status = status;
      setReport(JSON.parse(JSON.stringify(r)));
    };
    (async () => {
      for (const run of RUNS) {
        const out: any = { model: run.model, weights: run.weights, items: [] };
        r.runs.push(out);
        show(`${run.model} ${run.weights}: create`);
        let t0 = Date.now();
        const tts = await KittenTTS.create({ model: run.model, weights: run.weights, languages: ['en'] });
        out.createMs = Date.now() - t0;
        for (const lang of run.langs) {
          show(`${run.model} ${lang}`);
          t0 = Date.now();
          await tts.phonemize('Warm up.', lang);
          const firstUseMs = Date.now() - t0;
          for (const [i, text] of TEXTS[lang].entries()) {
            t0 = Date.now();
            const chunks = await tts.phonemize(text, lang);
            const phonMs = Date.now() - t0;
            t0 = Date.now();
            const res = await tts.generate(text, { language: lang });
            const genMs = Date.now() - t0;
            out.items.push({
              lang, i, text, chunks, phonMs, genMs, firstUseMs: i === 0 ? firstUseMs : undefined,
              seconds: +res.duration.toFixed(3), rtf: +(genMs / 1000 / res.duration).toFixed(4),
              peak: +res.samples.reduce((m, v) => Math.max(m, Math.abs(v)), 0).toFixed(3),
            });
          }
        }
        if (run.stream) {
          show(`${run.model}: streaming`);
          t0 = Date.now();
          let first = 0;
          let seconds = 0;
          let chunks = 0;
          for await (const part of tts.generateStreaming(LONG)) {
            if (!first) first = Date.now() - t0;
            seconds += part.duration;
            chunks += 1;
          }
          const totalMs = Date.now() - t0;
          out.stream = { chunks, firstChunkMs: first, totalMs, seconds: +seconds.toFixed(2), rtf: +(totalMs / 1000 / seconds).toFixed(4) };
        }
        const rtfs = out.items.map((it: any) => it.rtf);
        out.summary = {
          rtfMedian: +median(rtfs).toFixed(4), rtfMin: Math.min(...rtfs), rtfMax: Math.max(...rtfs),
          phonMsMedian: median(out.items.map((it: any) => it.phonMs)),
        };
        await tts.dispose();
      }
      r.done = true;
      show('done');
    })().catch(e => {
      r.error = String(e?.stack ?? e).slice(0, 2000);
      r.done = true;
      show('failed');
    });
  }, []);
  const json = JSON.stringify(report);
  return (
    <ScrollView contentContainerStyle={styles.box}>
      <Text style={styles.status}>{report.status}</Text>
      <Text accessibilityLabel="tiny-report" testID="tiny-report" style={styles.json}>{json}</Text>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  box: { padding: 16, paddingTop: 64 },
  status: { fontSize: 16, marginBottom: 12 },
  json: { fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace', fontSize: 9 },
});
