// Runs the on-device Kitten TTS 2 runtime against an exported model folder:
//
//   python tools/kitten-tts-2-onnx/export.py --out /tmp/kitten-tts-2-onnx
//   KITTEN_TTS_2_ONNX=/tmp/kitten-tts-2-onnx npm run test:live
//
// Needs onnxruntime-node where Node can resolve it (npm install --no-save
// onnxruntime-node). Skipped without either.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const pako = require('pako');

const { KittenTTS2OnDeviceBase } = require('../../lib/kittentts2/KittenTTS2OnDeviceBase');
const { Normalizer } = require('../../lib/kittentts2/normalizer/Normalizer');
const { QwenTokenizer } = require('../../lib/kittentts2/Tokenizer');
const text = require('../../lib/kittentts2/text');

const folder = process.env.KITTEN_TTS_2_ONNX;
let ort;
try {
  ort = require('onnxruntime-node');
} catch {
  ort = undefined;
}
const live = {
  skip: !folder ? 'KITTEN_TTS_2_ONNX is not set' : !ort ? 'onnxruntime-node is not installed' : false,
};

const file = name => path.join(folder, name);
const manifest = () => JSON.parse(fs.readFileSync(file('manifest.json'), 'utf8'));

test('text reaches the model exactly as the Python package sends it', live, () => {
  const m = manifest();
  const grammar = name => pako.ungzip(fs.readFileSync(file(name)));
  const normalizer = new Normalizer({
    tagger: grammar(m.normalizer.en.tagger),
    verbalizer: grammar(m.normalizer.en.verbalizer),
    post: grammar(m.normalizer.en.post),
  });
  const tokenizer = QwenTokenizer.fromFiles(
    fs.readFileSync(file(m.tokenizer.vocab), 'utf8'),
    fs.readFileSync(file(m.tokenizer.merges), 'utf8'),
    fs.readFileSync(file(m.tokenizer.added_tokens), 'utf8'),
  );
  // Written by the Python package (kittenml 0.9.7, kitten-text-processing 0.1.0).
  const cases = require('../fixtures/kitten-tts-2-text.json');
  for (const c of cases) {
    if (c.ids_raw) {
      assert.deepEqual(tokenizer.encode(c.text), c.ids_raw, c.text);
      continue;
    }
    const spoken = text.normalizeText(c.text, value => normalizer.normalizeText(value));
    assert.equal(spoken, c.spoken, c.text);
    assert.equal(text.normalizeText(c.text, null, false), c.raw, c.text);
    const chunks = text.splitForSynthesis(spoken, 380, 130);
    assert.deepEqual(chunks, c.chunks, c.text);
    assert.deepEqual(chunks.map(chunk => tokenizer.encode(chunk)), c.ids, c.text);
    assert.equal(text.hasExpressionTag(c.text), c.tag, c.text);
  }
});

test('generate() speaks a sentence on the CPU', live, async () => {
  const reader = {
    readText: async name => fs.readFileSync(file(name), 'utf8'),
    readBytes: async (name, offset = 0, length) => {
      const bytes = fs.readFileSync(file(name));
      return new Uint8Array(bytes.subarray(offset, length === undefined ? undefined : offset + length));
    },
    graph: async name => file(name),
  };
  const state = await KittenTTS2OnDeviceBase.open({
    repository: 'KittenML/kitten-tts-2',
    manifest: manifest(),
    reader,
    ort,
    sessionOptions: { intraOpNumThreads: 4, graphOptimizationLevel: 'all' },
    audioOutput: { play: async () => {}, stop: async () => {} },
  });
  const tts = new KittenTTS2OnDeviceBase(state);
  try {
    assert.ok(tts.availableVoices.includes('Bruno'));
    const result = await tts.generate('Hello from the device.', { voice: 'Bruno', advanced: { seed: 1 } });
    assert.ok(result.duration > 0.8 && result.duration < 6, `${result.duration} s`);
    let peak = 0;
    for (const value of result.samples) {
      assert.ok(Number.isFinite(value));
      peak = Math.max(peak, Math.abs(value));
    }
    assert.ok(peak > 0.05, `peak ${peak}`);
  } finally {
    await tts.dispose();
  }
});
