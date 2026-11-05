const test = require('node:test');
const assert = require('node:assert/strict');

const {
  KittenTTS2OnDeviceBase,
  isKittenTTS2Repository,
} = require('../lib/kittentts2/KittenTTS2OnDeviceBase');
const { LanguageModel } = require('../lib/kittentts2/LanguageModel');
const { AudioDecoder } = require('../lib/kittentts2/AudioDecoder');
const { Random, sampleToken } = require('../lib/kittentts2/sampling');
const text = require('../lib/kittentts2/text');
const { QwenTokenizer } = require('../lib/kittentts2/Tokenizer');
const { Graph, NoPathError } = require('../lib/kittentts2/normalizer/Graph');
const { VoiceFile, filesFor } = require('../lib/kittentts2/bundle');
const { decodeUTF8, encodeUTF8 } = require('../lib/kittentts2/utf8');

// A tiny model with the real token layout, small enough to fake ONNX Runtime with.
const TOKENS = {
  audio_id_base: 151675, num_audio_tokens: 6561, speech_start_id: 151669, speech_end_id: 151670,
  text_start_id: 151671, text_end_id: 151672, start_id: 151673, stop_id: 151674, final_seg_id: 158236,
};
const HEAD = 151669;
const HEAD_SIZE = 158241 - HEAD;
const LM = { layers: 2, kvHeads: 1, headDim: 4, headStart: HEAD, speakerDim: 4 };
const AUDIO = TOKENS.audio_id_base;

// --- Fakes -------------------------------------------------------------------------

class FakeTensor {
  constructor(type, data, dims) {
    this.type = type;
    this.data = data;
    this.dims = dims;
  }
}

/** Language model that writes `script` (codec ids), then `<speech_end>`. */
function fakeLanguageModelSession(script, calls = []) {
  let step = 0;
  return {
    calls,
    async run(feeds) {
      calls.push(feeds);
      if (feeds.past_key_0.dims[1] === 1) step = 0; // a new generation starts at the unused row
      const ids = [...feeds.input_ids.data].map(Number);
      const out = {};
      const logits = new Float32Array(HEAD_SIZE).fill(-50);
      const next = step < script.length ? AUDIO + script[step] : TOKENS.speech_end_id;
      logits[next - HEAD] = 50;
      step += 1;
      out.logits = new FakeTensor('float32', logits, [1, HEAD_SIZE]);
      for (let i = 0; i < LM.layers; i += 1) {
        // Each new row holds its token id, so the test can read the cache back.
        const rows = Float32Array.from({ length: ids.length * 4 }, (_, k) => ids[Math.floor(k / 4)]);
        out[`new_key_${i}`] = new FakeTensor('float32', rows, [1, ids.length, 1, 4]);
        out[`new_value_${i}`] = new FakeTensor('float32', rows.map(v => -v), [1, ids.length, 1, 4]);
      }
      return out;
    },
    async release() {},
  };
}

function fakeDecoderSessions(calls = []) {
  const flow = {
    async run(feeds) {
      calls.push(feeds);
      const frames = feeds.speech_tokens.dims[1] * 2;
      return { mel: new FakeTensor('float32', new Float32Array(80 * frames).fill(0.5), [1, 80, frames]) };
    },
    async release() {},
  };
  const hift = {
    async run(feeds) {
      const samples = feeds.mel.dims[2] * 480;
      return { wav: new FakeTensor('float32', new Float32Array(samples).fill(0.25), [1, samples]) };
    },
    async release() {},
  };
  return { flow, hift };
}

function fakeOrt(sessions) {
  return {
    Tensor: FakeTensor,
    InferenceSession: { create: async model => sessions[model] },
  };
}

/** voices.bin with one voice per name, as tools/kitten-tts-2-onnx writes it. */
function voicesBin(names) {
  const chunks = [];
  let length = 0;
  const put = (array) => {
    const bytes = new Uint8Array(array.buffer.slice(0));
    const offset = length;
    chunks.push(bytes);
    length += bytes.length;
    return [offset, array.length];
  };
  const voices = names.map((name, i) => ({
    name,
    id: `${i}_${name.toLowerCase()}`,
    language: 'en',
    transcript: 'Hi.',
    prefix: put(Int32Array.from([TOKENS.start_id, 9, 10, AUDIO + i])),
    speaker: put(new Float32Array(4).fill(i + 1)),
    prompt_token: put(Int32Array.from([1, 2])),
    prompt_feat: put(new Float32Array(4 * 80).fill(0.1)),
    embedding: put(new Float32Array(192).fill(0.2)),
  }));
  let header = new TextEncoder().encode(JSON.stringify({ version: 1, decoder: 'default', voices }));
  const padded = new Uint8Array(header.length + ((4 - ((12 + header.length) % 4)) % 4)).fill(32);
  padded.set(header);
  header = padded;
  const file = new Uint8Array(12 + header.length + length);
  file.set(new TextEncoder().encode('KTVOICE1'), 0);
  new DataView(file.buffer).setUint32(8, header.length, true);
  file.set(header, 12);
  let at = 12 + header.length;
  for (const chunk of chunks) {
    file.set(chunk, at);
    at += chunk.length;
  }
  return file;
}

function manifest() {
  return {
    format: 'kittentts-react-native/kitten-tts-2-onnx',
    version: 1,
    sample_rate: 24000,
    default_voice: 'Bruno',
    default_preset: 'stable',
    decode_presets: {
      stable: { temperature: 0.8, top_p: 0.8, top_k: 50, min_p: 0 },
      expressive: { temperature: 0.9, top_p: 0.9, top_k: 50, min_p: 0 },
    },
    generation: { use_reference_prompt: true, repetition_window: 50, emotion_control: '{emo: 1}', chunk_chars: 380, chunk_min_chars: 130 },
    token_map: TOKENS,
    lm: { graphs: { int8: 'lm.onnx', float32: 'lm-float32.onnx' }, default_precision: 'int8', speaker_dim: 4, head_start: HEAD, layers: LM.layers, kv_heads: 1, head_dim: 4 },
    decoder: { name: 'default', flow: 'flow.onnx', hift: 'hift.onnx', cfm_steps: 2, silence_token: 4299, token_mel_ratio: 2, samples_per_frame: 480, mels: 80 },
    tokenizer: { vocab: 'vocab.json', merges: 'merges.txt', added_tokens: 'added_tokens.json' },
    normalizer: { en: { tagger: 'tagger', verbalizer: 'verbalizer' } },
    voices: 'voices.bin',
    files: [],
  };
}

/** A byte-level vocabulary: every printable ASCII byte, plus one merge. */
function tokenizerFiles() {
  const vocab = {};
  for (let byte = 33; byte <= 126; byte += 1) vocab[String.fromCharCode(byte)] = byte;
  vocab['Ġ'] = 220; // the space byte, as GPT-2's table maps it
  vocab.he = 1000;
  return {
    vocab: JSON.stringify(vocab),
    merges: '#version: 0.2\nh e\n',
    added: JSON.stringify({ '<|speech_start|>': 151669 }),
  };
}

/** A grammar that copies its input: one state with a self-loop per byte. */
function identityGrammar() {
  const arcs = 256 - 1;
  const data = new Uint8Array(20 + 4 * (2 + 1 + 3 * arcs));
  const view = new DataView(data.buffer);
  data.set(new TextEncoder().encode('KITTEN1\0'));
  view.setUint32(8, 0, true); // start
  view.setUint32(12, 1, true); // states
  view.setUint32(16, arcs, true);
  let at = 20;
  view.setUint32(at, 0, true); view.setUint32(at + 4, arcs, true); at += 8; // offsets
  view.setFloat32(at, 0, true); at += 4; // finals
  for (let byte = 1; byte < 256; byte += 1) { view.setUint32(at, byte | (byte << 8), true); at += 4; }
  for (let byte = 1; byte < 256; byte += 1) { view.setUint32(at, 0, true); at += 4; }
  for (let byte = 1; byte < 256; byte += 1) { view.setFloat32(at, 0, true); at += 4; }
  return data;
}

async function createTTS({ script = [1, 2, 3], voices = ['Bruno', 'Luna'], lmCalls = [], flowCalls = [], precision, opened = [], native, decoder: decoderName } = {}) {
  const files = tokenizerFiles();
  const reader = {
    readText: async path => ({ 'vocab.json': files.vocab, 'merges.txt': files.merges, 'added_tokens.json': files.added })[path],
    readBytes: async (path, offset = 0, length) => {
      const bin = voicesBin(path === 'voices-student.bin' ? ['Student'] : voices);
      return path.startsWith('voices') ? bin.slice(offset, length === undefined ? undefined : offset + length) : new Uint8Array();
    },
    graph: async path => path,
  };
  const decoder = fakeDecoderSessions(flowCalls);
  const ort = fakeOrt({
    'lm.onnx': fakeLanguageModelSession(script, lmCalls),
    'lm-float32.onnx': fakeLanguageModelSession(script, lmCalls),
    'flow.onnx': decoder.flow,
    'hift.onnx': decoder.hift,
  });
  const create = ort.InferenceSession.create;
  ort.InferenceSession.create = async (model, options) => {
    opened.push(model);
    return create(model, options);
  };
  const state = await openWithoutGrammars({ reader, ort, precision, native, decoder: decoderName });
  return new KittenTTS2OnDeviceBase(state);
}

// The grammars copy their input, so normalization fails over to the cleaned text.
async function openWithoutGrammars({ reader, ort, precision, native, decoder }) {
  const pako = require('pako');
  const grammar = pako.gzip(identityGrammar());
  const withGrammars = { ...reader, readBytes: async (path, offset, length) => (path === 'tagger' || path === 'verbalizer' ? grammar : reader.readBytes(path, offset, length)) };
  return KittenTTS2OnDeviceBase.open({
    repository: 'KittenML/kitten-tts-2',
    manifest: { ...manifest(), decoders: { student_w4: { weights: 'student.safetensors', voices: 'voices-student.bin', cfm_steps: 1 } } },
    reader: withGrammars,
    ort,
    sessionOptions: {},
    audioOutput: { play: async () => {}, stop: async () => {} },
    precision,
    native,
    decoder,
  });
}

// --- Sampling ----------------------------------------------------------------------

const SETTINGS = {
  temperature: 0.8, topK: 50, topP: 0.8, minP: 0, repetitionPenalty: 1.1,
  repetitionWindow: 50, tokenRunPenalty: 1.3, tokenRunGrace: 10,
};
const IDS = { headStart: HEAD, audioStart: AUDIO, audioEnd: AUDIO + 6561, exempt: [TOKENS.speech_end_id, TOKENS.stop_id] };

test('the sampler is reproducible with a seed and draws from the head', () => {
  const draw = seed => {
    const random = new Random(seed);
    return Array.from({ length: 20 }, () => {
      const scores = Float32Array.from({ length: HEAD_SIZE }, (_, i) => Math.sin(i) * 3);
      return sampleToken(scores, [TOKENS.speech_start_id], 1, SETTINGS, IDS, random);
    });
  };
  assert.deepEqual(draw(7), draw(7));
  assert.notDeepEqual(draw(7), draw(8));
  for (const id of draw(9)) assert.ok(id >= HEAD && id < HEAD + HEAD_SIZE);
});

test('top-k and top-p leave only the likely tokens', () => {
  const random = new Random(1);
  const seen = new Set();
  for (let i = 0; i < 200; i += 1) {
    const scores = new Float32Array(HEAD_SIZE).fill(-20);
    scores[100] = 5;
    scores[101] = 4.9;
    scores[102] = 0;
    seen.add(sampleToken(scores, [TOKENS.speech_start_id], 1, { ...SETTINGS, topP: 0.6 }, IDS, random) - HEAD);
  }
  assert.deepEqual([...seen].sort(), [100, 101]);
});

test('a codec token held past the grace length is penalised', () => {
  const held = AUDIO + 5;
  const history = [TOKENS.speech_start_id, ...new Array(12).fill(held)];
  const scores = new Float32Array(HEAD_SIZE).fill(0);
  scores[held - HEAD] = 10;
  sampleToken(scores, history, 1, { ...SETTINGS, repetitionPenalty: 1, temperature: 1 }, IDS, new Random(1));
  // Run of 12, grace 10: (12 - 10 + 1) * 1.3 off the held token.
  assert.ok(Math.abs(scores[held - HEAD] - (10 - 3 * 1.3)) < 1e-5);
});

test('the repetition penalty spares <speech_end> and [STOP]', () => {
  const scores = new Float32Array(HEAD_SIZE).fill(0);
  scores[TOKENS.speech_end_id - HEAD] = 4;
  scores[AUDIO + 1 - HEAD] = 4;
  const history = [TOKENS.speech_start_id, AUDIO + 1, TOKENS.speech_end_id];
  sampleToken(scores, history, 1, { ...SETTINGS, temperature: 1, tokenRunPenalty: 0 }, IDS, new Random(1));
  assert.equal(scores[TOKENS.speech_end_id - HEAD], 4);
  assert.ok(Math.abs(scores[AUDIO + 1 - HEAD] - 4 / 1.1) < 1e-5);
});

// --- Text --------------------------------------------------------------------------

test('text without a normalizer is capitalized, terminated, and keeps its tags', () => {
  assert.equal(text.normalizeText('hello there', null), 'Hello there.');
  assert.equal(text.normalizeText('[joyful] we won <laugh> it was (((great)))', null), '[joyful] We won <laugh> it was (((great))).');
  assert.equal(text.normalizeText('wait — what -- really', null), 'Wait, what, really.');
  assert.equal(text.normalizeText('see (the appendix) here', null), 'See, the appendix, here.');
  assert.equal(text.normalizeText('x² ≥ 0', null), 'X 2 is greater than or equal to 0.');
});

test('a normalizer failure falls back to the cleaned text', () => {
  const failing = () => { throw new Error('boom'); };
  assert.equal(text.normalizeText('it costs 5 dollars', failing), 'It costs 5 dollars.');
});

test('expression tags are recognised', () => {
  assert.equal(text.hasExpressionTag('[Joyful] hi'), true);
  assert.equal(text.hasExpressionTag('so <sigh> tired'), true);
  assert.equal(text.hasExpressionTag('(((very))) good'), true);
  assert.equal(text.hasExpressionTag('section [3] and x < 5'), false);
});

test('long text is chunked at sentence boundaries with short chunks merged', () => {
  const sentence = 'This sentence has a handful of ordinary words in it. ';
  const chunks = text.splitForSynthesis(sentence.repeat(20).trim(), 380, 130);
  assert.ok(chunks.length > 1);
  for (const chunk of chunks) {
    assert.ok(chunk.length <= 380 * 1.4, `chunk of ${chunk.length}`);
    assert.ok(chunk.length >= 130 || chunks.length === 1);
    assert.match(chunk, /[.!?,;:]$/);
  }
  assert.deepEqual(text.splitForSynthesis('Short.', 380, 130), ['Short.']);
});

test('chunks are joined with one pause and faded edges', () => {
  const rate = 24000;
  const loud = () => Float32Array.from({ length: rate }, (_, i) => Math.sin(i / 10) * 0.5);
  const joined = text.joinChunks([loud(), loud()], rate, 0.16);
  const gap = Math.floor(rate * 0.16);
  assert.ok(Math.abs(joined.length - (2 * rate + gap)) < rate * 0.05);
  assert.equal(joined[0], 0);
  assert.equal(text.joinChunks([], rate), null);
});

test('UTF-8 round-trips, emoji included', () => {
  const sample = 'naïve café 日本 😀';
  assert.equal(decodeUTF8(encodeUTF8(sample)), sample);
  assert.deepEqual([...encodeUTF8('é')], [0xc3, 0xa9]);
});

// --- Tokenizer and grammar ---------------------------------------------------------

test('the tokenizer merges by rank and splits out added tokens', () => {
  const files = tokenizerFiles();
  const tokenizer = QwenTokenizer.fromFiles(files.vocab, files.merges, files.added);
  assert.deepEqual(tokenizer.encode('hex'), [1000, 'x'.charCodeAt(0)]);
  assert.deepEqual(tokenizer.encode('a<|speech_start|>b'), [97, 151669, 98]);
  assert.deepEqual(tokenizer.encode('a b'), [97, 220, 98]); // ' b' is one piece: the space byte, then b
});

test('a grammar rewrites byte by byte and refuses what it cannot read', () => {
  const graph = new Graph(identityGrammar());
  assert.equal(graph.rewrite('café ok'), 'café ok');
  assert.throws(() => new Graph(new Uint8Array(20)), /Invalid grammar/);
  assert.ok(new NoPathError('x') instanceof Error);
});

test('voices.bin is read one voice at a time, into buffers of their own', async () => {
  const bin = voicesBin(['Bruno', 'Luna']);
  const file = await VoiceFile.open(async (offset, length) => bin.slice(offset, offset + length));
  assert.deepEqual(file.entries.map(entry => entry.name), ['Bruno', 'Luna']);
  const luna = await file.load(file.entries[1]);
  assert.deepEqual(luna.prefix, [TOKENS.start_id, 9, 10, AUDIO + 1]);
  assert.deepEqual([...luna.speaker], [2, 2, 2, 2]);
  assert.equal(luna.conditioning.promptFeat.length, 320);
  for (const array of [luna.speaker, luna.conditioning.promptFeat, luna.conditioning.embedding]) {
    assert.equal(array.byteOffset, 0);
  }
});

// --- Language model and decoder ----------------------------------------------------

test('the language model passes the cache as offset-0 views and stops at <speech_end>', async () => {
  const calls = [];
  const lm = await LanguageModel.load(fakeOrt({ lm: fakeLanguageModelSession([7, 8, 9], calls) }), 'lm', LM, {
    speechEnd: TOKENS.speech_end_id, stop: TOKENS.stop_id, audioStart: AUDIO, audioEnd: AUDIO + 6561,
  }, {});
  const prompt = [TOKENS.start_id, TOKENS.text_start_id, 5, TOKENS.speech_start_id];
  const codes = await lm.generate(prompt, new Float32Array(4), { ...SETTINGS, temperature: 1 }, 100, new Random(1));
  assert.deepEqual(codes, [7, 8, 9]);
  assert.equal(calls.length, 4); // the prompt, then one call per kept token
  assert.deepEqual(calls[0].past_key_0.dims, [1, 1, 1, 4]); // only the unused row
  assert.deepEqual(calls[1].past_key_0.dims, [1, 1 + prompt.length, 1, 4]);
  for (const call of calls) assert.equal(call.past_key_0.data.byteOffset, 0);
  // Row 1 holds the first prompt token, written back from new_key_0.
  assert.equal(calls[1].past_key_0.data[4], TOKENS.start_id);
});

test('the language model honours maxNewTokens and cancellation', async () => {
  const load = () => LanguageModel.load(fakeOrt({ lm: fakeLanguageModelSession(new Array(50).fill(3)) }), 'lm', LM, {
    speechEnd: TOKENS.speech_end_id, stop: TOKENS.stop_id, audioStart: AUDIO, audioEnd: AUDIO + 6561,
  }, {});
  const settings = { ...SETTINGS, temperature: 1, tokenRunPenalty: 0, repetitionPenalty: 1 };
  assert.equal((await (await load()).generate([TOKENS.speech_start_id], new Float32Array(4), settings, 5, new Random(1))).length, 5);
  await assert.rejects((await load()).generate([TOKENS.speech_start_id], new Float32Array(4), settings, 5, new Random(1), { aborted: true }),
    error => error.code === 'CANCELLED');
});

test('decoder input and the fade are shared by both backends', () => {
  const { applyTrimFade, decoderTokens } = require('../lib/kittentts2/AudioDecoder');
  assert.deepEqual(decoderTokens([5, -1, 6561, 6], 4299), [5, 6, 4299, 4299, 4299]);
  assert.equal(decoderTokens([99999], 4299), null);
  const wav = applyTrimFade(new Float32Array(1200).fill(1));
  assert.equal(wav[0], 0);
  assert.equal(wav[479], 0);
  assert.equal(wav[480], 0);
  assert.ok(Math.abs(wav[959] - 1) < 1e-6);
  assert.equal(wav[1199], 1);
});

test('the decoder pads with silence, drops invalid codes and fades in', async () => {
  const calls = [];
  const sessions = fakeDecoderSessions(calls);
  const decoder = await AudioDecoder.load(fakeOrt({ flow: sessions.flow, hift: sessions.hift }),
    { model: 'flow', options: {} }, { model: 'hift', options: {} },
    { cfmSteps: 2, silenceToken: 4299, tokenMelRatio: 2, samplesPerFrame: 480, mels: 80 });
  const conditioning = { promptToken: Int32Array.from([1, 2]), promptFeat: new Float32Array(320), embedding: new Float32Array(192) };
  const wav = await decoder.decode([5, 99999, 6], conditioning, new Random(1));
  assert.deepEqual([...calls[0].speech_tokens.data].map(Number), [5, 6, 4299, 4299, 4299]);
  assert.deepEqual(calls[0].noise.dims, [1, 80, 2 * (2 + 5)]);
  assert.equal(wav.length, 5 * 2 * 480);
  assert.equal(wav[0], 0); // the first 20 ms are silenced
  assert.equal(wav[wav.length - 1], 0.25);
  assert.equal(await decoder.decode([99999], conditioning, new Random(1)), null);
});

// --- The class apps get ------------------------------------------------------------

test('generate() runs the pipeline with the Python prompt layout', async () => {
  const lmCalls = [];
  const tts = await createTTS({ lmCalls });
  assert.deepEqual(tts.availableVoices, ['Bruno', 'Luna']);
  assert.deepEqual(tts.presets, ['stable', 'expressive']);
  const result = await tts.generate('he', 'Luna');
  assert.equal(result.voice, 'Luna');
  assert.equal(result.sampleRate, 24000);
  assert.ok(result.samples.length > 0);
  const prompt = [...lmCalls[0].input_ids.data].map(Number);
  // The voice's reference prefix, the text, <final_seg>, <speech_start>.
  assert.deepEqual(prompt.slice(0, 4), [TOKENS.start_id, 9, 10, AUDIO + 1]);
  assert.equal(prompt[4], TOKENS.text_start_id);
  assert.deepEqual(prompt.slice(-2), [TOKENS.final_seg_id, TOKENS.speech_start_id]);
  assert.deepEqual([...lmCalls[0].speaker_embedding.data], [2, 2, 2, 2]);
});

test('the default voice, streaming and the refusals', async () => {
  const tts = await createTTS();
  assert.equal((await tts.generate('he')).voice, 'Bruno');
  const chunks = [];
  for await (const chunk of tts.generateStreaming('he', { voice: 'Bruno' })) chunks.push(chunk);
  assert.equal(chunks.length, 1);
  await assert.rejects(tts.generate('he', 'Nobody'), error => error.code === 'INVALID_ARGUMENT');
  await assert.rejects(tts.generate('he', { preset: 'loud' }), error => error.code === 'INVALID_ARGUMENT');
  await assert.rejects(tts.generate('he', { reference: 'file:///me.wav' }), error => error.code === 'INVALID_ARGUMENT');
  await assert.rejects(tts.generate('   '), error => error.code === 'EMPTY_INPUT');
  await tts.dispose();
  await assert.rejects(tts.generate('he'), error => error.code === 'ENGINE_NOT_READY');
});

test('precision picks the language model graph', async () => {
  const opened = [];
  assert.equal((await createTTS({ opened })).precision, 'int8');
  assert.equal(opened[0], 'lm.onnx');
  opened.length = 0;
  assert.equal((await createTTS({ opened, precision: 'float32' })).precision, 'float32');
  assert.equal(opened[0], 'lm-float32.onnx');
  await assert.rejects(createTTS({ precision: 'float16' }), error => error.code === 'INVALID_ARGUMENT');
});

test('a native engine takes the place of every ONNX graph', async () => {
  const opened = [];
  const calls = [];
  const decoded = [];
  const generator = {
    async generate(prompt, speaker, settings, maxNewTokens) {
      calls.push({ prompt, speaker: [...speaker], settings, maxNewTokens });
      return [4, 5, 6];
    },
    async release() {},
  };
  const decoder = {
    async decode(codes, conditioning) {
      decoded.push({ codes, frames: conditioning.promptFeat.length / 80 });
      return new Float32Array(2400).fill(0.1);
    },
    async release() {},
  };
  let tokens;
  const load = async t => { tokens = t; return { languageModel: generator, decoder }; };
  const tts = await createTTS({ opened, native: { backend: 'mlx', device: 'gpu', precision: 'bfloat16', load } });
  assert.equal(tts.backend, 'mlx');
  assert.equal(tts.device, 'gpu');
  assert.equal(tts.precision, 'bfloat16');
  assert.deepEqual(opened, [], `opened ${opened}`);
  assert.deepEqual(tokens, { speechEnd: TOKENS.speech_end_id, stop: TOKENS.stop_id, audioStart: AUDIO, audioEnd: AUDIO + 6561 });
  const result = await tts.generate('he', { voice: 'Luna', temperature: 0.7 });
  assert.equal(result.samples.length, 2400);
  assert.deepEqual(decoded, [{ codes: [4, 5, 6], frames: 4 }]);
  assert.equal(calls[0].settings.temperature, 0.7);
  assert.deepEqual(calls[0].speaker, [2, 2, 2, 2]);
  assert.deepEqual(calls[0].prompt.slice(-1), [TOKENS.speech_start_id]);
});

test('ggml on the CPU reports where it runs, and ONNX Runtime the CPU', async () => {
  const generator = { async generate() { return [4]; }, async release() {} };
  const decoder = { async decode() { return new Float32Array(480); }, async release() {} };
  const native = { backend: 'ggml', device: 'cpu', precision: 'int8', load: async () => ({ languageModel: generator, decoder }) };
  const tts = await createTTS({ native });
  assert.deepEqual([tts.backend, tts.device, tts.precision], ['ggml', 'cpu', 'int8']);
  const onnx = await createTTS({});
  assert.deepEqual([onnx.backend, onnx.device], ['onnx', 'cpu']);
});

test('a decoder has its own voices and files, and its name is checked', async () => {
  const generator = { async generate() { return [4]; }, async release() {} };
  const decoder = { async decode() { return new Float32Array(480); }, async release() {} };
  const native = { backend: 'mlx', device: 'gpu', precision: 'bfloat16', load: async () => ({ languageModel: generator, decoder }) };
  const tts = await createTTS({ native, decoder: 'student_w4' });
  assert.equal(tts.decoder, 'student_w4');
  assert.deepEqual(tts.availableVoices, ['Student']);
  assert.deepEqual(tts.availableDecoders, ['default', 'student_w4']);
  assert.equal((await createTTS({ native })).decoder, 'default');
  await assert.rejects(createTTS({ native, decoder: 'nope' }), /decoder must be one of: default, student_w4/);

  const files = [
    { path: 'shared', size: 1 },
    { path: 'flow.onnx', size: 1, backend: 'onnx' },
    { path: 'lm', size: 1, backend: 'native' },
    { path: 'voices.bin', size: 1, decoder: 'default' },
    { path: 'decoder', size: 1, backend: 'native', decoder: 'default' },
    { path: 'voices-student.bin', size: 1, decoder: 'student_w4' },
    { path: 'student', size: 1, backend: 'native', decoder: 'student_w4' },
  ];
  const paths = (backend, name) => filesFor({ files }, backend, name).map(file => file.path);
  assert.deepEqual(paths('native', 'student_w4'), ['shared', 'lm', 'voices-student.bin', 'student']);
  assert.deepEqual(paths('native', 'default'), ['shared', 'lm', 'voices.bin', 'decoder']);
  assert.deepEqual(paths('onnx', 'default'), ['shared', 'flow.onnx', 'voices.bin']);
});

test('repositories select the on-device model; API model IDs do not', () => {
  assert.equal(isKittenTTS2Repository('KittenML/kitten-tts-2'), true);
  assert.equal(isKittenTTS2Repository('someone/kitten-tts-2-mini'), true);
  assert.equal(isKittenTTS2Repository('kitten-tts-2'), false);
  assert.equal(isKittenTTS2Repository('kitten-tts-2-latest'), false);
  assert.equal(isKittenTTS2Repository('KittenML/kitten-tts-nano-0.8'), false);
});
