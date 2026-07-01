import React, {useState, useCallback, useEffect, useRef} from 'react';
import {
  SafeAreaView,
  ScrollView,
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  ActivityIndicator,
  Platform,
} from 'react-native';
import Sound from 'react-native-sound';
import {
  KittenTTS,
  KittenModel,
  KittenVoice,
  KittenTTSResult,
  modelDisplayName,
  voiceDisplayName,
  ALL_VOICES,
  createRNSoundPlayer,
} from '@kittentts/react-native';

type AppState =
  | {kind: 'idle'}
  | {kind: 'preparing'}
  | {kind: 'downloading'; progress: number}
  | {kind: 'generating'}
  | {
      kind: 'benchmarking';
      model: string;
      completed: number;
      total: number;
    }
  | {kind: 'playing'}
  | {kind: 'error'; message: string};

type BenchmarkRow = {
  model: string;
  modelDisplayName: string;
  status: 'passed' | 'failed';
  voice: string;
  voiceDisplayName: string;
  speed: number;
  firstGenerationMs?: number;
  firstGenerationSeconds?: number;
  firstRtf?: number;
  warmRunCount?: number;
  warmGenerationMs?: number[];
  warmGenerationSeconds?: number[];
  warmRtf?: number[];
  warmP50GenerationMs?: number;
  warmP50GenerationSeconds?: number;
  warmP95GenerationMs?: number;
  warmP95GenerationSeconds?: number;
  warmP50Rtf?: number;
  warmP95Rtf?: number;
  generationMs?: number;
  generationSeconds?: number;
  durationSeconds?: number;
  rtf?: number;
  sampleCount?: number;
  sampleRate?: number;
  sampleHash?: string;
  werReferenceText?: string;
  werAudioFormat?: 'wav-base64';
  werAudioSampleRate?: number;
  werAudioBase64?: string;
  werAudioBase64Length?: number;
  werAudioChunkCount?: number;
  werAudioChunkSize?: number;
  parakeetTranscript?: string;
  parakeetWer?: number;
  parakeetWerPercent?: number;
  parakeetStatus?: 'pending' | 'passed' | 'failed' | 'skipped';
  parakeetErrorSummary?: string;
  failedStage?: string;
  errorSummary?: string;
};

type BenchmarkReport = {
  schemaVersion: 1;
  sampleText: string;
  characterLength: number;
  voice: string;
  voiceDisplayName: string;
  speed: number;
  startedAt: string;
  finishedAt: string | null;
  rows: BenchmarkRow[];
};

const MODELS: KittenModel[] = [
  KittenModel.Nano,
  KittenModel.NanoInt8,
  KittenModel.Micro,
  KittenModel.Mini,
];

const SPEED_OPTIONS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0];
const BENCHMARK_MODEL_TIMEOUT_MS = 90 * 1000;
const BENCHMARK_WARM_RUNS = 5;
const WER_AUDIO_CHUNK_SIZE = 16000;

function e2eTextProps(testID: string) {
  if (Platform.OS === 'android') {
    return {testID, accessibilityLabel: testID};
  }

  return {testID};
}

function automationSlug(value: string) {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '');
}

function makeFailedBenchmarkRow(
  model: KittenModel,
  failedStage: string,
  errorSummary: string,
  voice: KittenVoice,
  speed: number,
): BenchmarkRow {
  return {
    model: String(model),
    modelDisplayName: modelDisplayName(model),
    status: 'failed',
    voice: String(voice),
    voiceDisplayName: voiceDisplayName(voice),
    speed,
    failedStage,
    errorSummary,
  };
}

export default function App() {
  const [tts, setTts] = useState<KittenTTS | null>(null);
  const ttsRef = useRef<KittenTTS | null>(null);
  const mountedRef = useRef(true);
  const [state, setState] = useState<AppState>({kind: 'idle'});
  const [inputText, setInputText] = useState(
    'Hello! Welcome to KittenTTS, a fast on-device text-to-speech engine.',
  );
  const [selectedModel, setSelectedModel] = useState(KittenModel.Nano);
  const [selectedVoice, setSelectedVoice] = useState(KittenVoice.Bella);
  const [selectedSpeed, setSelectedSpeed] = useState(1.0);
  const [result, setResult] = useState<KittenTTSResult | null>(null);
  const [benchmarkReport, setBenchmarkReport] =
    useState<BenchmarkReport | null>(null);

  const isWorking =
    state.kind === 'preparing' ||
    state.kind === 'downloading' ||
    state.kind === 'generating' ||
    state.kind === 'benchmarking' ||
    state.kind === 'playing';

  const initTTS = useCallback(async (model: KittenModel) => {
    try {
      await ttsRef.current?.dispose();
      setState({kind: 'preparing'});
      setResult(null);
      setBenchmarkReport(null);

      const instance = await KittenTTS.create(
        {model, player: createRNSoundPlayer(Sound)},
        (progress, info) => {
          if (mountedRef.current && info?.stage === 'downloading') {
            setState({
              kind: 'downloading',
              progress,
            });
          }
        },
      );

      if (!mountedRef.current) {
        if (!__DEV__) await instance.dispose();
        return;
      }

      ttsRef.current = instance;
      setTts(instance);
      setState({kind: 'idle'});
    } catch (error: unknown) {
      ttsRef.current = null;
      if (mountedRef.current) {
        setTts(null);
        setState({
          kind: 'error',
          message: getErrorMessage(error, 'Init failed'),
        });
      }
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    initTTS(selectedModel);
    return () => {
      mountedRef.current = false;
      // Fast Refresh can tear down the JS runtime while ONNX native objects are
      // still active, so avoid releasing the session during dev reloads.
      if (!__DEV__) {
        ttsRef.current?.dispose().catch(() => {});
      }
      ttsRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleGenerate = useCallback(async () => {
    if (!tts || !inputText.trim()) {
      return;
    }
    try {
      setState({kind: 'generating'});
      setBenchmarkReport(null);
      const res = await tts.generate(inputText, selectedVoice, selectedSpeed);
      setResult(res);
      setState({kind: 'idle'});
    } catch (error: unknown) {
      setState({
        kind: 'error',
        message: getErrorMessage(error, 'Generation failed'),
      });
    }
  }, [tts, inputText, selectedVoice, selectedSpeed]);

  const handleSpeak = useCallback(async () => {
    if (!tts || !inputText.trim()) {
      return;
    }
    try {
      setState({kind: 'playing'});
      setBenchmarkReport(null);
      const res = await tts.speak(inputText, selectedVoice, selectedSpeed);
      setResult(res);
      setState({kind: 'idle'});
    } catch (error: unknown) {
      setState({
        kind: 'error',
        message: getErrorMessage(error, 'Playback failed'),
      });
    }
  }, [tts, inputText, selectedVoice, selectedSpeed]);

  const handleBenchmark = useCallback(async () => {
    const sampleText = inputText.trim();

    if (!sampleText || !ttsRef.current) {
      return;
    }

    let lastResult: KittenTTSResult | null = null;

    try {
      setBenchmarkReport(null);
      setResult(null);
      const startedAt = new Date().toISOString();
      const rows = MODELS.map(model =>
        makeFailedBenchmarkRow(
          model,
          `Benchmark ${modelDisplayName(model)}`,
          'Benchmark did not run before the device session ended.',
          selectedVoice,
          selectedSpeed,
        ),
      );
      const publishReport = (finishedAt: string | null = null) => {
        setBenchmarkReport({
          schemaVersion: 1,
          sampleText,
          characterLength: Array.from(sampleText).length,
          voice: String(selectedVoice),
          voiceDisplayName: voiceDisplayName(selectedVoice),
          speed: selectedSpeed,
          startedAt,
          finishedAt,
          rows: [...rows],
        });
      };

      publishReport();

      for (let index = 0; index < MODELS.length; index += 1) {
        const model = MODELS[index];
        rows[index] = makeFailedBenchmarkRow(
          model,
          `Benchmark ${modelDisplayName(model)}`,
          'Benchmark started but did not finish before the device session ended.',
          selectedVoice,
          selectedSpeed,
        );
        publishReport();
        setState({
          kind: 'benchmarking',
          model: modelDisplayName(model),
          completed: index,
          total: MODELS.length,
        });

        const existingInstance =
          model === selectedModel ? ttsRef.current : null;
        let instance: KittenTTS | null = null;

        try {
          instance =
            existingInstance ??
            (await withTimeout(
              KittenTTS.create(
                {model, player: createRNSoundPlayer(Sound)},
                (progress, info) => {
                  if (mountedRef.current && info?.stage === 'downloading') {
                    setState({
                      kind: 'benchmarking',
                      model: `${modelDisplayName(model)} ${Math.round(
                        progress * 100,
                      )}%`,
                      completed: index,
                      total: MODELS.length,
                    });
                  }
                },
              ),
              BENCHMARK_MODEL_TIMEOUT_MS,
              `Timed out preparing ${modelDisplayName(model)}`,
            ));

          const firstRun = await measureGeneration(
            instance,
            sampleText,
            selectedVoice,
            selectedSpeed,
            `Timed out warming ${modelDisplayName(model)}`,
          );

          const measuredRuns: Array<{
            result: KittenTTSResult;
            generationMs: number;
          }> = [];

          for (let run = 0; run < BENCHMARK_WARM_RUNS; run += 1) {
            setState({
              kind: 'benchmarking',
              model: `${modelDisplayName(model)} run ${
                run + 1
              }/${BENCHMARK_WARM_RUNS}`,
              completed: index,
              total: MODELS.length,
            });
            measuredRuns.push(
              await measureGeneration(
                instance,
                sampleText,
                selectedVoice,
                selectedSpeed,
                `Timed out generating ${modelDisplayName(model)} warm run ${
                  run + 1
                }/${BENCHMARK_WARM_RUNS}`,
              ),
            );
          }

          const sortedWarmMs = measuredRuns
            .map(run => run.generationMs)
            .sort((a, b) => a - b);
          const bestRun = measuredRuns.reduce((best, candidate) =>
            candidate.generationMs < best.generationMs ? candidate : best,
          );
          const res = bestRun.result;
          const generationMs = bestRun.generationMs;
          const firstGenerationSeconds = firstRun.generationMs / 1000;
          const generationSeconds = generationMs / 1000;
          const durationSeconds = res.duration;
          const warmP50GenerationMs = percentile(sortedWarmMs, 50);
          const warmP95GenerationMs = percentile(sortedWarmMs, 95);
          const warmGenerationSeconds = measuredRuns.map(
            run => run.generationMs / 1000,
          );
          const warmRtf = warmGenerationSeconds.map(seconds =>
            durationSeconds > 0 ? seconds / durationSeconds : 0,
          );
          const wavBase64 = getWavBase64(res);

          rows[index] = {
            model: String(model),
            modelDisplayName: modelDisplayName(model),
            status: 'passed',
            voice: String(res.voice),
            voiceDisplayName: voiceDisplayName(res.voice),
            speed: res.effectiveSpeed,
            firstGenerationMs: firstRun.generationMs,
            firstGenerationSeconds,
            firstRtf:
              firstRun.result.duration > 0
                ? firstGenerationSeconds / firstRun.result.duration
                : 0,
            warmRunCount: BENCHMARK_WARM_RUNS,
            warmGenerationMs: measuredRuns.map(run => run.generationMs),
            warmGenerationSeconds,
            warmRtf,
            warmP50GenerationMs,
            warmP50GenerationSeconds: warmP50GenerationMs / 1000,
            warmP95GenerationMs,
            warmP95GenerationSeconds: warmP95GenerationMs / 1000,
            warmP50Rtf:
              durationSeconds > 0
                ? warmP50GenerationMs / 1000 / durationSeconds
                : 0,
            warmP95Rtf:
              durationSeconds > 0
                ? warmP95GenerationMs / 1000 / durationSeconds
                : 0,
            generationMs,
            generationSeconds,
            durationSeconds,
            rtf: durationSeconds > 0 ? generationSeconds / durationSeconds : 0,
            sampleCount: res.samples.length,
            sampleRate: res.sampleRate,
            sampleHash: computeSampleHash(res.samples),
            werReferenceText: sampleText,
            werAudioFormat: 'wav-base64',
            werAudioSampleRate: res.sampleRate,
            werAudioBase64: wavBase64,
            werAudioBase64Length: wavBase64?.length,
            werAudioChunkCount: wavBase64
              ? Math.ceil(wavBase64.length / WER_AUDIO_CHUNK_SIZE)
              : 0,
            werAudioChunkSize: WER_AUDIO_CHUNK_SIZE,
            parakeetStatus: 'pending',
          };
          publishReport();
          lastResult = res;
        } catch (error: unknown) {
          rows[index] = makeFailedBenchmarkRow(
            model,
            `Benchmark ${modelDisplayName(model)}`,
            getErrorMessage(error, 'Model benchmark failed'),
            selectedVoice,
            selectedSpeed,
          );
          publishReport();
        } finally {
          if (!existingInstance && instance) {
            await instance.dispose();
          }
        }
      }

      setResult(lastResult);
      publishReport(new Date().toISOString());
      setState({kind: 'idle'});
    } catch (error: unknown) {
      setState({
        kind: 'error',
        message: getErrorMessage(error, 'Benchmark failed'),
      });
    }
  }, [inputText, selectedModel, selectedSpeed, selectedVoice]);

  const handleModelChange = useCallback(
    (model: KittenModel) => {
      setSelectedModel(model);
      initTTS(model);
    },
    [initTTS],
  );

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.content}>
        <Text style={styles.title}>KittenTTS</Text>
        <Text style={styles.subtitle}>On-Device Text-to-Speech</Text>

        <StatusBanner state={state} />
        {benchmarkReport && <BenchmarkReportCard report={benchmarkReport} />}

        {/* Text Input */}
        <View style={styles.section}>
          <Text style={styles.label}>Text</Text>
          <TextInput
            testID="tts-input"
            accessibilityLabel="tts-input"
            style={styles.textInput}
            value={inputText}
            onChangeText={setInputText}
            multiline
            numberOfLines={4}
            editable={!isWorking}
            placeholder="Enter text to synthesise..."
            placeholderTextColor="#999"
          />
        </View>

        {/* Model Picker */}
        <View style={styles.section}>
          <Text style={styles.label}>Model</Text>
          <View style={styles.chipRow}>
            {MODELS.map(model => (
              <TouchableOpacity
                key={model}
                style={[
                  styles.chip,
                  selectedModel === model && styles.chipSelected,
                ]}
                onPress={() => handleModelChange(model)}
                disabled={isWorking}>
                <Text
                  style={[
                    styles.chipText,
                    selectedModel === model && styles.chipTextSelected,
                  ]}>
                  {modelDisplayName(model)}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>

        {/* Voice Picker */}
        <View style={styles.section}>
          <Text style={styles.label}>Voice</Text>
          <View style={styles.chipRow}>
            {ALL_VOICES.map(voice => (
              <TouchableOpacity
                key={voice}
                style={[
                  styles.chip,
                  selectedVoice === voice && styles.chipSelected,
                ]}
                onPress={() => setSelectedVoice(voice)}
                disabled={isWorking}>
                <Text
                  style={[
                    styles.chipText,
                    selectedVoice === voice && styles.chipTextSelected,
                  ]}>
                  {voiceDisplayName(voice)}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>

        {/* Speed Picker */}
        <View style={styles.section}>
          <Text style={styles.label}>Speed: {selectedSpeed.toFixed(1)}x</Text>
          <View style={styles.chipRow}>
            {SPEED_OPTIONS.map(speed => (
              <TouchableOpacity
                key={speed}
                style={[
                  styles.chip,
                  selectedSpeed === speed && styles.chipSelected,
                ]}
                onPress={() => setSelectedSpeed(speed)}
                disabled={isWorking}>
                <Text
                  style={[
                    styles.chipText,
                    selectedSpeed === speed && styles.chipTextSelected,
                  ]}>
                  {speed.toFixed(1)}x
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>

        {/* Action Buttons */}
        <View style={styles.buttonRow}>
          <TouchableOpacity
            testID="generate-button"
            accessibilityLabel="generate-button"
            style={[
              styles.button,
              styles.buttonPrimary,
              (isWorking || !inputText.trim() || !tts) && styles.buttonDisabled,
            ]}
            onPress={handleGenerate}
            disabled={isWorking || !inputText.trim() || !tts}>
            <Text style={styles.buttonPrimaryText}>Generate</Text>
          </TouchableOpacity>

          <TouchableOpacity
            testID="speak-button"
            accessibilityLabel="speak-button"
            style={[
              styles.button,
              styles.buttonSecondary,
              (isWorking || !inputText.trim() || !tts) && styles.buttonDisabled,
            ]}
            onPress={handleSpeak}
            disabled={isWorking || !inputText.trim() || !tts}>
            <Text style={styles.buttonSecondaryText}>Speak</Text>
          </TouchableOpacity>
        </View>

        <TouchableOpacity
          testID="benchmark-button"
          accessibilityLabel="benchmark-button"
          style={[
            styles.button,
            styles.buttonBenchmark,
            (isWorking || !inputText.trim() || !tts) && styles.buttonDisabled,
          ]}
          onPress={handleBenchmark}
          disabled={isWorking || !inputText.trim() || !tts}>
          <Text style={styles.buttonBenchmarkText}>Benchmark All Models</Text>
        </TouchableOpacity>

        {/* Result Card */}
        {result && <ResultCard result={result} />}
      </ScrollView>
    </SafeAreaView>
  );
}

function getErrorMessage(error: unknown, fallback: string): string {
  return error instanceof Error ? error.message : fallback;
}

function withTimeout<T>(
  promise: Promise<T>,
  timeoutMs: number,
  message: string,
): Promise<T> {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error(message)), timeoutMs);
    promise.then(
      value => {
        clearTimeout(timeout);
        resolve(value);
      },
      error => {
        clearTimeout(timeout);
        reject(error);
      },
    );
  });
}

async function measureGeneration(
  instance: KittenTTS,
  sampleText: string,
  voice: KittenVoice,
  speed: number,
  timeoutMessage: string,
): Promise<{result: KittenTTSResult; generationMs: number}> {
  const generationStartedAt = Date.now();
  const result = await withTimeout(
    instance.generate(sampleText, voice, speed),
    BENCHMARK_MODEL_TIMEOUT_MS,
    timeoutMessage,
  );
  return {
    result,
    generationMs: Date.now() - generationStartedAt,
  };
}

function percentile(sortedValues: number[], percentileValue: number): number {
  if (sortedValues.length === 0) return 0;
  const rank = Math.ceil((percentileValue / 100) * sortedValues.length) - 1;
  const index = Math.min(sortedValues.length - 1, Math.max(0, rank));
  return sortedValues[index];
}

function StatusBanner({state}: {state: AppState}) {
  switch (state.kind) {
    case 'idle':
      return null;
    case 'preparing':
      return (
        <View
          style={styles.banner}
          testID="status-banner"
          accessibilityLabel="status-banner">
          <ActivityIndicator size="small" color="#007AFF" />
          <Text
            style={styles.bannerText}
            {...e2eTextProps('status-label')}>
            Preparing model...
          </Text>
        </View>
      );
    case 'downloading':
      return (
        <View
          style={styles.banner}
          testID="status-banner"
          accessibilityLabel="status-banner">
          <ActivityIndicator size="small" color="#007AFF" />
          <Text
            style={styles.bannerText}
            {...e2eTextProps('status-label')}>
            Downloading model... {Math.round(state.progress * 100)}%
          </Text>
        </View>
      );
    case 'generating':
      return (
        <View
          style={styles.banner}
          testID="status-banner"
          accessibilityLabel="status-banner">
          <ActivityIndicator size="small" color="#007AFF" />
          <Text
            style={styles.bannerText}
            {...e2eTextProps('status-label')}>
            Generating speech...
          </Text>
        </View>
      );
    case 'benchmarking':
      return (
        <View
          style={styles.banner}
          testID="status-banner"
          accessibilityLabel="status-banner">
          <ActivityIndicator size="small" color="#007AFF" />
          <Text
            style={styles.bannerText}
            {...e2eTextProps('status-label')}>
            Benchmarking {state.model} ({state.completed + 1}/{state.total})...
          </Text>
        </View>
      );
    case 'playing':
      return (
        <View
          style={styles.banner}
          testID="status-banner"
          accessibilityLabel="status-banner">
          <ActivityIndicator size="small" color="#007AFF" />
          <Text
            style={styles.bannerText}
            {...e2eTextProps('status-label')}>
            Playing...
          </Text>
        </View>
      );
    case 'error':
      return (
        <View
          style={[styles.banner, styles.bannerError]}
          testID="error-banner"
          accessibilityLabel="error-banner">
          <Text
            style={styles.bannerErrorText}
            {...e2eTextProps('error-message')}>
            {state.message}
          </Text>
        </View>
      );
  }
}

function BenchmarkReportCard({report}: {report: BenchmarkReport}) {
  const displayReport = JSON.stringify(stripBenchmarkAudio(report));

  return (
    <View
      style={styles.resultCard}
      testID="benchmark-report"
      accessibilityLabel="benchmark-report">
      <Text style={styles.resultTitle}>Benchmark Report</Text>
      <View style={styles.resultRow}>
        <Text style={styles.resultLabel}>Sample Text</Text>
        <Text
          style={[styles.resultValue, styles.resultLongValue]}
          {...e2eTextProps('benchmark-sample-text')}>
          {report.sampleText}
        </Text>
      </View>
      <View style={styles.resultRow}>
        <Text style={styles.resultLabel}>Characters</Text>
        <Text
          style={styles.resultValue}
          {...e2eTextProps('benchmark-char-length')}>
          {report.characterLength}
        </Text>
      </View>
      <BenchmarkAudioChunks report={report} />
      {report.rows.map(row => (
        <View
          key={row.model}
          style={styles.benchmarkRow}
          testID={`benchmark-row-${row.model}`}
          accessibilityLabel={`benchmark-row-${row.model}`}>
          <Text style={styles.benchmarkModel}>{row.modelDisplayName}</Text>
          <Text style={styles.benchmarkMetric}>
            {row.status === 'passed'
              ? `Best ${((row.generationMs ?? 0) / 1000).toFixed(2)}s | p50 ${(
                  (row.warmP50GenerationMs ?? 0) / 1000
                ).toFixed(2)}s | p95 ${(
                  (row.warmP95GenerationMs ?? 0) / 1000
                ).toFixed(2)}s | RTF ${(row.rtf ?? 0).toFixed(3)}`
              : `Failed | ${row.failedStage}: ${row.errorSummary}`}
          </Text>
        </View>
      ))}
      <Text
        style={styles.benchmarkJson}
        {...e2eTextProps('benchmark-json-display')}
        selectable>
        {displayReport}
      </Text>
    </View>
  );
}

function BenchmarkAudioChunks({report}: {report: BenchmarkReport}) {
  return (
    <View style={styles.benchmarkAudioChunks} pointerEvents="none">
      {report.rows.flatMap(row => {
        if (row.status !== 'passed' || !row.werAudioBase64) {
          return [];
        }

        const rowSlug = automationSlug(row.model);
        const chunks = [];
        for (
          let offset = 0, index = 0;
          offset < row.werAudioBase64.length;
          offset += WER_AUDIO_CHUNK_SIZE, index += 1
        ) {
          chunks.push(
            <Text
              key={`${row.model}-${index}`}
              style={styles.benchmarkAudioChunk}
              {...e2eTextProps(`benchmark-audio-${rowSlug}-${index}`)}
              selectable>
              {row.werAudioBase64.slice(offset, offset + WER_AUDIO_CHUNK_SIZE)}
            </Text>,
          );
        }

        return chunks;
      })}
    </View>
  );
}

function stripBenchmarkAudio(report: BenchmarkReport): BenchmarkReport {
  return {
    ...report,
    rows: report.rows.map(row => {
      const {werAudioBase64: _werAudioBase64, ...rest} = row;
      return rest;
    }),
  };
}

function getWavBase64(result: KittenTTSResult): string | undefined {
  try {
    if (typeof result.wavBase64 === 'function') {
      return result.wavBase64();
    }
  } catch {
    return undefined;
  }

  return undefined;
}

function ResultCard({result}: {result: KittenTTSResult}) {
  return (
    <View
      style={styles.resultCard}
      testID="result-card"
      accessibilityLabel="result-card">
      <Text style={styles.resultTitle}>Generated Audio</Text>
      <View style={styles.resultRow}>
        <Text style={styles.resultLabel}>Voice</Text>
        <Text style={styles.resultValue}>{voiceDisplayName(result.voice)}</Text>
      </View>
      <View style={styles.resultRow}>
        <Text style={styles.resultLabel}>Duration</Text>
        <Text
          style={styles.resultValue}
          testID="duration"
          accessibilityLabel="duration">
          {result.duration.toFixed(2)}s
        </Text>
      </View>
      <View style={styles.resultRow}>
        <Text style={styles.resultLabel}>Samples</Text>
        <Text
          style={styles.resultValue}
          testID="sample-count"
          accessibilityLabel="sample-count">
          {result.samples.length.toLocaleString()}
        </Text>
      </View>
      <View style={styles.resultRow}>
        <Text style={styles.resultLabel}>Sample Rate</Text>
        <Text
          style={styles.resultValue}
          testID="sample-rate"
          accessibilityLabel="sample-rate">
          {result.sampleRate.toLocaleString()} Hz
        </Text>
      </View>
      <View style={styles.resultRow}>
        <Text style={styles.resultLabel}>Sample Hash</Text>
        <Text
          style={styles.resultValue}
          testID="sample-hash"
          accessibilityLabel="sample-hash">
          {computeSampleHash(result.samples)}
        </Text>
      </View>
    </View>
  );
}

function computeSampleHash(samples: ArrayLike<number>): string {
  let hash = 2166136261;

  for (let index = 0; index < samples.length; index += 1) {
    const pcm16 = Math.max(
      -32768,
      Math.min(32767, Math.round(samples[index] * 32767)),
    );
    hash ^= pcm16 & 0xff;
    hash = Math.imul(hash, 16777619);
    hash ^= (pcm16 >> 8) & 0xff;
    hash = Math.imul(hash, 16777619);
  }

  return (hash >>> 0).toString(16).padStart(8, '0');
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F2F2F7',
  },
  content: {
    padding: 20,
    paddingBottom: 40,
  },
  title: {
    fontSize: 28,
    fontWeight: '700',
    color: '#000',
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    marginBottom: 20,
  },
  section: {
    marginBottom: 16,
  },
  label: {
    fontSize: 13,
    fontWeight: '600',
    color: '#666',
    marginBottom: 6,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  textInput: {
    backgroundColor: '#FFF',
    borderRadius: 12,
    padding: 14,
    fontSize: 16,
    color: '#000',
    minHeight: 100,
    textAlignVertical: 'top',
    ...Platform.select({
      ios: {
        shadowColor: '#000',
        shadowOffset: {width: 0, height: 1},
        shadowOpacity: 0.05,
        shadowRadius: 3,
      },
      android: {
        elevation: 1,
      },
    }),
  },
  chipRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  chip: {
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 20,
    backgroundColor: '#FFF',
    borderWidth: 1,
    borderColor: '#E0E0E0',
  },
  chipSelected: {
    backgroundColor: '#007AFF',
    borderColor: '#007AFF',
  },
  chipText: {
    fontSize: 13,
    fontWeight: '500',
    color: '#333',
  },
  chipTextSelected: {
    color: '#FFF',
  },
  buttonRow: {
    flexDirection: 'row',
    gap: 12,
    marginTop: 8,
    marginBottom: 16,
  },
  button: {
    flex: 1,
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: 'center',
  },
  buttonPrimary: {
    backgroundColor: '#007AFF',
  },
  buttonSecondary: {
    backgroundColor: '#FFF',
    borderWidth: 1,
    borderColor: '#007AFF',
  },
  buttonBenchmark: {
    backgroundColor: '#111827',
    marginBottom: 16,
  },
  buttonDisabled: {
    opacity: 0.5,
  },
  buttonPrimaryText: {
    color: '#FFF',
    fontSize: 16,
    fontWeight: '600',
  },
  buttonSecondaryText: {
    color: '#007AFF',
    fontSize: 16,
    fontWeight: '600',
  },
  buttonBenchmarkText: {
    color: '#FFF',
    fontSize: 16,
    fontWeight: '600',
  },
  banner: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#F0F4FF',
    padding: 12,
    borderRadius: 12,
    marginBottom: 16,
    gap: 10,
  },
  bannerText: {
    fontSize: 14,
    color: '#007AFF',
  },
  bannerError: {
    backgroundColor: '#FFF0F0',
  },
  bannerErrorText: {
    fontSize: 14,
    color: '#FF3B30',
  },
  resultCard: {
    backgroundColor: '#FFF',
    borderRadius: 12,
    padding: 16,
    ...Platform.select({
      ios: {
        shadowColor: '#000',
        shadowOffset: {width: 0, height: 2},
        shadowOpacity: 0.08,
        shadowRadius: 4,
      },
      android: {
        elevation: 2,
      },
    }),
  },
  resultTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#34C759',
    marginBottom: 12,
  },
  resultRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 4,
  },
  resultLabel: {
    fontSize: 14,
    color: '#666',
  },
  resultValue: {
    fontSize: 14,
    fontWeight: '500',
    color: '#000',
  },
  resultLongValue: {
    flex: 1,
    marginLeft: 12,
    textAlign: 'right',
  },
  benchmarkRow: {
    borderTopWidth: 1,
    borderTopColor: '#E5E5EA',
    paddingTop: 10,
    marginTop: 10,
  },
  benchmarkModel: {
    fontSize: 14,
    fontWeight: '700',
    color: '#111827',
  },
  benchmarkMetric: {
    color: '#4B5563',
    fontSize: 13,
    marginTop: 2,
  },
  benchmarkJson: {
    color: '#6B7280',
    fontSize: 10,
    marginTop: 12,
  },
  benchmarkAudioChunks: {
    marginTop: 1,
  },
  benchmarkAudioChunk: {
    color: '#FFFFFF',
    fontSize: 1,
    height: 1,
    opacity: 0.01,
  },
});
