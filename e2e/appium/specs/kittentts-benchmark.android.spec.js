const fs = require("node:fs");
const path = require("node:path");

const EXPECTED_MODELS = [
  "kitten-tts-nano-0.8",
  "kitten-tts-nano-0.8-int8",
  "kitten-tts-micro-0.8",
  "kitten-tts-mini-0.8",
];
const BENCHMARK_REPORT_TIMEOUT_MS = Number(
  process.env.TESTMU_BENCHMARK_REPORT_TIMEOUT_MS || 9 * 60 * 1000
);
const APP_READY_TIMEOUT_MS = Number(
  process.env.TESTMU_APP_READY_TIMEOUT_MS || 6 * 60 * 1000
);

function slugify(value) {
  return String(value || "device")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
}

function parseBenchmarkJson(rawText) {
  const jsonStart = rawText.indexOf("{");
  const jsonEnd = rawText.lastIndexOf("}");

  if (jsonStart === -1 || jsonEnd === -1 || jsonEnd <= jsonStart) {
    throw new Error(`benchmark-json did not contain JSON: ${rawText}`);
  }

  return JSON.parse(rawText.slice(jsonStart, jsonEnd + 1));
}

function hasFinishedBenchmark(report) {
  return Boolean(report?.finishedAt);
}

async function readBenchmarkReport(accessibilityId) {
  try {
    const reportText = await $(`~${accessibilityId}`).getText();
    return parseBenchmarkJson(reportText);
  } catch {
    return null;
  }
}

async function getBenchmarkReportFromUi({ includeAudio = false } = {}) {
  if (includeAudio) {
    return (
      (await readBenchmarkReport("benchmark-json-with-audio")) ||
      (await readBenchmarkReport("benchmark-json"))
    );
  }

  return readBenchmarkReport("benchmark-json");
}

function markPartialReport(report, timeoutMessage) {
  return {
    ...report,
    status: "partial",
    finishedAt: report.finishedAt || null,
    rows: (report.rows || []).map((row) => {
      if (row.status !== "failed") {
        return row;
      }

      const summary = String(row.errorSummary || "");
      if (
        !/did not run|did not finish|in progress|session ended/i.test(summary)
      ) {
        return row;
      }

      return {
        ...row,
        failedStage: row.failedStage || "Benchmark timeout",
        errorSummary: `${summary} ${timeoutMessage}`.trim(),
      };
    }),
  };
}

function getDeviceName() {
  return (
    process.env.TESTMU_DEVICE ||
    process.env.TESTMU_ANDROID_DEVICE ||
    process.env.TESTMU_IOS_DEVICE ||
    "Pixel 5"
  );
}

function getPlatformName() {
  return (
    process.env.TESTMU_PLATFORM_NAME ||
    (process.env.TESTMU_IOS_DEVICE ? "iOS" : "Android")
  );
}

function getPlatformVersion() {
  return (
    process.env.TESTMU_PLATFORM_VERSION ||
    process.env.TESTMU_ANDROID_VERSION ||
    process.env.TESTMU_IOS_VERSION ||
    "12"
  );
}

function writeDeviceReport(report, startedAtMs) {
  const device = getDeviceName();
  const finishedAtMs = Date.now();
  const outputDir = path.resolve(__dirname, "..", "reports");
  fs.mkdirSync(outputDir, { recursive: true });

  const payload = {
    device,
    platformName: getPlatformName(),
    platformVersion: getPlatformVersion(),
    realDevice: process.env.TESTMU_REAL_DEVICE !== "false",
    sessionId: browser.sessionId,
    githubRunId: process.env.GITHUB_RUN_ID || null,
    githubSha: process.env.GITHUB_SHA || null,
    deviceStartedAt: new Date(startedAtMs).toISOString(),
    deviceFinishedAt: new Date(finishedAtMs).toISOString(),
    totalRuntimeMs: finishedAtMs - startedAtMs,
    totalRuntimeSeconds: Number(
      ((finishedAtMs - startedAtMs) / 1000).toFixed(3)
    ),
    capturedAt: new Date().toISOString(),
    ...report,
  };

  const outputPath = path.join(outputDir, `${slugify(device)}.json`);
  fs.writeFileSync(outputPath, `${JSON.stringify(payload, null, 2)}\n`);
}

async function getOptionalText(accessibilityId) {
  try {
    const element = await $(`~${accessibilityId}`);
    if (await element.isDisplayed()) {
      return await element.getText();
    }
  } catch {
    return null;
  }

  return null;
}

async function isDisplayed(accessibilityId) {
  try {
    const element = await $(`~${accessibilityId}`);
    return await element.isDisplayed();
  } catch {
    return false;
  }
}

async function getPageSourceSummary() {
  try {
    const source = await browser.getPageSource();
    return source
      .replace(/\s+/g, " ")
      .slice(0, 2000);
  } catch (error) {
    return `Could not read page source: ${error.message}`;
  }
}

async function waitForAppReady(timeoutMs) {
  const startedAt = Date.now();
  let lastStatus = "No app status captured yet.";

  while (Date.now() - startedAt < timeoutMs) {
    const errorMessage = await getOptionalText("error-message");
    if (errorMessage) {
      throw new Error(`App showed error-banner before benchmark: ${errorMessage}`);
    }

    const statusLabel = await getOptionalText("status-label");
    if (statusLabel && statusLabel !== lastStatus) {
      lastStatus = statusLabel;
      console.log(`[KittenTTS app status] ${statusLabel}`);
    }

    const inputVisible = await isDisplayed("tts-input");
    const benchmark = await $("~benchmark-button");
    if (inputVisible && (await benchmark.isEnabled().catch(() => false))) {
      return benchmark;
    }

    await browser.pause(5000);
  }

  const sourceSummary = await getPageSourceSummary();
  throw new Error(
    `Timed out waiting for app readiness after ${timeoutMs}ms. Last app status: ${lastStatus}. Page source: ${sourceSummary}`
  );
}

async function waitForBenchmarkReport(timeoutMs) {
  const startedAt = Date.now();
  let lastStatus = "No app status captured yet.";
  let lastReport = null;

  while (Date.now() - startedAt < timeoutMs) {
    const report = await getBenchmarkReportFromUi();
    if (report) {
      lastReport = report;
      if (hasFinishedBenchmark(report)) {
        return (await getBenchmarkReportFromUi({ includeAudio: true })) || report;
      }
    }

    const errorMessage = await getOptionalText("error-message");
    if (errorMessage) {
      throw new Error(`App showed error-banner: ${errorMessage}`);
    }

    const statusLabel = await getOptionalText("status-label");
    if (statusLabel && statusLabel !== lastStatus) {
      lastStatus = statusLabel;
      console.log(`[KittenTTS benchmark status] ${statusLabel}`);
    }

    await browser.pause(5000);
  }

  const timeoutMessage = `Timed out waiting for benchmark-report. Last app status: ${lastStatus}`;
  if (lastReport) {
    return markPartialReport(lastReport, timeoutMessage);
  }

  throw new Error(timeoutMessage);
}

describe("KittenTTS React Native benchmark", () => {
  it("benchmarks every bundled model and writes a device report", async () => {
    const deviceStartedAtMs = Date.now();
    const benchmark = await waitForAppReady(APP_READY_TIMEOUT_MS);
    const input = await $("~tts-input");

    const sampleText = process.env.TESTMU_SAMPLE_TEXT;
    if (sampleText) {
      const currentText = await input.getText().catch(() => "");
      if (currentText !== sampleText) {
        try {
          await input.click();
          await input.clearValue();
          await input.setValue(sampleText);
        } catch (error) {
          console.warn(
            `[KittenTTS benchmark] Could not override sample text; continuing with the app default. ${error.message}`
          );
        }
      }
    }

    await benchmark.click();

    const report = await waitForBenchmarkReport(BENCHMARK_REPORT_TIMEOUT_MS);

    expect(report.schemaVersion).toBe(1);
    expect(report.sampleText.length).toBeGreaterThan(0);
    expect(report.characterLength).toBeGreaterThan(0);
    expect(report.rows.length).toBe(EXPECTED_MODELS.length);

    for (const expectedModel of EXPECTED_MODELS) {
      const row = report.rows.find(
        (candidate) => candidate.model === expectedModel
      );

      if (!row) {
        throw new Error(`Missing benchmark row for model: ${expectedModel}`);
      }
      if (!["passed", "failed"].includes(row.status)) {
        throw new Error(
          `Invalid status for ${expectedModel}: ${String(row.status)}`
        );
      }

      if (row.status === "passed") {
        expect(row.generationMs).toBeGreaterThan(0);
        expect(row.generationSeconds).toBeGreaterThan(0);
        expect(row.firstGenerationMs).toBeGreaterThan(0);
        expect(row.firstGenerationSeconds).toBeGreaterThan(0);
        expect(row.warmRunCount).toBe(5);
        expect(row.warmGenerationMs.length).toBe(5);
        expect(row.warmRtf.length).toBe(5);
        expect(row.warmP50GenerationMs).toBeGreaterThan(0);
        expect(row.warmP95GenerationMs).toBeGreaterThan(0);
        expect(row.warmP50Rtf).toBeGreaterThan(0);
        expect(row.warmP95Rtf).toBeGreaterThan(0);
        expect(row.durationSeconds).toBeGreaterThan(0);
        expect(row.rtf).toBeGreaterThan(0);
        expect(row.sampleCount).toBeGreaterThan(0);
        expect(row.sampleRate).toBe(24000);
        if (!/^[0-9a-f]{8}$/.test(row.sampleHash)) {
          throw new Error(
            `Invalid sample hash for ${expectedModel}: ${row.sampleHash}`
          );
        }
        if (process.env.TESTMU_REQUIRE_WER_AUDIO === "true") {
          expect(row.werReferenceText).toBe(report.sampleText);
          expect(row.werAudioFormat).toBe("wav-base64");
          expect(row.werAudioSampleRate).toBe(24000);
          expect(row.werAudioBase64.length).toBeGreaterThan(1000);
          expect(row.parakeetStatus).toBe("pending");
        }
      } else {
        expect(row.failedStage.length).toBeGreaterThan(0);
        expect(row.errorSummary.length).toBeGreaterThan(0);
      }
    }

    writeDeviceReport(report, deviceStartedAtMs);
  });
});
