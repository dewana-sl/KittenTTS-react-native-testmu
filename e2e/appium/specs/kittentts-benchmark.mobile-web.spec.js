const fs = require("node:fs");
const path = require("node:path");

const EXPECTED_MODELS = [
  "kitten-tts-nano-0.8",
  "kitten-tts-nano-0.8-int8",
  "kitten-tts-micro-0.8",
  "kitten-tts-mini-0.8",
].filter((model) => {
  const expectedModels = process.env.TESTMU_EXPECTED_MODELS;
  if (!expectedModels) return true;
  return expectedModels
    .split(",")
    .map((value) => value.trim())
    .includes(model);
});
const EXPECTED_WARM_RUNS = Number(
  process.env.TESTMU_EXPECTED_WARM_RUNS || 5
);
const BENCHMARK_REPORT_TIMEOUT_MS = Number(
  process.env.TESTMU_BENCHMARK_REPORT_TIMEOUT_MS || 30 * 60 * 1000
);
const WEB_READY_TIMEOUT_MS = Number(
  process.env.TESTMU_WEB_READY_TIMEOUT_MS || 6 * 60 * 1000
);

function slugify(value) {
  return String(value || "device")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
}

function automationSlug(value) {
  return slugify(value);
}

function testIdSelector(testId) {
  return `[data-testid="${testId}"], [aria-label="${testId}"]`;
}

async function findByTestId(testId) {
  return $(testIdSelector(testId));
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

async function readElementText(testId) {
  const element = await findByTestId(testId);
  const directText = await element.getText().catch(() => "");
  if (directText && directText !== testId) {
    return directText;
  }

  const domText = await browser
    .execute((node) => {
      if (!node) return "";
      return (
        node.textContent || node.value || node.getAttribute("aria-label") || ""
      );
    }, element)
    .catch(() => "");

  return String(domText || "");
}

async function readBenchmarkReport(testId) {
  const webState = await getWebBenchmarkState();
  if (webState?.report) {
    return webState.report;
  }

  const globalReport = await browser
    .execute(() => globalThis.__KITTEN_BENCHMARK_REPORT__ || null)
    .catch(() => null);

  if (globalReport) {
    return globalReport;
  }

  try {
    const reportText = await readElementText(testId);
    return parseBenchmarkJson(reportText);
  } catch {
    return null;
  }
}

async function getWebBenchmarkState() {
  return browser
    .execute(() => globalThis.__KITTEN_BENCHMARK_STATE__ || null)
    .catch(() => null);
}

async function attachWerAudioChunksDirect(report) {
  const rows = [];

  for (const row of report.rows || []) {
    if (row.status !== "passed") {
      rows.push(row);
      continue;
    }

    const chunkCount = Number(row.werAudioChunkCount || 0);
    if (chunkCount <= 0) {
      rows.push(row);
      continue;
    }

    const rowSlug = automationSlug(row.model);
    const chunks = [];
    for (let index = 0; index < chunkCount; index += 1) {
      const testId = `benchmark-audio-${rowSlug}-${index}`;
      const chunk = await readElementText(testId);
      if (!chunk) {
        throw new Error(
          `Missing WER audio chunk ${index + 1}/${chunkCount} for ${
            row.model
          } (${testId}).`
        );
      }
      chunks.push(chunk);
    }

    const werAudioBase64 = chunks.join("");
    if (
      Number.isFinite(row.werAudioBase64Length) &&
      werAudioBase64.length !== row.werAudioBase64Length
    ) {
      throw new Error(
        `WER audio length mismatch for ${row.model}: expected ${row.werAudioBase64Length}, got ${werAudioBase64.length}.`
      );
    }

    rows.push({
      ...row,
      werAudioBase64,
    });
  }

  return {
    ...report,
    rows,
  };
}

async function getBenchmarkReportFromUi({ includeAudio = false } = {}) {
  const report = await readBenchmarkReport("benchmark-json-display");

  if (!report || !includeAudio) {
    return report;
  }

  return attachWerAudioChunksDirect(report);
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
  return process.env.TESTMU_DEVICE || "Pixel 8";
}

function getPlatformName() {
  return (
    browser?.capabilities?.platformName ||
    browser?.requestedCapabilities?.platformName ||
    process.env.TESTMU_PLATFORM_NAME ||
    "Android"
  );
}

function getCapabilityPlatformVersion() {
  return (
    browser?.capabilities?.platformVersion ||
    browser?.capabilities?.platform_version ||
    browser?.capabilities?.osVersion ||
    browser?.capabilities?.os_version ||
    browser?.requestedCapabilities?.platformVersion ||
    browser?.requestedCapabilities?.platform_version ||
    null
  );
}

function getPlatformVersion() {
  return (
    getCapabilityPlatformVersion() ||
    process.env.TESTMU_PLATFORM_VERSION ||
    null
  );
}

function getBrowserName() {
  return (
    browser?.capabilities?.browserName ||
    browser?.requestedCapabilities?.browserName ||
    process.env.TESTMU_BROWSER_NAME ||
    null
  );
}

function getBrowserVersion() {
  return (
    browser?.capabilities?.browserVersion ||
    browser?.capabilities?.browser_version ||
    browser?.requestedCapabilities?.browserVersion ||
    browser?.requestedCapabilities?.browser_version ||
    null
  );
}

function writeDeviceReport(report, startedAtMs) {
  const device = getDeviceName();
  const browserName = getBrowserName();
  const finishedAtMs = Date.now();
  const outputDir = path.resolve(__dirname, "..", "reports");
  fs.mkdirSync(outputDir, { recursive: true });

  const payload = {
    target: "web",
    device,
    platformName: getPlatformName(),
    platformVersion: getPlatformVersion(),
    requestedPlatformVersion: process.env.TESTMU_PLATFORM_VERSION || null,
    browserName,
    browserVersion: getBrowserVersion(),
    requestedBrowserName: process.env.TESTMU_BROWSER_NAME || null,
    webUrl: process.env.TESTMU_WEB_URL || null,
    tunnelName: process.env.TESTMU_TUNNEL_NAME || null,
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

  const outputPath = path.join(
    outputDir,
    `web-${slugify(device)}-${slugify(browserName || "browser")}.json`
  );
  fs.writeFileSync(outputPath, `${JSON.stringify(payload, null, 2)}\n`);
}

async function getOptionalText(testId) {
  try {
    const element = await findByTestId(testId);
    if (await element.isDisplayed()) {
      return await readElementText(testId);
    }
  } catch {
    return null;
  }

  return null;
}

async function isDisplayed(testId) {
  try {
    const element = await findByTestId(testId);
    return await element.isDisplayed();
  } catch {
    return false;
  }
}

async function getPageSourceSummary() {
  try {
    const source = await browser.getPageSource();
    return source.replace(/\s+/g, " ").slice(0, 2000);
  } catch (error) {
    return `Could not read page source: ${error.message}`;
  }
}

async function getBrowserDiagnostics() {
  const [state, title, url, bodyText, sourceSummary] = await Promise.all([
    getWebBenchmarkState(),
    browser.getTitle().catch((error) => `Could not read title: ${error.message}`),
    browser.getUrl().catch((error) => `Could not read URL: ${error.message}`),
    browser
      .execute(() => document.body?.innerText || document.body?.textContent || "")
      .catch((error) => `Could not read body text: ${error.message}`),
    getPageSourceSummary(),
  ]);

  return {
    title,
    url,
    state,
    bodyText: String(bodyText || "").replace(/\s+/g, " ").slice(0, 1200),
    sourceSummary,
  };
}

function summarizeDiagnostics(diagnostics) {
  return JSON.stringify(diagnostics, null, 2).slice(0, 5000);
}

async function waitForWebAppLoaded(timeoutMs) {
  const startedAt = Date.now();
  let lastState = null;

  while (Date.now() - startedAt < timeoutMs) {
    const state = await getWebBenchmarkState();
    if (state) {
      lastState = state;

      if (state.error) {
        throw new Error(`Web app error before benchmark: ${state.error}`);
      }

      if (state.appMounted || state.ready || state.report) {
        return state;
      }
    }

    await browser.pause(2000);
  }

  const diagnostics = await getBrowserDiagnostics();
  throw new Error(
    `Timed out waiting for web app load after ${timeoutMs}ms. Last bridge state: ${JSON.stringify(
      lastState
    )}. Diagnostics: ${summarizeDiagnostics(diagnostics)}`
  );
}

async function waitForWebReady(timeoutMs) {
  const startedAt = Date.now();
  let lastStatus = "No web status captured yet.";

  while (Date.now() - startedAt < timeoutMs) {
    const state = await getWebBenchmarkState();
    if (state) {
      if (state.error) {
        throw new Error(`Web app error before benchmark: ${state.error}`);
      }
      if (state.status && state.status !== lastStatus) {
        lastStatus = state.status;
        console.log(`[KittenTTS web status] ${state.status}`);
      }
      if (state.ready) {
        return findByTestId("benchmark-button");
      }
    }

    const errorMessage = await getOptionalText("error-message");
    if (errorMessage) {
      throw new Error(
        `Web app showed error-banner before benchmark: ${errorMessage}`
      );
    }

    const statusLabel = await getOptionalText("status-label");
    if (statusLabel && statusLabel !== lastStatus) {
      lastStatus = statusLabel;
      console.log(`[KittenTTS web status] ${statusLabel}`);
    }

    const inputVisible = await isDisplayed("tts-input");
    const benchmark = await findByTestId("benchmark-button");
    if (inputVisible && (await benchmark.isEnabled().catch(() => false))) {
      return benchmark;
    }

    await browser.pause(5000);
  }

  const diagnostics = await getBrowserDiagnostics();
  throw new Error(
    `Timed out waiting for web app readiness after ${timeoutMs}ms. Last web status: ${lastStatus}. Diagnostics: ${summarizeDiagnostics(
      diagnostics
    )}`
  );
}

async function waitForBenchmarkReport(timeoutMs) {
  const startedAt = Date.now();
  let lastStatus = "No benchmark status captured yet.";
  let lastReport = null;

  while (Date.now() - startedAt < timeoutMs) {
    const report = await getBenchmarkReportFromUi();
    if (report) {
      lastReport = report;
      if (hasFinishedBenchmark(report)) {
        const includeAudio = process.env.TESTMU_REQUIRE_WER_AUDIO !== "false";
        return (await getBenchmarkReportFromUi({ includeAudio })) || report;
      }
    }

    const state = await getWebBenchmarkState();
    if (state?.error) {
      throw new Error(`Web app benchmark error: ${state.error}`);
    }
    if (state?.status && state.status !== lastStatus) {
      lastStatus = state.status;
      console.log(`[KittenTTS web benchmark status] ${state.status}`);
    }

    const globalError = await browser
      .execute(() => globalThis.__KITTEN_BENCHMARK_ERROR__ || "")
      .catch(() => "");
    if (globalError) {
      throw new Error(`Web app benchmark error: ${globalError}`);
    }

    const errorMessage = await getOptionalText("error-message");
    if (errorMessage) {
      throw new Error(`Web app showed error-banner: ${errorMessage}`);
    }

    const statusLabel = await getOptionalText("status-label");
    if (statusLabel && statusLabel !== lastStatus) {
      lastStatus = statusLabel;
      console.log(`[KittenTTS web benchmark status] ${statusLabel}`);
    }

    await browser.pause(5000);
  }

  const timeoutMessage = `Timed out waiting for benchmark-report. Last web status: ${lastStatus}`;
  if (lastReport) {
    return markPartialReport(lastReport, timeoutMessage);
  }

  const diagnostics = await getBrowserDiagnostics();
  throw new Error(
    `${timeoutMessage}. Diagnostics: ${summarizeDiagnostics(diagnostics)}`
  );
}

describe("KittenTTS React Native web benchmark", () => {
  it("benchmarks every bundled model in a real mobile browser", async () => {
    const deviceStartedAtMs = Date.now();
    const webUrl = process.env.TESTMU_WEB_URL;
    const autoBenchmark = /benchmarkAutoStart=true/.test(webUrl);

    await browser.url(webUrl);
    const loadedState = await waitForWebAppLoaded(WEB_READY_TIMEOUT_MS);

    let benchmark = null;
    if (!autoBenchmark) {
      benchmark = await waitForWebReady(WEB_READY_TIMEOUT_MS);
    }

    const sampleText = process.env.TESTMU_SAMPLE_TEXT;
    if (sampleText && !autoBenchmark) {
      const input = await findByTestId("tts-input");
      const currentText = await readElementText("tts-input");
      if (currentText !== sampleText) {
        try {
          await input.click();
          await input.clearValue();
          await input.setValue(sampleText);
        } catch (error) {
          console.warn(
            `[KittenTTS web benchmark] Could not override sample text; continuing with the app default. ${error.message}`
          );
        }
      }
    }

    if (process.env.TESTMU_WEB_SMOKE_ONLY === "true") {
      const currentText = await readElementText("tts-input");
      const reportSampleText = sampleText || currentText || "";
      writeDeviceReport(
        {
          schemaVersion: 1,
          smokeOnly: true,
          status: "passed",
          startedAt: new Date(deviceStartedAtMs).toISOString(),
          finishedAt: new Date().toISOString(),
          sampleText: reportSampleText,
          characterLength: reportSampleText.length,
          rows: [],
        },
        deviceStartedAtMs
      );
      return;
    }

    if (!autoBenchmark) {
      await benchmark.click();
    } else if (
      loadedState?.ready &&
      loadedState?.autoStart &&
      !loadedState?.autoStartTriggered
    ) {
      console.warn(
        "[KittenTTS web benchmark] Auto-start was requested but not triggered after app load; clicking benchmark button as a fallback."
      );
      benchmark = await findByTestId("benchmark-button");
      await benchmark.click();
    }

    const report = await waitForBenchmarkReport(BENCHMARK_REPORT_TIMEOUT_MS);

    expect(report.schemaVersion).toBe(1);
    expect(report.sampleText.length).toBeGreaterThan(0);
    expect(report.characterLength).toBeGreaterThan(0);
    expect(report.rows.length).toBe(EXPECTED_MODELS.length);

    const failedRows = [];

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
        expect(row.warmRunCount).toBe(EXPECTED_WARM_RUNS);
        expect(row.warmGenerationMs.length).toBe(EXPECTED_WARM_RUNS);
        expect(row.warmRtf.length).toBe(EXPECTED_WARM_RUNS);
        expect(row.warmP50GenerationMs).toBeGreaterThan(0);
        expect(row.warmP95GenerationMs).toBeGreaterThan(0);
        expect(row.durationSeconds).toBeGreaterThan(0);
        expect(row.rtf).toBeGreaterThan(0);
        expect(row.sampleCount).toBeGreaterThan(0);
        expect(row.sampleRate).toBeGreaterThan(0);
        expect(row.sampleHash.length).toBeGreaterThan(0);

        if (process.env.TESTMU_REQUIRE_WER_AUDIO !== "false") {
          expect(row.werAudioFormat).toBe("wav-base64");
          expect(row.werAudioBase64.length).toBeGreaterThan(1000);
        }
      } else if (!row.errorSummary) {
        throw new Error(
          `Failed row for ${expectedModel} did not include an error summary.`
        );
      } else {
        failedRows.push(row);
      }
    }

    writeDeviceReport(report, deviceStartedAtMs);

    if (failedRows.length > 0) {
      throw new Error(
        `Benchmark completed with ${failedRows.length} failed model row(s): ${failedRows
          .map((row) => `${row.model}: ${row.errorSummary}`)
          .join("; ")}`
      );
    }
  });
});
