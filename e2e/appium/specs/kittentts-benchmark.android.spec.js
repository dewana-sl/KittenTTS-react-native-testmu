const fs = require("node:fs");
const path = require("node:path");

const EXPECTED_MODELS = [
  "kitten-tts-nano-0.8",
  "kitten-tts-nano-0.8-int8",
  "kitten-tts-micro-0.8",
  "kitten-tts-mini-0.8",
];
const BENCHMARK_REPORT_TIMEOUT_MS = Number(
  process.env.TESTMU_BENCHMARK_REPORT_TIMEOUT_MS || 30 * 60 * 1000
);
const APP_READY_TIMEOUT_MS = Number(
  process.env.TESTMU_APP_READY_TIMEOUT_MS || 6 * 60 * 1000
);
const ANDROID_APP_PACKAGE =
  process.env.TESTMU_ANDROID_APP_PACKAGE || "com.basicexample";

function slugify(value) {
  return String(value || "device")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
}

function automationSlug(value) {
  return slugify(value);
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

function isIosSession() {
  return /ios/i.test(
    String(
      browser?.capabilities?.platformName ||
        browser?.requestedCapabilities?.platformName ||
        getPlatformName()
    )
  );
}

function isAndroidSession() {
  return /android/i.test(
    String(
      browser?.capabilities?.platformName ||
        browser?.requestedCapabilities?.platformName ||
        getPlatformName()
    )
  );
}

async function activateAndroidBenchmarkApp() {
  if (!isAndroidSession() || typeof browser.activateApp !== "function") {
    return;
  }

  try {
    await browser.activateApp(ANDROID_APP_PACKAGE);
  } catch (error) {
    console.warn(
      `[KittenTTS benchmark] Could not activate ${ANDROID_APP_PACKAGE}: ${error.message}`
    );
  }
}

async function dismissAndroidSystemDialog() {
  if (!isAndroidSession()) {
    return false;
  }

  const selectors = [
    "id=android:id/button3",
    "id=android:id/button1",
    "id=android:id/button2",
    'android=new UiSelector().textMatches("(?i)^(OK|Got it|Close|Dismiss)$")',
  ];

  for (const selector of selectors) {
    try {
      const element = await $(selector);
      if (await element.isDisplayed()) {
        const text = (await element.getText().catch(() => selector)) || selector;
        console.log(`[KittenTTS benchmark] Dismissing Android dialog: ${text}`);
        await element.click();
        await browser.pause(1000);
        await activateAndroidBenchmarkApp();
        return true;
      }
    } catch {
      // Most devices will not have a blocking system dialog. Keep polling.
    }
  }

  return false;
}

function usableElementText(candidate, accessibilityId) {
  const text = String(candidate || "");
  return text.length > 0 && text !== accessibilityId ? text : "";
}

async function readElementText(accessibilityId) {
  const element = await $(`~${accessibilityId}`);

  const firstText = usableElementText(
    await element.getText().catch(() => ""),
    accessibilityId
  );

  if (firstText) {
    return firstText;
  }

  const attributeNames = isIosSession()
    ? ["label", "value", "name"]
    : ["text", "label", "value"];

  for (const attributeName of attributeNames) {
    const attributeText = usableElementText(
      await element.getAttribute(attributeName).catch(() => ""),
      accessibilityId
    );

    if (attributeText) {
      return attributeText;
    }
  }

  return "";
}

function shouldUseAudioPager(report) {
  const mode = String(
    process.env.TESTMU_AUDIO_EXTRACTION_MODE || "auto"
  ).toLowerCase();
  if (mode === "pager") return true;
  if (mode === "direct") return false;
  if (isIosSession()) return false;

  const version = Number.parseFloat(
    String(
      report?.platformVersion ||
        browser?.capabilities?.platformVersion ||
        browser?.requestedCapabilities?.platformVersion ||
        process.env.TESTMU_ANDROID_VERSION ||
        ""
    )
  );

  return !Number.isFinite(version) || version < 12;
}

function countExpectedAudioChunks(report) {
  return (report.rows || []).reduce(
    (total, row) =>
      row.status === "passed" ? total + Number(row.werAudioChunkCount || 0) : total,
    0
  );
}

async function attachWerAudioChunks(report) {
  const rows = [];
  const usePager = shouldUseAudioPager(report);
  const extractionMode = usePager ? "pager" : "direct";

  console.log(
    `Using ${extractionMode} WER audio extraction for ${report.device || "device"} ` +
      `(${report.platformName || "platform"} ${report.platformVersion || "unknown"}), ` +
      `${countExpectedAudioChunks(report)} chunk(s).`
  );

  if (usePager) {
    return attachWerAudioChunksFromPager(report);
  }

  try {
    return await attachWerAudioChunksDirect(report);
  } catch (error) {
    if (
      isIosSession() ||
      process.env.TESTMU_AUDIO_EXTRACTION_MODE === "direct"
    ) {
      throw error;
    }

    console.warn(
      `Direct WER audio extraction failed, falling back to pager: ${error.message}`
    );
    return attachWerAudioChunksFromPager(report);
  }
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
      const accessibilityId = `benchmark-audio-${rowSlug}-${index}`;
      const chunk = await readElementText(accessibilityId);
      if (!chunk) {
        throw new Error(
          `Missing WER audio chunk ${index + 1}/${chunkCount} for ${row.model} (${accessibilityId}).`
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

async function attachWerAudioChunksFromPager(report) {
  const rows = [];
  const expectedChunks = [];

  for (const row of report.rows || []) {
    if (row.status !== "passed") {
      continue;
    }

    const chunkCount = Number(row.werAudioChunkCount || 0);
    const rowSlug = automationSlug(row.model);
    for (let index = 0; index < chunkCount; index += 1) {
      expectedChunks.push({
        key: `${rowSlug}-${index}`,
        row,
        index,
        chunkCount,
      });
    }
  }

  const chunksByModel = new Map(
    (report.rows || []).map((row) => [row.model, []])
  );

  for (let globalIndex = 0; globalIndex < expectedChunks.length; globalIndex += 1) {
    const expected = expectedChunks[globalIndex];
    await waitForAudioPagerKey(expected.key, globalIndex, expectedChunks.length);

    const chunk = await readElementText("benchmark-audio-current");
    if (!chunk) {
      throw new Error(
        `Missing WER audio chunk ${expected.index + 1}/${expected.chunkCount} for ${expected.row.model} (${expected.key}).`
      );
    }

    chunksByModel.get(expected.row.model).push(chunk);

    if (globalIndex < expectedChunks.length - 1) {
      await $("~benchmark-audio-next").click();
    }
  }

  for (const row of report.rows || []) {
    if (row.status !== "passed") {
      rows.push(row);
      continue;
    }

    const chunks = chunksByModel.get(row.model) || [];
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

async function waitForAudioPagerKey(expectedKey, globalIndex, totalChunks) {
  const startedAt = Date.now();
  let lastKey = "";

  while (Date.now() - startedAt < 10_000) {
    lastKey = await readElementText("benchmark-audio-current-key");
    if (lastKey === expectedKey) {
      return;
    }

    await browser.pause(250);
  }

  throw new Error(
    `WER audio pager mismatch at chunk ${globalIndex + 1}/${totalChunks}: expected ${expectedKey}, got ${lastKey || "empty"}.`
  );
}

async function getBenchmarkReportFromUi({ includeAudio = false } = {}) {
  const report = await readBenchmarkReport("benchmark-json-display");

  if (!report || !includeAudio) {
    return report;
  }

  return attachWerAudioChunks(report);
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
    browser?.capabilities?.platformName ||
    browser?.requestedCapabilities?.platformName ||
    process.env.TESTMU_PLATFORM_NAME ||
    (process.env.TESTMU_IOS_DEVICE ? "iOS" : "Android")
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
    requestedPlatformVersion:
      process.env.TESTMU_PLATFORM_VERSION ||
      process.env.TESTMU_ANDROID_VERSION ||
      process.env.TESTMU_IOS_VERSION ||
      null,
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

  await activateAndroidBenchmarkApp();

  while (Date.now() - startedAt < timeoutMs) {
    if (await dismissAndroidSystemDialog()) {
      continue;
    }

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
          expect(row.werAudioChunkCount).toBeGreaterThan(0);
          expect(row.werAudioBase64Length).toBeGreaterThan(1000);
          if (typeof row.werAudioBase64 !== "string") {
            throw new Error(
              `Missing attached WER audio for ${expectedModel}; expected ${row.werAudioChunkCount} chunk(s) and ${row.werAudioBase64Length} base64 characters.`
            );
          }
          expect(row.werAudioBase64.length).toBe(row.werAudioBase64Length);
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
