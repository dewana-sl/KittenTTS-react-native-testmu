const fs = require("node:fs");
const path = require("node:path");

const EXPECTED_MODELS = ["nano", "nano-int8", "micro", "mini"];

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

function writeDeviceReport(report) {
  const device = process.env.TESTMU_ANDROID_DEVICE || "Pixel 5";
  const platformVersion = process.env.TESTMU_ANDROID_VERSION || "12";
  const outputDir = path.resolve(__dirname, "..", "reports");
  fs.mkdirSync(outputDir, { recursive: true });

  const payload = {
    device,
    platformName: "Android",
    platformVersion,
    realDevice: process.env.TESTMU_REAL_DEVICE !== "false",
    sessionId: browser.sessionId,
    githubRunId: process.env.GITHUB_RUN_ID || null,
    githubSha: process.env.GITHUB_SHA || null,
    capturedAt: new Date().toISOString(),
    ...report,
  };

  const outputPath = path.join(outputDir, `${slugify(device)}.json`);
  fs.writeFileSync(outputPath, `${JSON.stringify(payload, null, 2)}\n`);
}

describe("KittenTTS React Native benchmark", () => {
  it("benchmarks every bundled model and writes a device report", async () => {
    const input = await $("~tts-input");
    await input.waitForDisplayed({ timeout: 300000 });

    const sampleText = process.env.TESTMU_SAMPLE_TEXT;
    if (sampleText) {
      await input.click();
      await input.clearValue();
      await input.setValue(sampleText);
    }

    const benchmark = await $("~benchmark-button");
    await benchmark.waitForEnabled({ timeout: 300000 });
    await benchmark.click();

    const reportCard = await $("~benchmark-report");
    await reportCard.waitForDisplayed({ timeout: 1800000 });

    const reportText = await $("~benchmark-json").getText();
    const report = parseBenchmarkJson(reportText);

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
      expect(row.generationMs).toBeGreaterThan(0);
      expect(row.generationSeconds).toBeGreaterThan(0);
      expect(row.durationSeconds).toBeGreaterThan(0);
      expect(row.rtf).toBeGreaterThan(0);
      expect(row.sampleCount).toBeGreaterThan(0);
      expect(row.sampleRate).toBe(24000);
      if (!/^[0-9a-f]{8}$/.test(row.sampleHash)) {
        throw new Error(
          `Invalid sample hash for ${expectedModel}: ${row.sampleHash}`
        );
      }
    }

    writeDeviceReport(report);
  });
});
