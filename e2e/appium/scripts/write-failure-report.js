const fs = require("node:fs");
const path = require("node:path");

function stripAnsi(value) {
  return String(value || "").replace(/\x1b\[[0-9;]*m/g, "");
}

function slugify(value) {
  return String(value || "device")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
}

function findErrorSummary(logText) {
  const lines = stripAnsi(logText).split(/\r?\n/).filter(Boolean);
  const usefulPatterns = [
    /App showed error-banner/i,
    /Timed out waiting/i,
    /still not enabled/i,
    /waitForEnabled/i,
    /no such element/i,
    /invalid element state/i,
    /Session deleted/i,
    /ECONNRESET/i,
    /Error:/i,
    /Timeout/i,
  ];

  const line = lines
    .slice()
    .reverse()
    .find(
      (candidate) =>
        usefulPatterns.some((pattern) => pattern.test(candidate)) &&
        !/Spec Files:|^\s*FAILED\s+in/i.test(candidate)
    );

  return line
    ? line.trim().slice(0, 280)
    : "The Appium benchmark step failed before benchmark rows were collected.";
}

function readLog() {
  const logPath = path.join(process.cwd(), "reports", "appium-output.log");
  if (!fs.existsSync(logPath)) {
    return "";
  }

  return fs.readFileSync(logPath, "utf8");
}

function numberFromEnv(name) {
  const value = Number(process.env[name]);
  return Number.isFinite(value) ? value : null;
}

function main() {
  const logText = readLog();
  const device = process.env.TESTMU_DEVICE || "device";
  const target = process.env.TESTMU_TARGET || "app";
  const browserName = process.env.TESTMU_BROWSER_NAME || null;
  const platformName = process.env.TESTMU_PLATFORM_NAME || "Android";
  const startedAtMs = numberFromEnv("BENCHMARK_STARTED_AT_MS");
  const finishedAtMs = numberFromEnv("BENCHMARK_FINISHED_AT_MS") || Date.now();
  const totalRuntimeSeconds =
    numberFromEnv("BENCHMARK_TOTAL_RUNTIME_SECONDS") ||
    (startedAtMs
      ? Number(((finishedAtMs - startedAtMs) / 1000).toFixed(3))
      : null);
  const workflowRunUrl = process.env.GITHUB_RUN_URL || null;
  const logHint = workflowRunUrl
    ? `Open the workflow run logs and select the "${device}" benchmark job.`
    : "Open this benchmark job log in GitHub Actions for the full Appium output.";
  const sampleText = process.env.TESTMU_SAMPLE_TEXT || null;

  const report = {
    schemaVersion: 1,
    target,
    status: "failed",
    failedStage:
      process.env.BENCHMARK_FAILED_STAGE ||
      `Run TestMu ${platformName} benchmark`,
    errorSummary: findErrorSummary(logText),
    errorDetails: logHint,
    logUrl: workflowRunUrl,
    device,
    platformName,
    platformVersion: process.env.TESTMU_PLATFORM_VERSION || null,
    browserName,
    browserVersion: process.env.TESTMU_BROWSER_VERSION || null,
    requestedBrowserName: browserName,
    webUrl: process.env.TESTMU_WEB_URL || null,
    tunnelName: process.env.TESTMU_TUNNEL_NAME || null,
    realDevice: process.env.TESTMU_REAL_DEVICE !== "false",
    sessionId: null,
    githubRunId: process.env.GITHUB_RUN_ID || null,
    githubSha: process.env.GITHUB_SHA || null,
    deviceStartedAt: startedAtMs ? new Date(startedAtMs).toISOString() : null,
    deviceFinishedAt: new Date(finishedAtMs).toISOString(),
    totalRuntimeSeconds,
    capturedAt: new Date().toISOString(),
    sampleText,
    characterLength: sampleText ? Array.from(sampleText).length : null,
    rows: [],
  };

  fs.mkdirSync(path.join(process.cwd(), "reports"), { recursive: true });
  const filePrefix =
    target === "web"
      ? `web-${slugify(device)}-${slugify(browserName || "browser")}`
      : slugify(device);
  fs.writeFileSync(
    path.join(process.cwd(), "reports", `${filePrefix}.json`),
    `${JSON.stringify(report, null, 2)}\n`
  );
}

main();
