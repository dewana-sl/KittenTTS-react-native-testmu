const fs = require("node:fs");
const path = require("node:path");

const inputDir = path.resolve(process.argv[2] || "benchmark-results");
const outputDir = path.resolve(process.argv[3] || "benchmark-report");

function readJsonFiles(dir) {
  if (!fs.existsSync(dir)) {
    return [];
  }

  const entries = fs.readdirSync(dir, { withFileTypes: true });
  const files = [];

  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...readJsonFiles(fullPath));
    } else if (entry.isFile() && entry.name.endsWith(".json")) {
      files.push(fullPath);
    }
  }

  return files;
}

function formatNumber(value, digits = 3) {
  if (!Number.isFinite(value)) {
    return "";
  }

  return Number(value).toFixed(digits);
}

function escapeMarkdown(value) {
  return String(value ?? "")
    .replace(/\\/g, "\\\\")
    .replace(/\|/g, "\\|")
    .replace(/\n/g, " ");
}

function escapeCsv(value) {
  const text = String(value ?? "");
  if (!/[",\n]/.test(text)) {
    return text;
  }

  return `"${text.replace(/"/g, '""')}"`;
}

function getTotalRuntimeSeconds(report) {
  const explicit = Number(report.totalRuntimeSeconds);
  if (Number.isFinite(explicit)) {
    return explicit;
  }

  const runtimeMs = Number(report.totalRuntimeMs);
  if (Number.isFinite(runtimeMs)) {
    return runtimeMs / 1000;
  }

  return null;
}

function formatTotalRuntime(report) {
  const runtime = getTotalRuntimeSeconds(report);
  return Number.isFinite(runtime)
    ? `${formatNumber(runtime, 1)}s`
    : "unavailable";
}

function formatLogLink(report) {
  const url = report.logUrl || report.workflowRunUrl;
  if (url) {
    return `[Open workflow logs](${url})`;
  }

  return escapeMarkdown(report.errorDetails || "Open the device job logs.");
}

function buildDeviceTable(report) {
  if (report.status === "failed") {
    return [
      `### ${escapeMarkdown(report.device)} (${escapeMarkdown(
        report.platformName
      )} ${escapeMarkdown(report.platformVersion)})`,
      "",
      "| Status | Total run time | Failed stage | What happened | Logs |",
      "| --- | ---: | --- | --- | --- |",
      `| Failed | ${formatTotalRuntime(report)} | ${escapeMarkdown(
        report.failedStage
      )} | ${escapeMarkdown(report.errorSummary)} | ${formatLogLink(report)} |`,
    ].join("\n");
  }

  const lines = [
    `### ${escapeMarkdown(report.device)} (${escapeMarkdown(
      report.platformName
    )} ${escapeMarkdown(report.platformVersion)})`,
    "",
    `Total run time: ${formatTotalRuntime(report)}`,
    "",
    "| Model | Status | First gen (s) | Best warm (s) | Warm p50 (s) | Warm p95 (s) | Best RTF | Warm p50 RTF | Warm p95 RTF | Audio (s) | Samples | Sample hash / Error |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
  ];

  for (const row of report.rows || []) {
    if (row.status === "failed") {
      lines.push(
        `| ${escapeMarkdown(
          row.modelDisplayName || row.model
        )} | Failed |  |  |  |  |  |  |  |  |  | ${escapeMarkdown(
          `${row.failedStage || "Benchmark"}: ${
            row.errorSummary || "Unknown model failure"
          }`
        )} |`
      );
      continue;
    }

    lines.push(
      `| ${escapeMarkdown(
        row.modelDisplayName || row.model
      )} | Passed | ${formatNumber(
        row.firstGenerationSeconds
      )} | ${formatNumber(row.generationSeconds)} | ${formatNumber(
        row.warmP50GenerationSeconds
      )} | ${formatNumber(row.warmP95GenerationSeconds)} | ${formatNumber(
        row.rtf
      )} | ${formatNumber(row.warmP50Rtf)} | ${formatNumber(
        row.warmP95Rtf
      )} | ${formatNumber(row.durationSeconds)} | ${Number(
        row.sampleCount || 0
      ).toLocaleString("en-US")} | \`${escapeMarkdown(row.sampleHash)}\` |`
    );
  }

  return lines.join("\n");
}

function buildSummary(reports) {
  if (reports.length === 0) {
    return [
      "# KittenTTS TestMu Benchmark Report",
      "",
      "No benchmark result JSON files were found.",
      "",
    ].join("\n");
  }

  const completedReports = reports.filter(
    (report) => report.status !== "failed" && report.status !== "partial"
  );
  const partialReports = reports.filter(
    (report) => report.status === "partial"
  );
  const failedReports = reports.filter((report) => report.status === "failed");
  const first = reports.find((report) => report.sampleText) || reports[0];
  const firstPassed =
    completedReports[0] ||
    partialReports.find((report) => report.voice || report.voiceDisplayName) ||
    first;
  const lines = [
    "# KittenTTS TestMu Benchmark Report",
    "",
    "## Common Details",
    "",
    `- Sample text: ${escapeMarkdown(first.sampleText)}`,
    `- Character length: ${first.characterLength}`,
    `- Voice: ${escapeMarkdown(
      firstPassed.voiceDisplayName || firstPassed.voice || "unavailable"
    )}`,
    `- Speed: ${
      firstPassed.speed === undefined ? "unavailable" : `${firstPassed.speed}x`
    }`,
    `- Devices completed: ${completedReports.length}`,
    `- Devices partial: ${partialReports.length}`,
    `- Devices failed: ${failedReports.length}`,
    `- GitHub run: ${
      process.env.GITHUB_RUN_ID || first.githubRunId || "local"
    }`,
    `- Commit: ${process.env.GITHUB_SHA || first.githubSha || "local"}`,
    "",
  ];

  for (const report of reports) {
    lines.push(buildDeviceTable(report), "");
  }

  return lines.join("\n");
}

function buildCsv(reports) {
  const header = [
    "device",
    "platformName",
    "platformVersion",
    "sampleText",
    "characterLength",
    "voice",
    "speed",
    "totalRuntimeSeconds",
    "model",
    "modelDisplayName",
    "firstGenerationMs",
    "firstGenerationSeconds",
    "generationMs",
    "generationSeconds",
    "warmRunCount",
    "warmGenerationMs",
    "warmGenerationSeconds",
    "warmP50GenerationMs",
    "warmP50GenerationSeconds",
    "warmP95GenerationMs",
    "warmP95GenerationSeconds",
    "durationSeconds",
    "rtf",
    "warmRtf",
    "warmP50Rtf",
    "warmP95Rtf",
    "sampleCount",
    "sampleRate",
    "sampleHash",
    "status",
    "failedStage",
    "errorSummary",
    "errorDetails",
  ];
  const lines = [header.join(",")];
  const emptyMetricCells = Array(
    header.indexOf("status") - header.indexOf("model")
  ).fill("");

  for (const report of reports) {
    if (report.status === "failed") {
      lines.push(
        [
          report.device,
          report.platformName,
          report.platformVersion,
          report.sampleText,
          report.characterLength,
          report.voiceDisplayName || report.voice,
          report.speed,
          getTotalRuntimeSeconds(report),
          ...emptyMetricCells,
          report.status,
          report.failedStage,
          report.errorSummary,
          report.errorDetails,
        ]
          .map(escapeCsv)
          .join(",")
      );
      continue;
    }

    for (const row of report.rows || []) {
      lines.push(
        [
          report.device,
          report.platformName,
          report.platformVersion,
          report.sampleText,
          report.characterLength,
          report.voiceDisplayName || report.voice,
          report.speed,
          getTotalRuntimeSeconds(report),
          row.model,
          row.modelDisplayName,
          row.firstGenerationMs,
          row.firstGenerationSeconds,
          row.generationMs,
          row.generationSeconds,
          row.warmRunCount,
          Array.isArray(row.warmGenerationMs)
            ? row.warmGenerationMs.join("|")
            : "",
          Array.isArray(row.warmGenerationSeconds)
            ? row.warmGenerationSeconds.join("|")
            : "",
          row.warmP50GenerationMs,
          row.warmP50GenerationSeconds,
          row.warmP95GenerationMs,
          row.warmP95GenerationSeconds,
          row.durationSeconds,
          row.rtf,
          Array.isArray(row.warmRtf) ? row.warmRtf.join("|") : "",
          row.warmP50Rtf,
          row.warmP95Rtf,
          row.sampleCount,
          row.sampleRate,
          row.sampleHash,
          row.status || "passed",
          row.failedStage || "",
          row.errorSummary || "",
          "",
        ]
          .map(escapeCsv)
          .join(",")
      );
    }
  }

  return `${lines.join("\n")}\n`;
}

function main() {
  fs.mkdirSync(outputDir, { recursive: true });

  const reports = readJsonFiles(inputDir)
    .map((file) => JSON.parse(fs.readFileSync(file, "utf8")))
    .sort((a, b) => String(a.device).localeCompare(String(b.device)));

  const summary = buildSummary(reports);
  fs.writeFileSync(path.join(outputDir, "summary.md"), `${summary.trim()}\n`);
  fs.writeFileSync(path.join(outputDir, "summary.csv"), buildCsv(reports));
  fs.writeFileSync(
    path.join(outputDir, "summary.json"),
    `${JSON.stringify({ reports }, null, 2)}\n`
  );
  fs.writeFileSync(
    path.join(outputDir, "pr-comment.md"),
    `${summary.trim()}\n`
  );
}

main();
