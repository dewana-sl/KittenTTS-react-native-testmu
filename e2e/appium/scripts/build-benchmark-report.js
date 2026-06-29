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

function buildDeviceTable(report) {
  if (report.status === "failed") {
    return [
      `### ${escapeMarkdown(report.device)} (${escapeMarkdown(
        report.platformName
      )} ${escapeMarkdown(report.platformVersion)})`,
      "",
      "| Status | Failed stage | What happened | Details |",
      "| --- | --- | --- | --- |",
      `| Failed | ${escapeMarkdown(report.failedStage)} | ${escapeMarkdown(
        report.errorSummary
      )} | ${escapeMarkdown(report.errorDetails)} |`,
    ].join("\n");
  }

  const lines = [
    `### ${escapeMarkdown(report.device)} (${escapeMarkdown(
      report.platformName
    )} ${escapeMarkdown(report.platformVersion)})`,
    "",
    "| Model | Status | Generation time (s) | Audio duration (s) | RTF | Samples | Sample rate | Sample hash / Error |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
  ];

  for (const row of report.rows || []) {
    if (row.status === "failed") {
      lines.push(
        `| ${escapeMarkdown(
          row.modelDisplayName || row.model
        )} | Failed |  |  |  |  |  | ${escapeMarkdown(
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
      )} | Passed | ${formatNumber(row.generationSeconds)} | ${formatNumber(
        row.durationSeconds
      )} | ${formatNumber(row.rtf)} | ${Number(
        row.sampleCount || 0
      ).toLocaleString("en-US")} | ${row.sampleRate || ""} | \`${escapeMarkdown(
        row.sampleHash
      )}\` |`
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
  const partialReports = reports.filter((report) => report.status === "partial");
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
    "model",
    "modelDisplayName",
    "generationMs",
    "generationSeconds",
    "durationSeconds",
    "rtf",
    "sampleCount",
    "sampleRate",
    "sampleHash",
    "status",
    "failedStage",
    "errorSummary",
    "errorDetails",
  ];
  const lines = [header.join(",")];

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
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          "",
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
          row.model,
          row.modelDisplayName,
          row.generationMs,
          row.generationSeconds,
          row.durationSeconds,
          row.rtf,
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
