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
  return formatDuration(runtime);
}

function formatDuration(seconds) {
  const value = Number(seconds);
  if (!Number.isFinite(value)) {
    return "unavailable";
  }

  if (value < 60) {
    return `${formatNumber(value, 1)}s`;
  }

  const wholeSeconds = Math.round(value);
  const minutes = Math.floor(wholeSeconds / 60);
  const remainingSeconds = wholeSeconds % 60;
  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;

  if (hours > 0) {
    return `${hours}h ${remainingMinutes}m ${remainingSeconds}s`;
  }

  return `${minutes}m ${remainingSeconds}s`;
}

function formatLogLink(report) {
  const url = report.logUrl || report.workflowRunUrl;
  if (url) {
    return `[Open workflow logs](${url})`;
  }

  return escapeMarkdown(report.errorDetails || "Open the device job logs.");
}

function formatWer(row) {
  if (
    row.parakeetStatus === "passed" &&
    Number.isFinite(row.parakeetWerPercent)
  ) {
    return `${formatNumber(row.parakeetWerPercent, 2)}%`;
  }

  if (row.parakeetStatus) {
    return escapeMarkdown(row.parakeetStatus);
  }

  return "";
}

function formatWerDetails(row) {
  if (row.parakeetStatus === "passed") {
    const distance = Number(row.parakeetEditDistance);
    const words = Number(row.parakeetReferenceWordCount);
    if (Number.isFinite(distance) && Number.isFinite(words)) {
      return `${distance}/${words} edits`;
    }
    return "";
  }

  return escapeMarkdown(row.parakeetErrorSummary || "");
}

function formatTranscript(row) {
  if (row.parakeetStatus === "passed") {
    return escapeMarkdown(row.parakeetTranscript || "");
  }

  return escapeMarkdown(row.parakeetErrorSummary || "");
}

function formatAudioLink(row) {
  if (row.audioListenUrl) {
    return `[Listen](${row.audioListenUrl})`;
  }

  if (row.audioUploadStatus === "failed") {
    return `Upload failed: ${escapeMarkdown(row.audioUploadErrorSummary || "")}`;
  }

  if (row.audioUploadStatus === "skipped") {
    return `Skipped: ${escapeMarkdown(row.audioUploadErrorSummary || "")}`;
  }

  return "";
}

function formatAudioFolder(reports) {
  const report = reports.find(
    (item) =>
      item.audioUpload?.folderUrl &&
      Number(item.audioUpload.uploadedRows || 0) > 0
  );
  if (!report) {
    return "";
  }

  return `[Open folder](${report.audioUpload.folderUrl})`;
}

function countPassedRows(report) {
  return (report.rows || []).filter((row) => row.status === "passed").length;
}

function countFailedRows(report) {
  return (report.rows || []).filter((row) => row.status === "failed").length;
}

function average(values) {
  const finite = values.map(Number).filter(Number.isFinite);
  if (finite.length === 0) return null;
  return finite.reduce((sum, value) => sum + value, 0) / finite.length;
}

function summarizeDeviceWer(report) {
  const avg = average(
    (report.rows || [])
      .filter((row) => row.parakeetStatus === "passed")
      .map((row) => row.parakeetWerPercent)
  );
  return Number.isFinite(avg) ? `${formatNumber(avg, 2)}%` : "";
}

function summarizeDeviceStatus(report) {
  if (report.status === "failed") return "Failed";
  if (report.status === "partial") return "Partial";
  if (countFailedRows(report) > 0) return "Model failures";
  return "Passed";
}

function buildDeviceStatusTable(reports) {
  const lines = [
    "| Device | Platform | Status | Runtime | Models | Avg WER | Notes |",
    "| --- | --- | --- | ---: | ---: | ---: | --- |",
  ];

  for (const report of reports) {
    const modelSummary =
      report.status === "failed"
        ? ""
        : `${countPassedRows(report)}/${(report.rows || []).length || 4}`;
    const notes =
      report.status === "failed"
        ? report.errorSummary || report.failedStage || ""
        : countFailedRows(report) > 0
        ? `${countFailedRows(report)} model row(s) failed`
        : "";
    lines.push(
      `| ${escapeMarkdown(report.device)} | ${escapeMarkdown(
        `${report.platformName} ${report.platformVersion}`
      )} | ${summarizeDeviceStatus(report)} | ${formatTotalRuntime(
        report
      )} | ${modelSummary} | ${summarizeDeviceWer(report)} | ${escapeMarkdown(
        notes
      )} |`
    );
  }

  return lines.join("\n");
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
    `Runtime: ${formatTotalRuntime(report)}. Models passed: ${countPassedRows(
      report
    )}/${(report.rows || []).length}. Average WER: ${
      summarizeDeviceWer(report) || "unavailable"
    }.`,
    "",
    "#### Performance",
    "",
    "| Model | Status | First gen (s) | Best warm (s) | Warm p50/p95 (s) | Best RTF | Warm p50/p95 RTF | Audio (s) | Listen |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
  ];

  for (const row of report.rows || []) {
    if (row.status === "failed") {
      lines.push(
        `| ${escapeMarkdown(
          row.modelDisplayName || row.model
        )} | Failed |  |  |  |  |  |  | ${formatAudioLink(row)} |`
      );
      continue;
    }

    lines.push(
      `| ${escapeMarkdown(
        row.modelDisplayName || row.model
      )} | Passed | ${formatNumber(row.firstGenerationSeconds)} | ${formatNumber(
        row.generationSeconds
      )} | ${formatNumber(row.warmP50GenerationSeconds)} / ${formatNumber(
        row.warmP95GenerationSeconds
      )} | ${formatNumber(row.rtf)} | ${formatNumber(
        row.warmP50Rtf
      )} / ${formatNumber(row.warmP95Rtf)} | ${formatNumber(
        row.durationSeconds
      )} | ${formatAudioLink(row)} |`
    );
  }

  lines.push(
    "",
    "#### Parakeet ASR",
    "",
    "| Model | WER | Edits | Transcript / Error | Samples | Hash |",
    "| --- | ---: | ---: | --- | ---: | --- |"
  );

  for (const row of report.rows || []) {
    if (row.status === "failed") {
      lines.push(
        `| ${escapeMarkdown(
          row.modelDisplayName || row.model
        )} |  |  | ${escapeMarkdown(
          `${row.failedStage || "Benchmark"}: ${
            row.errorSummary || "Unknown model failure"
          }`
        )} |  |  |`
      );
      continue;
    }

    lines.push(
      `| ${escapeMarkdown(
        row.modelDisplayName || row.model
      )} | ${formatWer(row)} | ${formatWerDetails(row)} | ${formatTranscript(
        row
      )} | ${Number(
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
  const audioFolder = formatAudioFolder(reports);
  const lines = [
    "# KittenTTS TestMu Benchmark Report",
    "",
    "## Summary",
    "",
    "| Field | Value |",
    "| --- | --- |",
    `| Sample text | ${escapeMarkdown(first.sampleText)} |`,
    `| Character length | ${first.characterLength} |`,
    `| Voice | ${escapeMarkdown(
      firstPassed.voiceDisplayName || firstPassed.voice || "unavailable"
    )} |`,
    `| Speed | ${
      firstPassed.speed === undefined ? "unavailable" : `${firstPassed.speed}x`
    } |`,
    `| Devices completed / partial / failed | ${completedReports.length} / ${partialReports.length} / ${failedReports.length} |`,
    `| Parakeet WER rows | ${summarizeParakeetWer(reports)} |`,
    `| Audio files | ${audioFolder || summarizeAudioUpload(reports)} |`,
    `| WER normalization | Treats punctuation/case as insignificant and normalizes KittenTTS == Kitten TTS. |`,
    `| GitHub run | ${
      process.env.GITHUB_RUN_ID || first.githubRunId || "local"
    } |`,
    `| Commit | ${process.env.GITHUB_SHA || first.githubSha || "local"} |`,
    "",
    "## Device Status",
    "",
    buildDeviceStatusTable(reports),
    "",
    "## Device Details",
    "",
  ];

  for (const report of reports) {
    lines.push(buildDeviceTable(report), "");
  }

  return lines.join("\n");
}

function summarizeAudioUpload(reports) {
  const uploads = reports
    .map((report) => report.audioUpload)
    .filter(Boolean);
  if (uploads.length === 0) {
    return "unavailable";
  }

  const uploaded = uploads.reduce(
    (sum, upload) => sum + Number(upload.uploadedRows || 0),
    0
  );
  const skipped = uploads.reduce(
    (sum, upload) => sum + Number(upload.skippedRows || 0),
    0
  );
  const failed = uploads.reduce(
    (sum, upload) => sum + Number(upload.failedRows || 0),
    0
  );
  return `${uploaded} uploaded, ${skipped} skipped, ${failed} failed`;
}

function summarizeParakeetWer(reports) {
  const rows = reports.flatMap((report) => report.rows || []);
  const passed = rows.filter((row) => row.parakeetStatus === "passed").length;
  const skipped = rows.filter((row) => row.parakeetStatus === "skipped").length;
  const failed = rows.filter((row) => row.parakeetStatus === "failed").length;

  if (passed === 0 && skipped === 0 && failed === 0) {
    return "unavailable";
  }

  return `${passed} passed, ${skipped} skipped, ${failed} failed`;
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
    "parakeetStatus",
    "parakeetModel",
    "parakeetWer",
    "parakeetWerPercent",
    "parakeetTranscript",
    "parakeetErrorSummary",
    "sampleCount",
    "sampleRate",
    "sampleHash",
    "audioUploadStatus",
    "audioListenUrl",
    "audioFileName",
    "audioDriveFileId",
    "audioUploadErrorSummary",
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
          row.parakeetStatus || "",
          row.parakeetModel || "",
          row.parakeetWer,
          row.parakeetWerPercent,
          row.parakeetTranscript || "",
          row.parakeetErrorSummary || "",
          row.sampleCount,
          row.sampleRate,
          row.sampleHash,
          row.audioUploadStatus || "",
          row.audioListenUrl || "",
          row.audioFileName || "",
          row.audioDriveFileId || "",
          row.audioUploadErrorSummary || "",
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
