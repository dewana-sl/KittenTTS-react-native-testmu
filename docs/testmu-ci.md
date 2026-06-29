# TestMu Android Benchmark CI

This repository uses GitHub Actions plus LambdaTest/TestMu App Automation for a deterministic Android benchmark of `examples/BareRNExample`.

The workflow lives in `.github/workflows/rn-ci.yml` and does this:

1. Runs the SDK test/typecheck gate with `npm test`.
2. Builds `examples/BareRNExample` as a release Android APK.
3. Uploads the APK once to LambdaTest/TestMu.
4. Runs the same Appium benchmark on five Android real-device configs one after another.
5. Collects one JSON result per device.
6. Builds Markdown, CSV, and JSON summary reports.
7. Posts or updates the benchmark table as a PR comment.

KaneAI is not used in this flow. There is no `.lambdatest/config.yaml`, `configuration_id`, or `@KaneAI validate` trigger. The test is driven by the checked-in Appium spec so the results are repeatable.

## Required GitHub Secrets

Add these in GitHub under `Settings -> Secrets and variables -> Actions`:

- `LT_USERNAME`
- `LT_ACCESS_KEY`

Without these secrets, the TestMu upload job prints a skip message and exits successfully after the APK build.

The cloud run uses a release APK because React Native debug APKs expect a Metro server. A release APK is self-contained and can launch on a LambdaTest real device.

## Device Matrix

The workflow currently runs these devices sequentially with `max-parallel: 1`:

| Device              | OS         |
| ------------------- | ---------- |
| Pixel 5             | Android 12 |
| Galaxy Note10       | Android 12 |
| Galaxy S22 Ultra 5G | Android 12 |
| Galaxy S21          | Android 12 |
| Xiaomi Redmi Note 8 | Android 10 |

Edit the `testmu-android-benchmark.strategy.matrix.include` list in `.github/workflows/rn-ci.yml` to change the phones.

## Optional GitHub Variables

Add this under `Settings -> Secrets and variables -> Actions -> Variables` if you want a different benchmark sentence:

- `TESTMU_SAMPLE_TEXT`

Default sample text:

```text
Hello! Welcome to KittenTTS, a fast on-device text-to-speech engine.
```

The report records the exact sample text and character length so results from different runs can be compared honestly.

## Benchmark Criteria

The Appium test in `e2e/appium/specs/kittentts-benchmark.android.spec.js` validates objective generation metadata:

- `tts-input` is visible.
- `benchmark-button` is enabled.
- `benchmark-report` appears after all models finish.
- Every bundled model has one result row: `nano`, `nano-int8`, `micro`, and `mini`.
- Passed model rows include `generationMs`, audio duration, RTF, sample count, `sampleRate`, and `sampleHash`.
- Failed model rows include the model name, failed stage, and error summary.

The test does not judge subjective audio quality.

## Reports

Each device job uploads a JSON artifact named like:

```text
testmu-benchmark-pixel-5
```

The final report job combines those files into:

- `benchmark-report/summary.md`
- `benchmark-report/summary.csv`
- `benchmark-report/summary.json`
- `benchmark-report/pr-comment.md`

`summary.md` and the PR comment include common run details plus one table per device:

| Model | Generation time (s) | Audio duration (s) | RTF | Samples | Sample rate | Sample hash |
| ----- | ------------------: | -----------------: | --: | ------: | ----------: | ----------- |

The PR comment is scoped to the commit SHA. A new pushed commit gets a new benchmark report comment, while a manual rerun of the same commit refreshes only that commit's own report comment.

If a device fails before producing benchmark numbers, the device job writes a failure JSON artifact instead. The final report and PR comment then show the device, failed stage, summary, workflow link, and the instruction to open that specific device job log for the exact LambdaTest/Appium error.

If only one model hangs or fails, the app still writes the device report with that model marked as failed. This keeps the GitHub Action green when the automation successfully collected a truthful report.

Each model benchmark is capped inside the app, and each remote Appium device run has a CI timeout. If a real device freezes or the app never exposes `benchmark-report`, the report records the timeout and the Appium log tail instead of leaving the PR with empty benchmark tables.
