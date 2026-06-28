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
| Pixel 6             | Android 12 |

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
- `generationMs`, audio duration, RTF, and sample count are greater than zero.
- `sampleRate` is `24000`.
- `sampleHash` is an 8-character hex value.

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

The PR comment is updated in place using a hidden marker, so repeated workflow runs do not spam the pull request.
