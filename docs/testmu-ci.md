# TestMu Benchmark CI

This repository uses GitHub Actions plus LambdaTest/TestMu App Automation for a deterministic benchmark of `examples/BareRNExample`.

The workflow lives in `.github/workflows/rn-ci.yml` and does this:

1. Runs the SDK test/typecheck gate with `npm test`.
2. Builds the SDK TypeScript output, packs the SDK with `npm pack --ignore-scripts`, and installs that local tarball into `examples/BareRNExample`.
3. Builds `examples/BareRNExample` as a release Android APK.
4. Uploads the APK once to LambdaTest/TestMu.
5. Runs the same Appium benchmark on four Android real-device configs one after another.
6. Builds, uploads, and runs the same Appium benchmark on iPhone 14 when iOS signing secrets are configured. If signing is not configured, it can fall back to a pre-uploaded `TESTMU_IOS_APP_URL` or a downloadable `TESTMU_IOS_IPA_URL`.
7. Collects one JSON result per device.
8. Builds Markdown, CSV, and JSON summary reports.
9. Posts or updates the benchmark table as a PR comment.

KaneAI is not used in this flow. There is no `.lambdatest/config.yaml`, `configuration_id`, or `@KaneAI validate` trigger. The test is driven by the checked-in Appium spec so the results are repeatable.

## Required GitHub Secrets

Add these in GitHub under `Settings -> Secrets and variables -> Actions`:

- `LT_USERNAME`
- `LT_ACCESS_KEY`

Without these secrets, the TestMu upload job prints a skip message and exits successfully after the APK build.

The cloud run uses a release APK because React Native debug APKs expect a Metro server. A release APK is self-contained and can launch on a LambdaTest real device.

The APK is built after installing the SDK from the workflow-created `.tgz` package, not from npm. The workflow builds the TypeScript `lib/` output before packing and uses `--ignore-scripts` while packing because the CEPhonemizer rebuild needs Emscripten. The generated CEPhonemizer runtime is checked in so CI can make a complete local tarball without rebuilding it. The workflow fails early if the installed `@kittentts/react-native` package version does not match the repository root package version.

## iOS App Source

The best iOS path is to build the signed IPA in CI from the current commit. That proves the iPhone 14 benchmark is testing the same SDK tarball and app code as the PR, not an older upload.

Add these GitHub Actions secrets to enable that path:

- `IOS_CERTIFICATE_P12_BASE64`
- `IOS_CERTIFICATE_PASSWORD`
- `IOS_PROVISIONING_PROFILE_BASE64`
- `IOS_DEVELOPMENT_TEAM`

Optional iOS variables or secrets:

- `IOS_BUNDLE_IDENTIFIER` defaults to `org.reactjs.native.example.BasicExample`
- `IOS_EXPORT_METHOD` defaults to `development`
- `IOS_SIGNING_CERTIFICATE` defaults to `Apple Development`

The iOS build job also installs the SDK from the workflow-created `.tgz` package before archiving the app.

If CI signing is not available yet, the workflow can still run iOS with either:

- `TESTMU_IOS_APP_URL`: a pre-uploaded `lt://...` app URL from TestMu/LambdaTest.
- `TESTMU_IOS_IPA_URL`: a downloadable signed `.ipa`; CI downloads it, uploads it to TestMu, and uses the returned `lt://...` app URL.

Those fallback paths are useful for proving the device automation, but they may test a stale IPA unless you refresh the upload for every commit.

## Device Matrix

The workflow currently runs these devices sequentially with `max-parallel: 1`:

| Device              | OS         | Platform |
| ------------------- | ---------- | -------- |
| Pixel 5             | Android 12 | Android  |
| Galaxy Note10       | Android 12 | Android  |
| Galaxy S21          | Android 12 | Android  |
| Xiaomi Redmi Note 8 | Android 10 | Android  |
| iPhone 14           | iOS 16     | iOS      |

The iOS job is skipped when none of the iOS app sources above are configured. LambdaTest/TestMu iOS real-device runs need a signed IPA or an already uploaded `lt://...` app URL; unsigned simulator builds are not enough for this real-device benchmark.

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
- Passed model rows include the first generation timing, five warm measured runs, best warm generation timing, warm p50, warm p95, RTF values, audio duration, sample count, `sampleRate`, and `sampleHash`.
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

| Model | First gen (s) | Best warm (s) | Warm p50 (s) | Warm p95 (s) | Best RTF | Warm p50 RTF | Warm p95 RTF | Audio (s) | Samples | Sample hash |
| ----- | ------------: | ------------: | -----------: | -----------: | -------: | -----------: | -----------: | --------: | ------: | ----------- |

For each model, the app does one warm-up generation and then five measured warm generations. The main `Best warm` and `Best RTF` columns use the fastest of those five measured warm runs. The `Warm p50` and `Warm p95` columns show the median-ish and tail latency across the same five warm runs.

The PR comment is scoped to the commit SHA. A new pushed commit gets a new benchmark report comment, while a manual rerun of the same commit refreshes only that commit's own report comment.

If a device fails before producing benchmark numbers, the device job writes a failure JSON artifact instead. The final report and PR comment then show the device, total run time, failed stage, a short summary, and a link to the workflow logs.

If only one model hangs or fails, the app still writes a live device report with that model marked as failed or unfinished. This keeps the GitHub Action green when the automation successfully collected a truthful partial report.

Each model benchmark is capped inside the app, and each remote Appium device run has a CI timeout. If a real device freezes before the final report, Appium posts the latest partial model table. If the app never exposes any benchmark JSON, the fallback report records the timeout and the Appium log tail instead of leaving the PR with empty benchmark tables.
