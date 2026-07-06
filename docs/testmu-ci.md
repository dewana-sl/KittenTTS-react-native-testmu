# TestMu Benchmark CI

This repository uses GitHub Actions plus LambdaTest/TestMu App Automation and mobile browser automation for a deterministic benchmark of `benchmark/app`.

The workflow lives in `.github/workflows/rn-ci.yml` and does this:

1. Runs the SDK test/typecheck gate with `npm test`.
2. Builds the SDK TypeScript output, packs the SDK with `npm pack --ignore-scripts`, and installs that local tarball into `benchmark/app`.
3. Builds and uploads one unsigned iOS IPA.
4. Runs the uploaded IPA on the iOS TestMu device matrix one device after another.
5. Builds the same benchmark app for web with Vite and React Native Web.
6. Serves the web build locally in GitHub Actions, exposes it through LambdaTest Tunnel, and runs TestMu real-device mobile browsers against it.
7. Builds `benchmark/app` as a release Android APK.
8. Uploads the APK once to LambdaTest/TestMu.
9. Runs the same Appium benchmark on four Android real-device configs one after another.
10. Collects one JSON result per app device or browser device.
11. Optionally uploads each generated WAV to Google Drive and records a listen link.
12. Optionally transcribes each generated WAV with NVIDIA Parakeet and computes word error rate.
13. Builds Markdown, CSV, and JSON summary reports.
14. Posts or updates the benchmark table as a PR comment.

KaneAI is not used in this flow. There is no `.lambdatest/config.yaml`, `configuration_id`, or `@KaneAI validate` trigger. The test is driven by the checked-in Appium spec so the results are repeatable.

## Required GitHub Secrets

Add these in GitHub under `Settings -> Secrets and variables -> Actions`:

- `LT_USERNAME`
- `LT_ACCESS_KEY`
- `NVIDIA_API_KEY` if you want Parakeet WER values
- `GOOGLE_SERVICE_ACCOUNT_JSON` if you want Drive-hosted WAV listen links

Without these secrets, the SDK tests still run but the TestMu device jobs cannot upload apps.

Without `NVIDIA_API_KEY`, the report still passes and marks Parakeet WER as skipped. This keeps the device benchmark usable in forks while making WER available when the NVIDIA-hosted Parakeet ASR credential is configured.

Without `GOOGLE_SERVICE_ACCOUNT_JSON`, the report still passes and marks audio upload as skipped. Store the full service-account JSON file contents in the secret; do not commit the JSON file.

The Android cloud run uses a release APK because React Native debug APKs expect a Metro server. A release APK is self-contained and can launch on a LambdaTest real device.

Both the iOS and Android benchmark apps are built after installing the SDK from the workflow-created `.tgz` package, not from npm. The workflow builds the TypeScript `lib/` output before packing and uses `--ignore-scripts` while packing because the CEPhonemizer rebuild needs Emscripten. The generated CEPhonemizer runtime is checked in so CI can make a complete local tarball without rebuilding it. The workflow fails early if the installed `@kittentts/react-native` package version does not match the repository root package version.

The web benchmark is also built after installing that same local SDK tarball into `benchmark/app`. For local development, `benchmark/app` points at the workspace package with `file:../..` so `npm run build:web` can be run without waiting for a publish. In CI, the workflow overwrites that install with the generated `.tgz` and verifies the installed SDK version before building. The app uses `react-native-web` and Vite only as a web build target; Android and iOS continue to use the normal React Native app entrypoints.

## iOS App Source

The workflow builds the iOS benchmark app from the PR commit, packages the unsigned `.app` into an `.ipa`, and uploads that IPA to TestMu once. Separate iOS matrix jobs then run the returned `lt://...` app URL on each iOS device. This avoids Apple Developer signing secrets while still testing app code and SDK code from the current commit.

The iOS device matrix runs before Android. If any iOS device fails, Android does not start, which keeps the feedback loop focused on the highest-risk path first.

## Device Matrix

The app-device workflow currently runs these devices sequentially with `max-parallel: 1`:

| Device              | OS         | Platform |
| ------------------- | ---------- | -------- |
| Pixel 5             | Android 12 | Android  |
| Galaxy Note10       | Android 12 | Android  |
| Pixel 8             | Android 14 | Android  |
| Xiaomi Redmi Note 8 | Android 10 | Android  |
| iPhone 14           | iOS 16     | iOS      |
| iPad 10.9 (2022)    | iOS 18     | iOS      |

Edit the `testmu-ios-benchmark.strategy.matrix.include` or `testmu-android-benchmark.strategy.matrix.include` lists in `.github/workflows/rn-ci.yml` to change the devices.

The mobile-browser workflow runs the web build on these real-device browser configs:

| Device           | OS         | Browser |
| ---------------- | ---------- | ------- |
| Pixel 8          | Android 14 | Chrome  |
| Galaxy Note10    | Android 12 | Chrome  |
| iPhone 14        | iOS 16     | Safari  |
| iPad 10.9 (2022) | iOS 18     | Safari  |

Edit the `testmu-web-browser-benchmark.strategy.matrix.include` list in `.github/workflows/rn-ci.yml` to change the browser devices.

## Web Browser Benchmark

The web path reuses `benchmark/app/App.tsx`. `benchmark/app/vite.config.ts` aliases `react-native` to `react-native-web`, and the app imports `createBenchmarkPlayer()` from platform-specific helpers so native builds use `react-native-sound` while web builds use the SDK browser audio player.

CI builds the web bundle with:

```sh
cd benchmark/app
npm run build:web
```

Each TestMu browser job downloads that build artifact, starts `npm run preview:web` on `http://localhost:4173`, starts `LambdaTest/LambdaTest-tunnel-action@v2`, and runs `npm run test:web:browser` from `e2e/appium`. The WebdriverIO config uses the TestMu mobile hub, `isRealMobile: true`, the matrix browser/device/OS values, and the unique tunnel name `kittentts-web-${{ github.run_id }}-${{ matrix.slug }}`.

## Optional GitHub Variables

Add this under `Settings -> Secrets and variables -> Actions -> Variables` if you want a different benchmark sentence:

- `TESTMU_SAMPLE_TEXT`
- `GOOGLE_DRIVE_ROOT_FOLDER_ID` if Drive-hosted WAV listen links are enabled
- `GOOGLE_DRIVE_PUBLIC_AUDIO=true` if each uploaded WAV should be shared with anyone who has the link

Default sample text:

```text
KittenTTS runs fully on your device and creates clear speech quickly.
This benchmark compares every model for speed, quality, and consistency.
```

The report records the exact sample text and character length so results from different runs can be compared honestly.

## Google Drive Audio Links

When `GOOGLE_SERVICE_ACCOUNT_JSON` and `GOOGLE_DRIVE_ROOT_FOLDER_ID` are configured, the final report job uploads the generated WAV for every passed model before the base64 audio payload is stripped from the JSON artifacts.

The Drive uploader creates or reuses this folder layout under `GOOGLE_DRIVE_ROOT_FOLDER_ID`:

```text
Flutter SDK/
Web SDK/
Swift SDK/
RN SDK/
  PR #002/
    pixel-5__kitten-tts-nano__af6b1ce3.wav
Python SDK/
```

The RN benchmark uses the `RN SDK` folder and a zero-padded pull request folder like `PR #002`. The report table shows each audio URL as a compact `Listen` hyperlink instead of printing the full Drive URL. The CSV and JSON reports keep the raw `audioListenUrl`, `audioDriveFileId`, upload status, and any upload error summary for debugging.

Use a Google Shared Drive folder as the root when authenticating with a service account. Google service accounts do not have personal Drive storage quota and cannot upload non-empty files into a normal user's My Drive folder, even when that folder is shared with the service account. Share the Shared Drive or a folder inside it with the service account email as a content manager/editor. If the root folder is not publicly shared, only people with Drive access can open the `Listen` links unless `GOOGLE_DRIVE_PUBLIC_AUDIO=true` is set.

## Benchmark Criteria

The Appium test in `e2e/appium/specs/kittentts-benchmark.android.spec.js` validates objective generation metadata:

- `tts-input` is visible.
- `benchmark-button` is enabled.
- `benchmark-report` appears after all models finish.
- Every bundled model has one result row: `nano`, `nano-int8`, `micro`, and `mini`.
- Passed model rows include the first generation timing, five warm measured runs, best warm generation timing, warm p50, warm p95, RTF values, audio duration, sample count, `sampleRate`, `sampleHash`, and a WAV payload used by the report job for Parakeet WER.
- Failed model rows include the model name, failed stage, and error summary.

The device app does not judge subjective audio quality. The report job can add an objective ASR check by sending the generated WAV to NVIDIA Parakeet, normalizing the transcript and reference text, and computing word error rate.

## Reports

Each device job uploads a JSON artifact named like:

```text
testmu-benchmark-pixel-5
testmu-benchmark-web-pixel-8-chrome
```

The final report job combines those files into:

- `benchmark-report/summary.md`
- `benchmark-report/summary.csv`
- `benchmark-report/summary.json`
- `benchmark-report/pr-comment.md`

`summary.md` and the PR comment include common run details plus a device status table with average RTF and the worst model RTF, followed by one table per device. Browser rows show the browser in the platform cell, for example `iOS 18 / Safari`. The reported platform version comes from the Appium session when TestMu exposes it; otherwise the workflow falls back to the requested matrix version:

| Model | Status | First gen (s) | Best warm (s) | Warm p50/p95 (s) | Best RTF | Warm p50/p95 RTF | Audio (s) | Listen |
| ----- | ------ | ------------: | ------------: | ---------------: | -------: | ---------------: | --------: | ------ |

For each model, the app does one warm-up generation and then five measured warm generations. The main `Best warm` and `Best RTF` columns use the fastest of those five measured warm runs. The `Warm p50` and `Warm p95` columns show the median-ish and tail latency across the same five warm runs.

Parakeet WER is computed after all device jobs finish. The benchmark app exposes the best warm-run WAV for each passed model in an automation-only JSON field, the report job transcodes that audio to 16 kHz mono WAV, sends it to NVIDIA Parakeet, and strips the audio payload before writing the combined report artifacts.

The PR comment is scoped to the commit SHA. A new pushed commit gets a new benchmark report comment, while a manual rerun of the same commit refreshes only that commit's own report comment.

If a device fails before producing benchmark numbers, the device job writes a failure JSON artifact instead. The final report and PR comment then show the device, total run time, failed stage, a short summary, and a link to the workflow logs.

If only one model hangs or fails, the app still writes a live device report with that model marked as failed or unfinished. This keeps the GitHub Action green when the automation successfully collected a truthful partial report.

Each model benchmark is capped inside the app, and each remote Appium device run has a CI timeout. If a real device freezes before the final report, Appium posts the latest partial model table. If the app never exposes any benchmark JSON, the fallback report records the timeout and the Appium log tail instead of leaving the PR with empty benchmark tables.
