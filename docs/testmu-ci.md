# TestMu Android CI

This repo has a first-pass TestMu smoke setup for the React Native bare example.

The workflow lives in `.github/workflows/rn-ci.yml` and does three things:

1. Runs the SDK test/typecheck gate with `npm test`.
2. Builds `examples/BareRNExample` as a debug Android APK.
3. Uploads the APK to TestMu and runs `e2e/appium/specs/kittentts-smoke.android.spec.js` when TestMu credentials are configured.

## Required GitHub Secrets

Add these in GitHub under `Settings -> Secrets and variables -> Actions`:

- `LT_USERNAME`
- `LT_ACCESS_KEY`

Without these secrets, the TestMu job prints a skip message and exits successfully after the APK build.

## Optional GitHub Variables

Add these under `Settings -> Secrets and variables -> Actions -> Variables` if you want to override the default cloud device:

- `TESTMU_ANDROID_DEVICE`, default `Galaxy S21`
- `TESTMU_ANDROID_VERSION`, default `12`
- `TESTMU_REAL_DEVICE`, default `true`

Set `TESTMU_REAL_DEVICE=false` to use a TestMu virtual device configuration.

## TestMu GitHub App

Install the TestMu AI Cloud GitHub App on this repository. Then copy `.lambdatest/config.example.yaml` to `.lambdatest/config.yaml` and fill it from the TestMu integration page.

`.lambdatest/config.yaml` is intentionally gitignored because the values are account/project specific. The committed `.lambdatest/agent.md` gives KaneAI the app-specific smoke-test rules and automation IDs.

Trigger KaneAI/TestMu on a PR with:

```text
@TestMuAI Validate this PR
```

or:

```text
@KaneAI Validate this PR
```

## Smoke Criteria

The Appium smoke test validates objective generation metadata:

- `tts-input` is visible.
- `generate-button` is enabled.
- `result-card` appears after generation.
- `sample-count` is greater than zero.
- `duration` is greater than zero and less than 30 seconds.
- `sample-rate` is 24000.
- `sample-hash` is an eight-character lowercase hex value.

It does not judge subjective audio quality.
