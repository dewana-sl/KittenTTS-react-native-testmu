# TestMu Android CI

This repo has a first-pass TestMu smoke setup for the React Native bare example.

The workflow lives in `.github/workflows/rn-ci.yml` and does three things:

1. Runs the SDK test/typecheck gate with `npm test`.
2. Builds `examples/BareRNExample` as a release Android APK.
3. Uploads the APK to TestMu and runs `e2e/appium/specs/kittentts-smoke.android.spec.js` when TestMu credentials are configured.

## Required GitHub Secrets

Add these in GitHub under `Settings -> Secrets and variables -> Actions`:

- `LT_USERNAME`
- `LT_ACCESS_KEY`

Without these secrets, the TestMu job prints a skip message and exits successfully after the APK build.

The cloud run uses a release APK because React Native debug APKs expect a Metro server. A release APK is self-contained and can launch on a LambdaTest real device.

## Optional GitHub Variables

Add these under `Settings -> Secrets and variables -> Actions -> Variables` if you want to override the default cloud device:

- `TESTMU_ANDROID_DEVICE`, default `Galaxy S21`
- `TESTMU_ANDROID_VERSION`, default `12`
- `TESTMU_REAL_DEVICE`, default `true`

Set `TESTMU_REAL_DEVICE=false` to use a TestMu virtual device configuration.

## TestMu GitHub App

Install the TestMu AI Cloud GitHub App on this repository. The committed `.lambdatest/config.yaml` contains the project, folder, and assignee values from the app setup screen.

Before KaneAI can run, replace this placeholder with the real Test Run configuration value from LambdaTest:

```yaml
configuration_id: "your_test_run_configuration_id"
```

The committed `.lambdatest/agent.md` gives KaneAI the app-specific smoke-test rules and automation IDs.

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

The example app also displays `sample-rate` and `sample-hash` for manual/debug evidence, but the CI smoke test does not fail on those lower-page labels because they may be outside the accessible viewport on smaller real devices.

It does not judge subjective audio quality.
