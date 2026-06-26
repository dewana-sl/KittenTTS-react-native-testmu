# KittenTTS React Native TestMu Agent Notes

Focus only on the native Android `examples/BareRNExample` app.

Do not generate or author tests for GitHub Actions, CI configuration, GitHub pull request pages, repository pages, secrets configuration, or the `test_url`. The `test_url` exists only because the GitHub App requires a URL; it is not the application under test.

Generate exactly one smoke test for the installed native mobile app:

1. Wait for the text input to be visible.
2. Wait for the model to finish preparing/downloading and for `generate-button` to become enabled.
3. Tap `generate-button`.
4. Wait for `result-card`.
5. Assert `sample-count` is greater than zero.
6. Assert `duration` is greater than zero and less than 30 seconds.

Validate objective KittenTTS behavior:

- The app loads and the text input is visible.
- The `generate-button` becomes enabled.
- Generating a short sentence completes without showing `error-banner`.
- `result-card` appears.
- `sample-count` is greater than zero.
- `duration` is greater than zero and less than 30 seconds for the smoke sentence.

The app also exposes `sample-rate` and `sample-hash` as useful manual/debug evidence. Treat them as optional in cloud automation unless they are visible without extra scrolling.

Do not use subjective audio quality as a pass/fail gate. Playback quality should be reviewed manually or in a separate evaluation flow.

Prefer these automation IDs:

- `tts-input`
- `generate-button`
- `speak-button`
- `status-banner`
- `status-label`
- `error-banner`
- `error-message`
- `result-card`
- `sample-count`
- `duration`
- `sample-rate`
- `sample-hash`
