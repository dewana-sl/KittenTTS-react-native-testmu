# KittenTTS React Native TestMu Agent Notes

Focus on the `examples/BareRNExample` app first.

Validate objective KittenTTS behavior:

- The app loads and the text input is visible.
- The `generate-button` becomes enabled.
- Generating a short sentence completes without showing `error-banner`.
- `result-card` appears.
- `sample-count` is greater than zero.
- `duration` is greater than zero and less than 30 seconds for the smoke sentence.
- `sample-rate` is 24000 Hz.
- `sample-hash` is an eight-character lowercase hex value.

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
