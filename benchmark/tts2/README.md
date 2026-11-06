# Kitten TTS 2 on-device benchmark

An Expo app that runs Kitten TTS 2 on the device and shows a JSON report: load
time, the 8-sentence real-time factor per decoder, the decoder's own speed on the
same codec tokens, the first streamed chunk and cancellation. iOS runs MLX with
three decoders (`default`, `stock_1step`, `student_w4`); Android runs ggml with
`default`.

`.github/workflows/tts2-benchmark.yml` builds the model files with
`tools/kitten-tts-2-onnx/export.py`, builds this app against the SDK in this
branch, and runs it on TestMu (LambdaTest) real devices through
`e2e/tts2`. The model files are too big to ship in the app (TestMu takes apps up
to 1 GB), so the runner serves them through a LambdaTest tunnel at
`http://localhost.lambdatest.com:8788`.

`scripts/add-test-decoder.py` adds `stock_1step`, a test-only decoder;
`scripts/scene-fix.py` moves the app to the UIScene life cycle, which apps built
with the iOS 27 SDK need.
