// Kitten TTS 2's audio decoder on ggml: chatterbox-turbo's s3gen, as in
// ios/KittenTTS2AudioDecoder.swift. Codec tokens become a mel spectrogram (the
// flow: a conformer encoder, then two meanflow steps of a UNet estimator), and
// the mel becomes 24 kHz audio (the HiFT vocoder).
//
// It reads the same weights as the MLX decoder (tools/kitten-tts-2-onnx/decoder_mlx.py):
// float16, convolutions folded and stored [out, kernel, in], which loading turns
// into torch's [out, in, kernel] for ggml's direct convolution. The flow's activations
// are channels-last, [channels, time] in ggml's order, because it is mostly
// linear layers; the vocoder's are channels-first, [time, channels], because it
// is all convolutions. The sine source and the last overlap-add run on the CPU.

#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

#include "runtime.h"

namespace kt2 {

struct DecoderNoise {
    std::vector<float> flow;      // [frames, 80]
    std::vector<float> phase;     // [9]
    std::vector<float> source;    // [samples, 9]
};

class AudioDecoder {
public:
    AudioDecoder(const Runtime & runtime, const std::string & path);
    ~AudioDecoder();
    AudioDecoder(const AudioDecoder &) = delete;
    AudioDecoder & operator=(const AudioDecoder &) = delete;

    /**
     * Codec tokens (lookahead padding included) to audio samples. `promptFeat`
     * is [frames, 80]. `noise` exists for parity checks; normally it is drawn from `seed`.
     */
    std::vector<float> decode(const std::vector<int> & tokens, const std::vector<int> & promptToken,
                              const std::vector<float> & promptFeat, const std::vector<float> & embedding,
                              uint64_t seed, const DecoderNoise * noise = nullptr);

    /** The flow alone: mel frames [frames, 80], the prompt's frames removed. */
    std::vector<float> flow(const std::vector<int> & tokens, const std::vector<int> & promptToken,
                            const std::vector<float> & promptFeat, const std::vector<float> & embedding,
                            const std::vector<float> & noise, std::vector<float> * f0 = nullptr);
    /** The vocoder alone: mel frames [frames, 80] to samples, given f0 per frame. */
    std::vector<float> vocoder(const std::vector<float> & mel, const std::vector<float> & f0,
                               const std::vector<float> & phase, const std::vector<float> & sourceNoise);
    /** f0 per frame, as the vocoder's predictor finds it. */
    std::vector<float> predictF0(const std::vector<float> & mel);

    /** For parity checks: flow() returns this intermediate (embed, lookahead, layer0, encoders, mu, time_t0, velocity0) instead. */
    std::string probe;

private:
    struct Builder;

    ggml_tensor * weight(const std::string & name) const;
    ggml_tensor * meanVector(int64_t width) const;

    const Runtime & runtime_;
    ggml_context * weightsCtx_ = nullptr;
    ggml_backend_buffer_t weights_ = nullptr;
    // The vocoder's weights (mel2wav.*, the STFT bases): in the CPU's memory when
    // the vocoder runs there, which it does when the GPU lacks ggml_sin: its snake
    // activations would send every block to the CPU and back.
    bool vocoderOnCpu_ = false;
    ggml_context * vocoderCtx_ = nullptr;
    ggml_backend_buffer_t vocoderWeights_ = nullptr;
    std::map<std::string, ggml_tensor *> tensors_;
    std::map<int64_t, ggml_tensor *> means_;
    std::vector<float> sourceWeight_;   // m_source.l_linear, [9]
    float sourceBias_ = 0;
};

} // namespace kt2
