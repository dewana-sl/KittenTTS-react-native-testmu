// Kitten TTS 2's speech language model on ggml: the same model as the Python
// package's MLX backend (kittentts2/mlx/model.py) and ios/KittenTTS2LanguageModel.swift.
//
// The checkpoint's linear weights are ternary: in every 128-wide group each value
// is -scale, 0 or +scale. ggml's Q4_0 stores 32 weights as (code - 8) * d with one
// fp16 d, so codes 7, 8 and 9 with d = scale are those three values exactly.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "runtime.h"
#include "safetensors.h"

namespace kt2 {

struct LanguageModelConfig {
    int hiddenSize = 0;
    int layers = 0;
    int heads = 0;
    int kvHeads = 0;
    int headDim = 0;
    float ropeTheta = 1e6f;
    float rmsNormEps = 1e-6f;
    /** First token id the output head covers: the control tokens and the codec tokens follow it. */
    int headStart = 0;
};

class LanguageModel {
public:
    LanguageModel(const Runtime & runtime, const std::string & path, const LanguageModelConfig & config);
    ~LanguageModel();
    LanguageModel(const LanguageModel &) = delete;
    LanguageModel & operator=(const LanguageModel &) = delete;

    const LanguageModelConfig & config() const { return config_; }

    /** The speaker head: a 512-dim embedding becomes the hidden state of position 0. */
    std::vector<float> speakerHidden(const std::vector<float> & embedding) const;
    /**
     * Run `prompt` from an empty cache with position 0 replaced by `speaker`,
     * leaving room for `capacity` positions in all. Returns head logits.
     */
    std::vector<float> prefill(const std::vector<int> & prompt, const std::vector<float> & speaker, int capacity);
    /** One more token. Returns head logits. */
    std::vector<float> step(int token);

    /** For parity checks: when >= 0, prefill() and step() return the last position's hidden state after this layer. */
    int probeLayer = -1;
    /** For parity checks: what in that layer to return (0 layer output, 1 attention output, 2 MLP output, 3 q, 4 k, 5 v, 6 the MLP's down-projection input). */
    int probePart = 0;
    /** For parity checks: every position, not only the last. */
    bool probeAllRows = false;

private:
    struct Layer {
        ggml_tensor * inputNorm;
        ggml_tensor * postAttentionNorm;
        ggml_tensor * qNorm;
        ggml_tensor * kNorm;
        ggml_tensor * q;
        ggml_tensor * k;
        ggml_tensor * v;
        ggml_tensor * o;
        ggml_tensor * gate;
        ggml_tensor * up;
        ggml_tensor * down;
    };

    void embed(int token, float * out) const;
    void reserveCache(int capacity);
    void freeCache();
    std::vector<float> forward(const std::vector<float> & embeddings, int length);

    const Runtime & runtime_;
    LanguageModelConfig config_;
    std::unique_ptr<SafeTensors> file_;   // kept open: the embedding table is read from it
    const uint16_t * embedTable_ = nullptr;
    int64_t vocab_ = 0;
    std::vector<float> speakerWeight_, speakerBias_, speakerNormWeight_, speakerNormBias_;

    ggml_context * weightsCtx_ = nullptr;
    ggml_backend_buffer_t weights_ = nullptr;
    std::vector<Layer> layers_;
    ggml_tensor * norm_ = nullptr;
    ggml_tensor * head_ = nullptr;

    ggml_context * cacheCtx_ = nullptr;
    ggml_backend_buffer_t cache_ = nullptr;
    std::vector<ggml_tensor *> keys_, values_;
    int capacity_ = 0;
    int position_ = 0;

    ggml_backend_sched_t sched_ = nullptr;
};

} // namespace kt2
