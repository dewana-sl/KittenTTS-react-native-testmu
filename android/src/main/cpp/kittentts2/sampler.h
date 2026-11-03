// Kitten TTS 2's decode-time sampling: the same steps, in the same order, as
// src/kittentts2/sampling.ts and ios/KittenTTS2Sampler.swift, which port the
// Python package's logits.py:
//
//     token-run penalty -> repetition penalty -> temperature
//                       -> top_k -> top_p -> min_p -> sample
//
// The scores cover token ids from `headStart` up.

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

namespace kt2 {

struct SamplingSettings {
    float temperature = 1;
    int topK = 0;
    float topP = 1;
    float minP = 0;
    float repetitionPenalty = 1;
    /** Penalise only the last N generated tokens; empty scans the whole sequence. */
    std::optional<int> repetitionWindow;
    float tokenRunPenalty = 0;
    int tokenRunGrace = 10;
};

struct SamplerIds {
    int headStart = 0;
    int audioStart = 0;
    int audioEnd = 0;
    std::vector<int> exempt;
};

/** mulberry32, the generator the JavaScript sampler uses. */
class Random {
public:
    explicit Random(uint32_t seed) : state_(seed) {}
    double next();

private:
    uint32_t state_;
};

int sampleToken(const std::vector<float> & input, const std::vector<int> & history, int promptLength,
                const SamplingSettings & settings, const SamplerIds & ids, Random & random);

} // namespace kt2
