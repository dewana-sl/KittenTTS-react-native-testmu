// Kitten TTS 2 on ggml behind the React Native module (KittenTTS2GGMLModule.java,
// through jni.cpp): the language model and the audio decoder, with the token
// loop and sampling here, so nothing crosses to JavaScript per token. The Java
// side runs one call at a time; cancel() may come from any thread.

#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "audio_decoder.h"
#include "language_model.h"
#include "runtime.h"
#include "sampler.h"

namespace kt2 {

struct Cancelled : std::runtime_error {
    Cancelled() : std::runtime_error("The request was cancelled.") {}
};

struct GenerateOptions {
    SamplingSettings sampling;
    int speechEnd = 0;
    int stop = 0;
    int audioStart = 0;
    int audioEnd = 0;
    int maxNewTokens = 0;
    uint32_t seed = 0;
};

class Engine {
public:
    static Engine & shared();

    /** "Vulkan0: <GPU>" or "CPU": where the models run. */
    std::string device();
    void load(const std::string & path, const LanguageModelConfig & config);
    void loadDecoder(const std::string & path);
    /** The token ids written after the prompt, up to `<speech_end>` or `[STOP]`. */
    std::vector<int> generate(const std::vector<int> & prompt, const std::vector<float> & speaker,
                              const GenerateOptions & options);
    std::vector<float> decode(const std::vector<int> & tokens, const std::vector<int> & promptToken,
                              const std::vector<float> & promptFeat, const std::vector<float> & embedding,
                              uint32_t seed);
    void cancel() { cancelled_ = true; }
    void unload();

private:
    Runtime & runtime();

    std::mutex mutex_;
    std::unique_ptr<Runtime> runtime_;
    std::unique_ptr<LanguageModel> model_;
    std::unique_ptr<AudioDecoder> decoder_;
    std::atomic<bool> cancelled_{false};
};

/** Token generation with a model already loaded; the check tool uses it too. */
std::vector<int> generateTokens(LanguageModel & model, const std::vector<int> & prompt,
                                const std::vector<float> & speaker, const GenerateOptions & options,
                                const std::atomic<bool> * cancelled);

} // namespace kt2
