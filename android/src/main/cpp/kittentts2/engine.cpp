#include "engine.h"

namespace kt2 {

Engine & Engine::shared() {
    static Engine engine;
    return engine;
}

Runtime & Engine::runtime() {
    if (!runtime_) {
        // KITTENTTS2_DEVICE=cpu keeps everything off the GPU, for comparisons; =gpu
        // uses a GPU the SDK leaves alone by default (see knownGpuProblem).
        runtime_ = std::make_unique<Runtime>(gpuChoiceFromEnvironment(), 0);
    }
    return *runtime_;
}

std::string Engine::device() {
    std::lock_guard<std::mutex> lock(mutex_);
    return runtime().device();
}

void Engine::load(const std::string & path, const LanguageModelConfig & config) {
    std::lock_guard<std::mutex> lock(mutex_);
    model_.reset();
    model_ = std::make_unique<LanguageModel>(runtime(), path, config);
}

void Engine::loadDecoder(const std::string & path) {
    std::lock_guard<std::mutex> lock(mutex_);
    decoder_.reset();
    decoder_ = std::make_unique<AudioDecoder>(runtime(), path);
}

std::vector<int> Engine::generate(const std::vector<int> & prompt, const std::vector<float> & speaker,
                                  const GenerateOptions & options) {
    cancelled_ = false;
    std::lock_guard<std::mutex> lock(mutex_);
    if (!model_) throw std::runtime_error("The ggml model is not loaded.");
    return generateTokens(*model_, prompt, speaker, options, &cancelled_);
}

std::vector<float> Engine::decode(const std::vector<int> & tokens, const std::vector<int> & promptToken,
                                  const std::vector<float> & promptFeat, const std::vector<float> & embedding,
                                  uint32_t seed) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!decoder_) throw std::runtime_error("The ggml decoder is not loaded.");
    return decoder_->decode(tokens, promptToken, promptFeat, embedding, seed);
}

void Engine::unload() {
    cancelled_ = true;
    std::lock_guard<std::mutex> lock(mutex_);
    model_.reset();
    decoder_.reset();
}

std::vector<int> generateTokens(LanguageModel & model, const std::vector<int> & prompt,
                                const std::vector<float> & speaker, const GenerateOptions & options,
                                const std::atomic<bool> * cancelled) {
    SamplerIds ids{model.config().headStart, options.audioStart, options.audioEnd, {options.speechEnd, options.stop}};
    Random random(options.seed);
    const int promptLength = static_cast<int>(prompt.size());
    std::vector<float> logits = model.prefill(prompt, model.speakerHidden(speaker), promptLength + options.maxNewTokens);
    std::vector<int> history = prompt;
    std::vector<int> generated;
    while (static_cast<int>(generated.size()) < options.maxNewTokens) {
        const int token = sampleToken(logits, history, promptLength, options.sampling, ids, random);
        if (token == options.speechEnd || token == options.stop) break;
        history.push_back(token);
        generated.push_back(token);
        if (static_cast<int>(generated.size()) == options.maxNewTokens) break;
        if (cancelled && *cancelled) throw Cancelled();
        logits = model.step(token);
    }
    return generated;
}

} // namespace kt2
