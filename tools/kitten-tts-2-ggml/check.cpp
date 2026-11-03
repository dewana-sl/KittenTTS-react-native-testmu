// kt2check: the Android ggml engine (android/src/main/cpp/kittentts2) against
// reference outputs of the Python package, and its speed. See README.md.
//
//   kt2check device
//   kt2check lm <reference.json> <lm-ternary.safetensors>
//   kt2check decoder <decoder-reference.safetensors> <decoder-mlx.safetensors> [out.wav]
//
// KITTENTTS2_DEVICE=cpu keeps everything on the CPU.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

#include "audio_decoder.h"
#include "engine.h"
#include "json.h"
#include "language_model.h"
#include "safetensors.h"

using namespace kt2;

namespace {

double seconds(std::chrono::steady_clock::time_point since) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - since).count();
}

float maxDifference(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        std::printf("  size %zu vs %zu\n", a.size(), b.size());
        return INFINITY;
    }
    float out = 0;
    for (size_t i = 0; i < a.size(); i++) out = std::max(out, std::fabs(a[i] - b[i]));
    return out;
}

size_t nans(const std::vector<float> & values) {
    return static_cast<size_t>(std::count_if(values.begin(), values.end(), [](float v) { return std::isnan(v); }));
}

size_t argmax(const std::vector<float> & values) {
    return static_cast<size_t>(std::max_element(values.begin(), values.end()) - values.begin());
}

std::vector<int> ints(const TensorView & tensor) {
    const auto * values = reinterpret_cast<const int32_t *>(tensor.data);
    return {values, values + tensor.elements()};
}

std::unique_ptr<Runtime> makeRuntime() {
    const char * device = std::getenv("KITTENTTS2_DEVICE");
    auto runtime = std::make_unique<Runtime>(!(device && std::strcmp(device, "cpu") == 0), 0);
    std::printf("device: %s\n", runtime->device().c_str());
    return runtime;
}

void writeWav(const std::string & path, const std::vector<float> & samples, int rate) {
    std::ofstream out(path, std::ios::binary);
    auto u32 = [&](uint32_t v) { out.write(reinterpret_cast<const char *>(&v), 4); };
    auto u16 = [&](uint16_t v) { out.write(reinterpret_cast<const char *>(&v), 2); };
    const auto bytes = static_cast<uint32_t>(samples.size() * 2);
    out.write("RIFF", 4); u32(36 + bytes); out.write("WAVEfmt ", 8);
    u32(16); u16(1); u16(1); u32(static_cast<uint32_t>(rate)); u32(static_cast<uint32_t>(rate * 2)); u16(2); u16(16);
    out.write("data", 4); u32(bytes);
    for (float s : samples) u16(static_cast<uint16_t>(static_cast<int16_t>(std::lround(std::clamp(s, -1.0f, 1.0f) * 32767))));
}

int checkLanguageModel(const std::string & referencePath, const std::string & weightsPath) {
    std::ifstream file(referencePath);
    std::stringstream text;
    text << file.rdbuf();
    const Json reference = Json::parse(text.str());
    const Json & lm = reference["config"];
    LanguageModelConfig config;
    config.hiddenSize = static_cast<int>(lm["hidden_size"].integer());
    config.layers = static_cast<int>(lm["num_hidden_layers"].integer());
    config.heads = static_cast<int>(lm["num_attention_heads"].integer());
    config.kvHeads = static_cast<int>(lm["num_key_value_heads"].integer());
    config.headDim = static_cast<int>(lm["head_dim"].integer());
    config.ropeTheta = static_cast<float>(lm["rope_theta"].num());
    config.rmsNormEps = static_cast<float>(lm["rms_norm_eps"].num());
    config.headStart = static_cast<int>(reference["head_start"].integer());

    auto runtime = makeRuntime();
    auto started = std::chrono::steady_clock::now();
    LanguageModel model(*runtime, weightsPath, config);
    std::printf("load %.2fs\n", seconds(started));

    std::vector<int> prompt;
    for (const auto & id : reference["prompt"].array) prompt.push_back(static_cast<int>(id.integer()));
    std::vector<float> speaker;
    for (const auto & value : reference["speaker"].array) speaker.push_back(static_cast<float>(value.num()));
    std::vector<std::vector<float>> got{model.prefill(prompt, model.speakerHidden(speaker), static_cast<int>(prompt.size()) + 8)};
    for (const auto & token : reference["steps"].array) got.push_back(model.step(static_cast<int>(token.integer())));
    int failures = 0;
    for (size_t i = 0; i < got.size(); i++) {
        std::vector<float> expected;
        for (const auto & value : reference["logits"][i].array) expected.push_back(static_cast<float>(value.num()));
        const size_t ours = argmax(got[i]) + config.headStart, theirs = argmax(expected) + config.headStart;
        std::printf("step %zu: max |logit difference| %.4f, top token %zu vs Python MLX %zu\n", i,
                    maxDifference(got[i], expected), ours, theirs);
        if (ours != theirs) failures++;
    }

    // Generation speed with the stable preset.
    GenerateOptions options;
    options.sampling = SamplingSettings{0.8f, 50, 0.8f, 0, 1.1f, 50, 1.3f, 10};
    options.speechEnd = 151670;
    options.stop = 151674;
    options.audioStart = 151675;
    options.audioEnd = 151675 + 6561;
    options.maxNewTokens = 300;
    options.seed = 1;
    started = std::chrono::steady_clock::now();
    model.prefill(prompt, model.speakerHidden(speaker), static_cast<int>(prompt.size()) + 8);
    const double prefill = seconds(started);
    started = std::chrono::steady_clock::now();
    const std::vector<int> tokens = generateTokens(model, prompt, speaker, options, nullptr);
    const double total = seconds(started);
    const double decode = total - prefill;
    std::printf("prefill %zu tokens %.2fs, decode %zu tokens %.2fs = %.1f ms/token (%.2f x realtime at 25 tokens/s)\n",
                prompt.size(), prefill, tokens.size(), decode, decode / tokens.size() * 1000,
                decode / (static_cast<double>(tokens.size()) / 25));
    return failures == 0 ? 0 : 1;
}

int checkDecoder(const std::string & referencePath, const std::string & weightsPath, const char * wavPath) {
    SafeTensors reference(referencePath);
    auto runtime = makeRuntime();
    auto started = std::chrono::steady_clock::now();
    AudioDecoder decoder(*runtime, weightsPath);
    std::printf("decoder load %.2fs\n", seconds(started));

    const std::vector<int> tokens = ints(reference.get("tokens"));
    const std::vector<int> prompt = ints(reference.get("prompt_token"));
    const std::vector<float> promptFeat = toFloat(reference.get("prompt_feat"));
    const std::vector<float> embedding = toFloat(reference.get("embedding"));
    const std::vector<float> mel = toFloat(reference.get("mel"));

    started = std::chrono::steady_clock::now();
    const std::vector<float> ours = decoder.flow(tokens, prompt, promptFeat, embedding, toFloat(reference.get("flow_noise")));
    const float melDifference = maxDifference(ours, mel);
    std::printf("flow: %zu mel frames in %.2fs, max |difference| from torch %.4f\n", ours.size() / 80, seconds(started),
                melDifference);

    started = std::chrono::steady_clock::now();
    const std::vector<float> f0 = decoder.predictF0(mel);
    const std::vector<float> wav = decoder.vocoder(mel, f0, toFloat(reference.get("phase")),
                                                   toFloat(reference.get("source_noise")));
    const float wavDifference = maxDifference(wav, toFloat(reference.get("wav")));
    std::printf("hift: %zu samples in %.2fs, max |difference| from torch %.4f\n", wav.size(), seconds(started),
                wavDifference);

    std::vector<float> audio;
    for (int i = 0; i < 2; i++) {
        started = std::chrono::steady_clock::now();
        audio = decoder.decode(tokens, prompt, promptFeat, embedding, 1);
        const double took = seconds(started), length = audio.size() / 24000.0;
        std::printf("decode %.2fs of audio in %.3fs (%.2f x realtime)\n", length, took, took / length);
    }
    if (wavPath) writeWav(wavPath, audio, 24000);
    return melDifference < 0.1f && wavDifference < 0.05f ? 0 : 1;
}

/** The reference's language model config. */
LanguageModelConfig configOf(const Json & reference) {
    const Json & lm = reference["config"];
    LanguageModelConfig config;
    config.hiddenSize = static_cast<int>(lm["hidden_size"].integer());
    config.layers = static_cast<int>(lm["num_hidden_layers"].integer());
    config.heads = static_cast<int>(lm["num_attention_heads"].integer());
    config.kvHeads = static_cast<int>(lm["num_key_value_heads"].integer());
    config.headDim = static_cast<int>(lm["head_dim"].integer());
    config.ropeTheta = static_cast<float>(lm["rope_theta"].num());
    config.rmsNormEps = static_cast<float>(lm["rms_norm_eps"].num());
    config.headStart = static_cast<int>(reference["head_start"].integer());
    return config;
}

/**
 * Prefill on the GPU against the CPU, for prompts cut to a few lengths
 * (KT2_LENGTHS=a,b,...), or per layer at one length (KT2_LM_LAYERS=n).
 * KT2_LM_SWIGLU=n prints, on the CPU only, the largest input to each layer's
 * down projection: the values float16 has to hold.
 */
int compareLanguageModel(const std::string & referencePath, const std::string & weightsPath) {
    std::ifstream file(referencePath);
    std::stringstream text;
    text << file.rdbuf();
    const Json reference = Json::parse(text.str());
    const LanguageModelConfig config = configOf(reference);
    std::vector<int> prompt;
    for (const auto & id : reference["prompt"].array) prompt.push_back(static_cast<int>(id.integer()));
    std::vector<float> speaker;
    for (const auto & value : reference["speaker"].array) speaker.push_back(static_cast<float>(value.num()));
    auto cut = [&](int length) {
        return std::vector<int>(prompt.begin(), prompt.begin() + std::min<size_t>(length, prompt.size()));
    };

    Runtime cpu(false, 0);
    LanguageModel onCpu(cpu, weightsPath, config);
    if (const char * env = std::getenv("KT2_LM_SWIGLU")) {
        const int length = std::atoi(env);
        for (int layer = 0; layer < config.layers; layer++) {
            onCpu.probeLayer = layer;
            onCpu.probePart = 6;
            onCpu.probeAllRows = true;
            const std::vector<float> b = onCpu.prefill(cut(length), onCpu.speakerHidden(speaker), length + 8);
            float top = 0;
            size_t over = 0;
            for (float value : b) {
                top = std::max(top, std::fabs(value));
                if (std::fabs(value) > 65504) over++;
            }
            std::printf("layer %2d: max |swiglu| %12.1f, %zu values beyond float16\n", layer, top, over);
        }
        return 0;
    }

    Runtime gpu(true, 0);
    LanguageModel onGpu(gpu, weightsPath, config);
    if (const char * env = std::getenv("KT2_LM_LAYERS")) {
        const int length = std::atoi(env);
        static const char * names[] = {"output", "attention", "mlp"};
        for (int layer = 0; layer < config.layers; layer++) {
            for (int part = 0; part < 3; part++) {
                onGpu.probeLayer = onCpu.probeLayer = layer;
                onGpu.probePart = onCpu.probePart = part;
                const std::vector<float> a = onGpu.prefill(cut(length), onGpu.speakerHidden(speaker), length + 8);
                const std::vector<float> b = onCpu.prefill(cut(length), onCpu.speakerHidden(speaker), length + 8);
                float scale = 0;
                for (float value : b) scale = std::max(scale, std::fabs(value));
                std::printf("layer %2d %-9s max |GPU - CPU| %8.4f of %8.3f, NaN %zu / %zu\n", layer, names[part],
                            maxDifference(a, b), scale, nans(a), nans(b));
            }
        }
        return 0;
    }
    std::vector<int> lengths{2, 8, 16, 32, 33, 64, 65, 128, 200, 297};
    if (const char * env = std::getenv("KT2_LENGTHS")) {
        lengths.clear();
        for (std::stringstream list(env); list.good();) {
            std::string item;
            std::getline(list, item, ',');
            if (!item.empty()) lengths.push_back(std::atoi(item.c_str()));
        }
    }
    for (int length : lengths) {
        const std::vector<float> a = onGpu.prefill(cut(length), onGpu.speakerHidden(speaker), length + 8);
        const std::vector<float> b = onCpu.prefill(cut(length), onCpu.speakerHidden(speaker), length + 8);
        std::printf("prefill %3d tokens: max |GPU - CPU| %.4f, top %zu vs %zu, NaN %zu / %zu\n", length, maxDifference(a, b),
                    argmax(a), argmax(b), nans(a), nans(b));
    }
    return 0;
}

/** The flow's intermediates against torch's (mlxcheck's intermediates.safetensors), to find where a backend differs. */
int probeFlow(const std::string & referencePath, const std::string & intermediatesPath, const std::string & weightsPath) {
    SafeTensors reference(referencePath), intermediates(intermediatesPath);
    auto runtime = makeRuntime();
    AudioDecoder decoder(*runtime, weightsPath);
    for (const char * name : {"embed", "lookahead", "layer0", "encoders", "mu", "time_t0", "velocity0"}) {
        decoder.probe = name;
        const std::vector<float> ours = decoder.flow(ints(reference.get("tokens")), ints(reference.get("prompt_token")),
                                                     toFloat(reference.get("prompt_feat")), toFloat(reference.get("embedding")),
                                                     toFloat(reference.get("flow_noise")));
        const std::vector<float> theirs = toFloat(intermediates.get(name));
        float scale = 0;
        for (float value : theirs) scale = std::max(scale, std::fabs(value));
        std::printf("%-10s max |difference| %.4f (max |value| %.2f)\n", name, maxDifference(ours, theirs), scale);
    }
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc >= 2 && std::strcmp(argv[1], "device") == 0) {
            makeRuntime();
            return 0;
        }
        if (argc >= 4 && std::strcmp(argv[1], "lm") == 0) return checkLanguageModel(argv[2], argv[3]);
        if (argc >= 4 && std::strcmp(argv[1], "decoder") == 0) return checkDecoder(argv[2], argv[3], argc > 4 ? argv[4] : nullptr);
        if (argc >= 5 && std::strcmp(argv[1], "probe") == 0) return probeFlow(argv[2], argv[3], argv[4]);
        if (argc >= 4 && std::strcmp(argv[1], "lm-compare") == 0) return compareLanguageModel(argv[2], argv[3]);
        if (argc >= 6 && std::strcmp(argv[1], "dump") == 0) {
            // kt2check dump <decoder-reference.safetensors> <decoder weights> <intermediate> <out.f32>
            SafeTensors reference(argv[2]);
            auto runtime = makeRuntime();
            AudioDecoder decoder(*runtime, argv[3]);
            decoder.probe = argv[4];
            const std::vector<float> out = decoder.flow(ints(reference.get("tokens")), ints(reference.get("prompt_token")),
                                                        toFloat(reference.get("prompt_feat")), toFloat(reference.get("embedding")),
                                                        toFloat(reference.get("flow_noise")));
            std::ofstream(argv[5], std::ios::binary).write(reinterpret_cast<const char *>(out.data()), out.size() * 4);
            std::printf("%s: %zu values\n", argv[4], out.size());
            return 0;
        }
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 2;
    }
    std::fprintf(stderr, "usage: kt2check device | lm <reference.json> <lm-ternary.safetensors> | "
                         "decoder <decoder-reference.safetensors> <decoder-mlx.safetensors> [out.wav]\n");
    return 2;
}
