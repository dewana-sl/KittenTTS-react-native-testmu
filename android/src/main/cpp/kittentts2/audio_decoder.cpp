#include "audio_decoder.h"

#include "safetensors.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace kt2 {

namespace {

constexpr int kMels = 80;
constexpr int kFft = 16;
constexpr int kHop = 4;
constexpr int kBins = kFft / 2 + 1;
constexpr int kHarmonics = 9;
constexpr int kSamplesPerFrame = 480;
constexpr double kSampleRate = 24000;
constexpr double kPi = 3.14159265358979323846;
constexpr size_t kGraphSize = 32768;

bool endsWith(const std::string & text, const std::string & suffix) {
    return text.size() >= suffix.size() && text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool isTransposedConv(const std::string & name) {
    return name.rfind("mel2wav.ups.", 0) == 0 && endsWith(name, ".weight");
}

/** splitmix64, for the decoder's noise. */
class Noise {
public:
    explicit Noise(uint64_t seed) : state_(seed) {}

    double uniform() {
        uint64_t z = (state_ += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= z >> 31;
        return static_cast<double>(z >> 11) * (1.0 / 9007199254740992.0);
    }

    std::vector<float> normal(size_t count) {
        std::vector<float> out(count);
        for (size_t i = 0; i < count; i += 2) {
            const double u1 = std::max(uniform(), 1e-300), u2 = uniform();
            const double radius = std::sqrt(-2.0 * std::log(u1));
            out[i] = static_cast<float>(radius * std::cos(2 * kPi * u2));
            if (i + 1 < count) out[i + 1] = static_cast<float>(radius * std::sin(2 * kPi * u2));
        }
        return out;
    }

private:
    uint64_t state_;
};

std::vector<float> hannWindow() {
    std::vector<float> window(kFft);
    for (int i = 0; i < kFft; i++) window[i] = static_cast<float>(0.5 - 0.5 * std::cos(2 * kPi * i / kFft));
    return window;
}

/** Relative positions length - 1 down to -(length - 1), sines and cosines interleaved: [512, 2L - 1]. */
std::vector<float> positionalEncoding(int64_t length, int dims = 512) {
    std::vector<float> div(dims / 2);
    for (int i = 0; i < dims / 2; i++) div[i] = static_cast<float>(std::exp(2.0 * i * -(std::log(10000.0) / dims)));
    std::vector<float> out(static_cast<size_t>((2 * length - 1) * dims));
    for (int64_t p = 0; p < 2 * length - 1; p++) {
        const auto position = static_cast<float>(length - 1 - p);
        for (int i = 0; i < dims / 2; i++) {
            const float angle = position * div[i];
            out[p * dims + 2 * i] = std::sin(angle);
            out[p * dims + 2 * i + 1] = std::cos(angle);
        }
    }
    return out;
}

/** The meanflow time step's sinusoid, [320]. */
std::vector<float> timeSinusoid(float t) {
    const int half = 160;
    const double factor = std::log(10000.0) / (half - 1);
    std::vector<float> out(2 * half);
    for (int i = 0; i < half; i++) {
        const float angle = 1000.0f * t * static_cast<float>(std::exp(i * -factor));
        out[i] = std::sin(angle);
        out[half + i] = std::cos(angle);
    }
    return out;
}

void set(ggml_tensor * tensor, const std::vector<float> & values) {
    if (values.size() * sizeof(float) != ggml_nbytes(tensor)) throw std::runtime_error("Input size mismatch");
    ggml_backend_tensor_set(tensor, values.data(), 0, ggml_nbytes(tensor));
}

std::vector<float> get(ggml_tensor * tensor) {
    std::vector<float> out(static_cast<size_t>(ggml_nelements(tensor)));
    ggml_backend_tensor_get(tensor, out.data(), 0, ggml_nbytes(tensor));
    return out;
}

/** A scheduler for one call: its buffers are freed when the call ends, as the MLX decoder clears its cache. */
struct Scheduler {
    explicit Scheduler(const Runtime & runtime, bool cpuOnly = false) : sched(runtime.newScheduler(kGraphSize, cpuOnly)) {}
    ~Scheduler() { ggml_backend_sched_free(sched); }
    ggml_backend_sched_t sched;
};

} // namespace

AudioDecoder::AudioDecoder(const Runtime & runtime, const std::string & path) : runtime_(runtime) {
    SafeTensors file(path);
    auto format = file.metadata().find("format");
    if (format == file.metadata().end() || format->second != "kittentts-s3gen-mlx-1") {
        throw std::runtime_error("Not Kitten TTS 2 decoder weights: " + path);
    }
    for (const char * name : {"flow.input_embedding.weight", "mel2wav.conv_post.weight"}) file.get(name);

    ggml_init_params params{ggml_tensor_overhead() * (2 * file.tensors().size() + 64), nullptr, true};
    weightsCtx_ = ggml_init(params);
    vocoderCtx_ = ggml_init(params);
    vocoderOnCpu_ = runtime.hasGpu() && !runtime.gpuRunsSine();
    auto add = [&](const std::string & name, ggml_type type, std::vector<int64_t> ne) {
        const bool vocoder = name.rfind("mel2wav.", 0) == 0 || endsWith(name, "stft.basis");
        ggml_tensor * t = ggml_new_tensor(vocoder ? vocoderCtx_ : weightsCtx_, type, static_cast<int>(ne.size()), ne.data());
        ggml_set_name(t, name.c_str());
        tensors_[name] = t;
        return t;
    };
    for (const auto & [name, view] : file.tensors()) {
        std::vector<int64_t> ne(view.shape.rbegin(), view.shape.rend());
        if (isTransposedConv(name)) {
            // Stored [out, kernel, in]; ggml_conv_transpose_1d takes torch's [in, out, kernel].
            add(name, GGML_TYPE_F32, {view.shape[1], view.shape[0], view.shape[2]});
        } else if (view.shape.size() == 3) {
            // Stored [out, kernel, in]; ggml's CPU convolution takes torch's [out, in, kernel],
            // contiguous ([kernel, in, out] in ggml's order).
            add(name, GGML_TYPE_F16, {view.shape[1], view.shape[2], view.shape[0]});
        } else if (view.shape.size() == 1 || endsWith(name, "pos_bias_u") || endsWith(name, "pos_bias_v")) {
            add(name, GGML_TYPE_F32, ne);
            if (endsWith(name, ".alpha")) add(name + ".inverse", GGML_TYPE_F32, ne);
        } else {
            add(name, GGML_TYPE_F16, ne);
        }
    }
    // 1/C in every place, [C, 1], for layerNorm's mean over C channels.
    std::vector<int64_t> widths;
    for (const auto & [name, view] : file.tensors()) {
        if (view.shape.size() == 1 && std::find(widths.begin(), widths.end(), view.shape[0]) == widths.end()) {
            widths.push_back(view.shape[0]);
        }
    }
    for (int64_t width : widths) means_[width] = add("mean." + std::to_string(width), GGML_TYPE_F32, {width, 1});
    // A 16-point STFT as a convolution weight [out 18, in 1, kernel 16], and its inverse [16, 18].
    ggml_tensor * stft = add("stft.basis", GGML_TYPE_F32, {kFft, 1, 2 * kBins});
    ggml_tensor * istft = add("istft.basis", GGML_TYPE_F32, {2 * kBins, kFft});
    weights_ = allocateTensors(weightsCtx_, runtime.weightBuffer(), "audio decoder");
    vocoderWeights_ = allocateTensors(
        vocoderCtx_, vocoderOnCpu_ ? ggml_backend_get_default_buffer_type(runtime.cpu()) : runtime.weightBuffer(), "vocoder");

    for (const auto & [name, view] : file.tensors()) {
        ggml_tensor * t = tensors_.at(name);
        if (t->type == GGML_TYPE_F16) {
            if (view.dtype != "F16" || view.bytes != ggml_nbytes(t)) throw std::runtime_error(name + " is not float16");
            if (view.shape.size() != 3) {
                ggml_backend_tensor_set(t, view.data, 0, view.bytes);
                continue;
            }
            const int64_t out = view.shape[0], kernel = view.shape[1], in = view.shape[2];
            const auto * source = reinterpret_cast<const ggml_fp16_t *>(view.data);
            std::vector<ggml_fp16_t> torch(static_cast<size_t>(out * kernel * in));
            for (int64_t o = 0; o < out; o++)
                for (int64_t k = 0; k < kernel; k++)
                    for (int64_t i = 0; i < in; i++) torch[(o * in + i) * kernel + k] = source[(o * kernel + k) * in + i];
            ggml_backend_tensor_set(t, torch.data(), 0, view.bytes);
            continue;
        }
        std::vector<float> values = toFloat(view);
        if (isTransposedConv(name)) {
            const int64_t out = view.shape[0], kernel = view.shape[1], in = view.shape[2];
            std::vector<float> torch(values.size());
            for (int64_t o = 0; o < out; o++)
                for (int64_t k = 0; k < kernel; k++)
                    for (int64_t i = 0; i < in; i++) torch[(i * out + o) * kernel + k] = values[(o * kernel + k) * in + i];
            values.swap(torch);
        }
        set(t, values);
        if (endsWith(name, ".alpha")) {
            for (float & value : values) value = 1.0f / (value + 1e-9f);
            set(tensors_.at(name + ".inverse"), values);
        }
    }
    const std::vector<float> window = hannWindow();
    std::vector<float> forward(2 * kBins * kFft), inverse(kFft * 2 * kBins);
    for (int k = 0; k < kBins; k++) {
        const double scale = (k == 0 || k == kBins - 1) ? 1.0 : 2.0;
        for (int t = 0; t < kFft; t++) {
            const double angle = 2 * kPi * k * t / kFft;
            forward[k * kFft + t] = static_cast<float>(std::cos(angle) * window[t]);
            forward[(kBins + k) * kFft + t] = static_cast<float>(-std::sin(angle) * window[t]);
            inverse[t * 2 * kBins + k] = static_cast<float>(std::cos(angle) * scale / kFft * window[t]);
            const double sine = (k == 0 || k == kBins - 1) ? 0 : -std::sin(angle) * scale / kFft;
            inverse[t * 2 * kBins + kBins + k] = static_cast<float>(sine * window[t]);
        }
    }
    set(stft, forward);
    set(istft, inverse);
    for (const auto & [width, tensor] : means_) set(tensor, std::vector<float>(static_cast<size_t>(width), 1.0f / width));

    sourceWeight_ = toFloat(file.get("mel2wav.m_source.l_linear.weight"));
    sourceBias_ = toFloat(file.get("mel2wav.m_source.l_linear.bias")).at(0);
    if (sourceWeight_.size() != kHarmonics) throw std::runtime_error("Unexpected m_source weights");
}

AudioDecoder::~AudioDecoder() {
    if (weights_) ggml_backend_buffer_free(weights_);
    if (weightsCtx_) ggml_free(weightsCtx_);
    if (vocoderWeights_) ggml_backend_buffer_free(vocoderWeights_);
    if (vocoderCtx_) ggml_free(vocoderCtx_);
}

ggml_tensor * AudioDecoder::meanVector(int64_t width) const {
    auto it = means_.find(width);
    if (it == means_.end()) throw std::runtime_error("No mean vector for " + std::to_string(width) + " channels");
    return it->second;
}

ggml_tensor * AudioDecoder::weight(const std::string & name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) throw std::runtime_error("The decoder weights have no " + name);
    return it->second;
}

// MARK: - Layers

struct AudioDecoder::Builder {
    const AudioDecoder & decoder;
    Graph & g;
    ggml_context * ctx;
    std::map<std::string, ggml_tensor *> probes;   // intermediates by name, for AudioDecoder::probe

    ggml_tensor * w(const std::string & name) const { return decoder.weight(name); }
    /** Copies `t` where it is computed, when it is the decoder's probe, so later work cannot change it. */
    ggml_tensor * mark(const char * name, ggml_tensor * t) {
        if (decoder.probe == name) {
            ggml_tensor * copy = ggml_cont(ctx, t);
            ggml_set_name(copy, "probe");
            ggml_set_output(copy);
            g.expand(copy);
            probes[name] = copy;
        }
        return t;
    }

    ggml_tensor * linear(ggml_tensor * x, const std::string & name, bool bias = true) {
        ggml_tensor * y = matmul(ctx, w(name + ".weight"), x);
        return bias ? ggml_add(ctx, y, w(name + ".bias")) : y;
    }

    /**
     * Over the channels of a channels-last tensor: the mean taken off with a
     * matmul, then ggml_rms_norm, which is the same normalisation. ggml_norm's
     * Vulkan shader runs padding rows that write past the tensor, which only
     * GPUs that drop out-of-bounds writes survive (the emulator's does not).
     * The mean is a matrix-vector product, x's rows by 1/C: as one row times x,
     * ggml-vulkan runs it with its small tiles, which give wrong sums in float32
     * on Arm Mali GPUs.
     */
    ggml_tensor * layerNorm(ggml_tensor * x, const std::string & name, float eps) {
        ggml_tensor * rows = ggml_is_contiguous(x) ? x : ggml_cont(ctx, x);
        ggml_tensor * mean = ggml_reshape_2d(ctx, matmul(ctx, rows, decoder.meanVector(x->ne[0])), 1, x->ne[1]);  // [1, time]
        ggml_tensor * normalised = ggml_rms_norm(ctx, ggml_sub(ctx, x, mean), eps);
        return ggml_add(ctx, ggml_mul(ctx, normalised, w(name + ".weight")), w(name + ".bias"));
    }

    /** [dk, heads, time] as flash attention's [dk, time, heads], without copying. */
    ggml_tensor * byHead(ggml_tensor * t) { return ggml_permute(ctx, t, 0, 2, 1, 3); }

    /** Attention over every head at once; `mask` is added to the scaled scores. Returns [dk * heads, time]. */
    ggml_tensor * attention(ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, ggml_tensor * mask, float scale) {
        ggml_tensor * out = ggml_flash_attn_ext(ctx, byHead(q), byHead(k), byHead(v), mask, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
        return ggml_reshape_2d(ctx, out, out->ne[0] * out->ne[1], out->ne[2]);
    }

    /** A [kernel, in, out] weight as ggml_conv_2d_direct's [kernel, 1, in, out]. */
    ggml_tensor * kernelView(ggml_tensor * k) {
        return ggml_reshape_4d(ctx, k, k->ne[0], 1, k->ne[1], k->ne[2]);
    }

    /** [time, 1, in, 1] input to channels-first [time', out], no bias. */
    ggml_tensor * convolve(ggml_tensor * input, ggml_tensor * kernel, int stride, int padding, int dilation) {
        ggml_tensor * y = ggml_conv_2d_direct(ctx, kernelView(kernel), input, stride, 1, padding, 0, dilation, 1);
        return ggml_reshape_2d(ctx, y, y->ne[0], y->ne[2]);
    }

    ggml_tensor * channelBias(const std::string & name) {
        ggml_tensor * b = w(name + ".bias");
        return ggml_reshape_2d(ctx, b, 1, b->ne[0]);
    }

    /** Channels-first [time, channels] convolution with symmetric padding. */
    ggml_tensor * convCF(ggml_tensor * x, const std::string & name, int padding = 0, int stride = 1, int dilation = 1) {
        if (!ggml_is_contiguous(x)) x = ggml_cont(ctx, x);
        ggml_tensor * input = ggml_reshape_4d(ctx, x, x->ne[0], 1, x->ne[1], 1);
        return ggml_add(ctx, convolve(input, w(name + ".weight"), stride, padding, dilation), channelBias(name));
    }

    /** Channels-last [channels, time] convolution, stride 1, padded `left` and `right` zeros. */
    ggml_tensor * convCL(ggml_tensor * x, const std::string & name, int left, int right) {
        ggml_tensor * kernel = w(name + ".weight");
        const int64_t size = kernel->ne[0];
        if (size == 1 && left == 0 && right == 0) {
            ggml_tensor * matrix = ggml_reshape_2d(ctx, kernel, kernel->ne[1], kernel->ne[2]);
            return ggml_add(ctx, matmul(ctx, matrix, x), w(name + ".bias"));
        }
        const int64_t length = x->ne[1];
        const int padding = std::max(left, right);
        // im2col of the channels-first input, [in * kernel, time'], then one matmul,
        // which leaves the result channels-last.
        ggml_tensor * columns = ggml_im2col(ctx, kernel, ggml_cont(ctx, ggml_transpose(ctx, x)), 1, 0, padding, 0, 1, 0,
                                            false, GGML_TYPE_F32);
        ggml_tensor * matrix = ggml_reshape_2d(ctx, kernel, size * kernel->ne[1], kernel->ne[2]);
        ggml_tensor * y = matmul(ctx, matrix, ggml_reshape_2d(ctx, columns, columns->ne[0], columns->ne[1]));
        const int64_t outLength = length + left + right - size + 1;
        if (outLength != y->ne[1]) {
            y = ggml_view_2d(ctx, y, y->ne[0], outLength, y->nb[1], static_cast<size_t>(padding - left) * y->nb[1]);
        }
        return ggml_add(ctx, y, w(name + ".bias"));
    }

    /** Channels-first transposed convolution: torch's padding crops the full output on both sides. */
    ggml_tensor * convTranspose(ggml_tensor * x, const std::string & name, int stride, int padding) {
        ggml_tensor * y = ggml_conv_transpose_1d(ctx, w(name + ".weight"), x, stride, 0, 1);
        y = ggml_view_2d(ctx, y, y->ne[0] - 2 * padding, y->ne[1], y->nb[1], static_cast<size_t>(padding) * y->nb[0]);
        return ggml_add(ctx, y, channelBias(name));
    }

    ggml_tensor * mish(ggml_tensor * x) { return ggml_mul(ctx, x, ggml_tanh(ctx, ggml_softplus(ctx, x))); }

    /** Nearest-neighbour upsampling of a channels-last tensor's time axis, by 2. */
    ggml_tensor * doubleTime(ggml_tensor * x) {
        ggml_tensor * x3 = ggml_reshape_3d(ctx, x, x->ne[0], 1, x->ne[1]);
        x3 = ggml_repeat_4d(ctx, x3, x->ne[0], 2, x->ne[1], 1);
        return ggml_reshape_2d(ctx, x3, x->ne[0], 2 * x->ne[1]);
    }

    // MARK: Flow

    ggml_tensor * conformerLayer(ggml_tensor * x, ggml_tensor * pos, const std::string & p) {
        const int heads = 8, dk = 64;
        const int64_t length = x->ne[1], positions = pos->ne[1];
        ggml_tensor * h = layerNorm(x, p + ".norm_mha", 1e-12f);
        ggml_tensor * q = ggml_reshape_3d(ctx, linear(h, p + ".self_attn.linear_q"), dk, heads, length);
        ggml_tensor * k = ggml_reshape_3d(ctx, linear(h, p + ".self_attn.linear_k"), dk, heads, length);
        ggml_tensor * v = ggml_reshape_3d(ctx, linear(h, p + ".self_attn.linear_v"), dk, heads, length);
        ggml_tensor * pe = ggml_reshape_3d(ctx, linear(pos, p + ".self_attn.linear_pos", false), dk, heads, positions);
        ggml_tensor * qu = ggml_add(ctx, q, ggml_reshape_3d(ctx, w(p + ".self_attn.pos_bias_u"), dk, heads, 1));
        ggml_tensor * qv = ggml_add(ctx, q, ggml_reshape_3d(ctx, w(p + ".self_attn.pos_bias_v"), dk, heads, 1));
        const float scale = 1.0f / std::sqrt(static_cast<float>(dk));
        // The position scores of every head, [2L-1, L, heads], then Transformer-XL's
        // relative shift: a zero column in front, and each query's row read L
        // elements further on. Flash attention adds them, scaled, to the content
        // scores: softmax((q_u k + q_v pe) / sqrt(dk)) v.
        ggml_tensor * bd = matmul(ctx, ggml_cont(ctx, byHead(pe)), ggml_cont(ctx, byHead(qv)));
        bd = ggml_pad_ext(ctx, bd, 1, 0, 0, 0, 0, 0, 0, 0);
        bd = ggml_view_3d(ctx, bd, length, length, heads, (2 * length - 1) * sizeof(float), bd->nb[2], length * sizeof(float));
        ggml_tensor * mask = ggml_cast(ctx, ggml_scale(ctx, ggml_cont(ctx, bd), scale), GGML_TYPE_F16);
        ggml_tensor * attended = attention(qu, k, v, mask, scale);
        x = ggml_add(ctx, x, linear(attended, p + ".self_attn.linear_out"));
        ggml_tensor * f = layerNorm(x, p + ".norm_ff", 1e-12f);
        return ggml_add(ctx, x, linear(ggml_silu(ctx, linear(f, p + ".feed_forward.w_1")), p + ".feed_forward.w_2"));
    }

    ggml_tensor * encoder(ggml_tensor * x, ggml_tensor * pos, ggml_tensor * upPos) {
        const float scale = std::sqrt(512.0f);
        x = mark("embed", ggml_scale(ctx, layerNorm(linear(x, "flow.encoder.embed.out.0"), "flow.encoder.embed.out.1", 1e-5f), scale));
        // Look ahead three tokens, then a causal convolution, as a residual.
        ggml_tensor * h = ggml_leaky_relu(ctx, convCL(x, "flow.encoder.pre_lookahead_layer.conv1", 0, 3), 0.01f, false);
        h = convCL(h, "flow.encoder.pre_lookahead_layer.conv2", 2, 0);
        x = mark("lookahead", ggml_add(ctx, x, h));
        for (int i = 0; i < 6; i++) {
            x = conformerLayer(x, pos, "flow.encoder.encoders." + std::to_string(i));
            if (i == 0) mark("layer0", x);
        }
        mark("encoders", x);
        x = convCL(doubleTime(x), "flow.encoder.up_layer.conv", 4, 0);
        x = ggml_scale(ctx, layerNorm(linear(x, "flow.encoder.up_embed.out.0"), "flow.encoder.up_embed.out.1", 1e-5f), scale);
        for (int i = 0; i < 4; i++) x = conformerLayer(x, upPos, "flow.encoder.up_encoders." + std::to_string(i));
        return layerNorm(x, "flow.encoder.after_norm", 1e-5f);
    }

    ggml_tensor * timeEmbedding(ggml_tensor * sinusoid) {
        const std::string e = "flow.decoder.estimator.time_mlp";
        return linear(ggml_silu(ctx, linear(sinusoid, e + ".linear_1")), e + ".linear_2");
    }

    ggml_tensor * causalBlock(ggml_tensor * x, const std::string & p) {
        return mish(layerNorm(convCL(x, p + ".block.0", 2, 0), p + ".block.2", 1e-5f));
    }

    ggml_tensor * resnet(ggml_tensor * x, const std::string & p, ggml_tensor * time) {
        ggml_tensor * h = causalBlock(x, p + ".block1");
        h = ggml_add(ctx, h, linear(mish(time), p + ".mlp.1"));
        h = causalBlock(h, p + ".block2");
        return ggml_add(ctx, h, convCL(x, p + ".res_conv", 0, 0));
    }

    ggml_tensor * transformer(ggml_tensor * x, const std::string & p) {
        const int heads = 8, dk = 64;
        const int64_t length = x->ne[1];
        ggml_tensor * h = layerNorm(x, p + ".norm1", 1e-5f);
        auto split = [&](const std::string & name) {
            return ggml_reshape_3d(ctx, linear(h, p + name, false), dk, heads, length);
        };
        ggml_tensor * out = attention(split(".attn1.to_q"), split(".attn1.to_k"), split(".attn1.to_v"), nullptr,
                                      1.0f / std::sqrt(static_cast<float>(dk)));
        x = ggml_add(ctx, x, linear(out, p + ".attn1.to_out.0"));
        ggml_tensor * f = layerNorm(x, p + ".norm3", 1e-5f);
        return ggml_add(ctx, x, linear(ggml_gelu_erf(ctx, linear(f, p + ".ff.net.0.proj")), p + ".ff.net.2"));
    }

    ggml_tensor * estimator(ggml_tensor * input, ggml_tensor * mu, ggml_tensor * spks, ggml_tensor * cond,
                            ggml_tensor * time) {
        const std::string e = "flow.decoder.estimator";
        ggml_tensor * x = ggml_concat(ctx, ggml_concat(ctx, ggml_concat(ctx, input, mu, 0), spks, 0), cond, 0);
        x = resnet(x, e + ".down_blocks.0.0", time);
        for (int j = 0; j < 4; j++) x = transformer(x, e + ".down_blocks.0.1." + std::to_string(j));
        ggml_tensor * skip = x;
        x = convCL(x, e + ".down_blocks.0.2", 2, 0);
        for (int i = 0; i < 12; i++) {
            const std::string block = e + ".mid_blocks." + std::to_string(i);
            x = resnet(x, block + ".0", time);
            for (int j = 0; j < 4; j++) x = transformer(x, block + ".1." + std::to_string(j));
        }
        x = resnet(ggml_concat(ctx, x, skip, 0), e + ".up_blocks.0.0", time);
        for (int j = 0; j < 4; j++) x = transformer(x, e + ".up_blocks.0.1." + std::to_string(j));
        x = convCL(x, e + ".up_blocks.0.2", 2, 0);
        x = causalBlock(x, e + ".final_block");
        return convCL(x, e + ".final_proj", 0, 0);
    }

    // MARK: Vocoder

    /** f0 per frame from channels-first mel [frames, 80], as [1, frames]. */
    ggml_tensor * f0(ggml_tensor * mel) {
        ggml_tensor * f = mel;
        for (int i : {0, 2, 4, 6, 8}) f = ggml_elu(ctx, convCF(f, "mel2wav.f0_predictor.condnet." + std::to_string(i), 1));
        return ggml_abs(ctx, linear(ggml_cont(ctx, ggml_transpose(ctx, f)), "mel2wav.f0_predictor.classifier"));
    }

    ggml_tensor * snake(ggml_tensor * x, const std::string & name) {
        ggml_tensor * alpha = w(name);
        ggml_tensor * inverse = w(name + ".inverse");
        ggml_tensor * s = ggml_sin(ctx, ggml_mul(ctx, x, ggml_reshape_2d(ctx, alpha, 1, alpha->ne[0])));
        return ggml_add(ctx, x, ggml_mul(ctx, ggml_sqr(ctx, s), ggml_reshape_2d(ctx, inverse, 1, inverse->ne[0])));
    }

    ggml_tensor * resBlock(ggml_tensor * x, const std::string & p, int kernel) {
        const int dilations[] = {1, 3, 5};
        for (int i = 0; i < 3; i++) {
            const std::string index = std::to_string(i);
            const int d = dilations[i];
            ggml_tensor * t = snake(x, p + ".activations1." + index + ".alpha");
            t = convCF(t, p + ".convs1." + index, (kernel * d - d) / 2, 1, d);
            t = snake(t, p + ".activations2." + index + ".alpha");
            t = convCF(t, p + ".convs2." + index, (kernel - 1) / 2);
            x = ggml_add(ctx, x, t);
        }
        return x;
    }

    /** Channels-first mel [frames, 80] and the padded source [samples + 16, 1] to iSTFT frames [16, frames']. */
    ggml_tensor * vocoder(ggml_tensor * mel, ggml_tensor * source) {
        const std::string v = "mel2wav";
        ggml_tensor * spectrum = convolve(ggml_reshape_4d(ctx, source, source->ne[0], 1, 1, 1), w("stft.basis"), kHop, 0, 1);
        ggml_tensor * x = convCF(mel, v + ".conv_pre", 3);
        const int rates[] = {8, 5, 3}, kernels[] = {16, 11, 7};
        const int sourceStrides[] = {15, 3, 1}, sourcePadding[] = {7, 1, 0};
        const int sourceResKernels[] = {7, 7, 11}, resKernels[] = {3, 7, 11};
        for (int i = 0; i < 3; i++) {
            const std::string index = std::to_string(i);
            x = ggml_leaky_relu(ctx, x, 0.1f, false);
            x = convTranspose(x, v + ".ups." + index, rates[i], (kernels[i] - rates[i]) / 2);
            if (i == 2) {
                // Reflection pad (1, 0) in time.
                x = ggml_concat(ctx, ggml_cont(ctx, ggml_view_2d(ctx, x, 1, x->ne[1], x->nb[1], x->nb[0])), x, 0);
            }
            ggml_tensor * s = convCF(spectrum, v + ".source_downs." + index, sourcePadding[i], sourceStrides[i]);
            x = ggml_add(ctx, x, resBlock(s, v + ".source_resblocks." + index, sourceResKernels[i]));
            ggml_tensor * sum = nullptr;
            for (int j = 0; j < 3; j++) {
                ggml_tensor * y = resBlock(x, v + ".resblocks." + std::to_string(i * 3 + j), resKernels[j]);
                sum = sum ? ggml_add(ctx, sum, y) : y;
            }
            x = ggml_scale(ctx, sum, 1.0f / 3);
        }
        x = convCF(ggml_leaky_relu(ctx, x, 0.01f, false), v + ".conv_post", 3);                 // [frames', 18]
        const int64_t frames = x->ne[0];
        ggml_tensor * magnitude = ggml_clamp(ctx, ggml_exp(ctx, ggml_view_2d(ctx, x, frames, kBins, x->nb[1], 0)), 0.0f, 100.0f);
        ggml_tensor * angle = ggml_sin(ctx, ggml_view_2d(ctx, x, frames, kBins, x->nb[1], kBins * x->nb[1]));
        ggml_tensor * complex = ggml_concat(ctx, ggml_mul(ctx, magnitude, ggml_cos(ctx, angle)),
                                            ggml_mul(ctx, magnitude, ggml_sin(ctx, angle)), 1);
        // On the CPU, as the overlap-add after it: a 16-row product, which GPU
        // matmul shaders handle worst (ggml-vulkan's small tiles are wrong on Mali).
        ggml_tensor * frameSamples = matmul(ctx, w("istft.basis"), ggml_cont(ctx, ggml_transpose(ctx, complex)));
        g.onCpu(frameSamples);
        return frameSamples;
    }
};

// MARK: - Calls

std::vector<float> AudioDecoder::flow(const std::vector<int> & tokens, const std::vector<int> & promptToken,
                                      const std::vector<float> & promptFeat, const std::vector<float> & embedding,
                                      const std::vector<float> & noise, std::vector<float> * f0) {
    const auto count = static_cast<int64_t>(promptToken.size() + tokens.size());
    const int64_t frames = 2 * count;
    const int64_t promptFrames = static_cast<int64_t>(promptFeat.size()) / kMels;
    if (count == 0 || promptFrames >= frames || static_cast<int64_t>(noise.size()) != frames * kMels) {
        throw std::runtime_error("Bad decoder input sizes");
    }
    Scheduler scheduler(runtime_);
    Graph g(kGraphSize);
    Builder b{*this, g, g.ctx};
    ggml_context * ctx = g.ctx;

    ggml_tensor * ids = g.input(GGML_TYPE_I32, count);
    ggml_tensor * pos = g.input(GGML_TYPE_F32, 512, 2 * count - 1);
    ggml_tensor * upPos = g.input(GGML_TYPE_F32, 512, 2 * frames - 1);
    ggml_tensor * speaker = g.input(GGML_TYPE_F32, static_cast<int64_t>(embedding.size()));
    ggml_tensor * cond = g.input(GGML_TYPE_F32, kMels, frames);
    ggml_tensor * start = g.input(GGML_TYPE_F32, kMels, frames);
    const float steps[][2] = {{0.0f, 0.5f}, {0.5f, 1.0f}};
    ggml_tensor * sinusoids[2][2];
    for (auto & step : sinusoids)
        for (auto & s : step) s = g.input(GGML_TYPE_F32, 320);

    ggml_tensor * table = weight("flow.input_embedding.weight");
    ggml_tensor * x = ggml_get_rows(ctx, table, ids);                                                 // [512, N]
    ggml_tensor * encoded = b.encoder(x, pos, upPos);
    b.mark("mu_matmul", matmul(ctx, weight("flow.encoder_proj.weight"), encoded));
    ggml_tensor * mu = b.mark("mu", b.linear(encoded, "flow.encoder_proj"));                          // [80, 2N]
    ggml_tensor * spks = ggml_repeat_4d(ctx, b.linear(speaker, "flow.spk_embed_affine_layer"), kMels, frames, 1, 1);
    ggml_tensor * state = start;
    for (int i = 0; i < 2; i++) {
        ggml_tensor * time = b.linear(ggml_concat(ctx, b.timeEmbedding(sinusoids[i][0]), b.timeEmbedding(sinusoids[i][1]), 0),
                                      "flow.decoder.estimator.time_embed_mixer", false);
        ggml_tensor * velocity = b.estimator(state, mu, spks, cond, time);
        if (i == 0) {
            b.mark("time_t0", b.timeEmbedding(sinusoids[0][0]));
            b.mark("velocity0", velocity);
        }
        state = ggml_add(ctx, state, ggml_scale(ctx, velocity, steps[i][1] - steps[i][0]));
    }
    // Frame-major [80, F] in ggml's order, the prompt's frames dropped.
    ggml_tensor * mel = ggml_cont(ctx, ggml_view_2d(ctx, state, kMels, frames - promptFrames, state->nb[1],
                                                    promptFrames * state->nb[1]));
    ggml_set_output(mel);
    g.expand(mel);
    if (!probe.empty()) {
        auto found = b.probes.find(probe);
        if (found == b.probes.end()) throw std::runtime_error("No intermediate " + probe);
        mel = found->second;
        f0 = nullptr;
    }
    ggml_tensor * pitch = nullptr;
    if (f0) {
        pitch = b.f0(ggml_cont(ctx, ggml_transpose(ctx, mel)));
        ggml_set_output(pitch);
        g.expand(pitch);
    }

    g.allocate(runtime_, scheduler.sched);
    std::vector<int32_t> idValues(promptToken.begin(), promptToken.end());
    idValues.insert(idValues.end(), tokens.begin(), tokens.end());
    for (int32_t id : idValues) {
        if (id < 0 || id >= table->ne[1]) throw std::runtime_error("Codec token out of range");
    }
    ggml_backend_tensor_set(ids, idValues.data(), 0, ggml_nbytes(ids));
    set(pos, positionalEncoding(count));
    set(upPos, positionalEncoding(frames));
    {
        double norm = 0;
        for (float value : embedding) norm += static_cast<double>(value) * value;
        const float scale = 1.0f / std::max(static_cast<float>(std::sqrt(norm)), 1e-12f);
        std::vector<float> normalized(embedding);
        for (float & value : normalized) value *= scale;
        set(speaker, normalized);
    }
    std::vector<float> condition(static_cast<size_t>(frames * kMels), 0.0f);
    std::copy(promptFeat.begin(), promptFeat.end(), condition.begin());
    set(cond, condition);
    set(start, noise);
    for (int i = 0; i < 2; i++) {
        set(sinusoids[i][0], timeSinusoid(steps[i][0]));
        set(sinusoids[i][1], timeSinusoid(steps[i][1]));
    }
    g.compute(scheduler.sched);
    if (f0) *f0 = get(pitch);
    return get(mel);
}

std::vector<float> AudioDecoder::predictF0(const std::vector<float> & mel) {
    const int64_t frames = static_cast<int64_t>(mel.size()) / kMels;
    Scheduler scheduler(runtime_, vocoderOnCpu_);
    Graph g(4096);
    Builder b{*this, g, g.ctx};
    ggml_tensor * input = g.input(GGML_TYPE_F32, frames, kMels);
    ggml_tensor * pitch = b.f0(input);
    ggml_set_output(pitch);
    g.expand(pitch);
    g.allocate(runtime_, scheduler.sched);
    // Frame-major [frames, 80] to channels-first.
    std::vector<float> channelsFirst(mel.size());
    for (int64_t f = 0; f < frames; f++)
        for (int m = 0; m < kMels; m++) channelsFirst[m * frames + f] = mel[f * kMels + m];
    set(input, channelsFirst);
    g.compute(scheduler.sched);
    return get(pitch);
}

std::vector<float> AudioDecoder::vocoder(const std::vector<float> & mel, const std::vector<float> & f0,
                                         const std::vector<float> & phase, const std::vector<float> & sourceNoise) {
    const auto frames = static_cast<int64_t>(f0.size());
    const int64_t length = frames * kSamplesPerFrame;
    if (static_cast<int64_t>(mel.size()) != frames * kMels || phase.size() != kHarmonics
        || static_cast<int64_t>(sourceNoise.size()) != length * kHarmonics) {
        throw std::runtime_error("Bad vocoder input sizes");
    }

    // The harmonic source, as s3gen's SineGen computes it: nine sines at f0's
    // multiples (the phase summed in double, as torch's CPU cumsum does), noise
    // where unvoiced, then m_source's linear layer and tanh.
    std::vector<float> source(static_cast<size_t>(length), sourceBias_);
    for (int h = 0; h < kHarmonics; h++) {
        double sum = 0;
        const float weight = sourceWeight_[h];
        for (int64_t n = 0; n < length; n++) {
            const float f = f0[n / kSamplesPerFrame];
            sum += f * static_cast<float>(h + 1) / static_cast<float>(kSampleRate);
            const float theta = static_cast<float>(2 * kPi) * std::fmod(static_cast<float>(sum), 1.0f);
            const bool voiced = f > 10;
            const float amplitude = voiced ? 0.003f : 0.1f / 3;
            const float sine = voiced ? 0.1f * std::sin(theta + (h == 0 ? 0.0f : phase[h])) : 0.0f;
            source[n] += weight * (sine + amplitude * sourceNoise[n * kHarmonics + h]);
        }
    }
    const int half = kFft / 2;
    std::vector<float> padded(static_cast<size_t>(length + 2 * half));
    for (int i = 0; i < half; i++) {
        padded[i] = std::tanh(source[half - i]);
        padded[length + half + i] = std::tanh(source[length - 2 - i]);
    }
    for (int64_t n = 0; n < length; n++) padded[n + half] = std::tanh(source[n]);

    std::vector<float> channelsFirst(mel.size());
    for (int64_t f = 0; f < frames; f++)
        for (int m = 0; m < kMels; m++) channelsFirst[m * frames + f] = mel[f * kMels + m];

    std::vector<float> spectrumFrames;
    int64_t count = 0;
    {
        Scheduler scheduler(runtime_, vocoderOnCpu_);
        Graph g(kGraphSize);
        Builder b{*this, g, g.ctx};
        ggml_tensor * melInput = g.input(GGML_TYPE_F32, frames, kMels);
        ggml_tensor * sourceInput = g.input(GGML_TYPE_F32, length + 2 * half);
        ggml_tensor * out = b.vocoder(melInput, sourceInput);
        ggml_set_output(out);
        g.expand(out);
        g.allocate(runtime_, scheduler.sched);
        set(melInput, channelsFirst);
        set(sourceInput, padded);
        g.compute(scheduler.sched);
        count = out->ne[1];
        spectrumFrames = get(out);
    }

    // Inverse STFT's overlap-add, divided by the summed squared window.
    const std::vector<float> window = hannWindow();
    const int64_t total = (count - 1) * kHop + kFft;
    std::vector<float> signal(static_cast<size_t>(total), 0.0f), envelope(static_cast<size_t>(total), 0.0f);
    for (int64_t t = 0; t < count; t++) {
        for (int n = 0; n < kFft; n++) {
            signal[t * kHop + n] += spectrumFrames[t * kFft + n];
            envelope[t * kHop + n] += window[n] * window[n];
        }
    }
    std::vector<float> audio(static_cast<size_t>(total - 2 * half));
    for (size_t i = 0; i < audio.size(); i++) {
        const float value = signal[i + half] / std::max(envelope[i + half], 1e-11f);
        audio[i] = std::clamp(value, -0.99f, 0.99f);
    }
    return audio;
}

std::vector<float> AudioDecoder::decode(const std::vector<int> & tokens, const std::vector<int> & promptToken,
                                        const std::vector<float> & promptFeat, const std::vector<float> & embedding,
                                        uint64_t seed, const DecoderNoise * noise) {
    Noise random(seed);
    const auto frames = static_cast<int64_t>(2 * (promptToken.size() + tokens.size()));
    std::vector<float> f0;
    const std::vector<float> mel = flow(tokens, promptToken, promptFeat, embedding,
                                        noise ? noise->flow : random.normal(static_cast<size_t>(frames * kMels)), &f0);
    std::vector<float> phase(kHarmonics, 0.0f);
    if (noise) {
        phase = noise->phase;
    } else {
        for (int h = 1; h < kHarmonics; h++) phase[h] = static_cast<float>(-kPi + 2 * kPi * random.uniform());
    }
    const auto length = static_cast<size_t>(f0.size()) * kSamplesPerFrame;
    return vocoder(mel, f0, phase, noise ? noise->source : random.normal(length * kHarmonics));
}

} // namespace kt2
