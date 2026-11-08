#include "language_model.h"

#include "json.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace kt2 {

namespace {

constexpr int kQ4Block = 32;
constexpr float kScale = 32;
constexpr int kQ4BlockBytes = 2 + kQ4Block / 2;   // fp16 d, then 16 bytes of nibbles

struct TritTable {
    uint8_t trits[243][5];
    TritTable() {
        for (int value = 0; value < 243; value++) {
            int rest = value;
            for (auto & trit : trits[value]) {
                trit = static_cast<uint8_t>(rest % 3);
                rest /= 3;
            }
        }
    }
};

const TritTable & tritTable() {
    static const TritTable table;
    return table;
}

// Rows [rowBegin, rowEnd) of a packed ternary matrix as Q4_0 blocks. The codes
// are base-3 bytes, five trits each and lowest first, over the matrix in row-major
// order; trit 0, 1, 2 means -scale, 0, +scale.
void packRows(const uint8_t * codes, const uint16_t * scales, int64_t columns, int groupSize,
              int64_t rowBegin, int64_t rowEnd, uint8_t * out) {
    const auto & table = tritTable().trits;
    const int64_t groups = columns / groupSize;
    const int blocksPerGroup = groupSize / kQ4Block;
    int64_t flat = rowBegin * columns;
    int64_t byte = flat / 5;
    int position = static_cast<int>(flat % 5);
    uint8_t q[kQ4Block];
    for (int64_t row = rowBegin; row < rowEnd; row++) {
        for (int64_t group = 0; group < groups; group++) {
            ggml_fp16_t d = ggml_fp32_to_fp16(bf16ToFloat(scales[row * groups + group]));
            for (int block = 0; block < blocksPerGroup; block++) {
                for (auto & code : q) {
                    code = static_cast<uint8_t>(table[codes[byte]][position] + 7);
                    if (++position == 5) {
                        position = 0;
                        byte++;
                    }
                }
                std::memcpy(out, &d, sizeof(d));
                for (int j = 0; j < kQ4Block / 2; j++) out[2 + j] = static_cast<uint8_t>(q[j] | (q[j + 16] << 4));
                out += kQ4BlockBytes;
            }
        }
    }
}

void upload(ggml_tensor * tensor, const void * data, size_t bytes) {
    if (bytes != ggml_nbytes(tensor)) throw std::runtime_error(std::string("Size mismatch for ") + ggml_get_name(tensor));
    ggml_backend_tensor_set(tensor, data, 0, bytes);
}

} // namespace

LanguageModel::LanguageModel(const Runtime & runtime, const std::string & path, const LanguageModelConfig & config)
    : runtime_(runtime), config_(config), file_(std::make_unique<SafeTensors>(path)) {
    const auto & meta = file_->metadata();
    auto format = meta.find("format");
    if (format == meta.end() || format->second != "kittenml-ternary-1.58") {
        throw std::runtime_error("Not a packed Kitten TTS 2 checkpoint: " + path);
    }
    const int groupSize = std::stoi(meta.at("group_size"));
    if (groupSize % kQ4Block != 0) throw std::runtime_error("Unsupported group size " + std::to_string(groupSize));
    const Json packed = Json::parse(meta.at("packed"));

    const TensorView & embed = file_->get("model.embed_tokens.weight");
    if (embed.dtype != "BF16" || embed.shape.size() != 2 || embed.shape[1] != config.hiddenSize) {
        throw std::runtime_error("Unexpected embedding table in " + path);
    }
    vocab_ = embed.shape[0];
    embedTable_ = reinterpret_cast<const uint16_t *>(embed.data);
    speakerWeight_ = toFloat(file_->get("spk_proj.0.weight"));
    speakerBias_ = toFloat(file_->get("spk_proj.0.bias"));
    speakerNormWeight_ = toFloat(file_->get("spk_proj.1.weight"));
    speakerNormBias_ = toFloat(file_->get("spk_proj.1.bias"));

    // Tensors first, then one buffer for all of them, then the data.
    ggml_init_params params{ggml_tensor_overhead() * (12 * config.layers + 8), nullptr, true};
    weightsCtx_ = ggml_init(params);
    auto dense = [&](const std::string & name, int64_t size) {
        ggml_tensor * t = ggml_new_tensor_1d(weightsCtx_, GGML_TYPE_F32, size);
        ggml_set_name(t, name.c_str());
        return t;
    };
    auto shapeOf = [&](const std::string & name) {
        const Json & spec = packed[name];
        return std::pair<int64_t, int64_t>(spec["shape"][0].integer(), spec["shape"][1].integer());
    };
    auto linear = [&](const std::string & name) {
        auto [rows, columns] = shapeOf(name);
        ggml_tensor * t = ggml_new_tensor_2d(weightsCtx_, GGML_TYPE_Q4_0, columns, rows);
        ggml_set_name(t, name.c_str());
        return t;
    };
    for (int i = 0; i < config.layers; i++) {
        const std::string p = "model.layers." + std::to_string(i);
        layers_.push_back(Layer{
            dense(p + ".input_layernorm.weight", config.hiddenSize),
            dense(p + ".post_attention_layernorm.weight", config.hiddenSize),
            dense(p + ".self_attn.q_norm.weight", config.headDim),
            dense(p + ".self_attn.k_norm.weight", config.headDim),
            linear(p + ".self_attn.q_proj.weight"),
            linear(p + ".self_attn.k_proj.weight"),
            linear(p + ".self_attn.v_proj.weight"),
            linear(p + ".self_attn.o_proj.weight"),
            linear(p + ".mlp.gate_proj.weight"),
            linear(p + ".mlp.up_proj.weight"),
            linear(p + ".mlp.down_proj.weight"),
        });
    }
    norm_ = dense("model.norm.weight", config.hiddenSize);
    // The tied output head, cut to the rows the sampler draws from.
    head_ = ggml_new_tensor_2d(weightsCtx_, GGML_TYPE_F16, config.hiddenSize, vocab_ - config.headStart);
    ggml_set_name(head_, "head");
    weights_ = allocateTensors(weightsCtx_, runtime.weightBuffer(), "language model");

    const unsigned threads = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
    std::vector<uint8_t> scratch;
    for (ggml_tensor * t = ggml_get_first_tensor(weightsCtx_); t; t = ggml_get_next_tensor(weightsCtx_, t)) {
        const std::string name = ggml_get_name(t);
        if (t == head_) continue;
        if (t->type == GGML_TYPE_F32) {
            std::vector<float> values = toFloat(file_->get(name));
            upload(t, values.data(), values.size() * sizeof(float));
            continue;
        }
        const TensorView & codes = file_->get(name + "::codes");
        const TensorView & scales = file_->get(name + "::scale");
        const int64_t rows = t->ne[1], columns = t->ne[0];
        if (codes.bytes * 5 < static_cast<size_t>(rows * columns) || scales.dtype != "BF16"
            || scales.elements() != rows * (columns / groupSize)) {
            throw std::runtime_error(name + " is not packed as expected");
        }
        scratch.resize(ggml_nbytes(t));
        const int64_t rowBytes = static_cast<int64_t>(ggml_row_size(GGML_TYPE_Q4_0, columns));
        std::vector<std::thread> workers;
        for (unsigned w = 0; w < threads; w++) {
            int64_t begin = rows * w / threads, end = rows * (w + 1) / threads;
            workers.emplace_back(packRows, codes.data, reinterpret_cast<const uint16_t *>(scales.data), columns,
                                 groupSize, begin, end, scratch.data() + begin * rowBytes);
        }
        for (auto & worker : workers) worker.join();
        upload(t, scratch.data(), scratch.size());
    }
    {
        const int64_t rows = vocab_ - config.headStart, width = config.hiddenSize;
        std::vector<ggml_fp16_t> head(static_cast<size_t>(rows * width));
        const uint16_t * source = embedTable_ + config.headStart * width;
        for (int64_t i = 0; i < rows * width; i++) head[i] = ggml_fp32_to_fp16(bf16ToFloat(source[i]));
        upload(head_, head.data(), head.size() * sizeof(ggml_fp16_t));
    }
    sched_ = runtime.newScheduler(4096);
}

LanguageModel::~LanguageModel() {
    if (sched_) ggml_backend_sched_free(sched_);
    freeCache();
    if (weights_) ggml_backend_buffer_free(weights_);
    if (weightsCtx_) ggml_free(weightsCtx_);
}

void LanguageModel::embed(int token, float * out) const {
    if (token < 0 || token >= vocab_) throw std::runtime_error("Token " + std::to_string(token) + " is out of range");
    const uint16_t * row = embedTable_ + static_cast<int64_t>(token) * config_.hiddenSize;
    for (int i = 0; i < config_.hiddenSize; i++) out[i] = bf16ToFloat(row[i]);
}

std::vector<float> LanguageModel::speakerHidden(const std::vector<float> & embedding) const {
    const auto inputs = static_cast<int64_t>(embedding.size());
    const int hidden = config_.hiddenSize;
    if (inputs * hidden != static_cast<int64_t>(speakerWeight_.size())) {
        throw std::runtime_error("The speaker embedding has " + std::to_string(inputs) + " values");
    }
    std::vector<float> projected(hidden);
    for (int o = 0; o < hidden; o++) {
        const float * w = speakerWeight_.data() + static_cast<int64_t>(o) * inputs;
        double sum = speakerBias_[o];
        for (int64_t i = 0; i < inputs; i++) sum += static_cast<double>(w[i]) * embedding[i];
        projected[o] = static_cast<float>(sum);
    }
    double mean = 0, variance = 0;
    for (float value : projected) mean += value;
    mean /= hidden;
    for (float value : projected) variance += (value - mean) * (value - mean);
    variance /= hidden;
    const double inverse = 1.0 / std::sqrt(variance + 1e-5);
    for (int o = 0; o < hidden; o++) {
        projected[o] = static_cast<float>((projected[o] - mean) * inverse) * speakerNormWeight_[o] + speakerNormBias_[o];
    }
    return projected;
}

void LanguageModel::freeCache() {
    if (cache_) ggml_backend_buffer_free(cache_);
    if (cacheCtx_) ggml_free(cacheCtx_);
    cache_ = nullptr;
    cacheCtx_ = nullptr;
    keys_.clear();
    values_.clear();
    capacity_ = 0;
}

void LanguageModel::reserveCache(int capacity) {
    position_ = 0;
    if (capacity <= capacity_) return;
    freeCache();
    capacity = (capacity + 255) / 256 * 256;
    ggml_init_params params{ggml_tensor_overhead() * (2 * config_.layers + 2), nullptr, true};
    cacheCtx_ = ggml_init(params);
    for (int i = 0; i < config_.layers; i++) {
        const ggml_type type = runtime_.cacheType();
        keys_.push_back(ggml_new_tensor_3d(cacheCtx_, type, config_.headDim, config_.kvHeads, capacity));
        values_.push_back(ggml_new_tensor_3d(cacheCtx_, type, config_.headDim, config_.kvHeads, capacity));
    }
    cache_ = allocateTensors(cacheCtx_, runtime_.weightBuffer(), "language model's cache");
    ggml_backend_buffer_clear(cache_, 0);
    capacity_ = capacity;
}

std::vector<float> LanguageModel::prefill(const std::vector<int> & prompt, const std::vector<float> & speaker,
                                          int capacity) {
    const int length = static_cast<int>(prompt.size());
    if (length == 0) throw std::runtime_error("The prompt is empty");
    if (static_cast<int>(speaker.size()) != config_.hiddenSize) throw std::runtime_error("Bad speaker hidden state");
    reserveCache(std::max(capacity, length + 1));
    std::vector<float> embeddings(static_cast<size_t>(length) * config_.hiddenSize);
    std::copy(speaker.begin(), speaker.end(), embeddings.begin());
    for (int i = 1; i < length; i++) embed(prompt[i], embeddings.data() + static_cast<size_t>(i) * config_.hiddenSize);
    return forward(embeddings, length);
}

std::vector<float> LanguageModel::step(int token) {
    std::vector<float> embedding(config_.hiddenSize);
    embed(token, embedding.data());
    return forward(embedding, 1);
}

std::vector<float> LanguageModel::forward(const std::vector<float> & embeddings, int length) {
    const auto & c = config_;
    const int context = position_ + length;
    if (context > capacity_) throw std::runtime_error("The sequence is longer than the cache");

    Graph g(static_cast<size_t>(48 * c.layers + 64));
    ggml_context * ctx = g.ctx;
    ggml_tensor * input = g.input(GGML_TYPE_F32, c.hiddenSize, length);
    ggml_tensor * positions = g.input(GGML_TYPE_I32, length);
    ggml_tensor * mask = length > 1 ? g.input(GGML_TYPE_F16, context, length) : nullptr;
    const float scale = 1.0f / std::sqrt(static_cast<float>(c.headDim));

    ggml_tensor * x = input;
    ggml_tensor * probed = nullptr;
    for (int i = 0; i < c.layers; i++) {
        const Layer & layer = layers_[i];
        ggml_tensor * h = ggml_mul(ctx, ggml_rms_norm(ctx, x, c.rmsNormEps), layer.inputNorm);
        ggml_tensor * q = ggml_reshape_3d(ctx, matmul(ctx, layer.q, h), c.headDim, c.heads, length);
        q = ggml_mul(ctx, ggml_rms_norm(ctx, q, c.rmsNormEps), layer.qNorm);
        q = ggml_rope_ext(ctx, q, positions, nullptr, c.headDim, GGML_ROPE_TYPE_NEOX, 0, c.ropeTheta, 1.0f, 0.0f,
                          1.0f, 0.0f, 0.0f);
        ggml_tensor * k = ggml_reshape_3d(ctx, matmul(ctx, layer.k, h), c.headDim, c.kvHeads, length);
        k = ggml_mul(ctx, ggml_rms_norm(ctx, k, c.rmsNormEps), layer.kNorm);
        k = ggml_rope_ext(ctx, k, positions, nullptr, c.headDim, GGML_ROPE_TYPE_NEOX, 0, c.ropeTheta, 1.0f, 0.0f,
                          1.0f, 0.0f, 0.0f);
        ggml_tensor * v = ggml_reshape_3d(ctx, matmul(ctx, layer.v, h), c.headDim, c.kvHeads, length);

        // Append to the cache, then attend over everything in it. The copies come
        // first in the graph, so they run before the reads.
        ggml_tensor * keys = keys_[i];
        ggml_tensor * values = values_[i];
        g.expand(ggml_cpy(ctx, k, ggml_view_3d(ctx, keys, c.headDim, c.kvHeads, length, keys->nb[1], keys->nb[2],
                                               position_ * keys->nb[2])));
        g.expand(ggml_cpy(ctx, v, ggml_view_3d(ctx, values, c.headDim, c.kvHeads, length, values->nb[1],
                                               values->nb[2], position_ * values->nb[2])));
        ggml_tensor * allKeys = ggml_view_3d(ctx, keys, c.headDim, context, c.kvHeads, keys->nb[2], keys->nb[1], 0);
        ggml_tensor * allValues = ggml_view_3d(ctx, values, c.headDim, context, c.kvHeads, values->nb[2],
                                               values->nb[1], 0);
        ggml_tensor * attended = ggml_flash_attn_ext(ctx, ggml_permute(ctx, q, 0, 2, 1, 3), allKeys, allValues, mask,
                                                     scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attended, GGML_PREC_F32);
        attended = ggml_reshape_2d(ctx, attended, c.headDim * c.heads, length);
        // Scaled like the down projection below: in the last layer the output
        // reaches about 64,000, the edge of float16, which ggml-opencl's Adreno
        // kernels hold it in (it came out infinite at some prompt positions).
        x = ggml_add(ctx, x, ggml_scale(ctx, matmul(ctx, layer.o, ggml_scale(ctx, attended, 1.0f / kScale)), kScale));

        ggml_tensor * m = ggml_mul(ctx, ggml_rms_norm(ctx, x, c.rmsNormEps), layer.postAttentionNorm);
        ggml_tensor * mlp = ggml_swiglu_split(ctx, matmul(ctx, layer.gate, m), matmul(ctx, layer.up, m));
        // The down projection's input reaches a few hundred thousand in the last
        // layers, past float16, which GPU matmuls hold their inputs in: multiply by
        // a power of two before and after, which is exact (only values far below
        // what matters lose precision).
        // ggml-opencl holds the product itself in float16 too, and the last layer's
        // reaches about two million: 1/256 there.
        const float downScale = runtime_.openCL() ? 256 : kScale;
        ggml_tensor * mlpOut = ggml_scale(ctx, matmul(ctx, layer.down, ggml_scale(ctx, mlp, 1.0f / downScale)), downScale);
        x = ggml_add(ctx, x, mlpOut);
        if (i == probeLayer) {
            ggml_tensor * parts[] = {x, attended, mlpOut, q, k, v, mlp};
            probed = parts[probePart];
            if (probePart >= 3 && probePart <= 5) {
                probed = ggml_reshape_2d(ctx, ggml_cont(ctx, probed), probed->ne[0] * probed->ne[1], length);
            }
            break;
        }
    }
    // The last position's row, picked with get_rows as llama.cpp does.
    ggml_tensor * lastIndex = g.input(GGML_TYPE_I32, 1);
    ggml_tensor * logits = nullptr;
    if (probed) {
        logits = probeAllRows ? ggml_cont(ctx, probed) : ggml_get_rows(ctx, probed, lastIndex);
    } else {
        ggml_tensor * last = ggml_get_rows(ctx, x, lastIndex);
        last = ggml_mul(ctx, ggml_rms_norm(ctx, last, c.rmsNormEps), norm_);
        logits = matmul(ctx, head_, last);
    }
    ggml_set_output(logits);
    g.expand(logits);

    g.allocate(runtime_, sched_);
    ggml_backend_tensor_set(input, embeddings.data(), 0, ggml_nbytes(input));
    std::vector<int32_t> ids(length);
    for (int i = 0; i < length; i++) ids[i] = position_ + i;
    ggml_backend_tensor_set(positions, ids.data(), 0, ggml_nbytes(positions));
    const int32_t lastRow = length - 1;
    if (lastIndex->buffer) ggml_backend_tensor_set(lastIndex, &lastRow, 0, sizeof(lastRow));
    if (mask) {
        const ggml_fp16_t open = ggml_fp32_to_fp16(0.0f), closed = ggml_fp32_to_fp16(-INFINITY);
        std::vector<ggml_fp16_t> values(static_cast<size_t>(context) * length);
        for (int q = 0; q < length; q++) {
            for (int k = 0; k < context; k++) values[static_cast<size_t>(q) * context + k] = k <= position_ + q ? open : closed;
        }
        ggml_backend_tensor_set(mask, values.data(), 0, ggml_nbytes(mask));
    }
    g.compute(sched_);

    std::vector<float> out(static_cast<size_t>(ggml_nelements(logits)));
    ggml_backend_tensor_get(logits, out.data(), 0, ggml_nbytes(logits));
    position_ = context;
    return out;
}

} // namespace kt2
