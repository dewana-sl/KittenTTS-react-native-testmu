#include "gpu_check.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include "ggml-cpu.h"

namespace kt2 {

namespace {

using Clock = std::chrono::steady_clock;

// Graphs checked against the CPU in all (the first of each size), and lines written per graph.
constexpr int kMaxComparedGraphs = 12;
constexpr int kMaxLines = 40;

double msSince(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

bool isViewOp(ggml_op op) {
    return op == GGML_OP_NONE || op == GGML_OP_RESHAPE || op == GGML_OP_VIEW || op == GGML_OP_PERMUTE
           || op == GGML_OP_TRANSPOSE;
}

uint64_t hashBytes(const std::vector<uint8_t> & bytes) {
    uint64_t hash = 1469598103934665603ull;
    size_t i = 0;
    for (; i + 8 <= bytes.size(); i += 8) {
        uint64_t word;
        std::memcpy(&word, bytes.data() + i, sizeof(word));
        hash = (hash ^ word) * 1099511628211ull;
    }
    for (; i < bytes.size(); i++) hash = (hash ^ bytes[i]) * 1099511628211ull;
    return hash;
}

/** The bytes from the tensor's first element to its last, laid out as on its backend. */
std::vector<uint8_t> readBytes(const ggml_tensor * t) {
    std::vector<uint8_t> bytes(ggml_nbytes(t));
    if (!bytes.empty()) ggml_backend_tensor_get(t, bytes.data(), 0, bytes.size());
    return bytes;
}

std::string describe(const ggml_tensor * t) {
    std::string out = ggml_type_name(t->type);
    int dims = GGML_MAX_DIMS;
    while (dims > 1 && t->ne[dims - 1] == 1) dims--;
    out += "[";
    for (int d = 0; d < dims; d++) out += (d ? "," : "") + std::to_string(t->ne[d]);
    out += "]";
    if (!ggml_is_contiguous(t)) out += " strided";
    if (t->name[0]) out += std::string(" '") + t->name + "'";
    return out;
}

bool comparable(ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_I32;
}

double readValue(ggml_type type, const uint8_t * at) {
    switch (type) {
        case GGML_TYPE_F32: { float v; std::memcpy(&v, at, sizeof(v)); return v; }
        case GGML_TYPE_F16: { ggml_fp16_t v; std::memcpy(&v, at, sizeof(v)); return ggml_fp16_to_fp32(v); }
        default: { int32_t v; std::memcpy(&v, at, sizeof(v)); return v; }
    }
}

struct Difference {
    int64_t count = 0, nonFiniteGpu = 0, nonFiniteCpu = 0;
    double maxAbs = 0, maxRef = 0, sumSquares = 0, refSquares = 0;

    /** RMS of the difference over RMS of the CPU's result. */
    double relative() const {
        if (sumSquares == 0) return 0;
        return refSquares > 0 ? std::sqrt(sumSquares / refSquares) : INFINITY;
    }
    bool bad() const { return nonFiniteGpu != nonFiniteCpu || relative() > 0.05; }
};

Difference compareValues(const ggml_tensor * t, const std::vector<uint8_t> & gpu, const std::vector<uint8_t> & cpu) {
    Difference d;
    for (int64_t i3 = 0; i3 < t->ne[3]; i3++)
    for (int64_t i2 = 0; i2 < t->ne[2]; i2++)
    for (int64_t i1 = 0; i1 < t->ne[1]; i1++)
    for (int64_t i0 = 0; i0 < t->ne[0]; i0++) {
        const size_t offset = i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
        const double g = readValue(t->type, gpu.data() + offset);
        const double c = readValue(t->type, cpu.data() + offset);
        d.count++;
        if (!std::isfinite(g)) d.nonFiniteGpu++;
        if (!std::isfinite(c)) d.nonFiniteCpu++;
        if (!std::isfinite(g) || !std::isfinite(c)) continue;
        const double diff = std::fabs(g - c);
        d.maxAbs = std::max(d.maxAbs, diff);
        d.maxRef = std::max(d.maxRef, std::fabs(c));
        d.sumSquares += diff * diff;
        d.refSquares += c * c;
    }
    return d;
}

struct OpStats {
    int gpuNodes = 0, cpuNodes = 0, checked = 0, differ = 0;
    double gpuMs = 0, cpuMs = 0, checkMs = 0, worst = 0;
};

class Checker {
public:
    explicit Checker(const char * path) : path_(path), file_(std::fopen(path, "w")), cpu_(ggml_backend_cpu_init()) {
        if (cpu_) ggml_backend_cpu_set_n_threads(cpu_, 4);
        line("process %d", static_cast<int>(getpid()));
    }

    void compute(ggml_backend_sched_t sched, ggml_cgraph * graph) {
        std::lock_guard<std::mutex> lock(mutex_);
        const int nodes = ggml_graph_n_nodes(graph);
        // A single-position graph (a token step) and a longer one of the same size are different graphs.
        const std::string size = std::to_string(nodes) + (nodes && ggml_graph_node(graph, 0)->ne[1] == 1 ? "/1" : "/n");
        sched_ = sched;
        graphs_++;
        comparing_ = cpu_ && file_ && compared_ < kMaxComparedGraphs && sizes_.insert(size).second;
        made_.clear();
        written_.clear();
        node_ = 0;
        current_ = nullptr;
        differInGraph_ = changedInGraph_ = 0;
        if (comparing_) {
            compared_++;
            // Tensors that nodes of this graph write into (cache appends, in-place ops) change legitimately.
            for (int i = 0; i < nodes; i++) {
                const ggml_tensor * node = ggml_graph_node(graph, i);
                if (!isViewOp(node->op) && node->view_src) written_.insert(node->view_src);
            }
            line("graph %d: %d nodes, checked against the CPU", graphs_, nodes);
        }
        ggml_backend_sched_set_eval_callback(sched, &Checker::callback, this);
        ggml_status status = GGML_STATUS_FAILED;
        try {
            status = ggml_backend_sched_graph_compute(sched, graph);
        } catch (const std::exception & error) {
            ggml_backend_sched_set_eval_callback(sched, nullptr, nullptr);
            line("graph %d FAILED at node %d %s %s: %s", graphs_, node_, current_ ? ggml_op_desc(current_) : "?",
                 current_ ? describe(current_).c_str() : "", error.what());
            writeOps();
            throw;
        }
        ggml_backend_sched_set_eval_callback(sched, nullptr, nullptr);
        if (comparing_) line("graph %d: %d nodes differ, %d inputs changed", graphs_, differInGraph_, changedInGraph_);
        writeOps();
        if (status != GGML_STATUS_SUCCESS) {
            line("graph %d: status %d", graphs_, static_cast<int>(status));
            throw std::runtime_error("The computation failed");
        }
    }

private:
    struct Made {
        uint64_t hash;
        int node;
    };

    static bool callback(ggml_tensor * t, bool ask, void * self) {
        return static_cast<Checker *>(self)->onNode(t, ask);
    }

    // Asked about every node, so each runs alone: its inputs are read before it
    // runs (it may overwrite one in place) and its result after.
    bool onNode(ggml_tensor * t, bool ask) {
        if (ask) {
            node_++;
            current_ = t;
            ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched_, t);
            onGpu_ = backend && !ggml_backend_is_cpu(backend);
            checking_ = comparing_ && onGpu_ && !isViewOp(t->op);
            inputs_.clear();
            if (checking_) {
                for (const ggml_tensor * src : t->src) {
                    if (!src || inputs_.count(src)) continue;
                    inputs_[src] = readBytes(src);
                    checkUnchanged(src, t);
                }
            }
            started_ = Clock::now();
            return true;
        }
        const double ms = msSince(started_);
        if (!isViewOp(t->op)) {
            OpStats & stats = ops_[ggml_op_desc(t)];
            (onGpu_ ? stats.gpuNodes : stats.cpuNodes)++;
            (onGpu_ ? stats.gpuMs : stats.cpuMs) += ms;
        }
        if (checking_) check(t);
        return true;
    }

    void checkUnchanged(const ggml_tensor * src, const ggml_tensor * reader) {
        const ggml_tensor * base = src->view_src ? src->view_src : src;
        auto made = made_.find(base);
        if (made == made_.end() || written_.count(base)) return;
        const uint64_t hash = base == src ? hashBytes(inputs_[src]) : hashBytes(readBytes(base));
        if (hash == made->second.hash) return;
        if (++changedInGraph_ <= kMaxLines) {
            line("  input changed: %s %s, made by node %d, changed before node %d %s %s", ggml_op_desc(base),
                 describe(base).c_str(), made->second.node, node_, ggml_op_desc(reader), describe(reader).c_str());
        }
        made_.erase(made);
    }

    void check(ggml_tensor * t) {
        std::vector<uint8_t> gpu = readBytes(t);
        if (!t->view_src) made_[t] = Made{hashBytes(gpu), node_};
        if (!comparable(t->type) || gpu.empty()) return;

        // The same op on the CPU, over copies of the inputs the GPU node read.
        ggml_init_params params{
            ggml_tensor_overhead() * (GGML_MAX_SRC + 1) + ggml_graph_overhead_custom(GGML_MAX_SRC + 1, false),
            nullptr,
            true,
        };
        ggml_context * ctx = ggml_init(params);
        auto like = [&](const ggml_tensor * from) {
            ggml_tensor * to = ggml_new_tensor(ctx, from->type, GGML_MAX_DIMS, from->ne);
            std::copy(std::begin(from->nb), std::end(from->nb), std::begin(to->nb));
            return to;
        };
        ggml_tensor * node = like(t);
        node->op = t->op;
        std::memcpy(node->op_params, t->op_params, sizeof(node->op_params));
        std::map<const ggml_tensor *, ggml_tensor *> copies;
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            const ggml_tensor * src = t->src[s];
            if (!src) continue;
            ggml_tensor *& copy = copies[src];
            if (!copy) {
                copy = like(src);
                copy->data = inputs_[src].data();
            }
            node->src[s] = copy;
        }
        std::vector<uint8_t> cpu = gpu;   // a destination view keeps the bytes between its elements
        node->data = cpu.data();

        OpStats & stats = ops_[ggml_op_desc(t)];
        const Clock::time_point started = Clock::now();
        bool ran = false;
        if (ggml_backend_supports_op(cpu_, node)) {
            ggml_cgraph * graph = ggml_new_graph_custom(ctx, GGML_MAX_SRC + 1, false);
            ggml_build_forward_expand(graph, node);
            ran = ggml_backend_graph_compute(cpu_, graph) == GGML_STATUS_SUCCESS;
        }
        stats.checkMs += msSince(started);
        ggml_free(ctx);
        if (!ran) {
            line("  node %d %s %s: the CPU cannot run it", node_, ggml_op_desc(t), describe(t).c_str());
            return;
        }

        const Difference d = compareValues(t, gpu, cpu);
        stats.checked++;
        stats.worst = std::max(stats.worst, d.relative());
        if (!d.bad()) return;
        stats.differ++;
        if (++differInGraph_ > kMaxLines) return;
        std::string sources;
        for (int s = 0; s < GGML_MAX_SRC && t->src[s]; s++) sources += (s ? ", " : "") + describe(t->src[s]);
        line("  node %d %s %s = (%s): relative error %.3g, max error %.3g (CPU max %.3g), non-finite GPU %lld CPU %lld",
             node_, ggml_op_desc(t), describe(t).c_str(), sources.c_str(), d.relative(), d.maxAbs, d.maxRef,
             static_cast<long long>(d.nonFiniteGpu), static_cast<long long>(d.nonFiniteCpu));
    }

    void writeOps() {
        FILE * out = std::fopen((path_ + ".ops").c_str(), "w");
        if (!out) return;
        std::vector<std::pair<std::string, OpStats>> rows(ops_.begin(), ops_.end());
        std::sort(rows.begin(), rows.end(), [](const auto & a, const auto & b) {
            return a.second.gpuMs + a.second.cpuMs > b.second.gpuMs + b.second.cpuMs;
        });
        std::fprintf(out, "%d graphs, one node at a time\n", graphs_);
        std::fprintf(out, "op | GPU nodes | GPU ms | CPU nodes | CPU ms | checked | differ | worst relative error | ms to check\n");
        for (const auto & [op, s] : rows) {
            std::fprintf(out, "%s | %d | %.0f | %d | %.0f | %d | %d | %.3g | %.0f\n", op.c_str(), s.gpuNodes, s.gpuMs,
                         s.cpuNodes, s.cpuMs, s.checked, s.differ, s.worst, s.checkMs);
        }
        std::fclose(out);
    }

    __attribute__((format(printf, 2, 3))) void line(const char * format, ...) {
        if (!file_) return;
        va_list args;
        va_start(args, format);
        std::vfprintf(file_, format, args);
        va_end(args);
        std::fputc('\n', file_);
        std::fflush(file_);
    }

    std::mutex mutex_;
    std::string path_;
    FILE * file_ = nullptr;
    ggml_backend_t cpu_ = nullptr;
    std::set<std::string> sizes_;
    std::map<std::string, OpStats> ops_;
    int graphs_ = 0, compared_ = 0;

    // The graph being computed.
    ggml_backend_sched_t sched_ = nullptr;
    bool comparing_ = false;
    std::unordered_map<const ggml_tensor *, Made> made_;
    std::set<const ggml_tensor *> written_;
    int differInGraph_ = 0, changedInGraph_ = 0;

    // The node being computed.
    int node_ = 0;
    ggml_tensor * current_ = nullptr;
    bool onGpu_ = false, checking_ = false;
    std::map<const ggml_tensor *, std::vector<uint8_t>> inputs_;
    Clock::time_point started_;
};

} // namespace

bool computeChecked(ggml_backend_sched_t sched, ggml_cgraph * graph) {
    static const char * path = std::getenv("KITTENTTS2_CHECK");
    if (!path || !*path) return false;
    static Checker checker(path);
    checker.compute(sched, graph);
    return true;
}

} // namespace kt2
