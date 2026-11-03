#include "runtime.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <thread>

#include "ggml-cpu.h"

namespace kt2 {

Runtime::Runtime(bool useGpu, int threads) {
    if (useGpu) {
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            auto type = ggml_backend_dev_type(dev);
            if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
            try {
                gpu_ = ggml_backend_dev_init(dev, nullptr);
            } catch (const std::exception &) {
                // ggml-vulkan throws for a GPU it cannot use (no 16-bit storage, say): run on the CPU.
                gpu_ = nullptr;
            }
            if (gpu_) {
                device_ = std::string(ggml_backend_dev_name(dev)) + ": " + ggml_backend_dev_description(dev);
                break;
            }
        }
    }
    cpu_ = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu_) throw std::runtime_error("ggml has no CPU backend");
    if (threads <= 0) {
        // Phones pair a few fast cores with slow ones; more threads than fast cores only waits on the slow ones.
        threads = static_cast<int>(std::clamp(std::thread::hardware_concurrency() / 2, 2u, 4u));
    }
    ggml_backend_cpu_set_n_threads(cpu_, threads);
    if (!gpu_) device_ = "CPU";
}

Runtime::~Runtime() {
    if (gpu_) ggml_backend_free(gpu_);
    if (cpu_) ggml_backend_free(cpu_);
}

ggml_backend_buffer_type_t Runtime::weightBuffer() const {
    return ggml_backend_get_default_buffer_type(gpu_ ? gpu_ : cpu_);
}

ggml_backend_sched_t Runtime::newScheduler(size_t graphSize) const {
    std::vector<ggml_backend_t> backends;
    if (gpu_) backends.push_back(gpu_);
    backends.push_back(cpu_);
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends.data(), nullptr, static_cast<int>(backends.size()), graphSize, false, false);
    if (!sched) throw std::runtime_error("Could not create a ggml scheduler");
    return sched;
}

void Runtime::checkSupported(ggml_cgraph * graph) const {
    static const bool debug = std::getenv("KITTENTTS2_DEBUG") != nullptr;
    for (int i = 0; i < ggml_graph_n_nodes(graph); i++) {
        ggml_tensor * node = ggml_graph_node(graph, i);
        switch (node->op) {
            case GGML_OP_NONE: case GGML_OP_RESHAPE: case GGML_OP_VIEW: case GGML_OP_PERMUTE: case GGML_OP_TRANSPOSE:
                continue;
            default:
                break;
        }
        const bool onGpu = gpu_ && ggml_backend_supports_op(gpu_, node);
        if (debug && gpu_ && !onGpu) {
            std::fprintf(stderr, "kittentts2: node %d %s (%s) [%lld x %lld x %lld] runs on the CPU\n", i, ggml_op_desc(node),
                         ggml_get_name(node), (long long) node->ne[0], (long long) node->ne[1], (long long) node->ne[2]);
        }
        if (onGpu || ggml_backend_supports_op(cpu_, node)) continue;
        std::string shape;
        for (int d = 0; d < GGML_MAX_DIMS; d++) shape += (d ? "x" : "") + std::to_string(node->ne[d]);
        std::string sources;
        for (int s = 0; s < GGML_MAX_SRC && node->src[s]; s++) {
            sources += std::string(s ? ", " : "") + ggml_type_name(node->src[s]->type)
                       + (ggml_is_contiguous(node->src[s]) ? "" : " (strided)");
        }
        throw std::runtime_error(std::string("No backend runs ") + ggml_op_desc(node) + " [" + shape + "] of " + sources);
    }
}

Graph::Graph(size_t maxNodes) {
    ggml_init_params params{
        ggml_tensor_overhead() * maxNodes + ggml_graph_overhead_custom(maxNodes, false),
        nullptr,
        true,
    };
    ctx = ggml_init(params);
    if (!ctx) throw std::runtime_error("Could not create a ggml context");
    graph = ggml_new_graph_custom(ctx, maxNodes, false);
}

Graph::~Graph() {
    ggml_free(ctx);
}

ggml_tensor * Graph::input(ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2) {
    ggml_tensor * t = ggml_new_tensor_3d(ctx, type, ne0, ne1, ne2);
    ggml_set_input(t);
    return t;
}

void Graph::allocate(const Runtime & runtime, ggml_backend_sched_t sched) {
    runtime.checkSupported(graph);
    ggml_backend_sched_reset(sched);
    if (!ggml_backend_sched_alloc_graph(sched, graph)) {
        throw std::runtime_error("Not enough memory for the computation");
    }
    if (std::getenv("KITTENTTS2_DEBUG")) {
        std::fprintf(stderr, "kittentts2: %d nodes, %d splits, %d copies", ggml_graph_n_nodes(graph),
                     ggml_backend_sched_get_n_splits(sched), ggml_backend_sched_get_n_copies(sched));
        for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); i++) {
            ggml_backend_t backend = ggml_backend_sched_get_backend(sched, i);
            std::fprintf(stderr, ", %s buffer %.1f MB", ggml_backend_name(backend),
                         ggml_backend_sched_get_buffer_size(sched, backend) / 1048576.0);
        }
        std::fprintf(stderr, "\n");
    }
}

void Graph::compute(ggml_backend_sched_t sched) {
    if (ggml_backend_sched_graph_compute(sched, graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("The computation failed");
    }
}

ggml_backend_buffer_t allocateTensors(ggml_context * ctx, ggml_backend_buffer_type_t type, const char * what) {
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, type);
    if (!buffer) throw std::runtime_error(std::string("Not enough memory for the ") + what);
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    return buffer;
}

} // namespace kt2
