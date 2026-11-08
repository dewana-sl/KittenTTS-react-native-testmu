// The ggml backends the models run on: the GPU, through OpenCL on Adreno 7xx and
// newer or Vulkan elsewhere, when the device has one; and the CPU, which runs any
// op the GPU backend does not support, and everything when there is no GPU.

#pragma once

#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

namespace kt2 {

/**
 * Whether to use the GPU: never; when there is one that is not known to fail
 * (the default); or whenever there is one, for testing a GPU that is.
 */
enum class GpuChoice { Off, Auto, Always };

/** From KITTENTTS2_DEVICE: "cpu" is Off, "gpu" is Always, unset or anything else Auto. */
GpuChoice gpuChoiceFromEnvironment();

class Runtime {
public:
    /** `threads` 0 picks a count. */
    Runtime(GpuChoice gpu, int threads);
    ~Runtime();
    Runtime(const Runtime &) = delete;
    Runtime & operator=(const Runtime &) = delete;

    bool hasGpu() const { return gpu_ != nullptr; }
    ggml_backend_t cpu() const { return cpu_; }
    /**
     * The language model's cache type. float16, except on ggml-opencl: on Adreno
     * 7xx it runs flash attention only when the cache's type is the queries'
     * (float32), and an attention on the CPU in every layer of every token step
     * would cost more than the GPU saves.
     */
    ggml_type cacheType() const { return openCL_ ? GGML_TYPE_F32 : GGML_TYPE_F16; }
    /** "Vulkan0: <GPU name>", "GPUOpenCL: <GPU name>", or "CPU (<why the GPU is not used>)". */
    const std::string & device() const { return device_; }
    /** Where weights and caches live: the GPU's memory, or the CPU's. */
    ggml_backend_buffer_type_t weightBuffer() const;
    /** A scheduler over the GPU (when there is one, and not `cpuOnly`) and the CPU. The caller frees it. */
    ggml_backend_sched_t newScheduler(size_t graphSize, bool cpuOnly = false) const;
    /** Whether the GPU runs ggml_sin, which the vocoder needs in every block; ggml-opencl does not. */
    bool gpuRunsSine() const;
    /** Throws, naming the op, when neither backend can run a node (ggml would abort). */
    void checkSupported(ggml_cgraph * graph) const;

private:
    ggml_backend_t gpu_ = nullptr;
    ggml_backend_t cpu_ = nullptr;
    bool openCL_ = false;
    std::string device_;
};

/** A no-alloc ggml context and graph for one computation. */
class Graph {
public:
    explicit Graph(size_t maxNodes);
    ~Graph();
    Graph(const Graph &) = delete;
    Graph & operator=(const Graph &) = delete;

    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;

    /** A tensor the caller fills after allocation. */
    ggml_tensor * input(ggml_type type, int64_t ne0, int64_t ne1 = 1, int64_t ne2 = 1);
    void expand(ggml_tensor * t) { ggml_build_forward_expand(graph, t); }
    /** Run `t` on the CPU even when the GPU could. */
    void onCpu(ggml_tensor * t) { cpuNodes_.push_back(t); }
    /** Allocate on `sched`, then the caller sets inputs and calls compute(). */
    void allocate(const Runtime & runtime, ggml_backend_sched_t sched);
    void compute(ggml_backend_sched_t sched);

private:
    std::vector<ggml_tensor *> cpuNodes_;
};

/**
 * ggml_mul_mat summed in float32. ggml's Vulkan backend otherwise sums matrix
 * products in float16 on GPUs that have it, which these models' large
 * activations overflow.
 */
inline ggml_tensor * matmul(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) {
    ggml_tensor * out = ggml_mul_mat(ctx, a, b);
    ggml_mul_mat_set_prec(out, GGML_PREC_F32);
    return out;
}

/** Allocate every tensor of `ctx` in one buffer of `type`; throws when it does not fit. */
ggml_backend_buffer_t allocateTensors(ggml_context * ctx, ggml_backend_buffer_type_t type, const char * what);

} // namespace kt2
