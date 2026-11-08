#include "runtime.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>

#include "ggml-cpu.h"
#include "gpu_check.h"

#ifdef KITTENTTS2_VULKAN
#include <vulkan/vulkan.h>
#endif

namespace kt2 {

namespace {

#ifdef KITTENTTS2_VULKAN
std::string vulkanVersion(uint32_t version) {
    return std::to_string(VK_API_VERSION_MAJOR(version)) + "." + std::to_string(VK_API_VERSION_MINOR(version));
}

/** Calls `each` with the properties of every Vulkan device; false when Vulkan would not start. */
template <typename Each>
bool eachVulkanDevice(Each each) {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    info.pApplicationInfo = &app;
    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&info, nullptr, &instance) != VK_SUCCESS) return false;
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());
    for (VkPhysicalDevice device : devices) {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(device, &properties);
        each(properties);
    }
    vkDestroyInstance(instance, nullptr);
    return true;
}
#endif

/** The Vulkan the device offers, for when ggml-vulkan finds no GPU: it needs Vulkan 1.2. */
std::string vulkanSummary() {
#ifdef KITTENTTS2_VULKAN
    uint32_t instanceVersion = VK_API_VERSION_1_0;
    vkEnumerateInstanceVersion(&instanceVersion);
    std::string out = "Vulkan " + vulkanVersion(instanceVersion);
    const bool started = eachVulkanDevice([&](const VkPhysicalDeviceProperties & properties) {
        out += std::string(", ") + properties.deviceName + " with Vulkan " + vulkanVersion(properties.apiVersion);
    });
    return started ? out : out + ", which would not start";
#else
    return "a build without Vulkan";
#endif
}

/**
 * Why the GPU named `name` is left alone unless KITTENTTS2_DEVICE=gpu, or
 * nullptr. Each was measured on a real phone on TestMu (November 2026), with
 * every ggml-vulkan setting that might have helped (see "Android" in
 * docs/guides/kitten-tts-2-on-device.md). Matched by Vulkan vendor ID, so the
 * whole family is left alone, not only the GPU that was measured.
 */
const char * knownGpuProblem(const char * name) {
#ifdef KITTENTTS2_VULKAN
    uint32_t vendor = 0;
    eachVulkanDevice([&](const VkPhysicalDeviceProperties & properties) {
        if (std::strcmp(properties.deviceName, name) == 0) vendor = properties.vendorID;
    });
    if (vendor == 0x5143) return "on Qualcomm Adreno GPUs ggml-vulkan loses the device (Adreno 750)";
    if (vendor == 0x13B5) return "on Arm Mali GPUs ggml-vulkan gives noise, slower than the CPU (Mali-G715)";
#else
    (void)name;
#endif
    return nullptr;
}

} // namespace

GpuChoice gpuChoiceFromEnvironment() {
    const char * device = std::getenv("KITTENTTS2_DEVICE");
    if (device && std::strcmp(device, "cpu") == 0) return GpuChoice::Off;
    if (device && std::strcmp(device, "gpu") == 0) return GpuChoice::Always;
    return GpuChoice::Auto;
}

Runtime::Runtime(GpuChoice choice, int threads) {
    // Why the GPU is not used, reported as "CPU (<reason>)".
    std::string reason;
    if (choice == GpuChoice::Off) reason = "GPU turned off";
    for (size_t i = 0; choice != GpuChoice::Off && i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        auto type = ggml_backend_dev_type(dev);
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
        const char * name = ggml_backend_dev_description(dev);
        if (const char * problem = choice == GpuChoice::Auto ? knownGpuProblem(name) : nullptr) {
            // Skipped before ggml-vulkan starts it, which compiles its shaders and fills its memory.
            reason = std::string(name) + ": " + problem;
            continue;
        }
        try {
            gpu_ = ggml_backend_dev_init(dev, nullptr);
        } catch (const std::exception & error) {
            // ggml-vulkan throws for a GPU it cannot use (no 16-bit storage, say): run on the CPU.
            gpu_ = nullptr;
            reason = std::string(name) + ": " + error.what();
        }
        if (gpu_) {
            device_ = std::string(ggml_backend_dev_name(dev)) + ": " + name;
            break;
        }
    }
    if (reason.empty()) reason = "ggml-vulkan found no GPU it can use (it needs Vulkan 1.2): " + vulkanSummary();
    cpu_ = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu_) throw std::runtime_error("ggml has no CPU backend");
    if (threads <= 0) {
        // Phones pair a few fast cores with slow ones; more threads than fast cores only waits on the slow ones.
        threads = static_cast<int>(std::clamp(std::thread::hardware_concurrency() / 2, 2u, 4u));
    }
    ggml_backend_cpu_set_n_threads(cpu_, threads);
    if (!gpu_) device_ = "CPU (" + reason + ")";
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
    // KITTENTTS2_CHECK=<file>: a slow diagnostic for GPUs that fail (gpu_check.h).
    if (computeChecked(sched, graph)) return;
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
