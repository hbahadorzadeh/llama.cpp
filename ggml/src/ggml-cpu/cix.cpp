#include "cix.h"

#include "ggml-impl.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#include <cix_noe_standard_api.h>

namespace {
constexpr int width = 2560;
constexpr int outputs = 9728;

bool host_vector(const ggml_tensor * tensor, int length) {
    return tensor && tensor->type == GGML_TYPE_F32 && tensor->ne[0] == length &&
           tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1 &&
           ggml_is_contiguous(tensor) && tensor->data && tensor->buffer &&
           ggml_backend_buffer_is_host(tensor->buffer);
}

ggml_cgraph cpu_slice(const ggml_cgraph * graph, int first, int end) {
    ggml_cgraph slice{};
    slice.n_nodes = end - first;
    slice.nodes = graph->nodes + first;
    slice.use_counts = graph->use_counts;
    slice.visited_hash_set = graph->visited_hash_set;
    slice.order = graph->order;
    return slice;
}

// Serialize NOE contexts across CPU backends sharing the same AIPU.
std::mutex noe_mutex;
}

struct ggml_cix_projection::state {
    state(const char * graph_path, const char * weight_path) : graph_path(graph_path), weight_path(weight_path) {}

    ~state() {
        std::lock_guard<std::mutex> lock(noe_mutex);
        if (job_created) {
            noe_clean_job(context, job);
        }
        if (graph_loaded) {
            noe_unload_graph(context, graph);
        }
        if (context) {
            noe_deinit_context(context);
        }
    }

    bool initialize() {
        if (noe_init_context(&context) != 0 || noe_load_graph(context, graph_path.c_str(), &graph, nullptr) != 0) {
            return false;
        }
        graph_loaded = true;
        uint32_t inputs, results;
        if (noe_get_tensor_count(context, graph, NOE_TENSOR_TYPE_INPUT, &inputs) != 0 || inputs != 1 ||
            noe_get_tensor_count(context, graph, NOE_TENSOR_TYPE_OUTPUT, &results) != 0 || results != 1 ||
            noe_get_tensor_descriptor(context, graph, NOE_TENSOR_TYPE_INPUT, 0, &input) != 0 ||
            noe_get_tensor_descriptor(context, graph, NOE_TENSOR_TYPE_OUTPUT, 0, &output) != 0 ||
            input.size != width || output.size != outputs ||
            input.data_type != NOE_DATA_TYPE_S8 || output.data_type != NOE_DATA_TYPE_S8 ||
            !std::isfinite(input.scale) || !std::isfinite(output.scale) ||
            input.scale <= 0 || output.scale <= 0) {
            return false;
        }
        job_config_npu_t npu_config{};
        job_config_t config{&npu_config};
        job_created = noe_create_job(context, graph, &job, &config) == 0;
        return job_created;
    }

    bool matches(const ggml_tensor * node) {
        const auto * weight = node->src[0];
        if (node->op != GGML_OP_MUL_MAT || !weight || weight->type != GGML_TYPE_Q4_K ||
            weight->ne[0] != width || weight->ne[1] != outputs || weight->ne[2] != 1 || weight->ne[3] != 1 ||
            std::strcmp(weight->name, "blk.0.ffn_gate.weight") != 0 || !ggml_is_contiguous(weight) ||
            !weight->data || !weight->buffer || !ggml_backend_buffer_is_host(weight->buffer) ||
            !host_vector(node->src[1], width) || !host_vector(node, outputs)) {
            return false;
        }

        // The CIX graph contains fixed weights. Compare bytes rather than trusting a tensor name or address.
        if (weights.empty()) {
            weights.resize(ggml_nbytes(weight));
            std::ifstream file(weight_path, std::ios::binary);
            if (!file.read(weights.data(), weights.size()) || file.peek() != std::ifstream::traits_type::eof()) {
                GGML_LOG_WARN("CIX: unable to read projection weights; using CPU\n");
                disabled = true;
                return false;
            }
        }
        return std::memcmp(weights.data(), weight->data, weights.size()) == 0;
    }

    bool run(const ggml_tensor * activation, ggml_tensor * result) {
        std::lock_guard<std::mutex> lock(noe_mutex);

        std::array<int8_t, width> packed_input;
        std::array<int8_t, outputs> packed_output;
        const auto * x = static_cast<const float *>(activation->data);
        for (int i = 0; i < width; ++i) {
            if (!std::isfinite(x[i])) {
                return false;
            }
        }

        if (!job_created && !initialize()) {
            GGML_LOG_WARN("CIX: projection initialization failed; using CPU\n");
            disabled = true;
            return false;
        }

        for (int i = 0; i < width; ++i) {
            const float scaled = x[i] * input.scale - input.zero_point;
            packed_input[i] = static_cast<int8_t>(std::lrintf(std::fmin(127.0f, std::fmax(-128.0f, scaled))));
        }

        if (noe_load_tensor(context, job, 0, packed_input.data()) != 0 ||
            noe_job_infer_sync(context, job, 1000) != 0 ||
            noe_get_tensor(context, job, NOE_TENSOR_TYPE_OUTPUT, 0, packed_output.data()) != 0) {
            GGML_LOG_WARN("CIX: projection execution failed; using CPU\n");
            disabled = true;
            return false;
        }

        auto * y = static_cast<float *>(result->data);
        for (int i = 0; i < outputs; ++i) {
            y[i] = (static_cast<float>(packed_output[i]) + output.zero_point) / output.scale;
        }
        if (!reported) {
            GGML_LOG_INFO("CIX: Qwen layer 0 FFN gate executed on NPU\n");
            reported = true;
        }
        return true;
    }

    std::string graph_path, weight_path;
    std::vector<char> weights;
    context_handler_t * context = nullptr;
    uint64_t graph = 0;
    uint64_t job = 0;
    tensor_desc_t input{}, output{};
    bool graph_loaded = false;
    bool job_created = false;
    bool disabled = false;
    bool reported = false;
};

ggml_cix_projection::ggml_cix_projection() {
    const char * graph_path = std::getenv("GGML_CIX_GRAPH");
    const char * weight_path = std::getenv("GGML_CIX_WEIGHTS");
    if (graph_path && *graph_path && weight_path && *weight_path) {
        state_ = std::make_unique<state>(graph_path, weight_path);
    }
}

ggml_cix_projection::~ggml_cix_projection() = default;

bool ggml_cix_projection::enabled() const {
    return state_ && !state_->disabled;
}

ggml_status ggml_cix_projection::compute(ggml_backend_t backend, ggml_cgraph * graph,
                                         ggml_status (*cpu_compute)(ggml_backend_t, ggml_cgraph *)) {
    int start = 0;
    const int count = ggml_graph_n_nodes(graph);
    for (int i = 0; enabled() && i < count; ++i) {
        auto * node = ggml_graph_node(graph, i);
        if (!state_->matches(node)) {
            continue;
        }
        if (start < i) {
            auto prefix = cpu_slice(graph, start, i);
            const auto status = cpu_compute(backend, &prefix);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
        }
        start = state_->run(node->src[1], node) ? i + 1 : i;
    }
    if (start == 0) {
        return cpu_compute(backend, graph);
    }
    if (start == count) {
        return GGML_STATUS_SUCCESS;
    }
    auto suffix = cpu_slice(graph, start, count);
    return cpu_compute(backend, &suffix);
}
