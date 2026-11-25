#pragma once

#include "ggml-backend.h"

#include <memory>

class ggml_cix_projection {
public:
    ggml_cix_projection();
    ~ggml_cix_projection();

    bool enabled() const;
    ggml_status compute(ggml_backend_t backend, ggml_cgraph * graph,
                        ggml_status (*cpu_compute)(ggml_backend_t, ggml_cgraph *));

private:
    struct state;
    std::unique_ptr<state> state_;
};
