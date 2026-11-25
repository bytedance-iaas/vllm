// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "csrc/buffers/magic.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { deep_ep::magic::register_apis(m); }
