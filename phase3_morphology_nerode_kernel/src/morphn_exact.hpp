#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace morphn {
using Cost = std::int64_t;
constexpr Cost INF = std::numeric_limits<Cost>::max() / 8;
using Vector = std::vector<Cost>;
using Matrix = std::vector<Vector>;

struct Step {
    Cost increment;
    Vector residual;
};

inline Step advance(const Vector& residual, const Matrix& matrix) {
    const std::size_t n = residual.size();
    Vector raw(n, INF);
    for (std::size_t q = 0; q < n; ++q) {
        for (std::size_t p = 0; p < n; ++p) {
            if (residual[p] < INF && matrix[p][q] < INF) {
                raw[q] = std::min(raw[q], residual[p] + matrix[p][q]);
            }
        }
    }
    const Cost base = *std::min_element(raw.begin(), raw.end());
    if (base >= INF) throw std::runtime_error("dead configuration");
    for (Cost& value : raw) if (value < INF) value -= base;
    return {base, raw};
}
}
