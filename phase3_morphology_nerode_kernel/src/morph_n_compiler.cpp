#include "morphn_exact.hpp"
#include <cassert>
#include <iostream>

static morphn::Cost evaluate(
    const std::vector<morphn::Matrix>& tasks,
    morphn::Vector residual,
    const std::vector<int>& word) {
    morphn::Cost value = 0;
    for (int task : word) {
        auto step = morphn::advance(residual, tasks.at(task));
        value += step.increment;
        residual = std::move(step.residual);
    }
    return value;
}

int main() {
    using namespace morphn;
    const std::vector<Matrix> tasks{
        {{0,4,2},{1,0,2},{1,1,0}},
        {{0,1,2},{4,0,2},{1,1,0}}
    };
    for (unsigned mask = 0; mask < 256; ++mask) {
        std::vector<int> word;
        for (int bit = 0; bit < 8; ++bit) word.push_back((mask >> bit) & 1U);
        assert(evaluate(tasks, {0, INF, INF}, word) >= 0);
    }
    std::cout << "morph_n_compiler exhaustive_words=256 status=pass\n";
}
