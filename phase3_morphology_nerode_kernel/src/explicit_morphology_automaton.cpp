#include "morphn_exact.hpp"
#include <cassert>
#include <iostream>

int main() {
    using namespace morphn;
    const std::vector<Matrix> tasks{
        {{0,4,2},{1,0,2},{1,1,0}},
        {{0,1,2},{4,0,2},{1,1,0}}
    };
    Vector residual{0, INF, INF};
    Cost total = 0;
    for (int task : {0,1,1,0,1,0}) {
        Step step = advance(residual, tasks.at(task));
        total += step.increment;
        residual = std::move(step.residual);
    }
    assert(total == 0);
    std::cout << "explicit_automaton exact_total=" << total << "\n";
}
