#include "morphn_exact.hpp"
#include <cassert>
#include <iostream>
#include <map>
#include <queue>

int main() {
    using namespace morphn;
    const std::vector<Matrix> tasks{
        {{0,4,2},{1,0,2},{1,1,0}},
        {{0,1,2},{4,0,2},{1,1,0}}
    };
    std::map<Vector, std::size_t> index;
    std::queue<Vector> queue;
    Vector initial{0, INF, INF};
    index.emplace(initial, 0); queue.push(initial);
    std::size_t transitions = 0;
    while (!queue.empty()) {
        Vector source = queue.front(); queue.pop();
        for (const Matrix& task : tasks) {
            Vector target = advance(source, task).residual;
            if (index.emplace(target, index.size()).second) queue.push(target);
            ++transitions;
        }
    }
    assert(transitions == index.size() * tasks.size());
    std::cout << "residual_closure states=" << index.size()
              << " transitions=" << transitions << " unprocessed=0\n";
}
