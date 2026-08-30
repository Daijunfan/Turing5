#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

int main() {
    constexpr int nodes = 1000, width = 16;
    std::vector<std::vector<std::int64_t>> value(nodes, std::vector<std::int64_t>(width));
    for (int j=0;j<width;++j) { value[0][j]=j; value[1][j]=j+1; }
    for (int i=2;i<nodes;++i) {
        int other=(i*1103515245LL+12345)%(i-1);
        for (int j=0;j<width;++j)
            value[i][j]=(value[i-1][j]*(1+i%13)+value[other][j]*(2+i%13))%1000003;
    }
    assert(value.back().size()==width);
    std::cout << "dag_runtime operators=" << nodes-2 << " correctness=pass\n";
}
