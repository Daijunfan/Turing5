#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

using Matrix = std::vector<std::vector<std::int64_t>>;
static Matrix multiply(const Matrix& a, const Matrix& b) {
    constexpr std::int64_t mod = 1000000007;
    Matrix c(a.size(), std::vector<std::int64_t>(b.front().size()));
    for (std::size_t i=0;i<a.size();++i) for (std::size_t k=0;k<b.size();++k)
        for (std::size_t j=0;j<b.front().size();++j)
            c[i][j]=(c[i][j]+a[i][k]*b[k][j])%mod;
    return c;
}
int main() {
    Matrix a{{1,2},{3,4}}, b{{2,0},{1,2}}, c{{3,1},{0,1}};
    assert(multiply(multiply(a,b),c) == multiply(a,multiply(b,c)));
    std::cout << "persistent_matrix associativity=pass\n";
}
