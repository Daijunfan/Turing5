#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

int main() {
    const std::vector<std::int64_t> dims{1,2,4,1,6,6};
    const int n = static_cast<int>(dims.size()) - 1;
    const auto inf = std::numeric_limits<std::int64_t>::max() / 8;
    std::vector<std::vector<std::int64_t>> dp(n, std::vector<std::int64_t>(n, inf));
    for (int i = 0; i < n; ++i) dp[i][i] = 0;
    for (int length = 2; length <= n; ++length) {
        for (int i = 0; i + length <= n; ++i) {
            int j = i + length - 1;
            for (int k = i; k < j; ++k) {
                dp[i][j] = std::min(dp[i][j], dp[i][k] + dp[k+1][j]
                    + dims[i] * dims[k+1] * dims[j+1]);
            }
        }
    }
    assert(dp[0][n-1] == 52);
    std::cout << "interval_pricing exact_min_work=" << dp[0][n-1] << "\n";
}
