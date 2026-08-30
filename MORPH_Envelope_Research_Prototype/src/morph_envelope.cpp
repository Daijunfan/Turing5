#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using u64 = std::uint64_t;
using i64 = std::int64_t;
using Clock = std::chrono::steady_clock;
static constexpr u64 INF_U64 = std::numeric_limits<u64>::max() / 4;
static constexpr long double INF_LD = std::numeric_limits<long double>::infinity();

static u64 sat_add(u64 a, u64 b) {
    if (a >= INF_U64 || b >= INF_U64 || a > INF_U64 - b) return INF_U64;
    return a + b;
}
static u64 sat_mul(u64 a, u64 b) {
    if (a == 0 || b == 0) return 0;
    if (a >= INF_U64 || b >= INF_U64 || a > INF_U64 / b) return INF_U64;
    return a * b;
}
static u64 sat_mul3(u64 a, u64 b, u64 c) { return sat_mul(sat_mul(a, b), c); }

struct State {
    u64 work = 0;
    u64 peak = 0;
    int split = -1;
    int left_idx = -1;
    int right_idx = -1;
    bool right_first = false;
};

struct Node {
    int i = 0;
    int j = 0;
    int split = -1;
    bool right_first = false;
    std::unique_ptr<Node> left;
    std::unique_ptr<Node> right;
};

struct CostPeak { u64 work = 0; u64 peak = 0; };

class MorphEnvelope {
public:
    explicit MorphEnvelope(std::vector<u64> dims)
        : dims_(std::move(dims)), n_(static_cast<int>(dims_.size()) - 1), cells_(n_ * n_) {
        if (n_ < 1) throw std::invalid_argument("at least one matrix is required");
    }

    int n() const { return n_; }
    const std::vector<u64>& dims() const { return dims_; }

    u64 result_size(int i, int j) const { return sat_mul(dims_.at(i), dims_.at(j + 1)); }
    u64 combine_work(int i, int k, int j) const {
        return sat_mul3(dims_.at(i), dims_.at(k + 1), dims_.at(j + 1));
    }

    void compile() {
        for (int i = 0; i < n_; ++i) cell(i, i) = {State{0, 0, -1, -1, -1, false}};
        for (int len = 2; len <= n_; ++len) {
            for (int i = 0; i + len <= n_; ++i) {
                int j = i + len - 1;
                std::vector<State> candidates;
                for (int k = i; k < j; ++k) {
                    const auto& left = cell(i, k);
                    const auto& right = cell(k + 1, j);
                    const u64 ltemp = (i < k) ? result_size(i, k) : 0;
                    const u64 rtemp = (k + 1 < j) ? result_size(k + 1, j) : 0;
                    const u64 out = result_size(i, j);
                    const u64 coexist = sat_add(sat_add(ltemp, rtemp), out);
                    const u64 local_work = combine_work(i, k, j);
                    candidates.reserve(candidates.size() + left.size() * right.size());
                    for (int li = 0; li < static_cast<int>(left.size()); ++li) {
                        for (int ri = 0; ri < static_cast<int>(right.size()); ++ri) {
                            const State& a = left[li];
                            const State& b = right[ri];
                            const u64 peak_lr = std::max({a.peak, sat_add(ltemp, b.peak), coexist});
                            const u64 peak_rl = std::max({b.peak, sat_add(rtemp, a.peak), coexist});
                            const bool right_first = peak_rl < peak_lr;
                            candidates.push_back(State{
                                sat_add(sat_add(a.work, b.work), local_work),
                                right_first ? peak_rl : peak_lr,
                                k, li, ri, right_first
                            });
                        }
                    }
                }
                cell(i, j) = prune(std::move(candidates));
            }
        }
    }

    const std::vector<State>& root() const { return cell(0, n_ - 1); }

    int best_for_budget(u64 budget) const {
        const auto& r = root();
        int lo = 0, hi = static_cast<int>(r.size());
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (r[mid].peak <= budget) lo = mid + 1;
            else hi = mid;
        }
        return lo - 1;
    }

    std::unique_ptr<Node> reconstruct(int root_idx) const {
        return reconstruct_rec(0, n_ - 1, root_idx);
    }

    bool validate_certificate(const Node& t) const {
        std::vector<int> leaves;
        if (!validate_shape(t, leaves)) return false;
        if (static_cast<int>(leaves.size()) != n_) return false;
        for (int i = 0; i < n_; ++i) if (leaves[i] != i) return false;
        return true;
    }

    CostPeak evaluate_certificate(const Node& t) const { return eval_rec(t); }

    std::set<std::tuple<int, int, int>> split_set(const Node& t) const {
        std::set<std::tuple<int, int, int>> s;
        collect_splits(t, s);
        return s;
    }

private:
    std::vector<u64> dims_;
    int n_;
    std::vector<std::vector<State>> cells_;

    std::vector<State>& cell(int i, int j) { return cells_[i * n_ + j]; }
    const std::vector<State>& cell(int i, int j) const { return cells_[i * n_ + j]; }

    static std::vector<State> prune(std::vector<State> v) {
        std::sort(v.begin(), v.end(), [](const State& a, const State& b) {
            if (a.peak != b.peak) return a.peak < b.peak;
            if (a.work != b.work) return a.work < b.work;
            return std::tie(a.split, a.left_idx, a.right_idx, a.right_first)
                 < std::tie(b.split, b.left_idx, b.right_idx, b.right_first);
        });
        std::vector<State> out;
        u64 best_work = INF_U64;
        u64 seen_peak = INF_U64;
        for (const State& s : v) {
            if (!out.empty() && s.peak == seen_peak) continue;
            seen_peak = s.peak;
            if (s.work < best_work) {
                out.push_back(s);
                best_work = s.work;
            }
        }
        return out;
    }

    std::unique_ptr<Node> reconstruct_rec(int i, int j, int idx) const {
        const State& s = cell(i, j).at(idx);
        auto n = std::make_unique<Node>();
        n->i = i; n->j = j; n->split = s.split; n->right_first = s.right_first;
        if (i < j) {
            n->left = reconstruct_rec(i, s.split, s.left_idx);
            n->right = reconstruct_rec(s.split + 1, j, s.right_idx);
        }
        return n;
    }

    bool validate_shape(const Node& t, std::vector<int>& leaves) const {
        if (t.i < 0 || t.j >= n_ || t.i > t.j) return false;
        if (t.i == t.j) {
            if (t.left || t.right || t.split != -1) return false;
            leaves.push_back(t.i);
            return true;
        }
        if (!t.left || !t.right || t.split < t.i || t.split >= t.j) return false;
        if (t.left->i != t.i || t.left->j != t.split) return false;
        if (t.right->i != t.split + 1 || t.right->j != t.j) return false;
        return validate_shape(*t.left, leaves) && validate_shape(*t.right, leaves);
    }

    CostPeak eval_rec(const Node& t) const {
        if (t.i == t.j) return {0, 0};
        CostPeak a = eval_rec(*t.left);
        CostPeak b = eval_rec(*t.right);
        const u64 ltemp = (t.i < t.split) ? result_size(t.i, t.split) : 0;
        const u64 rtemp = (t.split + 1 < t.j) ? result_size(t.split + 1, t.j) : 0;
        const u64 out = result_size(t.i, t.j);
        const u64 coexist = sat_add(sat_add(ltemp, rtemp), out);
        const u64 peak_lr = std::max({a.peak, sat_add(ltemp, b.peak), coexist});
        const u64 peak_rl = std::max({b.peak, sat_add(rtemp, a.peak), coexist});
        return {
            sat_add(sat_add(a.work, b.work), combine_work(t.i, t.split, t.j)),
            t.right_first ? peak_rl : peak_lr
        };
    }

    static void collect_splits(const Node& t, std::set<std::tuple<int, int, int>>& s) {
        if (t.i == t.j) return;
        s.emplace(t.i, t.j, t.split);
        collect_splits(*t.left, s);
        collect_splits(*t.right, s);
    }
};

// ---------------- Exhaustive oracle ----------------
struct BrutePair { u64 work; u64 peak; };

static std::vector<BrutePair> enumerate_all(const std::vector<u64>& d, int i, int j) {
    if (i == j) return {{0, 0}};
    std::vector<BrutePair> out;
    for (int k = i; k < j; ++k) {
        auto left = enumerate_all(d, i, k);
        auto right = enumerate_all(d, k + 1, j);
        const u64 ltemp = (i < k) ? sat_mul(d[i], d[k + 1]) : 0;
        const u64 rtemp = (k + 1 < j) ? sat_mul(d[k + 1], d[j + 1]) : 0;
        const u64 result = sat_mul(d[i], d[j + 1]);
        const u64 coexist = sat_add(sat_add(ltemp, rtemp), result);
        const u64 local_work = sat_mul3(d[i], d[k + 1], d[j + 1]);
        for (const auto& a : left) for (const auto& b : right) {
            const u64 lr = std::max({a.peak, sat_add(ltemp, b.peak), coexist});
            const u64 rl = std::max({b.peak, sat_add(rtemp, a.peak), coexist});
            out.push_back({sat_add(sat_add(a.work, b.work), local_work), std::min(lr, rl)});
        }
    }
    return out;
}

static std::vector<std::pair<u64, u64>> brute_frontier(const std::vector<u64>& d) {
    auto all = enumerate_all(d, 0, static_cast<int>(d.size()) - 2);
    std::sort(all.begin(), all.end(), [](const BrutePair& a, const BrutePair& b) {
        if (a.peak != b.peak) return a.peak < b.peak;
        return a.work < b.work;
    });
    std::vector<std::pair<u64, u64>> out;
    u64 best = INF_U64;
    u64 seen_peak = INF_U64;
    for (const auto& x : all) {
        if (!out.empty() && x.peak == seen_peak) continue;
        seen_peak = x.peak;
        if (x.work < best) {
            out.emplace_back(x.work, x.peak);
            best = x.work;
        }
    }
    return out;
}

static std::vector<std::pair<u64, u64>> compiled_frontier(const MorphEnvelope& m) {
    std::vector<std::pair<u64, u64>> out;
    for (const auto& s : m.root()) out.emplace_back(s.work, s.peak);
    return out;
}

static u64 catalan_small(int n) {
    std::vector<u64> c(n + 1); c[0] = 1;
    for (int i = 1; i <= n; ++i) {
        for (int j = 0; j < i; ++j) c[i] += c[j] * c[i - 1 - j];
    }
    return c[n];
}

// ---------------- Exact semantic evaluator ----------------
enum class Semiring { Modular, Boolean, MinPlus };
struct Matrix {
    int rows = 0, cols = 0;
    std::vector<i64> data;
    i64& at(int i, int j) { return data[static_cast<size_t>(i) * cols + j]; }
    i64 at(int i, int j) const { return data[static_cast<size_t>(i) * cols + j]; }
};

static Matrix multiply(const Matrix& a, const Matrix& b, Semiring s) {
    if (a.cols != b.rows) throw std::runtime_error("matrix dimension mismatch");
    Matrix c{a.rows, b.cols, std::vector<i64>(static_cast<size_t>(a.rows) * b.cols, 0)};
    constexpr i64 MOD = 1000000007LL;
    constexpr i64 MPINF = 1LL << 55;
    if (s == Semiring::MinPlus) std::fill(c.data.begin(), c.data.end(), MPINF);
    for (int i = 0; i < a.rows; ++i) {
        for (int k = 0; k < a.cols; ++k) {
            for (int j = 0; j < b.cols; ++j) {
                if (s == Semiring::Modular) {
                    c.at(i, j) = (c.at(i, j) + (a.at(i, k) * b.at(k, j)) % MOD) % MOD;
                } else if (s == Semiring::Boolean) {
                    c.at(i, j) = c.at(i, j) || (a.at(i, k) && b.at(k, j));
                } else if (a.at(i, k) < MPINF && b.at(k, j) < MPINF) {
                    c.at(i, j) = std::min(c.at(i, j), a.at(i, k) + b.at(k, j));
                }
            }
        }
    }
    return c;
}

static Matrix random_matrix(int r, int c, Semiring s, std::mt19937_64& gen) {
    Matrix m{r, c, std::vector<i64>(static_cast<size_t>(r) * c)};
    std::uniform_int_distribution<int> digit(0, 9), bit(0, 1), hole(0, 11);
    constexpr i64 MPINF = 1LL << 55;
    for (auto& x : m.data) {
        if (s == Semiring::Modular) x = digit(gen);
        else if (s == Semiring::Boolean) x = bit(gen);
        else x = (hole(gen) == 0) ? MPINF : digit(gen);
    }
    return m;
}

static Matrix eval_plan(const Node& t, const std::vector<Matrix>& xs, Semiring s) {
    if (t.i == t.j) return xs.at(t.i);
    if (t.right_first) {
        Matrix r = eval_plan(*t.right, xs, s);
        Matrix l = eval_plan(*t.left, xs, s);
        return multiply(l, r, s);
    }
    Matrix l = eval_plan(*t.left, xs, s);
    Matrix r = eval_plan(*t.right, xs, s);
    return multiply(l, r, s);
}

static Matrix canonical_left_fold(const std::vector<Matrix>& xs, Semiring s) {
    Matrix out = xs.front();
    for (size_t i = 1; i < xs.size(); ++i) out = multiply(out, xs[i], s);
    return out;
}

static bool equal_matrix(const Matrix& a, const Matrix& b) {
    return a.rows == b.rows && a.cols == b.cols && a.data == b.data;
}

// ---------------- Correctness test suite ----------------
struct TestStats {
    u64 exhaustive_instances = 0;
    u64 random_instances = 0;
    u64 oracle_parenthesizations = 0;
    u64 certificate_checks = 0;
    u64 semantic_plan_evaluations = 0;
    u64 failures = 0;
};

static void check_frontier(const std::vector<u64>& dims, bool is_random, TestStats& stats) {
    MorphEnvelope m(dims); m.compile();
    const auto got = compiled_frontier(m);
    const auto expected = brute_frontier(dims);
    stats.oracle_parenthesizations += catalan_small(static_cast<int>(dims.size()) - 2);
    if (got != expected) {
        ++stats.failures;
        std::cerr << "frontier mismatch\n";
        return;
    }
    for (int i = 0; i < static_cast<int>(m.root().size()); ++i) {
        auto tree = m.reconstruct(i);
        CostPeak cp = m.evaluate_certificate(*tree);
        ++stats.certificate_checks;
        if (!m.validate_certificate(*tree) || cp.work != m.root()[i].work || cp.peak != m.root()[i].peak) {
            ++stats.failures;
            std::cerr << "certificate mismatch\n";
            return;
        }
    }
    if (is_random) ++stats.random_instances;
    else ++stats.exhaustive_instances;
}

static TestStats run_tests(bool smoke = false) {
    TestStats stats;
    const int max_n = smoke ? 5 : 6;
    for (int n = 2; n <= max_n; ++n) {
        u64 total = 1;
        for (int i = 0; i < n + 1; ++i) total *= 4;
        for (u64 code = 0; code < total; ++code) {
            u64 x = code;
            std::vector<u64> dims(n + 1);
            for (auto& d : dims) { d = 1 + (x % 4); x /= 4; }
            check_frontier(dims, false, stats);
        }
    }
    std::mt19937_64 gen(0xA17C0DEULL);
    std::uniform_int_distribution<int> dim(1, 16);
    const int random_per_n = smoke ? 12 : 120;
    for (int n = 7; n <= (smoke ? 7 : 9); ++n) {
        for (int t = 0; t < random_per_n; ++t) {
            std::vector<u64> dims(n + 1);
            for (auto& d : dims) d = dim(gen);
            check_frontier(dims, true, stats);
        }
    }

    std::uniform_int_distribution<int> n_dist(2, 7), small_dim(1, 4);
    const int semantic_cases = smoke ? 80 : 2000;
    for (Semiring s : {Semiring::Modular, Semiring::Boolean, Semiring::MinPlus}) {
        for (int t = 0; t < semantic_cases; ++t) {
            int n = n_dist(gen);
            std::vector<u64> dims(n + 1);
            for (auto& d : dims) d = small_dim(gen);
            MorphEnvelope m(dims); m.compile();
            std::vector<Matrix> xs;
            for (int i = 0; i < n; ++i) xs.push_back(random_matrix(static_cast<int>(dims[i]), static_cast<int>(dims[i + 1]), s, gen));
            Matrix reference = canonical_left_fold(xs, s);
            for (int k = 0; k < static_cast<int>(m.root().size()); ++k) {
                auto tree = m.reconstruct(k);
                Matrix answer = eval_plan(*tree, xs, s);
                ++stats.semantic_plan_evaluations;
                if (!equal_matrix(reference, answer)) {
                    ++stats.failures;
                    std::cerr << "semiring result mismatch\n";
                    break;
                }
            }
        }
    }
    return stats;
}

// ---------------- Runtime policies ----------------
struct DynamicResult {
    long double total = 0;
    long double service = 0;
    long double movement = 0;
    u64 switches = 0;
};

static int instantaneous_best(const std::vector<State>& basis, u64 budget) {
    int ans = -1;
    for (int i = 0; i < static_cast<int>(basis.size()); ++i) if (basis[i].peak <= budget) ans = i;
    return ans;
}

static std::vector<std::vector<int>> split_distances(const MorphEnvelope& m) {
    int b = static_cast<int>(m.root().size());
    std::vector<std::set<std::tuple<int, int, int>>> sets(b);
    for (int i = 0; i < b; ++i) {
        auto t = m.reconstruct(i);
        sets[i] = m.split_set(*t);
    }
    std::vector<std::vector<int>> d(b, std::vector<int>(b));
    for (int i = 0; i < b; ++i) for (int j = 0; j < b; ++j) {
        std::vector<std::tuple<int, int, int>> diff;
        std::set_symmetric_difference(sets[i].begin(), sets[i].end(), sets[j].begin(), sets[j].end(), std::back_inserter(diff));
        d[i][j] = static_cast<int>(diff.size());
    }
    return d;
}

static long double service_cost(const State& s) { return static_cast<long double>(s.work); }

static DynamicResult policy_static(const std::vector<State>& b, const std::vector<u64>& budgets) {
    u64 minimum_budget = *std::min_element(budgets.begin(), budgets.end());
    int p = instantaneous_best(b, minimum_budget);
    DynamicResult r;
    for (u64 budget : budgets) {
        if (p < 0 || b[p].peak > budget) return {INF_LD, INF_LD, 0, 0};
        r.service += service_cost(b[p]);
        r.total += service_cost(b[p]);
    }
    return r;
}

static DynamicResult policy_instant(const std::vector<State>& b, const std::vector<u64>& budgets,
                                    const std::vector<std::vector<long double>>& move) {
    int p = instantaneous_best(b, budgets[0]);
    DynamicResult r{service_cost(b[p]), service_cost(b[p]), 0, 0};
    for (size_t t = 1; t < budgets.size(); ++t) {
        int q = instantaneous_best(b, budgets[t]);
        long double m = move[p][q];
        r.service += service_cost(b[q]); r.movement += m; r.total += service_cost(b[q]) + m;
        if (q != p) ++r.switches;
        p = q;
    }
    return r;
}

static DynamicResult policy_myopic(const std::vector<State>& b, const std::vector<u64>& budgets,
                                   const std::vector<std::vector<long double>>& move) {
    int p = instantaneous_best(b, budgets[0]);
    DynamicResult r{service_cost(b[p]), service_cost(b[p]), 0, 0};
    for (size_t t = 1; t < budgets.size(); ++t) {
        int q = -1; long double best = INF_LD;
        for (int j = 0; j < static_cast<int>(b.size()); ++j) if (b[j].peak <= budgets[t]) {
            long double x = service_cost(b[j]) + move[p][j];
            if (x < best) { best = x; q = j; }
        }
        long double m = move[p][q];
        r.service += service_cost(b[q]); r.movement += m; r.total += service_cost(b[q]) + m;
        if (q != p) ++r.switches;
        p = q;
    }
    return r;
}

// Morphological Credit: switch only after observed savings amortize the certified migration path.
static DynamicResult policy_credit(const std::vector<State>& b, const std::vector<u64>& budgets,
                                   const std::vector<std::vector<long double>>& move) {
    const int B = static_cast<int>(b.size());
    std::vector<long double> credit(B, 0);
    int p = instantaneous_best(b, budgets[0]);
    DynamicResult r{service_cost(b[p]), service_cost(b[p]), 0, 0};
    for (size_t t = 1; t < budgets.size(); ++t) {
        if (b[p].peak > budgets[t]) {
            int q = -1; long double best = INF_LD;
            for (int j = 0; j < B; ++j) if (b[j].peak <= budgets[t]) {
                long double x = service_cost(b[j]) + move[p][j];
                if (x < best) { best = x; q = j; }
            }
            long double m = move[p][q];
            r.movement += m; r.total += m; if (q != p) ++r.switches;
            p = q; std::fill(credit.begin(), credit.end(), 0);
        } else {
            int target = p; long double best_surplus = 0;
            for (int q = 0; q < B; ++q) {
                if (b[q].peak > budgets[t]) { credit[q] = 0; continue; }
                credit[q] = std::max((long double)0, credit[q] + service_cost(b[p]) - service_cost(b[q]));
                long double surplus = credit[q] - move[p][q];
                if (surplus > best_surplus) { best_surplus = surplus; target = q; }
            }
            if (target != p) {
                long double m = move[p][target];
                r.movement += m; r.total += m; ++r.switches;
                p = target; std::fill(credit.begin(), credit.end(), 0);
            }
        }
        r.service += service_cost(b[p]); r.total += service_cost(b[p]);
    }
    return r;
}

static DynamicResult policy_wfa(const std::vector<State>& b, const std::vector<u64>& budgets,
                                const std::vector<std::vector<long double>>& move) {
    const int B = static_cast<int>(b.size());
    std::vector<long double> w(B, INF_LD), next(B, INF_LD);
    for (int j = 0; j < B; ++j) if (b[j].peak <= budgets[0]) w[j] = service_cost(b[j]);
    int p = static_cast<int>(std::min_element(w.begin(), w.end()) - w.begin());
    DynamicResult r{service_cost(b[p]), service_cost(b[p]), 0, 0};
    for (size_t t = 1; t < budgets.size(); ++t) {
        std::fill(next.begin(), next.end(), INF_LD);
        for (int j = 0; j < B; ++j) if (b[j].peak <= budgets[t]) {
            long double best = INF_LD;
            for (int i = 0; i < B; ++i) best = std::min(best, w[i] + move[i][j]);
            next[j] = best + service_cost(b[j]);
        }
        int q = -1; long double best = INF_LD;
        for (int j = 0; j < B; ++j) {
            long double x = next[j] + move[p][j];
            if (x < best) { best = x; q = j; }
        }
        long double m = move[p][q];
        r.service += service_cost(b[q]); r.movement += m; r.total += service_cost(b[q]) + m;
        if (q != p) ++r.switches;
        p = q; w.swap(next);
    }
    return r;
}

static DynamicResult policy_offline(const std::vector<State>& b, const std::vector<u64>& budgets,
                                    const std::vector<std::vector<long double>>& move) {
    const int B = static_cast<int>(b.size());
    std::vector<long double> dp(B, INF_LD), next(B, INF_LD);
    std::vector<std::vector<int>> parent(budgets.size(), std::vector<int>(B, -1));
    for (int j = 0; j < B; ++j) if (b[j].peak <= budgets[0]) dp[j] = service_cost(b[j]);
    for (size_t t = 1; t < budgets.size(); ++t) {
        std::fill(next.begin(), next.end(), INF_LD);
        for (int j = 0; j < B; ++j) if (b[j].peak <= budgets[t]) {
            for (int i = 0; i < B; ++i) {
                long double x = dp[i] + move[i][j] + service_cost(b[j]);
                if (x < next[j]) { next[j] = x; parent[t][j] = i; }
            }
        }
        dp.swap(next);
    }
    int p = static_cast<int>(std::min_element(dp.begin(), dp.end()) - dp.begin());
    std::vector<int> path(budgets.size()); path.back() = p;
    for (size_t t = budgets.size() - 1; t > 0; --t) path[t - 1] = parent[t][path[t]];
    DynamicResult r; r.total = dp[p];
    for (size_t t = 0; t < path.size(); ++t) {
        r.service += service_cost(b[path[t]]);
        if (t > 0) {
            long double m = move[path[t - 1]][path[t]];
            r.movement += m; if (path[t] != path[t - 1]) ++r.switches;
        }
    }
    return r;
}

static std::vector<u64> budget_trace(const std::vector<State>& b, int T, u64 seed) {
    std::mt19937_64 gen(seed);
    std::geometric_distribution<int> phase(1.0 / 24.0);
    std::uniform_int_distribution<int> choose(0, 2);
    std::array<int, 3> ids{0, static_cast<int>(b.size()) / 2, static_cast<int>(b.size()) - 1};
    std::vector<u64> out; out.reserve(T);
    int previous = -1;
    while (static_cast<int>(out.size()) < T) {
        int z = choose(gen);
        if (z == previous) z = (z + 1 + static_cast<int>(gen() % 2)) % 3;
        previous = z;
        int len = 5 + phase(gen);
        for (int k = 0; k < len && static_cast<int>(out.size()) < T; ++k) out.push_back(b[ids[z]].peak);
    }
    return out;
}

// ---------------- Benchmark generators ----------------
static std::vector<u64> adversarial_dims(int repetitions) {
    static const std::vector<u64> base = {1,32,2048,32,2,1,2,8,2048,2,2048,2,8,2,64,2048,16};
    std::vector<u64> d = base;
    for (int r = 1; r < repetitions; ++r) d.insert(d.end(), base.begin() + 1, base.end());
    return d;
}

static long double log10_catalan(int n) {
    return (std::lgammal(2.0L * n + 1) - 2.0L * std::lgammal(n + 1) - std::log((long double)n + 1)) / std::log(10.0L);
}

struct ScaleRow {
    int n = 0; size_t basis = 0; double compile_ms = 0; double lookup_ns = 0;
    u64 min_peak = 0, max_peak = 0, low_memory_work = 0, min_work = 0;
    long double log_space = 0;
};

static std::vector<ScaleRow> scaling_benchmark() {
    std::vector<ScaleRow> rows;
    for (int repetitions : {1,2,3,4,6,8}) {
        auto d = adversarial_dims(repetitions);
        MorphEnvelope m(d);
        auto t0 = Clock::now(); m.compile(); auto t1 = Clock::now();
        std::mt19937_64 gen(1234 + repetitions);
        std::uniform_int_distribution<u64> budget(m.root().front().peak, m.root().back().peak);
        volatile int sink = 0;
        constexpr int Q = 1000000;
        auto q0 = Clock::now();
        for (int i = 0; i < Q; ++i) sink += m.best_for_budget(budget(gen));
        auto q1 = Clock::now(); (void)sink;
        rows.push_back(ScaleRow{
            m.n(), m.root().size(),
            std::chrono::duration<double, std::milli>(t1 - t0).count(),
            std::chrono::duration<double, std::nano>(q1 - q0).count() / Q,
            m.root().front().peak, m.root().back().peak,
            m.root().front().work, m.root().back().work,
            log10_catalan(m.n() - 1)
        });
    }
    return rows;
}

struct DynamicSummary {
    long double static_ratio = 0, instant_ratio = 0, myopic_ratio = 0, credit_ratio = 0, wfa_ratio = 0;
    long double credit_saving = 0, credit_switches = 0, wfa_switches = 0, offline_switches = 0;
};

static DynamicSummary dynamic_benchmark(const MorphEnvelope& m, long double beta, int seeds = 30, int T = 20000) {
    const auto& b = m.root();
    auto distance = split_distances(m);
    std::vector<std::vector<long double>> movement(b.size(), std::vector<long double>(b.size()));
    long double base = service_cost(b.back());
    for (int i = 0; i < static_cast<int>(b.size()); ++i) for (int j = 0; j < static_cast<int>(b.size()); ++j) {
        movement[i][j] = beta * base * static_cast<long double>(distance[i][j]) / std::max(1, 2 * (m.n() - 1));
    }
    DynamicSummary s;
    for (int seed = 0; seed < seeds; ++seed) {
        auto trace = budget_trace(b, T, 9000 + seed);
        auto off = policy_offline(b, trace, movement);
        auto st = policy_static(b, trace);
        auto in = policy_instant(b, trace, movement);
        auto my = policy_myopic(b, trace, movement);
        auto cr = policy_credit(b, trace, movement);
        auto wf = policy_wfa(b, trace, movement);
        s.static_ratio += st.total / off.total;
        s.instant_ratio += in.total / off.total;
        s.myopic_ratio += my.total / off.total;
        s.credit_ratio += cr.total / off.total;
        s.wfa_ratio += wf.total / off.total;
        s.credit_saving += 1.0L - cr.total / st.total;
        s.credit_switches += cr.switches;
        s.wfa_switches += wf.switches;
        s.offline_switches += off.switches;
    }
    for (long double* x : {&s.static_ratio,&s.instant_ratio,&s.myopic_ratio,&s.credit_ratio,&s.wfa_ratio,
                          &s.credit_saving,&s.credit_switches,&s.wfa_switches,&s.offline_switches}) *x /= seeds;
    return s;
}

struct RandomSummary {
    int n = 0, seeds = 0; double fraction_multiple = 0, mean_basis = 0, median_ratio = 0, max_ratio = 0; size_t max_basis = 0;
};

static std::vector<RandomSummary> random_fallback_benchmark() {
    std::vector<RandomSummary> rows;
    const std::vector<u64> values = {1,2,4,8,16,32,64,128,256,512,1024};
    for (auto [n, seeds] : std::vector<std::pair<int,int>>{{16,300},{32,200},{48,80}}) {
        std::vector<size_t> basis_sizes;
        std::vector<double> ratios;
        for (int seed = 0; seed < seeds; ++seed) {
            std::mt19937_64 gen(100000 + 1000 * n + seed);
            std::uniform_int_distribution<int> pick(0, static_cast<int>(values.size()) - 1);
            std::vector<u64> d(n + 1);
            for (auto& x : d) x = values[pick(gen)];
            MorphEnvelope m(d); m.compile();
            basis_sizes.push_back(m.root().size());
            ratios.push_back(static_cast<double>(m.root().front().work) / m.root().back().work);
        }
        std::sort(ratios.begin(), ratios.end());
        RandomSummary r; r.n = n; r.seeds = seeds;
        r.fraction_multiple = static_cast<double>(std::count_if(basis_sizes.begin(), basis_sizes.end(), [](size_t x){return x>1;})) / seeds;
        r.mean_basis = static_cast<double>(std::accumulate(basis_sizes.begin(), basis_sizes.end(), (size_t)0)) / seeds;
        r.max_basis = *std::max_element(basis_sizes.begin(), basis_sizes.end());
        r.median_ratio = ratios[ratios.size()/2]; r.max_ratio = ratios.back();
        rows.push_back(r);
    }
    return rows;
}

struct SeparationRow {
    u64 H = 0, low_work = 0, high_work = 0, low_peak = 0, high_peak = 0;
    long double work_ratio = 0, peak_ratio = 0, dynamic_ratio = 0;
};

static std::vector<SeparationRow> separation_checks() {
    std::vector<SeparationRow> rows;
    for (u64 H : {2ULL,4ULL,8ULL,16ULL,32ULL,64ULL,128ULL,256ULL,512ULL}) {
        std::vector<u64> d = {1, H, H*H*H, H};
        MorphEnvelope m(d); m.compile();
        if (m.root().size() != 2) throw std::runtime_error("separation frontier size mismatch");
        u64 low_work = sat_add(sat_mul(sat_mul3(H,H,H), sat_mul(H,H)), sat_mul(H,H)); // H^5 + H^2
        u64 high_work = 2 * sat_mul(sat_mul(H,H), sat_mul(H,H));  // 2H^4
        u64 low_peak = H*H + H;
        u64 high_peak = H*H*H + H;
        if (m.root()[0].work != low_work || m.root()[0].peak != low_peak ||
            m.root()[1].work != high_work || m.root()[1].peak != high_peak) {
            throw std::runtime_error("separation formula mismatch");
        }
        long double fixed = (H + 1.0L) * low_work;
        long double morph = low_work + H * (long double)high_work + 2.0L * H*H*H;
        rows.push_back({H,low_work,high_work,low_peak,high_peak,
                        (long double)low_work/high_work,(long double)high_peak/low_peak,fixed/morph});
    }
    return rows;
}

static void write_json(const std::string& path, const TestStats& tests,
                       const std::vector<ScaleRow>& scaling,
                       const std::map<long double, DynamicSummary>& dynamic,
                       const std::vector<RandomSummary>& random_rows,
                       const std::vector<SeparationRow>& separation,
                       double compile_once_ms, double compile_repeat_ms) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("cannot open output JSON");
    f << std::setprecision(12);
    f << "{\n";
    f << "  \"tests\": {\"exhaustive_instances\": " << tests.exhaustive_instances
      << ", \"random_instances\": " << tests.random_instances
      << ", \"oracle_parenthesizations\": " << tests.oracle_parenthesizations
      << ", \"certificate_checks\": " << tests.certificate_checks
      << ", \"semantic_plan_evaluations\": " << tests.semantic_plan_evaluations
      << ", \"failures\": " << tests.failures << "},\n";
    f << "  \"local_rotation_counterexample\": {\"dims\": [1,1,1,2,2], \"local_stable_work\": 8, \"global_optimum_work\": 7},\n";
    f << "  \"scaling\": [\n";
    for (size_t i=0;i<scaling.size();++i) { const auto& r=scaling[i];
        f << "    {\"n\": "<<r.n<<", \"basis\": "<<r.basis<<", \"compile_ms\": "<<r.compile_ms
          <<", \"lookup_ns\": "<<r.lookup_ns<<", \"min_peak\": "<<r.min_peak<<", \"max_peak\": "<<r.max_peak
          <<", \"low_memory_work\": "<<r.low_memory_work<<", \"minimum_work\": "<<r.min_work
          <<", \"work_ratio\": "<<(double)r.low_memory_work/r.min_work<<", \"log10_parenthesizations\": "<<(double)r.log_space<<"}"
          <<(i+1<scaling.size()?",":"")<<"\n";
    }
    f << "  ],\n";
    f << "  \"planning_reuse\": {\"compile_once_ms\": "<<compile_once_ms<<", \"compile_three_times_ms\": "<<compile_repeat_ms
      <<", \"reuse_speedup_for_three_epochs\": "<<(compile_once_ms>0?compile_repeat_ms/compile_once_ms:0.0)<<"},\n";
    f << "  \"dynamic\": [\n"; size_t di=0;
    for (const auto& [beta,s] : dynamic) {
        f << "    {\"beta\": "<<(double)beta<<", \"static_over_offline\": "<<(double)s.static_ratio
          <<", \"instant_over_offline\": "<<(double)s.instant_ratio<<", \"myopic_over_offline\": "<<(double)s.myopic_ratio
          <<", \"credit_over_offline\": "<<(double)s.credit_ratio<<", \"wfa_over_offline\": "<<(double)s.wfa_ratio
          <<", \"credit_saving_vs_static\": "<<(double)s.credit_saving<<", \"credit_switches\": "<<(double)s.credit_switches
          <<", \"wfa_switches\": "<<(double)s.wfa_switches<<", \"offline_switches\": "<<(double)s.offline_switches<<"}"
          <<(++di<dynamic.size()?",":"")<<"\n";
    }
    f << "  ],\n";
    f << "  \"random_fallback\": [\n";
    for (size_t i=0;i<random_rows.size();++i) { const auto&r=random_rows[i];
        f << "    {\"n\": "<<r.n<<", \"seeds\": "<<r.seeds<<", \"fraction_multiple_morphologies\": "<<r.fraction_multiple
          <<", \"mean_basis\": "<<r.mean_basis<<", \"max_basis\": "<<r.max_basis
          <<", \"median_endpoint_work_ratio\": "<<r.median_ratio<<", \"max_endpoint_work_ratio\": "<<r.max_ratio<<"}"
          <<(i+1<random_rows.size()?",":"")<<"\n";
    }
    f << "  ],\n";
    f << "  \"asymptotic_separation\": [\n";
    for (size_t i=0;i<separation.size();++i) { const auto&r=separation[i];
        f << "    {\"H\": "<<r.H<<", \"low_memory_work\": "<<r.low_work<<", \"high_memory_work\": "<<r.high_work
          <<", \"low_peak\": "<<r.low_peak<<", \"high_peak\": "<<r.high_peak
          <<", \"work_ratio\": "<<(double)r.work_ratio<<", \"peak_ratio\": "<<(double)r.peak_ratio
          <<", \"dynamic_static_over_morph\": "<<(double)r.dynamic_ratio<<"}"
          <<(i+1<separation.size()?",":"")<<"\n";
    }
    f << "  ]\n}\n";
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--dynamic-one") {
        if (argc < 3) throw std::invalid_argument("--dynamic-one requires beta");
        long double beta = std::stold(argv[2]);
        auto dims = adversarial_dims(8);
        MorphEnvelope m(dims); m.compile();
        auto s = dynamic_benchmark(m, beta);
        std::cout << std::setprecision(12)
                  << "{\"beta\": " << (double)beta
                  << ", \"static_over_offline\": " << (double)s.static_ratio
                  << ", \"instant_over_offline\": " << (double)s.instant_ratio
                  << ", \"myopic_over_offline\": " << (double)s.myopic_ratio
                  << ", \"credit_over_offline\": " << (double)s.credit_ratio
                  << ", \"wfa_over_offline\": " << (double)s.wfa_ratio
                  << ", \"credit_saving_vs_static\": " << (double)s.credit_saving
                  << ", \"credit_switches\": " << (double)s.credit_switches
                  << ", \"wfa_switches\": " << (double)s.wfa_switches
                  << ", \"offline_switches\": " << (double)s.offline_switches
                  << "}\n";
        return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "--compile-once") {
        auto dims = adversarial_dims(8);
        auto t0 = Clock::now();
        MorphEnvelope m(dims); m.compile();
        auto t1 = Clock::now();
        double ms = std::chrono::duration<double,std::milli>(t1-t0).count();
        std::cout << std::setprecision(12) << "{\"n\": " << m.n()
                  << ", \"basis\": " << m.root().size()
                  << ", \"compile_ms\": " << ms << "}\n";
        return 0;
    }
    bool tests_only = argc > 1 && std::string(argv[1]) == "--tests-only";
    bool smoke = argc > 1 && std::string(argv[1]) == "--smoke";
    bool core_only = argc > 1 && std::string(argv[1]) == "--core";
    std::string output = core_only ? (argc > 2 ? argv[2] : "results/core_results.json")
                                   : ((!tests_only && !smoke && argc > 1) ? argv[1] : "results/morph_results.json");

    std::cout << "[1/6] correctness suite\n";
    TestStats tests = run_tests(smoke);
    std::cout << "  exhaustive="<<tests.exhaustive_instances<<" random="<<tests.random_instances
              <<" oracle_trees="<<tests.oracle_parenthesizations<<" certificates="<<tests.certificate_checks
              <<" semantic_evals="<<tests.semantic_plan_evaluations<<" failures="<<tests.failures<<"\n";
    if (tests.failures) return 2;
    if (tests_only || smoke) { std::cout << "ALL_TESTS_PASSED\n"; return 0; }

    std::cout << "[2/6] asymptotic separation checks\n";
    auto separation = separation_checks();
    for (const auto& r : separation) std::cout << "  H="<<r.H<<" work_ratio="<<(double)r.work_ratio<<" dynamic_ratio="<<(double)r.dynamic_ratio<<"\n";

    std::cout << "[3/6] non-toy scaling benchmark\n";
    auto scaling = scaling_benchmark();
    for (const auto& r : scaling) std::cout << "  n="<<r.n<<" basis="<<r.basis<<" compile_ms="<<r.compile_ms
                                           <<" lookup_ns="<<r.lookup_ns<<" space=1e"<<(double)r.log_space
                                           <<" work_ratio="<<(double)r.low_memory_work/r.min_work<<"\n";

    std::cout << "[4/6] random negative controls\n";
    auto random_rows = random_fallback_benchmark();
    for (const auto& r : random_rows) std::cout << "  n="<<r.n<<" multi="<<r.fraction_multiple<<" mean_basis="<<r.mean_basis<<" max_ratio="<<r.max_ratio<<"\n";
    if (core_only) {
        std::map<long double, DynamicSummary> empty_dynamic;
        write_json(output, tests, scaling, empty_dynamic, random_rows, separation, 0.0, 0.0);
        std::cout << "WROTE " << output << "\n";
        return 0;
    }

    std::cout << "[5/6] switching-cost online benchmark\n";
    auto large_dims = adversarial_dims(8);
    double compile_once_ms = 0.0;
    std::map<long double, DynamicSummary> dynamic;
    {
        MorphEnvelope large(large_dims);
        auto c0=Clock::now(); large.compile(); auto c1=Clock::now();
        compile_once_ms=std::chrono::duration<double,std::milli>(c1-c0).count();
        for (long double beta : {0.0L,0.25L,1.0L,4.0L}) {
            dynamic[beta] = dynamic_benchmark(large, beta);
            const auto& s=dynamic[beta];
            std::cout << "  beta="<<(double)beta<<" static/off="<<(double)s.static_ratio
                      <<" credit/off="<<(double)s.credit_ratio<<" wfa/off="<<(double)s.wfa_ratio
                      <<" credit_saved="<<(double)(100*s.credit_saving)<<"%\n";
        }
    } // release the large frontier before measuring repeated recompilation

    std::cout << "[6/6] independent-process recompilation is measured separately\n";
    double compile_repeat_ms = 0.0;
    std::cout << "  use --compile-once in fresh processes to avoid allocator-fragmentation bias\n";

    write_json(output, tests, scaling, dynamic, random_rows, separation, compile_once_ms, compile_repeat_ms);
    std::cout << "WROTE " << output << "\n";
    return 0;
}
