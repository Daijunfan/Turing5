#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results

# 单独执行，避免大规模基准与重型检查器争抢内存和编译时间。
g++ -std=c++20 -O0 -g -fsanitize=undefined -fno-omit-frame-pointer \
  -Wall -Wextra -Wpedantic src/morph_envelope.cpp -o morph_envelope_ubsan
UBSAN_OPTIONS=halt_on_error=1 ./morph_envelope_ubsan --tests-only | tee results/ubsan_tests.log

g++ -std=c++20 -O0 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
  -Wall -Wextra -Wpedantic src/morph_envelope.cpp -o morph_envelope_asan
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  ./morph_envelope_asan --smoke | tee results/asan_smoke.log

echo "检查器测试完成"
