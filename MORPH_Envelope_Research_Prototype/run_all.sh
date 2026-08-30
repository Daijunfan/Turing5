#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results

make
./morph_envelope --core results/core_results.json

for beta in 0 0.25 1 4; do
  ./morph_envelope --dynamic-one "$beta" > "results/dynamic_${beta}.json" &
done
wait

for i in 1 2 3; do
  ./morph_envelope --compile-once > "results/compile_once_${i}.json"
done

python3 tests/independent_runtime_check.py results/independent_runtime.json
python3 src/tensor_morphology.py results/tensor_results.json
python3 scripts/merge_results.py

echo "标准复现实验完成：results/morph_results_full.json"
echo "内存检查另运行：bash run_sanitizers.sh"
