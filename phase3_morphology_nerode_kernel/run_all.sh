#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONPATH="python:."
mkdir -p build results certificates counterexamples

echo "=== phase2 regression tests ==="
(cd ../phase2_transition_complete_kernel && PYTHONPATH=. python3 scripts/run_tests.py)

echo "=== phase3 Python tests ==="
python3 python/run_tests.py

echo "=== C++ exact smoke programs ==="
for source in \
  explicit_morphology_automaton \
  residual_closure \
  morph_n_compiler \
  interval_pricing_oracle \
  persistent_matrix_runtime \
  dag_rematerialization_runtime; do
  c++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror \
    "src/$source.cpp" -o "build/$source"
  "build/$source"
done

echo "=== phase2 clean-worktree result audit ==="
python3 python/phase2_reaudit.py \
  ../phase2_transition_complete_kernel \
  results/phase2_reaudit_raw \
  results/phase2_reaudit.json

echo "=== explicit infinite residual closure ==="
python3 python/run_explicit_closure.py
python3 python/independent_residual_checker.py \
  certificates/real_14_automaton.json \
  certificates/residual_closure_certificate.json

echo "=== counterexamples ==="
python3 python/counterexample_search.py

echo "=== implicit pricing scale (n=16,32,64,128) ==="
python3 python/run_pricing_scale.py

echo "=== actual shared-DAG rematerialization ==="
python3 python/run_dag_experiment.py

echo "=== formal proof status ==="
bash formal/check_lean.sh

echo "=== independent certificate audit and summaries ==="
python3 python/generate_summary.py
python3 python/plot_results.py
python3 python/independent_certificate_checker.py

echo "MORPH-N full reproduction completed"
