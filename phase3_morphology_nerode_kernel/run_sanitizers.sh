#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build/sanitizers

sources=(
  explicit_morphology_automaton
  residual_closure
  morph_n_compiler
  interval_pricing_oracle
  persistent_matrix_runtime
  dag_rematerialization_runtime
)

for sanitizer in undefined address; do
  for source in "${sources[@]}"; do
    output="build/sanitizers/${source}_${sanitizer}"
    c++ -std=c++20 -O1 -g -fno-omit-frame-pointer \
      -fsanitize="$sanitizer" -Wall -Wextra -Wpedantic -Werror \
      "src/$source.cpp" -o "$output"
    "$output"
  done
done

echo "sanitizer runs completed"
