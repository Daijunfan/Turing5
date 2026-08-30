#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from random import Random
from time import perf_counter

import numpy as np

from tcmk.kernel import ce_tcmk
from tcmk.local_space import local_rewrite_states
from tcmk.morphology import static_pareto
from tcmk.real_migration import exact_migration_totals


ROOT = Path(__file__).resolve().parents[1]


class PrecomputedModel:
    def __init__(self, source: np.ndarray, transition: np.ndarray):
        self.source = source
        self.transition = transition

    @property
    def state_count(self) -> int:
        return self.transition.shape[1]

    def service_cost(self, task: int, state: int) -> int:
        del task, state
        return 0

    def transition_cost(
        self, task: int, previous: int | None, state: int
    ) -> int:
        value = self.source[task, state] if previous is None else self.transition[task, previous, state]
        return int(value)


def edge_cost(totals) -> int:
    # Default real-model weights: work + (read + write + peak) / 8.
    return totals.total_work + (
        totals.read_bytes + totals.write_bytes + totals.peak_memory
    ) // 8


def make_model(dims: tuple[int, ...], states, updates: tuple[int, int]) -> PrecomputedModel:
    count = len(states)
    source = np.empty((2, count), dtype=np.int64)
    transition = np.empty((2, count, count), dtype=np.int64)
    for task, update in enumerate(updates):
        for target in range(count):
            source[task, target] = edge_cost(
                exact_migration_totals(dims, None, states[target].tree, update)
            )
        for previous in range(count):
            for target in range(count):
                transition[task, previous, target] = edge_cost(
                    exact_migration_totals(
                        dims, states[previous].tree, states[target].tree, update
                    )
                )
    return PrecomputedModel(source, transition)


def trace_batch(random: Random, checks: int) -> tuple[np.ndarray, Counter[str]]:
    patterns = (
        ("phase_switch", (0, 0, 0, 1, 1, 1)),
        ("hotspot", (0, 0, 0, 0, 0, 0)),
        ("periodic", (0, 1, 0, 1, 0, 1)),
        ("adversarial_switch", (0, 1, 1, 1, 1, 0)),
    )
    rows = []
    kinds: Counter[str] = Counter()
    for index in range(checks):
        if index < len(patterns):
            name, trace = patterns[index]
        else:
            name = "random_hotspot"
            trace = tuple(random.randrange(2) for _ in range(6))
        rows.append(trace)
        kinds[name] += 1
    return np.asarray(rows, dtype=np.int64), kinds


def batch_optimum(model: PrecomputedModel, traces: np.ndarray, allowed: tuple[int, ...]) -> np.ndarray:
    ids = np.asarray(allowed, dtype=np.int64)
    dp = model.source[traces[:, 0]][:, ids]
    for step in range(1, traces.shape[1]):
        matrices = model.transition[traces[:, step]][:, ids][:, :, ids]
        dp = np.min(dp[:, :, None] + matrices, axis=1)
    return np.min(dp, axis=1)


def dimensions(random: Random, matrix_count: int, index: int) -> tuple[int, ...]:
    mode = index % 4
    if mode == 0:
        return tuple(random.randrange(1, 9) for _ in range(matrix_count + 1))
    if mode == 1:
        return tuple(1 if item % 2 == 0 else 8 for item in range(matrix_count + 1))
    if mode == 2:
        cut = (matrix_count + 1) // 2
        return tuple(2 if item < cut else 7 for item in range(matrix_count + 1))
    return tuple(1 + ((item * 5 + index) % 8) for item in range(matrix_count + 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=int, default=10_000)
    parser.add_argument("--checks", type=int, default=100)
    args = parser.parse_args()
    master_seed = 0x5EED_2026_0830
    master = Random(master_seed)
    instance_seeds = [master.getrandbits(64) for _ in range(args.instances)]
    failures = 0
    kernels: Counter[str] = Counter()
    workload_counts: Counter[str] = Counter()
    started = perf_counter()

    for index, seed in enumerate(instance_seeds):
        random = Random(seed)
        matrix_count = 6 + index % 5
        dims = dimensions(random, matrix_count, index)
        states = local_rewrite_states(dims, limit=5, seed=seed)
        updates = (index % matrix_count, (index * 3 + 1) % matrix_count)
        model = make_model(dims, states, updates)
        static_ids = tuple(states.index(state) for state in static_pareto(states))
        certificate = ce_tcmk(model, (0, 1), 6, static_ids)
        if not certificate.final_separation.equivalent:
            raise AssertionError("CE-TCMK returned a non-equivalent candidate kernel")
        traces, kinds = trace_batch(random, args.checks)
        workload_counts.update(kinds)
        full = batch_optimum(model, traces, tuple(range(len(states))))
        kernel = batch_optimum(model, traces, certificate.kernel)
        failures += int(np.count_nonzero(full != kernel))
        kernels[str(len(certificate.kernel))] += 1
        if (index + 1) % 500 == 0:
            print(
                f"instances={index+1} checks={(index+1)*args.checks} "
                f"failures={failures} elapsed={perf_counter()-started:.1f}s",
                flush=True,
            )

    output = {
        "master_seed": master_seed,
        "instance_seeds": instance_seeds,
        "instances": args.instances,
        "matrix_count_range": [6, 10],
        "dimension_values": [1, 2, 3, 4, 5, 6, 7, 8],
        "candidate_states_per_instance": 5,
        "candidate_generation": "local associative rewrite graph",
        "transition_model": "exact fully-materialized real reuse accounting",
        "horizon": 6,
        "task_alphabet_size": 2,
        "equivalence_checks": args.instances * args.checks,
        "failures": failures,
        "kernel_size_histogram": dict(kernels),
        "workload_counts": dict(workload_counts),
        "wall_seconds": perf_counter() - started,
        "scope_warning": (
            "These n=6..10 checks are exact relative to each explicit five-state "
            "local-rewrite candidate graph. They are not full-Catalan oracle claims."
        ),
    }
    target = ROOT / "results" / "extended_validation.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key != "instance_seeds"}, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
