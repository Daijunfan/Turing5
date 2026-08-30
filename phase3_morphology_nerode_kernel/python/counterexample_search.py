#!/usr/bin/env python3
from __future__ import annotations

from itertools import product
import json
from pathlib import Path
from random import Random

from morphn.automaton import CostAutomaton, compare_automata, finite_horizon_gap
from morphn.compiler import compile_morph_n
from morphn.matrix_chain import (
    all_trees,
    build_real_automaton,
    migration_totals,
    tree_work_peak,
)
from morphn.pricing import price_transition


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "counterexamples"


def write(name: str, value: dict) -> None:
    TARGET.mkdir(exist_ok=True)
    (TARGET / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def finite_horizon_failure() -> dict:
    automaton, _, static = build_real_automaton((1, 2, 4, 1, 6, 6))
    k1 = automaton.subset(static)
    gap, witness = finite_horizon_gap(automaton, k1, 2)
    return {
        "type": "K_H_fails_at_H_plus_1",
        "H": 1,
        "failed_horizon": 2,
        "kernel": tuple(automaton.state_names[index] for index in static),
        "witness": tuple(automaton.tasks[index] for index in witness),
        "gap": str(gap),
    }


def equal_static_different_future() -> dict:
    dims = (1, 1, 1, 1)
    automaton, trees, _ = build_real_automaton(dims)
    left, right = 0, 1
    assert tree_work_peak(trees[left], dims) == tree_work_peak(trees[right], dims)
    initial_left = CostAutomaton(
        automaton.tasks,
        tuple(0 if state == left else None for state in range(len(trees))),
        automaton.matrices,
        automaton.state_names,
    )
    initial_right = CostAutomaton(
        automaton.tasks,
        tuple(0 if state == right else None for state in range(len(trees))),
        automaton.matrices,
        automaton.state_names,
    )
    comparison = compare_automata(initial_left, initial_right)
    assert not comparison.equivalent
    return {
        "type": "same_static_metrics_different_future",
        "dims": dims,
        "left": trees[left].render(),
        "right": trees[right].render(),
        "work_peak": tree_work_peak(trees[left], dims),
        "distinguishing_suffix": tuple(
            automaton.tasks[index] for index in comparison.witness
        ),
        "left_cost": str(initial_left.evaluate(comparison.witness)),
        "right_cost": str(initial_right.evaluate(comparison.witness)),
    }


def same_residual_multiple_executions() -> dict:
    automaton, _, _ = build_real_automaton((1, 2, 4, 1, 6, 6))
    closure = automaton.closure()
    for state, residual in enumerate(closure.states):
        optimal = tuple(
            automaton.state_names[index]
            for index, value in enumerate(residual)
            if value == 0
        )
        if len(optimal) > 1:
            return {
                "type": "same_behavior_residual_multiple_execution_architectures",
                "history": tuple(
                    automaton.tasks[index] for index in closure.witnesses[state]
                ),
                "residual": tuple("inf" if value is None else str(value) for value in residual),
                "optimal_execution_architectures": optimal,
            }
    raise AssertionError("expected a tied residual")


def small_behavior_large_execution() -> dict:
    count = 6
    tasks = tuple(f"task_{index}" for index in range(count))
    matrices = tuple(
        tuple(
            tuple(0 if target == task else 1 for target in range(count))
            for _source in range(count)
        )
        for task in range(count)
    )
    automaton = CostAutomaton(tasks, (0,) * count, matrices)
    result = compile_morph_n(automaton, (0,))
    return {
        "type": "behavior_kernel_small_execution_kernel_large",
        "H": count,
        "K": len(result.executable_kernel),
        "R": result.behavior_kernel_size,
        "executable_kernel": result.executable_kernel,
    }


def execution_small_behavior_large() -> dict:
    automaton, _, static = build_real_automaton((1, 2, 3, 1))
    result = compile_morph_n(automaton, static)
    return {
        "type": "execution_kernel_small_behavior_kernel_large",
        "H": automaton.state_count,
        "K": len(result.executable_kernel),
        "R": result.behavior_kernel_size,
    }


def growing_closure() -> dict:
    automaton = CostAutomaton(
        ("tick",),
        (0, 0),
        (((0, None), (None, 1)),),
    )
    partial = automaton.closure(state_limit=50)
    return {
        "type": "residual_closure_continues_growing",
        "closed": partial.closed,
        "states_at_limit": len(partial.states),
        "shortest_new_residual_word_length": len(partial.witnesses[-1]),
        "last_residual": tuple(
            "inf" if value is None else str(value) for value in partial.states[-1]
        ),
        "cause": "infinite edges prevent the finite-column-difference bound",
    }


def pricing_crosscheck() -> dict:
    random = Random(20260831)
    checks = 0
    for n in range(2, 10):
        for _ in range(20):
            dims = tuple(random.randrange(1, 6) for _ in range(n + 1))
            trees = all_trees(n)
            source = trees[random.randrange(len(trees))]
            update = random.randrange(n)
            priced = price_transition(dims, source, update)
            brute = min(
                migration_totals(dims, source, target, update).scalar_cost
                for target in trees
            )
            checks += 1
            if priced.total_cost != brute:
                return {
                    "type": "pricing_oracle_error",
                    "dims": dims,
                    "source": source.render(),
                    "update": update,
                    "priced": priced.total_cost,
                    "brute": brute,
                    "check_index": checks,
                }
    return {
        "type": "pricing_oracle_error_search",
        "counterexample_found": False,
        "checks": checks,
        "n_range": [2, 9],
        "seed": 20260831,
    }


def main() -> None:
    write("finite_horizon_failure.json", finite_horizon_failure())
    write("same_static_different_future.json", equal_static_different_future())
    write("same_residual_multiple_architectures.json", same_residual_multiple_executions())
    write("small_R_large_K.json", small_behavior_large_execution())
    write("small_K_large_R.json", execution_small_behavior_large())
    write("growing_residual_closure.json", growing_closure())
    write("pricing_oracle_search.json", pricing_crosscheck())
    print(f"wrote {len(tuple(TARGET.glob('*.json')))} counterexample/search files")


if __name__ == "__main__":
    main()
