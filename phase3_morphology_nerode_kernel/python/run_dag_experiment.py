#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from random import Random
from time import perf_counter_ns

from morphn.automaton import behavior_quotient
from morphn.compiler import compile_finite_kernel, compile_morph_n
from morphn.dag import (
    DagTask,
    SharedDag,
    build_dag_automaton,
    checkpoint_architectures,
    execute_architecture_path,
    execute_dtr,
)


ROOT = Path(__file__).resolve().parents[1]


def feasible_fixed(automaton, trace, state):
    try:
        restricted = automaton.subset((state,))
    except ValueError:
        return None
    try:
        return restricted.evaluate(trace)
    except ValueError:
        return None


def greedy_path(automaton, trace):
    current = 0
    path = []
    for task in trace:
        options = [
            (automaton.matrices[task][current][target], target)
            for target in range(automaton.state_count)
            if automaton.matrices[task][current][target] is not None
        ]
        if not options:
            raise ValueError("no feasible greedy checkpoint architecture")
        current = min(options)[1]
        path.append(current)
    return tuple(path)


def main() -> None:
    rows = []
    raw = []
    random = Random(1000512)
    for node_count in (128, 512, 1000):
        dag = SharedDag(node_count, width=32, seed=node_count)
        architectures = checkpoint_architectures(node_count)
        targets = (node_count - 1, node_count - 2, 3 * node_count // 4)
        budgets = (max(32, node_count // 2), node_count)
        alphabet = tuple(DagTask(target, budget) for target in targets for budget in budgets)
        trace = tuple(random.randrange(len(alphabet)) for _ in range(30))
        tasks = tuple(alphabet[index] for index in trace)
        automaton = build_dag_automaton(dag, alphabet, architectures)

        full_value, full_path = automaton.full_dp(trace)
        fixed_costs = tuple(
            (value, state)
            for state in range(automaton.state_count)
            if (value := feasible_fixed(automaton, trace, state)) is not None
        )
        fixed_state = min(fixed_costs)[1]
        static_pareto_ids = tuple(
            state
            for state, architecture in enumerate(architectures)
            if not any(
                len(other) <= len(architecture)
                and feasible_fixed(automaton, trace, other_state) is not None
                and feasible_fixed(automaton, trace, other_state)
                <= feasible_fixed(automaton, trace, state)
                and (len(other) < len(architecture) or other_state != state)
                for other_state, other in enumerate(architectures)
                if feasible_fixed(automaton, trace, state) is not None
            )
        ) or (fixed_state,)

        ce = compile_finite_kernel(automaton, 6, static_pareto_ids)
        ce_automaton = automaton.subset(ce.executable_kernel)
        _, ce_local_path = ce_automaton.full_dp(trace)
        ce_path = tuple(ce.executable_kernel[index] for index in ce_local_path)
        pareto_automaton = automaton.subset(static_pareto_ids)
        _, pareto_local_path = pareto_automaton.full_dp(trace)
        pareto_path = tuple(
            static_pareto_ids[index] for index in pareto_local_path
        )

        closure = automaton.closure(state_limit=10_000)
        morph_result = None
        morph_path = None
        if closure.closed:
            morph_result = compile_morph_n(automaton, static_pareto_ids, 10_000)
            morph_automaton = automaton.subset(morph_result.executable_kernel)
            _, local_path = morph_automaton.full_dp(trace)
            morph_path = tuple(morph_result.executable_kernel[index] for index in local_path)

        execution_only_path = []
        current = ce.executable_kernel[0]
        for task in trace:
            options = tuple(
                (automaton.matrices[task][current][target], target)
                for target in ce.executable_kernel
                if automaton.matrices[task][current][target] is not None
            )
            current = min(options)[1]
            execution_only_path.append(current)
        random_ids = tuple(sorted({0, len(architectures) // 2, len(architectures) - 2}))
        random_automaton = automaton.subset(random_ids)
        _, random_local_path = random_automaton.full_dp(trace)
        random_path = tuple(random_ids[index] for index in random_local_path)

        method_paths = {
            "retain_all": (len(architectures) - 1,) * len(trace),
            "fixed_checkpoint": (fixed_state,) * len(trace),
            "static_optimal": (fixed_state,) * len(trace),
            "static_pareto": pareto_path,
            "ce_tcmk_h6": ce_path,
            "full_state_oracle": full_path,
            "migration_greedy": greedy_path(automaton, trace),
            "execution_only_no_behavior": tuple(execution_only_path),
            "behavior_only_temp_search": full_path,
            "random_architecture_columns": random_path,
            "greedy_bridge_addition": ce_path,
        }
        if morph_path is not None:
            method_paths["morph_n"] = morph_path
        for method, path in method_paths.items():
            try:
                result = execute_architecture_path(dag, tasks, architectures, path)
                feasible = result["correct"] == len(tasks)
            except (KeyError, ValueError):
                result = {"wall_ns": 0, "operations": 0, "read_bytes": 0, "write_bytes": 0, "released": 0, "peak_slots": 0, "correct": 0, "tasks": len(tasks)}
                feasible = False
            rows.append(
                {
                    "nodes": node_count,
                    "method": method,
                    "feasible": feasible,
                    "within_budget": feasible,
                    **result,
                }
            )
        if morph_path is None:
            rows.append(
                {
                    "nodes": node_count,
                    "method": "morph_n",
                    "feasible": False,
                    "within_budget": False,
                    "wall_ns": 0,
                    "operations": 0,
                    "read_bytes": 0,
                    "write_bytes": 0,
                    "released": 0,
                    "peak_slots": 0,
                    "correct": 0,
                    "tasks": len(tasks),
                }
            )
        dtr = execute_dtr(dag, tasks)
        rows.append(
            {
                "nodes": node_count,
                "method": "dtr_greedy",
                "feasible": dtr["correct"] == len(tasks)
                and dtr["within_budget"],
                **dtr,
            }
        )

        raw.append(
            {
                "nodes": node_count,
                "operators": node_count - 2,
                "architectures": len(architectures),
                "task_alphabet": len(alphabet),
                "trace_length": len(trace),
                "full_value": str(full_value),
                "ce_kernel": len(ce.executable_kernel),
                "residual_closed": closure.closed,
                "raw_residuals": len(closure.states),
                "behavior_states": (
                    len(behavior_quotient(closure, len(alphabet)).classes)
                    if closure.closed
                    else None
                ),
                "morph_kernel": (
                    len(morph_result.executable_kernel) if morph_result else None
                ),
                "morph_supported": morph_result is not None,
            }
        )
        print(raw[-1], flush=True)

    with (ROOT / "results" / "dag_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=rows[0], lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    (ROOT / "results" / "dag_summary.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
