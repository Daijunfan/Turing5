#!/usr/bin/env python3
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from random import Random
from time import perf_counter

from tcmk.certificate_checker import check_migration_certificate
from tcmk.dynamics import offline_optimum
from tcmk.kernel import ce_tcmk, minimum_kernel_oracle
from tcmk.matrix_runtime import PersistentExecutor, random_matrix
from tcmk.morphology import canonical_morphologies, static_pareto
from tcmk.policies import all_baselines
from tcmk.pruning import block_context_kernel, service_anchor_pareto
from tcmk.real_migration import PersistentMatrixModel, RealTask, plan_migration
from tcmk.residual import exact_separation


ROOT = Path(__file__).resolve().parents[1]
REAL_DIMS = (1, 2, 4, 1, 6, 6)
UNLIMITED = 10**12


def cost(value: Fraction | None) -> str | None:
    if value is None:
        return None
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def real_bridge_experiment() -> dict:
    states = canonical_morphologies(REAL_DIMS)
    model = PersistentMatrixModel(REAL_DIMS, states)
    static_ids = tuple(states.index(state) for state in static_pareto(states))
    trace = tuple(RealTask(index, UNLIMITED) for index in (0, 4, 4, 2, 4, 2))
    full = offline_optimum(model, trace)
    static = offline_optimum(model, trace, static_ids)
    alphabet = tuple(RealTask(index, UNLIMITED) for index in range(5))
    ce = ce_tcmk(model, alphabet, 6, static_ids)
    minimum = minimum_kernel_oracle(model, alphabet, 6)

    methods = {
        "static_pareto": static_ids,
        "service_anchor_pareto": service_anchor_pareto(model, alphabet, static_ids),
        "safe_block_context": block_context_kernel(model, alphabet),
        "ce_tcmk": ce.kernel,
        "minimum_kernel_oracle": minimum.kernel,
    }
    comparison = {}
    for name, kernel in methods.items():
        separated = exact_separation(model, alphabet, 6, kernel)
        comparison[name] = {
            "size": len(kernel),
            "states": [states[index].name for index in kernel],
            "maximum_gap": cost(separated.gap),
            "equivalent": separated.equivalent,
            "reachable_residuals": separated.reachable_by_depth,
        }

    baselines = all_baselines(model, trace)
    return {
        "dims": REAL_DIMS,
        "state_count": len(states),
        "trace_updates": [task.update_index for task in trace],
        "full_cost": cost(full.cost),
        "static_cost": cost(static.cost),
        "full_path": [states[index].name for index in full.path],
        "static_path": [states[index].name for index in static.path],
        "static_states": [states[index].name for index in static_ids],
        "ce_iterations": [
            {
                "gap": cost(iteration.separation.gap),
                "trace": [task.update_index for task in iteration.separation.trace],
                "missing": [states[index].name for index in iteration.missing_path_states],
                "bridge_values": {
                    states[item.state].name: cost(item.value)
                    for item in iteration.bridge_values
                },
                "added": states[iteration.added_state].name,
            }
            for iteration in ce.iterations
        ],
        "ce_equals_minimum_kernel": ce.kernel == minimum.kernel,
        "pruning_comparison": comparison,
        "baselines_against_full_oracle": {
            name: None
            if result is None
            else {
                "cost": cost(result.cost),
                "ratio": None
                if result.cost is None or full.cost is None
                else float(result.cost / full.cost),
                "switches": result.switches,
            }
            for name, result in baselines.items()
        },
        "wfa_theoretical_status": "not_applicable_task_dependent_nonmetric_transition",
    }


def continuous_execution_experiment(steps: int = 1000) -> dict:
    dims = (2, 3, 2, 4, 2, 3)
    states = canonical_morphologies(dims)
    seed = 20260830
    random = Random(seed)
    leaves = [
        random_matrix(dims[index], dims[index + 1], random)
        for index in range(len(dims) - 1)
    ]
    executor = PersistentExecutor(dims, leaves, states[0].tree)
    checked = exact = total_work = read_bytes = write_bytes = peak = 0
    for _ in range(steps):
        update = random.randrange(len(leaves))
        target = states[random.randrange(len(states))].tree
        certificate = plan_migration(dims, executor.tree, target, update)
        result = check_migration_certificate(dims, executor.tree, target, certificate)
        if not result.valid:
            raise AssertionError(result.error)
        checked += 1
        exact += certificate.exact
        total_work += certificate.total_work
        read_bytes += certificate.read_bytes
        write_bytes += certificate.write_bytes
        peak = max(peak, certificate.peak_memory)
        executor.apply_update(
            update,
            random_matrix(dims[update], dims[update + 1], random),
            target,
            certificate,
        )
    return {
        "seed": seed,
        "updates": steps,
        "versions": executor.version,
        "reference_matches": steps,
        "certificates_checked": checked,
        "exact_plans": exact,
        "total_work": total_work,
        "read_bytes": read_bytes,
        "write_bytes": write_bytes,
        "maximum_peak_memory": peak,
    }


def compression_family() -> dict:
    random = Random(8675309)
    rows = []
    for matrix_count in range(4, 9):
        dims = tuple(random.randrange(1, 7) for _ in range(matrix_count + 1))
        states = canonical_morphologies(dims)
        static_ids = tuple(states.index(state) for state in static_pareto(states))
        alphabet = (RealTask(matrix_count // 2, UNLIMITED),)
        model = PersistentMatrixModel(dims, states)
        started = perf_counter()
        certificate = ce_tcmk(model, alphabet, 6, static_ids)
        elapsed = perf_counter() - started
        rows.append(
            {
                "matrix_count": matrix_count,
                "dims": dims,
                "full_states": len(states),
                "static_states": len(static_ids),
                "tcmk_states": len(certificate.kernel),
                "ratio": len(certificate.kernel) / len(states),
                "build_seconds": elapsed,
                "exact_equivalence": certificate.final_separation.equivalent,
            }
        )
    return {
        "seed": 8675309,
        "workload": "single persistent hotspot, all words through horizon 6",
        "rows": rows,
        "meets_five_x_compression": any(row["ratio"] <= 0.2 for row in rows),
    }


def optimizer_speed_experiment() -> dict:
    dims = (4, 1, 2, 3, 5, 6, 3, 2)
    states = canonical_morphologies(dims)
    alphabet = (RealTask(3, UNLIMITED),)
    query = alphabet * 50
    repetitions = 5

    full_model = PersistentMatrixModel(dims, states)
    started = perf_counter()
    for _ in range(repetitions):
        offline_optimum(full_model, query)
    full_seconds = perf_counter() - started

    kernel_model = PersistentMatrixModel(dims, states)
    static_ids = tuple(states.index(state) for state in static_pareto(states))
    started = perf_counter()
    certificate = ce_tcmk(kernel_model, alphabet, 6, static_ids)
    build_seconds = perf_counter() - started
    for _ in range(repetitions):
        offline_optimum(kernel_model, query, certificate.kernel)
    kernel_total_seconds = perf_counter() - started
    return {
        "dims": dims,
        "full_states": len(states),
        "kernel_states": len(certificate.kernel),
        "queries": repetitions,
        "query_horizon": len(query),
        "full_seconds": full_seconds,
        "kernel_build_seconds": build_seconds,
        "kernel_total_seconds_including_build": kernel_total_seconds,
        "speedup_including_build": full_seconds / kernel_total_seconds,
        "meets_two_x": full_seconds >= 2 * kernel_total_seconds,
    }


def main() -> None:
    started = perf_counter()
    output = {
        "real_bridge": real_bridge_experiment(),
        "continuous_execution": continuous_execution_experiment(),
        "compression_family": compression_family(),
        "optimizer_speed": optimizer_speed_experiment(),
    }
    output["wall_seconds"] = perf_counter() - started
    target = ROOT / "results" / "core_experiments.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
