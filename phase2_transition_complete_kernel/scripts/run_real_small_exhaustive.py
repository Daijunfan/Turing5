#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import product
from multiprocessing import Pool
from pathlib import Path
from time import perf_counter

from tcmk.kernel import ce_tcmk
from tcmk.morphology import canonical_morphologies, static_pareto
from tcmk.real_migration import exact_migration_totals


ROOT = Path(__file__).resolve().parents[1]


class IntegerEdgeModel:
    def __init__(self, source, transition):
        self.source = source
        self.transition = transition

    @property
    def state_count(self):
        return len(self.source[0])

    def service_cost(self, task, state):
        del task, state
        return 0

    def transition_cost(self, task, previous, state):
        return (
            self.source[task][state]
            if previous is None
            else self.transition[task][previous][state]
        )


def scalar_cost(totals) -> int:
    return totals.total_work + (
        totals.read_bytes + totals.write_bytes + totals.peak_memory
    ) // 8


def evaluate(dims: tuple[int, ...]) -> dict:
    states = canonical_morphologies(dims)
    count = len(states)
    source = []
    transition = []
    peaks = set()
    for update in range(len(dims) - 1):
        source_row = []
        transition_matrix = []
        for target in range(count):
            totals = exact_migration_totals(dims, None, states[target].tree, update)
            peaks.add(totals.peak_memory)
            source_row.append(scalar_cost(totals))
        for previous in range(count):
            row = []
            for target in range(count):
                totals = exact_migration_totals(
                    dims, states[previous].tree, states[target].tree, update
                )
                peaks.add(totals.peak_memory)
                row.append(scalar_cost(totals))
            transition_matrix.append(tuple(row))
        source.append(tuple(source_row))
        transition.append(tuple(transition_matrix))
    model = IntegerEdgeModel(tuple(source), tuple(transition))
    static_ids = tuple(states.index(state) for state in static_pareto(states))
    certificate = ce_tcmk(
        model, tuple(range(len(dims) - 1)), 6, static_ids
    )
    edge_count = (len(dims) - 1) * (count + 1) * count
    return {
        "dims": dims,
        "states": count,
        "static": len(static_ids),
        "kernel": len(certificate.kernel),
        "iterations": len(certificate.iterations),
        "equivalent": certificate.final_separation.equivalent,
        "residuals": sum(certificate.final_separation.reachable_by_depth),
        "distinct_peak_budgets": len(peaks),
        "budget_feasibility_checks": len(peaks) * edge_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-n", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    jobs = (
        dims
        for matrix_count in range(2, args.max_n + 1)
        for dims in product(range(1, 5), repeat=matrix_count + 1)
    )
    started = perf_counter()
    failures = residuals = budget_checks = total = 0
    kernels: Counter[str] = Counter()
    first_nontrivial = []
    with Pool(args.workers) as pool:
        for row in pool.imap_unordered(evaluate, jobs, chunksize=2):
            total += 1
            failures += not row["equivalent"]
            residuals += row["residuals"]
            budget_checks += row["budget_feasibility_checks"]
            kernels[str(row["kernel"])] += 1
            if row["iterations"] and len(first_nontrivial) < 100:
                first_nontrivial.append(row)
            if total % 100 == 0:
                print(
                    f"instances={total} failures={failures} elapsed={perf_counter()-started:.1f}s",
                    flush=True,
                )
    output = {
        "domain": {
            "model": "exact fully-materialized real matrix reuse",
            "matrix_count": [2, args.max_n],
            "dimension_values": [1, 2, 3, 4],
            "all_catalan_trees": True,
            "dynamic_task_alphabet": "every update index at nonbinding memory budget",
            "dynamic_horizons": [1, 6],
            "budget_validation": "every distinct exact migration peak threshold",
            "initial_state": "charged virtual source",
        },
        "instances": total,
        "failures": failures,
        "reachable_residual_states_checked": residuals,
        "budget_feasibility_checks": budget_checks,
        "kernel_size_histogram": dict(kernels),
        "first_nontrivial_instances": first_nontrivial,
        "wall_seconds": perf_counter() - started,
        "scope_warning": (
            "All peak budgets are exhaustively checked as edge-feasibility thresholds. "
            "The Horizon-6 dynamic alphabet uses update indices with a nonbinding budget; "
            "it does not cross-product every budget threshold into every task word."
        ),
    }
    target = ROOT / "results" / f"real_small_exhaustive_n{args.max_n}.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
