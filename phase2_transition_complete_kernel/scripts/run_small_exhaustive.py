#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import product
from multiprocessing import Pool
from pathlib import Path
from time import perf_counter

from tcmk.kernel import ce_tcmk, minimum_kernel_oracle
from tcmk.morphology import canonical_morphologies, static_pareto
from tcmk.synthetic import BudgetTask, SplitDistanceModel


ROOT = Path(__file__).resolve().parents[1]


def evaluate_instance(arguments: tuple[tuple[int, ...], int, bool]) -> dict:
    dims, horizon, check_minimum = arguments
    states = canonical_morphologies(dims)
    static_ids = tuple(states.index(state) for state in static_pareto(states))
    model = SplitDistanceModel(states, beta=4)
    alphabet = tuple(
        BudgetTask(peak) for peak in sorted({state.peak for state in states})
    )
    certificate = ce_tcmk(model, alphabet, horizon, static_ids)
    minimum_size = None
    if check_minimum:
        minimum_size = len(minimum_kernel_oracle(model, alphabet, horizon).kernel)
    return {
        "dims": dims,
        "states": len(states),
        "static": len(static_ids),
        "kernel": len(certificate.kernel),
        "iterations": len(certificate.iterations),
        "equivalent": certificate.final_separation.equivalent,
        "residuals": sum(certificate.final_separation.reachable_by_depth),
        "minimum_size": minimum_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-n", type=int, default=5)
    parser.add_argument("--min-n", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    started = perf_counter()
    total = failures = residuals = 0
    state_histogram: Counter[str] = Counter()
    kernel_histogram: Counter[str] = Counter()
    minimum_checks = 0
    nontrivial = []

    jobs = (
        (dims, args.horizon, matrix_count <= 4)
        for matrix_count in range(args.min_n, args.max_n + 1)
        for dims in product(range(1, 5), repeat=matrix_count + 1)
    )
    pool = Pool(args.workers) if args.workers > 1 else None
    results = pool.imap_unordered(evaluate_instance, jobs, chunksize=4) if pool else map(evaluate_instance, jobs)
    try:
        for row in results:
            total += 1
            residuals += row["residuals"]
            state_histogram[str(row["states"])] += 1
            kernel_histogram[str(row["kernel"])] += 1
            if not row["equivalent"]:
                failures += 1
            if row["iterations"] and len(nontrivial) < 100:
                nontrivial.append(row)
            if row["minimum_size"] is not None:
                minimum_checks += 1
                if row["kernel"] < row["minimum_size"]:
                    raise AssertionError("CE kernel cannot beat the cardinality oracle")
            if total % 100 == 0:
                print(
                    f"checked={total} failures={failures} elapsed={perf_counter()-started:.1f}s",
                    flush=True,
                )
    finally:
        if pool:
            pool.close()
            pool.join()

    output = {
        "domain": {
            "model": "legacy split-distance regression model",
            "matrix_count": [args.min_n, args.max_n],
            "dimension_values": [1, 2, 3, 4],
            "all_catalan_trees": True,
            "all_distinct_tree_peak_budgets": True,
            "horizons": [1, args.horizon],
            "arithmetic": "fractions.Fraction",
            "initial_state": "explicit empty-split source with charged construction",
        },
        "instances": total,
        "failures": failures,
        "dynamic_equivalence_rate": 1.0 if not total else (total - failures) / total,
        "reachable_residual_states_checked": residuals,
        "minimum_kernel_instances": minimum_checks,
        "full_state_histogram": dict(state_histogram),
        "kernel_size_histogram": dict(kernel_histogram),
        "first_nontrivial_instances": nontrivial,
        "wall_seconds": perf_counter() - started,
        "scope_warning": (
            "This exhaustive grid validates the exact TCMK machinery but uses the "
            "legacy split-distance model; final empirical claims use real_migration.py."
        ),
    }
    target = ROOT / "results" / f"small_exhaustive_n{args.min_n}_{args.max_n}.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
