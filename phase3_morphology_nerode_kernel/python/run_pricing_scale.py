#!/usr/bin/env python3
from __future__ import annotations

import csv
from math import lgamma, log
from pathlib import Path
from random import Random
import resource
from time import perf_counter

from morphn.matrix_chain import Tree
from morphn.pricing import PricingResourceLimit, price_transition


ROOT = Path(__file__).resolve().parents[1]


def balanced(i: int, j: int) -> Tree:
    if i == j:
        return Tree(i, j)
    split = (i + j) // 2
    return Tree(i, j, balanced(i, split), balanced(split + 1, j))


def log10_catalan(order: int) -> float:
    return (lgamma(2 * order + 1) - 2 * lgamma(order + 1) - log(order + 1)) / log(10)


def main() -> None:
    random = Random(128643216)
    rows = []
    for n in (16, 32, 64, 128):
        dims = tuple(random.randrange(1, 9) for _ in range(n + 1))
        columns = {balanced(0, n - 1)}
        calls = bridges = generated = retained = 0
        started = perf_counter()
        # Exact best responses from every generated predecessor for four hotspot tasks.
        completed = True
        failure = ""
        rounds = 2 if n <= 64 else 1
        try:
            # Exact best-response rounds. This is deliberately not called a
            # global dual-kernel fixed point.
            for _round in range(rounds):
                added = set()
                for source in tuple(columns):
                    for update in (0, n // 3, 2 * n // 3, n - 1):
                        result = price_transition(
                            dims,
                            source,
                            update,
                            max_generated_states=(5_000_000 if n >= 128 else None),
                        )
                        calls += 1
                        generated += result.dp_states_generated
                        retained += result.dp_states_retained
                        if result.tree not in columns:
                            added.add(result.tree)
                if not added:
                    break
                bridges += len(added)
                columns.update(added)
        except PricingResourceLimit as error:
            completed = False
            generated += error.generated
            retained += error.retained
            failure = str(error)
        elapsed = perf_counter() - started
        rows.append(
            {
                "matrix_count": n,
                "generated_columns": len(columns),
                "residual_states": "not_constructed_at_scale",
                "pricing_calls": calls,
                "kernel_seconds": elapsed,
                "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "bridges_added": bridges,
                "dp_states_generated": generated,
                "dp_states_retained": retained,
                "log10_catalan_architectures": log10_catalan(n - 1),
                "amortization_status": "not_measured",
                "completeness": "exact_fixed_predecessor_pricing_only",
                "requested_rounds": rounds,
                "completed": completed,
                "failure": failure,
            }
        )
        print(rows[-1], flush=True)
        target = ROOT / "results" / "scale_results.csv"
        with target.open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=rows[0], lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
