#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from random import Random
from time import perf_counter

from tcmk.kernel import ce_tcmk
from tcmk.local_space import local_rewrite_states
from tcmk.morphology import static_pareto
from tcmk.real_migration import PersistentMatrixModel, RealTask


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    random = Random(640032016)
    rows = []
    for matrix_count in (16, 32, 64):
        dims = tuple(random.randrange(1, 9) for _ in range(matrix_count + 1))
        started = perf_counter()
        states = local_rewrite_states(dims, limit=128, seed=matrix_count)
        enumeration_seconds = perf_counter() - started
        static = static_pareto(states)
        static_ids = tuple(states.index(state) for state in static)
        model = PersistentMatrixModel(dims, states, max_exact_nodes=20)
        alphabet = (RealTask(matrix_count // 2, 10**15),)
        started = perf_counter()
        certificate = ce_tcmk(model, alphabet, 6, static_ids)
        kernel_seconds = perf_counter() - started
        exact_edges = model.cached_plans_are_exact()
        rows.append(
            {
                "matrix_count": matrix_count,
                "candidate_states": len(states),
                "candidate_generation": "bounded_local_associative_rewrite_graph",
                "static_states": len(static_ids),
                "candidate_kernel_states": len(certificate.kernel),
                "candidate_space_equivalent_through_horizon_6": (
                    certificate.final_separation.equivalent
                ),
                "transition_cost_status": (
                    "exact" if exact_edges else "heuristic_upper_bound"
                ),
                "enumeration_seconds": enumeration_seconds,
                "kernel_seconds": kernel_seconds,
                "seed": 640032016,
                "dims": dims,
            }
        )
    output = {
        "scope_warning": (
            "Large-scale rows are relative to a bounded local-rewrite candidate graph, "
            "not the full Catalan architecture universe. Heuristic migration rows are "
            "upper bounds and are not exact-oracle evidence."
        ),
        "rows": rows,
    }
    target = ROOT / "results" / "scale_experiments.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
