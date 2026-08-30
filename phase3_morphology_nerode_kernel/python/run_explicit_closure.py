#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

from itertools import combinations

from morphn.automaton import behavior_quotient, encode_cost, finite_horizon_gap_curve
from morphn.compiler import compile_morph_n
from morphn.matrix_chain import build_real_automaton


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CERTIFICATES = ROOT / "certificates"


def automaton_json(automaton) -> dict:
    return {
        "tasks": automaton.tasks,
        "alpha": tuple(map(encode_cost, automaton.alpha)),
        "matrices": tuple(
            tuple(tuple(map(encode_cost, row)) for row in matrix)
            for matrix in automaton.matrices
        ),
        "state_names": automaton.state_names,
        "digest": automaton.digest(),
    }


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    CERTIFICATES.mkdir(exist_ok=True)
    dims = (1, 2, 4, 1, 6, 6)
    automaton, trees, static = build_real_automaton(dims)
    closure = automaton.closure()
    if not closure.closed or closure.unprocessed:
        raise AssertionError("explicit real residual closure did not terminate")
    certificate = automaton.closure_certificate(closure)
    quotient = behavior_quotient(closure, len(automaton.tasks))
    morph_n = compile_morph_n(automaton, static)

    (CERTIFICATES / "real_14_automaton.json").write_text(
        json.dumps(automaton_json(automaton), ensure_ascii=False, indent=2) + "\n"
    )
    (CERTIFICATES / "residual_closure_certificate.json").write_text(
        json.dumps(certificate, ensure_ascii=False, indent=2) + "\n"
    )

    infinite_ids = morph_n.executable_kernel
    subset_curves = {}
    for size in range(1, len(infinite_ids) + 1):
        for candidate in combinations(infinite_ids, size):
            try:
                restricted = automaton.subset(candidate)
            except ValueError:
                continue
            subset_curves[candidate] = finite_horizon_gap_curve(
                automaton, restricted, 20
            )

    rows = []
    previous: tuple[str, ...] | None = None
    for horizon in range(1, 21):
        exact_candidates = tuple(
            candidate
            for candidate, curve in subset_curves.items()
            if curve[horizon][0] == 0
        )
        if not exact_candidates:
            raise AssertionError(f"no Horizon-{horizon} exact subset of infinite kernel")
        chosen = min(exact_candidates, key=lambda item: (len(item), item))
        members = tuple(automaton.state_names[index] for index in chosen)
        added = tuple(sorted(set(members) - set(previous or ())))
        changed = previous is not None and members != previous
        rows.append(
            {
                "horizon": horizon,
                "kernel_size": len(members),
                "members": members,
                "added_bridges": added,
                "worst_gap": encode_cost(subset_curves[chosen][horizon][0]),
                "changed_from_previous": changed,
                "minimality_scope": "minimum cardinality among subsets of K_infinity",
            }
        )
        previous = members

    with (RESULTS / "kernel_growth.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "horizon",
                "kernel_size",
                "members",
                "added_bridges",
                "worst_gap",
                "changed_from_previous",
                "minimality_scope",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "members": "|".join(row["members"]), "added_bridges": "|".join(row["added_bridges"])})

    summary = {
        "schema": "morphn.explicit-closure.v1",
        "dims": dims,
        "H": len(trees),
        "K": len(morph_n.executable_kernel),
        "R": len(quotient.classes),
        "raw_residuals": len(closure.states),
        "ratios": {
            "R_over_H": len(quotient.classes) / len(trees),
            "K_over_H": len(morph_n.executable_kernel) / len(trees),
            "K_over_R": len(morph_n.executable_kernel) / len(quotient.classes),
        },
        "executable_kernel": tuple(
            automaton.state_names[index] for index in morph_n.executable_kernel
        ),
        "finite_column_difference_bound": automaton.finite_column_difference_bound(),
        "closure": {
            "queue_empty": True,
            "unprocessed": 0,
            "states": len(closure.states),
            "transitions": len(closure.transitions),
            "digest": certificate["closure_digest"],
        },
        "horizon_rows": rows,
        "infinite_kernel_equals_h20": set(rows[-1]["members"])
        == {
            automaton.state_names[index] for index in morph_n.executable_kernel
        },
    }
    (RESULTS / "explicit_closure_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({key: summary[key] for key in ("H", "K", "R", "raw_residuals", "infinite_kernel_equals_h20")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
