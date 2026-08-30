#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

from independent_residual_checker import check_certificate
from morphn.automaton import CostAutomaton, decode_cost


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    errors = []
    automaton_data = json.loads((ROOT / "certificates/real_14_automaton.json").read_text())
    automaton = CostAutomaton(
        automaton_data["tasks"],
        tuple(decode_cost(value) for value in automaton_data["alpha"]),
        tuple(
            tuple(tuple(decode_cost(value) for value in row) for row in matrix)
            for matrix in automaton_data["matrices"]
        ),
        automaton_data["state_names"],
    )
    closure = json.loads(
        (ROOT / "certificates/residual_closure_certificate.json").read_text()
    )
    errors.extend(check_certificate(automaton, closure))

    reaudit = json.loads((ROOT / "results/phase2_reaudit.json").read_text())
    if not all(reaudit["checks"].values()):
        errors.append("phase2 reaudit contains a failed check")
    if reaudit["residual_count_classification"]["answer"] != "A":
        errors.append("phase2 residual-count classification changed")

    pricing = json.loads((ROOT / "counterexamples/pricing_oracle_search.json").read_text())
    if pricing["counterexample_found"]:
        errors.append("pricing oracle counterexample exists")

    dag_rows = list(csv.DictReader((ROOT / "results/dag_results.csv").open()))
    for row in dag_rows:
        if row["feasible"] == "True" and row["correct"] != row["tasks"]:
            errors.append(f"DAG correctness mismatch: {row['nodes']}/{row['method']}")

    result = {
        "valid": not errors,
        "errors": errors,
        "closure_states_checked": closure["residual_state_count"],
        "closure_transitions_checked": closure["transition_count"],
        "dag_rows_checked": len(dag_rows),
    }
    print(json.dumps(result, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
