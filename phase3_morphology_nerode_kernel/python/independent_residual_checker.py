#!/usr/bin/env python3
from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import sys

from morphn.automaton import CostAutomaton, decode_cost, encode_cost, stable_hash


def check_certificate(automaton: CostAutomaton, certificate: dict) -> list[str]:
    errors = []
    states = tuple(tuple(decode_cost(value) for value in state) for state in certificate["states"])
    if certificate["automaton_digest"] != automaton.digest():
        errors.append("automaton digest mismatch")
    if certificate["unprocessed_state_count"] != 0 or not certificate["queue_empty"]:
        errors.append("closure queue is not empty")
    expected_edges = len(states) * len(automaton.tasks)
    if certificate["transition_count"] != expected_edges:
        errors.append("transition count is incomplete")
    edge_map = {
        (edge["source"], edge["task"]): edge for edge in certificate["transitions"]
    }
    state_index = {state: index for index, state in enumerate(states)}
    for source, residual in enumerate(states):
        for task in range(len(automaton.tasks)):
            edge = edge_map.get((source, task))
            if edge is None:
                errors.append(f"missing edge {source}/{task}")
                continue
            increment, target = automaton.step(residual, task)
            if state_index.get(target) != edge["target"]:
                errors.append(f"target mismatch {source}/{task}")
            if encode_cost(increment) != edge["increment"]:
                errors.append(f"increment mismatch {source}/{task}")
    state_hashes = [stable_hash(state) for state in certificate["states"]]
    transition_hashes = [stable_hash(edge) for edge in certificate["transitions"]]
    if stable_hash(state_hashes) != certificate["state_merkle_root"]:
        errors.append("state Merkle root mismatch")
    if stable_hash(transition_hashes) != certificate["transition_merkle_root"]:
        errors.append("transition Merkle root mismatch")
    return errors


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: independent_residual_checker.py AUTOMATON.json CERTIFICATE.json")
    automaton_data = json.loads(Path(sys.argv[1]).read_text())
    automaton = CostAutomaton(
        automaton_data["tasks"],
        tuple(decode_cost(value) for value in automaton_data["alpha"]),
        tuple(
            tuple(tuple(decode_cost(value) for value in row) for row in matrix)
            for matrix in automaton_data["matrices"]
        ),
        automaton_data["state_names"],
    )
    certificate = json.loads(Path(sys.argv[2]).read_text())
    errors = check_certificate(automaton, certificate)
    print(json.dumps({"valid": not errors, "errors": errors}, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
