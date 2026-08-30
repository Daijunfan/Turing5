#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
import csv
import gzip
import json
from pathlib import Path

from morphn.matrix_chain import build_real_automaton


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    explicit = json.loads((ROOT / "results/explicit_closure_summary.json").read_text())
    phase2 = json.loads((ROOT / "results/phase2_reaudit.json").read_text())
    dag = json.loads((ROOT / "results/dag_summary.json").read_text())

    automaton, _, _ = build_real_automaton((1, 2, 4, 1, 6, 6))
    closure = automaton.closure()
    depth_counts = Counter(map(len, closure.witnesses))
    with (ROOT / "results/residual_growth.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("depth", "new_residuals", "cumulative"),
            lineterminator="\n",
        )
        writer.writeheader()
        cumulative = 0
        for depth in sorted(depth_counts):
            cumulative += depth_counts[depth]
            writer.writerow({"depth": depth, "new_residuals": depth_counts[depth], "cumulative": cumulative})

    summary = {
        "stage": "MORPH-N: infinite-horizon dual-kernel morphology compilation",
        "final_judgment": "partially_supported",
        "supported": [
            "exact reduction to a min-plus weighted automaton",
            "infinite residual closure for the explicit 14-architecture nonbinding-budget matrix model",
            "independent closure certificate replay",
            "exact fixed-predecessor interval pricing through n=64",
            "all actual matrix and DAG outputs equal references",
        ],
        "partially_supported": [
            "dual-kernel compiler for finite explicit H with a finite closure",
            "implicit column generation: only fixed-predecessor pricing is globally exact",
        ],
        "falsified": [
            "phase2's 2,279,080 states were an infinite fixed-point closure",
            "finite normalized residual closure exists for every model",
            "infinite MORPH-N controller exists for the budgeted shared-DAG experiment",
            "MORPH-N achieves a 1.5x wall-clock gain in the second domain",
        ],
        "explicit": explicit,
        "phase2_reaudit": phase2["observed"],
        "dag": dag,
        "known_failures": [
            "n=128 interval pricing exceeded the configured DP state budget",
            "budgeted DAG residual closure exceeded 10,000 states",
            "Lean toolchain unavailable; no machine-proof claim",
            "full future-potential pricing oracle is not implemented",
        ],
    }
    (ROOT / "results/summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )

    dag_rows = list(csv.DictReader((ROOT / "results/dag_results.csv").open()))
    ablation = {
        "finite_horizon_kernel": {
            "H6_size": explicit["horizon_rows"][5]["kernel_size"],
            "scope": explicit["horizon_rows"][5]["minimality_scope"],
        },
        "infinite_residual_kernel": {
            "R": explicit["R"],
            "K": explicit["K"],
            "closure_certified": explicit["closure"]["unprocessed"] == 0,
        },
        "dag_methods": sorted({row["method"] for row in dag_rows}),
        "execution_only": "measured",
        "behavior_only_temp_search": "measured",
        "static_pareto": "measured",
        "ce_tcmk": "measured",
        "random_columns": "measured",
        "greedy_bridge": "measured",
        "full_state_oracle": "measured",
        "full_morph_n": "unsupported_on_budgeted_dag_due_to_non_closure",
    }
    (ROOT / "results/ablation_summary.json").write_text(
        json.dumps(ablation, ensure_ascii=False, indent=2) + "\n"
    )

    records = []
    for path in sorted((ROOT / "counterexamples").glob("*.json")):
        records.append({"kind": "counterexample", "path": str(path.relative_to(ROOT)), "data": json.loads(path.read_text())})
    records.append({"kind": "ablation", "data": ablation})
    records.append({"kind": "summary", "data": summary})
    with gzip.open(ROOT / "results/raw_runs.jsonl.gz", "wt", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"judgment": summary["final_judgment"], "raw_records": len(records)}))


if __name__ == "__main__":
    main()
