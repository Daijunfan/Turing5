#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent


def main() -> None:
    required = (
        "README.md",
        "REPORT_ZH.md",
        "THEORY_ZH.md",
        "LITERATURE_BOUNDARY_ZH.md",
        "results/phase2_reaudit.json",
        "results/summary.json",
        "results/raw_runs.jsonl.gz",
        "results/residual_growth.csv",
        "results/kernel_growth.csv",
        "results/scale_results.csv",
        "results/dag_results.csv",
        "results/final_run.log",
        "results/sanitizer_run.log",
        "certificates/residual_closure_certificate.json",
    )
    missing = [path for path in required if not (ROOT / path).exists()]
    if missing:
        raise AssertionError(f"missing release artifacts: {missing}")

    phase2 = json.loads((ROOT / "results/phase2_reaudit.json").read_text())
    closure = json.loads(
        (ROOT / "certificates/residual_closure_certificate.json").read_text()
    )
    summary = json.loads((ROOT / "results/summary.json").read_text())
    latest = json.loads((REPO / "LATEST_RESULTS.json").read_text())
    scale = list(csv.DictReader((ROOT / "results/scale_results.csv").open()))
    dag = list(csv.DictReader((ROOT / "results/dag_results.csv").open()))

    assert all(phase2["checks"].values())
    assert phase2["residual_count_classification"]["answer"] == "A"
    assert closure["queue_empty"] and closure["unprocessed_state_count"] == 0
    assert closure["transition_count"] == 1090
    assert summary["final_judgment"] == latest["final_judgment"] == "partially_supported"
    n64 = next(row for row in scale if row["matrix_count"] == "64")
    n128 = next(row for row in scale if row["matrix_count"] == "128")
    assert n64["completed"] == "True" and n128["completed"] == "False"
    assert all(
        row["correct"] == row["tasks"]
        for row in dag
        if row["feasible"] == "True"
    )
    assert "MORPH-N full reproduction completed" in (
        ROOT / "results/final_run.log"
    ).read_text()
    assert "sanitizer runs completed" in (
        ROOT / "results/sanitizer_run.log"
    ).read_text()

    forbidden = ("/" + "Users" + "/", "/" + "tmp" + "/")
    text_files = [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.suffix in {".py", ".cpp", ".hpp", ".md", ".json", ".sh", ".csv", ".log"}
        and ".toolchain" not in path.parts
    ]
    leaks = [
        str(path.relative_to(ROOT))
        for path in text_files
        if any(marker in path.read_text(errors="ignore") for marker in forbidden)
    ]
    if leaks:
        raise AssertionError(f"local absolute paths found: {leaks}")
    print(
        json.dumps(
            {
                "valid": True,
                "judgment": summary["final_judgment"],
                "required_artifacts": len(required),
                "dag_rows": len(dag),
            }
        )
    )


if __name__ == "__main__":
    main()
