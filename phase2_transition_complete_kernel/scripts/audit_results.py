#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    return json.loads((ROOT / "results" / name).read_text())


def main() -> None:
    counterexample = load("counterexample.json")
    assert counterexample["tree_count"] == 42
    assert counterexample["all_states_optimum"] == "2325.6"
    assert counterexample["static_pareto_optimum"] == "2497.6"
    assert counterexample["relative_gap_percent_legacy_display"] == "7.395940832473341"
    assert "A((B(C(DE)))F)" in counterexample["all_states_path"]

    core = load("core_experiments.json")
    bridge = core["real_bridge"]
    assert bridge["state_count"] == 14
    assert bridge["full_cost"] == "594" and bridge["static_cost"] == "738"
    assert bridge["ce_equals_minimum_kernel"]
    pruning = bridge["pruning_comparison"]
    assert not pruning["static_pareto"]["equivalent"]
    assert pruning["ce_tcmk"]["equivalent"]
    assert pruning["minimum_kernel_oracle"]["equivalent"]
    assert pruning["ce_tcmk"]["size"] == pruning["minimum_kernel_oracle"]["size"] == 4
    execution = core["continuous_execution"]
    assert execution["updates"] == execution["versions"] == 1000
    assert execution["reference_matches"] == execution["certificates_checked"] == 1000
    family = core["compression_family"]
    assert family["meets_five_x_compression"] and len(family["rows"]) >= 5
    assert all(row["exact_equivalence"] for row in family["rows"])
    speed = core["optimizer_speed"]
    assert speed["meets_two_x"] and speed["speedup_including_build"] >= 2

    small = (load("small_exhaustive_n2_4.json"), load("small_exhaustive_n5_5.json"))
    assert sum(item["instances"] for item in small) == 5440
    assert sum(item["failures"] for item in small) == 0
    assert small[0]["minimum_kernel_instances"] == 1344
    real_small = load("real_small_exhaustive_n5.json")
    assert real_small["instances"] == 5440 and real_small["failures"] == 0
    assert real_small["reachable_residual_states_checked"] >= 2_000_000
    assert real_small["budget_feasibility_checks"] >= 38_000_000

    extended = load("extended_validation.json")
    assert extended["instances"] >= 10_000
    assert len(extended["instance_seeds"]) == extended["instances"]
    assert extended["equivalence_checks"] >= 1_000_000
    assert extended["failures"] == 0
    assert {"phase_switch", "hotspot", "periodic", "adversarial_switch", "random_hotspot"} <= set(
        extended["workload_counts"]
    )

    scale = load("scale_experiments.json")
    assert [row["matrix_count"] for row in scale["rows"]] == [16, 32, 64]
    assert all(row["candidate_space_equivalent_through_horizon_6"] for row in scale["rows"])
    assert scale["rows"][1]["transition_cost_status"] == "heuristic_upper_bound"
    assert scale["rows"][2]["transition_cost_status"] == "heuristic_upper_bound"

    theory = (ROOT / "THEORY_ZH.md").read_text()
    report = (ROOT / "REPORT_ZH.md").read_text()
    for marker in (
        "静态 Pareto 裁剪不保持动态最优",
        "CE-TCMK 的有限终止与正确性",
        "安全块替换条件",
        "Min-plus/tropical weighted automata",
    ):
        assert marker in theory
    for marker in ("必须报告的失败条件", "未声称", "上界"):
        assert marker in report
    print("completion audit: PASS")


if __name__ == "__main__":
    main()
