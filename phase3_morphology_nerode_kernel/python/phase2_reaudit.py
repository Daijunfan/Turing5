#!/usr/bin/env python3
from __future__ import annotations

import ast
from hashlib import sha256
import json
from pathlib import Path
import sys


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def find_horizon_call(script: Path) -> bool:
    tree = ast.parse(script.read_text())
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ce_tcmk"
        and len(node.args) >= 3
        and isinstance(node.args[2], ast.Constant)
        and node.args[2].value == 6
        for node in ast.walk(tree)
    )


def find_bounded_residual_loop(script: Path) -> bool:
    tree = ast.parse(script.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "exact_separation"
    )
    return any(
        isinstance(node, ast.For)
        and isinstance(node.iter, ast.Call)
        and isinstance(node.iter.func, ast.Name)
        and node.iter.func.id == "range"
        and any(
            isinstance(argument, ast.BinOp)
            and isinstance(argument.left, ast.Name)
            and argument.left.id == "horizon"
            for argument in node.iter.args
        )
        for node in ast.walk(function)
    )


def find_residual_sum(script: Path) -> bool:
    tree = ast.parse(script.read_text())
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "sum"
        and any(
            isinstance(item, ast.Attribute)
            and item.attr == "reachable_by_depth"
            for item in ast.walk(node)
        )
        for node in ast.walk(tree)
    )


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: phase2_reaudit.py PHASE2_SOURCE FRESH_RESULTS OUTPUT")
    phase2 = Path(sys.argv[1]).resolve()
    fresh = Path(sys.argv[2]).resolve()
    output = Path(sys.argv[3]).resolve()

    counterexample = load(fresh / "counterexample.json")
    core = load(fresh / "core_experiments.json")
    real_small = load(fresh / "real_small_exhaustive_n5.json")
    extended = load(fresh / "extended_validation.json")
    bridge = core["real_bridge"]
    comparison = bridge["pruning_comparison"]
    family = core["compression_family"]["rows"]
    speed = core["optimizer_speed"]
    continuous = core["continuous_execution"]

    checks = {
        "counterexample_full": counterexample["all_states_optimum"] == "2325.6",
        "counterexample_static": counterexample["static_pareto_optimum"] == "2497.6",
        "counterexample_gap_display": counterexample["relative_gap_percent_legacy_display"] == "7.395940832473341",
        "real_bridge_full": bridge["full_cost"] == "594",
        "real_bridge_static": bridge["static_cost"] == "738",
        "ce_kernel_size": comparison["ce_tcmk"]["size"] == 4,
        "ce_equals_minimum": bridge["ce_equals_minimum_kernel"],
        "hotspot_two_of_429": any(row["full_states"] == 429 and row["tcmk_states"] == 2 for row in family),
        "speedup_about_4_277": 4.0 <= speed["speedup_including_build"] <= 4.6,
        "real_instances": real_small["instances"] == 5440,
        "residual_count": real_small["reachable_residual_states_checked"] == 2_279_080,
        "budget_checks": real_small["budget_feasibility_checks"] == 38_585_476,
        "dynamic_equivalence_checks": extended["equivalence_checks"] == 1_000_000,
        "continuous_updates": continuous["updates"] == continuous["reference_matches"] == 1000,
        "all_reported_failures_zero": real_small["failures"] == extended["failures"] == 0,
    }

    real_script = phase2 / "scripts" / "run_real_small_exhaustive.py"
    residual_script = phase2 / "tcmk" / "residual.py"
    ast_evidence = {
        "ce_tcmk_called_with_literal_horizon_6": find_horizon_call(real_script),
        "reported_count_sums_reachable_by_depth": find_residual_sum(real_script),
        "separation_expands_range_bounded_by_horizon": find_bounded_residual_loop(residual_script),
    }
    if not all(checks.values()) or not all(ast_evidence.values()):
        failed = [name for name, passed in {**checks, **ast_evidence}.items() if not passed]
        raise AssertionError(f"phase2 reaudit mismatch: {failed}")

    result = {
        "schema": "morphn.phase2-reaudit.v1",
        "source_commit": "ad267d87dad608e7556563c84b006ff8eadafaf9",
        "clean_worktree_reproduction": True,
        "checks": checks,
        "observed": {
            "counterexample": {
                "full": counterexample["all_states_optimum"],
                "static_pareto": counterexample["static_pareto_optimum"],
                "legacy_gap_percent": counterexample["relative_gap_percent_legacy_display"],
                "exact_gap": counterexample["relative_gap_exact"],
            },
            "real_bridge": {
                "full": bridge["full_cost"],
                "static_pareto": bridge["static_cost"],
                "ce_kernel_size": comparison["ce_tcmk"]["size"],
                "minimum_kernel_size": comparison["minimum_kernel_oracle"]["size"],
            },
            "hotspot_minimum_ratio": "2/429",
            "speedup_including_build": speed["speedup_including_build"],
            "real_small_instances": real_small["instances"],
            "reachable_residual_states": real_small["reachable_residual_states_checked"],
            "budget_feasibility_checks": real_small["budget_feasibility_checks"],
            "dynamic_equivalence_checks": extended["equivalence_checks"],
            "continuous_real_updates": continuous["updates"],
        },
        "residual_count_classification": {
            "answer": "A",
            "meaning": "有限 Horizon=6 内逐层搜索并汇总得到的可达成对残差状态数",
            "is_fixed_point_closure": False,
            "stopped_early_for_resources": False,
            "prescribed_horizon": 6,
            "machine_evidence": ast_evidence,
            "reason": (
                "run_real_small_exhaustive.py calls ce_tcmk with literal horizon 6 and sums "
                "reachable_by_depth; residual.py iterates only through the supplied horizon. "
                "It never drains an unbounded successor work queue and emits no closure certificate."
            ),
        },
        "source_evidence": {
            str(real_script.relative_to(phase2)): digest(real_script),
            str(residual_script.relative_to(phase2)): digest(residual_script),
        },
        "fresh_result_digests": {
            path.name: digest(path) for path in sorted(fresh.glob("*.json"))
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"valid": True, "classification": "A", "checks": len(checks)}))


if __name__ == "__main__":
    main()
