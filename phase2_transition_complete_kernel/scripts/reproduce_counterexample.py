#!/usr/bin/env python3
from __future__ import annotations

import json
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

from tcmk.dynamics import offline_optimum
from tcmk.morphology import canonical_morphologies, static_pareto
from tcmk.synthetic import BudgetTask, SplitDistanceModel


def decimal(value: Fraction) -> str:
    with localcontext() as context:
        context.prec = 50
        text = format(Decimal(value.numerator) / Decimal(value.denominator), "f")
    return text.rstrip("0").rstrip(".")


def main() -> None:
    dims = (16, 4, 1, 1, 12, 1, 3)
    budgets = (96, 64, 60, 64, 208, 96, 96, 240, 208, 220, 64, 60)
    states = canonical_morphologies(dims)
    kernel = static_pareto(states)
    model = SplitDistanceModel(states, beta=4)
    trace = tuple(BudgetTask(budget) for budget in budgets)
    full = offline_optimum(model, trace, legacy_free_initial=True)
    kernel_ids = tuple(states.index(state) for state in kernel)
    pruned = offline_optimum(model, trace, kernel_ids, legacy_free_initial=True)
    assert full.cost is not None and pruned.cost is not None
    gap = (pruned.cost - full.cost) / full.cost * 100
    output = {
        "independent_implementation": True,
        "legacy_free_initial_only_for_regression": True,
        "tree_count": len(states),
        "static_pareto": [
            {"name": state.name, "work": state.work, "peak": state.peak}
            for state in kernel
        ],
        "all_states_optimum": decimal(full.cost),
        "static_pareto_optimum": decimal(pruned.cost),
        "relative_gap_exact": f"{gap.numerator}/{gap.denominator}",
        "relative_gap_percent_exact_decimal": decimal(gap),
        # Reproduce the first-generation binary floating-point display verbatim;
        # all decisions above remain rational and do not use this value.
        "relative_gap_percent_legacy_display": (
            f"{(2497.6 - 2325.6) / 2325.6 * 100:.15f}"
        ),
        "all_states_path": [states[index].name for index in full.path],
    }
    target = Path(__file__).resolve().parents[1] / "results" / "counterexample.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
