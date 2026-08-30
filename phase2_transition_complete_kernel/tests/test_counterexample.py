from fractions import Fraction

from tcmk.dynamics import offline_optimum
from tcmk.morphology import canonical_morphologies, static_pareto
from tcmk.synthetic import BudgetTask, SplitDistanceModel


DIMS = (16, 4, 1, 1, 12, 1, 3)
BUDGETS = (96, 64, 60, 64, 208, 96, 96, 240, 208, 220, 64, 60)


def test_known_pcme_dynamic_counterexample_exactly():
    states = canonical_morphologies(DIMS)
    assert len(states) == 42
    assert [(state.name, state.work, state.peak) for state in static_pareto(states)] == [
        ("A(B((C(DE))F))", 220, 60),
        ("(A(B(C(DE))))F", 129, 64),
        ("(AB)((C(DE))F)", 128, 67),
    ]

    bridge = next(state for state in states if state.name == "A((B(C(DE)))F)")
    assert (bridge.work, bridge.peak) == (221, 60)

    model = SplitDistanceModel(states, beta=4)
    tasks = tuple(BudgetTask(budget) for budget in BUDGETS)
    all_result = offline_optimum(model, tasks, legacy_free_initial=True)
    pareto_ids = tuple(states.index(state) for state in static_pareto(states))
    pareto_result = offline_optimum(
        model, tasks, allowed=pareto_ids, legacy_free_initial=True
    )

    assert all_result.cost == Fraction(11628, 5)  # 2325.6
    assert pareto_result.cost == Fraction(12488, 5)  # 2497.6
    assert (
        (pareto_result.cost - all_result.cost) / all_result.cost * 100
        == Fraction(21500, 2907)
    )
    assert states.index(bridge) in all_result.path


def test_new_oracle_charges_source_construction():
    states = canonical_morphologies(DIMS)
    model = SplitDistanceModel(states, beta=4)
    tasks = tuple(BudgetTask(budget) for budget in BUDGETS)
    repaired = offline_optimum(model, tasks)
    legacy = offline_optimum(model, tasks, legacy_free_initial=True)
    assert repaired.cost is not None and legacy.cost is not None
    assert repaired.cost > legacy.cost

