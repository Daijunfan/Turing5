from random import Random

from morphn.matrix_chain import all_trees, migration_totals
from morphn.pricing import price_transition


def test_interval_pricing_matches_full_catalan_through_nine():
    random = Random(20260831)
    for n in range(2, 10):
        for _ in range(3):
            dims = tuple(random.randrange(1, 6) for _ in range(n + 1))
            trees = all_trees(n)
            source = trees[random.randrange(len(trees))]
            update = random.randrange(n)
            priced = price_transition(dims, source, update)
            brute = min(
                migration_totals(dims, source, target, update).scalar_cost
                for target in trees
            )
            assert priced.total_cost == brute
            assert (
                migration_totals(dims, source, priced.tree, update).scalar_cost
                == brute
            )


def test_pricing_memory_budget_is_exact():
    dims = (2, 3, 4, 2, 5, 3)
    source = all_trees(5)[0]
    unconstrained = price_transition(dims, source, 2)
    budget = unconstrained.peak_memory_bytes
    constrained = price_transition(dims, source, 2, budget)
    trees = all_trees(5)
    brute = min(
        totals.scalar_cost
        for target in trees
        if (totals := migration_totals(dims, source, target, 2)).peak_bytes <= budget
    )
    assert constrained.total_cost == brute
    assert (
        migration_totals(dims, source, constrained.tree, 2).scalar_cost == brute
    )
