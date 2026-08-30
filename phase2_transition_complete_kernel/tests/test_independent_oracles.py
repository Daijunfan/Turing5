from fractions import Fraction
from itertools import product
from random import Random

from tcmk.counterexamples import TableModel, TableTask
from tcmk.dynamics import offline_optimum
from tcmk.fast_table import fast_separation
from tcmk.morphology import canonical_morphologies, ordered_morphologies
from tcmk.residual import exact_separation


def test_paired_residual_search_against_explicit_word_enumeration():
    random = Random(9173)
    tasks = (TableTask(0), TableTask(1))
    for _ in range(50):
        service = tuple(tuple(random.randrange(6) for _ in range(3)) for _ in tasks)
        transition = tuple(
            tuple(
                tuple(0 if i == j else random.randrange(6) for j in range(3))
                for i in range(3)
            )
            for _ in tasks
        )
        model = TableModel(service, transition)
        kernel = tuple(sorted(random.sample(range(3), random.randrange(1, 4))))
        result = exact_separation(model, tasks, 4, kernel)
        maximum = Fraction(0)
        infeasible = False
        for length in range(1, 5):
            for trace in product(tasks, repeat=length):
                full = offline_optimum(model, trace)
                pruned = offline_optimum(model, trace, kernel)
                if full.cost is None:
                    continue
                if pruned.cost is None:
                    infeasible = True
                    continue
                maximum = max(maximum, pruned.cost - full.cost)
        assert result.kernel_infeasible == infeasible
        if not infeasible:
            assert result.gap == maximum


def test_evaluation_orders_are_explicit_and_legacy_merge_is_safe():
    dims = (2, 5, 3, 4, 2)
    ordered = tuple(ordered_morphologies(dims))
    canonical = canonical_morphologies(dims)
    # Catalan(3)=5 structures, with 2^3 recursive evaluation orders each.
    assert len(ordered) == 40
    assert len(canonical) == 5
    for state in canonical:
        same_tree = [candidate for candidate in ordered if candidate.tree == state.tree]
        assert state.peak == min(candidate.peak for candidate in same_tree)
        assert {candidate.work for candidate in same_tree} == {state.work}
        assert {candidate.splits for candidate in same_tree} == {state.splits}


def test_fast_integer_residual_search_matches_generic_oracle():
    import numpy as np

    random = Random(811)
    tasks = (TableTask(0), TableTask(1))
    for _ in range(20):
        service = tuple(tuple(random.randrange(6) for _ in range(3)) for _ in tasks)
        transition = tuple(
            tuple(
                tuple(0 if i == j else random.randrange(6) for j in range(3))
                for i in range(3)
            )
            for _ in tasks
        )
        model = TableModel(service, transition)
        source = np.asarray(service, dtype=np.int64)
        edges = np.asarray(transition, dtype=np.int64) + np.asarray(
            service, dtype=np.int64
        )[:, None, :]
        kernel = tuple(sorted(random.sample(range(3), random.randrange(1, 4))))
        generic = exact_separation(model, tasks, 4, kernel)
        fast = fast_separation(source, edges, 4, kernel)
        assert fast.gap == generic.gap
        assert fast.kernel_infeasible == generic.kernel_infeasible
        assert fast.reachable_by_depth == generic.reachable_by_depth
