from itertools import product

from morphn.matrix_chain import all_trees, build_real_automaton


def test_catalan_counts_and_real_cost_automaton():
    assert [len(all_trees(n)) for n in range(2, 7)] == [1, 2, 5, 14, 42]
    automaton, trees, static = build_real_automaton((1, 2, 4, 1, 6, 6))
    assert len(trees) == 14 and static
    assert automaton.finite_column_difference_bound() is not None
    for length in range(5):
        for word in product(range(5), repeat=length):
            assert automaton.evaluate(word) == automaton.full_dp(word)[0]
