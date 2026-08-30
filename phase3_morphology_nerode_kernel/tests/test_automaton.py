from fractions import Fraction
from itertools import product

from morphn.automaton import (
    CostAutomaton,
    behavior_quotient,
    compare_automata,
    minimum_kernel_for_horizon,
)
from morphn.compiler import compile_morph_n


def finite_example() -> CostAutomaton:
    return CostAutomaton(
        ("left", "right"),
        (0, None, None),
        (
            ((0, 4, 2), (1, 0, 2), (1, 1, 0)),
            ((0, 1, 2), (4, 0, 2), (1, 1, 0)),
        ),
        ("L", "R", "bridge"),
    )


def test_controller_matches_full_dynamic_program_for_all_words_through_8():
    automaton = finite_example()
    closure = automaton.closure()
    assert closure.closed and closure.unprocessed == 0
    table = closure.transition_table(len(automaton.tasks))
    for length in range(9):
        for word in product(range(2), repeat=length):
            state = 0
            value = automaton.initial_value
            for task in word:
                state, increment = table[state][task]
                value += increment
            assert value == automaton.evaluate(word) == automaton.full_dp(word)[0]


def test_finite_column_difference_bound_and_behavior_quotient():
    automaton = finite_example()
    bound = automaton.finite_column_difference_bound()
    assert bound is not None
    closure = automaton.closure()
    assert max(
        value
        for residual in closure.states
        for value in residual
        if value is not None
    ) <= bound
    quotient = behavior_quotient(closure, 2)
    assert len(quotient.classes) <= len(closure.states)


def test_unbounded_raw_residual_counterexample_uses_explicit_infinity():
    automaton = CostAutomaton(
        ("tick",),
        (0, 0),
        (((0, None), (None, 1)),),
    )
    closure = automaton.closure(state_limit=30)
    assert not closure.closed
    assert closure.states[-1] == (Fraction(0), Fraction(29))
    assert all(automaton.evaluate((0,) * length) == 0 for length in range(30))


def test_infinite_horizon_executable_kernel_and_finite_horizon_sequence():
    automaton = finite_example()
    result = compile_morph_n(automaton, (0, 1))
    assert result.equivalence.equivalent and result.equivalence.closed
    assert len(result.executable_kernel) <= automaton.state_count
    for horizon in range(1, 6):
        kernel, gap, _ = minimum_kernel_for_horizon(automaton, horizon)
        assert kernel and gap == 0
