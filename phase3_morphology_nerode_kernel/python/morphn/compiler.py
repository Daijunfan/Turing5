from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

from .automaton import (
    BehaviorQuotient,
    ComparisonResult,
    CostAutomaton,
    behavior_quotient,
    compare_automata,
    finite_horizon_gap,
)


@dataclass(frozen=True)
class ExecutionRefinement:
    kernel_before: tuple[int, ...]
    witness: tuple[int, ...]
    full_path: tuple[int, ...]
    missing_states: tuple[int, ...]
    added_state: int


@dataclass(frozen=True)
class MorphNResult:
    executable_kernel: tuple[int, ...]
    residual_state_count: int
    behavior_kernel_size: int
    behavior_quotient: BehaviorQuotient
    execution_refinements: tuple[ExecutionRefinement, ...]
    deleted_states: tuple[int, ...]
    equivalence: ComparisonResult


@dataclass(frozen=True)
class FiniteKernelResult:
    horizon: int
    executable_kernel: tuple[int, ...]
    refinements: tuple[ExecutionRefinement, ...]
    deleted_states: tuple[int, ...]
    worst_gap: Fraction
    witness: tuple[int, ...]


def _word_cost(automaton: CostAutomaton, word: Sequence[int]) -> Fraction:
    return automaton.evaluate(word)


def compile_morph_n(
    full: CostAutomaton,
    initial_kernel: Sequence[int],
    closure_limit: int | None = None,
) -> MorphNResult:
    """Dual loop: grow executable states, then quotient finite residual behavior."""

    kernel = set(initial_kernel)
    refinements = []
    while True:
        restricted = full.subset(tuple(kernel))
        comparison = compare_automata(full, restricted, closure_limit)
        if comparison.equivalent:
            break
        if not comparison.closed:
            raise RuntimeError("pair residual search did not reach a fixed point")
        _, full_path = full.full_dp(comparison.witness)
        missing = tuple(dict.fromkeys(state for state in full_path if state not in kernel))
        if not missing:
            raise AssertionError("counterexample path contains no missing architecture")
        current_cost = _word_cost(restricted, comparison.witness)
        choices = []
        for state in missing:
            candidate = full.subset(tuple(kernel | {state}))
            improvement = current_cost - _word_cost(candidate, comparison.witness)
            choices.append((improvement, -state, state))
        added = max(choices)[2]
        refinements.append(
            ExecutionRefinement(
                tuple(sorted(kernel)), comparison.witness, full_path, missing, added
            )
        )
        kernel.add(added)

    deleted = []
    for state in tuple(sorted(kernel)):
        candidate_ids = tuple(sorted(kernel - {state}))
        if not candidate_ids:
            continue
        try:
            candidate = full.subset(candidate_ids)
        except ValueError:
            continue
        check = compare_automata(full, candidate, closure_limit)
        if check.equivalent:
            kernel.remove(state)
            deleted.append(state)

    restricted = full.subset(tuple(sorted(kernel)))
    equivalence = compare_automata(full, restricted, closure_limit)
    if not equivalence.equivalent:
        raise AssertionError("deletion pass broke executable-kernel equivalence")
    closure = restricted.closure(closure_limit)
    if not closure.closed:
        raise RuntimeError("final residual closure did not terminate")
    quotient = behavior_quotient(closure, len(full.tasks))
    return MorphNResult(
        tuple(sorted(kernel)),
        len(closure.states),
        len(quotient.classes),
        quotient,
        tuple(refinements),
        tuple(deleted),
        equivalence,
    )


def compile_finite_kernel(
    full: CostAutomaton,
    horizon: int,
    initial_kernel: Sequence[int],
) -> FiniteKernelResult:
    kernel = set(initial_kernel)
    refinements = []
    while True:
        restricted = full.subset(tuple(kernel))
        gap, witness = finite_horizon_gap(full, restricted, horizon)
        if gap == 0:
            break
        _, full_path = full.full_dp(witness)
        missing = tuple(dict.fromkeys(state for state in full_path if state not in kernel))
        if not missing:
            raise AssertionError("positive finite-horizon gap without a missing state")
        current_cost = restricted.evaluate(witness)
        choices = []
        for state in missing:
            candidate = full.subset(tuple(kernel | {state}))
            choices.append((current_cost - candidate.evaluate(witness), -state, state))
        added = max(choices)[2]
        refinements.append(
            ExecutionRefinement(tuple(sorted(kernel)), witness, full_path, missing, added)
        )
        kernel.add(added)
    deleted = []
    for state in tuple(sorted(kernel)):
        candidate_ids = tuple(sorted(kernel - {state}))
        if not candidate_ids:
            continue
        try:
            candidate = full.subset(candidate_ids)
        except ValueError:
            continue
        candidate_gap, _ = finite_horizon_gap(full, candidate, horizon)
        if candidate_gap == 0:
            kernel.remove(state)
            deleted.append(state)
    final = full.subset(tuple(sorted(kernel)))
    gap, witness = finite_horizon_gap(full, final, horizon)
    if gap != 0:
        raise AssertionError("finite deletion pass broke equivalence")
    return FiniteKernelResult(
        horizon,
        tuple(sorted(kernel)),
        tuple(refinements),
        tuple(deleted),
        gap,
        witness,
    )
