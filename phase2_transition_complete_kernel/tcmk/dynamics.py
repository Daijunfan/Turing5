from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Hashable, Protocol, Sequence

Cost = Fraction
INF: Cost | None = None


class FiniteCostModel(Protocol):
    @property
    def state_count(self) -> int: ...

    def service_cost(self, task: Hashable, state: int) -> Cost | None: ...

    def transition_cost(self, task: Hashable, previous: int | None, state: int) -> Cost | None: ...


def add(left: Cost | None, right: Cost | None) -> Cost | None:
    return None if left is None or right is None else left + right


def minimum(values: Sequence[Cost | None]) -> Cost | None:
    finite = [value for value in values if value is not None]
    return min(finite) if finite else None


@dataclass(frozen=True)
class DynamicResult:
    cost: Cost | None
    path: tuple[int, ...]


def initial_vector(
    model: FiniteCostModel,
    task: Hashable,
    allowed: Sequence[int],
    initial_state: int | None,
    legacy_free_initial: bool = False,
) -> tuple[Cost | None, ...]:
    vector: list[Cost | None] = []
    for state in allowed:
        service = model.service_cost(task, state)
        if legacy_free_initial:
            vector.append(service)
        else:
            vector.append(add(model.transition_cost(task, initial_state, state), service))
    return tuple(vector)


def advance_vector(
    model: FiniteCostModel,
    task: Hashable,
    previous_states: Sequence[int],
    previous_costs: Sequence[Cost | None],
    next_states: Sequence[int],
) -> tuple[Cost | None, ...]:
    result: list[Cost | None] = []
    for state in next_states:
        service = model.service_cost(task, state)
        candidates = [
            add(cost, model.transition_cost(task, previous, state))
            for previous, cost in zip(previous_states, previous_costs)
        ]
        result.append(add(minimum(candidates), service))
    return tuple(result)


def offline_optimum(
    model: FiniteCostModel,
    trace: Sequence[Hashable],
    allowed: Sequence[int] | None = None,
    initial_state: int | None = None,
    legacy_free_initial: bool = False,
) -> DynamicResult:
    """Exact min-plus dynamic program with an explicit architecture or source.

    ``initial_state=None`` denotes a virtual source.  The model must provide its
    construction cost.  ``legacy_free_initial`` exists only for the isolated
    first-generation regression and must not be used by new experiments.
    """

    if not trace:
        return DynamicResult(Fraction(0), ())
    states = tuple(range(model.state_count)) if allowed is None else tuple(allowed)
    if not states:
        return DynamicResult(None, ())
    costs = initial_vector(model, trace[0], states, initial_state, legacy_free_initial)
    parents: list[tuple[int | None, ...]] = []
    for task in trace[1:]:
        next_costs: list[Cost | None] = []
        row: list[int | None] = []
        for state in states:
            service = model.service_cost(task, state)
            candidates = [
                add(cost, model.transition_cost(task, previous, state))
                for previous, cost in zip(states, costs)
            ]
            finite = [(value, index) for index, value in enumerate(candidates) if value is not None]
            if service is None or not finite:
                next_costs.append(None)
                row.append(None)
            else:
                value, index = min(finite)
                next_costs.append(value + service)
                row.append(index)
        costs = tuple(next_costs)
        parents.append(tuple(row))
    finite_ends = [(value, index) for index, value in enumerate(costs) if value is not None]
    if not finite_ends:
        return DynamicResult(None, ())
    value, position = min(finite_ends)
    path = [states[position]]
    for row in reversed(parents):
        parent = row[position]
        assert parent is not None
        position = parent
        path.append(states[position])
    path.reverse()
    return DynamicResult(value, tuple(path))

