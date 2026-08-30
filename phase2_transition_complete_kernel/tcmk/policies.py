from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Hashable, Sequence

from .dynamics import Cost, FiniteCostModel, add, minimum, offline_optimum


@dataclass(frozen=True)
class PolicyResult:
    cost: Cost | None
    path: tuple[int, ...]
    switches: int


def _path_cost(
    model: FiniteCostModel,
    trace: Sequence[Hashable],
    path: Sequence[int],
    initial_state: int | None,
) -> Cost | None:
    total: Cost | None = Fraction(0)
    previous = initial_state
    for task, state in zip(trace, path):
        total = add(total, model.transition_cost(task, previous, state))
        total = add(total, model.service_cost(task, state))
        previous = state
    return total


def _result(
    model: FiniteCostModel,
    trace: Sequence[Hashable],
    path: Sequence[int],
    initial_state: int | None,
) -> PolicyResult:
    switches = sum(left != right for left, right in zip(path, path[1:]))
    return PolicyResult(_path_cost(model, trace, path, initial_state), tuple(path), switches)


def best_fixed(
    model: FiniteCostModel,
    trace: Sequence[Hashable],
    allowed: Sequence[int] | None = None,
    initial_state: int | None = None,
) -> PolicyResult:
    states = tuple(range(model.state_count)) if allowed is None else tuple(allowed)
    candidates = []
    for state in states:
        path = (state,) * len(trace)
        cost = _path_cost(model, trace, path, initial_state)
        if cost is not None:
            candidates.append((cost, state, path))
    if not candidates:
        return PolicyResult(None, (), 0)
    _, _, path = min(candidates)
    return _result(model, trace, path, initial_state)


def instantaneous_best(
    model: FiniteCostModel,
    trace: Sequence[Hashable],
    allowed: Sequence[int] | None = None,
    initial_state: int | None = None,
) -> PolicyResult:
    states = tuple(range(model.state_count)) if allowed is None else tuple(allowed)
    path = []
    for task in trace:
        feasible = [
            (cost, state)
            for state in states
            if (cost := model.service_cost(task, state)) is not None
        ]
        if not feasible:
            return PolicyResult(None, (), 0)
        path.append(min(feasible)[1])
    return _result(model, trace, path, initial_state)


def migration_greedy(
    model: FiniteCostModel,
    trace: Sequence[Hashable],
    allowed: Sequence[int] | None = None,
    initial_state: int | None = None,
) -> PolicyResult:
    states = tuple(range(model.state_count)) if allowed is None else tuple(allowed)
    path: list[int] = []
    previous = initial_state
    for task in trace:
        candidates = []
        for state in states:
            cost = add(
                model.transition_cost(task, previous, state),
                model.service_cost(task, state),
            )
            if cost is not None:
                candidates.append((cost, state))
        if not candidates:
            return PolicyResult(None, (), 0)
        previous = min(candidates)[1]
        path.append(previous)
    return _result(model, trace, path, initial_state)


def morphological_credit_switching(
    model: FiniteCostModel,
    trace: Sequence[Hashable],
    allowed: Sequence[int] | None = None,
    initial_state: int | None = None,
) -> PolicyResult:
    """First-generation MCS, generalized to the exact cost interface."""

    states = tuple(range(model.state_count)) if allowed is None else tuple(allowed)
    if not trace:
        return PolicyResult(Fraction(0), (), 0)
    first = migration_greedy(model, trace[:1], states, initial_state)
    if first.cost is None:
        return first
    current = first.path[0]
    path = [current]
    credit = {state: Fraction(0) for state in states}
    for task in trace[1:]:
        current_service = model.service_cost(task, current)
        if current_service is None:
            candidates = []
            for state in states:
                step = add(
                    model.transition_cost(task, current, state),
                    model.service_cost(task, state),
                )
                if step is not None:
                    candidates.append((step, state))
            if not candidates:
                return PolicyResult(None, (), 0)
            current = min(candidates)[1]
            credit = {state: Fraction(0) for state in states}
        else:
            target = current
            best_surplus = Fraction(0)
            for state in states:
                service = model.service_cost(task, state)
                movement = model.transition_cost(task, current, state)
                if service is None or movement is None:
                    credit[state] = Fraction(0)
                    continue
                credit[state] = max(Fraction(0), credit[state] + current_service - service)
                surplus = credit[state] - movement
                if surplus > best_surplus:
                    best_surplus, target = surplus, state
            if target != current:
                current = target
                credit = {state: Fraction(0) for state in states}
        path.append(current)
    return _result(model, trace, path, initial_state)


def work_function_algorithm(
    model: FiniteCostModel,
    trace: Sequence[Hashable],
    allowed: Sequence[int] | None = None,
    initial_state: int | None = None,
) -> PolicyResult:
    """Run WFA only when the model explicitly certifies the MTS metric premise."""

    if not getattr(model, "is_metric", lambda: False)():
        raise ValueError("WFA theory requires a task-independent metric transition cost")
    states = tuple(range(model.state_count)) if allowed is None else tuple(allowed)
    if not trace:
        return PolicyResult(Fraction(0), (), 0)
    work = []
    for state in states:
        work.append(
            add(
                model.transition_cost(trace[0], initial_state, state),
                model.service_cost(trace[0], state),
            )
        )
    feasible = [(cost, pos) for pos, cost in enumerate(work) if cost is not None]
    if not feasible:
        return PolicyResult(None, (), 0)
    current_pos = min(feasible)[1]
    path = [states[current_pos]]
    for task in trace[1:]:
        following = []
        for target in states:
            arrivals = [
                add(cost, model.transition_cost(task, previous, target))
                for previous, cost in zip(states, work)
            ]
            following.append(add(minimum(arrivals), model.service_cost(task, target)))
        action_scores = [
            add(cost, model.transition_cost(task, states[current_pos], target))
            for target, cost in zip(states, following)
        ]
        feasible = [(cost, pos) for pos, cost in enumerate(action_scores) if cost is not None]
        if not feasible:
            return PolicyResult(None, (), 0)
        current_pos = min(feasible)[1]
        path.append(states[current_pos])
        work = following
    return _result(model, trace, path, initial_state)


def all_baselines(
    model: FiniteCostModel,
    trace: Sequence[Hashable],
    allowed: Sequence[int] | None = None,
    initial_state: int | None = None,
) -> dict[str, PolicyResult | None]:
    offline = offline_optimum(model, trace, allowed, initial_state)
    result: dict[str, PolicyResult | None] = {
        "offline": PolicyResult(
            offline.cost,
            offline.path,
            sum(a != b for a, b in zip(offline.path, offline.path[1:])),
        ),
        "fixed": best_fixed(model, trace, allowed, initial_state),
        "instantaneous": instantaneous_best(model, trace, allowed, initial_state),
        "migration_greedy": migration_greedy(model, trace, allowed, initial_state),
        "mcs": morphological_credit_switching(model, trace, allowed, initial_state),
    }
    result["wfa"] = (
        work_function_algorithm(model, trace, allowed, initial_state)
        if getattr(model, "is_metric", lambda: False)()
        else None
    )
    return result

