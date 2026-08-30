from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from itertools import combinations
import json
from typing import Iterable, Sequence


INF = None
Cost = Fraction
MaybeCost = Cost | None
Residual = tuple[MaybeCost, ...]


def add(left: MaybeCost, right: MaybeCost) -> MaybeCost:
    return None if left is None or right is None else left + right


def finite_min(values: Iterable[MaybeCost]) -> MaybeCost:
    finite = tuple(value for value in values if value is not None)
    return min(finite) if finite else None


def normalize(vector: Sequence[MaybeCost]) -> tuple[Cost, Residual]:
    base = finite_min(vector)
    if base is None:
        raise ValueError("configuration has no finite coordinate")
    return base, tuple(None if value is None else value - base for value in vector)


def encode_cost(value: MaybeCost) -> str:
    if value is None:
        return "inf"
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def decode_cost(value: str) -> MaybeCost:
    return None if value == "inf" else Fraction(value)


def stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(value: object) -> str:
    return sha256(stable_json(value).encode()).hexdigest()


@dataclass(frozen=True)
class ResidualTransition:
    source: int
    task: int
    target: int
    increment: Cost


@dataclass(frozen=True)
class ResidualClosure:
    states: tuple[Residual, ...]
    witnesses: tuple[tuple[int, ...], ...]
    transitions: tuple[ResidualTransition, ...]
    closed: bool
    unprocessed: int
    state_limit: int | None

    def transition_table(self, task_count: int) -> tuple[tuple[tuple[int, Cost], ...], ...]:
        table: list[list[tuple[int, Cost] | None]] = [
            [None] * task_count for _ in self.states
        ]
        for transition in self.transitions:
            table[transition.source][transition.task] = (
                transition.target,
                transition.increment,
            )
        if self.closed and any(item is None for row in table for item in row):
            raise AssertionError("closed residual graph has a missing transition")
        return tuple(
            tuple(item for item in row if item is not None) for row in table
        )


@dataclass(frozen=True)
class ComparisonResult:
    equivalent: bool
    witness: tuple[int, ...]
    explored_pairs: int
    closed: bool


@dataclass(frozen=True)
class BehaviorQuotient:
    state_to_class: tuple[int, ...]
    classes: tuple[tuple[int, ...], ...]
    refinement_rounds: int


class CostAutomaton:
    """Finite min-plus morphology cost automaton with explicit infinity."""

    def __init__(
        self,
        tasks: Sequence[str],
        alpha: Sequence[int | Fraction | None],
        matrices: Sequence[Sequence[Sequence[int | Fraction | None]]],
        state_names: Sequence[str] | None = None,
    ):
        self.tasks = tuple(tasks)
        self.alpha = tuple(None if value is None else Fraction(value) for value in alpha)
        self.matrices = tuple(
            tuple(
                tuple(None if value is None else Fraction(value) for value in row)
                for row in matrix
            )
            for matrix in matrices
        )
        self.state_names = tuple(state_names or (f"h{index}" for index in range(len(alpha))))
        self._validate()
        self.initial_value, self.initial_residual = normalize(self.alpha)

    @property
    def state_count(self) -> int:
        return len(self.alpha)

    def _validate(self) -> None:
        count = len(self.alpha)
        if not count or len(self.state_names) != count:
            raise ValueError("state metadata has inconsistent size")
        if len(self.tasks) != len(self.matrices):
            raise ValueError("one transition matrix is required per task")
        if any(len(matrix) != count or any(len(row) != count for row in matrix) for matrix in self.matrices):
            raise ValueError("transition matrices must be square")
        if finite_min(self.alpha) is None:
            raise ValueError("at least one initial state must be finite")

    def step(self, residual: Residual, task: int) -> tuple[Cost, Residual]:
        matrix = self.matrices[task]
        raw = tuple(
            finite_min(add(residual[previous], matrix[previous][target]) for previous in range(self.state_count))
            for target in range(self.state_count)
        )
        return normalize(raw)

    def evaluate(self, word: Sequence[int]) -> Cost:
        residual = self.initial_residual
        total = self.initial_value
        for task in word:
            increment, residual = self.step(residual, task)
            total += increment
        return total

    def full_dp(self, word: Sequence[int]) -> tuple[Cost, tuple[int, ...]]:
        costs = self.alpha
        parents: list[tuple[int | None, ...]] = []
        for task in word:
            following: list[MaybeCost] = []
            row: list[int | None] = []
            for target in range(self.state_count):
                candidates = tuple(
                    add(costs[previous], self.matrices[task][previous][target])
                    for previous in range(self.state_count)
                )
                finite = tuple(
                    (value, previous)
                    for previous, value in enumerate(candidates)
                    if value is not None
                )
                if finite:
                    value, previous = min(finite)
                    following.append(value)
                    row.append(previous)
                else:
                    following.append(None)
                    row.append(None)
            costs = tuple(following)
            parents.append(tuple(row))
        finite_end = tuple(
            (value, state) for state, value in enumerate(costs) if value is not None
        )
        if not finite_end:
            raise ValueError("word is infeasible")
        value, state = min(finite_end)
        path: list[int] = []
        for row in reversed(parents):
            path.append(state)
            previous = row[state]
            if previous is None:
                raise AssertionError("broken optimal-path parent")
            state = previous
        path.reverse()
        return value, tuple(path)

    def closure(self, state_limit: int | None = None) -> ResidualClosure:
        states = [self.initial_residual]
        witnesses: list[tuple[int, ...]] = [()]
        index = {self.initial_residual: 0}
        queue = deque([0])
        transitions: list[ResidualTransition] = []
        closed = True
        while queue:
            source = queue.popleft()
            for task in range(len(self.tasks)):
                increment, target_residual = self.step(states[source], task)
                target = index.get(target_residual)
                if target is None:
                    if state_limit is not None and len(states) >= state_limit:
                        closed = False
                        return ResidualClosure(
                            tuple(states),
                            tuple(witnesses),
                            tuple(transitions),
                            False,
                            len(queue) + 1,
                            state_limit,
                        )
                    target = len(states)
                    index[target_residual] = target
                    states.append(target_residual)
                    witnesses.append(witnesses[source] + (task,))
                    queue.append(target)
                transitions.append(ResidualTransition(source, task, target, increment))
        return ResidualClosure(
            tuple(states), tuple(witnesses), tuple(transitions), closed, 0, state_limit
        )

    def subset(self, states: Sequence[int]) -> CostAutomaton:
        selected = tuple(sorted(set(states)))
        return CostAutomaton(
            self.tasks,
            tuple(self.alpha[index] for index in selected),
            tuple(
                tuple(
                    tuple(matrix[left][right] for right in selected)
                    for left in selected
                )
                for matrix in self.matrices
            ),
            tuple(self.state_names[index] for index in selected),
        )

    def digest(self) -> str:
        return stable_hash(
            {
                "tasks": self.tasks,
                "alpha": tuple(map(encode_cost, self.alpha)),
                "matrices": tuple(
                    tuple(tuple(map(encode_cost, row)) for row in matrix)
                    for matrix in self.matrices
                ),
                "states": self.state_names,
            }
        )

    def finite_column_difference_bound(self) -> int | None:
        bound = Fraction(0)
        for matrix in self.matrices:
            if any(value is None for row in matrix for value in row):
                return None
            for left in range(self.state_count):
                for right in range(self.state_count):
                    bound = max(
                        bound,
                        max(
                            abs(matrix[previous][left] - matrix[previous][right])
                            for previous in range(self.state_count)
                        ),
                    )
        return int(bound) if bound.denominator == 1 else None

    def closure_certificate(self, closure: ResidualClosure) -> dict:
        if not closure.closed or closure.unprocessed:
            raise ValueError("only a completed closure can be certified")
        encoded_states = [tuple(map(encode_cost, state)) for state in closure.states]
        encoded_transitions = [
            {
                "source": transition.source,
                "task": transition.task,
                "target": transition.target,
                "increment": encode_cost(transition.increment),
            }
            for transition in closure.transitions
        ]
        finite_coordinates = [
            value for state in closure.states for value in state if value is not None
        ]
        infinity_patterns = sorted(
            {"".join("1" if value is None else "0" for value in state) for state in closure.states}
        )
        state_hashes = [stable_hash(state) for state in encoded_states]
        transition_hashes = [stable_hash(edge) for edge in encoded_transitions]
        return {
            "schema": "morphn.residual-closure.v1",
            "automaton_digest": self.digest(),
            "task_alphabet": self.tasks,
            "task_alphabet_digest": stable_hash(self.tasks),
            "cost_matrix_digest": stable_hash(
                tuple(
                    tuple(tuple(map(encode_cost, row)) for row in matrix)
                    for matrix in self.matrices
                )
            ),
            "initial_value": encode_cost(self.initial_value),
            "states": encoded_states,
            "witnesses": closure.witnesses,
            "transitions": encoded_transitions,
            "residual_state_count": len(closure.states),
            "transition_count": len(closure.transitions),
            "unprocessed_state_count": closure.unprocessed,
            "queue_empty": closure.unprocessed == 0,
            "all_successors_recorded": len(closure.transitions) == len(closure.states) * len(self.tasks),
            "maximum_coordinate": encode_cost(max(finite_coordinates, default=Fraction(0))),
            "infinity_coordinate_patterns": infinity_patterns,
            "state_merkle_root": stable_hash(state_hashes),
            "transition_merkle_root": stable_hash(transition_hashes),
            "closure_digest": stable_hash(
                {"states": encoded_states, "transitions": encoded_transitions}
            ),
        }


def behavior_quotient(closure: ResidualClosure, task_count: int) -> BehaviorQuotient:
    if not closure.closed:
        raise ValueError("behavior quotient requires a finite completed closure")
    table = closure.transition_table(task_count)
    blocks = tuple(0 for _ in closure.states)
    rounds = 0
    while True:
        signatures = tuple(
            tuple((table[state][task][1], blocks[table[state][task][0]]) for task in range(task_count))
            for state in range(len(closure.states))
        )
        signature_ids: dict[tuple, int] = {}
        following = []
        for signature in signatures:
            signature_ids.setdefault(signature, len(signature_ids))
            following.append(signature_ids[signature])
        following_tuple = tuple(following)
        if following_tuple == blocks:
            break
        blocks = following_tuple
        rounds += 1
    classes = tuple(
        tuple(index for index, block in enumerate(blocks) if block == class_id)
        for class_id in range(max(blocks, default=-1) + 1)
    )
    return BehaviorQuotient(blocks, classes, rounds)


def compare_automata(
    full: CostAutomaton,
    restricted: CostAutomaton,
    pair_limit: int | None = None,
) -> ComparisonResult:
    if full.tasks != restricted.tasks:
        raise ValueError("task alphabets differ")
    if full.initial_value != restricted.initial_value:
        return ComparisonResult(False, (), 1, True)
    start = (full.initial_residual, restricted.initial_residual)
    queue = deque([(start, ())])
    seen = {start}
    while queue:
        (left, right), prefix = queue.popleft()
        for task in range(len(full.tasks)):
            left_increment, left_next = full.step(left, task)
            right_increment, right_next = restricted.step(right, task)
            if left_increment != right_increment:
                return ComparisonResult(False, prefix + (task,), len(seen), True)
            pair = (left_next, right_next)
            if pair not in seen:
                if pair_limit is not None and len(seen) >= pair_limit:
                    return ComparisonResult(False, prefix + (task,), len(seen), False)
                seen.add(pair)
                queue.append((pair, prefix + (task,)))
    return ComparisonResult(True, (), len(seen), True)


def finite_horizon_gap(
    full: CostAutomaton,
    restricted: CostAutomaton,
    horizon: int,
) -> tuple[Cost, tuple[int, ...]]:
    return finite_horizon_gap_curve(full, restricted, horizon)[-1]


def finite_horizon_gap_curve(
    full: CostAutomaton,
    restricted: CostAutomaton,
    horizon: int,
) -> tuple[tuple[Cost, tuple[int, ...]], ...]:
    start_pair = (full.initial_residual, restricted.initial_residual)
    current = {
        start_pair: (restricted.initial_value - full.initial_value, ())
    }
    best = restricted.initial_value - full.initial_value
    witness: tuple[int, ...] = ()
    curve = [(best, witness)]
    transition_cache: dict[
        tuple[Residual, Residual, int],
        tuple[tuple[Residual, Residual], Cost],
    ] = {}
    for _ in range(horizon):
        following: dict[
            tuple[Residual, Residual], tuple[Cost, tuple[int, ...]]
        ] = {}
        for (left, right), (gap, prefix) in current.items():
            for task in range(len(full.tasks)):
                cache_key = (left, right, task)
                cached = transition_cache.get(cache_key)
                if cached is None:
                    left_increment, left_next = full.step(left, task)
                    right_increment, right_next = restricted.step(right, task)
                    cached = (
                        (left_next, right_next),
                        right_increment - left_increment,
                    )
                    transition_cache[cache_key] = cached
                pair, edge_gap = cached
                next_gap = gap + edge_gap
                word = prefix + (task,)
                previous = following.get(pair)
                if previous is None or next_gap > previous[0]:
                    following[pair] = (next_gap, word)
                if next_gap > best:
                    best, witness = next_gap, word
        current = following
        curve.append((best, witness))
    return tuple(curve)


def minimum_kernel_for_horizon(
    automaton: CostAutomaton, horizon: int
) -> tuple[tuple[int, ...], Cost, tuple[int, ...]]:
    universe = tuple(range(automaton.state_count))
    for size in range(1, len(universe) + 1):
        for candidate in combinations(universe, size):
            try:
                restricted = automaton.subset(candidate)
            except ValueError:
                continue
            gap, witness = finite_horizon_gap(automaton, restricted, horizon)
            if gap == 0:
                return candidate, gap, witness
    raise AssertionError("the full state set is always complete")
