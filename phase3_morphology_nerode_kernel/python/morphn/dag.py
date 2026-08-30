from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Sequence

import numpy as np

from .automaton import CostAutomaton


@dataclass(frozen=True)
class DagTask:
    target: int
    budget_slots: int


@dataclass(frozen=True)
class DagRun:
    output: np.ndarray
    cache: dict[int, np.ndarray]
    operations: int
    read_bytes: int
    write_bytes: int
    peak_slots: int
    released: int
    wall_ns: int


class SharedDag:
    def __init__(self, node_count: int, width: int = 32, seed: int = 20260831):
        if node_count < 3:
            raise ValueError("DAG needs two inputs and at least one operator")
        self.node_count = node_count
        self.width = width
        self.parents = tuple(
            None
            if node < 2
            else (node - 1, (node * 1103515245 + 12345) % (node - 1))
            for node in range(node_count)
        )
        random = np.random.default_rng(seed)
        self.inputs = {
            0: random.integers(0, 31, size=width, dtype=np.int64),
            1: random.integers(0, 31, size=width, dtype=np.int64),
        }

    def _required(self, roots: Sequence[int], cache: set[int]) -> set[int]:
        required: set[int] = set()
        stack = list(roots)
        while stack:
            node = stack.pop()
            if node < 2 or node in cache or node in required:
                continue
            required.add(node)
            parents = self.parents[node]
            assert parents is not None
            stack.extend(parents)
        return required

    def execute(
        self,
        task: DagTask,
        source_cache: dict[int, np.ndarray],
        target_keep: frozenset[int],
        *,
        actual: bool = True,
    ) -> DagRun:
        started = perf_counter_ns()
        roots = (task.target, *target_keep)
        required = self._required(roots, set(source_cache))
        useful_cached = set()
        for node in required:
            parents = self.parents[node]
            assert parents is not None
            useful_cached.update(parent for parent in parents if parent in source_cache)
        useful_cached.update(root for root in roots if root in source_cache)

        uses = {node: 0 for node in required | useful_cached}
        for node in required:
            parents = self.parents[node]
            assert parents is not None
            for parent in parents:
                if parent >= 2:
                    uses[parent] = uses.get(parent, 0) + 1
        for root in roots:
            if root >= 2:
                uses[root] = uses.get(root, 0) + 1

        values = {node: source_cache[node] for node in useful_cached} if actual else {}
        values.update(self.inputs if actual else {})
        live = len(useful_cached)
        peak = live
        operations = read = written = released = 0
        placeholder = np.zeros(self.width, dtype=np.int64)
        for node in sorted(required):
            left, right = self.parents[node]  # type: ignore[misc]
            if actual:
                left_value = values[left]
                right_value = values[right]
                coefficient = 1 + node % 13
                value = (left_value * coefficient + right_value * (coefficient + 1)) % 1_000_003
                values[node] = value
            operations += 1
            read += 2 * placeholder.nbytes
            written += placeholder.nbytes
            live += 1
            peak = max(peak, live)
            for parent in (left, right):
                if parent < 2:
                    continue
                uses[parent] -= 1
                if uses[parent] == 0 and parent not in target_keep:
                    if actual:
                        values.pop(parent, None)
                    live -= 1
                    released += 1

        output = values[task.target].copy() if actual else placeholder
        next_cache = (
            {node: values[node] for node in target_keep} if actual else {}
        )
        wall = perf_counter_ns() - started
        return DagRun(output, next_cache, operations, read, written, peak, released, wall)

    def reference(self, target: int) -> np.ndarray:
        return self.execute(
            DagTask(target, self.node_count), {}, frozenset(), actual=True
        ).output


def checkpoint_architectures(node_count: int) -> tuple[frozenset[int], ...]:
    candidates = [frozenset()]
    for stride in (64, 32, 16, 8, 4, 2):
        candidates.append(frozenset(range(2, node_count, stride)))
    candidates.append(frozenset(range(2, node_count)))
    unique = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return tuple(unique)


def build_dag_automaton(
    dag: SharedDag,
    tasks: Sequence[DagTask],
    architectures: Sequence[frozenset[int]],
) -> CostAutomaton:
    matrices = []
    for task in tasks:
        matrix = []
        for source in architectures:
            source_cache = {node: dag.inputs[0] for node in source}
            row = []
            for target in architectures:
                run = dag.execute(task, source_cache, target, actual=False)
                if len(target) > task.budget_slots or run.peak_slots > task.budget_slots:
                    row.append(None)
                else:
                    row.append(
                        run.operations * 100
                        + (run.read_bytes + run.write_bytes) // 8
                        + run.peak_slots
                    )
            matrix.append(tuple(row))
        matrices.append(tuple(matrix))
    return CostAutomaton(
        tuple(f"target_{task.target}_budget_{task.budget_slots}" for task in tasks),
        tuple(0 if index == 0 else None for index in range(len(architectures))),
        tuple(matrices),
        tuple(f"checkpoint_{index}_size_{len(value)}" for index, value in enumerate(architectures)),
    )


def execute_architecture_path(
    dag: SharedDag,
    tasks: Sequence[DagTask],
    architectures: Sequence[frozenset[int]],
    path: Sequence[int],
) -> dict:
    cache: dict[int, np.ndarray] = {}
    totals = {"wall_ns": 0, "operations": 0, "read_bytes": 0, "write_bytes": 0, "released": 0, "peak_slots": 0}
    correct = 0
    for task, architecture in zip(tasks, path):
        run = dag.execute(task, cache, architectures[architecture], actual=True)
        if (
            len(architectures[architecture]) > task.budget_slots
            or run.peak_slots > task.budget_slots
        ):
            raise ValueError("architecture path violates the task memory budget")
        cache = run.cache
        totals["wall_ns"] += run.wall_ns
        totals["operations"] += run.operations
        totals["read_bytes"] += run.read_bytes
        totals["write_bytes"] += run.write_bytes
        totals["released"] += run.released
        totals["peak_slots"] = max(totals["peak_slots"], run.peak_slots)
        correct += int(np.array_equal(run.output, dag.reference(task.target)))
    return {**totals, "correct": correct, "tasks": len(tasks)}


def execute_dtr(dag: SharedDag, tasks: Sequence[DagTask]) -> dict:
    cache: dict[int, np.ndarray] = {}
    totals = {"wall_ns": 0, "operations": 0, "read_bytes": 0, "write_bytes": 0, "released": 0, "peak_slots": 0}
    correct = 0
    within_budget = True
    for task in tasks:
        candidates = sorted(
            dag._required((task.target,), set(cache)) | set(cache), reverse=True
        )
        keep = frozenset(candidates[: max(0, task.budget_slots // 2)])
        run = dag.execute(task, cache, keep, actual=True)
        within_budget &= (
            len(keep) <= task.budget_slots and run.peak_slots <= task.budget_slots
        )
        cache = run.cache
        totals["wall_ns"] += run.wall_ns
        totals["operations"] += run.operations
        totals["read_bytes"] += run.read_bytes
        totals["write_bytes"] += run.write_bytes
        totals["released"] += run.released
        totals["peak_slots"] = max(totals["peak_slots"], run.peak_slots)
        correct += int(np.array_equal(run.output, dag.reference(task.target)))
    return {
        **totals,
        "correct": correct,
        "tasks": len(tasks),
        "within_budget": within_budget,
    }
