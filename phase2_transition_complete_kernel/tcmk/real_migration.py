from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Hashable, Sequence

from .morphology import Morphology, Tree, multiplication_work, result_size

Interval = tuple[int, int]


def interval_nodes(tree: Tree) -> dict[Interval, Tree]:
    return {(node.i, node.j): node for node in tree.internal_nodes()}


def interval_bytes(dims: Sequence[int], interval: Interval, element_bytes: int = 8) -> int:
    return result_size(dims, *interval) * element_bytes


@dataclass(frozen=True)
class MigrationCertificate:
    source: str
    target: str
    update_index: int
    reused_intervals: tuple[Interval, ...]
    invalidated_intervals: tuple[Interval, ...]
    recomputed_intervals: tuple[Interval, ...]
    execution_order: tuple[Interval, ...]
    released_intervals: tuple[Interval, ...]
    peak_memory: int
    total_work: int
    read_bytes: int
    write_bytes: int
    exact: bool


@dataclass(frozen=True)
class MigrationTotals:
    peak_memory: int
    total_work: int
    read_bytes: int
    write_bytes: int


def _dependencies(node: Tree) -> tuple[Interval, ...]:
    assert node.left is not None and node.right is not None
    return tuple(
        (child.i, child.j) for child in (node.left, node.right) if not child.leaf
    )


def _order_stats(
    dims: Sequence[int],
    nodes: dict[Interval, Tree],
    reused: set[Interval],
    order: Sequence[Interval],
    old_memory: int,
    element_bytes: int,
) -> tuple[int, int, int, int]:
    resident = sum(interval_bytes(dims, item, element_bytes) for item in reused)
    peak = max(old_memory, resident)
    work = read = written = 0
    available = set(reused)
    for interval in order:
        node = nodes[interval]
        if any(dependency not in available for dependency in _dependencies(node)):
            raise ValueError("execution order violates a target-tree dependency")
        output_bytes = interval_bytes(dims, interval, element_bytes)
        peak = max(peak, resident + output_bytes)
        resident += output_bytes
        i, k, j = node.i, node.split, node.j
        work += multiplication_work(dims, i, k, j)
        read += (
            result_size(dims, i, k) + result_size(dims, k + 1, j)
        ) * element_bytes
        written += output_bytes
        available.add(interval)
    return peak, work, read, written


def _exact_topological_order(
    dims: Sequence[int],
    nodes: dict[Interval, Tree],
    reused: set[Interval],
    recomputed: tuple[Interval, ...],
    old_memory: int,
    element_bytes: int,
) -> tuple[Interval, ...]:
    """Subset DP over all legal recomputation schedules."""

    position = {interval: index for index, interval in enumerate(recomputed)}
    dependency_masks = []
    for interval in recomputed:
        mask = 0
        for dependency in _dependencies(nodes[interval]):
            if dependency not in reused:
                mask |= 1 << position[dependency]
        dependency_masks.append(mask)
    sizes = [interval_bytes(dims, interval, element_bytes) for interval in recomputed]
    initial_memory = sum(interval_bytes(dims, item, element_bytes) for item in reused)

    @lru_cache(maxsize=None)
    def solve(mask: int) -> tuple[int, tuple[int, ...]]:
        if mask == (1 << len(recomputed)) - 1:
            return max(old_memory, initial_memory + sum(sizes)), ()
        resident = initial_memory + sum(
            size for index, size in enumerate(sizes) if mask & (1 << index)
        )
        choices = []
        for index, interval in enumerate(recomputed):
            bit = 1 << index
            if mask & bit or dependency_masks[index] & ~mask:
                continue
            suffix_peak, suffix = solve(mask | bit)
            peak = max(old_memory, resident + sizes[index], suffix_peak)
            choices.append((peak, (index,) + suffix, interval))
        if not choices:
            raise AssertionError("target tree contains a dependency cycle")
        peak, order, _ = min(choices, key=lambda item: (item[0], item[2], item[1]))
        return peak, order

    _, indices = solve(0)
    return tuple(recomputed[index] for index in indices)


def _greedy_topological_order(
    dims: Sequence[int],
    nodes: dict[Interval, Tree],
    reused: set[Interval],
    recomputed: tuple[Interval, ...],
    element_bytes: int,
) -> tuple[Interval, ...]:
    available = set(reused)
    pending = set(recomputed)
    order = []
    while pending:
        ready = [
            interval
            for interval in pending
            if all(item in available for item in _dependencies(nodes[interval]))
        ]
        if not ready:
            raise AssertionError("target tree contains a dependency cycle")
        chosen = min(ready, key=lambda item: (interval_bytes(dims, item, element_bytes), item))
        order.append(chosen)
        pending.remove(chosen)
        available.add(chosen)
    return tuple(order)


def exact_migration_totals(
    dims: Sequence[int],
    source: Tree | None,
    target: Tree,
    update_index: int,
    *,
    element_bytes: int = 8,
) -> MigrationTotals:
    """Closed-form exact totals for the fully-materialized-interval model.

    Releasing every non-reusable source interval first is always safe.  Every
    target interval must be resident at the end, and every recomputed interval's
    work and I/O are schedule-independent.  Thus ``max(old cache, target cache)``
    is both a peak lower bound and is attained by any topological recomputation.
    Small planners still use subset DP as an independent constructive check.
    """

    source_nodes = {} if source is None else interval_nodes(source)
    target_nodes = interval_nodes(target)
    source_set, target_set = set(source_nodes), set(target_nodes)
    invalidated = {
        interval for interval in source_set if interval[0] <= update_index <= interval[1]
    }
    reused = (source_set & target_set) - invalidated
    recomputed = target_set - reused
    old_memory = sum(
        interval_bytes(dims, interval, element_bytes) for interval in source_set
    )
    target_memory = sum(
        interval_bytes(dims, interval, element_bytes) for interval in target_set
    )
    work = read = written = 0
    for interval in recomputed:
        node = target_nodes[interval]
        i, k, j = node.i, node.split, node.j
        work += multiplication_work(dims, i, k, j)
        read += (
            result_size(dims, i, k) + result_size(dims, k + 1, j)
        ) * element_bytes
        written += interval_bytes(dims, interval, element_bytes)
    return MigrationTotals(max(old_memory, target_memory), work, read, written)


def plan_migration(
    dims: Sequence[int],
    source: Tree | None,
    target: Tree,
    update_index: int,
    *,
    element_bytes: int = 8,
    max_exact_nodes: int = 20,
) -> MigrationCertificate:
    """Plan a cache-valid transition after exactly one leaf replacement."""

    source_nodes = {} if source is None else interval_nodes(source)
    target_nodes = interval_nodes(target)
    source_intervals = set(source_nodes)
    target_intervals = set(target_nodes)
    invalidated = {
        interval for interval in source_intervals if interval[0] <= update_index <= interval[1]
    }
    reusable = (source_intervals & target_intervals) - invalidated
    released = source_intervals - reusable
    recomputed = tuple(sorted(target_intervals - reusable))
    old_memory = sum(
        interval_bytes(dims, interval, element_bytes) for interval in source_intervals
    )
    exact = len(recomputed) <= max_exact_nodes
    if exact:
        order = _exact_topological_order(
            dims,
            target_nodes,
            reusable,
            recomputed,
            old_memory,
            element_bytes,
        )
    else:
        order = _greedy_topological_order(
            dims, target_nodes, reusable, recomputed, element_bytes
        )
    peak, work, read, written = _order_stats(
        dims, target_nodes, reusable, order, old_memory, element_bytes
    )
    return MigrationCertificate(
        "SOURCE" if source is None else source.render(),
        target.render(),
        update_index,
        tuple(sorted(reusable)),
        tuple(sorted(invalidated)),
        recomputed,
        order,
        tuple(sorted(released)),
        peak,
        work,
        read,
        written,
        exact,
    )


@dataclass(frozen=True)
class RealTask:
    update_index: int
    memory_budget: int


@dataclass(frozen=True)
class RealCostWeights:
    work: Fraction = Fraction(1)
    read_byte: Fraction = Fraction(1, 8)
    write_byte: Fraction = Fraction(1, 8)
    peak_byte: Fraction = Fraction(1, 8)


class PersistentMatrixModel:
    """Task-dependent weighted automaton for maintained matrix-chain products."""

    def __init__(
        self,
        dims: Sequence[int],
        states: Sequence[Morphology],
        weights: RealCostWeights = RealCostWeights(),
        *,
        element_bytes: int = 8,
        max_exact_nodes: int = 20,
    ):
        self.dims = tuple(dims)
        self.states = tuple(states)
        self.weights = weights
        self.element_bytes = element_bytes
        self.max_exact_nodes = max_exact_nodes
        self._certificate_cache: dict[
            tuple[int, int | None, int], MigrationCertificate
        ] = {}

    @property
    def state_count(self) -> int:
        return len(self.states)

    def service_cost(self, task: Hashable, state: int) -> Fraction | None:
        del task, state
        # The update and all maintenance work are charged on the transition edge.
        return Fraction(0)

    def certificate(
        self, task: RealTask, previous: int | None, state: int
    ) -> MigrationCertificate:
        key = (task.update_index, previous, state)
        if key not in self._certificate_cache:
            source = None if previous is None else self.states[previous].tree
            self._certificate_cache[key] = plan_migration(
                self.dims,
                source,
                self.states[state].tree,
                task.update_index,
                element_bytes=self.element_bytes,
                max_exact_nodes=self.max_exact_nodes,
            )
        return self._certificate_cache[key]

    def transition_cost(
        self, task: Hashable, previous: int | None, state: int
    ) -> Fraction | None:
        assert isinstance(task, RealTask)
        certificate = self.certificate(task, previous, state)
        if certificate.peak_memory > task.memory_budget:
            return None
        return (
            self.weights.work * certificate.total_work
            + self.weights.read_byte * certificate.read_bytes
            + self.weights.write_byte * certificate.write_bytes
            + self.weights.peak_byte * certificate.peak_memory
        )

    def is_metric(self) -> bool:
        # Updating different leaves changes invalidation and edge costs; the model
        # is task-dependent and generally asymmetric, so classical MTS WFA theory
        # does not apply.
        return False

    def cached_plans_are_exact(self) -> bool:
        return all(certificate.exact for certificate in self._certificate_cache.values())
