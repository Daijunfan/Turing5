from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

from .automaton import CostAutomaton


@dataclass(frozen=True)
class Tree:
    i: int
    j: int
    left: Tree | None = None
    right: Tree | None = None

    @property
    def leaf(self) -> bool:
        return self.i == self.j

    @property
    def split(self) -> int:
        if self.left is None:
            raise ValueError("leaf has no split")
        return self.left.j

    def nodes(self) -> tuple[Tree, ...]:
        if self.leaf:
            return ()
        assert self.left is not None and self.right is not None
        return (self,) + self.left.nodes() + self.right.nodes()

    def intervals(self) -> frozenset[tuple[int, int]]:
        return frozenset((node.i, node.j) for node in self.nodes())

    def render(self, root: bool = True) -> str:
        if self.leaf:
            return chr(ord("A") + self.i) if self.i < 26 else f"A{self.i}"
        assert self.left is not None and self.right is not None
        text = f"({self.left.render(False)}{self.right.render(False)})"
        return text[1:-1] if root else text


@lru_cache(maxsize=None)
def interval_trees(i: int, j: int) -> tuple[Tree, ...]:
    if i == j:
        return (Tree(i, j),)
    return tuple(
        Tree(i, j, left, right)
        for split in range(i, j)
        for left in interval_trees(i, split)
        for right in interval_trees(split + 1, j)
    )


def all_trees(matrix_count: int) -> tuple[Tree, ...]:
    return interval_trees(0, matrix_count - 1)


def result_size(dims: Sequence[int], i: int, j: int) -> int:
    return dims[i] * dims[j + 1]


def tree_work_peak(tree: Tree, dims: Sequence[int]) -> tuple[int, int]:
    if tree.leaf:
        return 0, 0
    assert tree.left is not None and tree.right is not None
    left_work, left_peak = tree_work_peak(tree.left, dims)
    right_work, right_peak = tree_work_peak(tree.right, dims)
    i, split, j = tree.i, tree.split, tree.j
    left_temp = result_size(dims, i, split) if i < split else 0
    right_temp = result_size(dims, split + 1, j) if split + 1 < j else 0
    output = result_size(dims, i, j)
    coexist = left_temp + right_temp + output
    peak = min(
        max(left_peak, left_temp + right_peak, coexist),
        max(right_peak, right_temp + left_peak, coexist),
    )
    work = left_work + right_work + dims[i] * dims[split + 1] * dims[j + 1]
    return work, peak


def static_pareto_indices(trees: Sequence[Tree], dims: Sequence[int]) -> tuple[int, ...]:
    points = tuple(tree_work_peak(tree, dims) for tree in trees)
    result = []
    seen_points = set()
    for index, point in sorted(enumerate(points), key=lambda item: (item[1][1], item[1][0], trees[item[0]].render())):
        if point in seen_points:
            continue
        if not any(
            other[0] <= point[0]
            and other[1] <= point[1]
            and other != point
            for other in points
        ):
            result.append(index)
            seen_points.add(point)
    return tuple(result)


@dataclass(frozen=True)
class MigrationTotals:
    peak_bytes: int
    work: int
    read_bytes: int
    write_bytes: int

    @property
    def scalar_cost(self) -> int:
        return self.work + (self.read_bytes + self.write_bytes + self.peak_bytes) // 8


def migration_totals(
    dims: Sequence[int], source: Tree, target: Tree, update: int
) -> MigrationTotals:
    source_nodes = {(node.i, node.j): node for node in source.nodes()}
    target_nodes = {(node.i, node.j): node for node in target.nodes()}
    invalidated = {
        interval for interval in source_nodes if interval[0] <= update <= interval[1]
    }
    reused = (set(source_nodes) & set(target_nodes)) - invalidated
    recomputed = set(target_nodes) - reused
    source_memory = sum(result_size(dims, *interval) * 8 for interval in source_nodes)
    target_memory = sum(result_size(dims, *interval) * 8 for interval in target_nodes)
    work = read = written = 0
    for interval in recomputed:
        node = target_nodes[interval]
        i, split, j = node.i, node.split, node.j
        work += dims[i] * dims[split + 1] * dims[j + 1]
        read += (
            result_size(dims, i, split) + result_size(dims, split + 1, j)
        ) * 8
        written += result_size(dims, i, j) * 8
    return MigrationTotals(max(source_memory, target_memory), work, read, written)


def build_real_automaton(
    dims: Sequence[int], initial_state: int | None = None
) -> tuple[CostAutomaton, tuple[Tree, ...], tuple[int, ...]]:
    trees = all_trees(len(dims) - 1)
    static = static_pareto_indices(trees, dims)
    initial = static[0] if initial_state is None else initial_state
    alpha = tuple(0 if state == initial else None for state in range(len(trees)))
    matrices = tuple(
        tuple(
            tuple(
                migration_totals(dims, source, target, update).scalar_cost
                for target in trees
            )
            for source in trees
        )
        for update in range(len(dims) - 1)
    )
    automaton = CostAutomaton(
        tuple(f"update_{index}" for index in range(len(dims) - 1)),
        alpha,
        matrices,
        tuple(tree.render() for tree in trees),
    )
    return automaton, trees, static
