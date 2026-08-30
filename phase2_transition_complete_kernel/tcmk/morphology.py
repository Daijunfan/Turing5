from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from typing import Iterable, Iterator, Sequence


@dataclass(frozen=True)
class Tree:
    """An ordered full binary tree over a contiguous matrix-chain interval."""

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
            raise ValueError("a leaf has no split")
        return self.left.j

    def internal_nodes(self) -> tuple[Tree, ...]:
        if self.leaf:
            return ()
        assert self.left is not None and self.right is not None
        return (self,) + self.left.internal_nodes() + self.right.internal_nodes()

    def intervals(self) -> frozenset[tuple[int, int]]:
        return frozenset((node.i, node.j) for node in self.internal_nodes())

    def splits(self) -> frozenset[tuple[int, int, int]]:
        return frozenset((node.i, node.j, node.split) for node in self.internal_nodes())

    def render(self, root: bool = True) -> str:
        if self.leaf:
            return _leaf_name(self.i)
        assert self.left is not None and self.right is not None
        text = f"({self.left.render(False)}{self.right.render(False)})"
        return text[1:-1] if root else text


def _leaf_name(index: int) -> str:
    return chr(ord("A") + index) if index < 26 else f"A{index}"


@lru_cache(maxsize=None)
def _enumerate_interval(i: int, j: int) -> tuple[Tree, ...]:
    if i == j:
        return (Tree(i, j),)
    trees: list[Tree] = []
    for split in range(i, j):
        for left in _enumerate_interval(i, split):
            for right in _enumerate_interval(split + 1, j):
                trees.append(Tree(i, j, left, right))
    return tuple(trees)


def enumerate_trees(matrix_count: int) -> tuple[Tree, ...]:
    if matrix_count < 1:
        raise ValueError("matrix_count must be positive")
    return _enumerate_interval(0, matrix_count - 1)


@dataclass(frozen=True)
class Morphology:
    tree: Tree
    right_first: frozenset[tuple[int, int]]
    work: int
    peak: int

    @property
    def name(self) -> str:
        return self.tree.render()

    @property
    def splits(self) -> frozenset[tuple[int, int, int]]:
        return self.tree.splits()

    @property
    def intervals(self) -> frozenset[tuple[int, int]]:
        return self.tree.intervals()


def result_size(dims: Sequence[int], i: int, j: int) -> int:
    return dims[i] * dims[j + 1]


def multiplication_work(dims: Sequence[int], i: int, k: int, j: int) -> int:
    return dims[i] * dims[k + 1] * dims[j + 1]


def evaluate_tree(
    tree: Tree,
    dims: Sequence[int],
    right_first: frozenset[tuple[int, int]] = frozenset(),
) -> tuple[int, int]:
    if tree.leaf:
        return 0, 0
    assert tree.left is not None and tree.right is not None
    left_work, left_peak = evaluate_tree(tree.left, dims, right_first)
    right_work, right_peak = evaluate_tree(tree.right, dims, right_first)
    i, k, j = tree.i, tree.split, tree.j
    left_temp = result_size(dims, i, k) if i < k else 0
    right_temp = result_size(dims, k + 1, j) if k + 1 < j else 0
    coexist = left_temp + right_temp + result_size(dims, i, j)
    peak_lr = max(left_peak, left_temp + right_peak, coexist)
    peak_rl = max(right_peak, right_temp + left_peak, coexist)
    peak = peak_rl if (i, j) in right_first else peak_lr
    work = left_work + right_work + multiplication_work(dims, i, k, j)
    return work, peak


def ordered_morphologies(dims: Sequence[int]) -> Iterator[Morphology]:
    """Enumerate tree structure and every recursive left/right evaluation order."""

    for tree in enumerate_trees(len(dims) - 1):
        nodes = tuple((node.i, node.j) for node in tree.internal_nodes())
        for choices in product((False, True), repeat=len(nodes)):
            order = frozenset(node for node, right in zip(nodes, choices) if right)
            work, peak = evaluate_tree(tree, dims, order)
            yield Morphology(tree, order, work, peak)


def canonical_morphology(tree: Tree, dims: Sequence[int]) -> Morphology:
    """Choose an exact minimum-peak order; ties are deterministically left-first."""

    def visit(node: Tree) -> tuple[int, int, frozenset[tuple[int, int]]]:
        if node.leaf:
            return 0, 0, frozenset()
        assert node.left is not None and node.right is not None
        lw, lp, lo = visit(node.left)
        rw, rp, ro = visit(node.right)
        i, k, j = node.i, node.split, node.j
        lt = result_size(dims, i, k) if i < k else 0
        rt = result_size(dims, k + 1, j) if k + 1 < j else 0
        coexist = lt + rt + result_size(dims, i, j)
        peak_lr = max(lp, lt + rp, coexist)
        peak_rl = max(rp, rt + lp, coexist)
        order = lo | ro
        if peak_rl < peak_lr:
            order |= {(i, j)}
        return lw + rw + multiplication_work(dims, i, k, j), min(peak_lr, peak_rl), order

    work, peak, order = visit(tree)
    return Morphology(tree, order, work, peak)


def canonical_morphologies(dims: Sequence[int]) -> tuple[Morphology, ...]:
    """One safe representative per tree for the legacy split-distance model.

    Evaluation order cannot change work or split distance in that model.  The
    minimum-peak order therefore weakly dominates every other order of the same
    tree, so the orders can be merged there (and only there).
    """

    return tuple(canonical_morphology(tree, dims) for tree in enumerate_trees(len(dims) - 1))


def static_pareto(states: Iterable[Morphology]) -> tuple[Morphology, ...]:
    """Old PCME frontier, with one deterministic representative per cost point."""

    states = tuple(states)
    frontier = [
        state
        for state in states
        if not any(
            other.work <= state.work
            and other.peak <= state.peak
            and (other.work < state.work or other.peak < state.peak)
            for other in states
        )
    ]
    by_point: dict[tuple[int, int], Morphology] = {}
    for state in sorted(frontier, key=lambda item: item.name):
        by_point.setdefault((state.work, state.peak), state)
    return tuple(sorted(by_point.values(), key=lambda item: (item.peak, item.work, item.name)))

