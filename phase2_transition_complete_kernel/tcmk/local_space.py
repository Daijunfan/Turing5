from __future__ import annotations

from collections import deque
from random import Random
from typing import Sequence

from .morphology import Morphology, Tree, canonical_morphology


def join(left: Tree, right: Tree) -> Tree:
    return Tree(left.i, right.j, left, right)


def left_fold(i: int, j: int) -> Tree:
    tree = Tree(i, i)
    for leaf in range(i + 1, j + 1):
        tree = join(tree, Tree(leaf, leaf))
    return tree


def right_fold(i: int, j: int) -> Tree:
    tree = Tree(j, j)
    for leaf in range(j - 1, i - 1, -1):
        tree = join(Tree(leaf, leaf), tree)
    return tree


def balanced_tree(i: int, j: int) -> Tree:
    if i == j:
        return Tree(i, j)
    split = (i + j) // 2
    return join(balanced_tree(i, split), balanced_tree(split + 1, j))


def random_tree(i: int, j: int, random: Random) -> Tree:
    if i == j:
        return Tree(i, j)
    split = random.randrange(i, j)
    return join(random_tree(i, split, random), random_tree(split + 1, j, random))


def rotation_neighbors(tree: Tree) -> tuple[Tree, ...]:
    if tree.leaf:
        return ()
    assert tree.left is not None and tree.right is not None
    result: set[Tree] = set()
    left, right = tree.left, tree.right
    if not left.leaf:
        assert left.left is not None and left.right is not None
        result.add(join(left.left, join(left.right, right)))
    if not right.leaf:
        assert right.left is not None and right.right is not None
        result.add(join(join(left, right.left), right.right))
    for neighbor in rotation_neighbors(left):
        result.add(join(neighbor, right))
    for neighbor in rotation_neighbors(right):
        result.add(join(left, neighbor))
    return tuple(sorted(result, key=lambda item: item.render()))


def local_rewrite_states(
    dims: Sequence[int], limit: int = 128, seed: int = 0
) -> tuple[Morphology, ...]:
    """Build a bounded local-rewrite graph without enumerating Catalan space."""

    matrix_count = len(dims) - 1
    random = Random(seed)
    seeds = [
        left_fold(0, matrix_count - 1),
        right_fold(0, matrix_count - 1),
        balanced_tree(0, matrix_count - 1),
    ]
    seeds.extend(
        random_tree(0, matrix_count - 1, random) for _ in range(min(16, limit - 3))
    )
    queue = deque(seeds)
    seen: set[Tree] = set()
    while queue and len(seen) < limit:
        tree = queue.popleft()
        if tree in seen:
            continue
        seen.add(tree)
        queue.extend(rotation_neighbors(tree))
    return tuple(
        canonical_morphology(tree, dims)
        for tree in sorted(seen, key=lambda item: item.render())
    )

