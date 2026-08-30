from __future__ import annotations

from random import Random
from typing import Sequence

from .certificate_checker import check_migration_certificate
from .morphology import Tree
from .real_migration import Interval, MigrationCertificate, interval_nodes, plan_migration

Matrix = tuple[tuple[int, ...], ...]
MODULUS = 1_000_000_007


def multiply(left: Matrix, right: Matrix, modulus: int = MODULUS) -> Matrix:
    if not left or not right or len(left[0]) != len(right):
        raise ValueError("matrix dimension mismatch")
    rows, middle, columns = len(left), len(right), len(right[0])
    output = [[0] * columns for _ in range(rows)]
    for i in range(rows):
        for k in range(middle):
            value = left[i][k]
            for j in range(columns):
                output[i][j] = (output[i][j] + value * right[k][j]) % modulus
    return tuple(tuple(row) for row in output)


def random_matrix(rows: int, columns: int, random: Random) -> Matrix:
    return tuple(
        tuple(random.randrange(MODULUS) for _ in range(columns)) for _ in range(rows)
    )


def left_fold(leaves: Sequence[Matrix]) -> Matrix:
    value = leaves[0]
    for leaf in leaves[1:]:
        value = multiply(value, leaf)
    return value


class PersistentExecutor:
    """Execute certified migrations while maintaining every target interval."""

    def __init__(self, dims: Sequence[int], leaves: Sequence[Matrix], tree: Tree):
        self.dims = tuple(dims)
        self.leaves = list(leaves)
        self.tree = tree
        self.cache: dict[Interval, Matrix] = {}
        self._materialize(tree)
        self.version = 0
        self._check_reference()

    def _materialize(self, node: Tree) -> Matrix:
        if node.leaf:
            return self.leaves[node.i]
        assert node.left is not None and node.right is not None
        value = multiply(self._materialize(node.left), self._materialize(node.right))
        self.cache[(node.i, node.j)] = value
        return value

    def _check_reference(self) -> None:
        root = self.cache[(self.tree.i, self.tree.j)]
        if root != left_fold(self.leaves):
            raise AssertionError("maintained result differs from a from-scratch left fold")

    def apply_update(
        self,
        update_index: int,
        replacement: Matrix,
        target: Tree,
        certificate: MigrationCertificate | None = None,
    ) -> MigrationCertificate:
        certificate = certificate or plan_migration(
            self.dims, self.tree, target, update_index
        )
        checked = check_migration_certificate(
            self.dims, self.tree, target, certificate
        )
        if not checked.valid:
            raise ValueError(f"invalid migration certificate: {checked.error}")
        expected_shape = (self.dims[update_index], self.dims[update_index + 1])
        actual_shape = (len(replacement), len(replacement[0]) if replacement else 0)
        if actual_shape != expected_shape:
            raise ValueError("replacement matrix has the wrong shape")

        self.leaves[update_index] = replacement
        old_cache = self.cache
        self.cache = {
            interval: old_cache[interval] for interval in certificate.reused_intervals
        }
        nodes = interval_nodes(target)
        for interval in certificate.execution_order:
            node = nodes[interval]
            assert node.left is not None and node.right is not None
            left_key = (node.left.i, node.left.j)
            right_key = (node.right.i, node.right.j)
            left = self.leaves[node.left.i] if node.left.leaf else self.cache[left_key]
            right = self.leaves[node.right.i] if node.right.leaf else self.cache[right_key]
            self.cache[interval] = multiply(left, right)
        self.tree = target
        self.version += 1
        if set(self.cache) != set(interval_nodes(target)):
            raise AssertionError("migration did not materialize exactly the target intervals")
        self._check_reference()
        return certificate

