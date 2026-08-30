from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .morphology import Tree, multiplication_work, result_size
from .real_migration import (
    Interval,
    MigrationCertificate,
    interval_bytes,
    interval_nodes,
)


@dataclass(frozen=True)
class CheckResult:
    valid: bool
    error: str = ""


def check_migration_certificate(
    dims: Sequence[int],
    source: Tree | None,
    target: Tree,
    certificate: MigrationCertificate,
    *,
    element_bytes: int = 8,
) -> CheckResult:
    """Independent set/dependency/accounting checker; never invokes the planner."""

    source_nodes = {} if source is None else interval_nodes(source)
    target_nodes = interval_nodes(target)
    source_set, target_set = set(source_nodes), set(target_nodes)
    invalidated = {
        interval
        for interval in source_set
        if interval[0] <= certificate.update_index <= interval[1]
    }
    reused = (source_set & target_set) - invalidated
    recomputed = target_set - reused
    released = source_set - reused
    expected = (
        tuple(sorted(reused)),
        tuple(sorted(invalidated)),
        tuple(sorted(recomputed)),
        tuple(sorted(released)),
    )
    actual = (
        certificate.reused_intervals,
        certificate.invalidated_intervals,
        certificate.recomputed_intervals,
        certificate.released_intervals,
    )
    if actual != expected:
        return CheckResult(False, "reuse/invalidation/recompute/release sets differ")
    if set(certificate.execution_order) != recomputed or len(
        certificate.execution_order
    ) != len(recomputed):
        return CheckResult(False, "execution order is not a permutation of recomputations")

    old_memory = sum(
        interval_bytes(dims, interval, element_bytes) for interval in source_set
    )
    resident = sum(interval_bytes(dims, interval, element_bytes) for interval in reused)
    peak = max(old_memory, resident)
    available = set(reused)
    work = read = written = 0
    for interval in certificate.execution_order:
        node = target_nodes[interval]
        assert node.left is not None and node.right is not None
        dependencies = [
            (child.i, child.j)
            for child in (node.left, node.right)
            if not child.leaf
        ]
        if any(item not in available for item in dependencies):
            return CheckResult(False, f"dependency missing before {interval}")
        output = interval_bytes(dims, interval, element_bytes)
        peak = max(peak, resident + output)
        resident += output
        i, k, j = node.i, node.split, node.j
        work += multiplication_work(dims, i, k, j)
        read += (
            result_size(dims, i, k) + result_size(dims, k + 1, j)
        ) * element_bytes
        written += output
        available.add(interval)
    if available != target_set:
        return CheckResult(False, "final cache is not exactly the target cache")
    if (
        peak,
        work,
        read,
        written,
    ) != (
        certificate.peak_memory,
        certificate.total_work,
        certificate.read_bytes,
        certificate.write_bytes,
    ):
        return CheckResult(False, "work/byte/peak accounting differs")
    return CheckResult(True)

